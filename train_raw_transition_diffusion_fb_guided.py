"""
FB-guided Raw Transition Diffusion Training — Complete Single-File Script.
========================================================================

Ablation-aligned raw-transition version.
The intended experimental difference from the latent/VAE version is the representation:
raw 936-dim x=[s,a,s',next_qpos,next_qvel] instead of a 128-dim VAE latent.
Hidden size intentionally remains 1024 for the raw model. Bellman/Q semantics, ResNet ordering, diffusion
schedule/DDIM, split generation, action clipping, validation seeding, and L_B/L_F
semantics are aligned with train_latent_diffusion_fb_guided.py.

Key runtime choices in this version:
- Frozen FB guidance model: facebook/metamotivo-M-1.
- Raw diffusion denoiser: initialized from scratch (no diffusion checkpoint resume/load).
- Generated-vs-real distribution diagnostics (including covariance relative Frobenius
  error and sliced Wasserstein distance), plus expensive FB B-covariance cond/eigenvalue
  diagnostics: run once, after training completes.

Run directly via:
    python train_raw_transition_diffusion_fb_guided_scratch_M1.py
"""
from __future__ import annotations

import copy
import json
import time
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# Config
# ==============================================================================
def _first_existing(candidates: tuple[str, ...]) -> str | None:
    return next((path for path in candidates if Path(path).exists()), None)


@dataclass
class PathConfig:
    raw_hdf5: str = "/home/kyoungsookim/rho1/metamotivo-M-1-datasets/data/buffer_inference_500000.hdf5"
    save_dir: str | None = None
    # Optional: point this to the latent experiment's fb_guided_diffusion_split_indices.npz
    # for byte-for-byte identical train/validation membership. If omitted, the exact same
    # default_rng(seed) split algorithm as the latent code is used.
    shared_split_npz: str | None = None
    # Expert AMASS motions used only for expert-ASM z sampling.
    expert_motions: str = "/home/kyoungsookim/humenv/data_preparation/test_train_split/large1_small1_train_0.1.txt"
    expert_motions_root: str = "/home/kyoungsookim/humenv/data_preparation/humenv_amass"

    def resolve(self) -> None:
        env_save_dir = os.environ.get("SAVE_DIR")
        if env_save_dir:
            self.save_dir = env_save_dir
        elif not self.save_dir or self.save_dir.startswith("/home/kyoungsookim/rho"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = f"/home/kyoungsookim/rho1/fb_guided_raw_{timestamp}"

        self.shared_split_npz = os.environ.get(
            "SHARED_SPLIT_NPZ", self.shared_split_npz
        )
        self.expert_motions = os.environ.get(
            "EXPERT_MOTIONS", self.expert_motions
        )
        self.expert_motions_root = os.environ.get(
            "EXPERT_MOTIONS_ROOT", self.expert_motions_root
        )


@dataclass
class ModelConfig:
    transition_dim: int = 785  # s(358) + a(69) + s'(358)
    hidden_dim: int = 1024
    num_blocks: int = 6
    timesteps: int = 1000


@dataclass
class TrainConfig:
    seed: int = 42
    split_seed: int = 42
    validation_fraction: float = 0.05
    epochs: int = 500
    batch_size: int = 3072
    learning_rate: float = 1.0e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.99
    best_checkpoint_prior_tolerance: float = 0.02
    max_workers: int = 4
    max_grad_norm: float = 1.0
    log_interval: int = 10
    # Gradient diagnostics call autograd.grad several extra times.
    # Set to 0 to disable them entirely and reduce peak GPU memory usage.
    gradient_diagnostic_batches_per_epoch: int = 0
    checkpoint_epochs: tuple[int, ...] = (100, 200, 300, 400, 500)


@dataclass
class GuidanceConfig:
    # Fixed-weight guidance hyperparameters; FB guidance uses only x=[s,a,s'].
    ratio = 1.0
    lambda_bellman: float = 2.5e-5
    lambda_q: float = 4.0e-6
    lambda_b: float = 0.15
    lambda_f: float = 0.005
    discount: float = 0.98
    pessimism_penalty: float = 0.0
    actor_stddev_clip: float = 0.3
    batch_size: int = 3072
    start_epoch: int = 1
    ramp_epochs: int = 50
    validation_batches: int = 8
    action_clip: float | None = 1.0
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6
    expert_seq_length: int = 8

    prior_loss_threshold: float = 0.82
    adaptive_down: float = 0.90
    adaptive_up: float = 1.02
    adaptive_min: float = 0.10
    adaptive_max: float = 1.00


@dataclass
class EvaluationConfig:
    # Generated-vs-real distribution diagnostics are intentionally run only once,
    # after all training epochs finish. This avoids repeated DDIM sampling and
    # covariance/Frobenius/SWD calculations during training. Expensive FB covariance
    # condition/eigenvalue diagnostics are also deferred to this final evaluation.
    ddim_steps: int = 50
    final_samples: int = 10_000
    batch_size: int = 2_048
    swd_projections: int = 64
    seed: int = 2026


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        self.paths.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def guidance_ramp(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = (epoch - start_epoch + 1) / float(ramp_epochs)
    return min(1.0, max(0.0, progress))


# ==============================================================================
# Raw Transition Dataset & DataLoader
# ==============================================================================
class RawTransitionDataset(Dataset):
    """Dataset holding standardized 785-dim x=[s,a,s']."""

    def __init__(
        self,
        normalized_data: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
    ):
        self.data = normalized_data
        self.mean = mean
        self.std = std
        self.train_indices = train_indices
        self.val_indices = val_indices

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


@dataclass
class DataBundle:
    module: RawTransitionDataset
    train_loader: DataLoader
    val_loader: DataLoader
    train_indices: np.ndarray
    val_indices: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def build_data(config: ExperimentConfig, device: torch.device) -> DataBundle:
    raw_path = config.paths.raw_hdf5
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Raw dataset HDF5 not found at: {raw_path}")

    print(f"[Data] Loading raw transitions from {raw_path}...")
    with h5py.File(raw_path, "r") as hf:
        obs = hf["observation"][:]             # (N, 358)
        action = hf["action"][:]               # (N, 69)
        next_obs = hf["next_observation"][:]   # (N, 358)

    raw_transitions = np.concatenate(
        [obs, action, next_obs], axis=1
    ).astype(np.float32)
    num_samples = len(raw_transitions)

    if raw_transitions.shape[1] != config.model.transition_dim:
        raise ValueError(
            "Configured transition_dim does not match the loaded transition: "
            f"config={config.model.transition_dim}, loaded={raw_transitions.shape[1]} "
            f"(obs={obs.shape[1]}, action={action.shape[1]}, next_obs={next_obs.shape[1]})."
        )

    print(
        f"[Data] Loaded {num_samples:,} transitions of dimension "
        f"{raw_transitions.shape[1]} ([s,a,s'])"
    )

    # Match the latent code's split exactly. For the strictest ablation, set
    # SHARED_SPLIT_NPZ to the latent run's saved split file.
    shared_split = config.paths.shared_split_npz
    if shared_split:
        split_path = Path(shared_split)
        if not split_path.exists():
            raise FileNotFoundError(f"Shared split file not found: {split_path}")
        split = np.load(split_path, allow_pickle=False)
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        val_indices = np.asarray(split["val_indices"], dtype=np.int64)
        if len(train_indices) + len(val_indices) != num_samples:
            raise ValueError(
                "Shared split size does not match raw dataset: "
                f"train={len(train_indices)}, val={len(val_indices)}, N={num_samples}."
            )
        if np.intersect1d(train_indices, val_indices).size:
            raise ValueError("Shared train/val split contains overlapping indices.")
        if train_indices.min() < 0 or val_indices.min() < 0 or train_indices.max() >= num_samples or val_indices.max() >= num_samples:
            raise ValueError("Shared split contains out-of-range indices.")
        print(f"[Data] Loaded shared split: {split_path}")
    else:
        generator = np.random.default_rng(config.train.split_seed)
        all_indices = generator.permutation(num_samples)
        num_val = int(round(num_samples * config.train.validation_fraction))
        val_indices = all_indices[:num_val]
        train_indices = all_indices[num_val:]
        print("[Data] Using latent-aligned default_rng split algorithm.")

    # Match the latent code's train-only statistics precision/minimum std.
    train_raw = raw_transitions[train_indices]
    mean = train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)

    normalized = (raw_transitions - mean) / std
    normalized_tensor = torch.from_numpy(normalized).float()

    dataset_module = RawTransitionDataset(
        normalized_data=normalized_tensor,
        mean=mean,
        std=std,
        train_indices=train_indices,
        val_indices=val_indices,
    )

    train_subset = torch.utils.data.Subset(normalized_tensor, train_indices)
    val_subset = torch.utils.data.Subset(normalized_tensor, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.max_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.max_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(
        f"[Data] Train samples: {len(train_indices):,}, Val samples: {len(val_indices):,}, "
        f"Batch size: {config.train.batch_size}"
    )
    return DataBundle(
        module=dataset_module,
        train_loader=train_loader,
        val_loader=val_loader,
        train_indices=train_indices,
        val_indices=val_indices,
        mean=mean,
        std=std,
    )


# ==============================================================================
# Expert AMASS Trajectory Buffer (for expert-ASM z sampling only)
# ==============================================================================
def load_expert_trajectory_buffer(
    motions: str | Path,
    motions_root: str | Path,
    device: str,
    sequence_length: int,
):
    """Load HumEnv AMASS HDF5 motions into MetaMotivo's TrajectoryBuffer.

    This follows the FB-CPR expert sampling semantics, but stores only observations
    because expert-ASM encoding needs only consecutive expert states.
    """
    try:
        from humenv.misc.motionlib import canonicalize, load_episode_based_h5
        from metamotivo.buffers.buffers import TrajectoryBuffer
    except ImportError as error:
        raise ImportError(
            "HumEnv and MetaMotivo are required for expert AMASS loading."
        ) from error

    motions = Path(motions)
    motions_root = Path(motions_root)
    if not motions.exists():
        raise FileNotFoundError(f"Expert motion manifest not found: {motions}")
    if not motions_root.exists():
        raise FileNotFoundError(f"Expert motion root not found: {motions_root}")
    if sequence_length <= 0:
        raise ValueError("expert_seq_length must be positive.")

    with motions.open("r", encoding="utf-8") as txtf:
        h5files = [line.strip().replace(" ", "") for line in txtf if line.strip()]

    episodes: list[dict[str, np.ndarray]] = []
    skipped_short = 0
    print(f"[Expert ASM] Loading {len(h5files):,} AMASS motions from {motions_root}...")
    for rel_path in h5files:
        h5_path = canonicalize(rel_path, base_path=motions_root)
        loaded = load_episode_based_h5(h5_path, keys=["observation"])
        for episode in loaded:
            observation = np.asarray(episode["observation"], dtype=np.float32)
            # TrajectoryBuffer samples L transitions and offsets next-observation by 1,
            # so at least L+2 frames are required by its sampling rule.
            if observation.shape[0] <= sequence_length + 1:
                skipped_short += 1
                continue
            episodes.append({"observation": observation})

    if not episodes:
        raise RuntimeError("No valid expert AMASS trajectories were loaded.")

    buffer = TrajectoryBuffer(
        capacity=len(episodes),
        seq_length=sequence_length,
        device=device,
    )
    buffer.extend(episodes)
    print(
        f"[Expert ASM] Loaded {len(episodes):,} trajectories "
        f"(seq_length={sequence_length}, skipped_short={skipped_short:,})."
    )
    return buffer


# ==============================================================================
# Model Architecture (936-dim Augmented Raw Transition Denoising MLP)
# ==============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        frequency_scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32)
            * -frequency_scale
        )
        angles = x.float()[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResNetBlock1D(nn.Module):
    """Residual MLP block aligned with the latent denoiser (pre-norm ordering)."""

    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim),
        )
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)
        h = self.block2(h)
        return x + h


class RawTransitionDenoisingModel(nn.Module):
    """MLP denoising diffusion model on [s,a,s',next_qpos,next_qvel]."""

    def __init__(
        self,
        transition_dim: int = 785,
        hidden_dim: int = 1024,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.transition_dim = transition_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [
                ResNetBlock1D(dim=hidden_dim, time_emb_dim=hidden_dim)
                for _ in range(num_blocks)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, transition_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.sub_((1.0 - self.decay) * (s_param - m_param))


# ==============================================================================
# Gaussian Diffusion (v-parameterization)
# ==============================================================================
class GaussianDiffusion:
    """v-prediction diffusion aligned with the latent code's Zero-Terminal-SNR DDIM."""

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Union[str, torch.device] = "cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)

        if schedule == "cosine":
            s = 0.008
            x = torch.linspace(
                0,
                timesteps,
                timesteps + 1,
                device=self.device,
                dtype=torch.float64,
            )
            alpha_bar_unscaled = torch.cos(
                ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
            ).pow(2)
            alpha_bar_unscaled = alpha_bar_unscaled / alpha_bar_unscaled[0]

            # Match latent code: enforce terminal SNR exactly zero.
            sqrt_alpha_bar = torch.sqrt(alpha_bar_unscaled)
            sqrt_alpha_bar_0 = sqrt_alpha_bar[0]
            sqrt_alpha_bar_T = sqrt_alpha_bar[-1]
            sqrt_alpha_bar_rescaled = (
                (sqrt_alpha_bar - sqrt_alpha_bar_T)
                * sqrt_alpha_bar_0
                / (sqrt_alpha_bar_0 - sqrt_alpha_bar_T)
            )
            self.alphas_cumprod = sqrt_alpha_bar_rescaled.pow(2)[1:].to(torch.float32)
        elif schedule == "linear":
            betas = torch.linspace(
                beta_start,
                beta_end,
                timesteps,
                device=self.device,
                dtype=torch.float32,
            )
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            raise ValueError(f"Unsupported noise schedule: {schedule}")

        if self.alphas_cumprod.shape != (timesteps,):
            raise RuntimeError(
                f"alphas_cumprod must contain {timesteps} values, got "
                f"{tuple(self.alphas_cumprod.shape)}."
            )
        if schedule == "cosine" and abs(self.alphas_cumprod[-1].item()) > 1e-12:
            raise RuntimeError(
                "Zero-terminal-SNR failed: "
                f"alpha_bar_T={self.alphas_cumprod[-1].item():.8e}"
            )

        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1],
            (1, 0),
            value=1.0,
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            (1.0 - self.alphas_cumprod).clamp(min=0.0)
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        x_t = alpha_t * x_0 + sigma_t * noise
        target_v = alpha_t * noise - sigma_t * x_0
        return x_t, target_v

    def predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v_prediction: torch.Tensor,
    ) -> torch.Tensor:
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return alpha_t * x_t - sigma_t * v_prediction

    @torch.no_grad()
    def p_sample_ddim(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch_size = x_t.shape[0]
        v_theta = model(x_t, t)
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        x_0_pred = alpha_t * x_t - sigma_t * v_theta
        eps_pred = sigma_t * x_t + alpha_t * v_theta

        if (t_prev >= 0).all():
            alpha_bar_prev = self.alphas_cumprod[t_prev].view(-1, 1)
        else:
            alpha_bar_prev = torch.ones(
                (batch_size, 1), device=self.device, dtype=x_t.dtype
            )

        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        variance_term = (
            (1.0 - alpha_bar_prev)
            / (1.0 - self.alphas_cumprod[t].view(-1, 1)).clamp(min=1e-8)
            * (
                1.0
                - self.alphas_cumprod[t].view(-1, 1)
                / alpha_bar_prev.clamp(min=1e-8)
            )
        ).clamp(min=0.0)
        sigma = eta * torch.sqrt(variance_term)
        direction = torch.sqrt(
            (1.0 - alpha_bar_prev - sigma.pow(2)).clamp(min=0.0)
        ) * eps_pred

        if eta > 0.0:
            noise = torch.randn(
                x_t.shape,
                device=x_t.device,
                dtype=x_t.dtype,
                generator=generator,
            )
        else:
            noise = torch.zeros_like(x_t)

        return sqrt_alpha_bar_prev * x_0_pred + direction + sigma * noise

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: Tuple[int, int],
        ddim_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if ddim_steps < 2 or ddim_steps > self.timesteps:
            raise ValueError(f"ddim_steps must be between 2 and {self.timesteps}.")

        model.eval()
        x = torch.randn(shape, device=self.device, generator=generator)

        # Match latent code exactly: start at T-1 (=999 for T=1000), not 980.
        timesteps_seq = np.linspace(
            self.timesteps - 1, 0, ddim_steps, dtype=np.int64
        )
        for index, step in enumerate(timesteps_seq):
            t = torch.full(
                (shape[0],), int(step), device=self.device, dtype=torch.long
            )
            t_prev_value = (
                int(timesteps_seq[index + 1])
                if index < len(timesteps_seq) - 1
                else -1
            )
            t_prev = torch.full(
                (shape[0],), t_prev_value, device=self.device, dtype=torch.long
            )
            x = self.p_sample_ddim(
                model=model,
                x_t=x,
                t=t,
                t_prev=t_prev,
                eta=eta,
                generator=generator,
            )
        return x


# ==============================================================================
# FB Transition Constraint (Evaluated on Raw Transitions without VAE)
# ==============================================================================
def load_frozen_fb_model(device: torch.device) -> nn.Module:
    """Load the frozen MetaMotivo FB model used only for guidance.

    The diffusion denoiser itself is *not* loaded from a checkpoint; it is
    initialized from scratch in Trainer._build_model().
    """
    try:
        from metamotivo.fb_cpr.huggingface import FBcprModel

        model = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
        model.to(device)
    except ImportError as error:
        raise ImportError(
            'MetaMotivo is required. Install with: pip install '
            '"metamotivo[huggingface,humenv] @ '
            'git+https://github.com/facebookresearch/metamotivo.git"'
        ) from error

    model.eval()
    model.requires_grad_(False)
    return model


class FBRawTransitionConstraint(nn.Module):
    """FB guidance on the [s,a,s'] prefix; appended qpos/qvel are prior-only targets."""

    def __init__(
        self,
        fb_model: nn.Module,
        transition_mean: np.ndarray,
        transition_std: np.ndarray,
        device: torch.device,
        discount: float = 0.98,
        q_pessimism_penalty: float = 0.0,
        q_actor_stddev_clip: float = 0.3,
        action_clip: Optional[float] = 1.0,
        train_goal_ratio: float = 0.2,
        expert_asm_ratio: float = 0.6,
    ):
        super().__init__()
        self.fb_model = fb_model
        self.device = device
        self.discount = float(discount)
        self.q_pessimism_penalty = float(q_pessimism_penalty)
        self.q_actor_stddev_clip = float(q_actor_stddev_clip)
        self.action_clip = action_clip
        self.train_goal_ratio = float(train_goal_ratio)
        self.expert_asm_ratio = float(expert_asm_ratio)
        self.obs_dim = int(fb_model.cfg.obs_dim)
        self.action_dim = int(fb_model.cfg.action_dim)
        self.transition_dim = 2 * self.obs_dim + self.action_dim
        self.z_dim = int(fb_model.cfg.archi.z_dim)
        # Match latent code: actor distribution std comes from the frozen FB model.
        self.actor_std = float(fb_model.cfg.actor_std)

        self.register_buffer(
            "mean",
            torch.as_tensor(transition_mean, dtype=torch.float32, device=device).view(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.as_tensor(transition_std, dtype=torch.float32, device=device).view(1, -1),
            persistent=False,
        )

    def sample_mixed_z(
        self,
        num_samples: int,
        next_obs_fb: torch.Tensor,
        expert_encodings: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Samples mixed z context matching MetaMotivo FB-CPR agent logic.

        - Goal state z: Permutation of next_obs in batch passed through _backward_map.
        - Expert motion z: Expert encodings if provided.
        - Projected Gaussian z: Random prior noise on hypersphere.
        """
        z_raw = torch.randn(
            (num_samples, self.z_dim),
            device=next_obs_fb.device,
            dtype=next_obs_fb.dtype,
            generator=generator,
        )
        z = self.fb_model.project_z(z_raw)

        if self.train_goal_ratio <= 0.0 and self.expert_asm_ratio <= 0.0:
            return z

        p_goal = self.train_goal_ratio
        p_expert = self.expert_asm_ratio
        p_random = max(0.0, 1.0 - p_goal - p_expert)

        prob = torch.tensor(
            [p_goal, p_expert, p_random],
            dtype=torch.float32,
            device=next_obs_fb.device,
        )
        mix_idxs = torch.multinomial(
            prob, num_samples=num_samples, replacement=True, generator=generator
        ).reshape(-1, 1)

        # Goal state z: Permutation of batch's next_obs mapped via B(s_goal)
        if p_goal > 0.0:
            perm = torch.randperm(num_samples, device=next_obs_fb.device, generator=generator)
            goals = self.fb_model._backward_map(next_obs_fb[perm])
            goals = self.fb_model.project_z(goals)
            z = torch.where(mix_idxs == 0, goals, z)

        # Expert motion z from AMASS trajectory encodings.
        if p_expert > 0.0:
            if expert_encodings is None:
                raise ValueError(
                    "expert_asm_ratio > 0 requires AMASS expert_encodings."
                )
            if expert_encodings.shape[0] != num_samples:
                raise ValueError(
                    "expert_encodings batch mismatch: "
                    f"expected {num_samples}, got {expert_encodings.shape[0]}."
                )
            perm = torch.randperm(num_samples, device=next_obs_fb.device, generator=generator)
            z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    def unstandardize(
        self, x_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_phys = x_norm * self.std + self.mean
        obs = x_phys[:, : self.obs_dim]
        action_start = self.obs_dim
        action_end = action_start + self.action_dim
        action = x_phys[:, action_start:action_end]
        next_obs = x_phys[:, action_end : action_end + self.obs_dim]

        # Important ablation alignment: clip the current generated/target action too.
        if self.action_clip is not None:
            action = action.clamp(-self.action_clip, self.action_clip)
        return obs, action, next_obs

    def _normalize_fb_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fb_model._obs_normalizer(obs)

    @staticmethod
    def _parallelize(features: torch.Tensor) -> torch.Tensor:
        return features.unsqueeze(0) if features.ndim == 2 else features

    @staticmethod
    def _metamotivo_target(
        predictions: torch.Tensor,
        pessimism_penalty: float,
    ) -> torch.Tensor:
        if predictions.ndim < 2:
            return predictions
        predictions_mean = predictions.mean(dim=0)
        num_parallel = predictions.shape[0]
        if num_parallel <= 1 or pessimism_penalty == 0.0:
            return predictions_mean
        pairwise_differences = torch.abs(
            predictions.unsqueeze(0) - predictions.unsqueeze(1)
        )
        normalizer = num_parallel**2 - num_parallel
        uncertainty = pairwise_differences.sum(dim=(0, 1)) / normalizer
        return predictions_mean - pessimism_penalty * uncertainty

    def forward(
        self,
        predicted_x0_norm: torch.Tensor,
        max_samples: int,
        target_x0_norm: Optional[torch.Tensor] = None,
        expert_encodings: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = predicted_x0_norm.shape[0]
        num_samples = min(batch_size, max_samples)
        indices = None
        if num_samples < batch_size:
            indices = torch.randperm(
                batch_size,
                device=predicted_x0_norm.device,
                generator=generator,
            )[:num_samples]
            predicted_x0_norm = predicted_x0_norm[indices]

        obs, action, next_obs = self.unstandardize(predicted_x0_norm)
        obs_fb = self._normalize_fb_obs(obs)
        next_obs_fb = self._normalize_fb_obs(next_obs)

        # Use noise-free real transition s' (target_x0_norm) for goal state z sampling
        ref_x0 = (
            target_x0_norm
            if target_x0_norm is not None
            else predicted_x0_norm.detach()
        )
        if indices is not None and ref_x0.shape[0] > num_samples:
            ref_x0 = ref_x0[indices]

        with torch.no_grad():
            obs_gt, action_gt, next_obs_gt = self.unstandardize(ref_x0)
            obs_gt_fb = self._normalize_fb_obs(obs_gt)
            next_obs_gt_fb = self._normalize_fb_obs(next_obs_gt)

            context = self.sample_mixed_z(
                num_samples=num_samples,
                next_obs_fb=next_obs_gt_fb,
                expert_encodings=expert_encodings,
                generator=generator,
            )

        forward_sa = self._parallelize(
            self.fb_model._forward_map(obs_fb, context, action)
        )

        # Bellman next action uses actor mean, matching the latent implementation.
        actor_dist = self.fb_model._actor(next_obs_fb, context, self.actor_std)
        mean_value = actor_dist.mean
        next_action = mean_value() if callable(mean_value) else mean_value
        if self.action_clip is not None:
            next_action = next_action.clamp(-self.action_clip, self.action_clip)
        forward_next = self._parallelize(
            self.fb_model._forward_map(next_obs_fb, context, next_action)
        )

        backward_y = self.fb_model._backward_map(obs_fb)
        backward_next = self.fb_model._backward_map(next_obs_fb)

        # Memory-efficient exact computation of the all-pairs Bellman residual MSE.
        # Instead of materializing [K, N, N] = (F - gamma F') B^T, use
        #   ||D B^T||_F^2 = <D^T D, B^T B>_F,
        # where D = F - gamma F'. This preserves the original scalar objective exactly
        # (up to floating-point roundoff) while avoiding the O(N^2) activation tensor.
        delta_f = forward_sa - self.discount * forward_next  # [K, N, z]
        gram_f = torch.matmul(delta_f.transpose(-1, -2), delta_f)  # [K, z, z]
        gram_b = torch.matmul(backward_y.T, backward_y)  # [z, z]
        bellman_residual_mse = (
            gram_f * gram_b.unsqueeze(0)
        ).sum() / (
            delta_f.shape[0] * delta_f.shape[1] * backward_y.shape[0]
        )

        # Match latent/custom Bellman objective exactly.
        source = (
            forward_sa * backward_next.unsqueeze(0)
        ).sum(dim=-1).mean()
        bellman_loss = bellman_residual_mse - 2.0 * source

        # MetaMotivo-style q_loss aligned with latent code:
        # sampled next action + target forward map + ensemble/pessimism target.
        Qs = (forward_sa * context.unsqueeze(0)).sum(dim=-1)
        with torch.no_grad():
            q_actor_dist = self.fb_model._actor(
                next_obs_fb, context, self.actor_std
            )
            try:
                q_next_action = q_actor_dist.sample(
                    clip=self.q_actor_stddev_clip
                )
            except TypeError:
                q_next_action = q_actor_dist.sample()
                if self.action_clip is not None:
                    q_next_action = q_next_action.clamp(
                        -self.action_clip, self.action_clip
                    )

            target_forward_net = (
                self.fb_model._target_forward_map
                if hasattr(self.fb_model, "_target_forward_map")
                else self.fb_model._forward_map
            )
            target_Fs = self._parallelize(
                target_forward_net(next_obs_fb, context, q_next_action)
            )
            next_Qs = (target_Fs * context.unsqueeze(0)).sum(dim=-1)
            next_Q = self._metamotivo_target(
                next_Qs, self.q_pessimism_penalty
            )

            q_B = backward_next.detach()
            cov_B = torch.matmul(q_B.T, q_B) / q_B.shape[0]
            try:
                inv_cov_B = torch.inverse(cov_B)
            except RuntimeError as error:
                raise RuntimeError(
                    "MetaMotivo q_loss covariance inversion failed. "
                    "Keep fb_guidance_batch_size >= z_dim; covariance diagnostics "
                    "are otherwise deferred to the final post-training check."
                ) from error

            implicit_reward = (
                torch.matmul(q_B, inv_cov_B) * context
            ).sum(dim=-1)
            target_Q = implicit_reward.detach() + self.discount * next_Q
            expanded_targets = target_Q.expand(Qs.shape[0], -1)

            # cov_B and its inverse are part of the MetaMotivo-style implicit reward,
            # so they remain in the training objective. Expensive diagnostics such as
            # eigvalsh/cond are intentionally NOT computed per batch; they are checked
            # once after training finishes.

        q_loss = 0.5 * Qs.shape[0] * F.mse_loss(Qs, expanded_targets)

        # Align L_B/L_F semantics too: L_F uses an independent random z_rand,
        # not the Bellman/Q context.
        with torch.no_grad():
            b_s_gt = self.fb_model._backward_map(obs_gt_fb)
            b_s_next_gt = self.fb_model._backward_map(next_obs_gt_fb)

            z_rand_raw = torch.randn(
                (num_samples, self.z_dim),
                device=obs.device,
                dtype=obs.dtype,
                generator=generator,
            )
            z_rand = self.fb_model.project_z(z_rand_raw)
            f_sa_gt = self._parallelize(
                self.fb_model._forward_map(obs_gt_fb, z_rand, action_gt)
            )

        b_s_hat = backward_y
        b_s_next_hat = backward_next
        f_sa_hat = self._parallelize(
            self.fb_model._forward_map(obs_fb, z_rand, action)
        )

        loss_b = F.mse_loss(b_s_hat, b_s_gt) + F.mse_loss(
            b_s_next_hat, b_s_next_gt
        )
        loss_f = F.mse_loss(f_sa_hat, f_sa_gt)

        return {
            "bellman_loss": bellman_loss,
            "bellman_residual_mse": bellman_residual_mse,
            "bellman_source": source,
            "q_loss": q_loss,
            "loss_b": loss_b,
            "loss_f": loss_f,
            "raw_obs_abs_mean": obs.abs().mean(),
            "raw_action_abs_mean": action.abs().mean(),
            "raw_next_obs_abs_mean": next_obs.abs().mean(),
        }


# ==============================================================================
# Gradient Tracker & Helpers
# ==============================================================================
class MeanTracker:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.sums[k] += float(v)
            self.counts[k] += 1

    def averages(self) -> Dict[str, float]:
        return {k: self.sums[k] / max(1, self.counts[k]) for k in self.sums}


@dataclass
class GradientMetrics:
    prior_norm: float
    bellman_norm: float
    q_norm: float
    b_norm: float
    f_norm: float
    aux_norm: float
    prior_aux_cosine: float
    bellman_to_prior: float
    q_to_prior: float
    b_to_prior: float
    f_to_prior: float
    aux_to_prior: float
    effective_scale: float


class GradientTracker:
    def __init__(
        self,
        parameters: list[nn.Parameter],
        lambda_bellman: float,
        lambda_q: float,
        lambda_b: float,
        lambda_f: float,
    ):
        self.parameters = parameters
        self.lambda_bellman = lambda_bellman
        self.lambda_q = lambda_q
        self.lambda_b = lambda_b
        self.lambda_f = lambda_f

    def _flat_grad(self, loss: torch.Tensor, retain_graph: bool = True) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss, self.parameters, retain_graph=retain_graph, allow_unused=True
        )
        flat = []
        for g, p in zip(grads, self.parameters):
            if g is None:
                flat.append(torch.zeros_like(p).view(-1))
            else:
                flat.append(g.view(-1))
        return torch.cat(flat)

    def compute(
        self,
        prior_loss: torch.Tensor,
        bellman_loss: torch.Tensor,
        q_loss: torch.Tensor,
        loss_b: torch.Tensor,
        loss_f: torch.Tensor,
        scheduled_scale: float,
    ) -> GradientMetrics:
        g_prior = self._flat_grad(prior_loss, retain_graph=True)
        g_bellman = self._flat_grad(self.lambda_bellman * bellman_loss, retain_graph=True)
        g_q = self._flat_grad(self.lambda_q * q_loss, retain_graph=True)
        g_b = self._flat_grad(self.lambda_b * loss_b, retain_graph=True)
        g_f = self._flat_grad(self.lambda_f * loss_f, retain_graph=True)

        g_aux = g_bellman + g_q + g_b + g_f

        prior_norm = float(g_prior.norm())
        bellman_norm = float(g_bellman.norm())
        q_norm = float(g_q.norm())
        b_norm = float(g_b.norm())
        f_norm = float(g_f.norm())
        aux_norm = float(g_aux.norm())

        if prior_norm > 1e-8 and aux_norm > 1e-8:
            cosine = float(F.cosine_similarity(g_prior.unsqueeze(0), g_aux.unsqueeze(0)))
        else:
            cosine = 0.0

        p_norm_eps = max(prior_norm, 1e-8)
        return GradientMetrics(
            prior_norm=prior_norm,
            bellman_norm=bellman_norm,
            q_norm=q_norm,
            b_norm=b_norm,
            f_norm=f_norm,
            aux_norm=aux_norm,
            prior_aux_cosine=cosine,
            bellman_to_prior=bellman_norm / p_norm_eps,
            q_to_prior=q_norm / p_norm_eps,
            b_to_prior=b_norm / p_norm_eps,
            f_to_prior=f_norm / p_norm_eps,
            aux_to_prior=aux_norm / p_norm_eps,
            effective_scale=scheduled_scale,
        )


# ============================================================================== 
# Distribution Evaluation & Plotting
# ============================================================================== 
def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    projections: int = 64,
    seed: int = 2026,
) -> float:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Inputs must be 2D arrays with the same feature dimension.")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(x.shape[1], projections)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    x_proj = np.sort(x @ directions, axis=0)
    y_proj = np.sort(y @ directions, axis=0)
    count = min(len(x_proj), len(y_proj))
    if len(x_proj) != count:
        x_proj = x_proj[np.linspace(0, len(x_proj) - 1, count, dtype=np.int64)]
    if len(y_proj) != count:
        y_proj = y_proj[np.linspace(0, len(y_proj) - 1, count, dtype=np.int64)]
    return float(np.mean(np.mean(np.abs(x_proj - y_proj), axis=0)))


def distribution_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    swd_projections: int,
    seed: int,
) -> dict[str, float]:
    real_cov = np.cov(real, rowvar=False)
    generated_cov = np.cov(generated, rowvar=False)
    return {
        "mean_mae": float(np.mean(np.abs(generated.mean(0) - real.mean(0)))),
        "std_mae": float(np.mean(np.abs(generated.std(0) - real.std(0)))),
        "covariance_relative_frobenius": float(
            np.linalg.norm(generated_cov - real_cov, ord="fro")
            / (np.linalg.norm(real_cov, ord="fro") + 1e-12)
        ),
        "sliced_wasserstein": sliced_wasserstein(
            real, generated, projections=swd_projections, seed=seed
        ),
        "real_global_mean": float(real.mean()),
        "real_global_std": float(real.std()),
        "generated_global_mean": float(generated.mean()),
        "generated_global_std": float(generated.std()),
    }


@torch.no_grad()
def generate_normalized_raw(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    total_samples: int,
    transition_dim: int,
    batch_size: int,
    ddim_steps: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    chunks: list[np.ndarray] = []
    remaining = total_samples
    while remaining > 0:
        current = min(remaining, batch_size)
        generated = diffusion.sample_ddim(
            model=model,
            shape=(current, transition_dim),
            ddim_steps=ddim_steps,
            eta=0.0,
            generator=generator,
        )
        chunks.append(generated.cpu().numpy())
        remaining -= current
    return np.concatenate(chunks, axis=0)


def save_training_curves(history: dict[str, list], path: Path) -> None:
    epochs = history.get("epoch", [])
    if not epochs:
        return

    # Keep only the loss curves requested for the final training figure.
    # Guidance interaction, gradient norms, and distribution-quality panels are omitted.
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))

    def plot_if(ax, key, label, **kwargs):
        values = history.get(key, [])
        if len(values) == len(epochs):
            ax.plot(epochs, values, label=label, **kwargs)

    plot_if(axes[0, 0], "train_prior_loss", "Train prior")
    plot_if(axes[0, 0], "val_prior_loss", "Val prior")
    axes[0, 0].set_title("Diffusion prior loss")

    plot_if(axes[0, 1], "train_total_loss", "Train total")
    plot_if(axes[0, 1], "val_total_loss", "Val total")
    axes[0, 1].set_title("Total objective")

    plot_if(axes[1, 0], "train_bellman_loss", "Train Bellman")
    plot_if(axes[1, 0], "val_bellman_loss", "Val Bellman")
    plot_if(axes[1, 0], "train_bellman_residual_mse", "Train residual MSE")
    plot_if(axes[1, 0], "val_bellman_residual_mse", "Val residual MSE")
    axes[1, 0].set_title("Bellman objective / residual")

    plot_if(axes[1, 1], "train_q_loss", "Train Q")
    plot_if(axes[1, 1], "val_q_loss", "Val Q")
    axes[1, 1].set_title("MetaMotivo-style Q loss")

    plot_if(axes[2, 0], "train_loss_b", "Train L_B")
    plot_if(axes[2, 0], "val_loss_b", "Val L_B")
    plot_if(axes[2, 0], "train_loss_f", "Train L_F")
    plot_if(axes[2, 0], "val_loss_f", "Val L_F")
    axes[2, 0].set_title("B/F representation losses")

    plot_if(axes[2, 1], "train_loss_obs", "Train L_obs")
    plot_if(axes[2, 1], "val_loss_obs", "Val L_obs")
    axes[2, 1].set_title("Physics consistency loss (L_obs)")

    for ax in fig.axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ==============================================================================
# Trainer
# ==============================================================================
class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        set_seed(config.train.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(config.paths.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        self.data = build_data(config, self.device)
        self._save_split()
        self.constraint = self._build_constraint()
        self.expert_buffer = self._build_expert_buffer()
        self.model, self.ema = self._build_model()
        self.diffusion = GaussianDiffusion(
            timesteps=config.model.timesteps,
            schedule="cosine",
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.parameters = list(self.model.parameters())
        self.balancer = GradientTracker(
            self.parameters,
            lambda_bellman=config.guidance.lambda_bellman,
            lambda_q=config.guidance.lambda_q,
            lambda_b=config.guidance.lambda_b,
            lambda_f=config.guidance.lambda_f,
        )

        self.history: dict[str, list] = defaultdict(list)
        self.best_val_prior = float("inf")
        self.best_guided_score = float("inf")
        self.best_guided_prior = float("inf")
        self.has_guided_checkpoint = False
        self.prior_reference = float("nan")
        self.prior_limit = float("inf")
        self.adaptive_multiplier = 1.0

        self.best_prior_path = self.save_dir / "fb_guided_diffusion_raw_best_prior.pt"
        self.best_path = self.save_dir / "fb_guided_diffusion_raw_best.pt"

        print(f"[Device] {self.device}")
        print(f"[Model] Parameters: {sum(p.numel() for p in self.parameters):,}")
        print(f"[Training] Steps/epoch: {len(self.data.train_loader):,}")

    def _save_config(self) -> None:
        config_path = self.save_dir / "experiment_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
        print(f"[Config Saved] Training parameters saved to: {config_path}")

    def _save_split(self) -> None:
        data = self.data.module
        np.savez_compressed(
            self.save_dir / "fb_guided_diffusion_split_indices.npz",
            train_indices=data.train_indices,
            val_indices=data.val_indices,
            transition_mean=data.mean,
            transition_std=data.std,
            split_seed=np.asarray(self.config.train.split_seed),
            validation_fraction=np.asarray(self.config.train.validation_fraction),
        )

    def _build_constraint(self) -> FBRawTransitionConstraint:
        cfg = self.config
        fb_model = load_frozen_fb_model(self.device)
        return FBRawTransitionConstraint(
            fb_model=fb_model,
            transition_mean=self.data.mean,
            transition_std=self.data.std,
            device=self.device,
            discount=cfg.guidance.discount,
            q_pessimism_penalty=cfg.guidance.pessimism_penalty,
            q_actor_stddev_clip=cfg.guidance.actor_stddev_clip,
            action_clip=cfg.guidance.action_clip,
            train_goal_ratio=cfg.guidance.train_goal_ratio,
            expert_asm_ratio=cfg.guidance.expert_asm_ratio,
        ).to(self.device)

    def _build_expert_buffer(self):
        cfg = self.config
        if cfg.guidance.expert_asm_ratio <= 0.0:
            return None
        return load_expert_trajectory_buffer(
            motions=cfg.paths.expert_motions,
            motions_root=cfg.paths.expert_motions_root,
            device="cpu",
            sequence_length=cfg.guidance.expert_seq_length,
        )

    @torch.no_grad()
    def _sample_expert_encodings(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[torch.Tensor]:
        if self.config.guidance.expert_asm_ratio <= 0.0:
            return None
        if self.expert_buffer is None:
            raise RuntimeError("Expert AMASS buffer is not initialized.")

        seq_length = self.config.guidance.expert_seq_length
        # Match MetaMotivo TrajectoryBuffer semantics: sample trajectories uniformly,
        # then sample one contiguous L-step slice and encode its next observations.
        sample_count = int(math.ceil(num_samples / seq_length) * seq_length)
        num_slices = sample_count // seq_length

        priorities = self.expert_buffer.priorities.to(self.device)
        ep_indices = torch.multinomial(
            priorities,
            num_samples=num_slices,
            replacement=True,
            generator=generator,
        )
        uniforms = torch.rand(
            (num_slices,), device=self.device, generator=generator
        )
        ep_indices_cpu = ep_indices.cpu().tolist()
        uniforms_cpu = uniforms.cpu().tolist()

        next_obs_slices = []
        for ep_idx, uniform in zip(ep_indices_cpu, uniforms_cpu):
            episode = self.expert_buffer.storage[ep_idx]
            observation = episode["observation"]
            high = observation.shape[0] - seq_length - 1
            if high <= 0:
                raise RuntimeError(
                    "Expert trajectory is too short for the configured sequence length."
                )
            start = min(int(uniform * high), high - 1)
            next_obs_slices.append(
                observation[start + 1 : start + 1 + seq_length]
            )

        expert_next_obs = torch.cat(next_obs_slices, dim=0).to(
            self.device, non_blocking=True
        )
        expert_next_obs_fb = self.constraint._normalize_fb_obs(expert_next_obs)

        b_expert = self.constraint.fb_model._backward_map(expert_next_obs_fb).detach()
        z_expert = b_expert.view(
            num_slices, seq_length, b_expert.shape[-1]
        ).mean(dim=1)
        z_expert = self.constraint.fb_model.project_z(z_expert)
        z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)
        return z_expert[:num_samples]

    def _build_model(self) -> Tuple[nn.Module, EMA]:
        """Initialize a brand-new diffusion denoiser (scratch training)."""
        cfg = self.config
        model = RawTransitionDenoisingModel(
            transition_dim=cfg.model.transition_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_blocks=cfg.model.num_blocks,
        ).to(self.device)

        # No diffusion checkpoint / pretrained denoiser is loaded here.
        ema = EMA(model, decay=cfg.train.ema_decay)
        print("[Model Init] Raw diffusion denoiser initialized from scratch.")
        return model, ema

    def _scheduled_guidance(self, epoch: int) -> float:
        guidance = self.config.guidance
        return guidance_ramp(epoch, guidance.start_epoch, guidance.ramp_epochs) * self.adaptive_multiplier

    def train_epoch(self, epoch: int, scheduled_scale: float) -> dict[str, float]:
        cfg = self.config
        self.model.train()
        tracker = MeanTracker()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.train.seed * 100_000 + epoch)

        for batch_index, x_0 in enumerate(self.data.train_loader):
            x_0 = x_0.to(self.device, non_blocking=True)
            t = torch.randint(0, cfg.model.timesteps, (len(x_0),), device=self.device)
            x_t, target_v = self.diffusion.q_sample(x_0, t)
            v_pred = self.model(x_t, t)
            prior_loss = F.mse_loss(v_pred, target_v)

            predicted_x0 = self.diffusion.predict_x0_from_v(x_t, t, v_pred)
            fb_num_samples = min(len(x_0), cfg.guidance.batch_size)
            expert_encodings = self._sample_expert_encodings(
                fb_num_samples, generator=generator
            )
            fb = self.constraint(
                predicted_x0,
                max_samples=cfg.guidance.batch_size,
                target_x0_norm=x_0,
                expert_encodings=expert_encodings,
                generator=generator,
            )

            # Gradient diagnostics are expensive because GradientTracker.compute()
            # performs several additional autograd.grad passes. Measure only on the
            # first configured batch(es) of each epoch; the actual training loss is
            # still computed and optimized on every batch.
            grad = None
            if batch_index < cfg.train.gradient_diagnostic_batches_per_epoch:
                grad = self.balancer.compute(
                    prior_loss=prior_loss,
                    bellman_loss=fb["bellman_loss"],
                    q_loss=fb["q_loss"],
                    loss_b=fb["loss_b"],
                    loss_f=fb["loss_f"],
                    scheduled_scale=scheduled_scale,
                )

            total_loss = prior_loss + scheduled_scale * (
                cfg.guidance.lambda_bellman * fb["bellman_loss"]
                + cfg.guidance.lambda_q * fb["q_loss"]
                + cfg.guidance.lambda_b * fb["loss_b"]
                + cfg.guidance.lambda_f * fb["loss_f"]
            )

            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite total loss during training.")

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if cfg.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters, cfg.train.max_grad_norm)
            self.optimizer.step()
            self.ema.update(self.model)

            # Loss statistics are averaged over all training batches.
            tracker.update(
                total_loss=total_loss,
                prior_loss=prior_loss,
                bellman_loss=fb["bellman_loss"],
                bellman_residual_mse=fb["bellman_residual_mse"],
                bellman_source=fb["bellman_source"],
                q_loss=fb["q_loss"],
                loss_b=fb["loss_b"],
                loss_f=fb["loss_f"],
            )

            # Gradient statistics represent only the diagnostic batch(es), not an
            # all-batch epoch average. With the default N=1 this is one measurement
            # per epoch, taken from the first training batch.
            if grad is not None:
                tracker.update(
                    prior_grad_norm=grad.prior_norm,
                    bellman_grad_norm=grad.bellman_norm,
                    q_grad_norm=grad.q_norm,
                    b_grad_norm=grad.b_norm,
                    f_grad_norm=grad.f_norm,
                    aux_grad_norm=grad.aux_norm,
                    prior_aux_cosine=grad.prior_aux_cosine,
                    bellman_to_prior=grad.bellman_to_prior,
                    q_to_prior=grad.q_to_prior,
                    b_to_prior=grad.b_to_prior,
                    f_to_prior=grad.f_to_prior,
                    aux_to_prior=grad.aux_to_prior,
                    effective_scale=grad.effective_scale,
                )

        metrics = tracker.averages()
        return metrics

    def _validate(self, reference_scale: float) -> dict[str, float]:
        cfg = self.config
        self.ema.shadow.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.evaluation.seed)

        prior_squared_error = 0.0
        prior_elements = 0
        guidance_sums = {
            "bellman_loss": 0.0,
            "bellman_residual_mse": 0.0,
            "bellman_source": 0.0,
            "q_loss": 0.0,
            "loss_b": 0.0,
            "loss_f": 0.0,
        }
        guidance_count = 0

        with torch.no_grad():
            for batch_index, x_0 in enumerate(self.data.val_loader):
                x_0 = x_0.to(self.device, non_blocking=True)
                t = torch.randint(
                    0,
                    cfg.model.timesteps,
                    (len(x_0),),
                    device=self.device,
                    generator=generator,
                )
                noise = torch.randn(
                    x_0.shape,
                    device=self.device,
                    dtype=x_0.dtype,
                    generator=generator,
                )
                x_t, target_v = self.diffusion.q_sample(x_0, t, noise=noise)
                v_pred = self.ema.shadow(x_t, t)
                prior_squared_error += F.mse_loss(
                    v_pred, target_v, reduction="sum"
                ).item()
                prior_elements += target_v.numel()

                if batch_index < cfg.guidance.validation_batches:
                    predicted_x0 = self.diffusion.predict_x0_from_v(x_t, t, v_pred)
                    fb_num_samples = min(len(x_0), cfg.guidance.batch_size)
                    expert_encodings = self._sample_expert_encodings(
                        fb_num_samples, generator=generator
                    )
                    fb = self.constraint(
                        predicted_x0,
                        max_samples=cfg.guidance.batch_size,
                        target_x0_norm=x_0,
                        expert_encodings=expert_encodings,
                        generator=generator,
                    )
                    for key in guidance_sums:
                        guidance_sums[key] += float(fb[key].item())
                    guidance_count += 1

        metrics = {
            "prior_loss": prior_squared_error / max(prior_elements, 1),
            **{
                key: value / max(guidance_count, 1)
                for key, value in guidance_sums.items()
            },
        }
        metrics["total_loss"] = metrics["prior_loss"] + reference_scale * (
            cfg.guidance.lambda_bellman * metrics["bellman_loss"]
            + cfg.guidance.lambda_q * metrics["q_loss"]
            + cfg.guidance.lambda_b * metrics["loss_b"]
            + cfg.guidance.lambda_f * metrics["loss_f"]
        )
        return metrics

    def _evaluate_generation_final(self) -> None:
        """Run expensive generated-vs-real distribution diagnostics once, post-training."""
        cfg = self.config
        epoch = cfg.train.epochs
        count = min(cfg.evaluation.final_samples, len(self.data.val_indices))
        val_idx = self.data.val_indices[:count]
        real = self.data.module.data[val_idx].cpu().numpy()
        generated = generate_normalized_raw(
            model=self.ema.shadow,
            diffusion=self.diffusion,
            total_samples=count,
            transition_dim=cfg.model.transition_dim,
            batch_size=cfg.evaluation.batch_size,
            ddim_steps=cfg.evaluation.ddim_steps,
            device=self.device,
            seed=cfg.evaluation.seed,
        )
        metrics = distribution_metrics(
            real=real,
            generated=generated,
            swd_projections=cfg.evaluation.swd_projections,
            seed=cfg.evaluation.seed,
        )
        self.history["generation_epoch"].append(epoch)
        for key, value in metrics.items():
            self.history[f"generation_{key}"].append(float(value))

        fb_cov_metrics = self._evaluate_final_fb_covariance(generated)
        for key, value in fb_cov_metrics.items():
            self.history[f"final_{key}"].append(float(value))

        print(
            f"[Generation Eval] epoch={epoch} n={count:,} "
            f"SWD={metrics['sliced_wasserstein']:.5f} "
            f"CovRelF={metrics['covariance_relative_frobenius']:.5f}"
        )

    @torch.no_grad()
    def _evaluate_final_fb_covariance(self, generated_norm: np.ndarray) -> dict[str, float]:
        """Compute the expensive FB B-covariance diagnostics once, post-training."""
        x_norm = torch.as_tensor(generated_norm, dtype=torch.float32, device=self.device)
        _, _, next_obs = self.constraint.unstandardize(x_norm)
        next_obs_fb = self.constraint._normalize_fb_obs(next_obs)
        q_B = self.constraint.fb_model._backward_map(next_obs_fb)
        cov_B = torch.matmul(q_B.T, q_B) / q_B.shape[0]
        symmetric_cov_B = 0.5 * (cov_B + cov_B.T)
        eigenvalues = torch.linalg.eigvalsh(symmetric_cov_B)
        metrics = {
            "q_cov_condition": float(torch.linalg.cond(cov_B).item()),
            "q_cov_min_eigenvalue": float(eigenvalues.min().item()),
            "q_cov_max_eigenvalue": float(eigenvalues.max().item()),
        }
        print(
            "[Final FB Covariance] "
            f"cond(B)={metrics['q_cov_condition']:.5g} "
            f"eig_min={metrics['q_cov_min_eigenvalue']:.5g} "
            f"eig_max={metrics['q_cov_max_eigenvalue']:.5g}"
        )
        return metrics

    def _update_adaptive_guidance(self, val_prior_loss: float) -> None:
        cfg = self.config.guidance
        if val_prior_loss >= cfg.prior_loss_threshold:
            self.adaptive_multiplier = max(cfg.adaptive_min, self.adaptive_multiplier * cfg.adaptive_down)
        else:
            self.adaptive_multiplier = min(cfg.adaptive_max, self.adaptive_multiplier * cfg.adaptive_up)

    def _initialize_checkpoint_selection(self) -> None:
        baseline = self._validate(reference_scale=0.0)
        self.prior_reference = baseline["prior_loss"]
        tolerance = self.config.train.best_checkpoint_prior_tolerance
        self.prior_limit = self.prior_reference * (1.0 + tolerance)
        self.history["checkpoint_prior_reference"].append(self.prior_reference)
        self.history["checkpoint_prior_limit"].append(self.prior_limit)
        print(
            "[Checkpoint Selection] "
            f"initial EMA prior={self.prior_reference:.6f}, "
            f"allowed degradation={100.0 * tolerance:.2f}%, "
            f"prior limit={self.prior_limit:.6f}"
        )

    def _guided_selection_score(self, val: Mapping[str, float]) -> float:
        g = self.config.guidance
        return float(
            g.lambda_bellman * val["bellman_loss"]
            + g.lambda_q * val["q_loss"]
            + g.lambda_b * val["loss_b"]
            + g.lambda_f * val["loss_f"]
        )

    def _update_best_checkpoints(self, epoch: int, val: Mapping[str, float]) -> None:
        val_prior = float(val["prior_loss"])

        if val_prior < self.best_val_prior:
            self.best_val_prior = val_prior
            self._save_checkpoint(self.best_prior_path, epoch)

        guided_score = self._guided_selection_score(val)
        prior_is_preserved = val_prior <= self.prior_limit
        if prior_is_preserved and guided_score < self.best_guided_score:
            self.best_guided_score = guided_score
            self.best_guided_prior = val_prior
            self.has_guided_checkpoint = True
            self._save_checkpoint(self.best_path, epoch)
            print(
                "[Best Guided] "
                f"epoch={epoch}, score={guided_score:.6g}, "
                f"prior={val_prior:.6f} <= {self.prior_limit:.6f}"
            )

    def _record(self, epoch: int, train: dict, val: dict, scheduled_scale: float, lr: float) -> None:
        self.history["epoch"].append(epoch)
        for k, v in train.items():
            self.history[f"train_{k}"].append(v)
        for k, v in val.items():
            self.history[f"val_{k}"].append(v)
        self.history["scheduled_scale"].append(scheduled_scale)
        self.history["learning_rate"].append(lr)

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.shadow.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "transition_mean": self.data.mean,
            "transition_std": self.data.std,
        }
        torch.save(checkpoint, path)

    def _log(self, epoch: int, train: dict, val: dict, scheduled_scale: float, lr: float) -> None:
        if epoch % self.config.train.log_interval == 0 or epoch == 1:
            obs_info = f" | L_obs {train.get('loss_obs', 0.0):.4f}/{val.get('loss_obs', 0.0):.4f}" if "loss_obs" in train else ""
            print(
                f"[Epoch {epoch:03d}/{self.config.train.epochs:03d}] "
                f"Train Loss {train['total_loss']:.4f} (Prior {train['prior_loss']:.4f}) | "
                f"Val Loss {val['total_loss']:.4f} (Prior {val['prior_loss']:.4f})"
                f"{obs_info} | "
                f"Scale {scheduled_scale:.3f} | LR {lr:.3e}"
            )

    def fit(self) -> None:
        cfg = self.config
        self._initialize_checkpoint_selection()

        for epoch in range(1, cfg.train.epochs + 1):
            lr = cfg.train.learning_rate
            scheduled_scale = self._scheduled_guidance(epoch)

            train = self.train_epoch(epoch, scheduled_scale)
            val = self._validate(scheduled_scale)
            self._update_adaptive_guidance(val["prior_loss"])
            self._record(epoch, train, val, scheduled_scale, lr)

            # Expensive DDIM generation + distribution diagnostics are deliberately
            # skipped during training and run once after the final epoch.
            self._update_best_checkpoints(epoch, val)

            if epoch in cfg.train.checkpoint_epochs:
                self._save_checkpoint(
                    self.save_dir / f"fb_guided_diffusion_raw_epoch_{epoch:04d}.pt",
                    epoch,
                )
            self._log(epoch, train, val, scheduled_scale, lr)

        # One post-training distribution check only: mean/std errors, covariance
        # relative Frobenius error, and sliced Wasserstein distance.
        self._evaluate_generation_final()
        self._save_final()

    def _save_final(self) -> None:
        cfg = self.config
        final_path = self.save_dir / "fb_guided_diffusion_raw_785dim.pt"
        self._save_checkpoint(final_path, cfg.train.epochs)

        results_json = self.save_dir / "fb_guided_diffusion_training_results.json"
        serializable_history = {
            key: [float(x) for x in values]
            for key, values in self.history.items()
        }
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=2)

        summary = {
            "best_prior_loss": float(self.best_val_prior),
            "best_guided_score": float(self.best_guided_score),
            "best_guided_prior": float(self.best_guided_prior),
            "prior_reference": float(self.prior_reference),
            "prior_limit": float(self.prior_limit),
            "has_guided_checkpoint": bool(self.has_guided_checkpoint),
            "best_prior_checkpoint": str(self.best_prior_path),
            "best_guided_checkpoint": str(self.best_path) if self.has_guided_checkpoint else None,
            "final_checkpoint": str(final_path),
            "hidden_dim": int(cfg.model.hidden_dim),
            "transition_dim": int(cfg.model.transition_dim),
            "action_clip": cfg.guidance.action_clip,
            "shared_split_npz": cfg.paths.shared_split_npz,
            "final_sliced_wasserstein": float(
                self.history.get("generation_sliced_wasserstein", [float("nan")])[-1]
            ),
            "final_covariance_relative_frobenius": float(
                self.history.get("generation_covariance_relative_frobenius", [float("nan")])[-1]
            ),
            "final_q_cov_condition": float(
                self.history.get("final_q_cov_condition", [float("nan")])[-1]
            ),
            "final_q_cov_min_eigenvalue": float(
                self.history.get("final_q_cov_min_eigenvalue", [float("nan")])[-1]
            ),
            "final_q_cov_max_eigenvalue": float(
                self.history.get("final_q_cov_max_eigenvalue", [float("nan")])[-1]
            ),
        }
        with open(
            self.save_dir / "fb_guided_diffusion_raw_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        curve_path = self.save_dir / "fb_guided_diffusion_raw_training_curves.png"
        save_training_curves(self.history, curve_path)

        print(f"[Training Complete] Checkpoints/history saved to: {self.save_dir}")
        print(f"[Training Curves] {curve_path}")
        print(f"[Summary] {self.save_dir / 'fb_guided_diffusion_raw_summary.json'}")


# ==============================================================================
# Automated Dataset Generation Function (Augmented Raw Transitions)
# ==============================================================================
def generate_fb_guided_raw_dataset(
    output_path: str,
    orig_buffer_path: str,
    diffusion_ckpt_path: str,
    num_samples: int = 500_000,
    ddim_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Generate synthetic [s,a,s',next_qpos,next_qvel] transitions from the trained model."""
    print("=================================================================")
    print(f" Generating {num_samples:,} Raw Transition Dataset -> {output_path}")
    print("=================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(diffusion_ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    eval_cfg = cfg.get("evaluation", {})
    if ddim_steps is None:
        ddim_steps = int(eval_cfg.get("ddim_steps", 50))
    if batch_size is None:
        batch_size = int(eval_cfg.get("batch_size", 1000))
    ddim_steps = min(int(ddim_steps), int(cfg["model"]["timesteps"]))
    print(f"[Sampling Config] ddim_steps={ddim_steps}, batch_size={batch_size}")
    transition_mean = np.array(ckpt["transition_mean"], dtype=np.float32)
    transition_std = np.array(ckpt["transition_std"], dtype=np.float32)

    model = RawTransitionDenoisingModel(
        transition_dim=cfg["model"]["transition_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_blocks=cfg["model"]["num_blocks"],
    ).to(device)

    state_dict = ckpt.get("ema_state_dict", ckpt.get("model_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=cfg["model"]["timesteps"],
        schedule="cosine",
        device=device,
    )

    all_obs = []
    all_action = []
    all_next_obs = []
    all_next_qpos = []
    all_next_qvel = []

    num_batches = math.ceil(num_samples / batch_size)
    t0 = time.time()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.get("evaluation", {}).get("seed", 2026)))

    for b in range(num_batches):
        cur_batch = min(batch_size, num_samples - b * batch_size)
        with torch.no_grad():
            x_norm = diffusion.sample_ddim(
                model=model,
                shape=(cur_batch, cfg["model"]["transition_dim"]),
                ddim_steps=ddim_steps,
                eta=0.0,
                generator=generator,
            ).cpu().numpy()

        x_phys = x_norm * transition_std + transition_mean

        all_obs.append(x_phys[:, :358])
        all_action.append(x_phys[:, 358:427])
        all_next_obs.append(x_phys[:, 427:785])

    obs_arr = np.concatenate(all_obs, axis=0)[:num_samples]
    action_arr = np.concatenate(all_action, axis=0)[:num_samples]
    next_obs_arr = np.concatenate(all_next_obs, axis=0)[:num_samples]

    action_clip = cfg.get("guidance", {}).get("action_clip", 1.0)
    if action_clip is not None:
        action_arr = np.clip(action_arr, -float(action_clip), float(action_clip))

    print(f"[Sampling] Complete in {time.time() - t0:.2f}s. Saving HDF5...")

    with h5py.File(orig_buffer_path, "r") as hf_orig:
        with h5py.File(output_path, "w") as hf_out:
            for k in hf_orig.keys():
                if k not in [
                    "observation",
                    "action",
                    "next_observation",
                ]:
                    data_k = hf_orig[k][:num_samples]
                    hf_out.create_dataset(k, data=data_k, compression="gzip")

            hf_out.create_dataset("observation", data=obs_arr, compression="gzip")
            hf_out.create_dataset("action", data=action_arr, compression="gzip")
            hf_out.create_dataset("next_observation", data=next_obs_arr, compression="gzip")

    print(f"[HDF5 Save] Successfully created {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    config = ExperimentConfig()
    print("=================================================================")
    print(" STARTING FB-GUIDED RAW TRANSITION DIFFUSION TRAINING")
    print(f" Save Directory: {config.paths.save_dir}")
    print("=================================================================")
    trainer = Trainer(config)
    trainer.fit()
