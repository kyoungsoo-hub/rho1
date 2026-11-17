"""
FB-Reconstruction Guided Raw Transition Diffusion Training — Complete Single-File Script.
=========================================================================================

Trains/fine-tunes a raw 785-dim x=[s,a,s'] transition denoising diffusion model with
Frozen MetaMotivo FB Representation Reconstruction Losses (L_B and L_F) only:
- Starts from the pretrained Pure Diffusion checkpoint in:
  /home/kyoungsookim/rho1/pure_diffusion_raw_2026-11-12_20-40-29/raw_transition_diffusion_best.pt
- Uses the exact same train/val split (diffusion_split_indices.npz) and standardization.
- Adds FB Reconstruction losses without Bellman or Q losses, and without ramp-up:
      L_total = L_prior + lambda_b * L_B + lambda_f * L_F
  where lambda_b = 0.15, lambda_f = 0.005.
- Samples context z for L_F using the FB-CPR mixed probability distribution
  (20% batch goal state B(s'), 60% AMASS expert motion encoding, 20% random spherical noise).

Run directly via:
    python train_raw_transition_diffusion_fb_reconstruction.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

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
@dataclass
class PathConfig:
    raw_hdf5: str = "/home/kyoungsookim/rho1/metamotivo-M-1-datasets/data/buffer_inference_500000.hdf5"
    save_dir: str | None = None
    pretrained_ckpt: str | None = (
        "/home/kyoungsookim/rho1/pure_diffusion_raw_2026-11-12_20-40-29/raw_transition_diffusion_best.pt"
    )
    shared_split_npz: str | None = (
        "/home/kyoungsookim/rho1/pure_diffusion_raw_2026-11-12_20-40-29/diffusion_split_indices.npz"
    )
    # Expert AMASS motions used for expert-ASM z sampling in sample_mixed_z.
    expert_motions: str = "/home/kyoungsookim/humenv/data_preparation/test_train_split/large1_small1_train_0.1.txt"
    expert_motions_root: str = "/home/kyoungsookim/humenv/data_preparation/humenv_amass"

    def resolve(self) -> None:
        env_save_dir = os.environ.get("SAVE_DIR")
        if env_save_dir:
            self.save_dir = env_save_dir
        elif not self.save_dir or self.save_dir.startswith("/home/kyoungsookim/rho"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = f"/home/kyoungsookim/rho1/fb_recon_raw_{timestamp}"

        self.pretrained_ckpt = os.environ.get(
            "PRETRAINED_CKPT", self.pretrained_ckpt
        )
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
    batch_size: int = 4096
    learning_rate: float = 1.0e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    max_workers: int = 4
    max_grad_norm: float = 1.0
    log_interval: int = 10
    action_clip: float | None = 1.0
    checkpoint_epochs: tuple[int, ...] = (100, 200, 300, 400, 500)


@dataclass
class GuidanceConfig:
    # FB Reconstruction loss weights matching train_raw_transition_diffusion_fb_guided.py
    # Applied constantly from epoch 1 without ramp-up.
    lambda_b: float = 0.15
    lambda_f: float = 0.005
    batch_size: int = 3072
    validation_batches: int = 8
    action_clip: float | None = 1.0
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6
    expert_seq_length: int = 8


@dataclass
class EvaluationConfig:
    # Generated-vs-real distribution diagnostics are run once after training completes.
    ddim_steps: int = 250
    final_samples: int = 10_000
    batch_size: int = 4096
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
        if (
            train_indices.min() < 0
            or val_indices.min() < 0
            or train_indices.max() >= num_samples
            or val_indices.max() >= num_samples
        ):
            raise ValueError("Shared split contains out-of-range indices.")
        print(f"[Data] Loaded shared split: {split_path}")
    else:
        generator = np.random.default_rng(config.train.split_seed)
        all_indices = generator.permutation(num_samples)
        num_val = int(round(num_samples * config.train.validation_fraction))
        val_indices = all_indices[:num_val]
        train_indices = all_indices[num_val:]
        print("[Data] Using default_rng split algorithm.")

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
# Expert AMASS Trajectory Buffer (for expert-ASM z sampling in sample_mixed_z)
# ==============================================================================
def load_expert_trajectory_buffer(
    motions: str | Path,
    motions_root: str | Path,
    device: str,
    sequence_length: int,
):
    """Load HumEnv AMASS HDF5 motions into MetaMotivo's TrajectoryBuffer."""
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
# Model Architecture (785-dim Raw Transition Denoising MLP)
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
    """Residual MLP block (pre-norm ordering)."""

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
    """MLP denoising diffusion model on [s,a,s']."""

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
    """v-prediction diffusion with Zero-Terminal-SNR DDIM."""

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

            # Enforce terminal SNR exactly zero.
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
# FB Reconstruction Constraint (L_B and L_F with Mixed-Probability z)
# ==============================================================================
def load_frozen_fb_model(device: torch.device) -> nn.Module:
    """Load the frozen MetaMotivo FB model used for B/F representation reconstruction."""
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


class FBRawReconstructionConstraint(nn.Module):
    """FB Representation Reconstruction loss (L_B, L_F) on [s,a,s'] using mixed z."""

    def __init__(
        self,
        fb_model: nn.Module,
        transition_mean: np.ndarray,
        transition_std: np.ndarray,
        device: torch.device,
        action_clip: Optional[float] = 1.0,
        train_goal_ratio: float = 0.2,
        expert_asm_ratio: float = 0.6,
    ):
        super().__init__()
        self.fb_model = fb_model
        self.device = device
        self.action_clip = action_clip
        self.train_goal_ratio = float(train_goal_ratio)
        self.expert_asm_ratio = float(expert_asm_ratio)
        self.obs_dim = int(fb_model.cfg.obs_dim)
        self.action_dim = int(fb_model.cfg.action_dim)
        self.transition_dim = 2 * self.obs_dim + self.action_dim
        self.z_dim = int(fb_model.cfg.archi.z_dim)

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
        """Samples mixed z context matching MetaMotivo FB-CPR agent logic:
        - Goal state z (train_goal_ratio): Permutation of next_obs in batch passed through _backward_map.
        - Expert motion z (expert_asm_ratio): AMASS expert trajectory encodings.
        - Projected Gaussian z (remaining probability): Random prior noise projected onto hypersphere.
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

        # Expert motion z from AMASS trajectory encodings
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

        if self.action_clip is not None:
            action = action.clamp(-self.action_clip, self.action_clip)
        return obs, action, next_obs

    def _normalize_fb_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fb_model._obs_normalizer(obs)

    @staticmethod
    def _parallelize(features: torch.Tensor) -> torch.Tensor:
        return features.unsqueeze(0) if features.ndim == 2 else features

    def forward(
        self,
        predicted_x0_norm: torch.Tensor,
        max_samples: int,
        target_x0_norm: torch.Tensor,
        expert_encodings: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = predicted_x0_norm.shape[0]
        num_samples = min(batch_size, max_samples)
        if num_samples < batch_size:
            indices = torch.randperm(
                batch_size,
                device=predicted_x0_norm.device,
                generator=generator,
            )[:num_samples]
            predicted_x0_norm = predicted_x0_norm[indices]
            target_x0_norm = target_x0_norm[indices]

        obs_hat, action_hat, next_obs_hat = self.unstandardize(predicted_x0_norm)
        obs_hat_fb = self._normalize_fb_obs(obs_hat)
        next_obs_hat_fb = self._normalize_fb_obs(next_obs_hat)

        with torch.no_grad():
            obs_gt, action_gt, next_obs_gt = self.unstandardize(target_x0_norm)
            obs_gt_fb = self._normalize_fb_obs(obs_gt)
            next_obs_gt_fb = self._normalize_fb_obs(next_obs_gt)

            b_s_gt = self.fb_model._backward_map(obs_gt_fb)
            b_s_next_gt = self.fb_model._backward_map(next_obs_gt_fb)

            z_mixed = self.sample_mixed_z(
                num_samples=num_samples,
                next_obs_fb=next_obs_gt_fb,
                expert_encodings=expert_encodings,
                generator=generator,
            )
            f_sa_gt = self._parallelize(
                self.fb_model._forward_map(obs_gt_fb, z_mixed, action_gt)
            )

        b_s_hat = self.fb_model._backward_map(obs_hat_fb)
        b_s_next_hat = self.fb_model._backward_map(next_obs_hat_fb)
        f_sa_hat = self._parallelize(
            self.fb_model._forward_map(obs_hat_fb, z_mixed, action_hat)
        )

        loss_b = F.mse_loss(b_s_hat, b_s_gt) + F.mse_loss(
            b_s_next_hat, b_s_next_gt
        )
        loss_f = F.mse_loss(f_sa_hat, f_sa_gt)

        return {
            "loss_b": loss_b,
            "loss_f": loss_f,
        }


# ==============================================================================
# Metric Tracker & Distribution Evaluation
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def plot_if(ax, key, label, **kwargs):
        values = history.get(key, [])
        if len(values) == len(epochs):
            ax.plot(epochs, values, label=label, **kwargs)

    plot_if(axes[0, 0], "train_prior_loss", "Train prior")
    plot_if(axes[0, 0], "val_prior_loss", "Val prior")
    axes[0, 0].set_title("Diffusion Prior Loss (v-MSE)")

    plot_if(axes[0, 1], "train_total_loss", "Train total")
    plot_if(axes[0, 1], "val_total_loss", "Val total")
    axes[0, 1].set_title("Total Objective (Prior + 0.15*L_B + 0.005*L_F)")

    plot_if(axes[1, 0], "train_loss_b", "Train L_B")
    plot_if(axes[1, 0], "val_loss_b", "Val L_B")
    axes[1, 0].set_title("Backward Map Reconstruction Loss (L_B)")

    plot_if(axes[1, 1], "train_loss_f", "Train L_F")
    plot_if(axes[1, 1], "val_loss_f", "Val L_F")
    axes[1, 1].set_title("Forward Map Reconstruction Loss (L_F)")

    for ax in fig.axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
        handles, _ = ax.get_legend_handles_labels()
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

        self.history: dict[str, list] = defaultdict(list)
        self.best_val_total = float("inf")
        self.best_total_epoch = 0
        self.best_val_prior = float("inf")
        self.best_prior_epoch = 0

        # Primary best checkpoint (evaluated by deep_evaluate.py / evaluate_experiment.py)
        self.best_path = self.save_dir / "fb_guided_diffusion_raw_best.pt"
        self.compat_best_total_path = self.save_dir / "raw_transition_diffusion_best.pt"
        self.best_prior_path = self.save_dir / "fb_guided_diffusion_raw_best_prior.pt"
        self.compat_best_prior_path = self.save_dir / "model_best_prior.pt"

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
        for fname in ("diffusion_split_indices.npz", "fb_guided_diffusion_split_indices.npz"):
            np.savez_compressed(
                self.save_dir / fname,
                train_indices=data.train_indices,
                val_indices=data.val_indices,
                transition_mean=data.mean,
                transition_std=data.std,
                split_seed=np.asarray(self.config.train.split_seed),
                validation_fraction=np.asarray(self.config.train.validation_fraction),
            )

    def _build_constraint(self) -> FBRawReconstructionConstraint:
        cfg = self.config
        fb_model = load_frozen_fb_model(self.device)
        return FBRawReconstructionConstraint(
            fb_model=fb_model,
            transition_mean=self.data.mean,
            transition_std=self.data.std,
            device=self.device,
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
        """Initialize diffusion denoiser and load pretrained pure diffusion weights if configured."""
        cfg = self.config
        model = RawTransitionDenoisingModel(
            transition_dim=cfg.model.transition_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_blocks=cfg.model.num_blocks,
        ).to(self.device)

        ema = EMA(model, decay=cfg.train.ema_decay)

        pretrained_ckpt = cfg.paths.pretrained_ckpt
        if pretrained_ckpt:
            ckpt_path = Path(pretrained_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Pretrained diffusion checkpoint not found: {ckpt_path}"
                )
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model_state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict"))
            ema_state = ckpt.get("ema_state_dict", model_state)
            if model_state is None:
                raise KeyError(
                    f"Checkpoint {ckpt_path} does not contain model_state_dict or ema_state_dict."
                )
            model.load_state_dict(model_state)
            ema.shadow.load_state_dict(ema_state)
            ema.shadow.eval()
            loaded_epoch = ckpt.get("epoch", "unknown")
            print(
                f"[Model Init] Loaded pretrained diffusion checkpoint from: {ckpt_path} "
                f"(epoch={loaded_epoch})"
            )
        else:
            print("[Model Init] Raw diffusion denoiser initialized from scratch.")

        return model, ema

    def train_epoch(self, epoch: int) -> dict[str, float]:
        cfg = self.config
        self.model.train()
        tracker = MeanTracker()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.train.seed * 100_000 + epoch)

        for x_0 in self.data.train_loader:
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

            # Constant weights without ramp-up
            total_loss = (
                prior_loss
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

            tracker.update(
                total_loss=total_loss,
                prior_loss=prior_loss,
                loss_b=fb["loss_b"],
                loss_f=fb["loss_f"],
            )

        return tracker.averages()

    def _validate(self) -> dict[str, float]:
        cfg = self.config
        self.ema.shadow.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.evaluation.seed)

        prior_squared_error = 0.0
        prior_elements = 0
        guidance_sums = {
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

        prior_loss = prior_squared_error / max(prior_elements, 1)
        loss_b = guidance_sums["loss_b"] / max(guidance_count, 1)
        loss_f = guidance_sums["loss_f"] / max(guidance_count, 1)
        total_loss = (
            prior_loss
            + cfg.guidance.lambda_b * loss_b
            + cfg.guidance.lambda_f * loss_f
        )
        return {
            "prior_loss": prior_loss,
            "loss_b": loss_b,
            "loss_f": loss_f,
            "total_loss": total_loss,
        }

    def _evaluate_generation_final(self) -> None:
        """Run generated-vs-real distribution diagnostics once, post-training."""
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

        print(
            f"[Generation Eval] epoch={epoch} n={count:,} "
            f"SWD={metrics['sliced_wasserstein']:.5f} "
            f"CovRelF={metrics['covariance_relative_frobenius']:.5f}"
        )

    def _update_best_checkpoints(self, epoch: int, val: Mapping[str, float]) -> None:
        val_total = float(val["total_loss"])
        val_prior = float(val["prior_loss"])

        if val_total < self.best_val_total:
            self.best_val_total = val_total
            self.best_total_epoch = epoch
            self._save_checkpoint(self.best_path, epoch)
            self._save_checkpoint(self.compat_best_total_path, epoch)

        if val_prior < self.best_val_prior:
            self.best_val_prior = val_prior
            self.best_prior_epoch = epoch
            self._save_checkpoint(self.best_prior_path, epoch)
            self._save_checkpoint(self.compat_best_prior_path, epoch)

    def _record(self, epoch: int, train: dict, val: dict, lr: float) -> None:
        self.history["epoch"].append(epoch)
        for k, v in train.items():
            self.history[f"train_{k}"].append(v)
        for k, v in val.items():
            self.history[f"val_{k}"].append(v)
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

    def _log(self, epoch: int, train: dict, val: dict, lr: float) -> None:
        if epoch % self.config.train.log_interval == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:04d}/{self.config.train.epochs:04d}] "
                f"Total {train['total_loss']:.4f}/{val['total_loss']:.4f} "
                f"(Best {self.best_val_total:.4f} @ ep {self.best_total_epoch}) | "
                f"Prior {train['prior_loss']:.4f}/{val['prior_loss']:.4f} | "
                f"L_B {train['loss_b']:.4f}/{val['loss_b']:.4f} | "
                f"L_F {train['loss_f']:.4f}/{val['loss_f']:.4f} | "
                f"LR {lr:.3e}"
            )

    def fit(self) -> None:
        cfg = self.config

        for epoch in range(1, cfg.train.epochs + 1):
            lr = cfg.train.learning_rate

            train = self.train_epoch(epoch)
            val = self._validate()
            self._record(epoch, train, val, lr)
            self._update_best_checkpoints(epoch, val)

            if epoch in cfg.train.checkpoint_epochs:
                self._save_checkpoint(
                    self.save_dir / f"fb_recon_diffusion_raw_epoch_{epoch:04d}.pt",
                    epoch,
                )
            self._log(epoch, train, val, lr)

        self._evaluate_generation_final()
        self._save_final()

    def _save_final(self) -> None:
        cfg = self.config
        final_path = self.save_dir / "fb_guided_diffusion_raw_785dim.pt"
        compat_final_path = self.save_dir / "raw_transition_diffusion_785dim.pt"
        model_final_path = self.save_dir / "model_final.pt"
        self._save_checkpoint(final_path, cfg.train.epochs)
        self._save_checkpoint(compat_final_path, cfg.train.epochs)
        self._save_checkpoint(model_final_path, cfg.train.epochs)

        results_json = self.save_dir / "fb_recon_diffusion_training_results.json"
        serializable_history = {
            key: [float(x) for x in values]
            for key, values in self.history.items()
        }
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=2)

        summary = {
            "pretrained_ckpt": cfg.paths.pretrained_ckpt,
            "best_total_loss": float(self.best_val_total),
            "best_total_epoch": int(self.best_total_epoch),
            "best_prior_loss": float(self.best_val_prior),
            "best_prior_epoch": int(self.best_prior_epoch),
            "best_checkpoint": str(self.best_path),
            "best_prior_checkpoint": str(self.best_prior_path),
            "final_checkpoint": str(final_path),
            "hidden_dim": int(cfg.model.hidden_dim),
            "transition_dim": int(cfg.model.transition_dim),
            "lambda_b": float(cfg.guidance.lambda_b),
            "lambda_f": float(cfg.guidance.lambda_f),
            "train_goal_ratio": float(cfg.guidance.train_goal_ratio),
            "expert_asm_ratio": float(cfg.guidance.expert_asm_ratio),
            "action_clip": cfg.train.action_clip,
            "shared_split_npz": cfg.paths.shared_split_npz,
            "final_sliced_wasserstein": float(
                self.history.get("generation_sliced_wasserstein", [float("nan")])[-1]
            ),
            "final_covariance_relative_frobenius": float(
                self.history.get("generation_covariance_relative_frobenius", [float("nan")])[-1]
            ),
        }
        with open(
            self.save_dir / "fb_recon_diffusion_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        curve_path = self.save_dir / "fb_recon_diffusion_training_curves.png"
        save_training_curves(self.history, curve_path)

        print(f"[Training Complete] Checkpoints/history saved to: {self.save_dir}")
        print(f"[Training Curves] {curve_path}")
        print(f"[Summary] {self.save_dir / 'fb_recon_diffusion_summary.json'}")


if __name__ == "__main__":
    config = ExperimentConfig()
    print("=================================================================")
    print(" STARTING FB-RECONSTRUCTION RAW TRANSITION DIFFUSION TRAINING")
    print(f" Pretrained Checkpoint: {config.paths.pretrained_ckpt}")
    print(f" Save Directory: {config.paths.save_dir}")
    print("=================================================================")
    trainer = Trainer(config)
    trainer.fit()
