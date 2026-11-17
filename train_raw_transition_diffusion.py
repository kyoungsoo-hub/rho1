"""
Pure Raw Transition Diffusion Training — Complete Single-File Script.
====================================================================

Pure (unguided) diffusion version of train_raw_transition_diffusion_fb_guided.py.
Trains a raw 785-dim x=[s,a,s'] transition denoising diffusion model directly on
the replay buffer distribution without any FB / Bellman / Q / B / F guidance.

Key runtime choices:
- Raw diffusion denoiser: initialized from scratch (785-dim, hidden_dim=1024, 6 ResNet blocks).
- Noise schedule & parameterization: cosine schedule with Zero-Terminal-SNR and v-prediction.
- Split & standardization: identical default_rng(split_seed) split and train-only mean/std
  standardization as the FB-guided script (or optional SHARED_SPLIT_NPZ).
- Post-training evaluation: generated-vs-real distribution diagnostics (mean/std MAE,
  covariance relative Frobenius error, and sliced Wasserstein distance) via DDIM sampling.

Run directly via:
    python train_raw_transition_diffusion.py
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
    # Optional: point this to an existing split_indices.npz for identical train/val membership.
    shared_split_npz: str | None = None

    def resolve(self) -> None:
        env_save_dir = os.environ.get("SAVE_DIR")
        if env_save_dir:
            self.save_dir = env_save_dir
        elif not self.save_dir or self.save_dir.startswith("/home/kyoungsookim/rho"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = f"/home/kyoungsookim/rho1/pure_diffusion_raw_{timestamp}"

        self.shared_split_npz = os.environ.get(
            "SHARED_SPLIT_NPZ", self.shared_split_npz
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
    epochs: int = 5000
    batch_size: int = 4096
    learning_rate: float = 1.0e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    max_workers: int = 4
    max_grad_norm: float = 1.0
    log_interval: int = 10
    action_clip: float | None = 1.0
    checkpoint_epochs: tuple[int, ...] = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)
    


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

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    train_loss = history.get("train_prior_loss", [])
    val_loss = history.get("val_prior_loss", [])
    if len(train_loss) == len(epochs):
        ax.plot(epochs, train_loss, label="Train diffusion loss")
    if len(val_loss) == len(epochs):
        ax.plot(epochs, val_loss, label="Val diffusion loss")

    ax.set_title("Pure Raw Transition Diffusion Loss (v-MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
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
        self.best_val_prior = float("inf")
        self.best_epoch = 0

        self.best_prior_path = self.save_dir / "raw_transition_diffusion_best.pt"
        # Also save under model_best_prior.pt for compatibility with evaluation scripts.
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
        np.savez_compressed(
            self.save_dir / "diffusion_split_indices.npz",
            train_indices=data.train_indices,
            val_indices=data.val_indices,
            transition_mean=data.mean,
            transition_std=data.std,
            split_seed=np.asarray(self.config.train.split_seed),
            validation_fraction=np.asarray(self.config.train.validation_fraction),
        )

    def _build_model(self) -> Tuple[nn.Module, EMA]:
        """Initialize a brand-new diffusion denoiser from scratch."""
        cfg = self.config
        model = RawTransitionDenoisingModel(
            transition_dim=cfg.model.transition_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_blocks=cfg.model.num_blocks,
        ).to(self.device)

        ema = EMA(model, decay=cfg.train.ema_decay)
        print("[Model Init] Raw diffusion denoiser initialized from scratch.")
        return model, ema

    def train_epoch(self, epoch: int) -> dict[str, float]:
        cfg = self.config
        self.model.train()
        tracker = MeanTracker()

        for x_0 in self.data.train_loader:
            x_0 = x_0.to(self.device, non_blocking=True)
            t = torch.randint(0, cfg.model.timesteps, (len(x_0),), device=self.device)
            x_t, target_v = self.diffusion.q_sample(x_0, t)
            v_pred = self.model(x_t, t)
            prior_loss = F.mse_loss(v_pred, target_v)

            if not torch.isfinite(prior_loss):
                raise FloatingPointError("Non-finite diffusion loss during training.")

            self.optimizer.zero_grad(set_to_none=True)
            prior_loss.backward()
            if cfg.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters, cfg.train.max_grad_norm)
            self.optimizer.step()
            self.ema.update(self.model)

            tracker.update(
                total_loss=prior_loss,
                prior_loss=prior_loss,
            )

        return tracker.averages()

    def _validate(self) -> dict[str, float]:
        cfg = self.config
        self.ema.shadow.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.evaluation.seed)

        prior_squared_error = 0.0
        prior_elements = 0

        with torch.no_grad():
            for x_0 in self.data.val_loader:
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

        prior_loss = prior_squared_error / max(prior_elements, 1)
        return {
            "prior_loss": prior_loss,
            "total_loss": prior_loss,
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
        val_prior = float(val["prior_loss"])
        if val_prior < self.best_val_prior:
            self.best_val_prior = val_prior
            self.best_epoch = epoch
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
                f"Train Loss {train['prior_loss']:.4f} | "
                f"Val Loss {val['prior_loss']:.4f} "
                f"(Best {self.best_val_prior:.4f} @ ep {self.best_epoch}) | "
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
                    self.save_dir / f"raw_transition_diffusion_epoch_{epoch:04d}.pt",
                    epoch,
                )
            self._log(epoch, train, val, lr)

        self._evaluate_generation_final()
        self._save_final()

    def _save_final(self) -> None:
        cfg = self.config
        final_path = self.save_dir / "raw_transition_diffusion_785dim.pt"
        compat_final_path = self.save_dir / "model_final.pt"
        self._save_checkpoint(final_path, cfg.train.epochs)
        self._save_checkpoint(compat_final_path, cfg.train.epochs)

        results_json = self.save_dir / "raw_transition_diffusion_training_results.json"
        serializable_history = {
            key: [float(x) for x in values]
            for key, values in self.history.items()
        }
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=2)

        summary = {
            "best_prior_loss": float(self.best_val_prior),
            "best_epoch": int(self.best_epoch),
            "best_checkpoint": str(self.best_prior_path),
            "final_checkpoint": str(final_path),
            "hidden_dim": int(cfg.model.hidden_dim),
            "transition_dim": int(cfg.model.transition_dim),
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
            self.save_dir / "raw_transition_diffusion_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        curve_path = self.save_dir / "raw_transition_diffusion_training_curves.png"
        save_training_curves(self.history, curve_path)

        print(f"[Training Complete] Checkpoints/history saved to: {self.save_dir}")
        print(f"[Training Curves] {curve_path}")
        print(f"[Summary] {self.save_dir / 'raw_transition_diffusion_summary.json'}")


# ==============================================================================
# Automated Dataset Generation Function (Raw Transitions)
# ==============================================================================
def generate_raw_transition_dataset(
    output_path: str,
    orig_buffer_path: str,
    diffusion_ckpt_path: str,
    num_samples: int = 500_000,
    ddim_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Generate synthetic [s,a,s'] transitions from the trained diffusion model."""
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

    action_clip = cfg.get("train", {}).get(
        "action_clip", cfg.get("guidance", {}).get("action_clip", 1.0)
    )
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

    print(
        f"[HDF5 Save] Successfully created {output_path} "
        f"({os.path.getsize(output_path)/(1024*1024):.2f} MB)"
    )


if __name__ == "__main__":
    config = ExperimentConfig()
    print("=================================================================")
    print(" STARTING PURE RAW TRANSITION DIFFUSION TRAINING")
    print(f" Save Directory: {config.paths.save_dir}")
    print("=================================================================")
    trainer = Trainer(config)
    trainer.fit()
