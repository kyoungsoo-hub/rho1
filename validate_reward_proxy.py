#!/usr/bin/env python3
"""
Validate deep_evaluate.py's observation/action proxy rewards against the official
HumEnv reward on the ORIGINAL MetaMotivo inference buffer.

The comparison is row-aligned:
    official: r_true(qpos', qvel', a)
    proxy   : r_proxy(s', a)
for the exact same transition rows.

Outputs
-------
1) reward_proxy_validation.csv
   Per-task reward agreement, top-k overlap, and z cosine metrics.
2) reward_proxy_validation.json
   Full configuration, metrics, and optional rollout details.
3) plots/*.png (optional)
   Per-task r_true vs r_proxy scatter plots.

Optional paired rollout validation compares policies prompted with z_true and
z_proxy under the same HumEnv seeds, using native HumEnv reward for evaluation.

Example
-------
python validate_reward_proxy.py \
    --buffer /path/to/buffer_inference_500000.hdf5 \
    --deep_evaluate_py /path/to/deep_evaluate.py \
    --model_name facebook/metamotivo-M-1 \
    --num_samples 10000 \
    --max_workers 16 \
    --save_plots

# Add a focused rollout check after the reward/z diagnostics:
python validate_reward_proxy.py \
    --buffer /path/to/buffer_inference_500000.hdf5 \
    --deep_evaluate_py /path/to/deep_evaluate.py \
    --num_samples 10000 \
    --rollout_tasks move-ego-0-2,move-ego-0-4,jump-2,split-1 \
    --rollout_seeds 10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import FlattenObservation

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import relabel


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _sanitize_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _sanitize_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _sanitize_json(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _load_deep_evaluate(path: Path):
    """Load the exact deep_evaluate.py being validated."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # deep_evaluate.py imports project-local modules; make its directory importable.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("deep_evaluate_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    required = ["META_45_TASKS", "compute_reward_from_obs", "run_single_rollout", "proxy_metadata"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"{path} is missing required symbols: {missing}")
    return module


def _h5_paths(hf: h5py.File) -> list[str]:
    paths: list[str] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    hf.visititems(visit)
    return paths


def _resolve_dataset(hf: h5py.File, candidates: Iterable[str]) -> h5py.Dataset:
    for key in candidates:
        if key in hf and isinstance(hf[key], h5py.Dataset):
            return hf[key]
    available = _h5_paths(hf)
    raise KeyError(f"None of {list(candidates)} found in HDF5. Available datasets: {available}")


def _deterministic_indices(total: int, count: int, seed: int) -> np.ndarray:
    count = min(int(count), int(total))
    if count <= 0:
        raise ValueError("num_samples must be positive")
    if count == total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=count, replace=False)
    # h5py fancy indexing requires monotonically increasing indices.
    idx.sort()
    return idx.astype(np.int64)


def load_aligned_original_rows(
    buffer_path: Path,
    num_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Read row-aligned next_qpos, next_qvel, action, next_observation."""
    with h5py.File(buffer_path, "r") as hf:
        next_obs_ds = _resolve_dataset(hf, ["next_observation", "next/observation"])
        action_ds = _resolve_dataset(hf, ["action"])
        next_qpos_ds = _resolve_dataset(hf, ["next_qpos", "next/qpos"])
        next_qvel_ds = _resolve_dataset(hf, ["next_qvel", "next/qvel"])

        total = int(next_obs_ds.shape[0])
        for name, ds in {
            "action": action_ds,
            "next_qpos": next_qpos_ds,
            "next_qvel": next_qvel_ds,
        }.items():
            if int(ds.shape[0]) != total:
                raise ValueError(f"Row-count mismatch: next_observation={total}, {name}={ds.shape[0]}")

        idx = _deterministic_indices(total, num_samples, seed)
        data = {
            "indices": idx,
            "next_observation": np.asarray(next_obs_ds[idx], dtype=np.float32),
            "action": np.asarray(action_ds[idx], dtype=np.float32),
            "next_qpos": np.asarray(next_qpos_ds[idx]),
            "next_qvel": np.asarray(next_qvel_ds[idx]),
        }
    return data


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """scipy.stats.rankdata(method='average') without requiring scipy."""
    x = np.asarray(x).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        # 1-based average rank, matching scipy convention.
        avg_rank = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _safe_pearson(_rankdata_average(x), _rankdata_average(y))


def _top_fraction_overlap(x: np.ndarray, y: np.ndarray, fraction: float) -> dict[str, float]:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = len(x)
    k = max(1, int(math.ceil(n * float(fraction))))
    # argpartition avoids sorting all N values.
    x_top = set(np.argpartition(x, n - k)[n - k :].tolist())
    y_top = set(np.argpartition(y, n - k)[n - k :].tolist())
    inter = len(x_top & y_top)
    union = len(x_top | y_top)
    return {
        "k": k,
        "overlap": inter / k,
        "jaccard": inter / union if union else 1.0,
    }


def reward_metrics(r_true: np.ndarray, r_proxy: np.ndarray) -> dict[str, float]:
    t = np.asarray(r_true, dtype=np.float64).reshape(-1)
    p = np.asarray(r_proxy, dtype=np.float64).reshape(-1)
    if t.shape != p.shape:
        raise ValueError(f"Reward shape mismatch: true={t.shape}, proxy={p.shape}")
    if len(t) == 0:
        raise ValueError("Cannot compute reward metrics on an empty sample.")
    if not np.isfinite(t).all():
        bad = int((~np.isfinite(t)).sum())
        raise ValueError(f"Official reward contains {bad} non-finite values.")
    if not np.isfinite(p).all():
        bad = int((~np.isfinite(p)).sum())
        raise ValueError(f"Proxy reward contains {bad} non-finite values.")
    err = p - t
    out: dict[str, float] = {
        "n": float(len(t)),
        "true_mean": float(t.mean()),
        "true_std": float(t.std()),
        "true_min": float(t.min()),
        "true_max": float(t.max()),
        "proxy_mean": float(p.mean()),
        "proxy_std": float(p.std()),
        "proxy_min": float(p.min()),
        "proxy_max": float(p.max()),
        "bias_proxy_minus_true": float(err.mean()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "pearson": _safe_pearson(t, p),
        "spearman": _safe_spearman(t, p),
    }
    for pct in (1, 5, 10):
        overlap = _top_fraction_overlap(t, p, pct / 100.0)
        out[f"top{pct}_overlap"] = float(overlap["overlap"])
        out[f"top{pct}_jaccard"] = float(overlap["jaccard"])
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(1, -1)
    bb = b.reshape(1, -1)
    return float(F.cosine_similarity(aa, bb).item())


@torch.no_grad()
def infer_z_pair(
    model: FBcprModel,
    next_obs: np.ndarray,
    r_true: np.ndarray,
    r_proxy: np.ndarray,
) -> dict[str, Any]:
    device = model.cfg.device
    obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    true_t = torch.as_tensor(r_true, dtype=torch.float32, device=device).reshape(-1, 1)
    proxy_t = torch.as_tensor(r_proxy, dtype=torch.float32, device=device).reshape(-1, 1)

    z_true = model.reward_inference(obs_t, true_t, weight=None)
    z_proxy = model.reward_inference(obs_t, proxy_t, weight=None)

    z_true_wr = model.reward_wr_inference(obs_t, true_t)
    z_proxy_wr = model.reward_wr_inference(obs_t, proxy_t)

    return {
        "z_true": z_true,
        "z_proxy": z_proxy,
        "z_true_wr": z_true_wr,
        "z_proxy_wr": z_proxy_wr,
        "z_cosine_unweighted": _cosine(z_true, z_proxy),
        "z_cosine_wr": _cosine(z_true_wr, z_proxy_wr),
        "z_true_norm": float(torch.linalg.vector_norm(z_true).item()),
        "z_proxy_norm": float(torch.linalg.vector_norm(z_proxy).item()),
        "z_true_wr_norm": float(torch.linalg.vector_norm(z_true_wr).item()),
        "z_proxy_wr_norm": float(torch.linalg.vector_norm(z_proxy_wr).item()),
    }


def official_rewards_for_task(
    task_name: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    max_workers: int,
    process_executor: bool,
    process_context: str,
) -> np.ndarray:
    """Use the exact relabel path used by MetaMotivo RewardWrapper."""
    env, _ = make_humenv(task=task_name)
    try:
        rewards = relabel(
            env,
            qpos,
            qvel,
            action,
            env.unwrapped.task,
            max_workers=max_workers,
            process_executor=process_executor,
            process_context=process_context,
        )
    finally:
        env.close()
    return np.asarray(rewards, dtype=np.float32).reshape(-1, 1)


def observation_consistency_check(
    stored_next_obs: np.ndarray,
    next_qpos: np.ndarray,
    next_qvel: np.ndarray,
    num_checks: int,
    seed: int,
) -> dict[str, Any]:
    """
    Reconstruct flattened observations from qpos/qvel and compare with stored next_observation.

    This is intentionally a small sanity check because reset is performed row-by-row.
    """
    n = len(stored_next_obs)
    num_checks = min(int(num_checks), n)
    if num_checks <= 0:
        return {"num_checked": 0}

    rng = np.random.default_rng(seed)
    check_idx = np.sort(rng.choice(n, size=num_checks, replace=False))

    env, _ = make_humenv(
        num_envs=1,
        task="move-ego-0-0",
        state_init="Default",
        wrappers=[FlattenObservation],
    )
    max_abs_values: list[float] = []
    mae_values: list[float] = []
    try:
        for i in check_idx:
            obs, _ = env.reset(options={"qpos": next_qpos[i], "qvel": next_qvel[i]})
            regenerated = np.asarray(obs, dtype=np.float32).reshape(-1)
            stored = np.asarray(stored_next_obs[i], dtype=np.float32).reshape(-1)
            if regenerated.shape != stored.shape:
                raise ValueError(
                    f"Observation shape mismatch during consistency check: regenerated={regenerated.shape}, stored={stored.shape}"
                )
            diff = regenerated - stored
            max_abs_values.append(float(np.max(np.abs(diff))))
            mae_values.append(float(np.mean(np.abs(diff))))
    finally:
        env.close()

    return {
        "num_checked": num_checks,
        "max_abs_error_max": float(np.max(max_abs_values)),
        "max_abs_error_mean": float(np.mean(max_abs_values)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_max": float(np.max(mae_values)),
    }


def save_scatter_plot(
    path: Path,
    task: str,
    r_true: np.ndarray,
    r_proxy: np.ndarray,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(r_true).reshape(-1)
    y = np.asarray(r_proxy).reshape(-1)
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), max_points, replace=False)
        x = x[idx]
        y = y[idx]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, alpha=0.35)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("Official HumEnv reward")
    ax.set_ylabel("Proxy reward from (s', a)")
    ax.set_title(task)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_task_list(value: Optional[str], all_tasks: list[str]) -> list[str]:
    if value is None or not value.strip():
        return []
    if value.strip().lower() == "all":
        return list(all_tasks)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in all_tasks]
    if unknown:
        raise ValueError(f"Unknown rollout tasks: {unknown}")
    return requested


def paired_rollout_validation(
    deep_module: Any,
    model: FBcprModel,
    task_name: str,
    z_true: torch.Tensor,
    z_proxy: torch.Tensor,
    base_seed: int,
    num_seeds: int,
    num_steps: int,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for i in range(num_seeds):
        seed = base_seed + i
        true_roll = deep_module.run_single_rollout(
            model=model,
            task_name=task_name,
            z=z_true,
            seed=seed,
            num_steps=num_steps,
            video_output_path=None,
        )
        proxy_roll = deep_module.run_single_rollout(
            model=model,
            task_name=task_name,
            z=z_proxy,
            seed=seed,
            num_steps=num_steps,
            video_output_path=None,
        )
        q_true = float(true_roll["q_values"]["Q_gamma_0.99"])
        q_proxy = float(proxy_roll["q_values"]["Q_gamma_0.99"])
        ret_true = float(true_roll["undiscounted_return"])
        ret_proxy = float(proxy_roll["undiscounted_return"])
        pairs.append(
            {
                "seed": seed,
                "native_q99_true_z": q_true,
                "native_q99_proxy_z": q_proxy,
                "native_q99_diff_proxy_minus_true": q_proxy - q_true,
                "native_return_true_z": ret_true,
                "native_return_proxy_z": ret_proxy,
                "native_return_diff_proxy_minus_true": ret_proxy - ret_true,
            }
        )

    qdiff = np.asarray([p["native_q99_diff_proxy_minus_true"] for p in pairs], dtype=np.float64)
    rdiff = np.asarray([p["native_return_diff_proxy_minus_true"] for p in pairs], dtype=np.float64)
    return {
        "n": len(pairs),
        "native_q99_diff_mean": float(qdiff.mean()),
        "native_q99_diff_std": float(qdiff.std(ddof=1)) if len(qdiff) > 1 else 0.0,
        "native_return_diff_mean": float(rdiff.mean()),
        "native_return_diff_std": float(rdiff.std(ddof=1)) if len(rdiff) > 1 else 0.0,
        "pairs": pairs,
    }


# -----------------------------------------------------------------------------
# Main validation
# -----------------------------------------------------------------------------


def run_validation(args: argparse.Namespace) -> tuple[Path, Path]:
    t0 = time.time()
    buffer_path = Path(args.buffer).resolve()
    deep_path = Path(args.deep_evaluate_py).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    deep = _load_deep_evaluate(deep_path)
    all_tasks = list(deep.META_45_TASKS)
    tasks = all_tasks if not args.tasks else parse_task_list(args.tasks, all_tasks)
    rollout_tasks = parse_task_list(args.rollout_tasks, all_tasks)

    print("=" * 88)
    print(" Reward Proxy Validation: official HumEnv reward vs deep_evaluate proxy")
    print(f" Buffer      : {buffer_path}")
    print(f" deep_eval   : {deep_path}")
    print(f" Tasks       : {len(tasks)}")
    print(f" Samples     : {args.num_samples:,}")
    print("=" * 88)

    data = load_aligned_original_rows(buffer_path, args.num_samples, args.seed)
    next_obs = data["next_observation"]
    action = data["action"]
    next_qpos = data["next_qpos"]
    next_qvel = data["next_qvel"]

    consistency: dict[str, Any] = {"num_checked": 0}
    if args.check_obs_consistency > 0:
        print(f"[Sanity] Checking {args.check_obs_consistency} qpos/qvel -> observation reconstructions...")
        consistency = observation_consistency_check(
            next_obs,
            next_qpos,
            next_qvel,
            num_checks=args.check_obs_consistency,
            seed=args.seed + 77,
        )
        print(
            "[Sanity] "
            f"mean MAE={consistency.get('mae_mean', float('nan')):.3e}, "
            f"max |err|={consistency.get('max_abs_error_max', float('nan')):.3e}"
        )

    model: Optional[FBcprModel] = None
    if not args.no_z or rollout_tasks:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Model] Loading {args.model_name} on {device}...")
        model = FBcprModel.from_pretrained(args.model_name, device=str(device))

    plot_dir = out_dir / "plots"
    if args.save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    full_results: dict[str, Any] = {}
    rollout_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for task_i, task_name in enumerate(tasks, 1):
        print(f"\n[{task_i:02d}/{len(tasks):02d}] {task_name}")
        task_t0 = time.time()

        r_true = official_rewards_for_task(
            task_name,
            next_qpos,
            next_qvel,
            action,
            max_workers=args.max_workers,
            process_executor=args.process_executor,
            process_context=args.process_context,
        )
        try:
            r_proxy = deep.compute_reward_from_obs(next_obs, task_name, action=action)
        except TypeError:
            r_proxy = deep.compute_reward_from_obs(next_obs, task_name)
        r_proxy = np.asarray(r_proxy, dtype=np.float32).reshape(-1, 1)

        metrics = reward_metrics(r_true, r_proxy)
        meta = deep.proxy_metadata(task_name)
        metrics.update(
            {
                "task": task_name,
                "family": meta.family,
                "proxy_confidence": meta.confidence,
                "proxy_note": meta.note,
            }
        )

        z_info: dict[str, Any] = {}
        need_z_for_task = (not args.no_z) or (task_name in rollout_tasks)
        if need_z_for_task:
            assert model is not None
            z_info = infer_z_pair(model, next_obs, r_true, r_proxy)
            if not args.no_z:
                metrics.update(
                    {
                        "z_cosine_unweighted": z_info["z_cosine_unweighted"],
                        "z_cosine_wr": z_info["z_cosine_wr"],
                        "z_true_norm": z_info["z_true_norm"],
                        "z_proxy_norm": z_info["z_proxy_norm"],
                        "z_true_wr_norm": z_info["z_true_wr_norm"],
                        "z_proxy_wr_norm": z_info["z_proxy_wr_norm"],
                    }
                )
            if task_name in rollout_tasks:
                rollout_cache[task_name] = (
                    z_info["z_true"], z_info["z_proxy"], z_info["z_true_wr"], z_info["z_proxy_wr"]
                )

        if args.save_plots:
            save_scatter_plot(
                plot_dir / f"{task_name}.png",
                task_name,
                r_true,
                r_proxy,
                max_points=args.plot_max_points,
                seed=args.seed + task_i,
            )

        full_results[task_name] = {
            "proxy_metadata": {
                "family": meta.family,
                "confidence": meta.confidence,
                "note": meta.note,
            },
            "metrics": {k: v for k, v in metrics.items() if k not in {"proxy_note"}},
        }
        rows.append(metrics)

        msg = (
            f"Pearson={metrics['pearson']:.4f} | Spearman={metrics['spearman']:.4f} | "
            f"Top5={metrics['top5_overlap']:.3f}"
        )
        if not args.no_z:
            msg += (
                f" | z_cos={metrics['z_cosine_unweighted']:.4f}"
                f" | WR-z_cos={metrics['z_cosine_wr']:.4f}"
            )
        print(f"  {msg} | {time.time() - task_t0:.1f}s")

    # Optional native paired rollout validation.
    rollout_results: dict[str, Any] = {}
    if rollout_tasks:
        assert model is not None
        missing = [t for t in rollout_tasks if t not in rollout_cache]
        if missing:
            raise RuntimeError(
                f"Rollout tasks {missing} were not evaluated for z. Include them in --tasks or omit --tasks."
            )
        print("\n" + "=" * 88)
        print(f" Paired native rollout validation ({args.rollout_inference} z, {args.rollout_seeds} seeds/task)")
        print("=" * 88)
        for i, task_name in enumerate(rollout_tasks):
            z_true, z_proxy, z_true_wr, z_proxy_wr = rollout_cache[task_name]
            if args.rollout_inference == "wr":
                zt, zp = z_true_wr, z_proxy_wr
            else:
                zt, zp = z_true, z_proxy
            base_seed = args.rollout_seed + i * 10000
            result = paired_rollout_validation(
                deep,
                model,
                task_name,
                zt,
                zp,
                base_seed=base_seed,
                num_seeds=args.rollout_seeds,
                num_steps=args.rollout_steps,
            )
            rollout_results[task_name] = result
            print(
                f"[{task_name}] native Q99 Δ(proxy-z - true-z) "
                f"= {result['native_q99_diff_mean']:+.4f} ± {result['native_q99_diff_std']:.4f}"
            )

    # CSV: scalar columns only.
    csv_path = out_dir / "reward_proxy_validation.csv"
    preferred_columns = [
        "task", "family", "proxy_confidence", "proxy_note", "n",
        "true_mean", "proxy_mean", "true_std", "proxy_std",
        "bias_proxy_minus_true", "mae", "rmse", "pearson", "spearman",
        "top1_overlap", "top5_overlap", "top10_overlap",
        "top1_jaccard", "top5_jaccard", "top10_jaccard",
        "z_cosine_unweighted", "z_cosine_wr",
        "z_true_norm", "z_proxy_norm", "z_true_wr_norm", "z_proxy_wr_norm",
        "true_min", "true_max", "proxy_min", "proxy_max",
    ]
    fieldnames = [c for c in preferred_columns if any(c in row for row in rows)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _sanitize_json(row.get(k)) for k in fieldnames})

    # Useful aggregate diagnostics without imposing a pass/fail threshold.
    pearsons = np.asarray([r["pearson"] for r in rows], dtype=np.float64)
    spearmans = np.asarray([r["spearman"] for r in rows], dtype=np.float64)
    top5s = np.asarray([r["top5_overlap"] for r in rows], dtype=np.float64)
    aggregate: dict[str, Any] = {
        "tasks": len(rows),
        "mean_pearson_finite": float(np.nanmean(pearsons)),
        "mean_spearman_finite": float(np.nanmean(spearmans)),
        "mean_top5_overlap": float(np.nanmean(top5s)),
    }
    if not args.no_z:
        zc = np.asarray([r["z_cosine_unweighted"] for r in rows], dtype=np.float64)
        zw = np.asarray([r["z_cosine_wr"] for r in rows], dtype=np.float64)
        aggregate["mean_z_cosine_unweighted"] = float(np.nanmean(zc))
        aggregate["mean_z_cosine_wr"] = float(np.nanmean(zw))

    json_payload = {
        "validation_version": "official_humenv_vs_proxy_v1",
        "buffer": str(buffer_path),
        "deep_evaluate_py": str(deep_path),
        "model_name": args.model_name,
        "num_samples": int(len(next_obs)),
        "seed": int(args.seed),
        "sample_index_checksum": int(data["indices"].sum() % (2**31 - 1)),
        "official_reward_path": (
            "metamotivo.wrappers.humenvbench.relabel(env, next_qpos, next_qvel, action, env.unwrapped.task)"
        ),
        "proxy_reward_path": "deep_evaluate.compute_reward_from_obs(next_observation, task, action=action)",
        "observation_consistency": consistency,
        "aggregate": aggregate,
        "results_per_task": full_results,
        "paired_rollout_validation": rollout_results,
        "elapsed_seconds": time.time() - t0,
    }
    json_path = out_dir / "reward_proxy_validation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json(json_payload), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print(" VALIDATION COMPLETE")
    print(f" CSV : {csv_path}")
    print(f" JSON: {json_path}")
    if args.save_plots:
        print(f" Plots: {plot_dir}")
    print("=" * 88)
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate deep_evaluate proxy rewards against official HumEnv rewards on aligned original-buffer rows."
    )
    p.add_argument("--buffer", type=str, required=True, help="Original MetaMotivo inference HDF5.")
    p.add_argument(
        "--deep_evaluate_py",
        type=str,
        default="deep_evaluate.py",
        help="Path to the revised deep_evaluate.py whose compute_reward_from_obs is being tested.",
    )
    p.add_argument("--output_dir", type=str, default="reward_proxy_validation")
    p.add_argument("--model_name", type=str, default="facebook/metamotivo-M-1")
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--num_samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated subset of the 45 tasks. Omit for all tasks.",
    )

    # Official reward relabel parallelism.
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--process_executor", action="store_true")
    p.add_argument("--process_context", type=str, default="spawn")

    # Diagnostics.
    p.add_argument(
        "--check_obs_consistency",
        type=int,
        default=0,
        help="If >0, reconstruct this many observations from next_qpos/qvel and compare against stored next_observation.",
    )
    p.add_argument("--no_z", action="store_true", help="Skip z and WR-z cosine calculations.")
    p.add_argument("--save_plots", action="store_true")
    p.add_argument("--plot_max_points", type=int, default=5000)

    # Optional rollout validation.
    p.add_argument(
        "--rollout_tasks",
        type=str,
        default=None,
        help="Comma-separated tasks (or 'all') for paired native rollout validation. Disabled by default.",
    )
    p.add_argument("--rollout_inference", choices=["unweighted", "wr"], default="unweighted")
    p.add_argument("--rollout_seeds", type=int, default=10)
    p.add_argument("--rollout_steps", type=int, default=300)
    p.add_argument("--rollout_seed", type=int, default=50000)
    return p


if __name__ == "__main__":
    parser = build_parser()
    run_validation(parser.parse_args())
