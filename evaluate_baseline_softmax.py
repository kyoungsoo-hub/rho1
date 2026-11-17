"""
MetaMotivo Baseline 45-Task Evaluation using Official FB-CPR reward_wr_inference
================================================================================
Evaluates the official facebook/metamotivo-S-1 model on the original 500,000 dataset
(/home/kyoungsookim/rho/metamotivo-S-1-datasets/data/buffer_inference_500000.hdf5)
using metamotivo's built-in `model.reward_wr_inference(next_obs, reward)` which performs
softmax(10 * reward) weighted backward-map inference: z = project_z( sum_i r_i * softmax(10*r)_i * B(s'_i) ).

Outputs saved to /home/kyoungsookim/rho/baseline:
- Videos: /home/kyoungsookim/rho/baseline/videos/video_{task_name}.mp4
- JSON: /home/kyoungsookim/rho/baseline/baseline_evaluation_45tasks_results.json
- CSV: /home/kyoungsookim/rho/baseline/baseline_evaluation_45tasks_summary.csv
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import FlattenObservation

# Project imports
sys.path.append("/home/kyoungsookim/rho_M1")
sys.path.append("/home/kyoungsookim/rho")
from metamotivo.fb_cpr.huggingface import FBcprModel
from humenv import make_humenv
from deep_evaluate import (
    META_45_TASKS,
    compute_reward_from_obs,
    proxy_metadata,
    load_inference_obs,
    run_single_rollout,
)


@torch.no_grad()
def infer_z_with_metamotivo_wr(
    model: FBcprModel,
    next_obs: np.ndarray,
    task_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Uses MetaMotivo's built-in reward_wr_inference (softmax(10*r) weighted).
    Also computes unweighted z for cosine similarity comparison.
    """
    rewards = compute_reward_from_obs(next_obs, task_name).astype(np.float32)
    next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=model.cfg.device)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=model.cfg.device).reshape(-1, 1)

    # Built-in MetaMotivo FB-CPR reward_wr_inference:
    # return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))
    z_softmax = model.reward_wr_inference(next_obs_tensor, reward_tensor)

    # Unweighted z for reference
    z_unweighted = model.reward_inference(next_obs_tensor, reward_tensor, weight=None)

    stats = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_p95": float(np.quantile(rewards, 0.95)),
        "reward_p99": float(np.quantile(rewards, 0.99)),
    }
    return z_softmax, z_unweighted, stats


def evaluate_baseline_softmax(
    dataset_path: str = "/home/kyoungsookim/rho_M1/metamotivo-M-1-datasets/data/buffer_inference_500000.hdf5",
    output_dir: str = "/home/kyoungsookim/rho_M1/baseline_softmax_M1",
    model_name: str = "facebook/metamotivo-M-1",
    num_inference_samples: int = 500_000,
    seed: int = 1234,
    num_steps: int = 300,
    save_videos: bool = True,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    video_dir = out_path / "videos"
    if save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" METAMOTIVO BASELINE 45-TASK EVALUATION (FB-CPR reward_wr_inference)")
    print(f" Model     : {model_name}")
    print(f" Dataset   : {dataset_path}")
    print(f" Output Dir: {out_path}")
    print(f" Device    : {device}")
    print(f" Samples   : {num_inference_samples:,}")
    print(f" Inference : model.reward_wr_inference (softmax(10 * r) weighted)")
    print("=" * 80)

    print(f"[Model] Loading {model_name}...")
    model = FBcprModel.from_pretrained(model_name, device=device)

    print(f"[Dataset] Loading {num_inference_samples:,} inference samples from original buffer...")
    next_obs, sampled_indices = load_inference_obs(dataset_path, num_inference_samples, seed)

    results: dict[str, Any] = {
        "evaluation_type": "metamotivo_baseline_official_reward_wr_inference",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": dataset_path,
        "tasks_evaluated": len(META_45_TASKS),
        "inference": {
            "num_samples": num_inference_samples,
            "seed": seed,
            "weighting": "metamotivo.reward_wr_inference [softmax(10 * reward)]",
        },
        "rollout": {
            "num_steps": num_steps,
            "seed": seed,
            "save_videos": save_videos,
        },
        "results_per_task": {},
    }

    csv_rows = []
    start_all = time.time()

    for idx, task_name in enumerate(META_45_TASKS, 1):
        meta = proxy_metadata(task_name)
        t_task_start = time.time()

        z_softmax, z_unweighted, reward_stats = infer_z_with_metamotivo_wr(
            model=model,
            next_obs=next_obs,
            task_name=task_name,
        )

        z_cos_sim = float(F.cosine_similarity(z_softmax, z_unweighted).item())
        z_softmax_norm = float(z_softmax.norm().item())
        z_unweighted_norm = float(z_unweighted.norm().item())

        vid_path = str(video_dir / f"video_{task_name}.mp4") if save_videos else None

        rollout_res = run_single_rollout(
            model=model,
            task_name=task_name,
            z=z_softmax,
            seed=seed,
            num_steps=num_steps,
            video_output_path=vid_path,
        )

        q99 = rollout_res["q_values"]["Q_gamma_0.99"]
        q98 = rollout_res["q_values"]["Q_gamma_0.98"]
        q95 = rollout_res["q_values"]["Q_gamma_0.95"]
        undisc_ret = rollout_res["undiscounted_return"]
        mean_step_rew = rollout_res["mean_step_reward"]

        task_result = {
            "task": task_name,
            "proxy_family": meta.family,
            "proxy_confidence": meta.confidence,
            "z_softmax_norm": z_softmax_norm,
            "z_unweighted_norm": z_unweighted_norm,
            "z_cosine_similarity_with_unweighted": z_cos_sim,
            "reward_stats": reward_stats,
            "rollout_metrics": {
                "undiscounted_return": undisc_ret,
                "mean_step_reward": mean_step_rew,
                "Q_gamma_0.99": q99,
                "Q_gamma_0.98": q98,
                "Q_gamma_0.95": q95,
                "num_steps": rollout_res["num_steps"],
                "video_path": vid_path,
            },
            "elapsed_seconds": time.time() - t_task_start,
        }
        results["results_per_task"][task_name] = task_result

        csv_rows.append({
            "task": task_name,
            "proxy_family": meta.family,
            "proxy_confidence": meta.confidence,
            "z_softmax_norm": z_softmax_norm,
            "z_unweighted_norm": z_unweighted_norm,
            "z_cosine_vs_unweighted": z_cos_sim,
            "q99": q99,
            "q98": q98,
            "q95": q95,
            "undiscounted_return": undisc_ret,
            "mean_step_reward": mean_step_rew,
            "reward_mean": reward_stats["reward_mean"],
            "reward_p95": reward_stats["reward_p95"],
            "reward_max": reward_stats["reward_max"],
        })

        print(
            f"[{idx:02d}/45] {task_name:<20} | z_cos={z_cos_sim:.4f} | "
            f"Q99={q99:7.4f} | Q98={q98:7.4f} | Return={undisc_ret:7.4f} | "
            f"({time.time() - t_task_start:.1f}s)"
        )

    # Aggregate statistics
    q99_vals = [r["rollout_metrics"]["Q_gamma_0.99"] for r in results["results_per_task"].values()]
    returns = [r["rollout_metrics"]["undiscounted_return"] for r in results["results_per_task"].values()]
    cos_sims = [r["z_cosine_similarity_with_unweighted"] for r in results["results_per_task"].values()]

    results["aggregate"] = {
        "mean_q99": float(np.mean(q99_vals)),
        "median_q99": float(np.median(q99_vals)),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "mean_z_cosine_vs_unweighted": float(np.mean(cos_sims)),
        "median_z_cosine_vs_unweighted": float(np.median(cos_sims)),
        "total_elapsed_seconds": time.time() - start_all,
    }

    out_json = out_path / "baseline_evaluation_45tasks_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)

    out_csv = out_path / "baseline_evaluation_45tasks_summary.csv"
    if csv_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 80)
    print(" BASELINE SOFTMAX(10*r) EVALUATION COMPLETED")
    print(f" JSON: {out_json}")
    print(f" CSV : {out_csv}")
    if save_videos:
        print(f" Videos ({len(META_45_TASKS)} files): {video_dir}")
    print(f" Mean Q99: {results['aggregate']['mean_q99']:.4f}")
    print(f" Mean Return: {results['aggregate']['mean_return']:.4f}")
    print(f" Total Elapsed: {results['aggregate']['total_elapsed_seconds']:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline 45-Task Softmax(10*r) Evaluation (MetaMotivo M-1)")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/kyoungsookim/rho_M1/metamotivo-M-1-datasets/data/buffer_inference_500000.hdf5",
        help="Path to MetaMotivo M-1 transition dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/kyoungsookim/rho_M1/baseline_softmax_M1",
        help="Directory to save baseline evaluation results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/metamotivo-M-1",
        help="MetaMotivo model repository ID",
    )
    parser.add_argument("--num_inference_samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--no_videos", action="store_true")
    args = parser.parse_args()

    evaluate_baseline_softmax(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_inference_samples=args.num_inference_samples,
        seed=args.seed,
        num_steps=args.num_steps,
        save_videos=not args.no_videos,
    )
