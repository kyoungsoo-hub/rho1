"""
Integrated Experiment Evaluation Script
========================================
Automates evaluation for a trained FB-guided latent diffusion experiment directory:

1. Generates 500,000 synthetic transition dataset (buffer_fz_guided_500000.hdf5).
2. Generates 4-Panel 3D KDE Surface & 2D Contour Overlay plot (latent_tsne_3d_density_surface.png).
3. Builds 100% coverage Interactive HTML Pose & State Inspector (interactive_tsne_contour_inspector.html).
4. Infers z vectors using uniform observation-based reward (Method A) and renders 4 rollout MP4 videos.
5. Computes Cosine Similarity/Distance matrices & Discounted Return Q values, saving evaluation_summary_results.json.

Usage:
    python evaluate_experiment.py --exp_dir "/home/kyoungsookim/rho/fb guided 2026-08-10_17-50-31"
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import io
import time
import base64
import argparse
from pathlib import Path
import h5py
import json
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import mujoco
from gymnasium.wrappers import FlattenObservation

# Project imports
sys.path.append("/home/kyoungsookim/rho_M1")
sys.path.append("/home/kyoungsookim/rho")
from metamotivo.fb_cpr.huggingface import FBcprModel
from humenv import make_humenv
from train_raw_transition_diffusion_fb_guided import (
    RawTransitionDenoisingModel,
    GaussianDiffusion,
    generate_fb_guided_raw_dataset,
)


def compute_reward_from_obs(obs: np.ndarray, task_name: str) -> np.ndarray:
    """Direct reward computation from 358-dim observation vector s'."""
    if "move" in task_name or "run" in task_name:
        target_speed = 4.0
        if "0-2" in task_name:
            target_speed = 2.0
        elif "0-6" in task_name:
            target_speed = 6.0
        root_h = obs[:, 0]
        forward_vel = obs[:, 214]  # FORWARD velocity component (x-axis)
        standing = np.clip(1.0 - np.maximum(0.0, 1.4 - root_h) / 1.4, 0.01, 1.0)
        diff = np.abs(forward_vel - target_speed)
        move = np.exp(-0.5 * (diff / (target_speed / 2.0))**2)
        move = (5.0 * move + 1.0) / 6.0
        return (standing * move).reshape(-1, 1)
    elif "jump" in task_name:
        jump_height = 2.0
        root_h = obs[:, 0]
        up_vel = obs[:, 216]
        jumping = np.clip(1.0 - np.maximum(0.0, jump_height - root_h) / jump_height, 0.01, 1.0)
        up_velocity = np.clip(up_vel / 5.0, 0.0, 1.0)
        return (jumping * up_velocity).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported task: {task_name}")


def infer_z_obs_based(
    model: FBcprModel,
    dataset_path: str,
    task_name: str,
    num_inference_samples: int = 100_000,
) -> torch.Tensor:
    """Infer context vector z strictly using observation-based reward."""
    with h5py.File(dataset_path, "r") as hf:
        N = min(num_inference_samples, hf["next_observation"].shape[0])
        next_obs = hf["next_observation"][:N]

    rewards = compute_reward_from_obs(next_obs, task_name)
    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=model.cfg.device)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.cfg.device).reshape(-1, 1)

    z = model.reward_inference(next_obs_tensor, reward_tensor, weight=None)
    return z


def run_simulation_rollout(
    model: FBcprModel,
    task_name: str,
    z: torch.Tensor,
    video_output_path: str,
    num_steps: int = 300,
    fps: int = 50,
    gammas: list = [0.99, 0.98, 0.95],
) -> dict:
    """Run MuJoCo simulation rollout, save MP4 video, and compute discounted return Q values."""
    device = model.cfg.device

    env, _ = make_humenv(
        num_envs=1,
        task=task_name,
        state_init="Default",
        render_mode="rgb_array",
        render_width=640,
        render_height=480,
        wrappers=[FlattenObservation],
    )

    obs, _ = env.reset()
    frames = [env.render()]
    step_rewards = []

    for _ in range(num_steps):
        obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.act(obs_tensor, z, mean=True)
        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().ravel())
        step_rewards.append(float(reward))
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(video_output_path, frames, fps=fps)

    step_rewards = np.array(step_rewards, dtype=np.float64)
    T = len(step_rewards)
    timesteps = np.arange(T)
    undiscounted_return = float(np.sum(step_rewards))
    mean_step_reward = float(np.mean(step_rewards))

    q_values = {}
    for gamma in gammas:
        discounts = gamma ** timesteps
        q_val = float(np.sum(discounts * step_rewards))
        q_values[f"Q_gamma_{gamma}"] = q_val

    return {
        "video_path": video_output_path,
        "num_steps": int(T),
        "undiscounted_return": undiscounted_return,
        "mean_step_reward": mean_step_reward,
        "q_values": q_values,
    }


def evaluate_experiment(
    exp_dir: str,
    original_hdf5: str = "/home/kyoungsookim/rho_M1/metamotivo-M-1-datasets/data/buffer_inference_500000.hdf5",
    model_name: str = "facebook/metamotivo-M-1",
):
    print("=================================================================")
    print(f" INTEGRATED RAW EXPERIMENT EVALUATION: {exp_dir}")
    print("=================================================================")

    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    best_ckpt_path = exp_path / "fb_guided_diffusion_raw_best.pt"
    if not best_ckpt_path.exists():
        best_ckpt_path = exp_path / "fb_guided_diffusion_raw_best_prior.pt"
    if not best_ckpt_path.exists():
        best_ckpt_path = exp_path / "fb_guided_diffusion_raw_785dim.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"No valid raw model checkpoint found in {exp_dir}")

    orig_hdf5_path = original_hdf5
    output_hdf5_path = exp_path / "buffer_fz_guided_500000.hdf5"
    density_png_out = exp_path / "latent_tsne_3d_density_surface.png"
    html_out_path = exp_path / "interactive_tsne_contour_inspector.html"
    summary_json_out = exp_path / "evaluation_summary_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    print(f"[Checkpoint] Using: {best_ckpt_path}")
    print(f"[Dataset] Real baseline: {orig_hdf5_path}")

    # -------------------------------------------------------------------------
    # STEP 1: Generate 500,000 Synthetic Raw Dataset
    # -------------------------------------------------------------------------
    print("\n-----------------------------------------------------------------")
    print(" STEP 1: Generating 500,000 Synthetic Raw Dataset")
    print("-----------------------------------------------------------------")
    if not output_hdf5_path.exists():
        print(f"Generating synthetic raw dataset -> {output_hdf5_path}...")
        generate_fb_guided_raw_dataset(
            output_path=str(output_hdf5_path),
            orig_buffer_path=orig_hdf5_path,
            diffusion_ckpt_path=str(best_ckpt_path),
            num_samples=500_000,
        )
    else:
        print(f"Dataset already exists: {output_hdf5_path}")

    # -------------------------------------------------------------------------
    # STEP 2 & 3: 4-Panel t-SNE Density Plot & Interactive HTML Inspector
    # -------------------------------------------------------------------------
    print("\n-----------------------------------------------------------------")
    print(" STEP 2 & 3: 4-Panel t-SNE Plot & Interactive HTML Inspector")
    print("-----------------------------------------------------------------")
    diff_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    cfg = diff_ckpt["config"]
    transition_mean = np.array(diff_ckpt["transition_mean"], dtype=np.float32)
    transition_std = np.array(diff_ckpt["transition_std"], dtype=np.float32)

    # Load real validation raw transitions
    with h5py.File(orig_hdf5_path, "r") as hf_orig:
        obs_orig = hf_orig["observation"][:]
        act_orig = hf_orig["action"][:]
        next_obs_orig = hf_orig["next_observation"][:]

    raw_transitions_orig = np.concatenate([obs_orig, act_orig, next_obs_orig], axis=1).astype(np.float32)
    real_transitions_norm = (raw_transitions_orig - transition_mean) / transition_std

    split_npz = exp_path / "fb_guided_diffusion_split_indices.npz"
    if split_npz.exists():
        split_data = np.load(split_npz)
        val_indices = split_data["val_indices"]
        real_val_norm = real_transitions_norm[val_indices]
    else:
        real_val_norm = real_transitions_norm[475000:]

    num_samples = 3000
    rng = np.random.RandomState(42)
    real_sample_idx = rng.choice(len(real_val_norm), size=num_samples, replace=False)
    real_norm = real_val_norm[real_sample_idx]

    model = RawTransitionDenoisingModel(
        transition_dim=cfg["model"]["transition_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_blocks=cfg["model"]["num_blocks"],
    ).to(device)

    state_dict = diff_ckpt.get("ema_state_dict", diff_ckpt.get("model_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=cfg["model"]["timesteps"],
        schedule="cosine",
        device=device,
    )

    print(f"Sampling {num_samples} synthetic raw transitions with DDIM (50 steps)...")
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    with torch.no_grad():
        gen_norm = diffusion.sample_ddim(
            model=model,
            shape=(num_samples, cfg["model"]["transition_dim"]),
            ddim_steps=50,
            eta=0.0,
            generator=generator,
        ).cpu().numpy()

    print("Computing 2D t-SNE reduction (Perplexity=30, random_state=42)...")
    combined = np.concatenate([real_norm, gen_norm], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init="pca", learning_rate="auto")
    embedding_2d = tsne.fit_transform(combined)

    real_2d = embedding_2d[:num_samples]
    gen_2d = embedding_2d[num_samples:]

    print("Computing 2D Gaussian KDE for 3D Surface & Contours...")
    x_min = min(real_2d[:, 0].min(), gen_2d[:, 0].min()) - 3.0
    x_max = max(real_2d[:, 0].max(), gen_2d[:, 0].max()) + 3.0
    y_min = min(real_2d[:, 1].min(), gen_2d[:, 1].min()) - 3.0
    y_max = max(real_2d[:, 1].max(), gen_2d[:, 1].max()) + 3.0

    grid_resolution = 100
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])

    kde_real = gaussian_kde(real_2d.T)
    kde_gen = gaussian_kde(gen_2d.T)
    Z_real = kde_real(grid_coords).reshape(X.shape)
    Z_gen = kde_gen(grid_coords).reshape(X.shape)

    # Plot 4-Panel Density Surface
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        f"Raw Transition Space Density Visualization: Real vs FB-Guided ({exp_path.name})\n"
        "(t-SNE 2D Reduction -> 3D Surface Kernel Density Estimation)",
        fontsize=18, fontweight="bold", y=0.98,
    )

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(X, Y, Z_real, cmap="Blues", alpha=0.85, linewidth=0, antialiased=True)
    ax1.set_title("1. Real Raw Transition Density (3D Surface)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax1.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax1.set_zlabel("Density (Z)", fontsize=11)
    ax1.view_init(elev=35, azim=225)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label="Real Density")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(X, Y, Z_gen, cmap="Oranges", alpha=0.85, linewidth=0, antialiased=True)
    ax2.set_title("2. FB-Guided Generated Raw Density (3D Surface)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax2.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax2.set_zlabel("Density (Z)", fontsize=11)
    ax2.view_init(elev=35, azim=225)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label="Generated Density")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.plot_surface(X, Y, Z_real, cmap="Blues", alpha=0.5, linewidth=0.1, edgecolor="blue", label="Real Density")
    ax3.plot_surface(X, Y, Z_gen, cmap="Oranges", alpha=0.6, linewidth=0.1, edgecolor="red", label="Generated Density")
    ax3.set_title("3. Overlaid 3D Surface Density Comparison", fontsize=14, fontweight="bold", pad=12)
    ax3.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax3.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax3.set_zlabel("Density (Z)", fontsize=11)
    ax3.view_init(elev=35, azim=225)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(real_2d[:, 0], real_2d[:, 1], c="blue", alpha=0.15, s=8, label="Real Transitions")
    ax4.scatter(gen_2d[:, 0], gen_2d[:, 1], c="crimson", alpha=0.15, s=8, label="FB-Guided Generated")
    ax4.contour(X, Y, Z_real, levels=5, colors="blue", linewidths=1.5, alpha=0.7)
    ax4.contour(X, Y, Z_gen, levels=5, colors="crimson", linewidths=1.5, alpha=0.7)
    ax4.set_title("4. 2D t-SNE Scatter & Density Contour Overlay", fontsize=14, fontweight="bold")
    ax4.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax4.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=11)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(density_png_out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 4-panel density plot -> {density_png_out}")

    # Unstandardize physical transitions directly (NO VAE DECODER NEEDED!)
    real_phys = real_norm * transition_std + transition_mean
    gen_phys = gen_norm * transition_std + transition_mean

    real_next_obs = real_phys[:, 427:785]
    gen_next_obs = gen_phys[:, 427:785]

    real_h = real_next_obs[:, 0]
    gen_h = gen_next_obs[:, 0]
    real_vel = np.linalg.norm(real_next_obs[:, 214:216], axis=1)
    gen_vel = np.linalg.norm(gen_next_obs[:, 214:216], axis=1)
    real_up = real_next_obs[:, 216]
    gen_up = gen_next_obs[:, 216]

    real_r_move = compute_reward_from_obs(real_next_obs, "move-ego-0-4").ravel()
    gen_r_move = compute_reward_from_obs(gen_next_obs, "move-ego-0-4").ravel()
    real_r_jump = compute_reward_from_obs(real_next_obs, "jump-2").ravel()
    gen_r_jump = compute_reward_from_obs(gen_next_obs, "jump-2").ravel()

    print(f"Rendering 3D pose thumbnails for ALL {num_samples} Real + {num_samples} Generated states...")
    env_mujoco, _ = make_humenv(num_envs=1, task="move-ego-0-4", state_init="Default", render_mode="rgb_array")
    mj_data = env_mujoco.unwrapped.data
    mj_model = env_mujoco.unwrapped.model

    with h5py.File(orig_hdf5_path, "r") as hf_orig:
        orig_next_obs_search = hf_orig["next_observation"][:100000]
        orig_qpos_search = hf_orig["next_qpos"][:100000]
        orig_qvel_search = hf_orig["next_qvel"][:100000]

    def render_pose(obs_vec: np.ndarray) -> str:
        diffs = np.linalg.norm(orig_next_obs_search[:, :20] - obs_vec[:20], axis=1)
        best_idx = np.argmin(diffs)
        qpos = orig_qpos_search[best_idx].copy()
        qvel = orig_qvel_search[best_idx].copy()
        qpos[2] = obs_vec[0]
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = qvel
        mujoco.mj_forward(mj_model, mj_data)
        frame = env_mujoco.render()
        img = PIL.Image.fromarray(frame).resize((180, 180), PIL.Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    real_images = [render_pose(real_next_obs[i]) for i in range(num_samples)]
    gen_images = [render_pose(gen_next_obs[i]) for i in range(num_samples)]
    env_mujoco.close()

    hover_text_real = [
        f"<b>Class:</b> Real Validation Raw Transition<br><b>Sample ID:</b> #{i}<br>"
        f"<b>t-SNE:</b> ({real_2d[i, 0]:.2f}, {real_2d[i, 1]:.2f})<br>"
        f"<b>Root Height (h):</b> {real_h[i]:.3f} m<br><b>COM Speed:</b> {real_vel[i]:.3f} m/s<br>"
        f"<b>Upward Vel:</b> {real_up[i]:.3f} m/s<br><b>Move Reward:</b> {real_r_move[i]:.4f}<br>"
        f"<b>Jump Reward:</b> {real_r_jump[i]:.4f}"
        for i in range(num_samples)
    ]

    hover_text_gen = [
        f"<b>Class:</b> FB-Guided Raw Generated ({exp_path.name})<br><b>Sample ID:</b> #{i}<br>"
        f"<b>t-SNE:</b> ({gen_2d[i, 0]:.2f}, {gen_2d[i, 1]:.2f})<br>"
        f"<b>Root Height (h):</b> {gen_h[i]:.3f} m<br><b>COM Speed:</b> {gen_vel[i]:.3f} m/s<br>"
        f"<b>Upward Vel:</b> {gen_up[i]:.3f} m/s<br><b>Move Reward:</b> {gen_r_move[i]:.4f}<br>"
        f"<b>Jump Reward:</b> {gen_r_jump[i]:.4f}"
        for i in range(num_samples)
    ]

    trace_real_scatter = go.Scatter(
        x=real_2d[:, 0], y=real_2d[:, 1], mode="markers", name="Real Transitions (Scatter)",
        marker=dict(size=5, color="blue", opacity=0.35), hovertext=hover_text_real, hoverinfo="text",
        customdata=[[0, i] for i in range(num_samples)],
    )

    trace_gen_scatter = go.Scatter(
        x=gen_2d[:, 0], y=gen_2d[:, 1], mode="markers", name="FB-Guided Raw Generated (Scatter)",
        marker=dict(size=5, color="crimson", opacity=0.45), hovertext=hover_text_gen, hoverinfo="text",
        customdata=[[1, i] for i in range(num_samples)],
    )

    trace_real_contour = go.Contour(
        x=x_grid, y=y_grid, z=Z_real, colorscale=[[0, "rgba(0,0,255,0)"], [1, "rgba(0,0,255,0.7)"]],
        showscale=False, contours=dict(coloring="lines", showlabels=False), line=dict(color="blue", width=2),
        name="Real Density Contour", hoverinfo="skip",
    )

    trace_gen_contour = go.Contour(
        x=x_grid, y=y_grid, z=Z_gen, colorscale=[[0, "rgba(220,20,60,0)"], [1, "rgba(220,20,60,0.7)"]],
        showscale=False, contours=dict(coloring="lines", showlabels=False), line=dict(color="crimson", width=2),
        name="Generated Density Contour", hoverinfo="skip",
    )

    fig_interactive = go.Figure(data=[trace_real_contour, trace_gen_contour, trace_real_scatter, trace_gen_scatter])
    fig_interactive.update_layout(
        title=dict(text=f"4. 2D t-SNE Scatter & Density Contour Overlay ({exp_path.name} Inspector)", font=dict(size=18, family="Arial", color="#1E293B"), x=0.02),
        xaxis=dict(title="t-SNE Dimension 1", gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(title="t-SNE Dimension 2", gridcolor="#E2E8F0", zeroline=False),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F8FAFC", width=1100, height=750,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)", bordercolor="#CBD5E1", borderwidth=1),
        updatemenus=[
            dict(
                type="buttons", direction="right", x=0.5, y=1.12, xanchor="center", yanchor="top",
                buttons=[
                    dict(label="Show Both (All)", method="update", args=[{"visible": [True, True, True, True]}]),
                    dict(label="Real Only (Blue)", method="update", args=[{"visible": [True, False, True, False]}]),
                    dict(label="FB-Guided Only (Crimson)", method="update", args=[{"visible": [False, True, False, True]}]),
                    dict(label="Contours Only", method="update", args=[{"visible": [True, True, False, False]}]),
                ],
                bgcolor="#EDF2F7", bordercolor="#CBD5E1", font=dict(size=13, color="#2D3748"),
            )
        ],
    )

    plot_html = fig_interactive.to_html(include_plotlyjs="cdn", full_html=False)
    real_img_json = json.dumps(real_images)
    gen_img_json = json.dumps(gen_images)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>4. 2D t-SNE Scatter & Density Contour Overlay ({exp_path.name})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #F1F5F9; color: #0F172A; }}
        .container {{ display: flex; gap: 20px; max-width: 1550px; margin: 0 auto; }}
        .plot-container {{ flex: 1; background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }}
        .inspector-panel {{ width: 340px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); display: flex; flex-direction: column; align-items: center; }}
        .inspector-title {{ font-size: 16px; font-weight: 700; margin-bottom: 15px; color: #1E293B; text-align: center; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; width: 100%; }}
        .pose-card {{ width: 240px; height: 240px; border: 2px dashed #CBD5E1; border-radius: 10px; display: flex; align-items: center; justify-content: center; background: #F8FAFC; overflow: hidden; margin-bottom: 15px; }}
        .pose-card img {{ width: 100%; height: 100%; object-fit: cover; }}
        .state-details {{ width: 100%; font-size: 13px; line-height: 1.6; background: #F8FAFC; padding: 12px; border-radius: 8px; border: 1px solid #E2E8F0; }}
        .metric-badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
        .badge-real {{ background: #DBEAFE; color: #1E40AF; }}
        .badge-gen {{ background: #FEE2E2; color: #991B1B; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="plot-container">{plot_html}</div>
        <div class="inspector-panel">
            <div class="inspector-title">📷 3D Humanoid Pose & State Inspector</div>
            <div class="pose-card" id="pose-card">
                <span style="color:#94A3B8; font-size:13px; text-align:center;">Hover over points in 4th Figure (t-SNE Contour Overlay) to inspect 3D pose & simulation state s</span>
            </div>
            <div class="state-details" id="state-details">
                <b>Status:</b> Ready<br>
                <i>Hover over Scatter points to view exact simulation variables.</i>
            </div>
        </div>
    </div>
    <script>
        const realImages = {real_img_json};
        const genImages = {gen_img_json};
        document.addEventListener('DOMContentLoaded', function() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            plotDiv.on('plotly_hover', function(data) {{
                if (!data.points || data.points.length === 0) return;
                const pt = data.points[0];
                if (!pt.customdata || !Array.isArray(pt.customdata) || pt.customdata.length < 2) return;
                const isReal = (pt.customdata[0] === 0);
                const imgIdx = pt.customdata[1];
                const poseCard = document.getElementById('pose-card');
                const stateDetails = document.getElementById('state-details');
                let imgSrc = isReal ? realImages[imgIdx] : genImages[imgIdx];
                if (imgSrc) {{ poseCard.innerHTML = `<img src="${{imgSrc}}" alt="Humanoid Pose">`; }}
                const badgeClass = isReal ? 'badge-real' : 'badge-gen';
                const labelName = isReal ? 'Real Validation Raw Transition' : 'FB-Guided Raw Generated';
                stateDetails.innerHTML = `<span class="metric-badge ${{badgeClass}}">${{labelName}}</span><br><b>Point ID:</b> #${{imgIdx}}<br>${{pt.hovertext}}`;
            }});
        }});
    </script>
</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved interactive HTML inspector -> {html_out_path}")

    # -------------------------------------------------------------------------
    # STEP 4 & 5: z Inference, Rollout MP4 Videos & Cosine Similarity/Q JSON
    # -------------------------------------------------------------------------
    print("\n-----------------------------------------------------------------")
    print(" STEP 4 & 5: Method A z Inference, MP4 Videos & Cosine Sim / Q Stats")
    print("-----------------------------------------------------------------")
    print(f"[Model] Loading MetaMotivo model ({model_name})...")
    fb_model = FBcprModel.from_pretrained(model_name, device=str(device))

    z_move_orig = infer_z_obs_based(fb_model, orig_hdf5_path, "move-ego-0-4")
    z_move_guided = infer_z_obs_based(fb_model, str(output_hdf5_path), "move-ego-0-4")
    z_jump_orig = infer_z_obs_based(fb_model, orig_hdf5_path, "jump-2")
    z_jump_guided = infer_z_obs_based(fb_model, str(output_hdf5_path), "jump-2")

    zs = [z_move_orig.squeeze(), z_move_guided.squeeze(), z_jump_orig.squeeze(), z_jump_guided.squeeze()]
    labels = ["move (Orig)", f"move ({exp_path.name})", "jump (Orig)", f"jump ({exp_path.name})"]

    cos_sim = np.zeros((4, 4))
    cos_dist = np.zeros((4, 4))
    l2_dist = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            sim = F.cosine_similarity(zs[i].unsqueeze(0), zs[j].unsqueeze(0)).item()
            cos_sim[i, j] = sim
            cos_dist[i, j] = 1.0 - sim
            l2_dist[i, j] = (zs[i] - zs[j]).norm().item()

    print("Rendering 4 Rollout MP4 Videos & Computing Q Returns...")
    vids = [
        ("move-ego-0-4", "Original", z_move_orig, str(exp_path / "video_move-ego-0-4_original.mp4")),
        ("move-ego-0-4", "Guided", z_move_guided, str(exp_path / "video_move-ego-0-4_fz_guided.mp4")),
        ("jump-2", "Original", z_jump_orig, str(exp_path / "video_jump-2_original.mp4")),
        ("jump-2", "Guided", z_jump_guided, str(exp_path / "video_jump-2_fz_guided.mp4")),
    ]

    q_results = {}
    for task_name, dataset_label, z_vec, vid_path in vids:
        print(f"Running simulation rollout for '{task_name} | {dataset_label}' -> {vid_path}...")
        res = run_simulation_rollout(
            model=fb_model,
            task_name=task_name,
            z=z_vec,
            video_output_path=vid_path,
        )
        q_results[f"{task_name} | {dataset_label}"] = res

    summary_data = {
        "experiment_name": exp_path.name,
        "model_name": model_name,
        "labels": labels,
        "latent_z_norms": [z.norm().item() for z in zs],
        "cosine_similarity_matrix": cos_sim.tolist(),
        "cosine_distance_matrix": cos_dist.tolist(),
        "euclidean_distance_matrix": l2_dist.tolist(),
        "q_evaluation_rollouts": q_results,
    }

    with open(summary_json_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved evaluation summary JSON -> {summary_json_out}")
    print("=================================================================")
    print(" INTEGRATED RAW EVALUATION COMPLETED SUCCESSFULLY!")
    print(f" All outputs saved in: {exp_dir}")
    print("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Evaluation Script for Raw Transition Experiments")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to raw experiment directory")
    parser.add_argument(
        "--original_hdf5",
        type=str,
        default="/home/kyoungsookim/rho_M1/metamotivo-M-1-datasets/data/buffer_inference_500000.hdf5",
        help="Path to original real transition dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/metamotivo-M-1",
        help="MetaMotivo model repository ID",
    )
    args = parser.parse_args()

    evaluate_experiment(
        exp_dir=args.exp_dir,
        original_hdf5=args.original_hdf5,
        model_name=args.model_name,
    )
