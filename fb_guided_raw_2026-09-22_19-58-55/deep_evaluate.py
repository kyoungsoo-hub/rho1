"""
Deep Integrated 45-Task Evaluation Script (785-D [s, a, s'] with Unified Reward)
================================================================================
Evaluates Original vs FB-guided raw-transition datasets where the
Diffusion sample is x=[s, a, s'] (785-dim).

Key behavior in this version
----------------------------
1. Guided dataset generation uses the 785-D Diffusion checkpoint.
   Saved to: buffer_fz_guided_500000.hdf5
2. z context inference is computed from next_observation + action using a
   HumEnv-aligned observation/action proxy reward:
       z = E[r_proxy(s', a) * B(s')]
   via model.reward_inference(..., weight=None).
3. Rollout evaluation records BOTH:
   a) proxy_return: evaluated from observation s at each step using the exact
      same compute_reward_from_obs function as used in z-inference.
   b) native_return: evaluated by HumEnv/MuJoCo environment dynamics.
4. Both proxy and native metrics are logged into JSON and CSV summaries.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import h5py
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import FlattenObservation
from dm_control.utils import rewards as dm_rewards

# Project imports
sys.path.append("/home/kyoungsookim/rho1")
sys.path.append("/home/kyoungsookim/rho")
from metamotivo.fb_cpr.huggingface import FBcprModel
from humenv import make_humenv
from train_raw_transition_diffusion_fb_guided import generate_fb_guided_raw_dataset


META_45_TASKS = [
    "move-ego-0-0", "move-ego-low-0-0", "headstand", "move-ego-0-2", "move-ego-0-4",
    "move-ego--90-2", "move-ego--90-4", "move-ego-90-2", "move-ego-90-4", "move-ego-180-2",
    "move-ego-180-4", "move-ego-low-0-2", "move-ego-low--90-2", "move-ego-low-90-2",
    "move-ego-low-180-2", "jump-2", "rotate-x--5-0.8", "rotate-x-5-0.8", "rotate-y--5-0.8",
    "rotate-y-5-0.8", "rotate-z--5-0.8", "rotate-z-5-0.8", "raisearms-l-l", "raisearms-l-m",
    "raisearms-l-h", "raisearms-m-l", "raisearms-m-m", "raisearms-m-h", "raisearms-h-l",
    "raisearms-h-m", "raisearms-h-h", "crouch-0", "sitonground", "lieonground-up",
    "lieonground-down", "split-0.5", "split-1", "crawl-0.4-0-u", "crawl-0.4-2-u",
    "crawl-0.5-0-u", "crawl-0.5-2-u", "crawl-0.4-0-d", "crawl-0.4-2-d", "crawl-0.5-0-d",
    "crawl-0.5-2-d",
]


MOVE_RE = re.compile(
    r"^move-ego(?P<low>-low)?-(?P<angle>-?\d+(?:\.\d+)?)-(?P<speed>\d+(?:\.\d+)?)$"
)
ROTATE_RE = re.compile(
    r"^rotate-(?P<axis>[xyz])-(?P<omega>-?\d+(?:\.\d+)?)-(?P<height>\d+(?:\.\d+)?)$"
)
RAISE_RE = re.compile(r"^raisearms-(?P<left>[lmh])-(?P<right>[lmh])$")
CRAWL_RE = re.compile(
    r"^crawl-(?P<height>\d+(?:\.\d+)?)-(?P<speed>\d+(?:\.\d+)?)-(?P<facing>[ud])$"
)
SPLIT_RE = re.compile(r"^split-(?P<amount>\d+(?:\.\d+)?)$")
CROUCH_RE = re.compile(r"^crouch-(?P<height>\d+(?:\.\d+)?)$")
JUMP_RE = re.compile(r"^jump-(?P<height>\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class ProxyMetadata:
    family: str
    confidence: str
    note: str


# HumEnv 358-D flattened observation layout used by MetaMotivo:
#   0          : pelvis/root world height
#   1:70       : 23 non-root body positions in root-heading-local coordinates
#   70:214     : 24 body rotations as 6-D [local x-axis, local z-axis]
#   214:286    : 24 body linear velocities in root-heading-local coordinates
#   286:358    : 24 body angular velocities in root-heading-local coordinates
#
# Body order follows HumEnv's robot.xml / observation construction.
BODY_NAMES = (
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
)
BODY_INDEX = {name: i for i, name in enumerate(BODY_NAMES)}

ROTATION_OFFSET = 70
LINEAR_VEL_OFFSET = 214
ANGULAR_VEL_OFFSET = 286
OBS_DIM = 358

# HumEnv reward constants (mirrors humenv.rewards).
ALIGNMENT_BOUNDS = {
    "x": (-0.1, 0.1),
    "y": (0.9, float("inf")),
    "z": (-0.1, 0.1),
}
ARM_REWARD_LIMITS = {
    "l": (0.0, 0.8, 0.2),
    "m": (1.4, 1.6, 0.1),
    "h": (1.8, float("inf"), 0.2),
}


def proxy_metadata(task_name: str) -> ProxyMetadata:
    """Describe how faithfully the observation-only proxy matches HumEnv's native reward."""
    m = MOVE_RE.fullmatch(task_name)
    if m:
        return ProxyMetadata(
            family="move",
            confidence="medium",
            note=(
                "HumEnv-shaped proxy using head height, chest upright, pelvis-velocity COM proxy, "
                "egocentric direction and action control. Exact COM velocity is unavailable from the 358-D state."
            ),
        )
    if JUMP_RE.fullmatch(task_name):
        return ProxyMetadata(
            family="jump",
            confidence="medium",
            note="HumEnv-shaped head-height/upright/up-velocity reward; pelvis velocity approximates exact COM velocity.",
        )
    m = ROTATE_RE.fullmatch(task_name)
    if m:
        axis = m.group("axis")
        if axis == "z":
            return ProxyMetadata(
                family="rotate",
                confidence="high",
                note="HumEnv-shaped rotation reward; z angular velocity is preserved by heading normalization.",
            )
        return ProxyMetadata(
            family="rotate",
            confidence="low",
            note=(
                "HumEnv-shaped reward, but global x/y angular velocity cannot be recovered exactly after "
                "root-heading normalization; local x/y pelvis gyro is used as a documented proxy."
            ),
        )
    if RAISE_RE.fullmatch(task_name):
        return ProxyMetadata(
            family="raisearms",
            confidence="medium",
            note="Uses actual reconstructed L/R hand heights, head height, chest upright, motion and control proxies.",
        )
    if CRAWL_RE.fullmatch(task_name):
        return ProxyMetadata(
            family="crawl",
            confidence="medium",
            note="Reconstructs body heights/orientations and relative yaw; COM and global gyro terms remain approximate.",
        )
    if task_name in {"lieonground-up", "lieonground-down"}:
        return ProxyMetadata(
            family="ground_pose",
            confidence="medium",
            note="Reconstructs official ground-contact heights and vertical body-orientation terms from 358-D observation.",
        )
    if task_name == "headstand":
        return ProxyMetadata(
            family="ground_pose",
            confidence="medium",
            note="Reconstructs pelvis/feet/head heights and inversion; COM/global gyro terms remain approximate.",
        )
    if task_name == "sitonground":
        return ProxyMetadata(
            family="ground_pose",
            confidence="medium",
            note="HumEnv-shaped sitting reward using head/pelvis/knee heights, chest upright, motion and action control.",
        )
    if SPLIT_RE.fullmatch(task_name):
        return ProxyMetadata(
            family="split",
            confidence="high",
            note="Uses true ankle separation from local body positions plus HumEnv-shaped pelvis/head/motion/control terms.",
        )
    if CROUCH_RE.fullmatch(task_name):
        return ProxyMetadata(
            family="crouch",
            confidence="medium",
            note="HumEnv-shaped crouch reward using pelvis/head/knee heights, chest upright, motion and action control.",
        )
    return ProxyMetadata("unknown", "low", "Unknown task family; no proxy is defined.")


def _validate_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(f"Expected observation shape (N,{OBS_DIM}), got {obs.shape}")
    return obs


def _prepare_action(action: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if action is None:
        return None
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if action.ndim != 2 or action.shape[0] != n:
        raise ValueError(f"Expected action shape (N,A) with N={n}, got {action.shape}")
    return action


def _body_pos(obs: np.ndarray, name: str) -> np.ndarray:
    """Root-heading-local position relative to Pelvis. Pelvis itself is zero by construction."""
    b = BODY_INDEX[name]
    if b == 0:
        return np.zeros((obs.shape[0], 3), dtype=obs.dtype)
    start = 1 + 3 * (b - 1)
    return obs[:, start:start + 3]


def _body_height(obs: np.ndarray, name: str) -> np.ndarray:
    """World z-height. Heading removal is a yaw rotation, so local z is unchanged."""
    if name == "Pelvis":
        return obs[:, 0]
    return obs[:, 0] + _body_pos(obs, name)[:, 2]


def _body_rot6d(obs: np.ndarray, name: str) -> np.ndarray:
    b = BODY_INDEX[name]
    start = ROTATION_OFFSET + 6 * b
    return obs[:, start:start + 6]


def _body_rotmat(obs: np.ndarray, name: str) -> np.ndarray:
    """
    Reconstruct a proper rotation matrix from HumEnv's quat_to_tan_norm representation.

    quat_to_tan_norm stores the rotated local x-axis (first 3 values) followed by
    the rotated local z-axis (last 3 values). Generated diffusion observations may
    not be exactly orthonormal, so Gram-Schmidt re-orthogonalization is applied.
    """
    r6 = _body_rot6d(obs, name).astype(np.float64, copy=False)
    x_axis = r6[:, 0:3].copy()
    z_axis = r6[:, 3:6].copy()

    eps = 1e-8
    x_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)
    bad_x = x_norm[:, 0] < eps
    if np.any(bad_x):
        x_axis[bad_x] = np.array([1.0, 0.0, 0.0])
        x_norm[bad_x] = 1.0
    x_axis = x_axis / np.maximum(x_norm, eps)
    # Remove any component of z along x before normalizing.
    z_axis = z_axis - np.sum(z_axis * x_axis, axis=1, keepdims=True) * x_axis
    z_norm = np.linalg.norm(z_axis, axis=1, keepdims=True)

    # Extremely degenerate synthetic rotations: choose a stable axis not parallel to x.
    bad = (z_norm[:, 0] < eps)
    if np.any(bad):
        fallback = np.zeros((bad.sum(), 3), dtype=np.float64)
        xb = x_axis[bad]
        use_y = np.abs(xb[:, 2]) > 0.9
        fallback[:, 2] = 1.0
        fallback[use_y] = np.array([0.0, 1.0, 0.0])
        fallback = fallback - np.sum(fallback * xb, axis=1, keepdims=True) * xb
        z_axis[bad] = fallback
        z_norm[bad] = np.linalg.norm(fallback, axis=1, keepdims=True)

    z_axis = z_axis / np.maximum(z_norm, eps)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), eps)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), eps)

    # Rotation matrix columns are the body x, y, z axes in the heading-local frame.
    return np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)


def _body_vel(obs: np.ndarray, name: str) -> np.ndarray:
    b = BODY_INDEX[name]
    start = LINEAR_VEL_OFFSET + 3 * b
    return obs[:, start:start + 3]


def _body_angvel(obs: np.ndarray, name: str) -> np.ndarray:
    b = BODY_INDEX[name]
    start = ANGULAR_VEL_OFFSET + 3 * b
    return obs[:, start:start + 3]


def _proxy_com_velocity(obs: np.ndarray) -> np.ndarray:
    """
    Approximate HumEnv's global center-of-mass linear velocity.

    The 358-D observation does not contain body masses or the exact subtree COM
    sensor. Pelvis velocity is used as a conservative proxy; it is stored in the
    root-heading-local frame. Its z component and horizontal speed norm are frame
    invariant, while componentwise x/y quantities are only approximate.
    """
    return _body_vel(obs, "Pelvis")


def _chest_upright(obs: np.ndarray) -> np.ndarray:
    # HumEnv: data.xmat[Chest][-2] == R[2,1]. The third row is invariant to yaw-only
    # heading normalization, so this quantity is exactly recoverable from the 6-D state.
    return _body_rotmat(obs, "Chest")[:, 2, 1]


def _body_yaw(obs: np.ndarray, name: str) -> np.ndarray:
    r = _body_rotmat(obs, name)
    return np.arctan2(r[:, 1, 0], r[:, 0, 0])


def _small_control(action: Optional[np.ndarray], n: int) -> np.ndarray:
    """HumEnv's common small-control factor. If action is unavailable, omit this factor."""
    if action is None:
        return np.ones(n, dtype=np.float32)
    small = dm_rewards.tolerance(
        action,
        margin=1.0,
        value_at_margin=0.0,
        sigmoid="quadratic",
    ).mean(axis=1)
    return ((4.0 + small) / 5.0).astype(np.float32)


def _stand_upright_reward(obs: np.ndarray, stand_height: float = 1.4) -> np.ndarray:
    head_height = _body_height(obs, "Head")
    standing = dm_rewards.tolerance(
        head_height,
        bounds=(stand_height, float("inf")),
        margin=stand_height,
        value_at_margin=0.01,
        sigmoid="linear",
    )
    upright = dm_rewards.tolerance(
        _chest_upright(obs),
        bounds=(0.9, float("inf")),
        margin=1.9,
        value_at_margin=0.0,
        sigmoid="linear",
    )
    return standing * upright


def compute_reward_from_obs(
    obs: np.ndarray,
    task_name: str,
    action: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    HumEnv-aligned reward proxy computed from a 358-D flattened observation.

    This function is intentionally shared by BOTH z-inference and rollout proxy
    evaluation. When action is supplied, HumEnv's control term is reproduced too.

    Quantities exactly recoverable from the 358-D state include body heights,
    relative distances, vertical-orientation components, chest uprightness and
    z-axis angular velocity. The exact subtree COM velocity is not present and is
    approximated with Pelvis velocity. Global x/y angular velocity is also lost by
    heading normalization; rotate-x/y therefore remain explicitly low-confidence.
    """
    obs = _validate_obs(obs)
    n = obs.shape[0]
    action = _prepare_action(action, n)

    pelvis_h = _body_height(obs, "Pelvis")
    head_h = _body_height(obs, "Head")
    chest_upright = _chest_upright(obs)
    com_vel = _proxy_com_velocity(obs)
    control = _small_control(action, n)

    # ------------------------------------------------------------------
    # LocomotionReward
    # ------------------------------------------------------------------
    m = MOVE_RE.fullmatch(task_name)
    if m:
        stay_low = bool(m.group("low"))
        move_angle_deg = float(m.group("angle"))
        move_speed = float(m.group("speed"))

        if stay_low:
            low_height = 0.6
            standing = dm_rewards.tolerance(
                pelvis_h,
                bounds=(low_height / 2.0, low_height),
                margin=low_height / 2.0,
                value_at_margin=0.01,
                sigmoid="linear",
            )
            upright = dm_rewards.tolerance(
                chest_upright,
                bounds=(0.9, float("inf")),
                margin=1.9,
                value_at_margin=0.0,
                sigmoid="linear",
            )
            stand_reward = standing * upright
        else:
            stand_reward = _stand_upright_reward(obs, stand_height=1.4)

        horizontal_vel = com_vel[:, :2]
        if np.isclose(move_speed, 0.0):
            # Official reward is componentwise in the global frame. We only have the
            # heading-local frame, so this is numerically approximate but semantically
            # equivalent for near-zero velocity.
            dont_move = dm_rewards.tolerance(horizontal_vel, margin=0.5).mean(axis=1)
            reward = control * stand_reward * dont_move
            return reward.reshape(-1, 1).astype(np.float32)

        speed = np.linalg.norm(horizontal_vel, axis=1)
        move = dm_rewards.tolerance(
            speed,
            bounds=(0.9 * move_speed, 1.1 * move_speed),
            margin=move_speed / 2.0,
            value_at_margin=0.5,
            sigmoid="gaussian",
        )
        move = (5.0 * move + 1.0) / 6.0

        # HumEnv defines the humanoid's front as angle-90 degrees and then adds
        # Chest yaw. Since obs is root-heading-local, Chest local yaw gives the
        # corresponding target direction in the same local frame.
        target_angle = np.deg2rad(move_angle_deg - 90.0) + _body_yaw(obs, "Chest")
        speed_safe = speed + 1e-6
        direction = horizontal_vel / speed_safe[:, None]
        target_direction = np.stack([np.cos(target_angle), np.sin(target_angle)], axis=1)
        dot = np.sum(target_direction * direction, axis=1)
        angle_reward = np.where(np.isclose(speed, 0.0), 1.0, (dot + 1.0) / 2.0)
        angle_reward = np.clip(angle_reward, 0.0, 1.0)

        reward = control * stand_reward * move * angle_reward
        return reward.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # JumpReward
    # ------------------------------------------------------------------
    m = JUMP_RE.fullmatch(task_name)
    if m:
        jump_height = float(m.group("height"))
        jumping = dm_rewards.tolerance(
            head_h,
            bounds=(jump_height, jump_height + 0.1),
            margin=jump_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = dm_rewards.tolerance(
            chest_upright,
            bounds=(0.9, float("inf")),
            margin=1.9,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        max_velocity = 5.0
        up_velocity = dm_rewards.tolerance(
            com_vel[:, 2],
            bounds=(max_velocity, float("inf")),
            margin=max_velocity,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        return (jumping * upright * up_velocity).reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # RotationReward
    # ------------------------------------------------------------------
    m = ROTATE_RE.fullmatch(task_name)
    if m:
        axis = m.group("axis")
        target_ang_velocity = float(m.group("omega"))
        stand_pelvis_height = float(m.group("height"))

        pelvis_rot = _body_rotmat(obs, "Pelvis")
        pelvis_orientation = pelvis_rot[:, 2, :]  # exact under yaw-only heading removal
        pelvis_angvel_local = _body_angvel(obs, "Pelvis")
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]

        # z is exact under heading normalization; x/y are documented local-frame proxies.
        angular_component = pelvis_angvel_local[:, axis_index]
        direction = np.sign(target_ang_velocity)
        targ_av = abs(target_ang_velocity)
        move = dm_rewards.tolerance(
            direction * angular_component,
            bounds=(targ_av, targ_av + 5.0),
            margin=max(targ_av / 2.0, 1e-6),
            value_at_margin=0.0,
            sigmoid="linear",
        )
        height_reward = dm_rewards.tolerance(
            pelvis_h,
            bounds=(stand_pelvis_height, float("inf")),
            margin=stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        aligned = dm_rewards.tolerance(
            pelvis_orientation[:, axis_index],
            bounds=ALIGNMENT_BOUNDS[axis],
            margin=0.9,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        reward = move * height_reward * control * aligned
        return reward.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # ArmsReward
    # ------------------------------------------------------------------
    m = RAISE_RE.fullmatch(task_name)
    if m:
        left_limits = ARM_REWARD_LIMITS[m.group("left")]
        right_limits = ARM_REWARD_LIMITS[m.group("right")]
        left_height = _body_height(obs, "L_Hand")
        right_height = _body_height(obs, "R_Hand")

        stand_reward = _stand_upright_reward(obs, stand_height=1.4)
        dont_move = dm_rewards.tolerance(com_vel, margin=0.5).mean(axis=1)
        left_arm = dm_rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0.0,
            sigmoid="linear",
        )
        left_arm = (4.0 * left_arm + 1.0) / 5.0
        right_arm = dm_rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0.0,
            sigmoid="linear",
        )
        right_arm = (4.0 * right_arm + 1.0) / 5.0
        reward = control * stand_reward * dont_move * left_arm * right_arm
        return reward.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # SitOnGroundReward / crouch
    # ------------------------------------------------------------------
    m = CROUCH_RE.fullmatch(task_name)
    if m or task_name == "sitonground":
        if task_name == "sitonground":
            pelvis_height_th = 0.0
            constrained_knees = True
            knees_not_on_ground = False
        else:
            pelvis_height_th = float(m.group("height"))
            constrained_knees = False
            knees_not_on_ground = True

        head_pelvis_gap = 0.58
        standing_height = pelvis_height_th + head_pelvis_gap
        standing = dm_rewards.tolerance(
            head_h,
            bounds=(standing_height, standing_height + 0.2),
            margin=0.1,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = dm_rewards.tolerance(
            chest_upright,
            bounds=(0.85, float("inf")),
            margin=1.9,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        stand_reward = standing * upright
        dont_move = dm_rewards.tolerance(com_vel, margin=0.5).mean(axis=1)
        pelvis_reward = dm_rewards.tolerance(
            pelvis_h,
            bounds=(pelvis_height_th, pelvis_height_th + 0.15),
            margin=0.7,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        left_knee_h = _body_height(obs, "L_Knee")
        right_knee_h = _body_height(obs, "R_Knee")
        knee_reward = np.ones(n, dtype=np.float32)
        if constrained_knees:
            knee_reward *= dm_rewards.tolerance(
                left_knee_h, bounds=(0.0, 0.1), margin=0.7, value_at_margin=0.0, sigmoid="linear"
            )
            knee_reward *= dm_rewards.tolerance(
                right_knee_h, bounds=(0.0, 0.1), margin=0.7, value_at_margin=0.0, sigmoid="linear"
            )
        if knees_not_on_ground:
            knee_reward *= dm_rewards.tolerance(
                left_knee_h, bounds=(0.2, 1.0), margin=0.1, value_at_margin=0.0, sigmoid="linear"
            )
            knee_reward *= dm_rewards.tolerance(
                right_knee_h, bounds=(0.2, 1.0), margin=0.1, value_at_margin=0.0, sigmoid="linear"
            )

        reward = control * stand_reward * dont_move * pelvis_reward * knee_reward
        return reward.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # LieDownReward
    # ------------------------------------------------------------------
    if task_name in {"lieonground-up", "lieonground-down"}:
        direction = 1.0 if task_name.endswith("-up") else -1.0
        ground_bodies = ["Pelvis", "Head", "L_Knee", "R_Knee", "R_Ankle", "L_Ankle"]
        orient_bodies = ["Head", "Torso", "Chest", "Pelvis", "L_Knee", "L_Ankle", "R_Knee", "R_Ankle"]

        rew_ground = np.ones(n, dtype=np.float32)
        for body in ground_bodies:
            rew_ground *= dm_rewards.tolerance(
                _body_height(obs, body),
                bounds=(0.0, 0.2),
                margin=0.7,
                value_at_margin=0.0,
                sigmoid="linear",
            )
        flat_chest = dm_rewards.tolerance(
            chest_upright,
            bounds=(0.0, 0.2),
            margin=1.0,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ground_reward = rew_ground * flat_chest
        dont_move = dm_rewards.tolerance(com_vel, margin=0.5).mean(axis=1)

        orient_reward = np.ones(n, dtype=np.float32)
        for body in orient_bodies:
            # Official code rewards the z component of the body's local z-axis.
            r22 = _body_rotmat(obs, body)[:, 2, 2]
            orient_reward *= dm_rewards.tolerance(
                r22 * direction,
                bounds=(0.95, float("inf")),
                margin=1.9,
                value_at_margin=0.0,
                sigmoid="linear",
            )
        reward = control * ground_reward * dont_move * orient_reward
        return reward.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # SplitReward
    # ------------------------------------------------------------------
    m = SPLIT_RE.fullmatch(task_name)
    if m:
        target_distance = float(m.group("amount"))
        left_ankle = _body_pos(obs, "L_Ankle")
        right_ankle = _body_pos(obs, "R_Ankle")
        ankle_distance = np.linalg.norm(left_ankle - right_ankle, axis=1)

        split_rew = dm_rewards.tolerance(
            ankle_distance,
            bounds=(target_distance, float("inf")),
            margin=0.5,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        pelvis_pos = dm_rewards.tolerance(
            pelvis_h,
            bounds=(0.0, 0.2),
            margin=0.5,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        dont_move = dm_rewards.tolerance(com_vel, margin=0.5).mean(axis=1)
        head_rew = dm_rewards.tolerance(
            head_h,
            bounds=(0.5, float("inf")),
            margin=0.3,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        reward = head_rew * split_rew * pelvis_pos * dont_move * control
        return reward.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # HeadstandReward
    # ------------------------------------------------------------------
    if task_name == "headstand":
        stand_pelvis_height = 0.95
        pelvis_rot = _body_rotmat(obs, "Pelvis")
        pelvis_orientation = pelvis_rot[:, 2, :]
        pelvis_angvel = _body_angvel(obs, "Pelvis")

        height_reward = dm_rewards.tolerance(
            pelvis_h,
            bounds=(stand_pelvis_height, float("inf")),
            margin=stand_pelvis_height / 2.0,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        headstand = dm_rewards.tolerance(
            pelvis_orientation[:, 1],
            bounds=(-1.0, -0.9),
            margin=0.9,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        dont_move = dm_rewards.tolerance(com_vel, margin=0.5).mean(axis=1)
        # Official angular velocity is global; local-frame components are the best
        # available 358-D proxy. z is exact, x/y are heading-rotated.
        dont_rotate = dm_rewards.tolerance(
            pelvis_angvel,
            bounds=(-1.0, 1.0),
            margin=4.0,
            value_at_margin=0.1,
            sigmoid="linear",
        ).mean(axis=1)
        high_left_foot = dm_rewards.tolerance(
            _body_height(obs, "L_Ankle"),
            bounds=(stand_pelvis_height, float("inf")),
            margin=stand_pelvis_height / 2.0,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        high_right_foot = dm_rewards.tolerance(
            _body_height(obs, "R_Ankle"),
            bounds=(stand_pelvis_height, float("inf")),
            margin=stand_pelvis_height / 2.0,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        high_head = dm_rewards.tolerance(
            head_h,
            bounds=(0.3, float("inf")),
            margin=0.1,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        reward = (
            height_reward * control * headstand * dont_move * dont_rotate
            * high_left_foot * high_right_foot * high_head
        )
        return reward.reshape(-1, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # CrawlReward
    # ------------------------------------------------------------------
    m = CRAWL_RE.fullmatch(task_name)
    if m:
        spine_height = float(m.group("height"))
        move_speed = float(m.group("speed"))
        direction_sign = 1.0 if m.group("facing") == "u" else -1.0

        spine_high_reward = []
        orientation_reward = []
        for body in ["Spine", "Torso", "Chest", "Pelvis", "Head"]:
            z = _body_height(obs, body)
            up_bound = spine_height + 0.2
            down_bound = spine_height
            if body == "Head":
                up_bound = 1.0
                down_bound = 0.3
            spine_high_reward.append(
                dm_rewards.tolerance(
                    z,
                    bounds=(down_bound, up_bound),
                    margin=0.1,
                    value_at_margin=0.01,
                    sigmoid="linear",
                )
            )
            r22 = _body_rotmat(obs, body)[:, 2, 2]
            orientation_reward.append(
                dm_rewards.tolerance(
                    r22 * direction_sign,
                    bounds=(0.5, float("inf")),
                    margin=0.5,
                    value_at_margin=0.0,
                    sigmoid="linear",
                )
            )

        for body in ["L_Knee", "R_Knee", "L_Hip", "R_Hip", "L_Ankle", "R_Ankle"]:
            r22 = _body_rotmat(obs, body)[:, 2, 2]
            orientation_reward.append(
                dm_rewards.tolerance(
                    r22 * direction_sign,
                    bounds=(0.5, float("inf")),
                    margin=0.5,
                    value_at_margin=0.0,
                    sigmoid="linear",
                )
            )

        spine_prod = np.prod(np.stack(spine_high_reward, axis=0), axis=0)
        orient_prod = np.prod(np.stack(orientation_reward, axis=0), axis=0)
        pos_orient_reward = orient_prod * (1.0 + spine_prod) / 2.0

        chest_yaw = _body_yaw(obs, "Chest")
        angle_alignment_terms = []
        for body in ["Spine", "Torso", "Chest", "Pelvis"]:
            body_yaw = _body_yaw(obs, body)
            # Official reward compares each body's yaw to Chest yaw. Express this
            # as a wrapped relative angle so batched samples can share scalar bounds.
            yaw_error = np.arctan2(
                np.sin(body_yaw - chest_yaw),
                np.cos(body_yaw - chest_yaw),
            )
            angle_alignment_terms.append(
                dm_rewards.tolerance(
                    yaw_error,
                    bounds=(-0.1, 0.1),
                    margin=0.5,
                    value_at_margin=0.0,
                    sigmoid="linear",
                )
            )
        angle_alignment = (1.0 + np.prod(np.stack(angle_alignment_terms, axis=0), axis=0)) / 2.0

        pelvis_angvel = _body_angvel(obs, "Pelvis")
        dont_rotate = dm_rewards.tolerance(
            np.abs(pelvis_angvel),
            bounds=(0.0, 2.5),
            margin=2.0,
            value_at_margin=0.0,
            sigmoid="linear",
        ).mean(axis=1)
        dont_rotate = (1.0 + dont_rotate) / 2.0
        alignment_reward = dont_rotate * pos_orient_reward * angle_alignment

        horizontal_vel = com_vel[:, :2]
        if np.isclose(move_speed, 0.0):
            dont_move = dm_rewards.tolerance(horizontal_vel, margin=0.5).mean(axis=1)
            return (dont_move * alignment_reward).reshape(-1, 1).astype(np.float32)

        speed = np.linalg.norm(horizontal_vel, axis=1)
        move = dm_rewards.tolerance(
            speed,
            bounds=(0.9 * move_speed, 1.1 * move_speed),
            margin=move_speed / 2.0,
            value_at_margin=0.5,
            sigmoid="gaussian",
        )
        move = (5.0 * move + 1.0) / 6.0

        # CrawlReward has move_angle=0 by default. Convert official global target to
        # root-heading-local coordinates by adding Chest local yaw.
        target_angle = np.deg2rad(-90.0) + chest_yaw
        speed_safe = speed + 1e-6
        vel_dir = horizontal_vel / speed_safe[:, None]
        target_dir = np.stack([np.cos(target_angle), np.sin(target_angle)], axis=1)
        dot = np.sum(target_dir * vel_dir, axis=1)
        angle_reward = np.where(np.isclose(speed, 0.0), 1.0, (dot + 1.0) / 2.0)
        angle_reward = np.clip(angle_reward, 0.0, 1.0)
        return (alignment_reward * move * angle_reward).reshape(-1, 1).astype(np.float32)

    raise ValueError(f"No observation reward proxy implemented for task: {task_name}")

def deterministic_sample_indices(total: int, count: int, seed: int) -> np.ndarray:
    count = min(int(count), int(total))
    if count <= 0:
        raise ValueError("num_inference_samples must be positive")
    rng = np.random.default_rng(seed)
    if count == total:
        return np.arange(total, dtype=np.int64)
    indices = rng.choice(total, size=count, replace=False)
    indices.sort()
    return indices.astype(np.int64)


def load_inference_batch(
    dataset_path: str | Path,
    num_inference_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load matched next_observation/action rows used for HumEnv-shaped reward relabeling."""
    with h5py.File(dataset_path, "r") as hf:
        for key in ("next_observation", "action"):
            if key not in hf:
                raise KeyError(f"{dataset_path} has no {key!r} dataset")
        total = int(hf["next_observation"].shape[0])
        if int(hf["action"].shape[0]) != total:
            raise ValueError("action and next_observation row counts do not match")
        indices = deterministic_sample_indices(total, num_inference_samples, seed)
        next_obs = hf["next_observation"][indices]
        action = hf["action"][indices]
    return next_obs.astype(np.float32), action.astype(np.float32), indices


def load_inference_obs(
    dataset_path: str | Path,
    num_inference_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible helper retained for other evaluation scripts."""
    next_obs, _action, indices = load_inference_batch(dataset_path, num_inference_samples, seed)
    return next_obs, indices


@torch.no_grad()
def infer_z_from_obs(
    model: FBcprModel,
    obs: np.ndarray,
    task_name: str,
    action: Optional[np.ndarray] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute z from exactly the same HumEnv-aligned proxy used in rollout evaluation."""
    rewards = compute_reward_from_obs(obs, task_name, action=action)
    device = model.cfg.device
    next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)

    z = model.reward_inference(next_obs_tensor, reward_tensor, weight=None)
    stats = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_p95": float(np.quantile(rewards, 0.95)),
        "reward_p99": float(np.quantile(rewards, 0.99)),
    }
    return z, stats


def run_single_rollout(
    model: FBcprModel,
    task_name: str,
    z: torch.Tensor,
    seed: int,
    num_steps: int = 300,
    video_output_path: Optional[str] = None,
    fps: int = 50,
    gammas: tuple[float, ...] = (0.99, 0.98, 0.95),
) -> dict[str, Any]:
    """Run one rollout recording BOTH native environment rewards and observation proxy rewards."""
    device = model.cfg.device
    env = None
    try:
        env, _ = make_humenv(
            num_envs=1,
            task=task_name,
            state_init="Default",
            render_mode="rgb_array" if video_output_path else None,
            render_width=640,
            render_height=480,
            wrappers=[FlattenObservation],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to create HumEnv task '{task_name}'. Refusing fallback evaluation.") from exc

    frames: list[np.ndarray] = []
    step_native_rewards: list[float] = []
    step_proxy_rewards: list[float] = []
    try:
        obs, _ = env.reset(seed=seed)
        if video_output_path:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        for _ in range(num_steps):
            obs_tensor = torch.as_tensor(
                np.asarray(obs).reshape(1, -1),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                action = model.act(obs_tensor, z, mean=True)
            action_np = action.cpu().numpy().reshape(1, -1)
            obs, reward, terminated, truncated, _ = env.step(action_np.ravel())
            step_native_rewards.append(float(reward))

            # Use the same r(s', a) proxy as z-inference, including HumEnv's control term.
            proxy_r = float(
                compute_reward_from_obs(
                    np.asarray(obs).reshape(1, -1),
                    task_name,
                    action=action_np,
                )[0, 0]
            )
            step_proxy_rewards.append(proxy_r)

            if video_output_path:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            if terminated or truncated:
                break
    finally:
        env.close()

    if video_output_path and frames:
        os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
        imageio.mimsave(video_output_path, frames, fps=fps)

    native_np = np.asarray(step_native_rewards, dtype=np.float64)
    proxy_np = np.asarray(step_proxy_rewards, dtype=np.float64)
    t = np.arange(len(native_np), dtype=np.float64)

    native_q_values = {
        f"Q_gamma_{gamma}": float(np.sum((gamma ** t) * native_np))
        for gamma in gammas
    }
    proxy_q_values = {
        f"Q_gamma_{gamma}": float(np.sum((gamma ** t) * proxy_np))
        for gamma in gammas
    }

    return {
        "seed": int(seed),
        "num_steps": int(len(native_np)),
        "undiscounted_return": float(native_np.sum()) if len(native_np) else 0.0,
        "mean_step_reward": float(native_np.mean()) if len(native_np) else 0.0,
        "q_values": native_q_values,
        "proxy_undiscounted_return": float(proxy_np.sum()) if len(proxy_np) else 0.0,
        "proxy_mean_step_reward": float(proxy_np.mean()) if len(proxy_np) else 0.0,
        "proxy_q_values": proxy_q_values,
        "video_path": video_output_path,
    }


def _aggregate_scalar(values: list[float]) -> dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rollouts),
        "undiscounted_return": _aggregate_scalar([r["undiscounted_return"] for r in rollouts]),
        "mean_step_reward": _aggregate_scalar([r["mean_step_reward"] for r in rollouts]),
        "Q_gamma_0.99": _aggregate_scalar([r["q_values"]["Q_gamma_0.99"] for r in rollouts]),
        "Q_gamma_0.98": _aggregate_scalar([r["q_values"]["Q_gamma_0.98"] for r in rollouts]),
        "Q_gamma_0.95": _aggregate_scalar([r["q_values"]["Q_gamma_0.95"] for r in rollouts]),
        "proxy_undiscounted_return": _aggregate_scalar([r["proxy_undiscounted_return"] for r in rollouts]),
        "proxy_mean_step_reward": _aggregate_scalar([r["proxy_mean_step_reward"] for r in rollouts]),
        "proxy_Q_gamma_0.99": _aggregate_scalar([r["proxy_q_values"]["Q_gamma_0.99"] for r in rollouts]),
        "rollouts": rollouts,
    }


def _sanitize_for_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(v) for v in data]
    elif isinstance(data, (float, np.floating)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return _sanitize_for_json(data.tolist())
    return data


def find_best_checkpoint(exp_path: Path, checkpoint_arg: Optional[str] = None) -> Path:
    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if not p.is_absolute():
            p_in_exp = exp_path / checkpoint_arg
            if p.exists():
                return p.resolve()
            elif p_in_exp.exists():
                return p_in_exp.resolve()
            else:
                raise FileNotFoundError(
                    f"Specified checkpoint not found: {checkpoint_arg} (checked {p} and {p_in_exp})"
                )
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_arg}")

    candidates = [
        exp_path / "fb_guided_diffusion_raw_best.pt",
        exp_path / "fb_guided_diffusion_raw_best_prior.pt",
        exp_path / "fb_guided_diffusion_raw_785dim.pt",
        exp_path / "model_best_guided.pt",
        exp_path / "model_best_total.pt",
        exp_path / "model_best_prior.pt",
        exp_path / "model_final.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raw_numbered = sorted(exp_path.glob("fb_guided_diffusion_raw_epoch_*.pt"))
    if raw_numbered:
        return raw_numbered[-1]
    numbered = sorted(exp_path.glob("model_epoch_*.pt"))
    if numbered:
        return numbered[-1]
    raise FileNotFoundError(f"No valid checkpoint found in {exp_path}")


def _validate_augmented_checkpoint(checkpoint: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    dim = int(cfg.get("model", {}).get("transition_dim", -1))
    if dim != 785:
        raise ValueError(
            f"Expected 785-D checkpoint [s,a,s'], but {checkpoint} reports transition_dim={dim}."
        )


def _validate_augmented_guided_dataset(guided_hdf5: Path) -> None:
    required_shapes = {
        "observation": 358,
        "action": 69,
        "next_observation": 358,
    }
    with h5py.File(guided_hdf5, "r") as hf:
        for key, dim in required_shapes.items():
            if key not in hf:
                raise KeyError(f"Generated dataset {guided_hdf5} has no '{key}'.")
            if hf[key].ndim != 2 or hf[key].shape[1] != dim:
                raise ValueError(
                    f"Generated {key} has shape {hf[key].shape}; expected (N,{dim})."
                )
        n = int(hf["observation"].shape[0])
        if any(int(hf[k].shape[0]) != n for k in required_shapes):
            raise ValueError("Generated fields do not have matching row counts.")


def ensure_guided_dataset(
    exp_path: Path,
    original_hdf5: Path,
    regenerate: bool,
    checkpoint_arg: Optional[str] = None,
) -> Path:
    guided_hdf5 = exp_path / "buffer_fz_guided_500000.hdf5"
    if guided_hdf5.exists() and not regenerate:
        _validate_augmented_guided_dataset(guided_hdf5)
        print(f"[Dataset] Using existing guided dataset: {guided_hdf5}")
        return guided_hdf5

    checkpoint = find_best_checkpoint(exp_path, checkpoint_arg=checkpoint_arg)
    _validate_augmented_checkpoint(checkpoint)
    print(f"[Dataset] Generating 500,000 samples from: {checkpoint}")
    generate_fb_guided_raw_dataset(
        output_path=str(guided_hdf5),
        orig_buffer_path=str(original_hdf5),
        diffusion_ckpt_path=str(checkpoint),
        num_samples=500_000,
    )
    _validate_augmented_guided_dataset(guided_hdf5)
    return guided_hdf5


def deep_evaluate(
    exp_dir: str,
    original_hdf5: str,
    num_inference_samples: int,
    num_rollouts: int,
    num_steps: int,
    seed: int,
    regenerate_dataset: bool,
    save_videos: bool,
    ratio_baseline_threshold: float,
    model_name: str = "facebook/metamotivo-M-1",
    checkpoint: Optional[str] = None,
) -> Path:
    print("=" * 78)
    print(f" DEEP 45-TASK EVALUATION (785-D [s,a,s'] with Unified Reward): {exp_dir}")
    print(f" Baseline Dataset: {original_hdf5}")
    print(f" MetaMotivo Model: {model_name}")
    print("=" * 78)

    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(exp_path)

    original_path = Path(original_hdf5)
    if not original_path.exists():
        raise FileNotFoundError(original_path)

    video_dir = exp_path / "videos"
    if save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    guided_path = ensure_guided_dataset(
        exp_path, original_path, regenerate_dataset, checkpoint_arg=checkpoint
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    fb_model = FBcprModel.from_pretrained(model_name, device=str(device))

    print(f"[Inference] Sampling {num_inference_samples:,} rows per dataset...")
    original_obs, original_action, original_idx = load_inference_batch(original_path, num_inference_samples, seed)
    guided_obs, guided_action, guided_idx = load_inference_batch(guided_path, num_inference_samples, seed)

    results: dict[str, Any] = {
        "evaluation_version": "785d_humenv_aligned_reward_v2",
        "experiment_name": exp_path.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks_evaluated": len(META_45_TASKS),
        "reward_mode": "humenv_aligned_observation_action_proxy_and_native_return",
        "notes": (
            "z-inference and proxy rollout both evaluate the same HumEnv-aligned compute_reward_from_obs(s', a). "
            "Body geometry/orientation is reconstructed from the 358-D state; exact COM velocity and global x/y gyro "
            "are unavailable and use documented proxies. Native return remains the MuJoCo/HumEnv reward."
        ),
        "inference": {
            "num_samples": int(num_inference_samples),
            "original_seed": int(seed),
            "guided_seed": int(seed),
            "original_index_checksum": int(np.asarray(original_idx, dtype=np.int64).sum() % (2**31 - 1)),
            "guided_index_checksum": int(np.asarray(guided_idx, dtype=np.int64).sum() % (2**31 - 1)),
        },
        "rollout": {
            "num_rollouts": int(num_rollouts),
            "num_steps": int(num_steps),
            "base_seed": int(seed),
            "save_videos": bool(save_videos),
        },
        "results_per_task": {},
    }

    csv_rows: list[dict[str, Any]] = []
    start_all = time.time()

    for task_i, task_name in enumerate(META_45_TASKS, 1):
        print(f"\n[{task_i:02d}/{len(META_45_TASKS):02d}] {task_name}")
        meta = proxy_metadata(task_name)

        try:
            z_orig, reward_orig_stats = infer_z_from_obs(
                fb_model, original_obs, task_name, action=original_action
            )
            z_guided, reward_guided_stats = infer_z_from_obs(
                fb_model, guided_obs, task_name, action=guided_action
            )
        except Exception as exc:
            print(f"  [FAILED reward inference] {exc}")
            results["results_per_task"][task_name] = {
                "status": "reward_inference_failed",
                "error": repr(exc),
                "proxy_metadata": asdict(meta),
            }
            continue

        zo = z_orig.reshape(1, -1)
        zg = z_guided.reshape(1, -1)
        z_cos = float(F.cosine_similarity(zo, zg).item())
        z_orig_norm = float(torch.linalg.vector_norm(zo).item())
        z_guided_norm = float(torch.linalg.vector_norm(zg).item())
        z_norm_ratio = z_guided_norm / max(z_orig_norm, 1e-12)

        orig_rollouts: list[dict[str, Any]] = []
        guided_rollouts: list[dict[str, Any]] = []

        for rollout_i in range(num_rollouts):
            rollout_seed = seed + 10_000 * task_i + rollout_i
            orig_video = None
            guided_video = None
            if save_videos and rollout_i == 0:
                orig_video = str(video_dir / f"video_{task_name}_original.mp4")
                guided_video = str(video_dir / f"video_{task_name}_guided.mp4")

            try:
                orig_rollouts.append(
                    run_single_rollout(
                        fb_model, task_name, z_orig, rollout_seed,
                        num_steps=num_steps, video_output_path=orig_video,
                    )
                )
                guided_rollouts.append(
                    run_single_rollout(
                        fb_model, task_name, z_guided, rollout_seed,
                        num_steps=num_steps, video_output_path=guided_video,
                    )
                )
            except Exception as exc:
                print(f"  [FAILED rollout] {exc}")

        agg_orig = aggregate_rollouts(orig_rollouts)
        agg_guided = aggregate_rollouts(guided_rollouts)

        # Native return comparison
        q_orig = agg_orig["Q_gamma_0.99"]["mean"]
        q_guided = agg_guided["Q_gamma_0.99"]["mean"]
        q_diff = q_guided - q_orig
        q_ratio = float(q_guided / max(q_orig, 1e-8)) if abs(q_orig) >= ratio_baseline_threshold else None
        q_pct = float(q_diff / abs(q_orig) * 100.0) if abs(q_orig) >= ratio_baseline_threshold else None

        g_orig = agg_orig["undiscounted_return"]["mean"]
        g_guided = agg_guided["undiscounted_return"]["mean"]
        g_diff = g_guided - g_orig

        # Proxy return comparison (identical to z-inference)
        proxy_q_orig = agg_orig["proxy_Q_gamma_0.99"]["mean"]
        proxy_q_guided = agg_guided["proxy_Q_gamma_0.99"]["mean"]
        proxy_q_diff = proxy_q_guided - proxy_q_orig

        proxy_g_orig = agg_orig["proxy_undiscounted_return"]["mean"]
        proxy_g_guided = agg_guided["proxy_undiscounted_return"]["mean"]
        proxy_g_diff = proxy_g_guided - proxy_g_orig

        print(
            f"  z_cos={z_cos:.4f} | "
            f"Native Q99: orig={q_orig:.2f}, guided={q_guided:.2f} (diff={q_diff:+.2f}) | "
            f"Proxy Q99: orig={proxy_q_orig:.2f}, guided={proxy_q_guided:.2f} (diff={proxy_q_diff:+.2f})"
        )

        task_result = {
            "status": "ok",
            "proxy_metadata": asdict(meta),
            "z_cosine_similarity": z_cos,
            "z_original_norm": z_orig_norm,
            "z_guided_norm": z_guided_norm,
            "z_norm_ratio": z_norm_ratio,
            "reward_stats_original_inference_set": reward_orig_stats,
            "reward_stats_guided_inference_set": reward_guided_stats,
            "original_baseline": agg_orig,
            "guided_model": agg_guided,
            "q_0.99_difference": q_diff,
            "q_0.99_ratio": q_ratio,
            "q_0.99_percent_change": q_pct,
            "return_difference": g_diff,
            "proxy_q_0.99_difference": proxy_q_diff,
            "proxy_return_difference": proxy_g_diff,
        }
        results["results_per_task"][task_name] = task_result

        csv_rows.append({
            "task": task_name,
            "z_cosine": z_cos,
            "z_original_norm": z_orig_norm,
            "z_guided_norm": z_guided_norm,
            "z_norm_ratio": z_norm_ratio,
            "native_q99_orig": q_orig,
            "native_q99_guided": q_guided,
            "native_q99_diff": q_diff,
            "native_return_orig": g_orig,
            "native_return_guided": g_guided,
            "native_return_diff": g_diff,
            "proxy_q99_orig": proxy_q_orig,
            "proxy_q99_guided": proxy_q_guided,
            "proxy_q99_diff": proxy_q_diff,
            "proxy_return_orig": proxy_g_orig,
            "proxy_return_guided": proxy_g_guided,
            "proxy_return_diff": proxy_g_diff,
        })

    valid = [r for r in results["results_per_task"].values() if r.get("status") == "ok"]
    diffs = np.asarray([r["q_0.99_difference"] for r in valid], dtype=np.float64)
    proxy_diffs = np.asarray([r["proxy_q_0.99_difference"] for r in valid], dtype=np.float64)
    cos = np.asarray([r["z_cosine_similarity"] for r in valid], dtype=np.float64)

    results["aggregate"] = {
        "tasks_evaluated": len(valid),
        "native_tasks_improved": int((diffs > 1e-8).sum()),
        "native_tasks_degraded": int((diffs < -1e-8).sum()),
        "native_mean_q99_diff": float(diffs.mean()) if len(diffs) else 0.0,
        "proxy_tasks_improved": int((proxy_diffs > 1e-8).sum()),
        "proxy_tasks_degraded": int((proxy_diffs < -1e-8).sum()),
        "proxy_mean_q99_diff": float(proxy_diffs.mean()) if len(proxy_diffs) else 0.0,
        "mean_z_cosine": float(cos.mean()) if len(cos) else 0.0,
        "elapsed_seconds": time.time() - start_all,
    }

    sanitized_results = _sanitize_for_json(results)
    out_json = exp_path / "deep_evaluation_785d_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(sanitized_results, f, indent=2)

    out_csv = exp_path / "deep_evaluation_785d_summary.csv"
    if csv_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 78)
    print(" EVALUATION COMPLETED")
    print(f" JSON: {out_json}")
    print(f" CSV : {out_csv}")
    if save_videos:
        print(f" Videos: {video_dir}")
    print("=" * 78)
    return out_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="785-D Original-vs-Guided evaluation with Unified Reward")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument(
        "--original_hdf5",
        type=str,
        default="/home/kyoungsookim/rho1/metamotivo-M-1-datasets/data/buffer_inference_500000.hdf5",
    )
    parser.add_argument("--num_inference_samples", type=int, default=500_000)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--regenerate_dataset", action="store_true")
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument(
        "--ratio_baseline_threshold",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/metamotivo-M-1",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path or filename of a specific checkpoint in or outside exp_dir",
    )
    args = parser.parse_args()

    deep_evaluate(
        exp_dir=args.exp_dir,
        original_hdf5=args.original_hdf5,
        num_inference_samples=args.num_inference_samples,
        num_rollouts=args.num_rollouts,
        num_steps=args.num_steps,
        seed=args.seed,
        regenerate_dataset=args.regenerate_dataset,
        save_videos=not args.no_videos,
        ratio_baseline_threshold=args.ratio_baseline_threshold,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
    )
