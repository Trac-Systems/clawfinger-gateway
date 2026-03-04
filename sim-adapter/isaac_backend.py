"""Isaac Lab backend for the Clawfinger sim adapter.

Requires:
  - Isaac Sim 5.1 conda env (g1sim)
  - unitree_sim_isaaclab checked out at /home/muffin/robotics/unitree_sim_isaaclab

IMPORTANT: This module must be imported AFTER AppLauncher has been
initialized.  adapter.py handles this in the Isaac launch path.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import threading
import time
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("sim_adapter.isaac")


# ---------------------------------------------------------------------------
# Locomotion controller — wraps the pre-trained ONNX walking policy
# ---------------------------------------------------------------------------

# The 12 leg joints that the locomotion policy controls (output order).
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

# The 14 arm joints (used in observation vector).
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# The 29 joints in the specific order used for the "previous actions"
# dimension of the observation vector.
OLD_ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


class LocomotionController:
    """Wraps the pre-trained ONNX locomotion policy for the G1.

    Matches the DDSRLActionProvider from unitree_sim_isaaclab exactly:
    - 910-dim observation (91 features x 10 history frames)
    - 12 leg joint position deltas as output
    - 5-frame action delay buffer
    - ACTION_SCALE = 0.25

    Observation vector (91 per frame):
      - ang_vel [3]: body angular velocity in body frame
      - projected_gravity [3]: gravity vector projected into body frame
      - command [4]: [vx, vy, vyaw, height]
      - joint_pos_offset [26]: (leg+arm) joint positions - defaults
      - joint_vel_offset [26]: (leg+arm) joint velocities - defaults
      - prev_actions [29]: previous full action in old_action ordering
    """

    ACTION_SCALE = 0.25
    CLIP_OBS = 100.0
    CLIP_ACTIONS = 100.0
    DEFAULT_HEIGHT = 0.8
    # Max velocity change per step (m/s per step at 50Hz).
    # 0.02/step → 0→0.3 takes 15 steps = 0.3s ramp.
    COMMAND_RAMP_RATE = 0.02

    def __init__(self, env, policy_path: str, device: str = "cuda:0"):
        self.env = env
        self.device = device
        self._robot = env.scene["robot"]

        # Velocity command — _target is what the user asked for,
        # _command is what we actually feed to the policy (ramped).
        self._command = torch.tensor(
            [0.0, 0.0, 0.0, self.DEFAULT_HEIGHT],
            device=device, dtype=torch.float32,
        ).unsqueeze(0)  # [1, 4]
        self._target_command = torch.tensor(
            [0.0, 0.0, 0.0, self.DEFAULT_HEIGHT],
            device=device, dtype=torch.float32,
        ).unsqueeze(0)  # [1, 4]

        # Build joint index mappings
        all_names = list(self._robot.data.joint_names)
        self._name_to_idx = {n: i for i, n in enumerate(all_names)}
        self._n_joints = len(all_names)
        print(f"[locomotion] Joint names ({len(all_names)}): {all_names}", flush=True)

        self._leg_indices = [self._name_to_idx[n] for n in LEG_JOINT_NAMES]
        self._arm_indices = [self._name_to_idx[n] for n in ARM_JOINT_NAMES]
        self._waist_indices = [self._name_to_idx[n] for n in WAIST_JOINT_NAMES]
        # Observation joint order: legs + arms (26 joints)
        self._obs_indices = self._leg_indices + self._arm_indices
        # Old action joint order (29 joints) for prev_actions in obs
        self._old_action_indices = [self._name_to_idx[n] for n in OLD_ACTION_JOINT_NAMES]

        # Pre-compute index mapping: leg joint name → index in OLD_ACTION order
        self._leg_to_old = [OLD_ACTION_JOINT_NAMES.index(n) for n in LEG_JOINT_NAMES]
        # Pre-compute as tensors for vectorized scatter/gather
        self._leg_to_old_t = torch.tensor(self._leg_to_old, device=device, dtype=torch.long)

        # Default positions / velocities
        self._default_pos = self._robot.data.default_joint_pos  # [1, N]
        self._default_vel = self._robot.data.default_joint_vel  # [1, N]

        # Waist default positions in old_action order (for prev_action vector)
        waist_old_indices = [OLD_ACTION_JOINT_NAMES.index(n) for n in WAIST_JOINT_NAMES]
        self._waist_old_indices = waist_old_indices

        # Load ONNX policy
        import onnxruntime as ort
        self._ort_session = ort.InferenceSession(policy_path)
        self._input_name = self._ort_session.get_inputs()[0].name
        print(f"[locomotion] ONNX policy loaded: {policy_path}", flush=True)
        print(f"[locomotion] Input: {self._ort_session.get_inputs()[0].shape}", flush=True)
        print(f"[locomotion] Output: {self._ort_session.get_outputs()[0].shape}", flush=True)

        # History buffer (10 frames of 91-dim observations)
        from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
        self._obs_buffer = CircularBuffer(max_len=10, batch_size=1, device=device)
        # Action delay buffer (5 frames)
        n_old = len(OLD_ACTION_JOINT_NAMES)  # 29
        self._action_buffer = DelayBuffer(5, 1, device=device)
        self._action_buffer.compute(
            torch.zeros(1, n_old, dtype=torch.float32, device=device)
        )

        # Track previous raw action for observation building (avoids expensive
        # CircularBuffer.buffer property which does clone+roll+transpose)
        self._prev_raw_action = torch.zeros(
            1, n_old, dtype=torch.float32, device=device,
        )

        # Full action buffer (reused each step)
        self._full_action = torch.zeros(self._n_joints, device=device, dtype=torch.float32)

        # Reusable buffers
        self._full_old_action = torch.zeros(1, n_old, device=device, dtype=torch.float32)

        print(f"[locomotion] Controller ready: {self._n_joints} joints, "
              f"{len(self._leg_indices)} leg, {len(self._arm_indices)} arm", flush=True)

    def set_command(self, vx: float, vy: float, vyaw: float,
                    height: float | None = None) -> None:
        """Set velocity command target. Actual command ramps toward this."""
        h = height if height is not None else self.DEFAULT_HEIGHT
        old = self._target_command[0].tolist()
        self._target_command[0] = torch.tensor(
            [vx, vy, vyaw, h], device=self.device, dtype=torch.float32,
        )
        # Trace command changes for debugging
        if abs(vx) > 1e-3 or abs(vy) > 1e-3 or abs(vyaw) > 1e-3:
            print(f"[locomotion] set_command({vx:.3f},{vy:.3f},{vyaw:.3f}) "
                  f"old_target=({old[0]:.3f},{old[1]:.3f},{old[2]:.3f})", flush=True)
        elif abs(old[0]) > 1e-3 or abs(old[1]) > 1e-3 or abs(old[2]) > 1e-3:
            import traceback
            print(f"[locomotion] ZEROED target! was ({old[0]:.3f},{old[1]:.3f},{old[2]:.3f})", flush=True)
            traceback.print_stack()

    def stop(self) -> None:
        self.set_command(0.0, 0.0, 0.0)

    def clear_history(self) -> None:
        """Reset observation and action history buffers.

        Call after env.reset() to give the policy a clean start instead of
        stale history from the previous episode.
        """
        from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
        n_old = len(OLD_ACTION_JOINT_NAMES)
        # Re-create observation history buffer (10 frames of zeros)
        self._obs_buffer = CircularBuffer(max_len=10, batch_size=1, device=self.device)
        # Re-create action delay buffer (5 frames of zeros)
        self._action_buffer = DelayBuffer(5, 1, device=self.device)
        self._action_buffer.compute(
            torch.zeros(1, n_old, dtype=torch.float32, device=self.device)
        )
        self._prev_raw_action.zero_()
        self._full_old_action.zero_()
        self._full_action.zero_()
        self._command[0] = torch.tensor(
            [0.0, 0.0, 0.0, self.DEFAULT_HEIGHT],
            device=self.device, dtype=torch.float32,
        )
        self._target_command[0] = torch.tensor(
            [0.0, 0.0, 0.0, self.DEFAULT_HEIGHT],
            device=self.device, dtype=torch.float32,
        )
        print("[locomotion] History buffers cleared", flush=True)

    def _ramp_command(self) -> None:
        """Move _command toward _target_command by COMMAND_RAMP_RATE per step."""
        diff = self._target_command - self._command
        # Clamp each component's delta to ±COMMAND_RAMP_RATE
        clamped = torch.clamp(diff, -self.COMMAND_RAMP_RATE, self.COMMAND_RAMP_RATE)
        self._command += clamped

    @property
    def is_moving(self) -> bool:
        """True when a non-zero velocity command is set."""
        c = self._command[0]
        return abs(c[0].item()) > 1e-3 or abs(c[1].item()) > 1e-3 or abs(c[2].item()) > 1e-3

    def compute_action(self) -> torch.Tensor:
        """Build observation, run ONNX policy, return full joint action tensor.

        Returns [1, N_joints] tensor for env.step() with
        JointPositionActionCfg(scale=1.0, use_default_offset=True).
        """
        # Ramp actual command toward target (prevents abrupt velocity jumps)
        self._ramp_command()

        robot = self._robot

        # Current state
        ang_vel = robot.data.root_ang_vel_b  # [1, 3]
        gravity = robot.data.projected_gravity_b  # [1, 3]
        joint_pos = robot.data.joint_pos  # [1, N]
        joint_vel = robot.data.joint_vel  # [1, N]

        # Build 91-dim observation (matches DDSRLActionProvider exactly)
        obs = torch.cat([
            ang_vel,                                                        # [1, 3]
            gravity,                                                        # [1, 3]
            self._command,                                                  # [1, 4]
            (joint_pos[:, self._obs_indices] -
             self._default_pos[:, self._obs_indices]),                      # [1, 26]
            (joint_vel[:, self._obs_indices] -
             self._default_vel[:, self._obs_indices]),                      # [1, 26]
            self._prev_raw_action,                                          # [1, 29]
        ], dim=-1)  # [1, 91]

        # Append to history and get 910-dim input
        self._obs_buffer.append(obs)
        obs_hist = self._obs_buffer.buffer.reshape(1, -1)  # [1, 910]
        obs_hist = torch.clip(obs_hist, -self.CLIP_OBS, self.CLIP_OBS)

        # ONNX inference
        ort_input = {self._input_name: obs_hist.cpu().numpy()}
        ort_out = self._ort_session.run(None, ort_input)
        policy_output = torch.tensor(ort_out[0], device=self.device)  # [1, 12]

        # Build full action in old_action order (29 joints)
        self._full_old_action.zero_()
        # Scatter 12 policy outputs into old_action positions
        self._full_old_action[0].scatter_(
            0, self._leg_to_old_t, policy_output[0],
        )
        # Waist joints get default positions (matches original DDSRLActionProvider)
        for wi in self._waist_old_indices:
            waist_name = OLD_ACTION_JOINT_NAMES[wi]
            all_idx = self._name_to_idx[waist_name]
            self._full_old_action[0, wi] = self._default_pos[0, all_idx]

        # Store in delay buffer and update prev_raw_action
        self._prev_raw_action.copy_(self._full_old_action)
        delayed = self._action_buffer.compute(self._full_old_action)

        # Extract delayed leg actions using pre-computed index mapping
        delayed_leg = delayed[0].gather(0, self._leg_to_old_t)  # [12]
        clipped = torch.clip(delayed_leg, -self.CLIP_ACTIONS, self.CLIP_ACTIONS)

        # Build full action for env.step:
        # env uses JointPositionActionCfg(scale=1.0, use_default_offset=True)
        # → target = action * 1.0 + default_pos
        # Original code: target = clipped * ACTION_SCALE + default_pos
        # So: action = clipped * ACTION_SCALE
        self._full_action.zero_()
        for i, idx in enumerate(self._leg_indices):
            self._full_action[idx] = clipped[i] * self.ACTION_SCALE

        # Debug: log action magnitude and command every 50 steps
        self._dbg_count = getattr(self, '_dbg_count', 0) + 1
        if self._dbg_count % 50 == 0:
            cmd = self._command[0]
            act_mag = float(self._full_action.abs().max().item())
            leg_mag = float(clipped.abs().max().item())
            print(f"[locomotion] cmd=[{cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f}] "
                  f"leg_max={leg_mag:.3f} act_max={act_mag:.4f}", flush=True)

        return self._full_action.unsqueeze(0)  # [1, N_joints]


# ---------------------------------------------------------------------------
# Velocity Locomotion Controller — Isaac Lab built-in G1 locomotion policy
# ---------------------------------------------------------------------------

class VelocityLocomotionController:
    """Wraps an ONNX policy trained with Isaac Lab's velocity locomotion task.

    Observation vector (123-dim, single frame):
      - base_lin_vel [3]: base linear velocity in body frame
      - base_ang_vel [3]: base angular velocity in body frame
      - projected_gravity [3]: gravity in body frame
      - velocity_commands [3]: [vx, vy, vyaw]
      - joint_pos_rel [N]: joint positions - defaults
      - joint_vel_rel [N]: joint velocities - defaults  (vel scale 1.0)
      - last_action [N]: previous action

    Action: [N] joint position offsets, applied via
      JointPositionActionCfg(scale=0.5, use_default_offset=True)
      → target = action * 0.5 + default_pos
    """

    ACTION_SCALE = 0.5
    CLIP_ACTIONS = 100.0
    CLIP_OBS = 100.0
    # Max velocity change per step (m/s per step at 50Hz).
    COMMAND_RAMP_RATE = 0.02

    def __init__(self, env, policy_path: str, device: str = "cuda:0"):
        self.env = env
        self.device = device
        self._robot = env.scene["robot"]

        # Velocity command [vx, vy, vyaw]
        self._command = torch.zeros(1, 3, device=device, dtype=torch.float32)
        self._target_command = torch.zeros(1, 3, device=device, dtype=torch.float32)

        # Joint info
        all_names = list(self._robot.data.joint_names)
        self._n_joints = len(all_names)
        self._default_pos = self._robot.data.default_joint_pos  # [1, N]
        self._default_vel = self._robot.data.default_joint_vel  # [1, N]

        print(f"[velocity_loco] {self._n_joints} joints: {all_names}", flush=True)

        # Load ONNX policy
        import onnxruntime as ort
        self._ort_session = ort.InferenceSession(policy_path)
        self._input_name = self._ort_session.get_inputs()[0].name
        inp_shape = self._ort_session.get_inputs()[0].shape
        out_shape = self._ort_session.get_outputs()[0].shape
        print(f"[velocity_loco] ONNX: input={inp_shape} output={out_shape}", flush=True)

        # Previous action (for observation)
        self._prev_action = torch.zeros(1, self._n_joints, device=device, dtype=torch.float32)

        print(f"[velocity_loco] Controller ready", flush=True)

    def set_command(self, vx: float, vy: float, vyaw: float,
                    height: float | None = None) -> None:
        """Set velocity command target. Actual command ramps toward this."""
        old = self._target_command[0].tolist()
        self._target_command[0] = torch.tensor(
            [vx, vy, vyaw], device=self.device, dtype=torch.float32,
        )
        if abs(vx) > 1e-3 or abs(vy) > 1e-3 or abs(vyaw) > 1e-3:
            print(f"[velocity_loco] set_command({vx:.3f},{vy:.3f},{vyaw:.3f})", flush=True)
        elif abs(old[0]) > 1e-3 or abs(old[1]) > 1e-3 or abs(old[2]) > 1e-3:
            print(f"[velocity_loco] ZEROED target", flush=True)

    def stop(self) -> None:
        self.set_command(0.0, 0.0, 0.0)

    def clear_history(self) -> None:
        """Reset state after env reset."""
        self._prev_action.zero_()
        self._command.zero_()
        self._target_command.zero_()
        print("[velocity_loco] History cleared", flush=True)

    def _ramp_command(self) -> None:
        diff = self._target_command - self._command
        clamped = torch.clamp(diff, -self.COMMAND_RAMP_RATE, self.COMMAND_RAMP_RATE)
        self._command += clamped

    @property
    def is_moving(self) -> bool:
        c = self._command[0]
        return abs(c[0].item()) > 1e-3 or abs(c[1].item()) > 1e-3 or abs(c[2].item()) > 1e-3

    def compute_action(self) -> torch.Tensor:
        """Build 123-dim obs, run ONNX, return [1, N_joints] action."""
        self._ramp_command()
        robot = self._robot

        # Body-frame velocities
        lin_vel = robot.data.root_lin_vel_b    # [1, 3]
        ang_vel = robot.data.root_ang_vel_b    # [1, 3]
        gravity = robot.data.projected_gravity_b  # [1, 3]

        # Joint state relative to defaults
        joint_pos_rel = robot.data.joint_pos - self._default_pos  # [1, N]
        joint_vel_rel = robot.data.joint_vel - self._default_vel  # [1, N]

        # Build observation: [3+3+3+3+N+N+N] = [12 + 3*N]
        obs = torch.cat([
            lin_vel,           # [1, 3]
            ang_vel,           # [1, 3]
            gravity,           # [1, 3]
            self._command,     # [1, 3]
            joint_pos_rel,     # [1, N]
            joint_vel_rel,     # [1, N]
            self._prev_action, # [1, N]
        ], dim=-1)

        obs = torch.clip(obs, -self.CLIP_OBS, self.CLIP_OBS)

        # ONNX inference
        ort_input = {self._input_name: obs.cpu().numpy()}
        ort_out = self._ort_session.run(None, ort_input)
        action = torch.tensor(ort_out[0], device=self.device)  # [1, N]

        action = torch.clip(action, -self.CLIP_ACTIONS, self.CLIP_ACTIONS)

        # Store for next observation
        self._prev_action = action.clone()

        # Debug every 50 steps
        self._dbg_count = getattr(self, '_dbg_count', 0) + 1
        if self._dbg_count % 50 == 0:
            cmd = self._command[0]
            act_mag = float(action.abs().max().item())
            print(f"[velocity_loco] cmd=[{cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f}] "
                  f"act_max={act_mag:.4f}", flush=True)

        # Return raw action — env.step() applies JointPositionActionCfg
        # (scale=0.5, use_default_offset=True) → target = action * 0.5 + default_pos
        return action


# ---------------------------------------------------------------------------
# Isaac Sim Backend
# ---------------------------------------------------------------------------

class IsaacSimBackend:
    """Isaac Lab gymnasium env wrapper for the G1 robot."""

    def __init__(
        self,
        task: str = "Isaac-Move-Cylinder-G129-Dex1-Wholebody",
        device: str = "cuda:0",
        headless: bool = True,
        seed: int = 42,
    ):
        self.task = task
        self.device = device
        self.headless = headless
        self.seed = seed
        self.env = None
        self.obs = None
        self._step_count = 0
        self._walking = False
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self._joint_pos: list[float] = []
        self._joint_vel: list[float] = []
        self._sensors: dict = {}
        self._articulations: dict = {}
        self._locomotion: LocomotionController | VelocityLocomotionController | None = None

        # --- Thread-safe cached state (background threads read these, never PhysX) ---
        self._phys_lock = threading.Lock()  # Protects sim.step() / env.step()
        self._cached_pose: dict = {"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0}
        self._cached_imu: dict = {"quaternion": [1, 0, 0, 0], "gyroscope": [0, 0, 0]}
        self._cached_camera_frame: bytes | None = None
        self._camera_capture_interval = 0  # Set after init: sim_hz / camera_fps
        self._camera_step_counter = 0

        # --- Warmup: hold robot at spawn for N steps after reset ---
        # Policy runs at full STANDING_ACTION_SCALE (so it can balance), but
        # root pose is kinematically pinned to spawn each step. This fills the
        # observation buffer with real standing data before releasing control.
        self.WARMUP_STEPS = 200  # 200 steps at 50Hz = 4 seconds
        self._warmup_remaining = 0  # Decremented each step; 0 = warmup done

    def init(self) -> None:
        import gymnasium as gym

        is_velocity_task = "Velocity" in self.task

        if not is_velocity_task:
            # Register Unitree G1 wholebody tasks
            import tasks  # noqa: F401

        # Isaac Lab env config parsing (must be after AppLauncher init)
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        print(f"[isaac] Creating env: {self.task} (device={self.device})", flush=True)

        env_cfg = parse_env_cfg(self.task, device=self.device, num_envs=1)
        env_cfg.env_name = self.task
        env_cfg.seed = self.seed
        # Long episode to avoid 20-second truncation (default is 20s)
        env_cfg.episode_length_s = 3600.0

        if is_velocity_task:
            # Disable observation noise/corruption for deployment
            if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
                env_cfg.observations.policy.enable_corruption = False
            # Remove height scan for flat tasks (already None for flat, but be safe)
            if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations.policy, "height_scan"):
                env_cfg.observations.policy.height_scan = None
            # Disable random events
            if hasattr(env_cfg, "events"):
                if hasattr(env_cfg.events, "base_external_force_torque"):
                    env_cfg.events.base_external_force_torque = None
                if hasattr(env_cfg.events, "push_robot"):
                    env_cfg.events.push_robot = None

        # Add front camera to scene (velocity tasks don't include one by default).
        # Attach to pelvis (root body, always exists) with offset looking forward.
        if not hasattr(env_cfg.scene, "front_camera") or env_cfg.scene.front_camera is None:
            try:
                import isaaclab.sim as sim_utils
                from isaaclab.sensors import CameraCfg
                env_cfg.scene.front_camera = CameraCfg(
                    prim_path="/World/envs/env_.*/Robot/pelvis/front_cam",
                    update_period=0.5,  # 2 fps — matches adapter --camera-fps 2
                    height=480,
                    width=640,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=7.6,
                        focus_distance=400.0,
                        horizontal_aperture=20.0,
                        clipping_range=(0.1, 1.0e5),
                    ),
                    offset=CameraCfg.OffsetCfg(
                        pos=(0.25, 0.0, 0.45),  # Forward + up from pelvis ≈ head height
                        rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention: looking forward
                        convention="ros",
                    ),
                )
                print("[isaac] Added front_camera to scene config (pelvis mount)", flush=True)
            except Exception as exc:
                print(f"[isaac] WARNING: Could not add front_camera: {exc}", flush=True)

        self.env = gym.make(self.task, cfg=env_cfg).unwrapped
        self.env.seed(self.seed)

        # Discover sensors and articulations
        self._sensors = getattr(self.env.scene, "sensors", {})
        self._articulations = getattr(self.env.scene, "articulations", {})

        print(f"[isaac] Sensors: {list(self._sensors.keys())}", flush=True)
        print(f"[isaac] Articulations: {list(self._articulations.keys())}", flush=True)

        # Reset to get initial observations
        self.obs, _ = self.env.reset()
        self._extract_state()
        self._update_cached_state()
        self._warmup_remaining = self.WARMUP_STEPS

        # Set spawn position per task
        if is_velocity_task:
            self._SPAWN_POS = (0.0, 0.0, 0.74)
        else:
            self._SPAWN_POS = (-3.9, -2.81811, 0.8)

        print(f"[isaac] Env ready: {self.task} ({len(self._joint_pos)}-DOF)", flush=True)

        # Initialize locomotion controller
        if is_velocity_task:
            # Isaac Lab velocity locomotion task — use our trained ONNX policy
            script_dir = os.path.dirname(os.path.abspath(__file__))
            policy_path = os.path.join(script_dir, "assets", "g1_locomotion.onnx")
            if not os.path.isfile(policy_path):
                # Fallback: check home directory
                policy_path = os.path.expanduser("~/robotics/g1_locomotion.onnx")
            if os.path.isfile(policy_path):
                self._locomotion = VelocityLocomotionController(
                    self.env, policy_path, self.device,
                )
                print(f"[isaac] Velocity locomotion controller initialized", flush=True)
            else:
                print(f"[isaac] WARNING: No locomotion policy found", flush=True)
        elif "Wholebody" in self.task or "wholebody" in self.task:
            project_root = os.environ.get("PROJECT_ROOT", "")
            policy_path = os.path.join(project_root, "assets", "model", "policy.onnx")
            if os.path.isfile(policy_path):
                self._locomotion = LocomotionController(
                    self.env, policy_path, self.device,
                )
                print(f"[isaac] Wholebody locomotion controller initialized", flush=True)
            else:
                print(f"[isaac] WARNING: No locomotion policy at {policy_path}", flush=True)

    _fatal = False  # Set when env.reset() fails — signals process should exit
    _reset_requested = False  # Thread-safe reset flag (set by command handler, executed by main thread)
    _reset_result = None  # Result of last reset (True/False), read by command handler

    # Spawn position — set per task in init()
    _SPAWN_POS = (0.0, 0.0, 0.74)
    _SPAWN_QUAT = (1.0, 0.0, 0.0, 0.0)  # w,x,y,z

    # Standing damping: reduce policy actions when velocity command is zero.
    # The RL policy produces large actions (act_max ~5-7) even at zero command.
    # With env action scale=0.5, unscaled standing gives ~2.5 radian joint offsets
    # causing fast circular drift. Legacy wholebody (ACTION_SCALE=0.25) uses 0.65.
    # Velocity loco (env scale=0.5) needs a much tighter standing scale.
    STANDING_ACTION_SCALE = 0.65       # Legacy wholebody controller
    STANDING_ACTION_SCALE_VELO = 0.3   # Velocity controller: 5*0.3*0.5 = 0.75 rad

    def _reset_env(self, reason: str) -> bool:
        """Reset the environment and locomotion controller.

        MUST be called from the main thread (where Kit/physics runs).
        Returns True on success.
        """
        print(f"[isaac] Resetting env: {reason} (step {self._step_count})", flush=True)
        # Preserve walking state so we can re-apply after reset
        was_walking = self._walking
        saved_vx, saved_vy, saved_vyaw = self._vx, self._vy, self._vyaw
        _is_velocity_loco = isinstance(self._locomotion, VelocityLocomotionController)
        try:
            self.obs, _ = self.env.reset()
            # Force-teleport robot to spawn (env.reset may not move it)
            for _name, art in self._articulations.items():
                spawn_pose = torch.tensor(
                    [[*self._SPAWN_POS, *self._SPAWN_QUAT]],
                    device=self.device, dtype=torch.float32,
                )
                art.write_root_pose_to_sim(spawn_pose)
                # Zero velocities after teleport
                art.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))
                break
            # Step the sim once to apply the teleport before policy resumes
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            self.env.scene.update(dt=self.env.physics_dt)
            self.env.sim.render()
            self._extract_state()
            self._update_cached_state()
            if self._locomotion:
                self._locomotion.clear_history()
            self._step_count = 0
            if _is_velocity_loco:
                self._warmup_remaining = 0  # No warmup for velocity policy
            else:
                self._warmup_remaining = self.WARMUP_STEPS
            # Re-apply walking command if we were walking before the reset
            if was_walking and self._locomotion:
                self._walking = True
                self._vx, self._vy, self._vyaw = saved_vx, saved_vy, saved_vyaw
                self._locomotion.set_command(saved_vx, saved_vy, saved_vyaw)
                print(f"[isaac] Restored walk cmd: vx={saved_vx:.3f} vy={saved_vy:.3f} vyaw={saved_vyaw:.3f}", flush=True)
            pose = self._cached_pose
            print(f"[isaac] Reset successful — pos=({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})", flush=True)
            return True
        except Exception as exc:
            print(f"[isaac] FATAL: env.reset failed: {exc}", flush=True)
            print(f"[isaac] CUDA context corrupted — process must restart", flush=True)
            self._fatal = True
            return False

    def request_reset(self) -> None:
        """Request a reset from any thread. The main thread will execute it."""
        self._reset_requested = True

    def step(self) -> None:
        if self.env is None or self._fatal:
            return

        # Handle pending reset in the main thread (thread-safe)
        if self._reset_requested:
            self._reset_requested = False
            ok = self._reset_env("user requested reset")
            self._reset_result = ok
            return

        # --- Compute action (policy inference, no PhysX access) ---
        _is_velocity_loco = isinstance(self._locomotion, VelocityLocomotionController)
        try:
            if self._locomotion is not None:
                action = self._locomotion.compute_action()
                if _is_velocity_loco:
                    # Velocity loco: let the policy run unscaled (it needs full
                    # actions to balance). Drift is countered in the physics
                    # step below by zeroing root velocity when standing.
                    if self._warmup_remaining > 0:
                        self._warmup_remaining -= 1
                        if self._warmup_remaining == 0:
                            print(f"[isaac] Warmup complete — releasing kinematic hold", flush=True)
                else:
                    # Legacy wholebody controller: needs STANDING_ACTION_SCALE
                    if self._warmup_remaining > 0:
                        action = action * self.STANDING_ACTION_SCALE
                        self._warmup_remaining -= 1
                        if self._warmup_remaining == 0:
                            print(f"[isaac] Warmup complete — releasing kinematic hold", flush=True)
                    elif not self._locomotion.is_moving:
                        action = action * self.STANDING_ACTION_SCALE
            else:
                n = self.env.action_space.shape[-1]
                action = torch.zeros(1, n, device=self.device)
        except Exception as exc:
            self._reset_env(f"action compute: {exc}")
            return

        # --- Physics step under lock (prevents background thread PhysX access) ---
        with self._phys_lock:
            try:
                self.obs, reward, terminated, truncated, info = self.env.step(action)
            except Exception as exc:
                self._reset_env(f"{exc}")
                return

            # During warmup: pin the robot's root at spawn position each step.
            # The policy still runs (filling obs buffer), but the base can't drift.
            if self._warmup_remaining > 0:
                try:
                    for _name, art in self._articulations.items():
                        spawn_pose = torch.tensor(
                            [[*self._SPAWN_POS, *self._SPAWN_QUAT]],
                            device=self.device, dtype=torch.float32,
                        )
                        art.write_root_pose_to_sim(spawn_pose)
                        art.write_root_velocity_to_sim(
                            torch.zeros(1, 6, device=self.device),
                        )
                        break
                except Exception as exc:
                    logger.debug("Warmup kinematic hold failed: %s", exc)

            # Standing drift correction: when velocity command is zero, zero
            # out the root linear velocity each step. The policy keeps joints
            # active for balance but the base stays planted. This prevents the
            # slow circular drift caused by asymmetric policy outputs.
            elif _is_velocity_loco and not self._walking:
                try:
                    for _name, art in self._articulations.items():
                        art.write_root_velocity_to_sim(
                            torch.zeros(1, 6, device=self.device),
                        )
                        break
                except Exception:
                    pass

            self._step_count += 1
            self._extract_state()
            self._update_cached_state()

            # Main-thread camera capture (avoids PhysX threading issues)
            self._camera_step_counter += 1
            if self._camera_capture_interval > 0 and \
               self._camera_step_counter >= self._camera_capture_interval:
                self._camera_step_counter = 0
                self._cached_camera_frame = self._capture_camera_raw()

        # Periodic status logging (every 500 steps, outside lock)
        if self._step_count % 500 == 0:
            p = self._cached_pose
            print(f"[isaac] step={self._step_count} z={p['z']:.3f} "
                  f"x={p['x']:.2f} y={p['y']:.2f} "
                  f"walking={self._walking}", flush=True)

        # Auto-reset on termination
        if terminated.any() or truncated.any():
            self._reset_env("episode ended")

    def _extract_state(self) -> None:
        """Pull joint state from the first articulation. Called from main thread under lock."""
        try:
            for name, art in self._articulations.items():
                self._joint_pos = art.data.joint_pos[0].cpu().numpy().tolist()
                self._joint_vel = art.data.joint_vel[0].cpu().numpy().tolist()
                break
        except Exception as exc:
            logger.debug("Joint state extraction failed: %s", exc)

    def _update_cached_state(self) -> None:
        """Cache world pose and IMU from PhysX in the main thread.

        Background threads read these cached values instead of touching PhysX.
        """
        try:
            for _name, art in self._articulations.items():
                pos = art.data.root_pos_w[0].cpu().numpy()
                quat = art.data.root_quat_w[0].cpu().numpy()
                ang_vel = art.data.root_ang_vel_w[0].cpu().numpy()
                w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                theta = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                self._cached_pose = {
                    "x": float(pos[0]), "y": float(pos[1]),
                    "z": float(pos[2]), "theta": theta,
                }
                self._cached_imu = {
                    "quaternion": quat.tolist(),
                    "gyroscope": ang_vel.tolist(),
                }
                break
        except Exception as exc:
            logger.debug("Cached state update failed: %s", exc)

    def _capture_camera_raw(self) -> bytes | None:
        """Capture JPEG from the best camera sensor. Called from main thread under lock."""
        try:
            camera_sensor = None
            preferred = ["front_camera", "robot_camera", "head_camera"]
            for pref in preferred:
                if pref in self._sensors:
                    camera_sensor = self._sensors[pref]
                    break
            if camera_sensor is None:
                for sn, s in self._sensors.items():
                    if "camera" in sn.lower():
                        camera_sensor = s
                        break
            if camera_sensor is None:
                return None
            data = camera_sensor.data
            if not hasattr(data, "output"):
                return None
            rgb = data.output.get("rgb")
            if rgb is None:
                return None
            frame = rgb[0].cpu().numpy().astype(np.uint8)
            from PIL import Image
            img = Image.fromarray(frame[:, :, :3])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=50)
            return buf.getvalue()
        except Exception as exc:
            logger.debug("Camera capture failed: %s", exc)
            return None

    def capture_camera(self, name: str = "head_rgb") -> bytes | None:
        """Return the latest cached camera frame (captured in main thread)."""
        return self._cached_camera_frame

    def get_joint_positions(self) -> list[float]:
        return self._joint_pos

    def get_joint_velocities(self) -> list[float]:
        return self._joint_vel

    def get_imu(self) -> dict:
        """Return cached IMU data (updated in main thread each step)."""
        return self._cached_imu

    def get_world_pose(self) -> dict:
        """Return cached world pose (updated in main thread each step)."""
        return self._cached_pose

    def apply_walk(self, vx: float, vy: float, vyaw: float) -> None:
        self._walking = True
        self._vx = vx
        self._vy = vy
        self._vyaw = vyaw
        print(f"[isaac] apply_walk vx={vx:.3f} vy={vy:.3f} vyaw={vyaw:.3f}", flush=True)
        if self._locomotion:
            self._locomotion.set_command(vx, vy, vyaw)

    def apply_stand(self) -> None:
        self._walking = False
        self._vx = self._vy = self._vyaw = 0.0
        if self._locomotion:
            self._locomotion.stop()

    def apply_sit(self) -> None:
        self._walking = False
        if self._locomotion:
            self._locomotion.set_command(0.0, 0.0, 0.0, height=0.5)

    def apply_stop(self) -> None:
        self._walking = False
        self._vx = self._vy = self._vyaw = 0.0
        if self._locomotion:
            self._locomotion.stop()

    def apply_reset(self) -> bool:
        """Request environment reset (robot back to spawn).

        Thread-safe: sets a flag that the main thread picks up on the next
        sim step.  Waits up to 2 seconds for the main thread to execute it.
        Returns True on success.
        """
        self._walking = False
        self._vx = self._vy = self._vyaw = 0.0
        self._reset_result = None
        self.request_reset()
        # Wait for main thread to execute the reset (up to 2s)
        for _ in range(200):
            if self._reset_result is not None:
                return self._reset_result
            time.sleep(0.01)
        print("[isaac] WARNING: reset timed out waiting for main thread", flush=True)
        return False

    def shutdown(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:
                logger.warning("Error closing env: %s", exc)
            self.env = None
        print("[isaac] Env closed", flush=True)
