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

    def __init__(self, env, policy_path: str, device: str = "cuda:0"):
        self.env = env
        self.device = device
        self._robot = env.scene["robot"]

        # Velocity command [vx, vy, vyaw, height]
        self._command = torch.tensor(
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
        """Set velocity command. Sign convention matches gateway (positive = forward/left/CCW)."""
        h = height if height is not None else self.DEFAULT_HEIGHT
        self._command[0] = torch.tensor(
            [vx, vy, vyaw, h], device=self.device, dtype=torch.float32,
        )

    def stop(self) -> None:
        self.set_command(0.0, 0.0, 0.0)

    def compute_action(self) -> torch.Tensor:
        """Build observation, run ONNX policy, return full joint action tensor.

        Returns [1, N_joints] tensor for env.step() with
        JointPositionActionCfg(scale=1.0, use_default_offset=True).
        """
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

        return self._full_action.unsqueeze(0)  # [1, N_joints]


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
        self._locomotion: LocomotionController | None = None

    def init(self) -> None:
        import gymnasium as gym

        # Register Unitree G1 tasks — this triggers gym.register() calls
        import tasks  # noqa: F401

        # Isaac Lab env config parsing (must be after AppLauncher init)
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        print(f"[isaac] Creating env: {self.task} (device={self.device})", flush=True)

        env_cfg = parse_env_cfg(self.task, device=self.device, num_envs=1)
        env_cfg.env_name = self.task
        env_cfg.seed = self.seed
        # Long episode to avoid 20-second truncation (default is 20s)
        env_cfg.episode_length_s = 3600.0

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

        print(f"[isaac] Env ready: {self.task} ({len(self._joint_pos)}-DOF)", flush=True)

        # Initialize locomotion controller for Wholebody tasks
        if "Wholebody" in self.task or "wholebody" in self.task:
            project_root = os.environ.get("PROJECT_ROOT", "")
            policy_path = os.path.join(project_root, "assets", "model", "policy.onnx")
            if os.path.isfile(policy_path):
                self._locomotion = LocomotionController(
                    self.env, policy_path, self.device,
                )
                print(f"[isaac] Locomotion controller initialized", flush=True)
            else:
                print(f"[isaac] WARNING: No locomotion policy at {policy_path}", flush=True)

    def step(self) -> None:
        if self.env is None:
            return

        if self._locomotion is not None:
            # Use locomotion policy to compute joint targets
            action = self._locomotion.compute_action()
        else:
            # No locomotion — zero action (hold position), needs [1, N] shape
            n = self.env.action_space.shape[-1]
            action = torch.zeros(1, n, device=self.device)

        try:
            self.obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception as exc:
            # Physics error (e.g. robot fell, collision solver failed).
            # Reset the environment to recover.
            print(f"[isaac] env.step failed: {exc}, resetting env", flush=True)
            try:
                self.obs, _ = self.env.reset()
                self._extract_state()
                if self._locomotion:
                    self._locomotion.stop()
            except Exception as reset_exc:
                print(f"[isaac] env.reset also failed: {reset_exc}", flush=True)
            return

        self._step_count += 1
        self._extract_state()

        # Auto-reset on termination
        if terminated.any() or truncated.any():
            print(f"[isaac] Episode ended at step {self._step_count}, resetting", flush=True)
            self.obs, _ = self.env.reset()
            self._extract_state()
            if self._locomotion:
                self._locomotion.stop()

    def _extract_state(self) -> None:
        """Pull joint state from the first articulation."""
        try:
            for name, art in self._articulations.items():
                self._joint_pos = art.data.joint_pos[0].cpu().numpy().tolist()
                self._joint_vel = art.data.joint_vel[0].cpu().numpy().tolist()
                break
        except Exception as exc:
            logger.debug("Joint state extraction failed: %s", exc)

    def capture_camera(self, name: str = "head_rgb") -> bytes | None:
        """Capture JPEG from a camera sensor in the scene.

        Tries to match *name* against sensor names.  Falls back to the first
        camera sensor if no match.
        """
        try:
            camera_sensor = None

            # Try exact match first
            if name in self._sensors:
                camera_sensor = self._sensors[name]
            else:
                # Try partial match, or just grab the first camera
                for sensor_name, sensor in self._sensors.items():
                    if "camera" in sensor_name.lower():
                        camera_sensor = sensor
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

    def get_joint_positions(self) -> list[float]:
        return self._joint_pos

    def get_joint_velocities(self) -> list[float]:
        return self._joint_vel

    def get_imu(self) -> dict:
        try:
            for name, art in self._articulations.items():
                root_quat = art.data.root_quat_w[0].cpu().numpy().tolist()
                root_ang = art.data.root_ang_vel_w[0].cpu().numpy().tolist()
                return {"quaternion": root_quat, "gyroscope": root_ang}
        except Exception:
            pass
        return {"quaternion": [1, 0, 0, 0], "gyroscope": [0, 0, 0]}

    def get_world_pose(self) -> dict:
        try:
            for name, art in self._articulations.items():
                pos = art.data.root_pos_w[0].cpu().numpy()
                quat = art.data.root_quat_w[0].cpu().numpy()  # w,x,y,z
                # Extract yaw from quaternion
                w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                theta = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                return {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "theta": theta,
                }
        except Exception:
            pass
        return {"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0}

    def apply_walk(self, vx: float, vy: float, vyaw: float) -> None:
        self._walking = True
        self._vx = vx
        self._vy = vy
        self._vyaw = vyaw
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

    def shutdown(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:
                logger.warning("Error closing env: %s", exc)
            self.env = None
        print("[isaac] Env closed", flush=True)
