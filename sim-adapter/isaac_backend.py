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
import time
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("sim_adapter.isaac")


class IsaacSimBackend:
    """Isaac Lab gymnasium env wrapper for the G1 robot."""

    def __init__(
        self,
        task: str = "Isaac-PickPlace-Cylinder-G129-Dex1-Joint",
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

    def init(self) -> None:
        import gymnasium as gym

        # Register Unitree G1 tasks — this triggers gym.register() calls
        import tasks  # noqa: F401

        # Isaac Lab env config parsing (must be after AppLauncher init)
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        logger.info("Creating Isaac Lab env: %s (device=%s)", self.task, self.device)

        env_cfg = parse_env_cfg(self.task, device=self.device, num_envs=1)
        env_cfg.env_name = self.task
        env_cfg.seed = self.seed

        self.env = gym.make(self.task, cfg=env_cfg).unwrapped
        self.env.seed(self.seed)

        # Discover sensors and articulations
        self._sensors = getattr(self.env.scene, "sensors", {})
        self._articulations = getattr(self.env.scene, "articulations", {})

        logger.info("Sensors: %s", list(self._sensors.keys()))
        logger.info("Articulations: %s", list(self._articulations.keys()))

        # Reset to get initial observations
        self.obs, _ = self.env.reset()
        self._extract_state()

        logger.info("Isaac Lab env ready: %s (%d-DOF)",
                     self.task, len(self._joint_pos))

    def step(self) -> None:
        if self.env is None:
            return

        # Build zero action (idle) — the env maintains the robot pose
        action = torch.zeros(self.env.action_space.shape, device=self.device)

        # TODO: Map walk/stand commands to action dimensions when locomotion
        # policy is integrated.  For now the robot holds position.

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._extract_state()

        # Auto-reset on termination
        if terminated.any() or truncated.any():
            logger.debug("Episode ended at step %d, resetting", self._step_count)
            self.obs, _ = self.env.reset()

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

    def apply_stand(self) -> None:
        self._walking = False
        self._vx = self._vy = self._vyaw = 0.0

    def apply_sit(self) -> None:
        self._walking = False

    def apply_stop(self) -> None:
        self._walking = False
        self._vx = self._vy = self._vyaw = 0.0

    def shutdown(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:
                logger.warning("Error closing env: %s", exc)
            self.env = None
        logger.info("Isaac Lab env closed")
