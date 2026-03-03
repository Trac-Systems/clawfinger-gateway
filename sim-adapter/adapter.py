#!/usr/bin/env python3
"""Clawfinger Sim Adapter — bridges simulation ↔ Gateway WebSocket.

Pretends to be a Jetson: connects to the gateway's /ws/robot endpoint,
receives robot_command messages, translates them to sim actions, and
streams back camera frames + joint state.

Supports two backends:
  --backend mock    Use MockSim (no GPU, for testing the WS pipeline)
  --backend isaac   Use Isaac Lab with G1 (requires isaacsim conda env)

Usage (mock — works anywhere, no GPU):
    python adapter.py --gateway ws://127.0.0.1:18996/ws/robot --token localdev

Usage (Isaac Sim — RTX machine):
    conda activate g1sim
    python adapter.py --gateway ws://127.0.0.1:18996/ws/robot --token localdev \\
        --backend isaac --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint --headless
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import signal
import struct
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[sim-adapter] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sim_adapter")


# ---------------------------------------------------------------------------
# Abstract sim backend
# ---------------------------------------------------------------------------

class SimBackend:
    """Abstract interface for a simulation backend."""

    def init(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def capture_camera(self, name: str = "head_rgb") -> bytes | None:
        """Return JPEG bytes or None."""
        return None

    def get_joint_positions(self) -> list[float]:
        return []

    def get_joint_velocities(self) -> list[float]:
        return []

    def get_imu(self) -> dict:
        return {"quaternion": [1, 0, 0, 0], "gyroscope": [0, 0, 0]}

    def get_world_pose(self) -> dict:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0}

    def apply_walk(self, vx: float, vy: float, vyaw: float) -> None:
        pass

    def apply_stand(self) -> None:
        pass

    def apply_sit(self) -> None:
        pass

    def apply_stop(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock sim — no Isaac Sim needed, generates fake data
# ---------------------------------------------------------------------------

class MockSim(SimBackend):
    """Lightweight mock simulation for testing the adapter pipeline."""

    def __init__(self):
        self._step_count = 0
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._theta = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self._walking = False
        self._joint_pos = [0.0] * 29
        self._t0 = time.time()

    def init(self) -> None:
        logger.info("MockSim initialized (29-DOF G1 placeholder)")

    def step(self) -> None:
        self._step_count += 1
        dt = 0.02  # 50 Hz
        if self._walking:
            import math
            self._pos_x += self._vx * math.cos(self._theta) * dt
            self._pos_y += self._vx * math.sin(self._theta) * dt
            self._theta += self._vyaw * dt
            # Simulate slight joint oscillation while walking
            t = time.time() - self._t0
            for i in range(29):
                self._joint_pos[i] = 0.05 * __import__("math").sin(t * 3 + i * 0.3)

    def capture_camera(self, name: str = "head_rgb") -> bytes | None:
        """Generate a simple colored test frame."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            # Fallback: generate a minimal valid JPEG
            return self._tiny_jpeg()

        img = Image.new("RGB", (640, 480), color=(40, 60, 80))
        draw = ImageDraw.Draw(img)

        # Draw some info
        t = time.time() - self._t0
        lines = [
            f"Clawfinger Sim (Mock)",
            f"Camera: {name}",
            f"Step: {self._step_count}",
            f"Pos: ({self._pos_x:.2f}, {self._pos_y:.2f})",
            f"Theta: {self._theta:.2f} rad",
            f"Walking: {self._walking}",
            f"Time: {t:.1f}s",
        ]
        y = 20
        for line in lines:
            draw.text((20, y), line, fill=(200, 220, 255))
            y += 25

        # Draw a moving dot to show the sim is alive
        import math
        cx = 320 + int(100 * math.cos(t * 2))
        cy = 300 + int(100 * math.sin(t * 2))
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(0, 255, 100))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return buf.getvalue()

    def _tiny_jpeg(self) -> bytes:
        # Minimal 1x1 JPEG for when Pillow isn't available
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
            b"\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07"
            b"\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13"
            b"\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ,#\x1c\x1c28teleimager3@teleimager9=82<.342\xff\xc0\x00\x0b\x08\x00\x01"
            b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
            b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08"
            b"\x01\x01\x00\x00?\x00T\xdb\x9e\xa7(\xff\xd9"
        )

    def get_joint_positions(self) -> list[float]:
        return self._joint_pos[:]

    def get_joint_velocities(self) -> list[float]:
        return [0.0] * 29

    def get_imu(self) -> dict:
        return {"quaternion": [1, 0, 0, 0], "gyroscope": [0, 0, 0]}

    def get_world_pose(self) -> dict:
        return {
            "x": self._pos_x,
            "y": self._pos_y,
            "z": 0.0,
            "theta": self._theta,
        }

    def apply_walk(self, vx: float, vy: float, vyaw: float) -> None:
        self._walking = True
        self._vx = vx
        self._vy = vy
        self._vyaw = vyaw

    def apply_stand(self) -> None:
        self._walking = False
        self._vx = 0
        self._vy = 0
        self._vyaw = 0

    def apply_sit(self) -> None:
        self._walking = False

    def apply_stop(self) -> None:
        self._walking = False
        self._vx = 0
        self._vy = 0
        self._vyaw = 0

    def shutdown(self) -> None:
        logger.info("MockSim shutdown")


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

class CommandHandler:
    """Translates gateway robot_command messages to sim backend calls."""

    def __init__(self, sim: SimBackend):
        self.sim = sim

    async def handle(self, msg: dict) -> dict:
        command = msg.get("command", "")
        params = msg.get("params", {})
        req_id = msg.get("_req_id")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._dispatch, command, params)
        if req_id is not None:
            result["_resp_id"] = req_id
        return result

    def _dispatch(self, command: str, params: dict) -> dict:
        if command == "stand":
            self.sim.apply_stand()
            return {"ok": True, "detail": "standing (sim)"}

        elif command == "sit":
            self.sim.apply_sit()
            return {"ok": True, "detail": "sitting (sim)"}

        elif command == "walk":
            vx = params.get("vx", 0.3)
            vy = params.get("vy", 0.0)
            vyaw = params.get("vyaw", 0.0)
            self.sim.apply_walk(vx, vy, vyaw)
            return {"ok": True, "detail": f"walking vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}"}

        elif command == "turn":
            vyaw = params.get("vyaw", 0.5)
            self.sim.apply_walk(0, 0, vyaw)
            return {"ok": True, "detail": f"turning vyaw={vyaw:.2f}"}

        elif command in ("stop", "safety_stop"):
            self.sim.apply_stop()
            return {"ok": True, "detail": "stopped (sim)"}

        elif command == "camera_snapshot":
            source = params.get("source", "head_rgb")
            frame = self.sim.capture_camera(source)
            if frame:
                return {
                    "ok": True,
                    "image_base64": base64.b64encode(frame).decode(),
                    "width": params.get("width", 640),
                    "height": params.get("height", 480),
                    "source": source,
                }
            return {"ok": False, "error": "no camera data (sim)"}

        elif command == "camera_stream_start":
            return {"ok": True, "detail": "stream active (sim)"}

        elif command == "camera_stream_stop":
            return {"ok": True, "detail": "stream stopped (sim)"}

        elif command in ("wave", "nod", "thumbs_up", "shake_head", "point"):
            return {"ok": True, "detail": f"{command} gesture (sim)"}

        elif command in ("pick_up", "place", "hand_over", "reach", "grasp", "release", "pour"):
            return {"ok": True, "detail": f"{command} (sim placeholder)"}

        elif command in ("speak", "listen", "audio_monitor_start", "audio_monitor_stop"):
            return {"ok": True, "detail": f"{command} (sim, no audio)"}

        elif command in ("snapshot", "detect_object", "look", "camera_describe"):
            return {"ok": True, "detail": f"{command} (sim placeholder)"}

        else:
            logger.warning("Unknown command: %s", command)
            return {"ok": True, "detail": f"acked '{command}' (sim)"}


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

class GatewayClient:
    """Connects to the gateway's /ws/robot endpoint."""

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self._ws = None
        self._connected = False
        self._on_command = None
        self._heartbeat_task: asyncio.Task | None = None

    def on_command(self, handler):
        self._on_command = handler

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect_forever(self):
        """Connect with auto-reconnect."""
        try:
            import websockets
        except ImportError:
            logger.error("Install websockets: pip install websockets")
            return

        delay = 1.0
        while True:
            try:
                headers = {}
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"

                logger.info("Connecting to %s ...", self.url)
                # websockets 12+ renamed additional_headers to extra_headers
                ws_kwargs = dict(
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=2**22,  # 4 MB for camera frames
                )
                try:
                    self._ws = await websockets.connect(
                        self.url,
                        extra_headers=headers,
                        **ws_kwargs,
                    )
                except TypeError:
                    # Fallback for websockets <12
                    self._ws = await websockets.connect(
                        self.url,
                        additional_headers=headers,
                        **ws_kwargs,
                    )
                self._connected = True
                logger.info("Connected to gateway")
                delay = 1.0

                if self._heartbeat_task is None or self._heartbeat_task.done():
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                await self._recv_loop()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("Connection error: %s", exc)

            self._connected = False
            logger.info("Reconnecting in %.0fs...", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)

    async def send(self, msg: dict):
        if self._ws and self._connected:
            try:
                await self._ws.send(json.dumps(msg))
            except Exception:
                self._connected = False

    async def _recv_loop(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                await self._handle(msg)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Recv error: %s", exc)

    async def _handle(self, msg: dict):
        msg_type = msg.get("type", "")

        if msg_type == "heartbeat":
            await self.send({"type": "heartbeat_ack", "ts": time.time()})
            return

        if msg_type == "robot_command" and self._on_command:
            result = await self._on_command(msg)
            if result:
                await self.send(result)
            return

        # Gateway sends some commands directly (not wrapped in robot_command)
        # e.g. camera_snapshot, camera_stream_start, audio_monitor_start
        if msg_type in (
            "camera_snapshot", "camera_stream_start", "camera_stream_stop",
            "camera_describe", "audio_monitor_start", "audio_monitor_stop",
        ) and self._on_command:
            # Wrap as robot_command for the handler
            wrapped = {"command": msg_type, "params": msg, "_req_id": msg.get("_req_id")}
            result = await self._on_command(wrapped)
            if result:
                await self.send(result)
            return

        if msg_type == "tts_speak":
            text = msg.get("text", "")
            logger.info("TTS: %s", text[:100])
            return

    async def _heartbeat_loop(self):
        """Send heartbeat_ack proactively to keep connection alive."""
        while True:
            await asyncio.sleep(3)
            if self._connected:
                await self.send({"type": "heartbeat_ack", "ts": time.time()})

    async def close(self):
        self._connected = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._ws:
            await self._ws.close()


# ---------------------------------------------------------------------------
# Streaming loops
# ---------------------------------------------------------------------------

async def camera_stream_loop(sim: SimBackend, gw: GatewayClient, fps: float):
    """Stream camera frames to gateway.

    The gateway registers cameras as ``head_rgb`` etc., so the *source* field
    in outgoing messages must use the gateway's camera IDs.  Internally,
    ``sim.capture_camera()`` maps to the sim's native sensor names.
    """
    interval = 1.0 / max(fps, 0.1)
    seq = 0
    loop = asyncio.get_running_loop()
    gateway_source = "head_rgb"
    while True:
        await asyncio.sleep(interval)
        if not gw.connected:
            continue
        try:
            frame = await loop.run_in_executor(None, sim.capture_camera, gateway_source)
        except Exception as exc:
            print(f"[adapter] camera capture error: {exc}", flush=True)
            continue
        if frame:
            seq += 1
            await gw.send({
                "type": "camera_frame",
                "source": gateway_source,
                "image_base64": base64.b64encode(frame).decode(),
                "seq": seq,
                "ts": time.time(),
            })
            if seq == 1:
                print(f"[adapter] First camera_frame sent ({len(frame)} bytes)", flush=True)
            elif seq % 100 == 0:
                print(f"[adapter] Camera frames sent: {seq}", flush=True)


async def state_report_loop(sim: SimBackend, gw: GatewayClient, hz: float = 2.0):
    """Report joint state / pose periodically."""
    interval = 1.0 / hz
    loop = asyncio.get_running_loop()
    count = 0
    while True:
        await asyncio.sleep(interval)
        if not gw.connected:
            continue
        try:
            pose = await loop.run_in_executor(None, sim.get_world_pose)
            joints = await loop.run_in_executor(None, sim.get_joint_positions)
            vels = await loop.run_in_executor(None, sim.get_joint_velocities)
            imu = await loop.run_in_executor(None, sim.get_imu)
        except Exception as exc:
            print(f"[adapter] state report error: {exc}", flush=True)
            continue
        count += 1
        await gw.send({
            "type": "robot_event",
            "event": "state",
            "joint_positions": joints,
            "joint_velocities": vels,
            "imu": imu,
            "world_pose": pose,
            "timestamp": time.time(),
        })
        if count == 1:
            print(f"[adapter] First state report (pose={pose})", flush=True)


def sim_step_main_loop(sim: SimBackend, hz: float = 50.0):
    """Step the simulation in the MAIN thread at a fixed rate.

    CRITICAL: Kit's physics engine and async engine are both bound to the
    main thread.  sim.step() → env.step() → Kit sim.step() tries to run
    Kit's own asyncio loop, which crashes if called inside Python's asyncio
    event loop, and deadlocks if called from a thread pool thread.

    Solution: run sim stepping in the main thread (where Kit was initialized),
    and run Python asyncio in a background thread for WS / camera / state.
    """
    interval = 1.0 / hz
    step_count = 0
    print(f"[adapter] Sim step loop starting (main thread, {hz} Hz)", flush=True)
    while True:
        t0 = time.time()
        try:
            sim.step()
        except Exception as exc:
            print(f"[adapter] sim step error: {exc}", flush=True)
            time.sleep(1.0)
            continue
        step_count += 1
        if step_count == 1:
            print(f"[adapter] First sim step completed ({time.time()-t0:.2f}s)", flush=True)
        elif step_count == 100:
            print(f"[adapter] 100 sim steps done", flush=True)
        elif step_count % 1000 == 0:
            print(f"[adapter] {step_count} sim steps", flush=True)
        elapsed = time.time() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Clawfinger Sim Adapter")
    p.add_argument("--gateway", default="ws://127.0.0.1:18996/ws/robot",
                    help="Gateway WS URL")
    p.add_argument("--token", default="localdev", help="Bearer token")
    p.add_argument("--backend", choices=["mock", "isaac"], default="mock",
                    help="Sim backend")
    p.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex1-Joint",
                    help="Isaac Lab task (backend=isaac only)")
    p.add_argument("--headless", action="store_true", help="No GUI (isaac only)")
    p.add_argument("--camera-fps", type=float, default=5.0, help="Camera FPS to gateway")
    p.add_argument("--state-hz", type=float, default=2.0, help="State report rate")
    p.add_argument("--sim-hz", type=float, default=50.0, help="Sim step rate")
    p.add_argument("--device", default="cuda:0", help="Sim device (isaac only)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


async def run_async(args, sim: SimBackend):
    """Run async tasks (WS, camera, state) in a background thread's event loop.

    sim.step() is NOT called here — it runs in the main thread via
    sim_step_main_loop() because Kit requires the main thread.
    """
    cmd_handler = CommandHandler(sim)
    gw = GatewayClient(args.gateway, args.token)
    gw.on_command(cmd_handler.handle)

    print(f"[adapter] Async tasks starting (backend={args.backend})", flush=True)

    tasks = [
        asyncio.create_task(gw.connect_forever()),
        asyncio.create_task(camera_stream_loop(sim, gw, fps=args.camera_fps)),
        asyncio.create_task(state_report_loop(sim, gw, hz=args.state_hz)),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        print("[adapter] Async shutdown...", flush=True)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await gw.close()
        print("[adapter] Async done", flush=True)


def _create_isaac_backend(args):
    """Initialize Isaac Sim AppLauncher, then create the backend.

    The initialization order is critical:
      1. Set env vars (EULA, PROJECT_ROOT, Vulkan ICD)
      2. import isaacsim (bootstraps Kit kernel)
      3. Create AppLauncher (initializes the simulation app)
      4. Add unitree_sim_isaaclab to sys.path
      5. THEN import isaac_backend (which imports isaaclab_tasks etc.)

    NOTE: On headless Linux without a display, run via xvfb-run:
        xvfb-run -a python adapter.py --backend isaac --headless ...
    """
    import os
    import sys

    # 1) Set required environment variables
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "Y"

    # Force NVIDIA-only Vulkan ICD (avoids AMD/Intel ICDs failing)
    nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if os.path.isfile(nvidia_icd):
        os.environ.setdefault("VK_ICD_FILENAMES", nvidia_icd)

    # Set PROJECT_ROOT for unitree asset path resolution
    unitree_path = os.environ.get(
        "UNITREE_SIM_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "unitree_sim_isaaclab"),
    )
    unitree_path = os.path.abspath(unitree_path)
    os.environ.setdefault("PROJECT_ROOT", unitree_path)

    # 2) Bootstrap Isaac Sim kernel
    logger.info("Bootstrapping Isaac Sim...")
    import isaacsim  # noqa: F401

    # 3) Create AppLauncher with a minimal argparse namespace
    from isaaclab.app import AppLauncher

    launcher_argv = []
    if args.headless:
        launcher_argv.append("--headless")
    launcher_argv.append("--enable_cameras")

    import argparse as _argparse
    launcher_parser = _argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args = launcher_parser.parse_args(launcher_argv)
    launcher_args.device = args.device

    logger.info("Starting AppLauncher (headless=%s, device=%s)...", args.headless, args.device)
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app
    logger.info("AppLauncher ready")

    # 4) Add unitree_sim_isaaclab to path so `import tasks` works
    if os.path.isdir(unitree_path) and unitree_path not in sys.path:
        sys.path.insert(0, unitree_path)
        logger.info("Added to path: %s", unitree_path)

    # 5) Now it's safe to import isaac_backend
    from isaac_backend import IsaacSimBackend

    return IsaacSimBackend(
        task=args.task, device=args.device,
        headless=args.headless, seed=args.seed,
    )


def main():
    args = parse_args()
    print("[adapter] Clawfinger Sim Adapter", flush=True)
    print(f"[adapter]   Gateway:  {args.gateway}", flush=True)
    print(f"[adapter]   Backend:  {args.backend}", flush=True)

    # Create and init backend BEFORE starting any event loops.
    # Critical for Isaac: AppLauncher + env creation must happen in the
    # main thread, outside any asyncio event loop.
    if args.backend == "mock":
        sim = MockSim()
    elif args.backend == "isaac":
        print(f"[adapter]   Task:     {args.task}", flush=True)
        print(f"[adapter]   Headless: {args.headless}", flush=True)
        sim = _create_isaac_backend(args)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    sim.init()
    print("[adapter] Backend initialized", flush=True)

    if args.backend == "isaac":
        # Isaac backend: main thread runs sim stepping (Kit requires it),
        # asyncio runs in a background thread for WS/camera/state.
        import threading

        loop = asyncio.new_event_loop()

        def run_asyncio():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_async(args, sim))

        async_thread = threading.Thread(target=run_asyncio, daemon=True)
        async_thread.start()

        try:
            sim_step_main_loop(sim, hz=args.sim_hz)
        except KeyboardInterrupt:
            print("[adapter] Interrupted", flush=True)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            async_thread.join(timeout=5)
            sim.shutdown()
            print("[adapter] Done", flush=True)
    else:
        # Mock backend: everything runs in asyncio (no Kit constraints)
        asyncio.run(run_async(args, sim))


if __name__ == "__main__":
    main()
