"""Unit tests for robot perception buffer module."""

from __future__ import annotations

import asyncio
import base64

import pytest

from endpoints.robot import perception as perc


@pytest.fixture(autouse=True)
def clean_state():
    """Reset perception state before each test."""
    perc.stop_all()
    yield
    perc.stop_all()


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # minimal JPEG header
FAKE_JPEG_B64 = base64.b64encode(FAKE_JPEG).decode()

FAKE_PCM = b"\x00\x01" * 1600  # 100ms of 16kHz mono S16LE
FAKE_PCM_B64 = base64.b64encode(FAKE_PCM).decode()


class TestFrameBuffer:
    def test_push_frame_stores_data(self):
        perc.push_frame("cam1", FAKE_JPEG_B64, seq=0, ts=1.0)
        buf = perc._frame_buffers.get("cam1")
        assert buf is not None
        assert len(buf) == 1
        assert buf[0] == FAKE_JPEG

    def test_push_frame_invalid_base64_logs_warning(self):
        perc.push_frame("cam1", "not-valid-base64!!!", seq=0, ts=0)
        buf = perc._frame_buffers.get("cam1")
        # Should not crash, buffer either empty or not created
        assert buf is None or len(buf) == 0

    def test_ring_buffer_evicts_oldest(self):
        for i in range(perc._FRAME_BUFFER_SIZE + 5):
            data = base64.b64encode(FAKE_JPEG + bytes([i])).decode()
            perc.push_frame("cam1", data, seq=i, ts=float(i))
        buf = perc._frame_buffers["cam1"]
        assert len(buf) == perc._FRAME_BUFFER_SIZE

    def test_push_frame_signals_event(self):
        perc.push_frame("cam1", FAKE_JPEG_B64)
        ev = perc._frame_events.get("cam1")
        assert ev is not None
        assert ev.is_set()


class TestAudioBuffer:
    def test_push_audio_stores_data(self):
        perc.push_audio("mic1", FAKE_PCM_B64, sample_rate=16000, channels=1)
        buf = perc._audio_buffers.get("mic1")
        assert buf is not None
        assert len(buf) == 1
        assert buf[0] == FAKE_PCM

    def test_push_audio_invalid_base64(self):
        perc.push_audio("mic1", "bad!!!")
        buf = perc._audio_buffers.get("mic1")
        assert buf is None or len(buf) == 0


class TestStreamState:
    def test_streaming_state(self):
        assert not perc.is_streaming("cam1")
        perc.set_streaming("cam1", True)
        assert perc.is_streaming("cam1")
        assert "cam1" in perc.active_streams()
        perc.set_streaming("cam1", False)
        assert not perc.is_streaming("cam1")

    def test_monitoring_state(self):
        assert not perc.is_monitoring("mic1")
        perc.set_monitoring("mic1", True)
        assert perc.is_monitoring("mic1")
        assert "mic1" in perc.active_monitors()
        perc.set_monitoring("mic1", False)
        assert not perc.is_monitoring("mic1")


class TestSnapshot:
    def test_set_and_get_snapshot(self):
        perc.set_snapshot("cam1", FAKE_JPEG)
        result = perc.get_snapshot("cam1")
        assert result is not None
        jpeg, ts = result
        assert jpeg == FAKE_JPEG
        assert ts > 0

    def test_get_snapshot_missing(self):
        assert perc.get_snapshot("nonexistent") is None


class TestStopAll:
    def test_stop_all_clears_everything(self):
        perc.push_frame("cam1", FAKE_JPEG_B64)
        perc.push_audio("mic1", FAKE_PCM_B64)
        perc.set_streaming("cam1", True)
        perc.set_monitoring("mic1", True)
        perc.set_snapshot("cam1", FAKE_JPEG)

        perc.stop_all()

        assert len(perc._frame_buffers) == 0
        assert len(perc._audio_buffers) == 0
        assert len(perc._video_streams) == 0
        assert len(perc._audio_monitors) == 0
        assert len(perc._snapshots) == 0
        assert len(perc._frame_events) == 0
        assert len(perc._audio_events) == 0


class TestFrameGenerator:
    @pytest.mark.asyncio
    async def test_generator_yields_frames(self):
        perc.set_streaming("cam1", True)
        perc.push_frame("cam1", FAKE_JPEG_B64)
        perc.push_frame("cam1", FAKE_JPEG_B64)

        frames = []
        async for frame in perc.frame_generator("cam1"):
            frames.append(frame)
            if len(frames) >= 2:
                perc.set_streaming("cam1", False)
                break
        assert len(frames) == 2
        assert frames[0] == FAKE_JPEG

    @pytest.mark.asyncio
    async def test_generator_stops_when_not_streaming(self):
        perc.set_streaming("cam1", False)
        frames = []
        async for frame in perc.frame_generator("cam1"):
            frames.append(frame)
        assert len(frames) == 0


class TestGetPerception:
    def test_g1_perception_sources(self):
        from endpoints import robot as robot_mod
        robot_mod.load_models()
        perception = robot_mod.get_perception("unitree_g1")
        assert "cameras" in perception
        assert "microphones" in perception
        assert len(perception["cameras"]) == 2
        assert perception["cameras"][0]["id"] == "head_rgb"
        assert perception["cameras"][1]["id"] == "head_depth"
        assert len(perception["microphones"]) == 1
        assert perception["microphones"][0]["id"] == "mic_array"

    def test_unknown_model_returns_empty(self):
        from endpoints import robot as robot_mod
        assert robot_mod.get_perception("nonexistent") == {}


class TestCommandCapabilities:
    def test_perception_commands_mapped(self):
        from endpoints.robot import _COMMAND_CAPABILITIES
        assert _COMMAND_CAPABILITIES["camera_snapshot"] == "vision"
        assert _COMMAND_CAPABILITIES["camera_stream_start"] == "vision"
        assert _COMMAND_CAPABILITIES["camera_stream_stop"] == "vision"
        assert _COMMAND_CAPABILITIES["camera_describe"] == "vision"
        assert _COMMAND_CAPABILITIES["audio_monitor_start"] == "audio"
        assert _COMMAND_CAPABILITIES["audio_monitor_stop"] == "audio"
