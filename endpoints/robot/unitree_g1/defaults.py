"""Unitree G1 default configuration values."""

from __future__ import annotations

G1_DEFAULTS = {
    "jetson_ip": "192.168.123.164",
    "locomotion_ip": "192.168.123.161",
    "dds_domain": 0,
    "max_speed": 0.5,
    "enable_hands": True,
    "enable_low_level": False,
    "wifi_networks": [],
    "perception": {
        "cameras": [
            {"id": "head_rgb", "type": "rgb", "description": "RealSense D435 color", "default_resolution": [640, 480]},
            {"id": "head_depth", "type": "depth", "description": "RealSense D435 depth (colorized)", "default_resolution": [640, 480]},
        ],
        "microphones": [
            {"id": "mic_array", "type": "array", "channels": 4, "sample_rate": 16000, "description": "4-mic array"},
        ],
    },
}
