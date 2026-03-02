"""Unitree G1 model registration — capabilities and defaults."""

from __future__ import annotations

from endpoints.robot import register_model
from endpoints.robot.unitree_g1.commands import handle_command
from endpoints.robot.unitree_g1.defaults import G1_DEFAULTS

# Register G1 with its full capability set
register_model(
    name="unitree_g1",
    capabilities=[
        "locomotion",
        "posture",
        "manipulation",
        "gesture",
        "vision",
        "audio",
        "dexterous_hands",
    ],
    command_handler=handle_command,
    defaults=G1_DEFAULTS,
)
