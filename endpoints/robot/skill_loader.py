"""Robot skill package loader — scans skills/robot/ for skill.json manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "robot"
_SKILLS: dict[str, dict[str, Any]] = {}


def load_skills() -> None:
    """Scan skills/robot/ and parse all skill.json files."""
    _SKILLS.clear()
    if not _SKILLS_DIR.is_dir():
        return
    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        manifest = skill_dir / "skill.json"
        if not manifest.is_file():
            continue
        try:
            with manifest.open() as f:
                meta = json.load(f)
            meta["_dir"] = str(skill_dir)
            _SKILLS[meta["name"]] = meta
        except (json.JSONDecodeError, KeyError):
            continue


def list_skills() -> list[dict[str, Any]]:
    """Return all skills with name, description, execution_mode."""
    return [
        {
            "name": s["name"],
            "description": s.get("description", ""),
            "execution_mode": s.get("execution_mode", "slow"),
            "status": s.get("status", "available"),
            "required_capabilities": s.get("required_capabilities", []),
        }
        for s in _SKILLS.values()
    ]


def list_slow_skills() -> list[dict[str, Any]]:
    """Return only slow-path skills (for LLM system prompt)."""
    return [s for s in list_skills() if s["execution_mode"] == "slow"]


def get_skill(name: str) -> dict[str, Any] | None:
    """Return full skill metadata by name."""
    return _SKILLS.get(name)


def get_topic(skill_name: str, topic: str) -> str | None:
    """Return topic .md content for a slow-path skill."""
    skill = _SKILLS.get(skill_name)
    if not skill:
        return None
    if skill.get("execution_mode") == "fast":
        return None
    skill_dir = Path(skill["_dir"])
    topic_file = skill_dir / f"{topic}.md"
    if not topic_file.is_file():
        return None
    return topic_file.read_text()
