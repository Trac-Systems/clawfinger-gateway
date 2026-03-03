"""Unit tests for robot skill package loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from endpoints.robot import skill_loader


@pytest.fixture(autouse=True)
def _skills_dir(tmp_path, monkeypatch):
    """Create a temporary skills directory for testing."""
    skills_dir = tmp_path / "skills" / "robot"
    skills_dir.mkdir(parents=True)

    # Slow-path skill
    slow_dir = skills_dir / "household_objects"
    slow_dir.mkdir()
    (slow_dir / "skill.json").write_text(json.dumps({
        "name": "household_objects",
        "description": "Common household objects",
        "execution_mode": "slow",
        "topics": ["common"],
        "required_capabilities": ["vision", "locomotion"],
    }))
    (slow_dir / "common.md").write_text("# Common Objects\n\nKeys, phone, wallet.")

    # Fast-path skill
    fast_dir = skills_dir / "precision_grasp"
    fast_dir.mkdir()
    (fast_dir / "skill.json").write_text(json.dumps({
        "name": "precision_grasp",
        "description": "Trained RL policy",
        "execution_mode": "fast",
        "status": "coming_soon",
        "required_capabilities": ["manipulation", "vision"],
    }))

    # Invalid skill (malformed JSON)
    bad_dir = skills_dir / "broken"
    bad_dir.mkdir()
    (bad_dir / "skill.json").write_text("{invalid json")

    monkeypatch.setattr(skill_loader, "_SKILLS_DIR", skills_dir)
    skill_loader.load_skills()


class TestLoadSkills:
    def test_loads_both_skills(self):
        skills = skill_loader.list_skills()
        names = {s["name"] for s in skills}
        assert "household_objects" in names
        assert "precision_grasp" in names

    def test_ignores_malformed(self):
        skills = skill_loader.list_skills()
        names = {s["name"] for s in skills}
        assert "broken" not in names

    def test_skill_metadata(self):
        skills = skill_loader.list_skills()
        slow = next(s for s in skills if s["name"] == "household_objects")
        assert slow["execution_mode"] == "slow"
        assert slow["description"] == "Common household objects"
        assert "vision" in slow["required_capabilities"]

    def test_fast_path_status(self):
        skills = skill_loader.list_skills()
        fast = next(s for s in skills if s["name"] == "precision_grasp")
        assert fast["execution_mode"] == "fast"
        assert fast["status"] == "coming_soon"


class TestListSlowSkills:
    def test_only_slow(self):
        slow = skill_loader.list_slow_skills()
        assert len(slow) == 1
        assert slow[0]["name"] == "household_objects"


class TestGetTopic:
    def test_returns_content(self):
        content = skill_loader.get_topic("household_objects", "common")
        assert content is not None
        assert "Common Objects" in content

    def test_missing_skill(self):
        assert skill_loader.get_topic("nonexistent", "common") is None

    def test_missing_topic(self):
        assert skill_loader.get_topic("household_objects", "nonexistent") is None

    def test_fast_path_returns_none(self):
        assert skill_loader.get_topic("precision_grasp", "anything") is None


class TestGetSkill:
    def test_returns_full_metadata(self):
        skill = skill_loader.get_skill("household_objects")
        assert skill is not None
        assert skill["name"] == "household_objects"
        assert "_dir" in skill

    def test_missing_skill(self):
        assert skill_loader.get_skill("nonexistent") is None
