"""Integration tests for robot skill and project REST endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _setup_skills(tmp_path, monkeypatch):
    """Create temp skill packages and patch skill_loader."""
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
    }))

    from endpoints.robot import skill_loader
    monkeypatch.setattr(skill_loader, "_SKILLS_DIR", skills_dir)
    skill_loader.load_skills()


class TestListSkills:
    def test_get_robot_skills(self, client):
        c, _ = client
        resp = c.get("/api/robot/skills")
        assert resp.status_code == 200
        skills = resp.json()
        assert isinstance(skills, list)
        names = {s["name"] for s in skills}
        assert "household_objects" in names
        assert "precision_grasp" in names

    def test_skills_have_metadata(self, client):
        c, _ = client
        resp = c.get("/api/robot/skills")
        skills = resp.json()
        slow = next(s for s in skills if s["name"] == "household_objects")
        assert slow["execution_mode"] == "slow"
        assert slow["description"] == "Common household objects"


class TestGetSkillTopic:
    def test_get_topic_content(self, client):
        c, _ = client
        resp = c.get("/api/robot/skills/household_objects/common")
        assert resp.status_code == 200
        data = resp.json()
        assert "Common Objects" in data["content"]
        assert data["skill"] == "household_objects"
        assert data["topic"] == "common"

    def test_missing_skill_404(self, client):
        c, _ = client
        resp = c.get("/api/robot/skills/nonexistent/topic")
        assert resp.status_code == 404

    def test_missing_topic_404(self, client):
        c, _ = client
        resp = c.get("/api/robot/skills/household_objects/nonexistent")
        assert resp.status_code == 404

    def test_fast_path_skill_400(self, client):
        c, _ = client
        resp = c.get("/api/robot/skills/precision_grasp/anything")
        assert resp.status_code == 400


class TestProjectEndpoints:
    def test_project_idle(self, client):
        c, _ = client
        resp = c.get("/api/robot/project")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"

    def test_cancel_no_project(self, client):
        c, _ = client
        resp = c.post("/api/robot/project/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_project_active_state(self, client):
        """When a project is active, GET /api/robot/project returns its state."""
        from endpoints.robot import controller as robot_ctrl
        import time

        robot_ctrl._PROJECT = {
            "project_id": "test-123",
            "started_at": time.time(),
            "status": "active",
            "description": "Find my keys",
            "messages": [],
            "step": 3,
            "max_steps": 20,
            "timeout_sec": 120,
            "loaded_knowledge": ["household_objects/common"],
            "history": [{"step": 1}, {"step": 2}, {"step": 3}],
        }

        c, _ = client
        resp = c.get("/api/robot/project")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"
        assert data["description"] == "Find my keys"
        assert data["step"] == 3

        # Cleanup
        robot_ctrl._PROJECT = None
