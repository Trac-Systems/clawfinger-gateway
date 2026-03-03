"""Integration tests for robot spatial memory REST endpoints."""

from __future__ import annotations

import base64
from unittest.mock import patch

import pytest


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20
FAKE_JPEG_B64 = base64.b64encode(FAKE_JPEG).decode()


@pytest.fixture(autouse=True)
def memory_setup(tmp_path, client):
    """Initialize spatial memory with test DB for each test."""
    from endpoints.robot import memory as mem

    # Shutdown any existing state
    mem.shutdown()

    # Init with ChromaDB only (no CLIP)
    db_path = str(tmp_path / "test_spatial_memory")
    mem.init_memory_only(db_path)
    yield
    mem.shutdown()


# ---------------------------------------------------------------------------
# Persons
# ---------------------------------------------------------------------------

class TestPersonsAPI:
    def test_list_persons_empty(self, client):
        c, _ = client
        resp = c.get("/api/robot/memory/persons")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_add_person(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/persons", json={
            "name": "Alice",
            "description": "Test person",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Alice"
        assert body["description"] == "Test person"
        assert "id" in body

    def test_add_person_missing_name(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/persons", json={"description": "no name"})
        assert resp.status_code == 400

    def test_get_person(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/persons", json={"name": "Alice"})
        person_id = resp.json()["id"]

        resp = c.get(f"/api/robot/memory/persons/{person_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Alice"

    def test_get_person_not_found(self, client):
        c, _ = client
        resp = c.get("/api/robot/memory/persons/nonexistent")
        assert resp.status_code == 404

    def test_delete_person(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/persons", json={"name": "Alice"})
        person_id = resp.json()["id"]

        resp = c.delete(f"/api/robot/memory/persons/{person_id}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Should be gone
        resp = c.get(f"/api/robot/memory/persons/{person_id}")
        assert resp.status_code == 404

    def test_delete_person_not_found(self, client):
        c, _ = client
        resp = c.delete("/api/robot/memory/persons/nonexistent")
        assert resp.status_code == 404

    def test_list_persons_after_add(self, client):
        c, _ = client
        c.post("/api/robot/memory/persons", json={"name": "Alice"})
        c.post("/api/robot/memory/persons", json={"name": "Bob"})

        resp = c.get("/api/robot/memory/persons")
        assert resp.status_code == 200
        persons = resp.json()
        names = {p["name"] for p in persons}
        assert "Alice" in names
        assert "Bob" in names


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------

class TestObjectsAPI:
    def test_list_objects_empty(self, client):
        c, _ = client
        resp = c.get("/api/robot/memory/objects")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_add_object(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/objects", json={
            "name": "Keys",
            "description": "House keys",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Keys"

    def test_add_object_missing_name(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/objects", json={})
        assert resp.status_code == 400

    def test_get_object(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/objects", json={"name": "Keys"})
        oid = resp.json()["id"]
        resp = c.get(f"/api/robot/memory/objects/{oid}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Keys"

    def test_delete_object(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/objects", json={"name": "Keys"})
        oid = resp.json()["id"]
        resp = c.delete(f"/api/robot/memory/objects/{oid}")
        assert resp.status_code == 200
        resp = c.get(f"/api/robot/memory/objects/{oid}")
        assert resp.status_code == 404

    def test_delete_object_not_found(self, client):
        c, _ = client
        resp = c.delete("/api/robot/memory/objects/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------

class TestRoomsAPI:
    def test_list_rooms_empty(self, client):
        c, _ = client
        resp = c.get("/api/robot/memory/rooms")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_add_room(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/rooms", json={
            "name": "Kitchen",
            "description": "Main kitchen",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Kitchen"

    def test_add_room_missing_name(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/rooms", json={})
        assert resp.status_code == 400

    def test_delete_room(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/rooms", json={"name": "Kitchen"})
        rid = resp.json()["id"]
        resp = c.delete(f"/api/robot/memory/rooms/{rid}")
        assert resp.status_code == 200

    def test_delete_room_not_found(self, client):
        c, _ = client
        resp = c.delete("/api/robot/memory/rooms/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Routines
# ---------------------------------------------------------------------------

class TestRoutinesAPI:
    def test_list_routines_empty(self, client):
        c, _ = client
        resp = c.get("/api/robot/memory/routines")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_add_routine(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/routines", json={
            "name": "Morning check",
            "schedule": "daily 08:00",
            "description": "Check all rooms",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Morning check"
        assert body["schedule"] == "daily 08:00"

    def test_add_routine_missing_name(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/routines", json={
            "schedule": "daily 08:00",
        })
        assert resp.status_code == 400

    def test_delete_routine(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/routines", json={
            "name": "Morning", "schedule": "08:00",
        })
        rid = resp.json()["id"]
        resp = c.delete(f"/api/robot/memory/routines/{rid}")
        assert resp.status_code == 200

    def test_delete_routine_not_found(self, client):
        c, _ = client
        resp = c.delete("/api/robot/memory/routines/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class TestObservationsAPI:
    def test_add_observation_with_embedding(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/observations", json={
            "embedding": [0.1] * 512,
            "metadata": {
                "entity_type": "scene",
                "room": "kitchen",
                "timestamp": 1709500000.0,
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["id"].startswith("obs_")

    def test_add_observation_missing_both(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/observations", json={
            "metadata": {"room": "kitchen"},
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQueryAPI:
    def _seed(self, c):
        """Add observations for query tests."""
        for i in range(3):
            c.post("/api/robot/memory/observations", json={
                "embedding": [float(i) / 10] * 512,
                "metadata": {
                    "entity_type": "scene",
                    "room": "kitchen",
                    "description": f"Kitchen scene {i}",
                    "world_x": float(i),
                    "world_y": float(i),
                    "world_z": 0.0,
                    "timestamp": 1709500000.0 + i * 100,
                },
            })
        # Person observation
        c.post("/api/robot/memory/observations", json={
            "embedding": [0.5] * 512,
            "metadata": {
                "entity_type": "person",
                "entity_id": "alice123",
                "entity_name": "Alice",
                "room": "kitchen",
                "timestamp": 1709500500.0,
            },
        })

    def test_query_person_sightings(self, client):
        c, _ = client
        self._seed(c)
        resp = c.post("/api/robot/memory/query", json={
            "type": "person_sightings",
            "person_id": "alice123",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["results"][0]["entity_name"] == "Alice"

    def test_query_room_activity(self, client):
        c, _ = client
        self._seed(c)
        resp = c.post("/api/robot/memory/query", json={
            "type": "room_activity",
            "room": "kitchen",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 4  # 3 scenes + 1 person

    def test_query_nearby(self, client):
        c, _ = client
        self._seed(c)
        resp = c.post("/api/robot/memory/query", json={
            "type": "nearby",
            "x": 0.5,
            "y": 0.5,
            "z": 0.0,
            "radius": 1.5,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1

    def test_query_unknown_type(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/query", json={
            "type": "unknown_type",
        })
        assert resp.status_code == 400

    def test_query_text_missing(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/query", json={
            "type": "text",
        })
        assert resp.status_code == 400

    def test_query_person_sightings_missing_id(self, client):
        c, _ = client
        resp = c.post("/api/robot/memory/query", json={
            "type": "person_sightings",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStatsAPI:
    def test_stats_empty(self, client):
        c, _ = client
        resp = c.get("/api/robot/memory/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["initialized"] is True
        assert body["persons"] == 0
        assert body["observations"] == 0

    def test_stats_after_adds(self, client):
        c, _ = client
        c.post("/api/robot/memory/persons", json={"name": "Alice"})
        c.post("/api/robot/memory/objects", json={"name": "Keys"})
        c.post("/api/robot/memory/rooms", json={"name": "Kitchen"})
        c.post("/api/robot/memory/routines", json={
            "name": "Morning", "schedule": "08:00",
        })
        resp = c.get("/api/robot/memory/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["persons"] >= 1
        assert body["objects"] >= 1
        assert body["rooms"] == 1
        assert body["routines"] == 1


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_unauthenticated_request(self, client):
        c, _ = client
        from starlette.testclient import TestClient
        import app as app_module
        with TestClient(app_module.app) as raw_client:
            resp = raw_client.get("/api/robot/memory/stats")
            assert resp.status_code == 401
