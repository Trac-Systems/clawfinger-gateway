"""Unit tests for spatial memory module (ChromaDB + mocked CLIP)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from endpoints.robot import memory as mem


# ---------------------------------------------------------------------------
# Deterministic CLIP mock — returns seeded 512-dim embeddings
# ---------------------------------------------------------------------------

_CALL_COUNT = 0


def _fake_embed_image(image_bytes: bytes) -> list[float]:
    """Return a deterministic 512-dim embedding based on image content hash."""
    global _CALL_COUNT
    _CALL_COUNT += 1
    seed = hash(image_bytes) % 10000
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(512)]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


def _fake_embed_text(text: str) -> list[float]:
    """Return a deterministic 512-dim embedding based on text hash."""
    seed = hash(text) % 10000
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(512)]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def clean_memory(tmp_path):
    """Initialize fresh memory with mocked CLIP for each test."""
    global _CALL_COUNT
    _CALL_COUNT = 0

    # Reset module state
    mem.shutdown()

    # Init with ChromaDB but no CLIP
    db_path = str(tmp_path / "test_memory")
    mem.init_memory_only(db_path)

    # Patch CLIP functions
    with patch.object(mem, "_embed_image", side_effect=_fake_embed_image), \
         patch.object(mem, "clip_available", return_value=True):
        yield

    mem.shutdown()


# ---------------------------------------------------------------------------
# Person CRUD
# ---------------------------------------------------------------------------

class TestPersonCRUD:
    def test_add_person_no_images(self):
        result = mem.add_person("Alice", "Test person")
        assert result["name"] == "Alice"
        assert result["description"] == "Test person"
        assert "id" in result
        assert result["image_count"] == 0

    def test_add_person_with_images(self):
        images = [b"\xff\xd8\xff\xe0" + b"\x00" * 20,
                  b"\xff\xd8\xff\xe0" + b"\x01" * 20]
        result = mem.add_person("Bob", "Another person", reference_images=images)
        assert result["name"] == "Bob"
        assert result["image_count"] == 2

    def test_list_persons(self):
        mem.add_person("Alice")
        mem.add_person("Bob")
        persons = mem.list_persons()
        names = {p["name"] for p in persons}
        assert "Alice" in names
        assert "Bob" in names
        assert len(persons) == 2

    def test_get_person(self):
        result = mem.add_person("Alice", "desc")
        person = mem.get_person(result["id"])
        assert person is not None
        assert person["name"] == "Alice"
        assert person["description"] == "desc"

    def test_get_person_not_found(self):
        assert mem.get_person("nonexistent") is None

    def test_delete_person(self):
        result = mem.add_person("Alice")
        assert mem.delete_person(result["id"]) is True
        assert mem.get_person(result["id"]) is None

    def test_delete_person_not_found(self):
        assert mem.delete_person("nonexistent") is False

    def test_add_person_reference(self):
        result = mem.add_person("Alice", reference_images=[b"\xff" * 20])
        assert result["image_count"] == 1

        ref = mem.add_person_reference(result["id"], b"\xfe" * 20)
        assert ref is not None
        assert ref["ref_index"] == 1

        person = mem.get_person(result["id"])
        assert person["image_count"] == 2


# ---------------------------------------------------------------------------
# Object CRUD
# ---------------------------------------------------------------------------

class TestObjectCRUD:
    def test_add_object_no_images(self):
        result = mem.add_object("Keys", "My house keys")
        assert result["name"] == "Keys"
        assert result["description"] == "My house keys"
        assert result["image_count"] == 0

    def test_add_object_with_images(self):
        images = [b"\xff\xd8" + b"\x00" * 20]
        result = mem.add_object("Keys", reference_images=images)
        assert result["image_count"] == 1

    def test_list_objects(self):
        mem.add_object("Keys")
        mem.add_object("Wallet")
        objects = mem.list_objects()
        names = {o["name"] for o in objects}
        assert "Keys" in names
        assert "Wallet" in names

    def test_get_object(self):
        result = mem.add_object("Keys", "My keys")
        obj = mem.get_object(result["id"])
        assert obj is not None
        assert obj["name"] == "Keys"

    def test_delete_object(self):
        result = mem.add_object("Keys")
        assert mem.delete_object(result["id"]) is True
        assert mem.get_object(result["id"]) is None

    def test_add_object_reference(self):
        result = mem.add_object("Keys")
        # No initial images, so image_count = 0
        ref = mem.add_object_reference(result["id"], b"\xfe" * 20)
        assert ref is not None


# ---------------------------------------------------------------------------
# Room CRUD
# ---------------------------------------------------------------------------

class TestRoomCRUD:
    def test_add_room(self):
        result = mem.add_room("Kitchen", "The main kitchen")
        assert result["name"] == "Kitchen"
        assert result["description"] == "The main kitchen"
        assert "id" in result

    def test_list_rooms(self):
        mem.add_room("Kitchen")
        mem.add_room("Living Room")
        rooms = mem.list_rooms()
        names = {r["name"] for r in rooms}
        assert "Kitchen" in names
        assert "Living Room" in names

    def test_get_room(self):
        result = mem.add_room("Kitchen", "desc")
        room = mem.get_room(result["id"])
        assert room is not None
        assert room["name"] == "Kitchen"

    def test_delete_room(self):
        result = mem.add_room("Kitchen")
        assert mem.delete_room(result["id"]) is True
        assert mem.get_room(result["id"]) is None

    def test_delete_room_not_found(self):
        assert mem.delete_room("nonexistent") is False


# ---------------------------------------------------------------------------
# Routine CRUD
# ---------------------------------------------------------------------------

class TestRoutineCRUD:
    def test_add_routine(self):
        result = mem.add_routine("Morning check", "daily 08:00", "Check all rooms")
        assert result["name"] == "Morning check"
        assert result["schedule"] == "daily 08:00"
        assert result["description"] == "Check all rooms"

    def test_list_routines(self):
        mem.add_routine("Morning", "08:00", "morning tasks")
        mem.add_routine("Evening", "20:00", "evening tasks")
        routines = mem.list_routines()
        names = {r["name"] for r in routines}
        assert "Morning" in names
        assert "Evening" in names

    def test_delete_routine(self):
        result = mem.add_routine("Morning", "08:00")
        assert mem.delete_routine(result["id"]) is True
        routines = mem.list_routines()
        assert len(routines) == 0

    def test_delete_routine_not_found(self):
        assert mem.delete_routine("nonexistent") is False


# ---------------------------------------------------------------------------
# Observation ingestion
# ---------------------------------------------------------------------------

class TestObservations:
    def test_add_observation(self):
        embedding = [0.1] * 512
        metadata = {
            "entity_type": "person",
            "entity_id": "abc",
            "entity_name": "Alice",
            "room": "kitchen",
            "description": "Alice near the counter",
            "world_x": 2.3,
            "world_y": 1.4,
            "world_z": 0.9,
            "timestamp": 1709500000.0,
        }
        obs_id = mem.add_observation(embedding, metadata)
        assert obs_id.startswith("obs_")

    def test_add_multiple_observations(self):
        for i in range(5):
            embedding = [float(i) / 10] * 512
            mem.add_observation(embedding, {
                "entity_type": "scene",
                "room": "kitchen",
                "timestamp": 1709500000.0 + i,
            })
        s = mem.stats()
        assert s["observations"] == 5

    def test_observation_defaults(self):
        embedding = [0.1] * 512
        obs_id = mem.add_observation(embedding, {})
        # Should not crash — all metadata has defaults
        assert obs_id.startswith("obs_")

    def test_ingest_frame_without_clip(self):
        """ingest_frame needs CLIP — test that it calls _embed_image."""
        # Our mock clip_available returns True
        obs_id = mem.ingest_frame(b"\xff\xd8" + b"\x00" * 20, {
            "room": "kitchen",
        })
        assert obs_id.startswith("obs_")
        s = mem.stats()
        assert s["observations"] == 1


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:
    def _seed_observations(self):
        """Add some test observations."""
        for i in range(5):
            embedding = _fake_embed_image(f"scene_{i}".encode())
            mem.add_observation(embedding, {
                "entity_type": "scene",
                "room": "kitchen" if i < 3 else "bedroom",
                "description": f"Scene {i}",
                "world_x": float(i),
                "world_y": float(i),
                "world_z": 0.0,
                "timestamp": 1709500000.0 + i * 100,
            })
        # Add a person observation
        embedding = _fake_embed_image(b"person_alice")
        mem.add_observation(embedding, {
            "entity_type": "person",
            "entity_id": "alice123",
            "entity_name": "Alice",
            "room": "kitchen",
            "description": "Alice in kitchen",
            "world_x": 2.0,
            "world_y": 1.0,
            "world_z": 0.0,
            "timestamp": 1709500500.0,
        })

    def test_query_text(self):
        self._seed_observations()
        with patch.object(mem, "_embed_text", side_effect=_fake_embed_text):
            results = mem.query_text("kitchen scene")
        assert len(results) > 0
        # All results should have similarity scores
        assert all("similarity" in r for r in results)

    def test_query_text_with_room_filter(self):
        self._seed_observations()
        with patch.object(mem, "_embed_text", side_effect=_fake_embed_text):
            results = mem.query_text("scene", filters={"room": "kitchen"})
        # All results should be from kitchen
        for r in results:
            assert r.get("room") == "kitchen"

    def test_query_text_with_time_filter(self):
        self._seed_observations()
        with patch.object(mem, "_embed_text", side_effect=_fake_embed_text):
            results = mem.query_text("scene", filters={
                "time_start": 1709500000.0,
                "time_end": 1709500200.0,
            })
        for r in results:
            ts = r.get("timestamp", 0)
            assert 1709500000.0 <= ts <= 1709500200.0

    def test_person_sightings(self):
        self._seed_observations()
        results = mem.person_sightings("alice123")
        assert len(results) == 1
        assert results[0]["entity_name"] == "Alice"
        assert results[0]["room"] == "kitchen"

    def test_person_sightings_empty(self):
        self._seed_observations()
        results = mem.person_sightings("nonexistent")
        assert len(results) == 0

    def test_object_sightings(self):
        # Add an object observation
        embedding = _fake_embed_image(b"keys")
        mem.add_observation(embedding, {
            "entity_type": "object",
            "entity_id": "keys123",
            "entity_name": "Keys",
            "room": "kitchen",
            "timestamp": 1709500000.0,
        })
        results = mem.object_sightings("keys123")
        assert len(results) == 1
        assert results[0]["entity_name"] == "Keys"

    def test_room_activity(self):
        self._seed_observations()
        results = mem.room_activity("kitchen")
        # 3 scene observations + 1 person observation in kitchen
        assert len(results) == 4

    def test_room_activity_with_time_filter(self):
        self._seed_observations()
        results = mem.room_activity("kitchen",
                                    time_start=1709500000.0,
                                    time_end=1709500100.0)
        # Only scene 0 and scene 1 (timestamps 1709500000 and 1709500100)
        assert len(results) >= 1
        for r in results:
            assert r.get("room") == "kitchen"

    def test_query_nearby(self):
        self._seed_observations()
        results = mem.query_nearby(0.5, 0.5, 0.0, radius=1.5)
        # Should find observations near (0.5, 0.5, 0) — scene 0 at (0,0,0) and scene 1 at (1,1,0)
        assert len(results) >= 1
        for r in results:
            assert r.get("distance", 999) <= 1.5


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty(self):
        s = mem.stats()
        assert s["initialized"] is True
        assert s["persons"] == 0
        assert s["objects"] == 0
        assert s["rooms"] == 0
        assert s["routines"] == 0
        assert s["observations"] == 0

    def test_stats_after_adds(self):
        mem.add_person("Alice")
        mem.add_object("Keys")
        mem.add_room("Kitchen")
        mem.add_routine("Morning", "08:00")
        embedding = [0.1] * 512
        mem.add_observation(embedding, {"room": "kitchen"})

        s = mem.stats()
        assert s["persons"] >= 1  # At least the metadata entry
        assert s["objects"] >= 1
        assert s["rooms"] == 1
        assert s["routines"] == 1
        assert s["observations"] == 1


# ---------------------------------------------------------------------------
# Init / shutdown safety
# ---------------------------------------------------------------------------

class TestInitShutdown:
    def test_double_init_is_safe(self, tmp_path):
        # Already initialized by fixture — second init should warn but not crash
        mem.init_memory_only(str(tmp_path / "test_memory2"))
        assert mem._initialized is True

    def test_shutdown_and_reinit(self, tmp_path):
        mem.shutdown()
        assert mem._initialized is False
        assert mem.stats() == {"initialized": False}

        mem.init_memory_only(str(tmp_path / "test_memory3"))
        assert mem._initialized is True
        s = mem.stats()
        assert s["initialized"] is True

    def test_shutdown_without_init(self):
        mem.shutdown()
        mem.shutdown()  # Double shutdown should be safe

    def test_operations_after_shutdown(self):
        mem.shutdown()
        assert mem.list_persons() == []
        assert mem.list_objects() == []
        assert mem.list_rooms() == []
        assert mem.list_routines() == []
        assert mem.get_person("x") is None
        assert mem.delete_person("x") is False
