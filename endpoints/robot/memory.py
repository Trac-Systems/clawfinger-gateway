"""Spatial memory — persistent environment knowledge backed by ChromaDB + CLIP.

Two layers:
1. Reference knowledge: persons, objects, rooms, routines taught by the user.
2. Runtime observations: camera frames with spatial metadata, accumulated
   from the robot observer (Jetson or sim adapter).

Observations arrive as pre-computed CLIP embeddings over Intercom (~2KB each).
CLIP on the Mac Mini is only used for teaching (reference photos) and text queries.

Schema is voxel-ready — every observation stores (x, y, z) world coordinates
for future 3D spatial indexing.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from io import BytesIO
from typing import Any

logger = logging.getLogger("gateway.robot.memory")

# ---------------------------------------------------------------------------
# Module-level state (same pattern as perception.py, skill_loader.py)
# ---------------------------------------------------------------------------

_client: Any = None               # chromadb.PersistentClient
_clip_model: Any = None           # open_clip model
_clip_preprocess: Any = None      # image transform
_clip_tokenizer: Any = None       # text tokenizer
_initialized = False

# ChromaDB collection handles
_persons: Any = None
_objects: Any = None
_rooms: Any = None
_routines: Any = None
_observations: Any = None

_EMBEDDING_DIM = 512  # ViT-B-32 output dimension


# ---------------------------------------------------------------------------
# Init / shutdown
# ---------------------------------------------------------------------------

def init(db_path: str, embedding_model: str = "ViT-B-32",
         device: str = "mps") -> None:
    """Initialize ChromaDB persistent store and load CLIP model."""
    global _client, _persons, _objects, _rooms, _routines, _observations
    global _initialized

    if _initialized:
        logger.warning("Spatial memory already initialized — skipping")
        return

    import chromadb

    _client = chromadb.PersistentClient(path=db_path)

    # Create/get collections (ChromaDB is idempotent on get_or_create)
    _persons = _client.get_or_create_collection(
        name="persons", metadata={"hnsw:space": "cosine"})
    _objects = _client.get_or_create_collection(
        name="objects", metadata={"hnsw:space": "cosine"})
    _rooms = _client.get_or_create_collection(
        name="rooms", metadata={"hnsw:space": "cosine"})
    _routines = _client.get_or_create_collection(name="routines")
    _observations = _client.get_or_create_collection(
        name="observations", metadata={"hnsw:space": "cosine"})

    # Load CLIP (may take a few seconds on first call)
    _load_clip_model(embedding_model, device)

    _initialized = True
    logger.info("Spatial memory initialized (db=%s, clip=%s, device=%s)",
                db_path, embedding_model, device)


def init_memory_only(db_path: str) -> None:
    """Initialize ChromaDB without loading CLIP (for testing)."""
    global _client, _persons, _objects, _rooms, _routines, _observations
    global _initialized

    if _initialized:
        return

    import chromadb

    _client = chromadb.PersistentClient(path=db_path)
    _persons = _client.get_or_create_collection(
        name="persons", metadata={"hnsw:space": "cosine"})
    _objects = _client.get_or_create_collection(
        name="objects", metadata={"hnsw:space": "cosine"})
    _rooms = _client.get_or_create_collection(
        name="rooms", metadata={"hnsw:space": "cosine"})
    _routines = _client.get_or_create_collection(name="routines")
    _observations = _client.get_or_create_collection(
        name="observations", metadata={"hnsw:space": "cosine"})

    _initialized = True
    logger.info("Spatial memory initialized (db=%s, no CLIP)", db_path)


def shutdown() -> None:
    """Close ChromaDB cleanly."""
    global _client, _clip_model, _clip_preprocess, _clip_tokenizer
    global _persons, _objects, _rooms, _routines, _observations
    global _initialized

    if not _initialized:
        return

    _client = None
    _clip_model = None
    _clip_preprocess = None
    _clip_tokenizer = None
    _persons = None
    _objects = None
    _rooms = None
    _routines = None
    _observations = None
    _initialized = False
    logger.info("Spatial memory shut down")


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def _load_clip_model(model_name: str, device: str) -> None:
    """Load open_clip model + preprocessing + tokenizer."""
    global _clip_model, _clip_preprocess, _clip_tokenizer

    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    _clip_model = model
    _clip_preprocess = preprocess
    _clip_tokenizer = tokenizer

    logger.info("CLIP model loaded: %s on %s", model_name, device)


def _embed_image(image_bytes: bytes) -> list[float]:
    """Embed an image using CLIP. Returns 512-dim vector."""
    import torch
    from PIL import Image

    if _clip_model is None or _clip_preprocess is None:
        raise RuntimeError("CLIP model not loaded — call init() first")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    device = next(_clip_model.parameters()).device

    with torch.no_grad():
        image_tensor = _clip_preprocess(image).unsqueeze(0).to(device)
        features = _clip_model.encode_image(image_tensor)
        features /= features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()


def _embed_text(text: str) -> list[float]:
    """Embed text using CLIP. Returns 512-dim vector."""
    import torch

    if _clip_model is None or _clip_tokenizer is None:
        raise RuntimeError("CLIP model not loaded — call init() first")

    device = next(_clip_model.parameters()).device

    with torch.no_grad():
        tokens = _clip_tokenizer([text]).to(device)
        features = _clip_model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()


def clip_available() -> bool:
    """Check if CLIP model is loaded."""
    return _clip_model is not None


# ---------------------------------------------------------------------------
# Person CRUD
# ---------------------------------------------------------------------------

def add_person(name: str, description: str = "",
               reference_images: list[bytes] | None = None) -> dict:
    """Create a person entry with optional reference photo embeddings."""
    if not _initialized:
        raise RuntimeError("Spatial memory not initialized")

    person_id = str(uuid.uuid4())[:8]

    # Embed reference images if CLIP is available
    embeddings = []
    if reference_images and clip_available():
        for img in reference_images:
            embeddings.append(_embed_image(img))

    # Store person metadata in the persons collection
    # If we have embeddings, store each as a separate document with the person_id
    if embeddings:
        ids = [f"{person_id}_ref_{i}" for i in range(len(embeddings))]
        metadatas = [{"person_id": person_id, "name": name,
                      "description": description, "type": "reference",
                      "ref_index": i} for i in range(len(embeddings))]
        documents = [f"Reference photo {i} of {name}" for i in range(len(embeddings))]
        _persons.add(ids=ids, embeddings=embeddings, metadatas=metadatas,
                     documents=documents)
    else:
        # Store at least a metadata entry (no embedding)
        _persons.add(
            ids=[f"{person_id}_meta"],
            embeddings=[[0.0] * _EMBEDDING_DIM],  # placeholder
            metadatas=[{"person_id": person_id, "name": name,
                        "description": description, "type": "metadata"}],
            documents=[f"Person: {name}. {description}"],
        )

    return {"id": person_id, "name": name, "description": description,
            "image_count": len(embeddings)}


def list_persons() -> list[dict]:
    """Return all known persons."""
    if not _initialized:
        return []

    results = _persons.get()
    # Deduplicate by person_id
    seen = {}
    for i, meta in enumerate(results.get("metadatas", [])):
        pid = meta.get("person_id", "")
        if pid not in seen:
            seen[pid] = {
                "id": pid,
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "image_count": 0,
            }
        if meta.get("type") == "reference":
            seen[pid]["image_count"] += 1

    return list(seen.values())


def get_person(person_id: str) -> dict | None:
    """Get a single person by ID."""
    if not _initialized:
        return None

    results = _persons.get(where={"person_id": person_id})
    if not results["ids"]:
        return None

    name = ""
    description = ""
    image_count = 0
    for meta in results["metadatas"]:
        name = meta.get("name", name)
        description = meta.get("description", description)
        if meta.get("type") == "reference":
            image_count += 1

    return {"id": person_id, "name": name, "description": description,
            "image_count": image_count}


def delete_person(person_id: str) -> bool:
    """Remove a person and all their embeddings."""
    if not _initialized:
        return False

    results = _persons.get(where={"person_id": person_id})
    if not results["ids"]:
        return False

    _persons.delete(ids=results["ids"])
    return True


def add_person_reference(person_id: str, image_bytes: bytes) -> dict | None:
    """Add a reference photo to an existing person."""
    if not _initialized or not clip_available():
        return None

    person = get_person(person_id)
    if not person:
        return None

    embedding = _embed_image(image_bytes)
    ref_idx = person["image_count"]
    doc_id = f"{person_id}_ref_{ref_idx}"

    _persons.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"person_id": person_id, "name": person["name"],
                    "description": person["description"],
                    "type": "reference", "ref_index": ref_idx}],
        documents=[f"Reference photo {ref_idx} of {person['name']}"],
    )

    return {"id": person_id, "ref_index": ref_idx}


# ---------------------------------------------------------------------------
# Object CRUD
# ---------------------------------------------------------------------------

def add_object(name: str, description: str = "",
               reference_images: list[bytes] | None = None) -> dict:
    """Create an object entry with optional reference photo embeddings."""
    if not _initialized:
        raise RuntimeError("Spatial memory not initialized")

    object_id = str(uuid.uuid4())[:8]

    embeddings = []
    if reference_images and clip_available():
        for img in reference_images:
            embeddings.append(_embed_image(img))

    if embeddings:
        ids = [f"{object_id}_ref_{i}" for i in range(len(embeddings))]
        metadatas = [{"object_id": object_id, "name": name,
                      "description": description, "type": "reference",
                      "ref_index": i} for i in range(len(embeddings))]
        documents = [f"Reference photo {i} of {name}" for i in range(len(embeddings))]
        _objects.add(ids=ids, embeddings=embeddings, metadatas=metadatas,
                     documents=documents)
    else:
        _objects.add(
            ids=[f"{object_id}_meta"],
            embeddings=[[0.0] * _EMBEDDING_DIM],
            metadatas=[{"object_id": object_id, "name": name,
                        "description": description, "type": "metadata"}],
            documents=[f"Object: {name}. {description}"],
        )

    return {"id": object_id, "name": name, "description": description,
            "image_count": len(embeddings)}


def list_objects() -> list[dict]:
    """Return all known objects."""
    if not _initialized:
        return []

    results = _objects.get()
    seen = {}
    for meta in results.get("metadatas", []):
        oid = meta.get("object_id", "")
        if oid not in seen:
            seen[oid] = {
                "id": oid,
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "image_count": 0,
            }
        if meta.get("type") == "reference":
            seen[oid]["image_count"] += 1

    return list(seen.values())


def get_object(object_id: str) -> dict | None:
    """Get a single object by ID."""
    if not _initialized:
        return None

    results = _objects.get(where={"object_id": object_id})
    if not results["ids"]:
        return None

    name = ""
    description = ""
    image_count = 0
    for meta in results["metadatas"]:
        name = meta.get("name", name)
        description = meta.get("description", description)
        if meta.get("type") == "reference":
            image_count += 1

    return {"id": object_id, "name": name, "description": description,
            "image_count": image_count}


def delete_object(object_id: str) -> bool:
    """Remove an object and all its embeddings."""
    if not _initialized:
        return False

    results = _objects.get(where={"object_id": object_id})
    if not results["ids"]:
        return False

    _objects.delete(ids=results["ids"])
    return True


def add_object_reference(object_id: str, image_bytes: bytes) -> dict | None:
    """Add a reference photo to an existing object."""
    if not _initialized or not clip_available():
        return None

    obj = get_object(object_id)
    if not obj:
        return None

    embedding = _embed_image(image_bytes)
    ref_idx = obj["image_count"]
    doc_id = f"{object_id}_ref_{ref_idx}"

    _objects.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"object_id": object_id, "name": obj["name"],
                    "description": obj["description"],
                    "type": "reference", "ref_index": ref_idx}],
        documents=[f"Reference photo {ref_idx} of {obj['name']}"],
    )

    return {"id": object_id, "ref_index": ref_idx}


# ---------------------------------------------------------------------------
# Room CRUD
# ---------------------------------------------------------------------------

def add_room(name: str, description: str = "",
             reference_image: bytes | None = None) -> dict:
    """Define a room/zone."""
    if not _initialized:
        raise RuntimeError("Spatial memory not initialized")

    room_id = str(uuid.uuid4())[:8]

    if reference_image and clip_available():
        embedding = _embed_image(reference_image)
    else:
        embedding = [0.0] * _EMBEDDING_DIM

    _rooms.add(
        ids=[room_id],
        embeddings=[embedding],
        metadatas=[{"name": name, "description": description,
                    "has_image": reference_image is not None and clip_available()}],
        documents=[f"Room: {name}. {description}"],
    )

    return {"id": room_id, "name": name, "description": description}


def list_rooms() -> list[dict]:
    """Return all known rooms."""
    if not _initialized:
        return []

    results = _rooms.get()
    rooms = []
    for i, rid in enumerate(results.get("ids", [])):
        meta = results["metadatas"][i]
        rooms.append({
            "id": rid,
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
        })
    return rooms


def get_room(room_id: str) -> dict | None:
    """Get a single room by ID."""
    if not _initialized:
        return None

    results = _rooms.get(ids=[room_id])
    if not results["ids"]:
        return None

    meta = results["metadatas"][0]
    return {"id": room_id, "name": meta.get("name", ""),
            "description": meta.get("description", "")}


def delete_room(room_id: str) -> bool:
    """Remove a room."""
    if not _initialized:
        return False

    try:
        results = _rooms.get(ids=[room_id])
        if not results["ids"]:
            return False
        _rooms.delete(ids=[room_id])
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Routine CRUD
# ---------------------------------------------------------------------------

def add_routine(name: str, schedule: str, description: str = "") -> dict:
    """Create a text-based routine."""
    if not _initialized:
        raise RuntimeError("Spatial memory not initialized")

    routine_id = str(uuid.uuid4())[:8]

    _routines.add(
        ids=[routine_id],
        metadatas=[{"name": name, "schedule": schedule,
                    "description": description}],
        documents=[f"Routine: {name}. Schedule: {schedule}. {description}"],
    )

    return {"id": routine_id, "name": name, "schedule": schedule,
            "description": description}


def list_routines() -> list[dict]:
    """Return all routines."""
    if not _initialized:
        return []

    results = _routines.get()
    routines = []
    for i, rid in enumerate(results.get("ids", [])):
        meta = results["metadatas"][i]
        routines.append({
            "id": rid,
            "name": meta.get("name", ""),
            "schedule": meta.get("schedule", ""),
            "description": meta.get("description", ""),
        })
    return routines


def delete_routine(routine_id: str) -> bool:
    """Remove a routine."""
    if not _initialized:
        return False

    try:
        results = _routines.get(ids=[routine_id])
        if not results["ids"]:
            return False
        _routines.delete(ids=[routine_id])
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Observations (primary pipeline — pre-computed embeddings from observer)
# ---------------------------------------------------------------------------

def add_observation(embedding: list[float], metadata: dict) -> str:
    """Store a pre-computed embedding + metadata.

    This is the primary pipeline: the observer (Jetson/sim) runs CLIP locally
    and sends the embedding over Intercom. No CLIP computation on Mac Mini.
    """
    if not _initialized:
        raise RuntimeError("Spatial memory not initialized")

    obs_id = f"obs_{str(uuid.uuid4())[:8]}"

    # Ensure required metadata defaults
    meta = {
        "entity_type": metadata.get("entity_type", "scene"),
        "entity_id": metadata.get("entity_id", ""),
        "entity_name": metadata.get("entity_name", ""),
        "room": metadata.get("room", ""),
        "description": metadata.get("description", ""),
        "labels": metadata.get("labels", ""),
        "world_x": float(metadata.get("world_x", 0.0)),
        "world_y": float(metadata.get("world_y", 0.0)),
        "world_z": float(metadata.get("world_z", 0.0)),
        "robot_x": float(metadata.get("robot_x", 0.0)),
        "robot_y": float(metadata.get("robot_y", 0.0)),
        "robot_theta": float(metadata.get("robot_theta", 0.0)),
        "timestamp": float(metadata.get("timestamp", time.time())),
        "source": metadata.get("source", "head_rgb"),
        "depth_available": bool(metadata.get("depth_available", False)),
    }

    _observations.add(
        ids=[obs_id],
        embeddings=[embedding],
        metadatas=[meta],
        documents=[meta["description"] or f"Observation at {meta['timestamp']}"],
    )

    return obs_id


def ingest_frame(image_bytes: bytes, metadata: dict) -> str:
    """Embed an image on Mac Mini + store. Convenience for testing/manual upload.

    NOT the runtime pipeline — runtime uses add_observation with pre-computed
    embeddings from the observer.
    """
    if not _initialized or not clip_available():
        raise RuntimeError("Spatial memory or CLIP not available")

    embedding = _embed_image(image_bytes)
    return add_observation(embedding, metadata)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def _build_where_filter(filters: dict) -> dict | None:
    """Build a ChromaDB where clause from query filters."""
    conditions = []

    if filters.get("room"):
        conditions.append({"room": filters["room"]})
    if filters.get("entity_type"):
        conditions.append({"entity_type": filters["entity_type"]})
    if filters.get("time_start") is not None:
        conditions.append({"timestamp": {"$gte": float(filters["time_start"])}})
    if filters.get("time_end") is not None:
        conditions.append({"timestamp": {"$lte": float(filters["time_end"])}})

    # Geometric bounding box (square approximation — true radius filtered post-query)
    if all(k in filters for k in ("near_x", "near_y", "near_z", "radius")):
        r = float(filters["radius"])
        cx, cy, cz = float(filters["near_x"]), float(filters["near_y"]), float(filters["near_z"])
        conditions.extend([
            {"world_x": {"$gte": cx - r}},
            {"world_x": {"$lte": cx + r}},
            {"world_y": {"$gte": cy - r}},
            {"world_y": {"$lte": cy + r}},
            {"world_z": {"$gte": cz - r}},
            {"world_z": {"$lte": cz + r}},
        ])

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _filter_by_radius(results: list[dict], cx: float, cy: float,
                       cz: float, radius: float) -> list[dict]:
    """Post-filter results by true spherical distance."""
    filtered = []
    for r in results:
        dx = r.get("world_x", 0) - cx
        dy = r.get("world_y", 0) - cy
        dz = r.get("world_z", 0) - cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= radius:
            r["distance"] = round(dist, 3)
            filtered.append(r)
    return filtered


def _format_results(query_results: dict) -> list[dict]:
    """Convert ChromaDB query results to flat list of dicts."""
    items = []
    ids = query_results.get("ids", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]
    documents = query_results.get("documents", [[]])[0]

    for i, obs_id in enumerate(ids):
        item = {"id": obs_id}
        if i < len(metadatas):
            item.update(metadatas[i])
        if i < len(distances):
            item["similarity"] = round(1.0 - distances[i], 4)  # cosine distance → similarity
        if i < len(documents):
            item["document"] = documents[i]
        items.append(item)

    return items


def query_text(text: str, n_results: int = 10,
               filters: dict | None = None) -> list[dict]:
    """Semantic text search across observations."""
    if not _initialized or not clip_available():
        return []

    embedding = _embed_text(text)
    where = _build_where_filter(filters or {})

    kwargs: dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where

    results = _observations.query(**kwargs)
    items = _format_results(results)

    # Post-filter by radius if specified
    if filters and all(k in filters for k in ("near_x", "near_y", "near_z", "radius")):
        items = _filter_by_radius(
            items, float(filters["near_x"]), float(filters["near_y"]),
            float(filters["near_z"]), float(filters["radius"]))

    return items


def query_image(image_bytes: bytes, n_results: int = 10,
                filters: dict | None = None) -> list[dict]:
    """Visual similarity search across observations."""
    if not _initialized or not clip_available():
        return []

    embedding = _embed_image(image_bytes)
    where = _build_where_filter(filters or {})

    kwargs: dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where

    results = _observations.query(**kwargs)
    return _format_results(results)


def query_nearby(x: float, y: float, z: float, radius: float,
                 n_results: int = 50, filters: dict | None = None) -> list[dict]:
    """Geometric proximity — all observations within radius of (x, y, z)."""
    if not _initialized:
        return []

    f = dict(filters or {})
    f["near_x"] = x
    f["near_y"] = y
    f["near_z"] = z
    f["radius"] = radius
    where = _build_where_filter(f)

    # Get all results within bounding box via ChromaDB metadata filter
    results = _observations.get(where=where, limit=n_results)

    items = []
    for i, obs_id in enumerate(results.get("ids", [])):
        item = {"id": obs_id}
        if i < len(results.get("metadatas", [])):
            item.update(results["metadatas"][i])
        items.append(item)

    # Post-filter by true radius
    return _filter_by_radius(items, x, y, z, radius)


def person_sightings(person_id: str, time_start: float | None = None,
                     time_end: float | None = None,
                     room: str | None = None) -> list[dict]:
    """Filter observations for a specific person."""
    if not _initialized:
        return []

    filters: dict[str, Any] = {"entity_type": "person"}
    if time_start is not None:
        filters["time_start"] = time_start
    if time_end is not None:
        filters["time_end"] = time_end
    if room:
        filters["room"] = room

    where_conditions = [{"entity_id": person_id}]
    base_where = _build_where_filter(filters)
    if base_where:
        if "$and" in base_where:
            where_conditions.extend(base_where["$and"])
        else:
            where_conditions.append(base_where)

    where = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

    results = _observations.get(where=where)
    items = []
    for i, obs_id in enumerate(results.get("ids", [])):
        item = {"id": obs_id}
        if i < len(results.get("metadatas", [])):
            item.update(results["metadatas"][i])
        items.append(item)
    return items


def object_sightings(object_id: str, time_start: float | None = None,
                     time_end: float | None = None,
                     room: str | None = None) -> list[dict]:
    """Filter observations for a specific object."""
    if not _initialized:
        return []

    filters: dict[str, Any] = {"entity_type": "object"}
    if time_start is not None:
        filters["time_start"] = time_start
    if time_end is not None:
        filters["time_end"] = time_end
    if room:
        filters["room"] = room

    where_conditions = [{"entity_id": object_id}]
    base_where = _build_where_filter(filters)
    if base_where:
        if "$and" in base_where:
            where_conditions.extend(base_where["$and"])
        else:
            where_conditions.append(base_where)

    where = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

    results = _observations.get(where=where)
    items = []
    for i, obs_id in enumerate(results.get("ids", [])):
        item = {"id": obs_id}
        if i < len(results.get("metadatas", [])):
            item.update(results["metadatas"][i])
        items.append(item)
    return items


def room_activity(room: str, time_start: float | None = None,
                  time_end: float | None = None) -> list[dict]:
    """Aggregate observations by room."""
    if not _initialized:
        return []

    filters: dict[str, Any] = {"room": room}
    if time_start is not None:
        filters["time_start"] = time_start
    if time_end is not None:
        filters["time_end"] = time_end

    where = _build_where_filter(filters)
    results = _observations.get(where=where)

    items = []
    for i, obs_id in enumerate(results.get("ids", [])):
        item = {"id": obs_id}
        if i < len(results.get("metadatas", [])):
            item.update(results["metadatas"][i])
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats() -> dict:
    """Return collection counts and model info."""
    if not _initialized:
        return {"initialized": False}

    return {
        "initialized": True,
        "persons": _persons.count(),
        "objects": _objects.count(),
        "rooms": _rooms.count(),
        "routines": _routines.count(),
        "observations": _observations.count(),
        "clip_loaded": clip_available(),
    }
