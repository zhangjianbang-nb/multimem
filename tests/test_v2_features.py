"""v0.2 feature tests: dual-space cross-modal retrieval, video timestamps,
retrieval feedback, migration, cosine regression."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from multimem import Memory, MemoryConfig, EPISODIC, SEMANTIC
from multimem.encode import ClipStyleEmbedder, SyntheticClipEmbedder
from multimem.media import sample_video_keyframes
from multimem.store import MemoryStore


# ---------------------------------------------------------------- dual space
def test_store_accepts_two_spaces(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    from multimem.types import MemoryItem

    item = MemoryItem(kind=EPISODIC, content="a red bike")
    store.add(item, np.array([1.0, 0.0], dtype=np.float32), "clip:x", space="text")
    store.upsert_embedding(
        item.id, np.array([0.0, 1.0], dtype=np.float32), "clip:x", space="vision"
    )
    rows_t, mat_t = store.rows_with_vectors("clip:x", space="text")
    rows_v, mat_v = store.rows_with_vectors("clip:x", space="vision")
    assert len(rows_t) == 1 and len(rows_v) == 1
    assert mat_t[0][1] == 0.0 and mat_v[0][1] == 1.0


def test_synthetic_clip_cross_modal(tmp_path):
    """A text query sharing tokens with the image filename must hit the
    image's *vision* vector — the cross-modal path."""
    img = tmp_path / "red_bike.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(img)
    clip = SyntheticClipEmbedder(dim=64)
    mem = Memory(":memory:", clip=clip)
    mem.add_image(img, caption="unrelated caption words")
    # "red bike" matches the vision vector via filename tokens, not the caption
    hits = mem.search("red bike", k=1)
    assert hits and hits[0].item.modality == "image"
    assert hits[0].parts["relevance"] > 0.9


def test_clip_fusion_prefers_vision_when_caption_poor(tmp_path):
    img = tmp_path / "sunset_road.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(img)
    clip = SyntheticClipEmbedder(dim=64)
    mem = Memory(":memory:", clip=clip)
    mem.add_image(img, caption="photo 001")
    mem.add_text("photo 001 archive", kind=SEMANTIC)
    hits = mem.search("sunset road", k=2)
    assert hits[0].item.modality == "image"  # found through the vision tower


def test_clip_broken_clip_degrades(tmp_path):
    """A clip embedder that fails to load must not break add/search."""

    class Broken:
        name = "clip:broken"
        dim = 8

        def embed(self, texts):
            raise RuntimeError("model not loaded")

        def embed_images(self, paths):
            raise RuntimeError("model not loaded")

    img = tmp_path / "x.png"
    from PIL import Image

    Image.new("RGB", (4, 4)).save(img)
    mem = Memory(":memory:", clip=Broken())
    item = mem.add_image(img, caption="some caption")
    assert item.id
    hits = mem.search("some caption", k=1)
    assert hits  # text space still works


# ---------------------------------------------------------------- video time
def test_sample_video_keyframes_signature():
    """The v0.2 sampler returns (path, t_seconds) tuples."""
    import inspect

    sig = inspect.signature(sample_video_keyframes)
    assert list(sig.parameters) == ["source", "out_dir", "max_frames"]


def test_search_video_within_window(tmp_path):
    mem = Memory(":memory:")
    for t in (5.0, 60.0, 300.0):
        mem.add_text(
            f"frame at {t}s",
            modality="video",
            meta={"video": "x.mp4", "t_seconds": t},
        )
    hits = mem.search("frame", k=10, video_within=(0.0, 120.0))
    assert {round(h.item.meta["t_seconds"]) for h in hits} == {5, 60}
    # without the window, all three come back
    assert len(mem.search("frame", k=10)) == 3


# ------------------------------------------------------------------ feedback
def test_retrieval_feedback_bumps_use_count():
    cfg = MemoryConfig(feedback=True, feedback_importance_delta=0.05)
    mem = Memory(":memory:", config=cfg)
    item = mem.add_text("retrieved often", importance=0.5)
    before = mem.get(item.id).importance
    mem.search("retrieved", k=1, feedback=True)
    got = mem.get(item.id)
    assert got.use_count == 1
    assert got.importance == 0.0 or True  # placeholder removed below
    assert got.importance == pytest.approx(before + 0.05)
    # feedback=False overrides config
    mem.search("retrieved", k=1, feedback=False)
    assert mem.get(item.id).use_count == 1


def test_feedback_off_by_default():
    mem = Memory(":memory:")
    item = mem.add_text("quiet item")
    mem.search("quiet", k=1)
    assert mem.get(item.id).use_count == 0


# ----------------------------------------------------------------- migration
def test_v1_database_migrates(tmp_path):
    """Open a v0.1-schema db; it must gain use_count + space column and work."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
            modality TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',
            importance REAL NOT NULL DEFAULT 0.5, created_at TEXT NOT NULL,
            valid_at TEXT, invalid_at TEXT,
            attachments_json TEXT NOT NULL DEFAULT '[]',
            meta_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE embeddings (
            item_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            dim INTEGER NOT NULL, embedder TEXT NOT NULL, vector BLOB NOT NULL
        );
        INSERT INTO memories (id, kind, content, modality, importance, created_at)
        VALUES ('abc123', 'episodic', 'legacy fact', 'text', 0.5, '2026-01-01T00:00:00+00:00');
        INSERT INTO embeddings VALUES ('abc123', 2, 'hash-256', x'0000803f00000000');
        """
    )
    conn.commit()
    conn.close()

    mem = Memory(db)
    got = mem.get("abc123")
    assert got is not None and got.content == "legacy fact"
    assert got.use_count == 0
    # the hand-made 2-dim legacy vector cannot match the live 256-dim
    # embedder — by design (v0.1: switch embedder -> old vectors don't
    # participate) it is filtered out rather than crashing the matmul
    assert mem.search("legacy", k=1) == []
    # still writable in v0.2 spaces
    mem.add_text("new after migration", kind=SEMANTIC)
    assert len(mem.search("after migration", k=1)) == 1


# ------------------------------------------------------------ clip adapter
def test_clipstyle_lazy_import_error(monkeypatch):
    """Without the optional dependency, ClipStyleEmbedder must raise a
    clear RuntimeError instead of ImportError leakage."""
    import sys

    # make `from transformers import ...` fail as if not installed
    monkeypatch.setitem(sys.modules, "transformers", None)
    c = ClipStyleEmbedder("nonexistent/model")
    with pytest.raises(RuntimeError, match="optional dependency"):
        c.embed(["hello"])
