"""Storage-layer tests (all :memory:, no disk)."""
import numpy as np
import pytest
from multimem.store import MemoryStore
from multimem.types import MemoryItem


@pytest.fixture
def store():
    return MemoryStore(":memory:")


def _item(content="c", kind="episodic", **kw):
    return MemoryItem(kind=kind, content=content, **kw)


def test_add_and_get(store):
    it = _item("hello")
    store.add(it, np.ones(4, dtype=np.float32), "test")
    got = store.get(it.id)
    assert got is not None and got.content == "hello"


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_count(store):
    for i in range(3):
        store.add(_item(f"m{i}"), np.ones(4, dtype=np.float32), "test")
    assert store.count() == 3


def test_rows_with_vectors(store):
    v = np.array([1, 0, 0, 0], dtype=np.float32)
    store.add(_item("a"), v, "test")
    rows, mat = store.rows_with_vectors("test")
    assert len(rows) == 1 and mat.shape == (1, 4)
    assert np.allclose(mat[0], v)


def test_rows_with_vectors_wrong_embedder_empty(store):
    store.add(_item("a"), np.ones(4, dtype=np.float32), "test")
    rows, mat = store.rows_with_vectors("other")
    assert rows == []


def test_invalidate(store):
    it = _item("a")
    store.add(it, np.ones(4, dtype=np.float32), "test")
    store.invalidate(it.id, "2026-01-01T00:00:00+00:00")
    assert store.get(it.id).invalid_at == "2026-01-01T00:00:00+00:00"


def test_delete_cascades(store):
    it = _item("a")
    store.add(it, np.ones(4, dtype=np.float32), "test")
    store.delete(it.id)
    assert store.get(it.id) is None
    assert store.all_embeddings_meta() == []


def test_iter_rows_kind_filter(store):
    store.add(_item("a", kind="semantic"), np.ones(4, dtype=np.float32), "t")
    store.add(_item("b", kind="episodic"), np.ones(4, dtype=np.float32), "t")
    rows = store.iter_rows(kinds=["semantic"])
    assert len(rows) == 1 and rows[0]["kind"] == "semantic"


def test_get_many(store):
    ids = []
    for i in range(3):
        it = _item(f"m{i}")
        store.add(it, np.ones(4, dtype=np.float32), "t")
        ids.append(it.id)
    got = store.get_many(ids[:2])
    assert len(got) == 2
    assert store.get_many([]) == []


def test_file_store_roundtrip(tmp_path):
    db = tmp_path / "mem.db"
    s1 = MemoryStore(db)
    it = _item("persist me")
    s1.add(it, np.ones(4, dtype=np.float32), "t")
    s1.close()
    s2 = MemoryStore(db)
    assert s2.get(it.id).content == "persist me"
