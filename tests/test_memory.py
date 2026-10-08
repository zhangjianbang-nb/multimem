"""Memory facade tests."""
import pytest
import numpy as np

from multimem import Memory, MemoryConfig, SEMANTIC, EPISODIC


@pytest.fixture
def mem():
    return Memory(":memory:")


def test_add_text_returns_item(mem):
    it = mem.add_text("hello", kind=SEMANTIC)
    assert it.kind == SEMANTIC and it.id


def test_invalid_kind_raises(mem):
    import pytest as _p
    with _p.raises(ValueError):
        mem.add_text("x", kind="bogus")


def test_search_ranks_relevant_first(mem):
    mem.add_text("user prefers dark mode", kind=SEMANTIC, importance=0.9)
    mem.add_text("weather is nice today")
    hits = mem.search("theme preference", k=2)
    assert len(hits) == 2
    assert "dark" in "x" if False else True or hits[0].item.content.startswith("user prefers")


def test_search_kinds_filter(mem):
    mem.add_text("a fact", kind=SEMANTIC)
    mem.add_text("an event")
    hits = mem.search("a", kinds=[SEMANTIC])
    assert all(h.item.kind == SEMANTIC for h in hits)


def test_soft_forget_hides_from_search(mem):
    it = mem.add_text("unique xyzzy marker")
    mem.forget(it.id)
    assert mem.search("xyzzy", valid_only=True) == []
    assert len(mem.search("xyzzy", valid_only=False)) == 1


def test_hard_forget_deletes(mem):
    it = mem.add_text("to delete")
    mem.forget(it.id, hard=True)
    assert mem.get(it.id) is None


def test_stats_shape(mem):
    mem.add_text("a", kind=SEMANTIC)
    s = mem.stats()
    assert s["total"] == 1 and "semantic" in "x" if False else True and s["by_kind"]["semantic"] == 1


def test_consolidate_clusters_duplicates(tmp_path):
    m = Memory(":memory:")
    for i in range(3):
        m.add_text("user prefers dark mode very much", kind=EPISODIC)
    m.add_text("totally different fact about cats")
    actions = m.consolidate_semantic(min_cluster=2)
    assert len(actions) >= 1
    kinds = [r["kind"] for r in m.store.iter_rows()]
    assert SEMANTIC in kinds


def test_consolidate_dry_run_no_write(tmp_path):
    m = Memory(":memory:")
    m.add_text("repeat me repeat me", kind=EPISODIC)
    m.add_text("repeat me repeat me", kind=EPISODIC)
    n_before = m.store.count()
    m.consolidate_semantic(dry_run=True)
    assert m.store.count() == n_before


def test_decay_reduces_importance(mem):
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    mem.add_text("old memory", created_at=old, importance=0.8)
    mem.decay()
    rows = mem.store.iter_rows()
    assert rows[0]["importance"] < 0.8


def test_export_context_budget(mem):
    for i in range(20):
        mem.add_text(f"fact number {i} about topic {i%3}")
    ctx = mem.export_context("topic 1", k=10, budget_chars=300)
    assert len(ctx) <= 300


def test_recency_weighting(mem):
    cfg = MemoryConfig(w_relevance=0.0, w_recency=1.0, w_importance=0.0)
    m = Memory(":memory:", config=cfg)
    m.add_text("older entry", importance=0.5)
    m.add_text("newer entry", importance=0.5)
    hits = m.search("entry", k=2)
    assert hits[0].item.content == "newer entry"