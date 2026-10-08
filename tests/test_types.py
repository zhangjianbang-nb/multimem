"""Data-model tests."""
import json
from multimem.types import MemoryItem, Attachment, SearchHit


def test_item_roundtrip():
    item = MemoryItem(kind="semantic", content="fact", tags=["a", "b"], importance=0.8)
    row = item.to_row()
    assert row["tags"] == "a,b"
    clone = MemoryItem.from_row(row)
    assert clone.id == item.id and clone.tags == ["a", "b"]
    assert clone.importance == 0.8


def test_attachment_roundtrip():
    att = Attachment(media_type="image/png", path="/x/y.png", caption="cap")
    raw = att.to_dict()
    clone = Attachment.from_dict(raw)
    assert clone == att


def test_attachment_meta_default_not_shared():
    a1, a2 = Attachment("image/png"), Attachment("image/png")
    a1.meta["x"] = 1
    assert a2.meta == {}


def test_searchhit_brief():
    item = MemoryItem(kind="episodic", content="hello world")
    hit = SearchHit(item=item, score=0.5, parts={"relevance": 0.5})
    assert "episodic" in hit.brief() and "hello" in hit.brief()


def test_ids_unique():
    a, b = MemoryItem(kind="semantic", content="x"), MemoryItem(kind="semantic", content="x")
    assert a.id != b.id
