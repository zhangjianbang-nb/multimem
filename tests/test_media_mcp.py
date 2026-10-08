"""Media + MCP server tests."""
import json
import pytest

from multimem.media import MediaStore, default_caption, guess_media_type
from multimem.memory import Memory
from multimem.mcp_server import make_server, handle_request, _frame
from multimem.types import Attachment


def _png(tmp_path):
    from PIL import Image
    p = tmp_path / "dashboard.png"
    Image.new("RGB", (32, 16), (200, 30, 30)).save(p)
    return p


def test_media_store_dedup(tmp_path):
    ms = MediaStore(tmp_path)
    p = _png(tmp_path)
    a = ms.save(p)
    b = ms.save(p)
    assert a == b and a.exists()


def test_default_caption_has_dims(tmp_path):
    p = _png(tmp_path)
    cap = default_caption(p)
    assert "32x16" in cap


def test_guess_media_type(tmp_path):
    assert guess_media_type(_png(tmp_path)) == "image/png"


def test_add_image_memory(tmp_path):
    m = Memory(str(tmp_path))
    p = _png(tmp_path)
    item = m.add_image(p, caption="red rectangle dashboard")
    assert item.modality == "image"
    assert item.attachments[0].path is not None
    hits = m.search("red rectangle dashboard")
    assert len(hits) == 1


def test_add_image_missing_file(tmp_path):
    m = Memory(str(tmp_path)) if False else Memory(tmp_path)
    with pytest.raises(FileNotFoundError):
        m.add_image(tmp_path / "nope.png")


def test_frame_format():
    fr = _frame({"a": 1})
    assert fr.startswith(b"Content-Length: ") and fr.endswith(b"\r\n\r\n" if True else b"")


def _roundtrip(mem, registry, msg):
    return handle_request(mem, msg, registry)


def test_mcp_initialize_and_tools(tmp_path):
    m = Memory(tmp_path)
    reg = make_server(m)
    resp = _roundtrip(m, reg, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["serverInfo"]["name"] == "multimem"
    resp = _roundtrip(m, reg, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in resp["result"]["tools"]]
    assert "search_memory" in names and "add_memory" in names


def test_mcp_add_search_forget(tmp_path):
    m = Memory(tmp_path)
    reg = make_server(m)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}}
    call["params"] = {"name": "add_memory", "arguments": {"content": "prefers window seat", "kind": "semantic"}}
    r1 = _roundtrip(m, reg, call)
    item_id = json.loads(r1["result"]["content"][0]["text"])["id"]
    call["params"] = {"name": "search_memory", "arguments": {"query": "seat preference"}}
    r2 = _roundtrip(m, reg, call)
    assert "prefers window seat" in r2["result"]["content"][0]["text"]
    call["params"] = {"name": "forget_memory", "arguments": {"id": item_id}}
    _roundtrip(m, reg, call)
    assert m.search("window seat") == []


def test_mcp_unknown_method(tmp_path):
    m = Memory(tmp_path)
    reg = make_server(m)
    resp = _roundtrip(m, reg, {"jsonrpc": "2.0", "id": 9, "method": "bogus", "params": {}})
    assert resp["error"]["code"] == -32601