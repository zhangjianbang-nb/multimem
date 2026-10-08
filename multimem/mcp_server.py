"""A dependency-free MCP (Model Context Protocol) stdio server.

Exposes the memory store to any MCP client (Claude Desktop, ZCode, ...):

    multimem serve ./agent_mem

Implements the required JSON-RPC 2.0 handshake and the tools/* methods over
stdin/stdout with LSP-style Content-Length framing — no SDK needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .memory import Memory

PROTOCOL_VERSION = "2024-11-05"


def _frame(msg: dict) -> bytes:
    body = json.dumps(msg, ensure_ascii=False).encode()
    return b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body + b"\r\n\r\n"


def read_message() -> dict | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, val = line.decode().partition(":")
        headers[key.strip().lower()] = val.strip()
    length = int(headers.get("content-length", "0"))
    body = sys.stdin.buffer.read(length)
    return json.loads(body)


def write_message(msg: dict) -> None:
    sys.stdout.buffer.write(_frame(msg))
    sys.stdout.buffer.flush()


def make_server(mem: Memory) -> dict:
    """Return the tool registry: name -> (schema, handler(request_dict))."""

    def tool_add(req: dict) -> dict:
        a = req["arguments"]
        if a.get("path"):
            item = mem.add_image(a["path"], caption=a.get("caption"))
            return {"id": item.id, "modality": item.modality}
        item = mem.add_text(
            a["content"],
            kind=a.get("kind", "episodic"),
            importance=a.get("importance", 0.5),
            tags=a.get("tags"),
        )
        return {"id": item.id, "modality": item.modality}

    def tool_search(req: dict) -> dict:
        hits = mem.search(req["arguments"]["query"], k=req["arguments"].get("k", 5))
        return {"hits": [h.brief() for h in hits]}

    def tool_list(req: dict) -> dict:
        rows = mem.store.iter_rows()
        return {"memories": [f"[{r['kind']}/{r['modality']}] {r['content'][:80]}" for r in rows]}

    def tool_forget(req: dict) -> dict:
        mem.forget(req["arguments"]["id"], hard=bool(req["arguments"].get("hard", False)))
        return {"ok": True}

    return {
        "add_memory": {
            "description": "Store a memory (text, or an image by path with caption).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "path": {"type": "string", "description": "image file path"},
                    "caption": {"type": "string"},
                    "kind": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                    "importance": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
            "handler": tool_add,
        },
        "search_memory": {
            "description": "Hybrid search over all memories; returns ranked briefs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            "handler": tool_search,
        },
        "list_memories": {
            "description": "List all memories as brief lines.",
            "inputSchema": {"type": "object", "properties": {}},
            "handler": tool_list,
        },
        "forget_memory": {
            "description": "Soft-invalidate (default) or hard-delete a memory by id.",
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "hard": {"type": "boolean"}},
                "required": ["id"],
            },
            "handler": tool_forget,
        },
    }


def handle_request(mem: Memory, req: dict, registry: dict) -> dict | None:
    method = req.get("method", "")
    rid = req.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": "multimem", "version": "0.2.0"},
                "capabilities": {"tools": {}},
            },
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {"tools": [
                {"name": name, "description": spec["description"],
                 "inputSchema": spec["inputSchema"]}
                for name, spec in registry.items()
            ]},
        }
    if method == "tools/call":
        name = req["params"]["name"]
        spec = registry.get(name)
        if not spec:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602,
                     "message": f"unknown tool {name}"}}
        result = spec["handler"](req["params"])
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": json.dumps(
                    result, ensure_ascii=False)}]}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601,
                 "message": f"method not found: {method}"}}
    return None


def serve(path: str) -> None:
    store_path = Path(path) if path != ":memory:" else ":memory:"
    mem = Memory(store_path)
    registry = make_server(mem)
    while True:
        req = read_message()
        if req is None:
            break
        resp = handle_request(mem, req, registry)
        if resp is not None:
            write_message(resp)
