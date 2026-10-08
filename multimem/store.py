"""SQLite-backed vector store.

Uses sqlite3 from the stdlib (WAL mode) plus numpy brute-force cosine
search — deliberate: agent memories are 10^3–10^6 items where a 256–4096
dim numpy scan takes <50 ms, and this keeps the package dependency-free.
Swap in faiss/lancedb behind the same interface if you — out of scope here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .types import MemoryItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    modality TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    valid_at TEXT,
    invalid_at TEXT,
    attachments_json TEXT NOT NULL DEFAULT '[]',
    meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE TABLE IF NOT EXISTS embeddings (
    item_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    embedder TEXT NOT NULL,
    vector BLOB NOT NULL
);
"""

_COLS = (
    "id, kind, content, modality, tags, importance, created_at, "
    "valid_at, invalid_at, attachments_json, meta_json"
)


def _to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()

    # -- writes ------------------------------------------------------------
    def add(self, item: MemoryItem, embedding: np.ndarray, embedder: str) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, kind, content, modality, tags, importance, created_at,
                valid_at, invalid_at, attachments_json, meta_json)
               VALUES (:id, :kind, :content, :modality, :tags, :importance,
                       :created_at, :valid_at, :invalid_at,
                       :attachments_json, :meta_json)""",
            item.to_row(),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (item_id, dim, embedder, vector)"
            " VALUES (?, ?, ?, ?)",
            (item.id, int(embedding.shape[-1]), embedder, _to_blob(embedding)),
        )
        self.conn.commit()

    def upsert_embedding(self, item_id: str, embedding: np.ndarray, embedder: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (item_id, dim, embedder, vector)"
            " VALUES (?, ?, ?, ?)",
            (item_id, int(embedding.shape[-1]), embedder, _to_blob(embedding)),
        )
        self.conn.commit()

    def delete(self, item_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
        self.conn.commit()

    def invalidate(self, item_id: str, when: str) -> None:
        self.conn.execute(
            "UPDATE memories SET invalid_at = ? WHERE id = ?", (when, item_id)
        )
        self.conn.commit()

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0])

    # -- reads -------------------------------------------------------------
    def get(self, item_id: str) -> Optional[MemoryItem]:
        row = self.conn.execute(
            f"SELECT {_COLS} FROM memories WHERE id = ?", (item_id,)
        ).fetchone()
        return MemoryItem.from_row(dict(row)) if row else None

    def get_many(self, ids: Iterable[str]) -> list[MemoryItem]:
        ids = list(ids)
        if not ids:
            return []
        placeholders = []
        for _ in ids:
            placeholders.append("?")
        sql = f"SELECT {_COLS} FROM memories WHERE id IN ({','.join(placeholders)})"
        rows = self.conn.execute(sql, ids).fetchall()
        return [MemoryItem.from_row(dict(r)) for r in rows]

    def iter_rows(
        self,
        kinds: Optional[Iterable[str]] = None,
        where: str = "",
        params: tuple = (),
    ) -> list[dict]:
        sql = f"SELECT {_COLS} FROM memories"
        conds: list[str] = []
        args: list = []
        if kinds:
            kind_list = list(kinds)
            qs = []
            for _ in kind_list:
                qs.append("?")
            conds.append("kind IN ( " + ",".join(qs) + " )")
            args.extend(kind_list)
        if where:
            conds.append(where)
            args.extend(params)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY created_at"
        rows = self.conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def rows_with_vectors(self, embedder: str) -> tuple[list[dict], np.ndarray]:
        sql = (
            "SELECT m.*, e.dim AS vdim, e.vector AS vvec FROM memories m"
            " JOIN embeddings e ON e.item_id = m.id WHERE e.embedder = ?"
        )
        rows = self.conn.execute(sql, (embedder,)).fetchall()
        metas = [dict(r) for r in rows]
        if not metas:
            return [], np.zeros((0, 1), dtype=np.float32)
        dim = int(rows[0]["vdim"])
        mat = np.stack([_from_blob(r["vvec"], dim) for r in rows])
        return metas, mat

    def all_embeddings_meta(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT item_id, embedder, dim FROM embeddings"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
