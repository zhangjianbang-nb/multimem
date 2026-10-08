"""SQLite-backed vector store.

Uses sqlite3 from the stdlib (WAL mode) plus numpy brute-force cosine
search — deliberate: agent memories are 10^3–10^6 items where a 256–4096
dim numpy scan takes <50 ms, and this keeps the package dependency-free.
Swap in faiss/lancedb behind the same interface if you — out of scope here.

v0.2 additions:
- embeddings table gains a ``space`` column ("text" or "vision") so a CLIP-style
  embedder can store a caption vector and an image vector for the same item
  (PK is now (item_id, embedder, space)). Old single-space databases are
  migrated in place on open.
- memories table gains ``use_count`` (retrieval feedback, Generative Agents
  style); old databases get the column via ALTER TABLE.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .types import MemoryItem

SCHEMA_VERSION = 2

_SCHEMA_V2 = """
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
    item_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    embedder TEXT NOT NULL,
    space TEXT NOT NULL DEFAULT 'text',
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (item_id, embedder, space)
);
"""

_COLS = (
    "id, kind, content, modality, tags, importance, use_count, created_at, "
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
        self._migrate()
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()

    # -- schema / migration -------------------------------------------------
    def _migrate(self) -> None:
        """Create the v2 schema; upgrade v1 databases in place."""
        self.conn.executescript(_SCHEMA_V2)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(memories)")}
        if "use_count" not in cols:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0"
            )
        # v1 embeddings tables had no space column and PK on item_id only.
        ecols = {r[1] for r in self.conn.execute("PRAGMA table_info(embeddings)")}
        if ecols and "space" not in ecols:
            self.conn.executescript(
                """
                CREATE TABLE embeddings_v2 (
                    item_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    embedder TEXT NOT NULL,
                    space TEXT NOT NULL DEFAULT 'text',
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (item_id, embedder, space)
                );
                INSERT OR IGNORE INTO embeddings_v2 (item_id, embedder, space, dim, vector)
                    SELECT item_id, embedder, 'text', dim, vector FROM embeddings;
                DROP TABLE embeddings;
                ALTER TABLE embeddings_v2 RENAME TO embeddings;
                """
            )
        self.conn.commit()

    # -- writes ------------------------------------------------------------
    def add(self, item: MemoryItem, embedding: np.ndarray, embedder: str,
            space: str = "text") -> None:
        row = item.to_row()
        row["use_count"] = getattr(item, "use_count", 0)
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        self.conn.execute(
            f"INSERT OR REPLACE INTO memories ({cols}) VALUES ({placeholders})",
            row,
        )
        self.upsert_embedding(item.id, embedding, embedder, space)
        self.conn.commit()

    def upsert_embedding(self, item_id: str, embedding: np.ndarray, embedder: str,
                          space: str = "text") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (item_id, embedder, space, dim, vector)"
            " VALUES (?, ?, ?, ?, ?)",
            (item_id, embedder, space, int(embedding.shape[-1]), _to_blob(embedding)),
        )
        self.conn.commit()

    def bump_use(self, item_id: str, importance_delta: float = 0.0) -> None:
        """Retrieval feedback: use_count += 1 and optionally nudge importance."""
        if importance_delta:
            self.conn.execute(
                "UPDATE memories SET use_count = use_count + 1, "
                "importance = min(1.0, importance + ?) WHERE id = ?",
                (importance_delta, item_id),
            )
        else:
            self.conn.execute(
                "UPDATE memories SET use_count = use_count + 1 WHERE id = ?",
                (item_id,),
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

    def rows_with_vectors(
        self, embedder: str, space: str = "text", dim: int | None = None
    ) -> tuple[list[dict], np.ndarray]:
        """Rows joined with their vectors in ``space`` (items without one are dropped).

        ``dim``: when given, rows whose vectors have a different dimensionality
        are excluded (vectors from another embedder generation cannot be
        compared and would crash the matmul). When None, the majority dim wins.
        """
        sql = (
            "SELECT m.*, e.dim AS vdim, e.vector AS vvec FROM memories m"
            " JOIN embeddings e ON e.item_id = m.id"
            " WHERE e.embedder = ? AND e.space = ?"
        )
        rows = self.conn.execute(sql, (embedder, space)).fetchall()
        metas = [dict(r) for r in rows]
        if not metas:
            return [], np.zeros((0, 1), dtype=np.float32)
        if dim is None:
            from collections import Counter

            dim, _ = Counter(int(r["vdim"]) for r in metas).most_common(1)[0]
        metas = [r for r in metas if int(r["vdim"]) == dim]
        if not metas:
            return [], np.zeros((0, dim), dtype=np.float32)
        mat = np.stack([_from_blob(r["vvec"], dim) for r in metas])
        return metas, mat

    def has_embedding(self, item_id: str, embedder: str, space: str = "text") -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE item_id=? AND embedder=? AND space=?",
            (item_id, embedder, space),
        ).fetchone()
        return row is not None

    def all_embeddings_meta(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT item_id, embedder, space, dim FROM embeddings"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
