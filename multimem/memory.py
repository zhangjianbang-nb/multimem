"""Memory: the main entry point.

    from multimem import Memory

    mem = Memory("./agent_mem")            # or Memory(":memory:") for tests
    mem.add_text("User prefers dark mode", kind="semantic", importance=0.9)
    mem.add_image("shot.png", caption="dashboard with red error banner")
    hits = mem.search("what theme does the user like?", k=3)
    print(hits[0].brief())

Scoring follows the Generative Agents recipe, extended with validity
filtering and multi-hop expansion (see docs/DESIGN.md section 3):

    score = w_relevance * relevance + w_recency * recency + w_importance * importance
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .encode import EmbeddingProvider, HashingEmbedder
from .media import ImageCaptionFn, MediaStore, guess_media_type, ingest_image
from .store import MemoryStore
from .types import (
    EPISODIC,
    KINDS,
    MOD_IMAGE,
    MOD_MULTI,
    MOD_TEXT,
    MOD_VIDEO,
    SEMANTIC,
    Attachment,
    MemoryItem,
    SearchHit,
    now_iso,
    parse_ts,
)

__all__ = ["Memory", "MemoryConfig"]


@dataclass
class MemoryConfig:
    # scoring weights (Generative Agents: relevance / recency / importance)
    w_relevance: float = 0.65
    w_recency: float = 0.20
    w_importance: float = 0.15
    recency_halflife_hours: float = 168.0  # one week
    # retrieval
    default_k: int = 5
    # consolidation
    semantic_similarity_threshold: float = 0.88
    enable_consolidation: bool = True
    # embedder: "" -> HashingEmbedder; or point openai_* fields at any
    # OpenAI-compatible /v1/embeddings endpoint (vLLM, Ollama, Jina, ...)
    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key: str = ""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _recency(created_iso: str, now: datetime, halflife_hours: float) -> float:
    dt = (now - parse_ts(created_iso)).total_seconds() / 3600.0
    return float(0.5 ** (max(dt, 0.0) / halflife_hours))


class Memory:
    def __init__(
        self,
        path: str | Path = ":memory:",
        config: MemoryConfig | None = None,
        caption_fn: Optional[ImageCaptionFn] = None,
    ):
        self.config = config or MemoryConfig()
        self.embedder: EmbeddingProvider = self._build_embedder()
        p = Path(path)
        # Accept either a directory (-> dir/memory.db) or an explicit .db
        # file path; ":memory:" keeps the in-memory behaviour end to end.
        if str(path) == ":memory:":
            self.path = ":memory:"
            self.root = None
        elif p.suffix == ".db" or p.suffix == ".sqlite":
            self.path = str(p)
            self.root = p.parent
        else:
            self.path = str(p / "memory.db")
            self.root = p
        self.store = MemoryStore(self.path)
        self.media = None if self.root is None else MediaStore(self.root)
        self.caption_fn = caption_fn

    # ------------------------------------------------------------------ setup
    def _build_embedder(self) -> EmbeddingProvider:
        cfg = self.config
        if cfg.openai_base_url and cfg.openai_model:
            from .encode import OpenAICompatEmbedder

            return OpenAICompatEmbedder(
                cfg.openai_base_url, cfg.openai_model, cfg.openai_api_key
            )
        return HashingEmbedder()

    # ------------------------------------------------------------------ write
    def add_text(
        self,
        content: str,
        kind: str = EPISODIC,
        importance: float = 0.5,
        tags: list[str] | None = None,
        modality: str = MOD_TEXT,
        meta: dict | None = None,
        created_at: str | None = None,
        item_id: str | None = None,
    ) -> MemoryItem:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        item = MemoryItem(
            kind=kind,
            content=content,
            modality=modality,
            tags=list(tags or []),
            importance=importance,
            created_at=created_at or now_iso(),
            meta=dict(meta or {}),
        )
        if item_id:
            item.id = item_id
        vec = self.embedder.embed([self._embed_text_of(item)])[0]
        self.store.add(item, vec, self.embedder.name)
        return item

    def add_image(
        self,
        path: str | Path,
        caption: str | None = None,
        kind: str = EPISODIC,
        importance: float = 0.5,
        tags: list[str] | None = None,
        meta: dict | None = None,
    ) -> MemoryItem:
        """Add an image memory; the caption becomes the searchable body.

        Bytes are copied into the store's media dir (except :memory: mode).
        Plug a real VLM in via ``caption_fn`` for meaningful captions, or a
        CLIP-style embedder for true cross-modal vectors (DESIGN.md section 4).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if self.media is None:
            stored_path = path
        else:
            stored_path = self.media.save(path)
        if caption is None:
            if self.caption_fn is not None:
                caption = self.caption_fn(stored_path)
            else:
                from .media import default_caption

                caption = default_caption(stored_path)
        att = Attachment(
            media_type=guess_media_type(stored_path),
            path=str(stored_path),
            caption=caption,
        )
        item = MemoryItem(
            kind=kind,
            content=caption,
            modality=MOD_IMAGE,
            tags=list(tags or []),
            importance=importance,
            created_at=now_iso(),
            attachments=[att],
            meta=dict(meta or {}),
        )
        vec = self.embedder.embed([self._embed_text_of(item)])[0]
        self.store.add(item, vec, self.embedder.name)
        return item

    def add_video(
        self,
        path: str | Path,
        max_frames: int = 8,
        caption_fn: Optional[ImageCaptionFn] = None,
        tags: list[str] | None = None,
    ) -> list[MemoryItem]:
        """Sample keyframes from a video; each becomes one image memory that
        references the source file via meta["video"]. Requires ffmpeg on PATH.
        Falls back to a single text-only item if ffmpeg is unavailable.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        from .media import sample_video_keyframes

        try:
            frames = sample_video_keyframes(
                path, Path(self.path) / "frames" if self.path != ":memory:" else Path("/tmp/multimem_frames"),
                max_frames=max_frames,
            )
        except (FileNotFoundError, RuntimeError):
            return [
                self.add_text(
                    f"video (unsampled): {path.name}",
                    kind=EPISODIC,
                    modality=MOD_VIDEO,
                    tags=list(tags or []),
                    meta={"video": str(path)},
                )
            ]
        items: list[MemoryItem] = []
        for frame in frames:
            cap_fn = caption_fn or self.caption_fn
            items.append(
                self.add_image(
                    frame,
                    caption=None if cap_fn is None else cap_fn(frame),
                    tags=list(tags or []),
                    meta={"video": str(path)},
                )
            )
        return items

    # ------------------------------------------------------------------- read
    def _embed_text_of(self, item: MemoryItem) -> str:
        """Text that gets embedded: body + captions of attachments."""
        parts = [item.content]
        for att in item.attachments:
            if att.caption:
                parts.append(att.caption)
        return "\n".join(parts)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self.store.get(item_id)

    def search(
        self,
        query: str,
        k: int | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        valid_only: bool = True,
    ) -> list[SearchHit]:
        """Hybrid search: cosine similarity over stored vectors, blended with
        recency and importance. Invalidated semantic facts are excluded by
        default (bi-temporal filtering, cf. Zep/Graphiti).
        """
        cfg = self.config
        k = k or cfg.default_k
        qvec = self.embedder.embed([query])[0]
        rows, mat = self.store.rows_with_vectors(self.embedder.name)
        if not rows:
            return []
        if kinds:
            allowed = set(kinds)
            keep = [i for i, r in enumerate(rows) if r["kind"] in allowed]
            rows = [rows[i] for i in keep]
            mat = mat[keep]
        if not rows:
            return []
        now = _utcnow()
        sims = (mat @ qvec) / (
            np.linalg.norm(mat, axis=1) * 1
        )
        hits: list[SearchHit] = []
        for i, row in enumerate(rows):
            if valid_only and row["invalid_at"]:
                continue
            if tags and not (set(tags) & set(t for t in (row["tags"] or "").split(",") if t)):
                continue
            norm = np.linalg.norm(mat[i])
            rel = float(sims[i] / norm) if norm > 0 else 0.0
            rec = _recency(row["created_at"], now, cfg.recency_halflife_hours)
            imp = float(row["importance"])
            score = cfg.w_relevance * rel + cfg.w_recency * rec + cfg.w_importance * imp
            hits.append(
                SearchHit(
                    item=MemoryItem.from_row(row),
                    score=score,
                    parts={"relevance": rel, "recency": rec, "importance": imp},
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def export_context(self, query: str, k: int | None = None, budget_chars: int = 4000) -> str:
        """Render top hits as an LLM-ready context block (memory-in-prompt)."""
        hits = self.search(query, k=k or self.config.default_k)
        lines: list[str] = []
        used = 0
        for h in hits:
            line = h.brief()
            if used + len(line) > budget_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    # ---------------------------------------------------------- consolidation
    def consolidate_semantic(self, min_cluster: int = 2, dry_run: bool = False) -> list[dict]:
        """Naive semantic distillation: cluster near-duplicate episodic items
        by embedding similarity; each cluster of >= min_cluster becomes a
        semantic fact (count-annotated), originals are kept (episodic is the
        audit log). Returns the actions taken.
        """
        thr = self.config.semantic_similarity_threshold
        rows, mat = self.store.rows_with_vectors(self.embedder.name)
        epis = [i for i, r in enumerate(rows) if r["kind"] == EPISODIC]
        if len(epis) < min_cluster:
            return []
        norms = np.linalg.norm(mat[epis], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = mat[epis] / norms
        sim = emb @ emb.T
        assigned: set[int] = set()
        actions: list[dict] = []
        for a in range(len(epis)):
            if a in assigned:
                continue
            cluster = [b for b in range(a, len(epis)) if sim[a][b] >= thr and b not in assigned]
            if len(cluster) < min_cluster:
                continue
            assigned.update(cluster)
            members = [rows[epis[b]] for b in cluster]
            # representative = most important member's content
            rep = max(members, key=lambda r: r["importance"])
            fact = f"{len(members)}x recurring episode: {rep['content']}"
            action = {"fact": fact, "sources": [r["id"] for r in members]}
            actions.append(action)
            if not dry_run:
                self.add_text(
                    fact,
                    kind=SEMANTIC,
                    importance=min(1.0, rep["importance"] + 0.1),
                    meta={"consolidated_from": action["sources"]},
                )
        return actions

    # ------------------------------------------------------------------ forget
    def forget(self, item_id: str, hard: bool = False) -> None:
        """Soft-forget (invalidate, bi-temporal) or hard-delete an item."""
        if hard:
            self.store.delete(item_id)
        else:
            self.store.invalidate(item_id, now_iso())

    def decay(self, floor_importance: float = 0.1, drop_below: float = 0.05) -> int:
        """Time-based decay pass: lower importance of old items; hard-delete
        those that fall below ``drop_below``. Returns number affected.
        """
        rows = self.store.iter_rows()
        now = _utcnow()
        n = 0
        for row in rows:
            rec = _recency(row["created_at"], now, self.config.recency_halflife_hours)
            new_imp = max(floor_importance, row["importance"] * (0.5 + 0.5 * rec))
            if row["importance"] <= drop_below and new_imp <= drop_below:
                self.store.delete(row["id"])
            elif abs(new_imp - row["importance"]) > 1e-9:
                self.store.conn.execute(
                    "UPDATE memories SET importance=? WHERE id=?", (new_imp, row["id"])
                )
                n += 1
        self.store.conn.commit()
        return n

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        rows = self.store.iter_rows()
        by_kind: dict[str, int] = {}
        by_mod: dict[str, int] = {}
        for r in rows:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
            by_mod[r["modality"]] = by_mod.get(r["modality"], 0) + 1
        return {"total": len(rows), "by_kind": by_kind, "by_modality": by_mod}
