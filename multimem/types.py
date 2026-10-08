"""Core data model: memory items, attachments, search hits.

A memory item is a single unit of remembered information. Three kinds are
supported (mirroring the taxonomy used by MIRIX / MemGPT-style systems):

- ``episodic``: an event that happened at a point in time ("user uploaded a
  photo of a whiteboard at 3pm"), possibly carrying multimodal attachments.
- ``semantic``: a distilled fact that holds over a time interval ("user
  prefers dark mode"), produced by extraction or consolidation.
- ``procedural``: a reusable recipe ("how to summarizing screenshots of
  dashboards: first OCR the header...").
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

EPISODIC = "episodic"
SEMANTIC = "semantic"
PROCEDURAL = "procedural"
KINDS = (EPISODIC, SEMANTIC, PROCEDURAL)

MOD_TEXT = "text"
MOD_IMAGE = "image"
MOD_VIDEO = "video"
MOD_AUDIO = "audio"
MOD_MULTI = "multimodal"
MODALITIES = (MOD_TEXT, MOD_IMAGE, MOD_VIDEO, MOD_AUDIO, MOD_MULTI)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return utcnow().isoformat()


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass
class Attachment:
    """A binary media object referenced by a memory item.

    The bytes themselves stay on disk (or are inlined as base64 for tiny
    payloads); we only keep the path plus a caption used for retrieval.
    """

    media_type: str  # RFC-2045 type, e.g. image/png, video/mp4
    path: Optional[str] = None
    caption: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Attachment":
        return cls(
            media_type=raw["media_type"],
            path=raw.get("path"),
            caption=raw.get("caption", ""),
            meta=raw.get("meta") or {},
        )


@dataclass
class MemoryItem:
    kind: str  # one of KINDS
    content: str  # natural-language body, always searchable
    modality: str = MOD_TEXT  # primary modality
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0..1
    use_count: int = 0  # retrieval feedback (v0.2); shapes decay resistance
    created_at: str = field(default_factory=now_iso)
    # validity interval for semantic facts (bi-temporal, cf. Zep/Graphiti)
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    attachments: list[Attachment] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    # -- serialization -----------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "modality": self.modality,
            "tags": ",".join(self.tags),
            "importance": self.importance,
            "use_count": self.use_count,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "attachments_json": __import__("json").dumps(
                [a.to_dict() for a in self.attachments], ensure_ascii=False
            ),
            "meta_json": __import__("json").dumps(self.meta, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any], embedding=None) -> "MemoryItem":
        import json

        return cls(
            kind=row["kind"],
            content=row["content"],
            modality=row["modality"],
            tags=[t for t in (row["tags"] or "").split(",") if t],
            importance=row["importance"],
            use_count=int(row.get("use_count") or 0),
            created_at=row["created_at"],
            valid_at=row["valid_at"],
            invalid_at=row["invalid_at"],
            attachments=[
                Attachment.from_dict(a) for a in json.loads(row["attachments_json"] or "[]")
            ],
            meta=json.loads(row["meta_json"] or "{}"),
            id=row["id"],
        )


@dataclass
class SearchHit:
    item: MemoryItem
    score: float
    parts: dict[str, float]  # score decomposition: relevance/recency/importance

    def brief(self) -> str:
        head = self.item.content[:80].replace("\n", " ")
        return f"[{self.item.kind}/{self.item.modality} {self.item.created_at[:10]}] {head} (score={self.score:.3f})"
