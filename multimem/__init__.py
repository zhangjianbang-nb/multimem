"""multimem — multimodal agent memory: episodic + semantic memory over
text / images / video frames with hybrid retrieval, consolidation and a
dependency-free MCP server."""

from .memory import Memory, MemoryConfig
from .store import MemoryStore
from .types import (
    EPISODIC,
    KINDS,
    MODALITIES,
    MOD_AUDIO,
    MOD_IMAGE,
    MOD_MULTI,
    MOD_TEXT,
    MOD_VIDEO,
    PROCEDURAL,
    SEMANTIC,
    Attachment,
    MemoryItem,
    SearchHit,
)

__version__ = "0.2.0"

__all__ = [
    "Memory",
    "MemoryConfig",
    "MemoryStore",
    "MemoryItem",
    "Attachment",
    "SearchHit",
    "EPISODIC",
    "SEMANTIC",
    "PROCEDURAL",
    "MOD_TEXT",
    "MOD_IMAGE",
    "MOD_VIDEO",
    "MOD_AUDIO",
    "MOD_MULTI",
    "MODALITIES",
    "KINDS",
    "__version__",
]
