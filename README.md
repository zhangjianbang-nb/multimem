# MultiMem

**Multimodal agent memory, batteries included — and dependency-light.**

[![tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)]() [![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![license](https://img.shields.io/badge/license-MIT-informational)]()

MultiMem gives a multimodal agent a persistent memory: **episodic / semantic /
procedural** items over **text and images (video frames)**, with hybrid
retrieval, **true cross-modal search** (find a picture by what is in it, even with a poor
caption), auditable consolidation, bi-temporal
forgetting — and a **dependency-free MCP server** so any agent can use it as
a tool.

Design decisions are grounded in a survey of the 2023–2026 memory landscape
(MemGPT/Letta, Mem0, Graphiti/Zep, A-MEM, MemOS, MIRIX, M3-Agent, HippoRAG,
LangMem, Cognee + 2026 video-memory work). The full research notes ship in
[`docs/research/`](docs/research/) and the rationale in
[`docs/DESIGN.md`](docs/DESIGN.md) (v0.1) / [`docs/DESIGN-v2.md`](docs/DESIGN-v2.md) (v0.2).

## Why this memory library

| | mem0 | letta | graphiti | MIRIX | **multimem** |
|---|---|---|---|---|---|
| multimodal items | text-facts only | text | text | text+vision, heavy | text + image/video-frame items |
| cross-modal retrieval | ✗ | ✗ | ✗ | partial | ✓ (paired CLIP towers, caption+vision fusion) |
| video timestamp index | ✗ | ✗ | ✗ | partial | ✓ (`meta["t_seconds"]` + `video_within=(a,b)`) |
| bi-temporal forgetting | ✗ | ✗ | ✓ | partial | ✓ (`valid_at/invalid_at`, soft-forget) |
| auditable consolidation | opaque | opaque | ✗ | `dry_run` ✓ | ✓ (`consolidate_semantic(dry_run=True)`) |
| retrieval feedback loop | ✗ | ✗ | ✗ | ✗ | ✓ (`use_count` shapes decay & importance) |
| install weight | SDK+pydantic stack | SDK | SDK | heavy | **numpy + pillow only** |
| MCP server | via SDK | built-in | ✗ | ✗ | **zero-SDK stdio server** |

## Quickstart

```bash
pip install -e .            # or: uv pip install -e .
python examples/demo.py     # v0.1 tour
python examples/demo_v2.py  # v0.2 tour: cross-modal + timestamps + feedback
```

```python
from multimem import Memory, MemoryConfig, SEMANTIC
from multimem.encode import ClipStyleEmbedder

mem = Memory("./agent_mem")            # or Memory(":memory:")

mem.add_text("User prefers dark mode", kind=SEMANTIC, importance=0.9)
mem.add_image("dashboard.png", caption="ops dashboard, red error banner")
mem.add_video("meeting.mp4", max_frames=8)   # keyframes carry t_seconds

hits = mem.search("what theme does the user like?", k=3)
print(hits[0].brief())

# "what was shown in the video between 2 and 5 minutes?"
hits = mem.search("error banner", video_within=(120, 300))

# true cross-modal: plug any CLIP/SigLIP/CN-CLIP checkpoint (optional dep)
clip = ClipStyleEmbedder("google/siglip2-base-patch16-256")
mem2 = Memory("./agent_mem2", clip=clip)     # images get vision vectors too
```

CLI:

```bash
multimem add ./mem --text "prefers dark mode" --kind semantic
multimem add ./mem --image shot.png --caption "dashboard"
multimem search ./mem "theme?" -k 3
multimem consolidate ./mem --dry-run
multimem stats ./mem
```

MCP server (Claude Desktop / any MCP client):

```bash
multimem serve ./mem
```

Exposes `add_memory`, `search_memory`, `list_memories`, `forget_memory`
over stdio JSON-RPC — no MCP SDK involved.

## Retrieval model

```
score = 0.65·relevance + 0.20·recency + 0.15·importance   (recency half-life 1 week)

relevance = max( cos(query, caption_vector),                 # text tower
                 cos(query, vision_vector) )                 # vision tower, if paired clip is set
```

- invalid facts (soft-forgotten) are excluded by default (bi-temporal filter)
- `kinds`/`tags`/`video_within` prune before the vector scan
- retrieval feedback (`MemoryConfig(feedback=True)`) counts hits into
  `use_count`, nudges importance, and slows decay of frequently-used memories
- embedder is pluggable: offline `HashingEmbedder` by default (deterministic,
  no network), or any OpenAI-compatible `/v1/embeddings` endpoint
  (`MemoryConfig(openai_base_url=..., openai_model=...)`); for a shared
  text↔image space use the paired `ClipStyleEmbedder` (lazy-imports
  transformers; SigLIP2 = Apache-2.0, Chinese-CLIP = MIT — see
  `docs/research/05-embedding-index-2026.md` for the license-checked survey)

## Storage

SQLite (WAL) + float32 vectors in one file; numpy brute-force cosine
(<50 ms at 10^6×256). One directory per store:

```
./agent_mem/
├── memory.db          # items + embeddings (text & vision spaces)
├── media/             # attachment bytes (dedup by sha1)
└── frames/            # sampled video keyframes (+ timestamps in meta)
```

Databases created by v0.1 are migrated in place on open (new `use_count`
column, embeddings keyed by space).

## License

MIT. Research notes in `docs/research/` inherit the citations of their sources.
