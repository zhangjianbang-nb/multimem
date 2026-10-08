# MultiMem

**Multimodal agent memory, batteries included — and dependency-light.**

[![tests](https://img.shields.io/badge/tests-45%20passing-brightgreen)]() [![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![license](https://img.shields.io/badge/license-MIT-informational)]()

MultiMem gives a multimodal agent a persistent memory: **episodic / semantic /
procedural** items over **text and images (video frames)**, with hybrid
retrieval, auditable consolidation, bi-temporal forgetting — and a
**dependency-free MCP server** so any agent can use it as a tool.

Design decisions are grounded in a survey of the 2023–2026 memory landscape
(MemGPT/Letta, Mem0, Graphiti/Zep, A-MEM, MemOS, MIRIX, M3-Agent, HippoRAG,
LangMem, Cognee + 2026 multicue work). The full research notes ship in
[`docs/research/`](docs/research/) and the rationale in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Why another memory library

| | mem0 | letta | graphiti | MIRIX | **multimem** |
|---|---|---|---|---|---|
| multimodal items | text-facts only | text | text | text+vision, heavy | text + image/video-frame items |
| bi-temporal forgetting | ✗ | ✗ | ✓ | partial | ✓ (`valid_at/invalid_at`, soft-forget) |
| auditable consolidation | opaque | opaque | ✗ | `dry_run` ✓ | ✓ (`consolidate_semantic(dry_run=True)`) |
| install weight | SDK+pydantic stack | SDK | SDK | heavy | **numpy + pillow only** |
| MCP server | via SDK | built-in | ✗ | ✗ | **zero-SDK stdio server** |

## Quickstart

```bash
pip install -e .            # or: uv pip install -e .
python examples/demo.py     # end-to-end tour, exits 0 on success
```

```python
from multimem import Memory, SEMANTIC

mem = Memory("./agent_mem")            # or Memory(":memory:")

mem.add_text("User prefers dark mode", kind=SEMANTIC, importance=0.9)
mem.add_image("dashboard.png", caption="ops dashboard, red error banner")
mem.add_video("meeting.mp4", max_frames=8)   # keyframes → image memories

hits = mem.search("what theme does the user like?", k=3)
print(hits[0].brief())

ctx = mem.export_context("user preferences", budget_chars=2000)  # prompt-ready
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
```

- invalid facts (soft-forgotten) are excluded by default (bi-temporal filter)
- `kinds`/`tags` prune before the vector scan
- embedder is pluggable: offline `HashingEmbedder` by default (deterministic,
  no network), or any OpenAI-compatible `/v1/embeddings` endpoint
  (`MemoryConfig(openai_base_url=..., openai_model=...)`); for true
  text↔image shared space, plug a CLIP/SigLIP/jina-clip-v2 provider.

## Storage

SQLite (WAL) + float32 vectors in one file; numpy brute-force cosine
(<50 ms at 10^6×256). One directory per store:

```
./agent_mem/
├── memory.db          # items + embeddings
├── media/             # attachment bytes (dedup by sha1)
└── frames/            # sampled video keyframes
```

## License

MIT. Research notes in `docs/research/` inherit the citations of their sources.
