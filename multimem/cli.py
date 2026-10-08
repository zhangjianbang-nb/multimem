"""Command-line interface.

    multimem init ./agent_mem
    multimem add ./agent_mem --text "user prefers dark mode" --kind semantic
    multimem add ./agent_mem --image shot.png --caption "dashboard"
    multimem search ./agent_mem "what theme?" -k 3
    multimem stats ./agent_mem
    multimem consolidate ./agent_mem --dry-run
    multimem serve ./agent_mem        # MCP stdio server
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _open(path: str):
    from .memory import Memory

    return Memory(Path(path) if path != ":memory:" else ":memory:")


def cmd_add(args) -> int:
    mem = _open(args.path)
    if args.text:
        item = mem.add_text(
            args.text, kind=args.kind, importance=args.importance,
            tags=args.tags.split(",") if args.tags else None,
        )
        print(json.dumps({"id": item.id, "kind": item.kind}))
    elif args.image:
        item = mem.add_image(
            args.image, caption=args.caption, kind=args.kind,
            importance=args.importance,
            tags=args.tags.split(",") if args.tags else None,
        )
        print(json.dumps({"id": item.id, "kind": item.kind, "modality": item.modality}))
    elif args.video:
        items = mem.add_video(args.video, max_frames=args.max_frames)
        print(json.dumps({"count": len(items)}))
    else:
        print("nothing to add: use --text/--image/--video", file=sys.stderr)
        return 2
    return 0


def cmd_search(args) -> int:
    mem = _open(args.path)
    hits = mem.search(args.query, k=args.k, kinds=args.kinds.split(",") if args.kinds else None)
    for h in hits:
        print(h.brief())
    return 0


def cmd_stats(args) -> int:
    mem = _open(args.path)
    print(json.dumps(mem.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_consolidate(args) -> int:
    mem = _open(args.path)
    actions = mem.consolidate_semantic(dry_run=args.dry_run)
    print(json.dumps(actions, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args) -> int:
    from .mcp_server import serve

    serve(args.path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="multimem", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("add", help="add a memory")
    sp.add_argument("path")
    sp.add_argument("--text")
    sp.add_argument("--image")
    sp.add_argument("--video")
    sp.add_argument("--caption")
    sp.add_argument("--kind", default="episodic", choices=["episodic", "semantic", "procedural"])
    sp.add_argument("--importance", type=float, default=0.5)
    sp.add_argument("--tags", default="")
    sp.add_argument("--max-frames", type=int, default=8)
    sp.set_defaults(fn=cmd_add)

    sp = sub.add_parser("search", help="search memories")
    sp.add_argument("path")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5)
    sp.add_argument("--kinds", default="")
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("stats", help="memory statistics")
    sp.add_argument("path")
    sp.set_defaults(fn=cmd_stats)

    sp = sub.add_parser("consolidate", help="distill episodic -> semantic")
    sp.add_argument("path")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=cmd_consolidate)

    sp = sub.add_parser("serve", help="run MCP stdio server")
    sp.add_argument("path")
    sp.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
