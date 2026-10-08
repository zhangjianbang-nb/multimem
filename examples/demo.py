#!/usr/bin/env python3
"""End-to-end demo: build a small multimodal memory, query it, consolidate.

    python examples/demo.py

Uses only the offline default embedder — no API keys, no network.
Exit code 0 on success; any failed stage raises.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from multimem import Memory, SEMANTIC


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="multimem-demo-"))
    print(f"[demo] store at {root}")
    mem = Memory(root)

    # 1. text memories across kinds
    mem.add_text("User prefers dark mode in all IDEs", kind=SEMANTIC, importance=0.9)
    mem.add_text("Deployed v2.3 to staging at 3pm", importance=0.4)
    mem.add_text("User's dog is called Biscuit", kind=SEMANTIC, importance=0.6)

    # 2. an image memory (generated on the fly)
    img = root / "chart.png"
    Image.new("RGB", (120, 80), (30, 120, 200)).save(img)
    mem.add_image(img, caption="blue throughput chart from the ops dashboard")

    # 3. search across modalities
    hits = mem.search("what theme does the user like?", k=2)
    assert hits, "search returned nothing"
    assert hits[0].item.content.startswith("User prefers dark mode"), hits[0].brief()
    print(f"[demo] top hit: {hits[0].brief()}")

    hits = mem.search("ops dashboard picture", k=1)
    assert "throughput" in hits[0].item.content, hits[0].brief()
    print(f"[demo] image hit: {hits[0].brief()}")

    # 4. consolidation: recurring episodes -> semantic fact
    for _ in range(3):
        mem.add_text("user reviewed the latency dashboard this morning")
    actions = mem.consolidate_semantic(min_cluster=2)
    assert actions, "consolidation produced nothing"
    print(f"[demo] consolidated {len(actions)} fact(s): {actions[0]['fact'][:60]}")

    # 5. soft-forget keeps the audit trail
    # (locate the exact item with valid_only=False: with the offline hashing
    # embedder, relevance is sparse — rank by filtered search would be unstable)
    hits = mem.search("dog", k=5, valid_only=False)
    fact_id = next(h.item.id for h in hits if "dog" in h.item.content.lower())
    mem.forget(fact_id)
    assert mem.search("dog", k=5, valid_only=True) == [] or all(
        "dog" not in h.item.content.lower() for h in mem.search("dog", k=5, valid_only=True)
    )
    print("[demo] soft-forgotten item correctly hidden")

    print(f"[demo] stats: {mem.stats()}")
    shutil.rmtree(root)
    print("[demo] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
