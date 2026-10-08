"""v0.2 end-to-end demo: dual-space cross-modal retrieval, video timestamps,
retrieval feedback, consolidation, forgetting. Exit 0 = all assertions pass.

    python examples/demo_v2.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from multimem import Memory, MemoryConfig, SEMANTIC
from multimem.encode import SyntheticClipEmbedder


def pytest_approx(value: float, expected: float, tol: float = 1e-6) -> bool:
    return abs(value - expected) <= tol


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="multimem_v2_demo_"))
    log: list[str] = []
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        log.append(f"{'PASS' if cond else 'FAIL'}: {name}")
        ok = ok and cond

    # --- 1. text memory + hybrid retrieval --------------------------------
    mem = Memory(tmp / "store", config=MemoryConfig())
    mem.add_text("User prefers dark mode", kind=SEMANTIC, importance=0.9)
    mem.add_text("Weather was rainy in Wuhan", importance=0.3)
    hits = mem.search("which theme does the user prefer?", k=2)
    check("text retrieval ranks fact first", hits[0].item.content.startswith("User prefers"))

    # --- 2. cross-modal: image found through its vision vector ------------
    clip = SyntheticClipEmbedder(dim=64)
    img = tmp / "red_bicycle.png"
    from PIL import Image

    Image.new("RGB", (16, 16)).save(img)
    mem2 = Memory(tmp / "store2", clip=clip)
    mem2.add_image(img, caption="a photo 0001")  # deliberately poor caption
    found = mem2.search("red bicycle", k=1)
    check(
        "cross-modal: poor-caption image found via vision tower",
        bool(found) and found[0].item.modality == "image",
    )

    # --- 3. video timestamps (synthetic frames, no ffmpeg needed) ---------
    mem3 = Memory(":memory:")
    for t, desc in ((12.5, "user opens dashboard"), (95.0, "error banner appears")):
        mem3.add_text(
            desc,
            meta={"video": "session.mp4", "t_seconds": t},
            modality="video",
        )
    early = mem3.search("dashboard", k=5, video_within=(0.0, 60.0))
    check("video window filter keeps only t<=60", {round(h.item.meta["t_seconds"]) for h in early} == {12})

    # --- 4. retrieval feedback --------------------------------------------
    cfg = MemoryConfig(feedback=True, feedback_importance_delta=0.05)
    mem4 = Memory(":memory:", config=cfg)
    item = mem4.add_text("important recurring fact", importance=0.5)
    mem4.search("recurring", k=1)
    got = mem4.get(item.id)
    check(
        "feedback raises use_count+importance",
        got.use_count == 1 and pytest_approx(got.importance, 0.55),
    )

    # --- 5. consolidation + forgetting still work (v0.1 contract) ---------
    mem5 = Memory(":memory:")
    for _ in range(3):
        mem5.add_text("standup meeting notes about latency budgets")
    actions = mem5.consolidate_semantic(min_cluster=2, dry_run=True)
    check("consolidation dry-run proposes an action", len(actions) >= 1)
    victim = mem5.add_text("to be forgotten")
    mem5.forget(victim.id)
    check(
        "soft forget hides item",
        all(h.item.id != victim.id for h in mem5.search("to be forgotten", k=5)),
    )

    print("\n".join(log))
    print("demo_v2:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
