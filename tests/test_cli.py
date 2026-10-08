"""CLI end-to-end tests (subprocess, real files)."""
import json
import subprocess
import sys

import pytest

REPO = "/home/whu/workspace/ai_workspace/multimem"


def _cli(*args):
    r = subprocess.run(
        [sys.executable, "-m", "multimem.cli"] + list(args),
        capture_output=True, text=True, cwd=REPO, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    return r


def test_cli_add_text_search_stats(tmp_path):
    db = tmp_path / "store"
    r = _cli("add", str(db), "--text", "user prefers dark mode", "--kind", "semantic")
    payload = json.loads(r.stdout)
    assert payload["kind"] == "semantic"
    r = _cli("search", str(db) if False else db, "theme preference", "-k", "2")
    assert "dark mode" in r.stdout
    r = _cli("stats", str(db) if False else db)
    stats = json.loads(r.stdout)
    assert stats["total"] == 1


def test_cli_add_image(tmp_path):
    from PIL import Image

    db = tmp_path / "store2"
    img = tmp_path / "shot.png"
    Image.new("RGB", (10, 10), (255, 0, 0)).save(img)
    r = _cli("add", db, "--image", str(img), "--caption", "red square")
    payload = json.loads(r.stdout)
    assert payload["modality"] == "image"
    r = _cli("search", db, "red square", "-k", "1")
    assert "red square" in r.stdout


def test_cli_consolidate_dry_run(tmp_path):
    db = tmp_path / "store3"
    _cli("add", db, "--text", "user likes dark mode", "--kind", "episodic")
    _cli("add", db, "--text", "user likes dark mode", "--kind", "episodic")
    r = _cli("consolidate", db, "--dry-run")
    assert "recurring" in r.stdout


def test_cli_requires_something_to_add(tmp_path):
    with pytest.raises(AssertionError):
        _cli("add", tmp_path / "s4")
