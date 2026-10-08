"""Multimodal helpers: attachment ingestion, captioning hook, frame sampling.

Design (see docs/DESIGN-v2.md): every memory keeps natural-language text as the
canonical searchable body; binaries live on disk referenced by path. Images
contribute (a) their caption text and (b) optionally a vision embedding.
Videos are sampled into keyframes carrying timestamps; each keyframe becomes
one episodic item linked back to the source video.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .types import Attachment

ImageCaptionFn = Callable[[Path], str]
"""A captioner maps an image path to a text description (any VLM callable).

Default when no VLM is wired: derive a stable pseudo-caption from the file
name + metadata. Plug in Qwen-VL/InternVL/GPT-4o here in production.
"""


def default_caption(path: Path) -> str:
    """Offline fallback caption: filename words + dims via Pillow if readable."""
    words = path.stem.replace("-", " ").replace("_", " ")
    desc = f"image file {words}"
    try:
        from PIL import Image

        with Image.open(path) as im:
            desc += f" ({im.format}, {im.width}x{im.height})"
    except Exception:
        pass
    return desc


@dataclass
class IngestResult:
    attachment: Attachment
    text: str  # text actually embedded for retrieval (caption or fallback)


def guess_media_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


class MediaStore:
    """Manages where attachment bytes live: {root}/media/<xx>/<name>."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.media_dir = self.root / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def save(self, src: Path) -> Path:
        src = Path(src)
        mime = guess_media_type(src)
        prefix = {
            "image": "img",
            "video": "vid",
            "audio": "aud",
        }.get(mime.split("/")[0], "file")
        digest = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
        name = f"{prefix}-{digest}{src.suffix.lower() or ''}"
        dest = self.media_dir / name[:2] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        return dest

    def save_bytes(self, data: bytes, suffix: str, mime: str) -> Path:
        digest = hashlib.sha1(data).hexdigest()[:10]
        prefix = mime.split("/")[0]
        name = f"{prefix}-{digest}{suffix}"
        dest = self.media_dir / name[:2] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        return dest

    def to_data_uri(self, att: Attachment) -> str:
        if not att.path:
            raise ValueError("attachment has no path")
        data = Path(att.path).read_bytes()
        b64 = base64.b64encode(data).decode()
        return f"data:{att.media_type};base64,{b64}"


def ingest_image(
    media: MediaStore,
    path: Path,
    caption: Optional[str] = None,
    caption_fn: Optional[ImageCaptionFn] = None,
) -> IngestResult:
    """Copy an image into the store and build its Attachment + retrieval text."""
    path = Path(path)
    dest = media.save(path)
    if caption is None:
        caption = caption_fn(dest) if caption_fn else default_caption(dest)
    att = Attachment(
        media_type=guess_media_type(dest),
        path=dest,
        caption=caption,
    )
    return IngestResult(attachment=att, text=caption)


def sample_video_keyframes(
    source: Path,
    out_dir: Path,
    max_frames: int = 8,
) -> list[tuple[Path, float | None]]:
    """Sample keyframes from a video.

    Returns a list of ``(frame_path, t_seconds_or_None)``. Strategy
    (docs/research/03-multimodal.md section 1.2): scene-boundary sampling
    first (``select='gt(scene,0.3)'``) because cut points mark event
    boundaries better than even spacing; if no scene cuts are found, fall
    back to ffmpeg's ``thumbnail`` representative-frame filter. Timestamps
    are parsed from ffmpeg's ``showinfo`` log. Raises RuntimeError when
    ffmpeg is missing or the file cannot be decoded.
    """
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    def _extract(filter_graph: str, tag: str) -> list[tuple[Path, float | None]]:
        for old in out_dir.glob(f"{tag}-*.jpg"):
            old.unlink()
        output_pattern = out_dir / (tag + "-%04d.jpg")
        proc = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-nostats",
                "-i", source,
                "-vf", filter_graph,
                "-frames:v", max_frames,
                "-q:v", "3",
                output_pattern,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not any(out_dir.glob(f"{tag}-*.jpg")):
            raise RuntimeError(f"ffmpeg sampling failed: {proc.stderr[-400:]}")
        frames = sorted(out_dir.glob(f"{tag}-*.jpg"))
        # parse pts_time from showinfo lines: "[Parsed_showinfo_2 ...] n: 0 pts_time:3.7"
        times: dict[int, float] = {}
        for m in re.finditer(r"n:\s*(\d+)\s+.*?pts_time:([\d.]+)", proc.stderr):
            times[int(m.group(1))] = float(m.group(2))
        out: list[tuple[Path, float | None]] = []
        for i, f in enumerate(frames):
            out.append((f, times.get(i)))
        return out

    try:
        frames = _extract("select=gt(scene,0.3)", "scene")
        if len(frames) >= 2:
            return frames
    except RuntimeError:
        pass
    # fallback: ffmpeg's own representative frames (even-ish coverage)
    return _extract("thumbnail", "thumb")
