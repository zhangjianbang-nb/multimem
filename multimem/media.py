"""Multimodal helpers: attachment ingestion, captioning hook, frame sampling.

Design (see docs/DESIGN.md): every memory keeps natural-language text as the
canonical searchable body; binaries live on disk referenced by path. Images
contribute (a) their caption text and (b) optionally an image embedding.
Videos are sampled into keyframes; each keyframe becomes one episodic item
linked back to the source video.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
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
