"""Embedding providers.

Two built-ins:

- :class:`HashingEmbedder` — deterministic, dependency-free, offline. Hashes
  tokens (with CJK bigram support) into a fixed-dim bag vector. Good for
  tests, demos and as a graceful default when no model is reachable.
- :class:`OpenAICompatEmbedder` — calls any OpenAI-compatible
  ``POST /v1/embeddings`` endpoint (OpenAI, vLLM, Ollama, Jina, ...).

Multimodal note: with a plain text embedder, images are embedded through
their caption text. Plug a CLIP/SigLIP/Jina-CLIP-v2 provider here to place
image vectors into the same space as text queries (see docs/DESIGN.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class EmbeddingProvider(Protocol):
    dim: int
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class ImageEmbeddingProvider(Protocol):
    """Vision-tower provider: maps image paths to vectors.

    A CLIP/SigLIP-style model implements BOTH protocols under the same
    ``name``; that shared name is what lets the retrieval layer fuse
    text-space and vision-space vectors (DESIGN-v2 section 1).
    """

    dim: int
    name: str

    def embed_images(self, paths: Sequence[str]) -> np.ndarray: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    # group consecutive CJK chars into bigrams for better matching
    out: list[str] = []
    buf: list[str] = []
    for tok in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]", tok):
            buf.append(tok)
        else:
            if len(buf) == 1:
                out.append(buf[0])
            elif len(buf) >= 2:
                out.extend(buf[i] + buf[i + 1] for i in range(len(buf) - 1))
            buf = []
            out.append(tok)
    if len(buf) == 1:
        out.append(buf[0])
    elif len(buf) >= 2:
        out.extend(buf[i] + buf[i + 1] for i in range(len(buf) - 1))
    return out


class HashingEmbedder:
    """Deterministic feature-hashing embedder (offline default)."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokenize(text or ""):
                idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % self.dim
                vecs[i, idx] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class OpenAICompatEmbedder:
    """Any endpoint implementing POST {base_url}/embeddings."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"openai:{model}"
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            vec = self.embed(["warmup"])[0]
            self._dim = int(vec.shape[0])
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        body = json.dumps({"input": list(texts), "model": self.model}).encode()
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        rows = sorted(data["data"], key=lambda d: d["index"])
        return np.array([r["embedding"] for r in rows], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / na / nb)


class ClipStyleEmbedder:
    """Paired text+vision embedder for true cross-modal retrieval.

    Wraps any local CLIP/SigLIP/CN-CLIP checkpoint. The heavy dependency
    (transformers + torch, or onnxruntime) is imported lazily inside
    :meth:`embed` / :meth:`embed_images`, so ``import multimem`` never pulls
    them in. ``model_id`` is any HF id or local path, e.g.
    ``google/siglip2-base-patch16-256`` or ``OFA-Sys/chinese-clip-vit-base-patch16``.

    Text and image vectors live in the SAME space only when both towers come
    from the same checkpoint — hence one class, one shared ``name``.
    """

    def __init__(self, model_id: str, device: str = "cpu", name: str = ""):
        self.model_id = model_id
        self.device = device
        self.name = name or f"clip:{model_id}"
        self._dim: int | None = None
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError as e:  # pragma: no cover - env without transformers
                raise RuntimeError(
                    "ClipStyleEmbedder requires the optional dependency "
                    "'transformers' (pip install transformers torch). "
                    "Multimem itself stays dependency-free."
                ) from e
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._model, self._processor

    @property
    def dim(self) -> int:
        if self._dim is None:
            probe = self.embed(["dim probe"])
            self._dim = int(probe.shape[-1])
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model, processor = self._ensure_loaded()
        inputs = processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        return _l2(feats.cpu().numpy())

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        from PIL import Image

        import torch

        model, processor = self._ensure_loaded()
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        return _l2(feats.cpu().numpy())


class SyntheticClipEmbedder:
    """Deterministic paired embedder for tests/demos (no network, no weights).

    Hashes tokens into a text vector; images map through their path-name
    tokens so a synthetic text query that shares words with an image's file
    name hits its *vision* vector too. Not semantically meaningful — it only
    for exercising the dual-space plumbing end to end.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.name = f"synthetic-clip-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text or ""):
            idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % self.dim
            vec[idx] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(Path(p).stem) for p in paths])


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)
