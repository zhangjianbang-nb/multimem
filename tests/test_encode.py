"""Embedder tests."""
import numpy as np
import pytest

from multimem.encode import HashingEmbedder, _tokenize, cosine


def test_deterministic():
    e = HashingEmbedder(dim=64)
    a = e.embed(["hello world"])
    b = e.embed(["hello world"])
    assert np.allclose(a, b)


def test_normalized():
    e = HashingEmbedder(dim=64)
    vec = e.embed(["some text here"])[0]
    n = float(np.linalg.norm(vec))
    assert abs(n - 1.0) < 1e-5


def test_similarity_ordering():
    e = HashingEmbedder(dim=128)
    va = e.embed(["user prefers dark mode"])[0]
    vb = e.embed(["user likes dark theme"])[0]
    vd = e.embed(["the cat sat on the mat"])[0]
    assert cosine(va, vb) > cosine(va, vd)


def test_cosine_zero_vector_safe():
    z = np.zeros(4)
    assert cosine(z, z) == 0.0


def test_tokenizer_cjk_bigram():
    toks = _tokenize("我喜欢黑暗模式")
    assert "喜欢" in toks