"""Tests for QJL-backed latent index."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex


def test_index_empty_search_returns_empty() -> None:
    idx = LatentIndex(IndexConfig(in_dim=32, bits=128))
    q = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    assert idx.search(q, k=4) == []


def test_index_add_and_search_topk() -> None:
    idx = LatentIndex(IndexConfig(in_dim=64, bits=512, seed=0))
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((50, 64)).astype(np.float32)
    payloads = [f"sol-{i}" for i in range(50)]
    idx.add(latents, payloads)
    assert len(idx) == 50

    # Query one of the cached vectors → should retrieve itself first.
    target = 17
    results = idx.search(latents[target], k=5)
    assert len(results) == 5
    sims = [s for s, _ in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0][1] == "sol-17"
    assert results[0][0] == pytest.approx(1.0)


def test_index_add_payload_count_mismatch() -> None:
    idx = LatentIndex(IndexConfig(in_dim=16))
    with pytest.raises(ValueError):
        idx.add(np.zeros((3, 16), dtype=np.float32), ["a", "b"])


def test_index_invalid_k() -> None:
    idx = LatentIndex(IndexConfig(in_dim=16))
    idx.add(np.zeros((1, 16), dtype=np.float32), ["x"])
    with pytest.raises(ValueError):
        idx.search(np.zeros(16, dtype=np.float32), k=0)


def test_index_two_batches_concatenate() -> None:
    idx = LatentIndex(IndexConfig(in_dim=16, bits=256, seed=1))
    rng = np.random.default_rng(0)
    a = rng.standard_normal((10, 16)).astype(np.float32)
    b = rng.standard_normal((5, 16)).astype(np.float32)
    idx.add(a, [f"a{i}" for i in range(10)])
    idx.add(b, [f"b{i}" for i in range(5)])
    assert len(idx) == 15
