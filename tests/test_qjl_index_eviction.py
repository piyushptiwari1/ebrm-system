"""Tests for bounded LatentIndex with eviction (LRU + FIFO)."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex

D = 8


def _vec(x: float) -> np.ndarray:
    v = np.zeros(D, dtype=np.float32)
    v[0] = x
    return v


def test_unbounded_index_is_default() -> None:
    idx = LatentIndex(IndexConfig(in_dim=D))
    for i in range(50):
        evicted = idx.add(np.stack([_vec(float(i))]), [f"p{i}"])
        assert evicted == 0
    assert len(idx) == 50


def test_max_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_size"):
        IndexConfig(in_dim=D, max_size=0)


def test_unknown_evict_policy_rejected() -> None:
    with pytest.raises(ValueError, match="evict_policy"):
        IndexConfig(in_dim=D, max_size=4, evict_policy="random")


def test_fifo_evicts_oldest_inserts_first() -> None:
    idx = LatentIndex(IndexConfig(in_dim=D, max_size=3, evict_policy="fifo"))
    for i in range(5):
        idx.add(np.stack([_vec(float(i))]), [f"p{i}"])
    assert len(idx) == 3
    # Top-3 nearest to a tall query should now be p2,p3,p4 (the freshly added).
    hits = idx.search(_vec(10.0), k=3)
    payloads = {h[1] for h in hits}
    assert payloads == {"p2", "p3", "p4"}


def test_lru_keeps_recently_searched_entries() -> None:
    idx = LatentIndex(IndexConfig(in_dim=D, max_size=3, evict_policy="lru"))
    # Seed with three entries.
    for i in range(3):
        idx.add(np.stack([_vec(float(i))]), [f"p{i}"])
    # Touch p0 via search so it becomes most-recently-accessed.
    idx.search(_vec(0.0), k=1)
    # Now add a fourth entry — eviction should drop p1 (least-recent), not p0.
    evicted = idx.add(np.stack([_vec(99.0)]), ["p99"])
    assert evicted == 1
    payloads = {h[1] for h in idx.search(_vec(50.0), k=3)}
    assert "p1" not in payloads
    assert "p0" in payloads


def test_eviction_keeps_codes_payloads_and_timestamps_aligned() -> None:
    idx = LatentIndex(IndexConfig(in_dim=D, max_size=2, evict_policy="fifo"))
    idx.add(np.stack([_vec(0.0), _vec(1.0), _vec(2.0)]), ["p0", "p1", "p2"])
    assert len(idx) == 2
    # Internal arrays must remain aligned and searchable. The two survivors
    # must be the two newest (p1, p2). QJL is a *bit-quantized* projection
    # so we don't assert exact ranking — only that p0 was evicted.
    hits = idx.search(_vec(2.0), k=2)
    assert len(hits) == 2
    payloads = {h[1] for h in hits}
    assert payloads == {"p1", "p2"}


def test_search_returns_empty_for_empty_index() -> None:
    idx = LatentIndex(IndexConfig(in_dim=D, max_size=10))
    assert idx.search(_vec(0.0), k=4) == []


def test_eviction_is_correct_when_batch_overflows() -> None:
    """Adding a single batch larger than max_size must trim down to max_size."""
    idx = LatentIndex(IndexConfig(in_dim=D, max_size=3, evict_policy="fifo"))
    latents = np.stack([_vec(float(i)) for i in range(7)])
    payloads: list[object] = [f"p{i}" for i in range(7)]
    evicted = idx.add(latents, payloads)
    assert evicted == 4
    assert len(idx) == 3
