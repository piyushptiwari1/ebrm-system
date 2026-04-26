"""Tests for the Hindsight/Letta-style 3-tier memory (v0.9)."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.memory import (
    MemoryTier,
    TierConfig,
    TieredMemory,
    TieredMemoryConfig,
)

D = 8


def _latent(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(D).astype(np.float32)


class TestTierConfigValidation:
    def test_rejects_non_positive_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            TierConfig(max_size=0)

    def test_rejects_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError, match="ttl_ticks"):
            TierConfig(max_size=1, ttl_ticks=0)

    def test_rejects_non_positive_promote_threshold(self) -> None:
        with pytest.raises(ValueError, match="promote_after_hits"):
            TierConfig(max_size=1, promote_after_hits=0)


class TestBasicAddSearch:
    def test_add_lands_in_working_tier(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=D))
        v = _latent(0)
        mem.add(v[np.newaxis, :], [v])
        assert mem.tier_size(MemoryTier.WORKING) == 1
        assert mem.tier_size(MemoryTier.EPISODIC) == 0
        assert mem.tier_size(MemoryTier.SEMANTIC) == 0
        assert len(mem) == 1

    def test_search_returns_inserted_payload(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=D))
        v = _latent(1)
        mem.add(v[np.newaxis, :], [v])
        hits = mem.search(v, k=1)
        assert len(hits) == 1
        assert hits[0][1] is v

    def test_empty_search_returns_empty(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=D))
        assert mem.search(_latent(0), k=4) == []

    def test_payload_count_mismatch_raises(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=D))
        v = _latent(0)
        with pytest.raises(ValueError, match="payload count"):
            mem.add(v[np.newaxis, :], [v, v])


class TestPromotion:
    def test_entry_promotes_after_threshold_hits(self) -> None:
        mem = TieredMemory(
            TieredMemoryConfig(
                in_dim=D,
                working=TierConfig(max_size=8, promote_after_hits=2),
                episodic=TierConfig(max_size=8, promote_after_hits=3),
                semantic=TierConfig(max_size=8),
            )
        )
        v = _latent(2)
        mem.add(v[np.newaxis, :], [v])
        # Two hits should trigger promotion to episodic.
        mem.search(v, k=1)
        mem.search(v, k=1)
        assert mem.tier_size(MemoryTier.WORKING) == 0
        assert mem.tier_size(MemoryTier.EPISODIC) == 1

    def test_chained_promotion_to_semantic(self) -> None:
        mem = TieredMemory(
            TieredMemoryConfig(
                in_dim=D,
                working=TierConfig(max_size=8, promote_after_hits=1),
                episodic=TierConfig(max_size=8, promote_after_hits=1),
                semantic=TierConfig(max_size=8),
            )
        )
        v = _latent(3)
        mem.add(v[np.newaxis, :], [v])
        mem.search(v, k=1)  # working -> episodic
        mem.search(v, k=1)  # episodic -> semantic
        assert mem.tier_size(MemoryTier.SEMANTIC) == 1


class TestTTLExpiry:
    def test_entries_expire_after_ttl(self) -> None:
        mem = TieredMemory(
            TieredMemoryConfig(
                in_dim=D,
                working=TierConfig(max_size=8, ttl_ticks=2),
                episodic=TierConfig(max_size=8),
                semantic=TierConfig(max_size=8),
            )
        )
        v = _latent(4)
        mem.add(v[np.newaxis, :], [v])
        # Bump the clock past ttl_ticks via a few unrelated adds.
        for _ in range(5):
            other = _latent(100)
            mem.add(other[np.newaxis, :], [other])
        # The original 'v' must be gone.
        hits = [p for _, p in mem.search(v, k=4) if p is v]
        assert hits == []


class TestCapacityEviction:
    def test_working_tier_respects_max_size(self) -> None:
        mem = TieredMemory(
            TieredMemoryConfig(
                in_dim=D,
                working=TierConfig(max_size=3),
                episodic=TierConfig(max_size=8),
                semantic=TierConfig(max_size=8),
            )
        )
        for i in range(10):
            v = _latent(i)
            mem.add(v[np.newaxis, :], [v])
        assert mem.tier_size(MemoryTier.WORKING) <= 3


class TestSummarizerOnEviction:
    def test_summarizer_reroutes_to_next_tier(self) -> None:
        captured: list[int] = []

        def summarize(latents, payloads):
            captured.append(len(payloads))
            # Compress N entries into a single mean-pooled latent.
            mean = np.mean(np.stack(latents), axis=0).astype(np.float32)
            return [mean], [("summary", len(payloads))]

        mem = TieredMemory(
            TieredMemoryConfig(
                in_dim=D,
                working=TierConfig(max_size=8, ttl_ticks=2),
                episodic=TierConfig(max_size=8),
                semantic=TierConfig(max_size=8),
                summarizer=summarize,
            )
        )
        # Force several TTL evictions from the working tier.
        for i in range(6):
            v = _latent(i)
            mem.add(v[np.newaxis, :], [v])
        for _ in range(4):
            other = _latent(999)
            mem.add(other[np.newaxis, :], [other])
        assert captured, "summarizer should have been called on TTL eviction"
        assert mem.tier_size(MemoryTier.EPISODIC) >= 1


class TestStatsAndDuckType:
    def test_stats_reports_per_tier_counts(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=D))
        v = _latent(0)
        mem.add(v[np.newaxis, :], [v])
        stats = mem.stats()
        assert stats["working"] == 1
        assert stats["episodic"] == 0
        assert stats["semantic"] == 0

    def test_drop_in_compatible_with_generate_candidates(self) -> None:
        """TieredMemory must satisfy the LatentIndex duck-typed protocol."""
        from ebrm_system.inference.candidates import (
            CandidateConfig,
            generate_candidates,
        )

        mem = TieredMemory(TieredMemoryConfig(in_dim=D))
        for i in range(4):
            v = _latent(i)
            mem.add(v[np.newaxis, :], [v])

        seed = _latent(0)

        def energy(z: np.ndarray) -> float:
            return float(np.sum(z * z))

        cfg = CandidateConfig(num_candidates=4, num_steps=3, warmstart_k=2, seed=0)
        cands = generate_candidates(seed, energy, config=cfg, index=mem)  # type: ignore[arg-type]
        assert len(cands) == 4
        # At least one candidate should have been warm-started from the index.
        assert any(c.warmstart for c in cands)
