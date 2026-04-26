"""Three-tier memory stack inspired by Hindsight (LongMemEval) and Letta/MemGPT.

Why this exists
---------------
A flat :class:`~ebrm_system.reward.qjl_index.LatentIndex` either grows
unbounded (memory leak) or evicts blindly via LRU/FIFO (loses high-utility
context). Production reasoning agents need a *graded* memory: a small fast
working set, a medium episodic store, and a large long-lived semantic store,
with promotion driven by re-use and eviction softened by summarization.

Hindsight's LongMemEval result (91.4%, beats vanilla RAG) and Letta's
agent-memory pattern both validate this design. This module brings the same
shape to EBRM at the latent level — three composed
:class:`~ebrm_system.reward.qjl_index.LatentIndex` instances with TTL
expiry, hit-count promotion, and an optional summarization hook on
eviction.

Drop-in replacement for ``LatentIndex`` in
:func:`ebrm_system.inference.candidates.generate_candidates`: the duck-typed
interface (``__len__`` + ``search``) is preserved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex


class MemoryTier(str, Enum):
    """Tier names. Order = promotion direction (working → episodic → semantic)."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


_TIER_ORDER: tuple[MemoryTier, ...] = (
    MemoryTier.WORKING,
    MemoryTier.EPISODIC,
    MemoryTier.SEMANTIC,
)


@dataclass(frozen=True)
class TierConfig:
    """Per-tier knobs."""

    max_size: int
    """Hard capacity. Excess entries are evicted on every add."""

    ttl_ticks: int | None = None
    """Maximum age (in logical clock ticks) before forced expiry. ``None``
    disables time-based eviction."""

    promote_after_hits: int | None = None
    """Promote an entry to the next tier after this many search hits.
    ``None`` disables promotion (sink tier)."""

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.ttl_ticks is not None and self.ttl_ticks <= 0:
            raise ValueError("ttl_ticks must be positive when set")
        if self.promote_after_hits is not None and self.promote_after_hits <= 0:
            raise ValueError("promote_after_hits must be positive when set")


# A summarizer compresses a list of (latent, payload) pairs being evicted
# into a smaller list. Returning ``[]`` discards them; returning the input
# unchanged disables summarization. The output payloads must be ndarray
# latents so they remain compatible with the warm-start path.
Summarizer = Callable[
    [list[NDArray[np.float32]], list[object]],
    tuple[list[NDArray[np.float32]], list[object]],
]


@dataclass(frozen=True)
class TieredMemoryConfig:
    """End-to-end tiered-memory configuration."""

    in_dim: int
    bits: int = 512
    seed: int = 0
    working: TierConfig = field(
        default_factory=lambda: TierConfig(max_size=64, promote_after_hits=2)
    )
    episodic: TierConfig = field(
        default_factory=lambda: TierConfig(max_size=512, promote_after_hits=3)
    )
    semantic: TierConfig = field(default_factory=lambda: TierConfig(max_size=4096))
    summarizer: Summarizer | None = None
    """Called on entries evicted from a *non-sink* tier; the result is
    promoted into the next tier instead of being dropped. Eviction from the
    final (semantic) tier always discards. Hindsight-style."""

    def per_tier(self) -> dict[MemoryTier, TierConfig]:
        return {
            MemoryTier.WORKING: self.working,
            MemoryTier.EPISODIC: self.episodic,
            MemoryTier.SEMANTIC: self.semantic,
        }


class _TierState:
    """Holds the LatentIndex plus per-entry hit counts and birth ticks."""

    def __init__(self, index_cfg: IndexConfig) -> None:
        self.index = LatentIndex(index_cfg)
        self.hits: list[int] = []
        self.born: list[int] = []

    def __len__(self) -> int:
        return len(self.index)


class TieredMemory:
    """Hindsight/Letta-style 3-tier latent memory.

    Duck-typed compatible with :class:`LatentIndex`: exposes ``__len__`` and
    ``search`` so it drops into
    :func:`~ebrm_system.inference.candidates.generate_candidates`.

    All adds enter the working tier. Search hits across any tier increment
    the entry's hit counter; when the counter reaches the tier's
    ``promote_after_hits`` threshold, the entry is moved to the next tier
    on the next maintenance pass. TTL expiry runs on every add and search.
    """

    def __init__(self, config: TieredMemoryConfig) -> None:
        self.config = config
        self._tick = 0
        self._tiers: dict[MemoryTier, _TierState] = {
            tier: _TierState(
                IndexConfig(
                    in_dim=config.in_dim,
                    bits=config.bits,
                    seed=config.seed,
                    max_size=cfg.max_size,
                    evict_policy="lru",
                )
            )
            for tier, cfg in config.per_tier().items()
        }

    # -- public, LatentIndex-compatible API -------------------------------

    def __len__(self) -> int:
        return sum(len(t) for t in self._tiers.values())

    def add(self, latents: NDArray[np.float32], payloads: list[object]) -> dict[str, int]:
        """Insert into the working tier. Returns a per-tier eviction count."""
        if latents.shape[0] != len(payloads):
            raise ValueError("payload count must match latent count")
        self._tick += 1
        working = self._tiers[MemoryTier.WORKING]
        evicted_before = self._snapshot_evicted_payloads(working)
        working.index.add(latents, payloads)
        working.hits.extend([0] * latents.shape[0])
        working.born.extend([self._tick] * latents.shape[0])
        # Indexing trims to max_size already; sync our parallel arrays.
        self._sync_parallel_arrays(working)
        report = self._maintain()
        # Account for evictions caused by the add itself.
        report.setdefault(MemoryTier.WORKING.value, 0)
        report[MemoryTier.WORKING.value] += max(
            0, len(evicted_before) - len(self._tiers[MemoryTier.WORKING].index)
        )
        return report

    def search(self, query: NDArray[np.float32], k: int = 8) -> list[tuple[float, object]]:
        """Search every tier, merge by similarity, refresh hits + ticks."""
        self._tick += 1
        merged: list[tuple[float, object, MemoryTier, int]] = []
        for tier, state in self._tiers.items():
            if len(state) == 0:
                continue
            hits = state.index.search(query, k=min(k, len(state)))
            for sim, payload in hits:
                idx = self._payload_index(state, payload)
                if idx is None:
                    continue
                state.hits[idx] += 1
                merged.append((sim, payload, tier, idx))
        merged.sort(key=lambda x: -x[0])
        top = merged[:k]
        self._maintain()
        return [(sim, payload) for sim, payload, _t, _i in top]

    # -- internals --------------------------------------------------------

    def _maintain(self) -> dict[str, int]:
        """Run TTL expiry and hit-driven promotions; return eviction counts."""
        report: dict[str, int] = {}
        cfg = self.config.per_tier()
        for i, tier in enumerate(_TIER_ORDER):
            state = self._tiers[tier]
            tier_cfg = cfg[tier]

            # 1) TTL expiry — collect ages first so deletions don't race.
            if tier_cfg.ttl_ticks is not None and len(state) > 0:
                expired = [
                    j for j, b in enumerate(state.born) if (self._tick - b) > tier_cfg.ttl_ticks
                ]
                if expired:
                    self._evict(tier, expired, summarize=(i < len(_TIER_ORDER) - 1))
                    report[tier.value] = report.get(tier.value, 0) + len(expired)

            # 2) Promotion — non-sink tiers only.
            if tier_cfg.promote_after_hits is not None and i < len(_TIER_ORDER) - 1:
                state = self._tiers[tier]
                promote_idx = [
                    j for j, h in enumerate(state.hits) if h >= tier_cfg.promote_after_hits
                ]
                if promote_idx:
                    self._promote(tier, _TIER_ORDER[i + 1], promote_idx)
        return report

    def _promote(self, src: MemoryTier, dst: MemoryTier, idx: list[int]) -> None:
        src_state = self._tiers[src]
        dst_state = self._tiers[dst]
        latents, payloads = self._gather(src_state, idx)
        if not payloads:
            return
        # Drop from source first so the parallel arrays line up.
        self._evict(src, idx, summarize=False)
        # Insert into destination.
        dst_state.index.add(np.stack(latents), payloads)
        dst_state.hits.extend([0] * len(payloads))
        dst_state.born.extend([self._tick] * len(payloads))
        self._sync_parallel_arrays(dst_state)

    def _evict(self, tier: MemoryTier, idx: list[int], *, summarize: bool) -> None:
        state = self._tiers[tier]
        if not idx:
            return
        latents, payloads = self._gather(state, idx)

        # Rebuild the LatentIndex without the evicted rows. The QJL index
        # has no public delete; we replace it wholesale to keep the array
        # contracts simple and bug-free.
        keep_mask = np.ones(len(state), dtype=bool)
        keep_mask[idx] = False
        kept_payloads = [p for p, k in zip(state.index._payloads, keep_mask, strict=True) if k]
        kept_codes = state.index._codes[keep_mask] if state.index._codes is not None else None
        kept_access = [t for t, k in zip(state.index._last_access, keep_mask, strict=True) if k]
        state.index._codes = kept_codes
        state.index._payloads = kept_payloads
        state.index._last_access = kept_access
        state.hits = [h for h, k in zip(state.hits, keep_mask, strict=True) if k]
        state.born = [b for b, k in zip(state.born, keep_mask, strict=True) if k]

        if summarize and self.config.summarizer is not None and payloads:
            sum_latents, sum_payloads = self.config.summarizer(latents, payloads)
            if sum_payloads:
                # Push summary into the *next* tier.
                tier_idx = _TIER_ORDER.index(tier)
                if tier_idx + 1 < len(_TIER_ORDER):
                    nxt = self._tiers[_TIER_ORDER[tier_idx + 1]]
                    nxt.index.add(np.stack(sum_latents), list(sum_payloads))
                    nxt.hits.extend([0] * len(sum_payloads))
                    nxt.born.extend([self._tick] * len(sum_payloads))
                    self._sync_parallel_arrays(nxt)

    @staticmethod
    def _gather(
        state: _TierState, idx: list[int]
    ) -> tuple[list[NDArray[np.float32]], list[object]]:
        latents: list[NDArray[np.float32]] = []
        payloads: list[object] = []
        for j in idx:
            payload = state.index._payloads[j]
            if isinstance(payload, np.ndarray):
                latents.append(payload.astype(np.float32, copy=False))
                payloads.append(payload)
        return latents, payloads

    @staticmethod
    def _payload_index(state: _TierState, payload: object) -> int | None:
        for i, p in enumerate(state.index._payloads):
            if p is payload:
                return i
        return None

    @staticmethod
    def _sync_parallel_arrays(state: _TierState) -> None:
        """LatentIndex.add may have evicted entries to honour ``max_size``;
        trim our parallel hit/born arrays to match its current length."""
        n = len(state.index)
        if len(state.hits) > n:
            state.hits = state.hits[-n:]
        if len(state.born) > n:
            state.born = state.born[-n:]

    @staticmethod
    def _snapshot_evicted_payloads(state: _TierState) -> list[object]:
        return list(state.index._payloads)

    # -- inspection helpers ----------------------------------------------

    def tier_size(self, tier: MemoryTier) -> int:
        return len(self._tiers[tier])

    def stats(self) -> dict[str, int]:
        return {tier.value: len(state) for tier, state in self._tiers.items()}


__all__ = [
    "MemoryTier",
    "Summarizer",
    "TierConfig",
    "TieredMemory",
    "TieredMemoryConfig",
]
