"""QJL-quantized latent index for warm-start retrieval.

Given a corpus of canonical solution latents (e.g. EBRM v2 projector outputs
for the GSM8K training set), build a compact bit-code index. At query time,
look up the k nearest cached latents and use them to seed Langevin restarts —
this dramatically reduces the number of steps needed to reach a low-energy
attractor.

Memory budget for 1 M latents at d=768, m=512 bits:
    raw fp32 :  1 M * 768 * 4  ~  3.0 GB
    QJL      :  1 M * 64       ~   64 MB    (47x smaller)

This is a pure numpy module; no torch, no FAISS. For >10 M entries, swap in
FAISS-binary or hnswlib via the same ``LatentIndex`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ebrm_system.inference.qjl import QJLConfig, QJLProjector


@dataclass(frozen=True)
class IndexConfig:
    """Latent-index configuration.

    Parameters
    ----------
    in_dim
        Latent dimensionality.
    bits
        QJL projection size in bits.
    seed
        Projector seed.
    max_size
        Maximum number of entries. ``None`` means unbounded (the default,
        backward-compatible). When set, exceeding the limit triggers
        eviction governed by :attr:`evict_policy`. Must be > 0 if set.
    evict_policy
        ``"lru"`` evicts least-recently-accessed entries (touched on
        :meth:`LatentIndex.search`). ``"fifo"`` evicts in insertion order.
        Ignored when ``max_size`` is ``None``.
    """

    in_dim: int
    bits: int = 512
    seed: int = 0
    max_size: int | None = None
    evict_policy: str = "lru"

    def __post_init__(self) -> None:
        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("max_size must be positive when set")
        if self.evict_policy not in {"lru", "fifo"}:
            raise ValueError(f"unknown evict_policy: {self.evict_policy!r}")

    def to_qjl(self) -> QJLConfig:
        return QJLConfig(in_dim=self.in_dim, out_bits=self.bits, seed=self.seed)


class LatentIndex:
    """Bit-code latent index with cosine-approximate kNN lookup.

    Bounded by ``IndexConfig.max_size`` (optional). When set, eviction runs
    on every :meth:`add` call and is logically O(evicted) per insert.
    """

    def __init__(self, config: IndexConfig) -> None:
        self.config = config
        self._projector = QJLProjector(config.to_qjl())
        self._codes: NDArray[np.uint8] | None = None
        self._payloads: list[object] = []
        # Monotonic logical clock — bumps on every add and (for LRU) on every
        # successful search hit. Used as an integer recency stamp.
        self._tick: int = 0
        # Per-entry recency. Aligned with ``_payloads`` and the rows of
        # ``_codes``. Empty until the first add.
        self._last_access: list[int] = []

    def add(self, latents: NDArray[np.float32], payloads: list[object]) -> int:
        """Add a batch of latents with parallel payloads.

        Returns the number of evicted entries (0 when the index is unbounded
        or still has room).
        """
        if len(payloads) != latents.shape[0]:
            raise ValueError(f"payload count {len(payloads)} != latent count {latents.shape[0]}")
        new_codes = self._projector.project_batch(latents)
        self._codes = (
            new_codes if self._codes is None else np.concatenate([self._codes, new_codes], axis=0)
        )
        self._payloads.extend(payloads)
        # Per-row monotonic stamps so FIFO eviction is deterministic on ties.
        new_stamps = list(range(self._tick + 1, self._tick + 1 + latents.shape[0]))
        self._tick += latents.shape[0]
        self._last_access.extend(new_stamps)
        return self._evict_if_needed()

    def __len__(self) -> int:
        return 0 if self._codes is None else self._codes.shape[0]

    def search(self, query: NDArray[np.float32], k: int = 8) -> list[tuple[float, object]]:
        """Return top-k entries by approximate cosine similarity (high → low).

        Distance metric is computed in code space via Hamming distance:
        h ∈ [0, 1] → cos(θ) ≈ cos(π · h).

        Side effect: when ``evict_policy == "lru"``, accessed rows have
        their last-access stamp refreshed so they survive the next eviction.
        """
        if self._codes is None or len(self) == 0:
            return []
        if k <= 0:
            raise ValueError("k must be positive")

        q_code = self._projector.project(query)
        # Vectorized Hamming distance: XOR + popcount across rows.
        xor = np.bitwise_xor(self._codes, q_code[np.newaxis, :])
        # popcount via unpackbits + sum (fast enough for ≤ 1 M rows; swap in
        # numpy 2.0 ``bit_count`` when the deps allow).
        popcount = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)
        normalized = popcount / self.config.bits
        sims = np.cos(np.pi * normalized)

        top_k = min(k, len(self))
        # argpartition is O(n); slice + sort the small top-k.
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]

        if self.config.evict_policy == "lru":
            self._tick += 1
            for i in idx:
                self._last_access[int(i)] = self._tick

        return [(float(sims[i]), self._payloads[int(i)]) for i in idx]

    # -- internals ------------------------------------------------------- #

    def _evict_if_needed(self) -> int:
        max_size = self.config.max_size
        if max_size is None or self._codes is None:
            return 0
        excess = self._codes.shape[0] - max_size
        if excess <= 0:
            return 0
        # Both policies pick the ``excess`` smallest stamps; FIFO uses
        # insertion order (stamps are monotonically increasing, so the
        # earliest-added rows have the smallest stamps), LRU uses the
        # access-refreshed stamps.
        stamps = np.asarray(self._last_access, dtype=np.int64)
        evict_idx = np.argpartition(stamps, excess - 1)[:excess]
        keep_mask = np.ones(self._codes.shape[0], dtype=bool)
        keep_mask[evict_idx] = False
        self._codes = self._codes[keep_mask]
        self._payloads = [p for p, keep in zip(self._payloads, keep_mask, strict=True) if keep]
        self._last_access = [
            t for t, keep in zip(self._last_access, keep_mask, strict=True) if keep
        ]
        return int(excess)
