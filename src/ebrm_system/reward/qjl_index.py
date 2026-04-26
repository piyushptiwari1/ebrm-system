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
    """Latent-index configuration."""

    in_dim: int
    bits: int = 512
    seed: int = 0

    def to_qjl(self) -> QJLConfig:
        return QJLConfig(in_dim=self.in_dim, out_bits=self.bits, seed=self.seed)


class LatentIndex:
    """Bit-code latent index with cosine-approximate kNN lookup."""

    def __init__(self, config: IndexConfig) -> None:
        self.config = config
        self._projector = QJLProjector(config.to_qjl())
        self._codes: NDArray[np.uint8] | None = None
        self._payloads: list[object] = []

    def add(self, latents: NDArray[np.float32], payloads: list[object]) -> None:
        """Add a batch of latents with parallel payloads (e.g. solution strings)."""
        if len(payloads) != latents.shape[0]:
            raise ValueError(
                f"payload count {len(payloads)} != latent count {latents.shape[0]}"
            )
        new_codes = self._projector.project_batch(latents)
        self._codes = (
            new_codes if self._codes is None else np.concatenate([self._codes, new_codes], axis=0)
        )
        self._payloads.extend(payloads)

    def __len__(self) -> int:
        return 0 if self._codes is None else self._codes.shape[0]

    def search(
        self, query: NDArray[np.float32], k: int = 8
    ) -> list[tuple[float, object]]:
        """Return top-k entries by approximate cosine similarity (high → low).

        Distance metric is computed in code space via Hamming distance:
        h ∈ [0, 1] → cos(θ) ≈ cos(π · h).
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
        return [(float(sims[i]), self._payloads[int(i)]) for i in idx]
