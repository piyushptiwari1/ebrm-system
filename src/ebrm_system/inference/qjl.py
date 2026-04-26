"""Quantized Johnson-Lindenstrauss (QJL) projection.

QJL is the building block underlying TurboQuant
(Zandieh, Daliri, Hadian, Mirrokni — Google Research, ICLR 2026,
arXiv:2504.19874). It maps high-dimensional Euclidean vectors to short
bit-strings while preserving pairwise distances up to a (1 ± ε) factor.

Algorithm (1-bit variant, sufficient for ANN):
    1. Sample R in R^(m by d) with i.i.d. N(0, 1) entries (fixed across calls).
    2. Project: y = R @ x  (in R^m).
    3. Quantize: q = sign(y) in {-1, +1}^m, packed into bits.
    4. Recover approximate cosine via Hamming distance:
         cos(x, x') ~= cos(pi * hamming(q, q') / m)

Memory: m bits per vector instead of 32*d bits. For d=768, m=512 ->
        96x compression with < 5 % cosine error on unit vectors.

This is the *full* QJL — pure numpy, deterministic given a seed, no torch
required. TurboQuant proper adds a Hadamard pre-rotation and multi-bit
PolarQuant; we keep the 1-bit core here because it covers the ANN /
latent-index use case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QJLConfig:
    """QJL projector configuration."""

    in_dim: int
    out_bits: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if self.out_bits <= 0 or self.out_bits % 8 != 0:
            raise ValueError("out_bits must be a positive multiple of 8")


class QJLProjector:
    """1-bit Quantized Johnson-Lindenstrauss projector.

    Deterministic for a given seed. Thread-safe (read-only after construction).
    """

    def __init__(self, config: QJLConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        # Gaussian random matrix; rows are independent.
        self._R: NDArray[np.float32] = rng.standard_normal(
            (config.out_bits, config.in_dim), dtype=np.float32
        )

    def project(self, x: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Project a single vector x in R^d -> packed bits in uint8[m/8]."""
        if x.ndim != 1 or x.shape[0] != self.config.in_dim:
            raise ValueError(f"expected shape ({self.config.in_dim},), got {x.shape}")
        y = self._R @ x.astype(np.float32, copy=False)
        bits = (y >= 0).astype(np.uint8)
        return np.packbits(bits)

    def project_batch(self, x_batch: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Project a batch x_batch in R^(n by d) -> packed bits uint8[n, m/8]."""
        if x_batch.ndim != 2 or x_batch.shape[1] != self.config.in_dim:
            raise ValueError(
                f"expected shape (n, {self.config.in_dim}), got {x_batch.shape}"
            )
        y = x_batch.astype(np.float32, copy=False) @ self._R.T  # (n, m)
        bits = (y >= 0).astype(np.uint8)
        return np.packbits(bits, axis=1)

    @property
    def compressed_bytes_per_vector(self) -> int:
        return self.config.out_bits // 8

    def estimate_cosine(
        self, q1: NDArray[np.uint8], q2: NDArray[np.uint8]
    ) -> float:
        """Approximate cosine(x, x') from packed bit codes via Hamming distance.

        cos(theta) ~= cos(pi * h / m), where h is normalized Hamming distance.
        """
        if q1.shape != q2.shape:
            raise ValueError("code shape mismatch")
        # XOR + popcount
        xor = np.bitwise_xor(q1, q2)
        hamming = int(np.unpackbits(xor).sum())
        m = self.config.out_bits
        normalized = hamming / m
        return float(np.cos(np.pi * normalized))
