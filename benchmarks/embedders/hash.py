"""Hash-projection embedder — deterministic baseline (no semantics).

Existed as ``hash_embed`` in ``benchmarks.longmemeval``; here we expose it as
an :class:`Embedder` so the harness can be parameterised over backends.
This is the baseline against which trained / API embedders must improve.
"""

from __future__ import annotations

import hashlib

import numpy as np


class HashEmbedder:
    """Deterministic random projection from text → unit sphere.

    Useful for offline tests and as a no-semantics floor. Identical to the
    original ``benchmarks.longmemeval.hash_embed`` function.
    """

    name = "hash-projection"

    def __init__(self, dim: int = 128, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self._seed = seed

    def _embed_one(self, text: str) -> np.ndarray:
        h = hashlib.blake2b(
            text.encode("utf-8"),
            digest_size=8,
            salt=self._seed.to_bytes(8, "little", signed=False),
        ).digest()
        rng = np.random.default_rng(int.from_bytes(h, "little"))
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts], axis=0)
