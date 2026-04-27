"""Embedder protocol — pluggable text-to-vector backends.

All embedders must:
- Be deterministic for a given input batch.
- Produce ``np.float32`` row-vectors.
- Return L2-normalised vectors so that cosine similarity reduces to dot product.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Embed a batch of strings into a ``(n, dim)`` float32 matrix.

    Implementations must return L2-normalised rows so downstream cosine
    similarity reduces to a single matmul.
    """

    dim: int
    """Output embedding dimensionality."""

    name: str
    """Stable identifier for audit/logging (e.g. ``"azure-text-embedding-3-small-2"``)."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), self.dim)`` float32 matrix of L2-normalised rows."""
        ...
