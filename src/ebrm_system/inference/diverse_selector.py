"""Diverse Verifier Tree Search-style candidate selection.

Why this exists
---------------
Flat self-consistency over N i.i.d. candidates is wasteful: many candidates
land on the same (often wrong) attractor. The HuggingFaceH4
"Scaling test-time compute" report and follow-ups
(https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)
show that **Diverse Verifier Tree Search (DVTS)** beats beam-search and
best-of-N at every compute budget by enforcing diversity across the
candidate pool before the verifier picks a winner.

EBRM produces candidate *latents*, not step-level traces, so we adapt the
DVTS recipe to our setting:

    1. Cluster the N latent candidates into M groups by greedy farthest-first
       traversal in latent space (deterministic, no random init).
    2. From each group keep only the lowest-energy candidate (the
       "tree-search winner" of that subtree).
    3. Return the M survivors. The reasoner then votes over those instead of
       the raw N.

This module is pure-Python + NumPy, deterministic, side-effect free,
torch-optional.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LatentT = NDArray[np.float32]


@dataclass(frozen=True)
class DiverseSelectionConfig:
    """Knobs for :func:`select_diverse`."""

    num_groups: int = 4
    """Number of diverse subtrees (M). The selector returns at most this
    many survivors. ``num_groups <= 1`` disables clustering."""

    min_candidates: int = 4
    """If the input pool has fewer than this many candidates, the selector
    is a no-op (clustering on tiny pools is meaningless)."""

    def __post_init__(self) -> None:
        if self.num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {self.num_groups}")
        if self.min_candidates < 1:
            raise ValueError(f"min_candidates must be >= 1, got {self.min_candidates}")


def select_diverse(
    latents: Sequence[LatentT],
    energies: Sequence[float],
    config: DiverseSelectionConfig | None = None,
) -> list[int]:
    """Return indices of survivors after diverse-cluster filtering.

    Parameters
    ----------
    latents
        Per-candidate latent vectors (1-D arrays of equal shape).
    energies
        Per-candidate scalar energies (lower = more plausible).
    config
        Optional :class:`DiverseSelectionConfig`.

    Returns
    -------
    list[int]
        Indices into ``latents`` (length ``<= num_groups``), one per cluster,
        each pointing at the lowest-energy member of its cluster. Order is
        deterministic: it matches the cluster-discovery order from
        farthest-first traversal.

    Notes
    -----
    The clustering is greedy farthest-first traversal seeded at the global
    minimum-energy candidate. This is O(N * M) which is fine for the
    candidate counts EBRM uses (<= 32). It is fully deterministic.

    Edge cases:

    * If ``len(latents) < config.min_candidates`` or ``num_groups <= 1``,
      returns ``list(range(len(latents)))`` unchanged (no-op).
    * If ``num_groups >= len(latents)``, returns all indices sorted by
      energy ascending.
    """
    cfg = config or DiverseSelectionConfig()
    n = len(latents)
    if n != len(energies):
        raise ValueError(f"latents/energies length mismatch: {n} vs {len(energies)}")
    if n == 0:
        return []
    if n < cfg.min_candidates or cfg.num_groups <= 1:
        return list(range(n))
    if cfg.num_groups >= n:
        return sorted(range(n), key=lambda i: float(energies[i]))

    arr = np.stack([np.asarray(z, dtype=np.float32) for z in latents], axis=0)
    eng = np.asarray(energies, dtype=np.float64)

    # Seed: lowest-energy candidate.
    seed = int(np.argmin(eng))
    centers: list[int] = [seed]

    # Squared distance from each point to its nearest center (so far).
    dist = np.sum((arr - arr[seed]) ** 2, axis=1)

    while len(centers) < cfg.num_groups:
        # Pick the point farthest from any current center.
        nxt = int(np.argmax(dist))
        if dist[nxt] == 0.0:  # everyone collapsed to a center already
            break
        centers.append(nxt)
        new_d = np.sum((arr - arr[nxt]) ** 2, axis=1)
        dist = np.minimum(dist, new_d)

    # Assign every candidate to its nearest center.
    assignments = np.zeros(n, dtype=np.int64)
    for i in range(n):
        d_to_centers = [float(np.sum((arr[i] - arr[c]) ** 2)) for c in centers]
        assignments[i] = int(np.argmin(d_to_centers))

    # From each cluster, keep the lowest-energy member.
    survivors: list[int] = []
    for cluster_idx, _ in enumerate(centers):
        members = [i for i in range(n) if int(assignments[i]) == cluster_idx]
        if not members:
            continue
        winner = min(members, key=lambda i: float(eng[i]))
        survivors.append(winner)

    return survivors


__all__ = ["DiverseSelectionConfig", "select_diverse"]
