"""ReST-MCTS*-style Monte Carlo Tree Search over generated latent candidates.

Why this exists
---------------
`ReST-MCTS* <https://arxiv.org/abs/2406.03816>`_ couples a process reward
model with MCTS to allocate test-time compute toward candidates whose
partial reasoning looks most promising. Concurrent work
(`AlphaProof <https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/>`_,
LeanDojo) shows the same value-guided search beats flat self-consistency
on hard reasoning.

EBRM v0.7 already produces:
    * a pool of candidate latents with per-candidate ``energy``,
    * a pluggable PRM via ``verifiers.prm.GenerativePRMVerifier``.

This module wires those into a UCB1-driven tree search that re-ranks the
pool *under a fixed compute budget* (``num_simulations`` PRM calls,
defaulted to small enough for production). The output is a permutation of
the input candidates ordered by MCTS visit count — a Bayesian estimate of
which candidate is best given the available value signal.

Design
------
The tree is shallow on purpose:

    root
     ├── cluster₀  (DVTS-style farthest-first cluster)
     │     ├── cand_a
     │     └── cand_b
     ├── cluster₁
     │     └── cand_c
     └── …

`select` walks UCB1 down to a leaf, `evaluate` calls the value function
once on the leaf candidate, and `backup` propagates the score to the
ancestors. This stays torch-free, deterministic given the seed, and
fully covered by unit tests against a synthetic value function.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ebrm_system.inference.diverse_selector import (
    DiverseSelectionConfig,
    select_diverse,
)

LatentT = NDArray[np.float32]
ValueFn = Callable[[int], float]
"""``(candidate_index) -> value`` in [0, 1]. Higher is better. Wrap a
``GenerativePRMVerifier`` or a normalized energy score in a closure."""


@dataclass(frozen=True)
class MCTSConfig:
    """ReST-MCTS* configuration."""

    num_simulations: int = 16
    """Total number of select→evaluate→backup iterations. The PRM is called
    at most ``num_simulations`` times, regardless of pool size."""

    exploration_c: float = 1.4
    """UCB1 exploration constant. ``sqrt(2)`` is the textbook default."""

    num_clusters: int = 4
    """How many DVTS-style clusters to build at depth 1. The root has at
    most ``num_clusters`` children. Set to 1 to disable clustering."""

    seed: int = 0
    """Seed for tie-breaking during selection."""

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.exploration_c < 0:
            raise ValueError("exploration_c must be non-negative")
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive")


@dataclass
class _Node:
    """An MCTS tree node."""

    children: list[int] = field(default_factory=list)
    """Indices of child nodes in the flat node list. Empty for leaves."""

    candidate_idx: int | None = None
    """Index into the *original* candidate pool. Only set on leaves."""

    parent: int | None = None
    visits: int = 0
    total_value: float = 0.0

    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.total_value / self.visits


@dataclass(frozen=True)
class MCTSResult:
    """Output of :func:`mcts_select`."""

    ranking: tuple[int, ...]
    """Indices into the original pool, ordered best → worst by MCTS visits."""

    visits: tuple[int, ...]
    """Visit count for each candidate, aligned with ``ranking``."""

    values: tuple[float, ...]
    """Mean value for each candidate, aligned with ``ranking``."""

    simulations_run: int
    """Total simulations actually executed (== ``num_simulations``)."""


def _build_tree(
    latents: list[LatentT],
    config: MCTSConfig,
) -> list[_Node]:
    """Construct the shallow root → clusters → candidates tree."""
    n = len(latents)
    nodes: list[_Node] = [_Node()]  # root at index 0

    if config.num_clusters >= n:
        # Degenerate case: each candidate gets its own cluster.
        for cand_idx in range(n):
            cluster = _Node(parent=0)
            nodes.append(cluster)
            cluster_node_idx = len(nodes) - 1
            nodes[0].children.append(cluster_node_idx)
            leaf = _Node(parent=cluster_node_idx, candidate_idx=cand_idx)
            nodes.append(leaf)
            cluster.children.append(len(nodes) - 1)
        return nodes

    # DVTS-style clustering: pick num_clusters representatives by farthest-
    # first traversal, then assign every candidate to its nearest representative.
    cluster_cfg = DiverseSelectionConfig(
        num_groups=config.num_clusters,
        min_candidates=1,
    )
    # Energies are not used as a representative-selection criterion here;
    # we want geometric diversity. select_diverse picks lowest-energy per
    # cluster, but we only need the *centres*, so we feed uniform energies.
    centre_idx = select_diverse(latents, [0.0] * n, config=cluster_cfg)

    centres = np.stack([latents[i] for i in centre_idx])
    arr = np.stack(latents)
    # Squared euclidean distance from each candidate to each centre.
    diffs = arr[:, np.newaxis, :] - centres[np.newaxis, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    assignment = np.argmin(d2, axis=1)

    # Build cluster nodes.
    cluster_node_indices: list[int] = []
    for _ in centre_idx:
        cluster = _Node(parent=0)
        nodes.append(cluster)
        cluster_node_indices.append(len(nodes) - 1)
        nodes[0].children.append(len(nodes) - 1)

    # Attach candidates as leaves under the right cluster.
    for cand_idx, cluster_local_idx in enumerate(assignment):
        cluster_node_idx = cluster_node_indices[int(cluster_local_idx)]
        leaf = _Node(parent=cluster_node_idx, candidate_idx=cand_idx)
        nodes.append(leaf)
        nodes[cluster_node_idx].children.append(len(nodes) - 1)

    # Drop empty clusters (no assignments) so UCB never picks a dead branch.
    nodes[0].children = [c for c in nodes[0].children if nodes[c].children]
    return nodes


def _ucb1(child: _Node, parent_visits: int, c: float) -> float:
    if child.visits == 0:
        return math.inf
    exploit = child.mean_value()
    explore = c * math.sqrt(math.log(max(parent_visits, 1)) / child.visits)
    return exploit + explore


def _select(nodes: list[_Node], rng: np.random.Generator, c: float) -> list[int]:
    """Walk UCB1 from root to leaf. Returns the path of node indices."""
    path = [0]
    while nodes[path[-1]].children:
        cur = nodes[path[-1]]
        # UCB1 with random tie-breaking.
        scores = [_ucb1(nodes[ch], cur.visits, c) for ch in cur.children]
        best = max(scores)
        # Pick among the best uniformly to break ties deterministically by seed.
        candidates_at_best = [ch for ch, s in zip(cur.children, scores, strict=True) if s == best]
        chosen = candidates_at_best[int(rng.integers(0, len(candidates_at_best)))]
        path.append(chosen)
    return path


def _backup(nodes: list[_Node], path: list[int], value: float) -> None:
    for node_idx in path:
        nodes[node_idx].visits += 1
        nodes[node_idx].total_value += value


def mcts_select(
    latents: list[LatentT],
    value_fn: ValueFn,
    *,
    config: MCTSConfig | None = None,
) -> MCTSResult:
    """Run ReST-MCTS*-style search over a candidate pool.

    Parameters
    ----------
    latents
        The candidate latents (already generated by Langevin / DVTS / etc.).
    value_fn
        ``(candidate_index) -> value`` in ``[0, 1]``. Called at most
        ``config.num_simulations`` times. Wrap your PRM here.
    config
        :class:`MCTSConfig`. Defaults to a 16-simulation, 4-cluster search.

    Returns
    -------
    MCTSResult
        Re-ranking of the pool. The candidate at ``ranking[0]`` is the
        MCTS recommendation.

    Notes
    -----
    If ``len(latents) == 0``, returns an empty result with zero simulations
    run. If ``len(latents) == 1``, the value function is called once and
    that candidate is returned.
    """
    cfg = config or MCTSConfig()
    if not latents:
        return MCTSResult(ranking=(), visits=(), values=(), simulations_run=0)

    rng = np.random.default_rng(cfg.seed)
    nodes = _build_tree(latents, cfg)

    # Cache value-fn calls per candidate so the same leaf doesn't burn the
    # PRM budget on repeated visits.
    cached: dict[int, float] = {}

    sims = 0
    for _ in range(cfg.num_simulations):
        path = _select(nodes, rng, cfg.exploration_c)
        leaf = nodes[path[-1]]
        cand = leaf.candidate_idx
        if cand is None:
            # Should not happen with a well-formed tree, but guard anyway.
            break
        if cand not in cached:
            cached[cand] = float(value_fn(cand))
        _backup(nodes, path, cached[cand])
        sims += 1

    # Aggregate per-candidate stats from leaf nodes.
    leaf_stats: dict[int, tuple[int, float]] = {}
    for node in nodes:
        if node.candidate_idx is not None:
            leaf_stats[node.candidate_idx] = (node.visits, node.mean_value())

    pool_size = len(latents)
    visits = [leaf_stats.get(i, (0, 0.0))[0] for i in range(pool_size)]
    values = [leaf_stats.get(i, (0, 0.0))[1] for i in range(pool_size)]

    # Sort: more visits first; break ties by higher mean value.
    order = sorted(
        range(pool_size),
        key=lambda i: (-visits[i], -values[i]),
    )
    return MCTSResult(
        ranking=tuple(order),
        visits=tuple(visits[i] for i in order),
        values=tuple(values[i] for i in order),
        simulations_run=sims,
    )


__all__ = [
    "MCTSConfig",
    "MCTSResult",
    "ValueFn",
    "mcts_select",
]
