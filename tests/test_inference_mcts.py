"""Tests for ReST-MCTS*-style search over candidate pools (v0.10)."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.inference.mcts import MCTSConfig, mcts_select

D = 6


def _latents(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(D).astype(np.float32) for _ in range(n)]


class TestMCTSConfig:
    def test_rejects_non_positive_simulations(self) -> None:
        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=0)

    def test_rejects_negative_exploration(self) -> None:
        with pytest.raises(ValueError, match="exploration_c"):
            MCTSConfig(exploration_c=-0.5)

    def test_rejects_non_positive_clusters(self) -> None:
        with pytest.raises(ValueError, match="num_clusters"):
            MCTSConfig(num_clusters=0)


class TestMCTSSelect:
    def test_empty_pool_returns_empty(self) -> None:
        result = mcts_select([], lambda i: 0.0, config=MCTSConfig(num_simulations=4))
        assert result.ranking == ()
        assert result.simulations_run == 0

    def test_single_candidate_returned(self) -> None:
        latents = _latents(1)
        result = mcts_select(latents, lambda i: 0.5, config=MCTSConfig(num_simulations=4))
        assert result.ranking == (0,)
        assert result.visits[0] >= 1

    def test_best_candidate_ranked_first(self) -> None:
        """A clearly-best candidate should win the most visits."""
        latents = _latents(8, seed=1)
        # candidate 3 is the unique best.
        target = 3

        def vfn(i: int) -> float:
            return 1.0 if i == target else 0.1

        result = mcts_select(
            latents,
            vfn,
            config=MCTSConfig(num_simulations=64, num_clusters=4, seed=0),
        )
        assert result.ranking[0] == target
        assert result.visits[0] >= result.visits[-1]

    def test_simulations_count_is_respected(self) -> None:
        latents = _latents(4)
        called: list[int] = []

        def vfn(i: int) -> float:
            called.append(i)
            return 0.5

        result = mcts_select(latents, vfn, config=MCTSConfig(num_simulations=10, num_clusters=2))
        # At most 10 simulations; value-fn calls cached per candidate so
        # they may be fewer than simulations_run.
        assert result.simulations_run == 10
        assert len(set(called)) <= 4
        assert sum(result.visits) == 10

    def test_deterministic_given_seed(self) -> None:
        latents = _latents(6, seed=42)

        def vfn(i: int) -> float:
            return [0.1, 0.9, 0.3, 0.4, 0.2, 0.7][i]

        cfg = MCTSConfig(num_simulations=20, num_clusters=3, seed=7)
        a = mcts_select(latents, vfn, config=cfg)
        b = mcts_select(latents, vfn, config=cfg)
        assert a.ranking == b.ranking
        assert a.visits == b.visits

    def test_value_fn_called_at_most_once_per_candidate(self) -> None:
        latents = _latents(3)
        calls: dict[int, int] = {}

        def vfn(i: int) -> float:
            calls[i] = calls.get(i, 0) + 1
            return 0.5

        mcts_select(
            latents,
            vfn,
            config=MCTSConfig(num_simulations=30, num_clusters=2),
        )
        # Cached: each candidate should be evaluated at most once.
        assert all(v == 1 for v in calls.values())

    def test_ranking_is_a_permutation(self) -> None:
        latents = _latents(7, seed=3)

        def vfn(i: int) -> float:
            return 0.5

        result = mcts_select(
            latents,
            vfn,
            config=MCTSConfig(num_simulations=12, num_clusters=3),
        )
        assert sorted(result.ranking) == list(range(7))

    def test_more_clusters_than_candidates_degenerates_cleanly(self) -> None:
        latents = _latents(3)

        def vfn(i: int) -> float:
            return [0.2, 0.9, 0.1][i]

        result = mcts_select(
            latents,
            vfn,
            config=MCTSConfig(num_simulations=12, num_clusters=10, seed=0),
        )
        assert result.ranking[0] == 1
