"""Unit tests for ``ebrm_system.inference.diverse_selector``."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.inference.diverse_selector import (
    DiverseSelectionConfig,
    select_diverse,
)


def _l(a: float, b: float) -> np.ndarray:
    return np.asarray([a, b], dtype=np.float32)


class TestDiverseSelectionConfig:
    def test_defaults(self) -> None:
        cfg = DiverseSelectionConfig()
        assert cfg.num_groups == 4
        assert cfg.min_candidates == 4

    def test_rejects_zero_groups(self) -> None:
        with pytest.raises(ValueError, match="num_groups"):
            DiverseSelectionConfig(num_groups=0)

    def test_rejects_zero_min_candidates(self) -> None:
        with pytest.raises(ValueError, match="min_candidates"):
            DiverseSelectionConfig(min_candidates=0)


class TestSelectDiverse:
    def test_empty_input(self) -> None:
        assert select_diverse([], []) == []

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            select_diverse([_l(0, 0)], [0.1, 0.2])

    def test_below_min_candidates_is_noop(self) -> None:
        cfg = DiverseSelectionConfig(num_groups=2, min_candidates=4)
        latents = [_l(0, 0), _l(10, 10)]
        out = select_diverse(latents, [0.5, 0.6], cfg)
        assert out == [0, 1]

    def test_num_groups_one_is_noop(self) -> None:
        cfg = DiverseSelectionConfig(num_groups=1, min_candidates=1)
        latents = [_l(0, 0), _l(10, 10), _l(20, 20)]
        out = select_diverse(latents, [0.1, 0.2, 0.3], cfg)
        assert out == [0, 1, 2]

    def test_groups_geq_n_returns_all_sorted_by_energy(self) -> None:
        cfg = DiverseSelectionConfig(num_groups=10, min_candidates=1)
        latents = [_l(0, 0), _l(1, 1), _l(2, 2)]
        energies = [0.9, 0.1, 0.5]
        out = select_diverse(latents, energies, cfg)
        assert out == [1, 2, 0]  # sorted ascending by energy

    def test_picks_lowest_energy_per_cluster(self) -> None:
        cfg = DiverseSelectionConfig(num_groups=2, min_candidates=2)
        # Two well-separated clusters in latent space:
        #   cluster A near (0,0): indices 0 (energy 0.9), 1 (energy 0.1) -> winner = 1
        #   cluster B near (10,10): indices 2 (energy 0.5), 3 (energy 0.2) -> winner = 3
        latents = [_l(0, 0), _l(0.1, 0.1), _l(10, 10), _l(10.1, 10.1)]
        energies = [0.9, 0.1, 0.5, 0.2]
        out = select_diverse(latents, energies, cfg)
        assert sorted(out) == [1, 3]

    def test_deterministic(self) -> None:
        cfg = DiverseSelectionConfig(num_groups=3, min_candidates=2)
        latents = [_l(i, i * 0.5) for i in range(12)]
        energies = [float(((i * 7) % 5) / 10.0) for i in range(12)]
        out1 = select_diverse(latents, energies, cfg)
        out2 = select_diverse(latents, energies, cfg)
        assert out1 == out2

    def test_diverse_winner_can_beat_lowest_energy_only(self) -> None:
        """If three of four candidates collapse onto the same wrong answer
        cluster, DVTS should still surface the diverse minority."""
        cfg = DiverseSelectionConfig(num_groups=2, min_candidates=2)
        latents = [
            _l(0, 0),  # majority cluster
            _l(0.05, 0.05),
            _l(0.1, 0.0),
            _l(50, 50),  # lone diverse candidate
        ]
        energies = [0.10, 0.05, 0.20, 0.40]
        out = select_diverse(latents, energies, cfg)
        # Survivor list must include the diverse outlier despite higher energy
        assert 3 in out
        assert len(out) == 2
