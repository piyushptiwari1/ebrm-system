"""Tests for the multi-seed candidate generator."""

from __future__ import annotations

import numpy as np

from ebrm_system.inference.candidates import (
    CandidateConfig,
    generate_candidates,
)


def quadratic_energy(s: np.ndarray) -> float:
    """Simple convex bowl with minimum at the origin."""
    return float(np.sum(s * s))


def test_generate_candidates_count_and_sort() -> None:
    seed_latent = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    cfg = CandidateConfig(num_candidates=5, num_steps=10, step_size=0.1, seed=0)
    cands = generate_candidates(seed_latent, quadratic_energy, cfg)
    assert len(cands) == 5
    energies = [c.energy for c in cands]
    assert energies == sorted(energies)


def test_generate_candidates_descend_energy() -> None:
    """After enough Langevin steps the best candidate should beat the seed."""
    seed_latent = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    cfg = CandidateConfig(num_candidates=8, num_steps=50, step_size=0.1, noise_scale=0.01, seed=1)
    cands = generate_candidates(seed_latent, quadratic_energy, cfg)
    assert cands[0].energy < quadratic_energy(seed_latent)


def test_generate_candidates_distinct_seeds() -> None:
    seed_latent = np.zeros(4, dtype=np.float32)
    cfg = CandidateConfig(num_candidates=6, num_steps=2, seed=7)
    cands = generate_candidates(seed_latent, quadratic_energy, cfg)
    seeds = [c.seed for c in cands]
    assert len(set(seeds)) == len(seeds)


def test_candidate_config_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        CandidateConfig(num_candidates=0)
    with pytest.raises(ValueError):
        CandidateConfig(num_steps=-1)
