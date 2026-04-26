"""Tests for the multi-seed candidate generator."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.inference.candidates import (
    CandidateConfig,
    generate_candidates,
)
from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex


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


def test_warmstart_k_validation() -> None:
    with pytest.raises(ValueError):
        CandidateConfig(num_candidates=4, warmstart_k=-1)
    with pytest.raises(ValueError):
        CandidateConfig(num_candidates=4, warmstart_k=5)


def test_warmstart_seeds_first_k_candidates() -> None:
    """When warmstart_k > 0 and index has matching latents, the first k
    candidates should be flagged as warm-started."""
    rng = np.random.default_rng(0)
    seed_latent = rng.standard_normal(8).astype(np.float32)

    # Build an index with 3 cached latents that are *near* the seed.
    cached = [seed_latent + 0.01 * rng.standard_normal(8).astype(np.float32) for _ in range(3)]
    idx = LatentIndex(IndexConfig(in_dim=8, bits=256, seed=0))
    idx.add(np.stack(cached), list(cached))  # payloads are the latents themselves

    cfg = CandidateConfig(num_candidates=5, num_steps=2, step_size=0.05, warmstart_k=3, seed=0)
    cands = generate_candidates(seed_latent, quadratic_energy, cfg, index=idx)
    warm_count = sum(1 for c in cands if c.warmstart)
    assert warm_count == 3


def test_warmstart_with_no_index_falls_back_to_gaussian() -> None:
    seed_latent = np.zeros(4, dtype=np.float32)
    cfg = CandidateConfig(num_candidates=4, num_steps=1, warmstart_k=2, seed=0)
    cands = generate_candidates(seed_latent, quadratic_energy, cfg, index=None)
    assert all(not c.warmstart for c in cands)


def test_warmstart_with_empty_index_falls_back_to_gaussian() -> None:
    seed_latent = np.zeros(4, dtype=np.float32)
    idx = LatentIndex(IndexConfig(in_dim=4, bits=128))
    cfg = CandidateConfig(num_candidates=4, num_steps=1, warmstart_k=2, seed=0)
    cands = generate_candidates(seed_latent, quadratic_energy, cfg, index=idx)
    assert all(not c.warmstart for c in cands)


def test_warmstart_skips_payloads_with_wrong_shape() -> None:
    """Payloads that aren't ndarrays of the right shape are silently dropped."""
    seed_latent = np.zeros(4, dtype=np.float32)
    idx = LatentIndex(IndexConfig(in_dim=4, bits=128))
    idx.add(np.zeros((2, 4), dtype=np.float32), ["string-payload-1", "string-payload-2"])
    cfg = CandidateConfig(num_candidates=3, num_steps=1, warmstart_k=2, seed=0)
    cands = generate_candidates(seed_latent, quadratic_energy, cfg, index=idx)
    assert all(not c.warmstart for c in cands)
