"""Tests for the torch-native Langevin step."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ebrm_system.inference.torch_langevin import (  # noqa: E402
    generate_torch_candidates,
    torch_langevin_step,
)


def quadratic_energy(s: torch.Tensor) -> torch.Tensor:
    """Convex bowl with minimum at the origin."""
    return (s * s).sum()


def test_torch_langevin_step_descends_on_quadratic() -> None:
    s = torch.tensor([2.0, 2.0, 2.0])
    gen = torch.Generator().manual_seed(0)
    e0 = quadratic_energy(s).item()
    for _ in range(20):
        s = torch_langevin_step(
            s, quadratic_energy, step_size=0.1, noise_scale=0.001, generator=gen
        )
    assert quadratic_energy(s).item() < e0


def test_torch_langevin_step_rejects_non_scalar_energy() -> None:
    s = torch.zeros(3)
    with pytest.raises(ValueError, match="0-dim"):
        torch_langevin_step(s, lambda x: x * 2.0, step_size=0.1, noise_scale=0.0)


def test_generate_torch_candidates_count_and_sort() -> None:
    seed_latent = torch.tensor([1.0, 1.0, 1.0])
    cands = generate_torch_candidates(
        seed_latent, quadratic_energy, num_candidates=5, num_steps=10, step_size=0.1, seed=0
    )
    assert len(cands) == 5
    energies = [c.energy for c in cands]
    assert energies == sorted(energies)


def test_generate_torch_candidates_descend_energy() -> None:
    seed_latent = torch.tensor([2.0, 2.0, 2.0])
    cands = generate_torch_candidates(
        seed_latent,
        quadratic_energy,
        num_candidates=8,
        num_steps=50,
        step_size=0.1,
        noise_scale=0.01,
        seed=1,
    )
    assert cands[0].energy < quadratic_energy(seed_latent).item()


def test_generate_torch_candidates_distinct_seeds() -> None:
    seed_latent = torch.zeros(4)
    cands = generate_torch_candidates(
        seed_latent, quadratic_energy, num_candidates=6, num_steps=2, seed=7
    )
    seeds = [c.seed for c in cands]
    assert len(set(seeds)) == len(seeds)


def test_generate_torch_candidates_reproducible() -> None:
    seed_latent = torch.zeros(4)
    a = generate_torch_candidates(
        seed_latent, quadratic_energy, num_candidates=3, num_steps=5, seed=42
    )
    b = generate_torch_candidates(
        seed_latent, quadratic_energy, num_candidates=3, num_steps=5, seed=42
    )
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x.latent, y.latent)
        assert x.energy == y.energy
        assert x.seed == y.seed


def test_generate_torch_candidates_validation() -> None:
    seed_latent = torch.zeros(2)
    with pytest.raises(ValueError):
        generate_torch_candidates(seed_latent, quadratic_energy, num_candidates=0)
    with pytest.raises(ValueError):
        generate_torch_candidates(seed_latent, quadratic_energy, num_steps=-1)


def test_generate_torch_candidates_preserves_dtype_and_device() -> None:
    seed_latent = torch.zeros(3, dtype=torch.float64)
    cands = generate_torch_candidates(
        seed_latent, quadratic_energy, num_candidates=2, num_steps=2, seed=0
    )
    for c in cands:
        assert c.latent.dtype == torch.float64
        assert c.latent.device == seed_latent.device
