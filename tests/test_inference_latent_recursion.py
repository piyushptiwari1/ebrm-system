"""Tests for Coconut-style latent recursion (v0.8)."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.inference.halt import PlateauHalt
from ebrm_system.inference.latent_recursion import (
    RecursionConfig,
    gradient_step,
    recurse_latent,
)


def _quadratic_energy(z: np.ndarray) -> float:
    """Bowl centred at origin; gradient = 2z."""
    return float(np.sum(z * z))


class TestRecursionConfig:
    def test_defaults_disable_recursion(self) -> None:
        cfg = RecursionConfig()
        assert cfg.max_steps == 0

    def test_rejects_negative_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            RecursionConfig(max_steps=-1)

    def test_rejects_non_positive_step_size(self) -> None:
        with pytest.raises(ValueError, match="step_size"):
            RecursionConfig(step_size=0.0)

    def test_rejects_non_positive_fd_eps(self) -> None:
        with pytest.raises(ValueError, match="fd_eps"):
            RecursionConfig(fd_eps=0.0)


class TestRecurseLatent:
    def test_max_steps_zero_is_noop(self) -> None:
        seed = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        step_fn = gradient_step(_quadratic_energy)
        result = recurse_latent(seed, step_fn, config=RecursionConfig(max_steps=0))
        assert result.steps_run == 0
        assert result.halted_early is False
        assert result.energy_trajectory == ()
        np.testing.assert_array_equal(result.latent, seed)

    def test_rejects_non_1d_seed(self) -> None:
        seed = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            recurse_latent(
                seed,
                gradient_step(_quadratic_energy),
                config=RecursionConfig(max_steps=1),
            )

    def test_descends_quadratic_bowl(self) -> None:
        seed = np.array([5.0, -4.0, 3.0], dtype=np.float32)
        step_fn = gradient_step(_quadratic_energy, step_size=0.1)
        result = recurse_latent(
            seed,
            step_fn,
            config=RecursionConfig(max_steps=20),
            energy_fn=_quadratic_energy,
        )
        assert result.steps_run == 20
        assert result.energy_trajectory[-1] < result.energy_trajectory[0]
        assert np.linalg.norm(result.latent) < np.linalg.norm(seed)

    def test_trajectory_length_matches_steps(self) -> None:
        seed = np.array([1.0, 0.0], dtype=np.float32)
        step_fn = gradient_step(_quadratic_energy)
        result = recurse_latent(
            seed,
            step_fn,
            config=RecursionConfig(max_steps=5),
            energy_fn=_quadratic_energy,
        )
        # +1 because we record the seed energy too.
        assert len(result.energy_trajectory) == result.steps_run + 1

    def test_no_trajectory_without_energy_fn(self) -> None:
        seed = np.array([1.0, 0.0], dtype=np.float32)
        step_fn = gradient_step(_quadratic_energy)
        result = recurse_latent(
            seed,
            step_fn,
            config=RecursionConfig(max_steps=3),
        )
        assert result.energy_trajectory == ()
        assert result.steps_run == 3

    def test_plateau_halt_fires_early(self) -> None:
        # Step that lands on the minimum after one move, then plateaus.
        def constant_step(z: np.ndarray, _: int) -> np.ndarray:
            return np.zeros_like(z)

        seed = np.array([2.0, 2.0], dtype=np.float32)
        halt = PlateauHalt(window=3, threshold=1e-6, min_steps=1)
        result = recurse_latent(
            seed,
            constant_step,
            config=RecursionConfig(max_steps=50),
            energy_fn=_quadratic_energy,
            halt_policy=halt,
        )
        assert result.halted_early is True
        assert result.steps_run < 50

    def test_deterministic_given_step_fn(self) -> None:
        seed = np.array([1.5, -2.5, 0.5], dtype=np.float32)
        step_fn = gradient_step(_quadratic_energy)
        cfg = RecursionConfig(max_steps=7)
        a = recurse_latent(seed, step_fn, config=cfg)
        b = recurse_latent(seed, step_fn, config=cfg)
        np.testing.assert_array_equal(a.latent, b.latent)


class TestGradientStep:
    def test_step_decreases_energy(self) -> None:
        z = np.array([3.0, -2.0], dtype=np.float32)
        step_fn = gradient_step(_quadratic_energy, step_size=0.05)
        z_next = step_fn(z, 0)
        assert _quadratic_energy(z_next) < _quadratic_energy(z)

    def test_returns_float32(self) -> None:
        z = np.array([1.0, 2.0], dtype=np.float32)
        out = gradient_step(_quadratic_energy)(z, 0)
        assert out.dtype == np.float32
