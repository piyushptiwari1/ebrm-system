"""Tests for ``ebrm_system.inference.halt`` policies and integration with
``generate_candidates``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.inference.candidates import CandidateConfig, generate_candidates
from ebrm_system.inference.halt import NeverHalt, PlateauHalt


def _bowl_energy(target: float = 0.0):
    """Return an energy fn whose minimum is at ``target`` along axis 0."""

    def _e(x):
        return float((x[0] - target) ** 2)

    return _e


def test_never_halt_runs_full_trajectory() -> None:
    cfg = CandidateConfig(num_candidates=2, num_steps=20, seed=0)
    seed = np.zeros(4, dtype=np.float32)
    cands = generate_candidates(seed, _bowl_energy(), cfg, halt_policy=NeverHalt())
    assert all(c.steps_run == 20 for c in cands)


def test_plateau_halt_stops_early_in_a_basin() -> None:
    """A flat-energy fn should trigger PlateauHalt almost immediately."""

    def flat(_x):
        return 1.0  # constant → variance = 0 → plateau detected

    cfg = CandidateConfig(num_candidates=3, num_steps=200, seed=0)
    policy = PlateauHalt(window=4, threshold=1e-6, min_steps=4)
    seed = np.zeros(4, dtype=np.float32)
    cands = generate_candidates(seed, flat, cfg, halt_policy=policy)
    # Should halt right after min_steps once the buffer fills.
    assert all(0 < c.steps_run < 200 for c in cands)
    assert max(c.steps_run for c in cands) <= 10


def test_plateau_halt_respects_min_steps() -> None:
    """Even on a flat energy, halting cannot fire before ``min_steps``."""

    def flat(_x):
        return 1.0

    cfg = CandidateConfig(num_candidates=1, num_steps=100, seed=0)
    policy = PlateauHalt(window=2, threshold=1e-6, min_steps=15)
    seed = np.zeros(4, dtype=np.float32)
    cands = generate_candidates(seed, flat, cfg, halt_policy=policy)
    assert cands[0].steps_run >= 15


def test_plateau_halt_keeps_running_when_energy_moves() -> None:
    """A monotonically-changing energy should never look like a plateau."""

    counter = {"n": 0}

    def moving(_x):
        counter["n"] += 1
        return float(counter["n"])  # strictly increasing each call

    cfg = CandidateConfig(num_candidates=1, num_steps=30, seed=0)
    policy = PlateauHalt(window=4, threshold=0.5, min_steps=4)
    seed = np.zeros(4, dtype=np.float32)
    cands = generate_candidates(seed, moving, cfg, halt_policy=policy)
    assert cands[0].steps_run == 30


def test_plateau_halt_validates_args() -> None:
    with pytest.raises(ValueError, match="window"):
        PlateauHalt(window=1)
    with pytest.raises(ValueError, match="threshold"):
        PlateauHalt(threshold=-0.1)
    with pytest.raises(ValueError, match="min_steps"):
        PlateauHalt(min_steps=-1)


def test_default_no_halt_policy_preserves_old_behaviour() -> None:
    """Calling generate_candidates without halt_policy still runs num_steps."""
    cfg = CandidateConfig(num_candidates=2, num_steps=8, seed=0)
    seed = np.zeros(4, dtype=np.float32)
    cands = generate_candidates(seed, _bowl_energy(), cfg)  # no halt_policy
    assert all(c.steps_run == 8 for c in cands)
