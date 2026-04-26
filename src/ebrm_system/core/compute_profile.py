"""Difficulty-adaptive compute profile.

Scales the per-intent compute budget (Langevin steps, restart count, trace
count) up or down based on the user's chosen profile.

Motivated by the test-time-compute scaling literature
(Snell et al., HuggingFaceH4 search-and-learn 2024; "The Reasoning Model
Revolution: How Test-Time Compute Is Reshaping AI in 2026"). The optimal
inference cost depends on problem difficulty: easy problems should get
N=1 greedy decoding; hard problems benefit superlinearly from extra samples.

This module is pure-Python, side-effect free, and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ebrm_system.intent import IntentPrediction


class ComputeProfile(str, Enum):
    """How aggressively to scale inference compute.

    The reasoner multiplies the intent classifier's *suggested* budgets by a
    profile-dependent factor. ``balanced`` is a no-op; ``economy`` collapses
    easy questions to a single greedy decode; ``max_quality`` doubles the
    candidate count and Langevin steps for difficulty >= 0.5.
    """

    ECONOMY = "economy"
    BALANCED = "balanced"
    MAX_QUALITY = "max_quality"


@dataclass(frozen=True)
class ScaledBudget:
    """Output of :func:`scale_budget` — concrete compute knobs."""

    num_candidates: int
    num_steps: int
    num_restarts: int
    profile: ComputeProfile
    """The profile that produced these numbers (for audit)."""


def scale_budget(
    prediction: IntentPrediction,
    profile: ComputeProfile = ComputeProfile.BALANCED,
) -> ScaledBudget:
    """Scale the intent's suggested compute budget by the chosen profile.

    Parameters
    ----------
    prediction
        Output of the intent classifier (carries ``difficulty`` and the three
        ``suggested_*`` knobs).
    profile
        Compute aggressiveness. Default is :attr:`ComputeProfile.BALANCED`,
        which returns the classifier's suggestions unchanged.

    Returns
    -------
    ScaledBudget
        Concrete (candidates, steps, restarts) triple, all >= 1.

    Notes
    -----
    Rules:

    * **economy** — for ``difficulty < 0.3``, force N=1 / 1 restart and shrink
      Langevin steps to one quarter of the suggestion. Above the threshold,
      keep the suggestion (we never starve hard problems).
    * **balanced** — pass through.
    * **max_quality** — for ``difficulty >= 0.5`` double candidates and steps
      and add 50% restarts (capped at 32 / 4000 / 16). Cheap problems are
      left alone — we only spend extra compute where it pays.
    """
    n = prediction.suggested_trace_count
    s = prediction.suggested_langevin_steps
    r = prediction.suggested_restarts
    d = prediction.difficulty

    if profile is ComputeProfile.ECONOMY and d < 0.3:
        n, s, r = 1, max(10, s // 4), 1
    elif profile is ComputeProfile.MAX_QUALITY and d >= 0.5:
        n = min(32, n * 2)
        s = min(4000, s * 2)
        r = min(16, max(r, 1) * 3 // 2)

    return ScaledBudget(
        num_candidates=max(1, n),
        num_steps=max(1, s),
        num_restarts=max(1, r),
        profile=profile,
    )


__all__ = ["ComputeProfile", "ScaledBudget", "scale_budget"]
