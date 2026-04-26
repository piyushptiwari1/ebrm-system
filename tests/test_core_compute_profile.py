"""Unit tests for ``ebrm_system.core.compute_profile``."""

from __future__ import annotations

import pytest

from ebrm_system.core import ComputeProfile, ScaledBudget, scale_budget
from ebrm_system.intent import Intent, IntentPrediction


def _pred(difficulty: float, n: int = 8, s: int = 500, r: int = 4) -> IntentPrediction:
    return IntentPrediction(
        intent=Intent.MATH_REASONING,
        difficulty=difficulty,
        reasoning="test",
        suggested_trace_count=n,
        suggested_langevin_steps=s,
        suggested_restarts=r,
    )


class TestScaleBudgetBalanced:
    def test_balanced_is_passthrough(self) -> None:
        p = _pred(0.5, n=8, s=500, r=4)
        b = scale_budget(p, ComputeProfile.BALANCED)
        assert b.num_candidates == 8
        assert b.num_steps == 500
        assert b.num_restarts == 4
        assert b.profile is ComputeProfile.BALANCED


class TestScaleBudgetEconomy:
    def test_easy_collapses_to_single_candidate(self) -> None:
        p = _pred(0.1, n=8, s=500, r=4)
        b = scale_budget(p, ComputeProfile.ECONOMY)
        assert b.num_candidates == 1
        assert b.num_restarts == 1
        assert b.num_steps == max(10, 500 // 4)

    def test_hard_question_left_alone_in_economy(self) -> None:
        p = _pred(0.7, n=12, s=1000, r=6)
        b = scale_budget(p, ComputeProfile.ECONOMY)
        assert b.num_candidates == 12
        assert b.num_steps == 1000
        assert b.num_restarts == 6

    def test_economy_floor_for_steps(self) -> None:
        p = _pred(0.0, n=8, s=20, r=4)
        b = scale_budget(p, ComputeProfile.ECONOMY)
        assert b.num_steps >= 10  # floor


class TestScaleBudgetMaxQuality:
    def test_doubles_for_hard_questions(self) -> None:
        p = _pred(0.6, n=8, s=500, r=4)
        b = scale_budget(p, ComputeProfile.MAX_QUALITY)
        assert b.num_candidates == 16
        assert b.num_steps == 1000
        assert b.num_restarts == 6  # 4 * 3 // 2

    def test_easy_question_left_alone_in_max(self) -> None:
        p = _pred(0.2, n=2, s=100, r=1)
        b = scale_budget(p, ComputeProfile.MAX_QUALITY)
        assert b.num_candidates == 2
        assert b.num_steps == 100
        assert b.num_restarts == 1

    def test_caps_at_safety_limits(self) -> None:
        p = _pred(0.99, n=20, s=3000, r=12)
        b = scale_budget(p, ComputeProfile.MAX_QUALITY)
        assert b.num_candidates <= 32
        assert b.num_steps <= 4000
        assert b.num_restarts <= 16


class TestScaledBudgetInvariants:
    @pytest.mark.parametrize("profile", list(ComputeProfile))
    @pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_all_outputs_at_least_one(self, profile: ComputeProfile, difficulty: float) -> None:
        p = _pred(difficulty, n=1, s=1, r=1)
        b = scale_budget(p, profile)
        assert b.num_candidates >= 1
        assert b.num_steps >= 1
        assert b.num_restarts >= 1
        assert isinstance(b, ScaledBudget)
