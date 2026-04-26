"""Integration tests for v0.7 features wired into HierarchicalLatentReasoner."""

from __future__ import annotations

import numpy as np

from ebrm_system.core import (
    DiverseSelectionConfig,
    HierarchicalLatentReasoner,
    ReasonerConfig,
)
from ebrm_system.intent import Intent, IntentPrediction

D = 8


def _encoder_for(value: float):
    def _enc(_: str) -> np.ndarray:
        v = np.zeros(D, dtype=np.float32)
        v[0] = value
        return v

    return _enc


def _decoder(latent: np.ndarray) -> str:
    return str(round(float(latent[0])))


def _energy_for(target: float):
    def _e(latent: np.ndarray) -> float:
        return float((latent[0] - target) ** 2)

    return _e


class _FixedClassifier:
    def __init__(self, prediction: IntentPrediction) -> None:
        self._p = prediction

    def classify(self, _question: str) -> IntentPrediction:
        return self._p


class TestDiverseSelectionWiredIn:
    def test_disabled_by_default_reports_none(self) -> None:
        pred = IntentPrediction(
            intent=Intent.UNKNOWN,
            difficulty=0.5,
            reasoning="x",
            suggested_trace_count=8,
            suggested_langevin_steps=50,
            suggested_restarts=1,
        )
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(pred),
            config=ReasonerConfig(seed=0),
        )
        result = reasoner.solve("Q")
        assert result.details["diverse_selection"] is None

    def test_enabled_records_survivor_count(self) -> None:
        pred = IntentPrediction(
            intent=Intent.UNKNOWN,
            difficulty=0.5,
            reasoning="x",
            suggested_trace_count=8,
            suggested_langevin_steps=50,
            suggested_restarts=1,
        )
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(pred),
            config=ReasonerConfig(
                seed=0,
                diverse_selection=DiverseSelectionConfig(num_groups=3, min_candidates=2),
            ),
        )
        result = reasoner.solve("Q")
        info = result.details["diverse_selection"]
        assert info is not None
        assert info["input"] == 8
        assert 1 <= info["survivors"] <= 3

    def test_diverse_survivors_subset_of_traces(self) -> None:
        """The voted answer must come from the trace pool."""
        pred = IntentPrediction(
            intent=Intent.UNKNOWN,
            difficulty=0.5,
            reasoning="x",
            suggested_trace_count=6,
            suggested_langevin_steps=30,
            suggested_restarts=1,
        )
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(7.0),
            decoder=_decoder,
            energy_fn=_energy_for(7.0),
            classifier=_FixedClassifier(pred),
            config=ReasonerConfig(
                seed=0,
                diverse_selection=DiverseSelectionConfig(num_groups=2, min_candidates=2),
            ),
        )
        result = reasoner.solve("Q")
        all_answers = {t.answer for t in result.traces}
        assert str(result.answer) in all_answers
