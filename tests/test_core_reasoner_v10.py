"""Integration tests for v0.10 ReST-MCTS* in HierarchicalLatentReasoner."""

from __future__ import annotations

import numpy as np

from ebrm_system.core import (
    HierarchicalLatentReasoner,
    MCTSConfig,
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

    def classify(self, _q: str) -> IntentPrediction:
        return self._p


def _pred() -> IntentPrediction:
    return IntentPrediction(
        intent=Intent.UNKNOWN,
        difficulty=0.5,
        reasoning="x",
        suggested_trace_count=6,
        suggested_langevin_steps=20,
        suggested_restarts=1,
    )


class TestMCTSWiredIn:
    def test_disabled_by_default(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(seed=0),
        )
        result = reasoner.solve("Q")
        assert result.details["mcts"] is None

    def test_enabled_records_audit(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(3.0),
            decoder=_decoder,
            energy_fn=_energy_for(3.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(
                seed=0,
                mcts=MCTSConfig(num_simulations=8, num_clusters=2),
            ),
        )
        result = reasoner.solve("Q")
        audit = result.details["mcts"]
        assert audit is not None
        assert audit["simulations_run"] == 8
        assert audit["pool_size"] >= 1
        assert len(audit["top_visits"]) >= 1

    def test_custom_value_fn_is_used(self) -> None:
        called: list[float] = []

        def vfn(trace) -> float:
            called.append(trace.energy)
            return 1.0 - min(1.0, trace.energy)

        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(4.0),
            decoder=_decoder,
            energy_fn=_energy_for(4.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(
                seed=0,
                mcts=MCTSConfig(num_simulations=6, num_clusters=2),
            ),
            mcts_value_fn=vfn,
        )
        reasoner.solve("Q")
        assert len(called) >= 1
