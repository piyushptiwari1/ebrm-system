"""Tests for v0.11 episodic write-back to the index after solve."""

from __future__ import annotations

import numpy as np

from ebrm_system.core import HierarchicalLatentReasoner, ReasonerConfig
from ebrm_system.intent import Intent, IntentPrediction
from ebrm_system.memory import TieredMemory, TieredMemoryConfig
from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex

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
        suggested_trace_count=4,
        suggested_langevin_steps=20,
        suggested_restarts=1,
    )


class TestWriteBack:
    def test_disabled_by_default(self) -> None:
        index = LatentIndex(IndexConfig(in_dim=D))
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(_pred()),
            index=index,
            config=ReasonerConfig(seed=0),
        )
        result = reasoner.solve("Q")
        assert result.details["memory_write"] is None
        assert len(index) == 0

    def test_writes_into_latent_index_when_enabled(self) -> None:
        index = LatentIndex(IndexConfig(in_dim=D))
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(3.0),
            decoder=_decoder,
            energy_fn=_energy_for(3.0),
            classifier=_FixedClassifier(_pred()),
            index=index,
            config=ReasonerConfig(seed=0, learn_from_solves=True),
        )
        result = reasoner.solve("Q")
        audit = result.details["memory_write"]
        assert audit is not None
        assert audit["written"] is True
        assert len(index) == 1

    def test_writes_into_tiered_memory_when_enabled(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=D))
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(4.0),
            decoder=_decoder,
            energy_fn=_energy_for(4.0),
            classifier=_FixedClassifier(_pred()),
            index=mem,  # type: ignore[arg-type]
            config=ReasonerConfig(seed=0, learn_from_solves=True),
        )
        result = reasoner.solve("Q")
        assert result.details["memory_write"]["written"] is True
        assert len(mem) == 1

    def test_no_index_is_safe(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(seed=0, learn_from_solves=True),
        )
        result = reasoner.solve("Q")
        # No index attached → write-back must be a clean no-op.
        assert result.details["memory_write"] is None

    def test_subsequent_solve_can_warmstart_from_written_back_latent(self) -> None:
        index = LatentIndex(IndexConfig(in_dim=D))
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(5.0),
            decoder=_decoder,
            energy_fn=_energy_for(5.0),
            classifier=_FixedClassifier(_pred()),
            index=index,
            config=ReasonerConfig(seed=0, learn_from_solves=True),
        )
        reasoner.solve("Q1")
        first_count = len(index)
        reasoner.solve("Q2")
        assert len(index) > first_count
