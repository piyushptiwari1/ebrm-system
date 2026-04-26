"""Integration tests for v0.8 Coconut latent recursion in HierarchicalLatentReasoner."""

from __future__ import annotations

import numpy as np

from ebrm_system.core import (
    HierarchicalLatentReasoner,
    ReasonerConfig,
    RecursionConfig,
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


def _pred() -> IntentPrediction:
    return IntentPrediction(
        intent=Intent.UNKNOWN,
        difficulty=0.5,
        reasoning="x",
        suggested_trace_count=4,
        suggested_langevin_steps=20,
        suggested_restarts=1,
    )


class TestLatentRecursionWiredIn:
    def test_disabled_by_default_reports_none(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(seed=0),
        )
        result = reasoner.solve("Q")
        assert result.details["latent_recursion"] is None

    def test_enabled_records_audit(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(5.0),
            decoder=_decoder,
            energy_fn=_energy_for(0.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(
                seed=0,
                latent_recursion=RecursionConfig(max_steps=4, step_size=0.05),
            ),
        )
        result = reasoner.solve("Q")
        audit = result.details["latent_recursion"]
        assert audit is not None
        assert audit["steps_run"] == 4
        assert audit["halted_early"] is False
        # Energy is well-defined on a quadratic bowl: descent should drop it.
        assert audit["energy_end"] < audit["energy_start"]

    def test_max_steps_zero_is_disabled(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(
                seed=0,
                latent_recursion=RecursionConfig(max_steps=0),
            ),
        )
        result = reasoner.solve("Q")
        assert result.details["latent_recursion"] is None

    def test_custom_step_fn_is_used(self) -> None:
        calls: list[int] = []

        def custom_step(z: np.ndarray, step: int) -> np.ndarray:
            calls.append(step)
            return z * 0.5

        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(8.0),
            decoder=_decoder,
            energy_fn=_energy_for(0.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(
                seed=0,
                latent_recursion=RecursionConfig(max_steps=3),
            ),
            recursion_step_fn=custom_step,
        )
        result = reasoner.solve("Q")
        assert calls == [0, 1, 2]
        audit = result.details["latent_recursion"]
        assert audit["steps_run"] == 3
