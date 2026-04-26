"""Tests for v0.12 PRM-guided refinement (extra_verifiers + context flow)."""

from __future__ import annotations

import numpy as np

from ebrm_system.core import (
    HierarchicalLatentReasoner,
    ReasonerConfig,
    RefinementConfig,
)
from ebrm_system.intent import Intent, IntentPrediction
from ebrm_system.verifiers.prm import GenerativePRMVerifier, PRMVerdict

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
        suggested_langevin_steps=15,
        suggested_restarts=1,
    )


class TestExtraVerifiers:
    def test_extra_verifiers_run_in_chain(self) -> None:
        seen_questions: list[str] = []

        def prm_fn(question: str, candidate: object) -> PRMVerdict:
            seen_questions.append(question)
            return PRMVerdict(verified=True, confidence=0.9, reasoning="looks good")

        prm = GenerativePRMVerifier(prm_fn)
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(seed=0),
            extra_verifiers=[prm],
        )
        result = reasoner.solve("What is 2?")
        assert seen_questions, "PRM verifier should have been called"
        assert all(q == "What is 2?" for q in seen_questions)
        # All traces must record the PRM verifier's result.
        assert all(any(r.verifier == prm.name for r in t.verifier_results) for t in result.traces)

    def test_prm_critique_drives_refinement(self) -> None:
        """When the PRM rejects, refinement should run with the PRM critique."""
        rejection_reason = "intermediate step missed a unit conversion"

        def prm_fn(_question: str, _candidate: object) -> PRMVerdict:
            return PRMVerdict(verified=False, confidence=0.1, reasoning=rejection_reason)

        prm = GenerativePRMVerifier(prm_fn)
        seen_questions: list[str] = []

        def encoder(q: str) -> np.ndarray:
            seen_questions.append(q)
            v = np.zeros(D, dtype=np.float32)
            v[0] = 1.0
            return v

        reasoner = HierarchicalLatentReasoner(
            encoder=encoder,
            decoder=_decoder,
            energy_fn=_energy_for(1.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(
                seed=0,
                refinement=RefinementConfig(
                    max_rounds=1,
                    trigger_threshold=0.99,
                    max_critiques=2,
                ),
            ),
            extra_verifiers=[prm],
        )
        result = reasoner.solve("Original question")
        # Round 0 + at least one refinement round.
        assert result.details["refinement_rounds"] >= 1
        # The PRM's reasoning must appear in the refined question.
        assert any(rejection_reason in q for q in seen_questions[1:])

    def test_no_extra_verifiers_is_default_behaviour(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(2.0),
            decoder=_decoder,
            energy_fn=_energy_for(2.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(seed=0),
        )
        result = reasoner.solve("Q")
        # No extra verifiers attached; UNKNOWN intent has no chain → no
        # verifier results.
        assert all(t.verifier_results == () for t in result.traces)
