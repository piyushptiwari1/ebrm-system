"""Integration tests for v0.6 features wired into HierarchicalLatentReasoner.

Covers:
* Compute profile actually changes the produced trace count.
* Refinement loop fires when verifiers reject and adds extra traces.
* Refinement is a no-op when no verifier rejected anything.
"""

from __future__ import annotations

import numpy as np

from ebrm_system.core import (
    ComputeProfile,
    HierarchicalLatentReasoner,
    ReasonerConfig,
    RefinementConfig,
)
from ebrm_system.intent import Intent, IntentPrediction
from ebrm_system.verifiers import routing
from ebrm_system.verifiers.base import VerificationResult, Verifier

D = 8


# -------------------------------------------------------------------------- #
# Mocks                                                                       #
# -------------------------------------------------------------------------- #


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
    """Returns a fixed prediction (lets us drive difficulty in tests)."""

    def __init__(self, prediction: IntentPrediction) -> None:
        self._p = prediction

    def classify(self, _question: str) -> IntentPrediction:
        return self._p


class _AlwaysReject(Verifier):
    name = "always_reject"

    def check(self, candidate: object, context=None) -> VerificationResult:
        return VerificationResult(
            verifier=self.name,
            verified=False,
            confidence=0.0,
            reason=f"answer '{candidate}' is not allowed",
        )


# -------------------------------------------------------------------------- #
# Compute profile                                                             #
# -------------------------------------------------------------------------- #


class TestComputeProfileWiredIn:
    def test_economy_collapses_easy_question_to_one_trace(self) -> None:
        pred = IntentPrediction(
            intent=Intent.UNKNOWN,  # UNKNOWN intent has no verifier chain
            difficulty=0.1,
            reasoning="trivial",
            suggested_trace_count=8,
            suggested_langevin_steps=200,
            suggested_restarts=4,
        )
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(3.0),
            decoder=_decoder,
            energy_fn=_energy_for(3.0),
            classifier=_FixedClassifier(pred),
            config=ReasonerConfig(seed=0, compute_profile=ComputeProfile.ECONOMY),
        )
        result = reasoner.solve("trivial Q")
        assert result.details["budget"]["num_candidates"] == 1
        assert len(result.traces) == 1

    def test_max_quality_doubles_hard_question(self) -> None:
        pred = IntentPrediction(
            intent=Intent.UNKNOWN,
            difficulty=0.8,
            reasoning="hard",
            suggested_trace_count=4,
            suggested_langevin_steps=200,
            suggested_restarts=2,
        )
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(3.0),
            decoder=_decoder,
            energy_fn=_energy_for(3.0),
            classifier=_FixedClassifier(pred),
            config=ReasonerConfig(seed=0, compute_profile=ComputeProfile.MAX_QUALITY),
        )
        result = reasoner.solve("hard Q")
        assert result.details["budget"]["num_candidates"] == 8


# -------------------------------------------------------------------------- #
# Refinement                                                                  #
# -------------------------------------------------------------------------- #


class TestRefinementWiredIn:
    def test_refinement_disabled_by_default(self) -> None:
        pred = IntentPrediction(
            intent=Intent.UNKNOWN,
            difficulty=0.5,
            reasoning="x",
            suggested_trace_count=3,
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
        assert result.details["refinement_rounds"] == 0

    def test_refinement_fires_when_verifier_rejects_all(self, monkeypatch) -> None:
        # Force the routing to use our always-rejecting verifier
        from ebrm_system.verifiers.base import VerifierChain

        def fake_chain_for_intent(_intent):
            return VerifierChain([_AlwaysReject()])

        monkeypatch.setattr(routing, "chain_for_intent", fake_chain_for_intent)
        monkeypatch.setattr(
            "ebrm_system.core.reasoner.chain_for_intent",
            fake_chain_for_intent,
        )

        pred = IntentPrediction(
            intent=Intent.MATH_REASONING,
            difficulty=0.5,
            reasoning="x",
            suggested_trace_count=3,
            suggested_langevin_steps=50,
            suggested_restarts=1,
        )
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(7.0),
            decoder=_decoder,
            energy_fn=_energy_for(7.0),
            classifier=_FixedClassifier(pred),
            config=ReasonerConfig(
                seed=0,
                refinement=RefinementConfig(max_rounds=2, trigger_threshold=1.0),
            ),
        )
        result = reasoner.solve("Q")
        # All candidates rejected → at least one refinement round should fire
        assert result.details["refinement_rounds"] >= 1
        # Total traces = round0 + at least one extra round
        assert len(result.traces) > 3
        assert result.verified_fraction == 0.0
