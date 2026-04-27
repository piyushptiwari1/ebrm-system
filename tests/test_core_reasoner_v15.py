"""Integration tests for v0.15 MCTS-seeded refinement."""

from __future__ import annotations

import numpy as np

from ebrm_system.core import (
    HierarchicalLatentReasoner,
    MCTSConfig,
    ReasonerConfig,
    RefinementConfig,
)
from ebrm_system.intent import Intent, IntentPrediction
from ebrm_system.verifiers.base import VerificationResult

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
        suggested_langevin_steps=10,
        suggested_restarts=1,
    )


class _AlwaysFail:
    name = "always_fail"

    def check(self, _candidate, _ctx=None) -> VerificationResult:
        return VerificationResult(
            verifier=self.name,
            verified=False,
            confidence=0.0,
            reason="forced rejection",
        )


class TestRefinementConfigUseMctsSeed:
    def test_default_disabled(self) -> None:
        cfg = RefinementConfig()
        assert cfg.use_mcts_seed is False

    def test_can_enable(self) -> None:
        cfg = RefinementConfig(max_rounds=1, use_mcts_seed=True)
        assert cfg.use_mcts_seed is True


class TestMctsSeededRefinement:
    def _make(self, *, use_mcts_seed: bool) -> HierarchicalLatentReasoner:
        return HierarchicalLatentReasoner(
            encoder=_encoder_for(5.0),
            decoder=_decoder,
            energy_fn=_energy_for(5.0),
            classifier=_FixedClassifier(_pred()),
            extra_verifiers=[_AlwaysFail()],
            config=ReasonerConfig(
                seed=0,
                mcts=MCTSConfig(num_simulations=8, num_clusters=2),
                refinement=RefinementConfig(
                    max_rounds=1,
                    trigger_threshold=1.0,
                    use_mcts_seed=use_mcts_seed,
                ),
            ),
        )

    def test_disabled_runs_normally(self) -> None:
        reasoner = self._make(use_mcts_seed=False)
        result = reasoner.solve("Q")
        assert result.details["refinement_rounds"] >= 1

    def test_enabled_runs_and_completes(self) -> None:
        reasoner = self._make(use_mcts_seed=True)
        result = reasoner.solve("Q")
        # Refinement still ran; MCTS-seeded path produced traces.
        assert result.details["refinement_rounds"] >= 1
        assert len(result.traces) > 0


class TestReasonOnceSeedLatent:
    def test_mcts_top1_returns_none_when_disabled(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(0.0),
            decoder=_decoder,
            energy_fn=_energy_for(0.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(seed=0),
        )
        assert reasoner._mcts_top1_latent([]) is None

    def test_mcts_top1_returns_latent_when_enabled(self) -> None:
        reasoner = HierarchicalLatentReasoner(
            encoder=_encoder_for(0.0),
            decoder=_decoder,
            energy_fn=_energy_for(0.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(
                seed=0,
                mcts=MCTSConfig(num_simulations=4, num_clusters=2),
            ),
        )
        result = reasoner.solve("Q")
        top = reasoner._mcts_top1_latent(list(result.traces))
        assert top is not None
        assert top.shape == (D,)

    def test_seed_latent_param_overrides_encoder(self) -> None:
        """When seed_latent is passed to _reason_once, encoder is bypassed."""
        from ebrm_system.core import ComputeProfile
        from ebrm_system.core.compute_profile import ScaledBudget

        encoder_calls: list[str] = []

        def tracking_encoder(q: str) -> np.ndarray:
            encoder_calls.append(q)
            return np.zeros(D, dtype=np.float32)

        reasoner = HierarchicalLatentReasoner(
            encoder=tracking_encoder,
            decoder=_decoder,
            energy_fn=_energy_for(3.0),
            classifier=_FixedClassifier(_pred()),
            config=ReasonerConfig(seed=0),
        )
        custom_seed = np.zeros(D, dtype=np.float32)
        custom_seed[0] = 3.0
        budget = ScaledBudget(
            num_candidates=4, num_steps=5, num_restarts=1, profile=ComputeProfile.BALANCED
        )
        n_before = len(encoder_calls)
        traces, _ = reasoner._reason_once(
            "Q", _pred(), budget, seed_offset=0, seed_latent=custom_seed
        )
        assert len(encoder_calls) == n_before  # encoder not called
        assert len(traces) > 0
