"""Core reasoning primitives — the hierarchical latent reasoner.

This module wires the existing pieces of `ebrm-system` into a single
end-to-end reasoner:

    question
        ─► intent classifier  (hierarchy level 0: routing)
        ─► encoder             (hierarchy level 1: question → seed latent)
        ─► candidate generator (hierarchy level 2: latent samples)
            (optional QJL warm-start from a LatentIndex)
        ─► decoder             (hierarchy level 3: latent → answer string)
        ─► verifier chain      (hard checks per intent)
        ─► self-consistency vote
        ─► ReasoningResult with full audit trail

The encoder, decoder, and energy function are injected as callables so this
module is torch-optional and unit-testable on CPU. In production they wrap
the EBRM v2 heads (pooler, projector, energy_head, answer_decoder) loaded
from `piyushptiwari/ebrm-v2-qwen3-4b` on Hugging Face.

The whole thing is deterministic given a seed and pure-Python in its control
flow. No global state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ebrm_system.core.compute_profile import (
    ComputeProfile,
    ScaledBudget,
    scale_budget,
)
from ebrm_system.core.refinement import (
    RefinementConfig,
    build_refined_question,
    collect_critiques,
    should_refine,
)
from ebrm_system.inference.candidates import (
    Candidate as LatentCandidate,
)
from ebrm_system.inference.candidates import (
    CandidateConfig,
    generate_candidates,
)
from ebrm_system.inference.diverse_selector import (
    DiverseSelectionConfig,
    select_diverse,
)
from ebrm_system.intent import Classifier, IntentPrediction, RuleBasedClassifier
from ebrm_system.verifiers.base import VerificationResult, VerifierChain
from ebrm_system.verifiers.routing import chain_for_intent
from ebrm_system.voting import SelfConsistencyVoter, VoteResult
from ebrm_system.voting.voter import Candidate as VoteCandidate

if TYPE_CHECKING:
    from ebrm_system.reward.qjl_index import LatentIndex

LatentT = NDArray[np.float32]
EncoderFn = Callable[[str], LatentT]
"""Maps a question string to a seed latent (e.g. EBRM projector output)."""

DecoderFn = Callable[[LatentT], str]
"""Maps a candidate latent back to an answer string."""

EnergyFn = Callable[[LatentT], float]
"""Scalar energy of a latent (lower = more plausible)."""


@dataclass(frozen=True)
class TraceItem:
    """One reasoning trace: latent → decoded answer → verifier evidence."""

    latent: LatentT
    answer: str
    energy: float
    seed: int
    warmstart: bool
    verified: bool
    verifier_results: tuple[VerificationResult, ...]


@dataclass(frozen=True)
class ReasoningResult:
    """End-to-end output of the hierarchical reasoner."""

    answer: object
    intent: IntentPrediction
    vote: VoteResult
    traces: tuple[TraceItem, ...]
    verified_fraction: float
    """Share of candidates that passed every hard verifier in the chain."""
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasonerConfig:
    """Fixed knobs for the reasoner (compute-budget hints come from intent)."""

    weight_by: str = "inverse_energy"
    """Voting weight: 'uniform' | 'confidence' | 'inverse_energy'."""
    numerical_tolerance: float | None = None
    """If set, bucket numeric answers within this absolute tolerance."""
    require_verification: bool = False
    """If True, vote only over candidates that pass the verifier chain.

    If no candidate passes, falls back to voting over all candidates so the
    pipeline never returns empty.
    """
    seed: int = 0
    compute_profile: ComputeProfile = ComputeProfile.BALANCED
    """Auto-scale candidate count / Langevin steps by question difficulty."""
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    """Verification-and-refinement loop (disabled when ``max_rounds == 0``)."""
    diverse_selection: DiverseSelectionConfig | None = None
    """DVTS-style diverse-cluster pre-vote filter. ``None`` disables it.

    When set, candidates are clustered in latent space by greedy
    farthest-first traversal; only the lowest-energy candidate from each
    cluster reaches the voter. Reference: HuggingFaceH4 'Scaling test-time
    compute' (DVTS).
    """


class HierarchicalLatentReasoner:
    """Compose intent routing, latent candidate sampling, decoding, and voting.

    Parameters
    ----------
    encoder
        ``str -> NDArray[float32]``. Wraps the EBRM pooler+projector heads.
    decoder
        ``NDArray[float32] -> str``. Wraps the EBRM answer_decoder head
        (greedy or temperature-sampled — the caller's choice).
    energy_fn
        ``NDArray[float32] -> float``. Wraps the EBRM energy_head.
    classifier
        Intent classifier. Defaults to :class:`RuleBasedClassifier`.
    index
        Optional :class:`LatentIndex` for QJL warm-start retrieval.
    config
        Reasoner configuration (voting strategy, seed, …).

    Notes
    -----
    The compute budget (Langevin steps, restart count, candidate count) is
    chosen *automatically* from the intent classifier's output, so the same
    instance handles trivial arithmetic and hard reasoning without manual
    tuning.
    """

    def __init__(
        self,
        encoder: EncoderFn,
        decoder: DecoderFn,
        energy_fn: EnergyFn,
        *,
        classifier: Classifier | None = None,
        index: LatentIndex | None = None,
        config: ReasonerConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.energy_fn = energy_fn
        self.classifier = classifier or RuleBasedClassifier()
        self.index = index
        self.config = config or ReasonerConfig()

    def solve(self, question: str) -> ReasoningResult:
        """Run the full hierarchical reasoning pipeline on one question.

        When :attr:`ReasonerConfig.refinement` enables refinement and the
        first pass fails to verify enough candidates, the question is
        re-encoded with appended verifier critiques and reasoning is run
        again. Verified candidates from all rounds are pooled before voting.
        """
        prediction = self.classifier.classify(question)
        budget = scale_budget(prediction, self.config.compute_profile)

        # Round 0
        traces, chain = self._reason_once(question, prediction, budget, seed_offset=0)
        n_rounds = 1
        all_traces: list[TraceItem] = list(traces)

        # Verification-and-refinement (arXiv:2507.15855)
        for round_idx in range(self.config.refinement.max_rounds):
            verified_fraction = self._verified_fraction(all_traces, chain)
            critiques = collect_critiques(
                [list(t.verifier_results) for t in all_traces if not t.verified],
                max_critiques=self.config.refinement.max_critiques,
            )
            if not should_refine(verified_fraction, critiques, self.config.refinement):
                break
            refined_q = build_refined_question(question, critiques, self.config.refinement)
            refined_pred = self.classifier.classify(refined_q)
            refined_budget = scale_budget(refined_pred, self.config.compute_profile)
            extra_traces, _ = self._reason_once(
                refined_q,
                refined_pred,
                refined_budget,
                seed_offset=(round_idx + 1) * 1009,
            )
            all_traces.extend(extra_traces)
            n_rounds += 1

        # Vote over the (possibly enlarged) trace pool
        pool = self._select_voting_pool(all_traces)
        n_pre_diverse = len(pool)
        pool = self._apply_diverse_selection(pool)
        vote_candidates = [
            VoteCandidate(answer=t.answer, energy=t.energy, trace_id=t.seed) for t in pool
        ]
        voter = SelfConsistencyVoter(
            numerical=self.config.numerical_tolerance is not None,
            tolerance=self.config.numerical_tolerance or 1e-6,
            weight_by=self.config.weight_by,
        )
        vote = voter.vote(vote_candidates)

        verified_fraction = self._verified_fraction(all_traces, chain)
        details: dict[str, object] = {
            "intent": prediction.intent.value,
            "difficulty": prediction.difficulty,
            "n_candidates": len(all_traces),
            "n_warmstart": sum(1 for t in all_traces if t.warmstart),
            "voting_pool": "verified" if pool is not all_traces else "all",
            "compute_profile": self.config.compute_profile.value,
            "budget": {
                "num_candidates": budget.num_candidates,
                "num_steps": budget.num_steps,
                "num_restarts": budget.num_restarts,
            },
            "refinement_rounds": n_rounds - 1,
            "diverse_selection": (
                {"survivors": len(pool), "input": n_pre_diverse}
                if self.config.diverse_selection is not None
                else None
            ),
        }
        return ReasoningResult(
            answer=vote.answer,
            intent=prediction,
            vote=vote,
            traces=tuple(all_traces),
            verified_fraction=verified_fraction,
            details=details,
        )

    # -- pipeline core (one round) ---------------------------------------

    def _reason_once(
        self,
        question: str,
        prediction: IntentPrediction,
        budget: ScaledBudget,
        *,
        seed_offset: int,
    ) -> tuple[list[TraceItem], VerifierChain]:
        """Run encode → generate → decode → verify exactly once.

        Returns ``(traces, chain)``. The chain is returned so the caller can
        check whether verifiers were available for the routed intent.
        """
        seed_latent = self._as_float32(self.encoder(question))
        cand_cfg = CandidateConfig(
            num_candidates=budget.num_candidates,
            num_steps=budget.num_steps,
            warmstart_k=(
                min(budget.num_restarts, budget.num_candidates) if self.index is not None else 0
            ),
            seed=self.config.seed + seed_offset,
        )
        latent_candidates = generate_candidates(
            seed_latent,
            self.energy_fn,
            config=cand_cfg,
            index=self.index,
        )
        chain = chain_for_intent(prediction.intent)
        traces: list[TraceItem] = []
        for lc in latent_candidates:
            answer = self.decoder(lc.latent)
            results = chain.verify(answer) if chain.verifiers else []
            ok = bool(chain.verifiers) and bool(results) and chain.all_passed(results)
            traces.append(
                TraceItem(
                    latent=lc.latent,
                    answer=answer,
                    energy=lc.energy,
                    seed=lc.seed,
                    warmstart=lc.warmstart,
                    verified=ok,
                    verifier_results=tuple(results),
                )
            )
        return traces, chain

    @staticmethod
    def _verified_fraction(traces: list[TraceItem], chain: VerifierChain) -> float:
        if not traces or not chain.verifiers:
            return 0.0
        return sum(1 for t in traces if t.verified) / len(traces)

    # -- internals --------------------------------------------------------

    def _select_voting_pool(self, traces: list[TraceItem]) -> list[TraceItem]:
        if not self.config.require_verification:
            return traces
        verified = [t for t in traces if t.verified]
        return verified if verified else traces

    def _apply_diverse_selection(self, traces: list[TraceItem]) -> list[TraceItem]:
        """Apply DVTS-style cluster filtering before voting (no-op if disabled)."""
        cfg = self.config.diverse_selection
        if cfg is None or not traces:
            return traces
        survivors = select_diverse(
            [t.latent for t in traces],
            [t.energy for t in traces],
            config=cfg,
        )
        return [traces[i] for i in survivors]

    @staticmethod
    def _as_float32(x: LatentT) -> LatentT:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError(f"encoder must return a 1-D latent, got shape {a.shape}")
        return a


__all__ = [
    "DecoderFn",
    "EncoderFn",
    "EnergyFn",
    "HierarchicalLatentReasoner",
    "LatentCandidate",
    "ReasonerConfig",
    "ReasoningResult",
    "TraceItem",
]
