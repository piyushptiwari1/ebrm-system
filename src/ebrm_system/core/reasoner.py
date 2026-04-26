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

from ebrm_system.inference.candidates import (
    Candidate as LatentCandidate,
)
from ebrm_system.inference.candidates import (
    CandidateConfig,
    generate_candidates,
)
from ebrm_system.intent import Classifier, IntentPrediction, RuleBasedClassifier
from ebrm_system.verifiers.base import VerificationResult
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
        """Run the full hierarchical reasoning pipeline on one question."""
        # Level 0: intent + budget
        prediction = self.classifier.classify(question)

        # Level 1: encode
        seed_latent = self._as_float32(self.encoder(question))

        # Level 2: candidate generation
        cand_cfg = CandidateConfig(
            num_candidates=prediction.suggested_trace_count,
            num_steps=prediction.suggested_langevin_steps,
            warmstart_k=min(
                prediction.suggested_restarts,
                prediction.suggested_trace_count,
            )
            if self.index is not None
            else 0,
            seed=self.config.seed,
        )
        latent_candidates = generate_candidates(
            seed_latent,
            self.energy_fn,
            config=cand_cfg,
            index=self.index,
        )

        # Level 3: decode + verify
        chain = chain_for_intent(prediction.intent)
        traces: list[TraceItem] = []
        n_verified = 0
        for lc in latent_candidates:
            answer = self.decoder(lc.latent)
            results = chain.verify(answer) if chain.verifiers else []
            ok = bool(results) and chain.all_passed(results)
            if not chain.verifiers:
                ok = False  # cannot claim verification when chain is empty
            else:
                n_verified += int(ok)
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

        # Level 4: vote
        pool = self._select_voting_pool(traces)
        vote_candidates = [
            VoteCandidate(answer=t.answer, energy=t.energy, trace_id=t.seed) for t in pool
        ]
        voter = SelfConsistencyVoter(
            numerical=self.config.numerical_tolerance is not None,
            tolerance=self.config.numerical_tolerance or 1e-6,
            weight_by=self.config.weight_by,
        )
        vote = voter.vote(vote_candidates)

        verified_fraction = (n_verified / len(traces)) if traces and chain.verifiers else 0.0
        details: dict[str, object] = {
            "intent": prediction.intent.value,
            "difficulty": prediction.difficulty,
            "n_candidates": len(traces),
            "n_warmstart": sum(1 for t in traces if t.warmstart),
            "voting_pool": "verified" if pool is not traces else "all",
        }
        return ReasoningResult(
            answer=vote.answer,
            intent=prediction,
            vote=vote,
            traces=tuple(traces),
            verified_fraction=verified_fraction,
            details=details,
        )

    # -- internals --------------------------------------------------------

    def _select_voting_pool(self, traces: list[TraceItem]) -> list[TraceItem]:
        if not self.config.require_verification:
            return traces
        verified = [t for t in traces if t.verified]
        return verified if verified else traces

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
