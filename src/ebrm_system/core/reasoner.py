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
from ebrm_system.inference.latent_recursion import (
    RecursionConfig,
    StepFn,
    gradient_step,
    recurse_latent,
)
from ebrm_system.inference.mcts import MCTSConfig, mcts_select
from ebrm_system.intent import Classifier, IntentPrediction, RuleBasedClassifier
from ebrm_system.verifiers.base import VerificationResult, Verifier, VerifierChain
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
    latent_recursion: RecursionConfig | None = None
    """Coconut-style recursion applied to the seed latent before candidate
    generation. ``None`` disables it. When set with ``max_steps > 0``, the
    seed latent is iterated through ``step_fn`` (default: finite-difference
    energy gradient descent on the reasoner's ``energy_fn``) up to
    ``max_steps`` times, with optional plateau-based early halting.
    Reference: Coconut (arXiv:2412.06769), recurrent-depth (OpenReview
    D6o6Bwtq7h), LatentSeek.
    """
    mcts: MCTSConfig | None = None
    """ReST-MCTS*-style search over the candidate pool before voting.

    ``None`` disables it. When set, the candidate pool (post diverse
    selection if any) is re-ranked by UCB1-driven MCTS with a value
    function. By default the value function maps each candidate's energy
    into ``[0, 1]`` (lower energy → higher value), but callers can supply
    their own ``mcts_value_fn`` for PRM-guided search.
    Reference: ReST-MCTS* (arXiv:2406.03816), AlphaProof.
    """
    learn_from_solves: bool = False
    """Write each solved trace's winning latent back into the index after
    voting. ``False`` preserves the previous (read-only) behaviour.

    When enabled, the reasoner picks the lowest-energy verified candidate
    (or the lowest-energy candidate overall if none verified) and pushes
    its latent + answer payload into ``self.index`` via the duck-typed
    ``add(latents, payloads)`` method. Compatible with both
    :class:`~ebrm_system.reward.qjl_index.LatentIndex` and
    :class:`~ebrm_system.memory.TieredMemory`. Closes the loop so the
    agent learns across solves (Letta-style episodic write-back).
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
        recursion_step_fn: StepFn | None = None,
        mcts_value_fn: Callable[[TraceItem], float] | None = None,
        extra_verifiers: list[Verifier] | None = None,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.energy_fn = energy_fn
        self.classifier = classifier or RuleBasedClassifier()
        self.index = index
        self.config = config or ReasonerConfig()
        self.recursion_step_fn = recursion_step_fn
        self.mcts_value_fn = mcts_value_fn
        self.extra_verifiers: list[Verifier] = list(extra_verifiers or [])

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
        pool = self._apply_mcts(pool)
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
            "latent_recursion": getattr(self, "_last_recursion", None),
            "mcts": getattr(self, "_last_mcts", None),
            "memory_write": self._maybe_write_back(all_traces, vote),
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
        seed_latent = self._maybe_recurse_latent(seed_latent)
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
        if self.extra_verifiers:
            chain = VerifierChain(list(chain.verifiers) + list(self.extra_verifiers))
        verifier_context: dict[str, object] = {"question": question}
        traces: list[TraceItem] = []
        for lc in latent_candidates:
            answer = self.decoder(lc.latent)
            results = chain.verify(answer, verifier_context) if chain.verifiers else []
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

    def _apply_mcts(self, traces: list[TraceItem]) -> list[TraceItem]:
        """Re-rank candidates with ReST-MCTS* (no-op if disabled).

        The default value function maps each trace's energy into ``[0, 1]``
        via ``1 / (1 + energy_relative_to_min)`` so that the lowest-energy
        candidate maps to ``1.0``. Callers can override by passing
        ``mcts_value_fn`` (TraceItem -> float in [0, 1]) at construction.

        Stores audit info at ``self._last_mcts`` so ``solve()`` can surface
        it in :attr:`ReasoningResult.details`.
        """
        cfg = self.config.mcts
        if cfg is None or len(traces) <= 1:
            self._last_mcts = None
            return traces
        if self.mcts_value_fn is not None:
            value_fn = self.mcts_value_fn

            def _vfn(i: int) -> float:
                return float(value_fn(traces[i]))
        else:
            energies = np.array([t.energy for t in traces], dtype=np.float64)
            e_min = float(energies.min())

            def _vfn(i: int) -> float:
                # Lower energy -> higher value; scale-invariant.
                return float(1.0 / (1.0 + (traces[i].energy - e_min)))

        result = mcts_select(
            [t.latent for t in traces],
            _vfn,
            config=cfg,
        )
        self._last_mcts = {
            "simulations_run": result.simulations_run,
            "top_visits": result.visits[:3],
            "top_values": result.values[:3],
            "pool_size": len(traces),
        }
        # MCTS output is a strict permutation; honour it order-preservingly.
        return [traces[i] for i in result.ranking]

    def _maybe_write_back(
        self, traces: list[TraceItem], vote: VoteResult
    ) -> dict[str, object] | None:
        """Push the winning trace's latent into the index (Letta-style episodic write-back).

        No-op when ``learn_from_solves`` is False, or when the index has no
        ``add`` method, or when the trace pool is empty. Returns the audit
        dict surfaced in :attr:`ReasoningResult.details`.
        """
        if not self.config.learn_from_solves or self.index is None or not traces:
            return None
        add = getattr(self.index, "add", None)
        if add is None:
            return None
        # Pick the verified trace whose answer matches the vote winner with
        # the lowest energy. Fall back to lowest-energy candidate overall.
        winner_answer = str(vote.answer)
        verified_winners = [t for t in traces if t.verified and str(t.answer) == winner_answer]
        if verified_winners:
            chosen = min(verified_winners, key=lambda t: t.energy)
            kind = "verified"
        else:
            chosen = min(traces, key=lambda t: t.energy)
            kind = "fallback"
        latent = np.asarray(chosen.latent, dtype=np.float32)[np.newaxis, :]
        try:
            add(latent, [chosen.latent])
        except Exception as exc:
            return {"written": False, "error": str(exc)}
        return {
            "written": True,
            "kind": kind,
            "energy": chosen.energy,
            "answer": winner_answer,
        }

    def _maybe_recurse_latent(self, seed_latent: LatentT) -> LatentT:
        """Apply Coconut-style latent recursion to the seed (no-op if disabled).

        Stores a per-call audit dict at ``self._last_recursion`` so
        ``solve()`` can surface it in :attr:`ReasoningResult.details`.
        """
        cfg = self.config.latent_recursion
        if cfg is None or cfg.max_steps == 0:
            self._last_recursion = None
            return seed_latent
        step_fn: StepFn = self.recursion_step_fn or gradient_step(
            self.energy_fn, step_size=cfg.step_size, fd_eps=cfg.fd_eps
        )
        result = recurse_latent(
            seed_latent,
            step_fn,
            config=cfg,
            energy_fn=self.energy_fn,
        )
        self._last_recursion = {
            "steps_run": result.steps_run,
            "halted_early": result.halted_early,
            "energy_start": (result.energy_trajectory[0] if result.energy_trajectory else None),
            "energy_end": (result.energy_trajectory[-1] if result.energy_trajectory else None),
        }
        return result.latent

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
