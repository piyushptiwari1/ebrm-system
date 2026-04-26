"""Pluggable Process Reward Model (PRM) verifier.

This wraps an arbitrary scoring callable into a :class:`Verifier` that can
sit anywhere in the verifier chain. It is designed to be the integration
point for trained PRMs such as ThinkPRM and Athena-PRM:

* `ThinkPRM <https://mukhal.github.io/projects/thinkprm/>`_ — emits a
  step-by-step verbal verification chain and a verdict.
* `Athena-PRM <https://openreview.net/forum?id=YyCgWQFGtL>`_ — multimodal
  PRM trained with weak/strong agreement labels.

You bring the model and expose it as a callable; this module gives you the
``Verifier`` plumbing for free.

Two ready-made adapters:

* :class:`ScalarPRMVerifier` — wraps ``(question, answer) -> float`` plus a
  threshold. Verdict is ``score >= threshold``.
* :class:`GenerativePRMVerifier` — wraps ``(question, answer) -> PRMVerdict``
  for models that emit a verdict, a confidence, and a reasoning chain
  (e.g. ThinkPRM-style).

Both are pure-Python, deterministic *if your callable is deterministic*,
and torch-optional. They never call any model themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ebrm_system.verifiers.base import VerificationResult


@dataclass(frozen=True)
class PRMVerdict:
    """Output contract for generative PRM callables.

    Attributes
    ----------
    verified
        ``True`` iff the PRM accepts the answer.
    confidence
        In ``[0, 1]``. For ThinkPRM-style models, set this to the
        probability of the ``<correct>`` token.
    reasoning
        Optional verbalized chain explaining the verdict. Carried into
        ``VerificationResult.reason`` so the refinement loop can use it as
        a critique.
    evidence
        Free-form structured evidence (e.g. per-step labels). Optional.
    """

    verified: bool
    confidence: float
    reasoning: str = ""
    evidence: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


ScalarPRMFn = Callable[[str, object], float]
"""Signature for :class:`ScalarPRMVerifier`. Returns a score in any range;
the verifier compares it against a threshold."""

GenerativePRMFn = Callable[[str, object], PRMVerdict]
"""Signature for :class:`GenerativePRMVerifier`. Returns a full verdict."""


class ScalarPRMVerifier:
    """Verifier that thresholds a scalar PRM score.

    Parameters
    ----------
    score_fn
        ``(question, candidate_answer) -> float``. May return any real
        value; only the comparison against ``threshold`` matters.
    threshold
        Acceptance threshold (inclusive).
    name
        Verifier name (default ``"scalar_prm"``).
    question_key
        Key in the verifier ``context`` dict that carries the question.
        Defaults to ``"question"``.

    Notes
    -----
    Use this when you have any scalar reward signal — a trained value head,
    an outcome reward model, or an LLM-as-judge that returns a single
    number. For richer step-level feedback, use
    :class:`GenerativePRMVerifier` instead.
    """

    def __init__(
        self,
        score_fn: ScalarPRMFn,
        *,
        threshold: float,
        name: str = "scalar_prm",
        question_key: str = "question",
    ) -> None:
        self._score = score_fn
        self.threshold = threshold
        self.name = name
        self.question_key = question_key

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        question = self._extract_question(context)
        score = float(self._score(question, candidate))
        verified = score >= self.threshold
        # Map the score into [0, 1] using a simple clamp around threshold.
        confidence = 1.0 if verified else 0.0
        return VerificationResult(
            verifier=self.name,
            verified=verified,
            confidence=confidence,
            reason=(f"score {score:.4f} {'>=' if verified else '<'} threshold {self.threshold}"),
            evidence={"score": score, "threshold": self.threshold},
        )

    def _extract_question(self, context: dict[str, object] | None) -> str:
        if context is None:
            return ""
        q = context.get(self.question_key, "")
        return str(q) if q is not None else ""


class GenerativePRMVerifier:
    """Verifier that delegates to a generative PRM callable.

    Parameters
    ----------
    verdict_fn
        ``(question, candidate_answer) -> PRMVerdict``. Wrap your trained
        PRM (ThinkPRM, Athena-PRM, …) in this signature.
    name
        Verifier name (default ``"generative_prm"``).
    question_key
        Key in the verifier ``context`` dict that carries the question.

    Notes
    -----
    Carries the PRM's ``reasoning`` chain into
    :attr:`VerificationResult.reason` so the
    :mod:`ebrm_system.core.refinement` loop can fold the rejection
    rationale back into the next prompt — exactly the behaviour the
    IMO-2025-gold pipeline (arXiv:2507.15855) relies on.
    """

    def __init__(
        self,
        verdict_fn: GenerativePRMFn,
        *,
        name: str = "generative_prm",
        question_key: str = "question",
    ) -> None:
        self._verdict = verdict_fn
        self.name = name
        self.question_key = question_key

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        question = self._extract_question(context)
        verdict = self._verdict(question, candidate)
        evidence: dict[str, object] = {}
        if verdict.evidence is not None:
            evidence.update(verdict.evidence)
        return VerificationResult(
            verifier=self.name,
            verified=verdict.verified,
            confidence=verdict.confidence,
            reason=verdict.reasoning,
            evidence=evidence,
        )

    def _extract_question(self, context: dict[str, object] | None) -> str:
        if context is None:
            return ""
        q = context.get(self.question_key, "")
        return str(q) if q is not None else ""


__all__ = [
    "GenerativePRMFn",
    "GenerativePRMVerifier",
    "PRMVerdict",
    "ScalarPRMFn",
    "ScalarPRMVerifier",
]
