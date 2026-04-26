"""Verification-and-refinement loop.

When the verifier chain rejects a candidate, capture its rejection ``reason``
and re-encode the question augmented with that critique. This is the same
mechanism that turned silver-medal LLMs into IMO-2025-gold per
"Winning Gold at IMO 2025 with a Model-Agnostic Verification-and-Refinement
Pipeline" (arXiv:2507.15855).

This module is intentionally tiny: it produces an *augmented question*, the
caller (``HierarchicalLatentReasoner``) re-runs its full encoder→generate→
decode→verify pipeline on it, and the two rounds' verified candidates are
pooled before voting. We do not modify the decoder, the energy head, or
the candidate generator — refinement lives entirely in question space.

Pure-Python, deterministic, no torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ebrm_system.verifiers.base import VerificationResult


@dataclass(frozen=True)
class RefinementConfig:
    """Knobs for the refinement loop."""

    max_rounds: int = 0
    """Number of additional refinement passes (0 disables refinement)."""

    trigger_threshold: float = 0.5
    """Refine if ``verified_fraction < trigger_threshold``. Set to 1.0 to
    always refine when at least one verifier rejected something."""

    max_critiques: int = 3
    """Cap how many distinct rejection reasons feed into the prompt."""

    template: str = (
        "{question}\n\n"
        "[Critique of previous attempt(s)]\n"
        "{critiques}\n\n"
        "Re-examine the problem given the critique above and produce a corrected solution."
    )

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise ValueError(f"max_rounds must be >= 0, got {self.max_rounds}")
        if not 0.0 <= self.trigger_threshold <= 1.0:
            raise ValueError(f"trigger_threshold must be in [0,1], got {self.trigger_threshold}")
        if self.max_critiques < 1:
            raise ValueError(f"max_critiques must be >= 1, got {self.max_critiques}")


def collect_critiques(
    results_per_trace: Sequence[Sequence[VerificationResult]],
    *,
    max_critiques: int = 3,
) -> list[str]:
    """Collect distinct rejection reasons across the verifier results.

    Returns the most informative *unique* failure reasons (preserving first-
    occurrence order). Empty reasons and accept results are skipped.
    """
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    for results in results_per_trace:
        for r in results:
            if r.verified:
                continue
            key = (r.verifier, r.reason.strip())
            label = f"[{r.verifier}] {r.reason.strip()}" if r.reason.strip() else ""
            if not label or key in seen:
                continue
            seen.add(key)
            out.append(label)
            if len(out) >= max_critiques:
                return out
    return out


def should_refine(
    verified_fraction: float,
    critiques: Sequence[str],
    config: RefinementConfig,
) -> bool:
    """True iff a refinement pass is warranted under ``config``."""
    if config.max_rounds <= 0:
        return False
    if not critiques:
        return False
    return verified_fraction < config.trigger_threshold


def build_refined_question(
    question: str,
    critiques: Sequence[str],
    config: RefinementConfig,
) -> str:
    """Render the augmented question for the next reasoning pass."""
    if not critiques:
        return question
    bullet = "\n".join(f"- {c}" for c in critiques[: config.max_critiques])
    return config.template.format(question=question, critiques=bullet)


__all__ = [
    "RefinementConfig",
    "build_refined_question",
    "collect_critiques",
    "should_refine",
]
