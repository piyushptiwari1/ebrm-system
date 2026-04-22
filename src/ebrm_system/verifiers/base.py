"""Verifier protocol + registry.

A Verifier takes a proposed answer (and optional context) and returns a
VerificationResult. Verifiers are pure, deterministic, and CPU-only.

The verifier chain is evaluated in order: the first verifier to reject a
candidate short-circuits the chain. Verifiers never hallucinate — they only
confirm or reject what SymPy / Python / regex can mechanically check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VerificationResult:
    """Result of a single verifier invocation."""

    verifier: str
    verified: bool
    confidence: float  # 0.0 = reject, 1.0 = accept (never between for symbolic)
    reason: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


@runtime_checkable
class Verifier(Protocol):
    """Pure verifier interface."""

    name: str

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        """Verify a candidate answer. Must be side-effect free and deterministic."""
        ...


class VerifierChain:
    """Run multiple verifiers in order, short-circuit on first rejection."""

    def __init__(self, verifiers: list[Verifier]) -> None:
        self.verifiers = verifiers

    def verify(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for v in self.verifiers:
            result = v.check(candidate, context)
            results.append(result)
            if not result.verified:
                break
        return results

    def all_passed(self, results: list[VerificationResult]) -> bool:
        return all(r.verified for r in results)
