"""SymPy-based algebraic verifier.

Checks whether a proposed answer symbolically equals a reference expression,
or satisfies a given algebraic identity. Fully deterministic.
"""

from __future__ import annotations

import sympy as sp

from ebrm_system.verifiers.base import VerificationResult


class SymPyVerifier:
    """Verify an answer using symbolic algebra.

    Context keys:
        expected: expected value (str expression, number, or sp.Expr)
        tolerance: numerical tolerance for float comparison (default 1e-6)
    """

    name = "sympy"

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        context = context or {}
        expected = context.get("expected")
        if expected is None:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="no expected value in context",
            )

        try:
            cand_expr = self._parse(candidate)
            exp_expr = self._parse(expected)
        except (sp.SympifyError, TypeError, ValueError) as exc:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"parse error: {exc}",
            )

        # Symbolic equality check
        try:
            diff = sp.simplify(cand_expr - exp_expr)
        except (TypeError, ValueError) as exc:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"simplify error: {exc}",
            )

        if diff == 0:
            return VerificationResult(
                verifier=self.name,
                verified=True,
                confidence=1.0,
                reason="symbolically equal",
                evidence={"candidate": str(cand_expr), "expected": str(exp_expr)},
            )

        # Numerical fallback for float-valued answers
        try:
            cand_num = float(cand_expr)
            exp_num = float(exp_expr)
            if abs(cand_num - exp_num) <= self.tolerance * max(abs(exp_num), 1.0):
                return VerificationResult(
                    verifier=self.name,
                    verified=True,
                    confidence=1.0,
                    reason=f"numerically equal within {self.tolerance}",
                    evidence={"candidate": cand_num, "expected": exp_num},
                )
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"numerical mismatch: {cand_num} vs {exp_num}",
                evidence={"candidate": cand_num, "expected": exp_num},
            )
        except (TypeError, ValueError):
            pass

        return VerificationResult(
            verifier=self.name,
            verified=False,
            confidence=0.0,
            reason="not symbolically or numerically equal",
            evidence={"diff": str(diff)},
        )

    @staticmethod
    def _parse(value: object) -> sp.Expr:
        if isinstance(value, sp.Expr):
            return value
        if isinstance(value, (int, float)):
            return (
                sp.Rational(value).limit_denominator(10**9)
                if isinstance(value, float)
                else sp.Integer(value)
            )
        if isinstance(value, str):
            return sp.sympify(value)
        raise TypeError(f"cannot parse {type(value).__name__} as sympy expression")
