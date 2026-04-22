"""Tests for SymPyVerifier."""

from __future__ import annotations

import sympy as sp

from ebrm_system.verifiers import SymPyVerifier


def test_symbolic_equality_identity(sympy_verifier: SymPyVerifier) -> None:
    result = sympy_verifier.check("(x+1)**2", {"expected": "x**2 + 2*x + 1"})
    assert result.verified is True
    assert result.confidence == 1.0


def test_numerical_equality_within_tolerance(sympy_verifier: SymPyVerifier) -> None:
    result = sympy_verifier.check(3.141593, {"expected": sp.pi})
    assert result.verified is True


def test_rejects_wrong_answer(sympy_verifier: SymPyVerifier) -> None:
    result = sympy_verifier.check("x + 1", {"expected": "x + 2"})
    assert result.verified is False
    assert result.confidence == 0.0


def test_missing_expected_returns_unverified(sympy_verifier: SymPyVerifier) -> None:
    result = sympy_verifier.check("x + 1", {})
    assert result.verified is False
    assert "expected" in (result.reason or "")


def test_parse_error_handled_gracefully(sympy_verifier: SymPyVerifier) -> None:
    # Unbalanced parens are unambiguously unparseable across SymPy versions.
    result = sympy_verifier.check("1 + + ))", {"expected": "1"})
    assert result.verified is False


def test_integer_equality(sympy_verifier: SymPyVerifier) -> None:
    result = sympy_verifier.check(42, {"expected": 42})
    assert result.verified is True


def test_tolerance_respected() -> None:
    strict = SymPyVerifier(tolerance=1e-10)
    result = strict.check(1.0000001, {"expected": 1.0})
    assert result.verified is False

    loose = SymPyVerifier(tolerance=1e-3)
    result = loose.check(1.0001, {"expected": 1.0})
    assert result.verified is True
