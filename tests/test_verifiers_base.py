"""Tests for the Verifier base Protocol and VerifierChain."""

from __future__ import annotations

import pytest

from ebrm_system.verifiers import (
    RegexVerifier,
    SymPyVerifier,
    VerificationResult,
    VerifierChain,
)


def test_verification_result_validates_confidence() -> None:
    with pytest.raises(ValueError):
        VerificationResult(verifier="x", verified=True, confidence=1.5)
    with pytest.raises(ValueError):
        VerificationResult(verifier="x", verified=True, confidence=-0.1)


def test_chain_short_circuits_on_rejection() -> None:
    regex = RegexVerifier()
    sympy = SymPyVerifier()
    chain = VerifierChain([regex, sympy])
    # Regex rejects first -> sympy never runs
    results = chain.verify("not-a-digit", {"pattern": r"\d+", "expected": "1"})
    assert len(results) == 1
    assert results[0].verifier == "regex"
    assert chain.all_passed(results) is False


def test_chain_all_pass() -> None:
    regex = RegexVerifier()
    sympy = SymPyVerifier()
    chain = VerifierChain([regex, sympy])
    results = chain.verify("42", {"pattern": r"\d+", "expected": 42})
    assert len(results) == 2
    assert chain.all_passed(results) is True


def test_empty_chain() -> None:
    chain = VerifierChain([])
    results = chain.verify("anything")
    assert results == []
    assert chain.all_passed(results) is True
