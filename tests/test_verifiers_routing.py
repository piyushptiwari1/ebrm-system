"""Tests for intent-routed verifier chains."""

from __future__ import annotations

from ebrm_system.intent import Intent
from ebrm_system.verifiers.dri import DRIVerifier
from ebrm_system.verifiers.exec_verifier import ExecVerifier
from ebrm_system.verifiers.lean import LeanVerifier
from ebrm_system.verifiers.regex_verifier import RegexVerifier
from ebrm_system.verifiers.routing import advice_chain, chain_for_intent
from ebrm_system.verifiers.sympy_verifier import SymPyVerifier


def _names(chain: object) -> list[str]:
    return [v.name for v in chain.verifiers]  # type: ignore[attr-defined]


def test_chain_for_math_includes_sympy_and_lean() -> None:
    chain = chain_for_intent(Intent.MATH_REASONING)
    types = [type(v) for v in chain.verifiers]
    assert SymPyVerifier in types
    assert LeanVerifier in types


def test_chain_for_arithmetic_includes_sympy_and_lean() -> None:
    chain = chain_for_intent(Intent.ARITHMETIC)
    types = [type(v) for v in chain.verifiers]
    assert SymPyVerifier in types
    assert LeanVerifier in types


def test_chain_for_code_includes_exec_and_regex() -> None:
    chain = chain_for_intent(Intent.CODE)
    types = [type(v) for v in chain.verifiers]
    assert ExecVerifier in types
    assert RegexVerifier in types


def test_chain_for_factual_is_regex_only() -> None:
    chain = chain_for_intent(Intent.FACTUAL)
    types = [type(v) for v in chain.verifiers]
    assert types == [RegexVerifier]


def test_chain_for_creative_is_empty() -> None:
    assert chain_for_intent(Intent.CREATIVE).verifiers == []


def test_chain_for_dialogue_is_empty() -> None:
    assert chain_for_intent(Intent.DIALOGUE).verifiers == []


def test_chain_for_unknown_is_empty() -> None:
    assert chain_for_intent(Intent.UNKNOWN).verifiers == []


def test_advice_chain_is_dri() -> None:
    chain = advice_chain()
    assert [type(v) for v in chain.verifiers] == [DRIVerifier]
