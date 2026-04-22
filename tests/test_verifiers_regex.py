"""Tests for RegexVerifier."""

from __future__ import annotations

import re

from ebrm_system.verifiers import RegexVerifier


def test_fullmatch_success(regex_verifier: RegexVerifier) -> None:
    result = regex_verifier.check("hello", {"pattern": r"[a-z]+"})
    assert result.verified is True


def test_partial_match_rejected(regex_verifier: RegexVerifier) -> None:
    result = regex_verifier.check("hello world", {"pattern": r"[a-z]+"})
    assert result.verified is False


def test_flags_ignorecase(regex_verifier: RegexVerifier) -> None:
    result = regex_verifier.check("HELLO", {"pattern": r"hello", "flags": re.IGNORECASE})
    assert result.verified is True


def test_missing_pattern(regex_verifier: RegexVerifier) -> None:
    result = regex_verifier.check("hello", {})
    assert result.verified is False


def test_non_string_candidate(regex_verifier: RegexVerifier) -> None:
    result = regex_verifier.check(42, {"pattern": r"\d+"})
    assert result.verified is False


def test_invalid_regex_handled(regex_verifier: RegexVerifier) -> None:
    result = regex_verifier.check("abc", {"pattern": "["})
    assert result.verified is False
    assert "invalid" in (result.reason or "")
