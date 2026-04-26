"""Unit tests for ``ebrm_system.verifiers.prm``."""

from __future__ import annotations

import pytest

from ebrm_system.verifiers.prm import (
    GenerativePRMVerifier,
    PRMVerdict,
    ScalarPRMVerifier,
)


class TestPRMVerdict:
    def test_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            PRMVerdict(verified=True, confidence=1.5)

    def test_defaults(self) -> None:
        v = PRMVerdict(verified=True, confidence=0.9)
        assert v.reasoning == ""
        assert v.evidence is None


class TestScalarPRMVerifier:
    def test_passes_above_threshold(self) -> None:
        v = ScalarPRMVerifier(lambda q, a: 0.8, threshold=0.5)
        result = v.check("42", {"question": "q"})
        assert result.verified is True
        assert result.confidence == 1.0
        assert "0.8000" in result.reason
        assert result.evidence["score"] == pytest.approx(0.8)
        assert result.evidence["threshold"] == 0.5

    def test_rejects_below_threshold(self) -> None:
        v = ScalarPRMVerifier(lambda q, a: 0.1, threshold=0.5)
        result = v.check("42", {"question": "q"})
        assert result.verified is False
        assert result.confidence == 0.0

    def test_uses_question_from_context(self) -> None:
        seen: list[str] = []

        def score(q: str, a: object) -> float:
            seen.append(q)
            return 0.6

        v = ScalarPRMVerifier(score, threshold=0.5)
        v.check("answer", {"question": "what?"})
        assert seen == ["what?"]

    def test_handles_missing_context(self) -> None:
        v = ScalarPRMVerifier(lambda q, a: 1.0, threshold=0.5)
        result = v.check("answer")
        assert result.verified is True

    def test_custom_question_key(self) -> None:
        seen: list[str] = []

        def score(q: str, a: object) -> float:
            seen.append(q)
            return 0.6

        v = ScalarPRMVerifier(score, threshold=0.5, question_key="prompt")
        v.check("answer", {"prompt": "hi"})
        assert seen == ["hi"]


class TestGenerativePRMVerifier:
    def test_passes_through_verdict(self) -> None:
        verdict = PRMVerdict(
            verified=True,
            confidence=0.95,
            reasoning="Step 1 ok; Step 2 ok",
            evidence={"steps": [True, True]},
        )
        v = GenerativePRMVerifier(lambda q, a: verdict)
        result = v.check("42", {"question": "q"})
        assert result.verified is True
        assert result.confidence == 0.95
        assert result.reason == "Step 1 ok; Step 2 ok"
        assert result.evidence["steps"] == [True, True]

    def test_rejection_carries_critique(self) -> None:
        verdict = PRMVerdict(
            verified=False,
            confidence=0.1,
            reasoning="Step 2 contradicts Step 1",
        )
        v = GenerativePRMVerifier(lambda q, a: verdict)
        result = v.check("42", {"question": "q"})
        assert result.verified is False
        assert result.reason == "Step 2 contradicts Step 1"

    def test_uses_question_from_context(self) -> None:
        seen: list[str] = []

        def fn(q: str, a: object) -> PRMVerdict:
            seen.append(q)
            return PRMVerdict(verified=True, confidence=1.0)

        v = GenerativePRMVerifier(fn)
        v.check("ans", {"question": "the q"})
        assert seen == ["the q"]
