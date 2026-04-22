"""Tests for RuleBasedClassifier."""

from __future__ import annotations

import pytest

from ebrm_system.intent import Intent, IntentPrediction, RuleBasedClassifier


def test_empty_query_is_unknown(classifier: RuleBasedClassifier) -> None:
    pred = classifier.classify("")
    assert pred.intent == Intent.UNKNOWN
    assert pred.difficulty == 0.0


def test_code_query(classifier: RuleBasedClassifier) -> None:
    pred = classifier.classify("write a function in python to sort a list")
    assert pred.intent == Intent.CODE


def test_arithmetic(classifier: RuleBasedClassifier) -> None:
    pred = classifier.classify("2 + 2")
    assert pred.intent == Intent.ARITHMETIC


def test_math_word_problem(classifier: RuleBasedClassifier) -> None:
    pred = classifier.classify("Solve: If Alice has 3 apples and Bob has 5, what is the sum?")
    assert pred.intent in {Intent.MATH_REASONING, Intent.ARITHMETIC}


def test_creative(classifier: RuleBasedClassifier) -> None:
    pred = classifier.classify("write a story about a robot")
    assert pred.intent == Intent.CREATIVE


def test_factual(classifier: RuleBasedClassifier) -> None:
    pred = classifier.classify("who is Ada Lovelace?")
    assert pred.intent == Intent.FACTUAL


def test_dialogue(classifier: RuleBasedClassifier) -> None:
    pred = classifier.classify("hello")
    assert pred.intent == Intent.DIALOGUE
    assert pred.difficulty < 0.1


def test_budget_scales_with_difficulty(classifier: RuleBasedClassifier) -> None:
    easy = classifier.classify("2+2")
    hard = classifier.classify(
        "Compute the integral of sin(x)^2 from 0 to pi and then prove "
        "the result matches the identity cos(2x) = 1 - 2*sin(x)^2 "
        "using numbers 1 2 3 4 5 6 7."
    )
    assert hard.suggested_langevin_steps >= easy.suggested_langevin_steps
    assert hard.suggested_trace_count >= easy.suggested_trace_count


def test_prediction_validates_difficulty() -> None:
    with pytest.raises(ValueError):
        IntentPrediction(
            intent=Intent.UNKNOWN,
            difficulty=1.5,
            suggested_langevin_steps=10,
            suggested_restarts=1,
            suggested_trace_count=1,
            reasoning="",
        )


def test_prediction_validates_steps() -> None:
    with pytest.raises(ValueError):
        IntentPrediction(
            intent=Intent.UNKNOWN,
            difficulty=0.5,
            suggested_langevin_steps=0,
            suggested_restarts=1,
            suggested_trace_count=1,
            reasoning="",
        )
