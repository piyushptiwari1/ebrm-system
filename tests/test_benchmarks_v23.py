"""Tests for v0.23 reader prompt: preference and aggregation handling."""

from __future__ import annotations

import pytest

pytest.importorskip("openai", reason="reader tests need openai")

from benchmarks.reader.azure_llm import _READER_SYSTEM


def test_reader_prompt_handles_recommendation_questions() -> None:
    # Reader must NOT say IDK on suggest/recommend/advice questions.
    assert "recommendation" in _READER_SYSTEM
    assert "preferences" in _READER_SYSTEM
    assert "DO\nNOT say 'I don't know'" in _READER_SYSTEM or "DO NOT" in _READER_SYSTEM


def test_reader_prompt_handles_aggregation_questions() -> None:
    # Reader must enumerate all matches before reporting totals.
    assert "TOTAL" in _READER_SYSTEM
    assert "enumerate" in _READER_SYSTEM
    assert "Do not stop at the first match" in _READER_SYSTEM


def test_reader_prompt_keeps_arithmetic_and_chronology_rules() -> None:
    # v0.22 rules must remain.
    assert "do the arithmetic" in _READER_SYSTEM
    assert "session dates" in _READER_SYSTEM
