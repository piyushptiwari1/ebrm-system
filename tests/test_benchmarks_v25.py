"""Tests for v0.25 aggregation CoT reader path."""

from __future__ import annotations

import pytest

pytest.importorskip("openai", reason="reader tests need openai")

from benchmarks.reader.azure_llm import (
    _AGGREGATION_USER_TEMPLATE,
    _final_answer,
)


def test_aggregation_template_demands_numbered_items() -> None:
    assert "ITEMS:" in _AGGREGATION_USER_TEMPLATE
    assert "TOTAL:" in _AGGREGATION_USER_TEMPLATE
    assert "ANSWER:" in _AGGREGATION_USER_TEMPLATE
    assert "1., 2., 3." in _AGGREGATION_USER_TEMPLATE


def test_final_answer_extracts_answer_line() -> None:
    raw = "ITEMS:\n1. boots\n2. shirt\n3. jacket\nTOTAL: 3 items\nANSWER: 3"
    assert _final_answer(raw) == "3"


def test_final_answer_handles_lowercase_and_whitespace() -> None:
    raw = "items:\n1. a\ntotal: 1\nanswer:   42  "
    assert _final_answer(raw) == "42"


def test_final_answer_falls_back_to_raw_when_no_marker() -> None:
    raw = "Just a flat answer."
    assert _final_answer(raw) == "Just a flat answer."


def test_final_answer_picks_last_answer_line() -> None:
    raw = "ANSWER: wrong\nmore text\nANSWER: right"
    assert _final_answer(raw) == "right"


def test_final_answer_empty_after_marker_falls_back() -> None:
    raw = "ITEMS:\n1. x\nANSWER:"
    # Empty answer string falls back to full raw.
    assert _final_answer(raw) == raw.strip()
