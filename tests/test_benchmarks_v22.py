"""Tests for v0.22 reader chronological sort and judge prompt version."""

from __future__ import annotations

import pytest

pytest.importorskip("openai", reason="reader/judge tests need openai")

from benchmarks.datasets.longmemeval_official import OfficialTurn
from benchmarks.judges.azure_llm import _JUDGE_PROMPT_VERSION, _JUDGE_SYSTEM
from benchmarks.reader.azure_llm import _chronological


def _t(date: str, content: str = "x", idx: int = 0) -> OfficialTurn:
    return OfficialTurn(
        session_id=f"s{idx}",
        session_idx=idx,
        turn_idx=0,
        role="user",
        content=content,
        session_date=date,
        has_answer=False,
    )


def test_chronological_sort_orders_oldest_first() -> None:
    turns = [
        _t("2024/03/15 (Fri) 09:00", "B", 1),
        _t("2024/01/02 (Tue) 10:30", "A", 0),
        _t("2024/06/01 (Sat) 18:45", "C", 2),
    ]
    out = _chronological(turns)
    assert [t.content for t in out] == ["A", "B", "C"]


def test_chronological_sort_pushes_unparseable_to_end() -> None:
    turns = [
        _t("garbage", "X", 0),
        _t("2024/01/01 (Mon) 10:00", "A", 1),
        _t("2024/06/01 (Sat) 12:00", "B", 2),
    ]
    out = _chronological(turns)
    assert [t.content for t in out] == ["A", "B", "X"]


def test_chronological_sort_handles_date_only_format() -> None:
    turns = [
        _t("2024/06/01", "later", 1),
        _t("2024/01/15", "earlier", 0),
    ]
    out = _chronological(turns)
    assert [t.content for t in out] == ["earlier", "later"]


def test_chronological_sort_empty() -> None:
    assert _chronological([]) == []


def test_judge_prompt_is_lenient_v2() -> None:
    # Cache-busting tag must change when prompt changes.
    assert _JUDGE_PROMPT_VERSION == "v2-lenient"
    # Lenient prompt must explicitly accept verbose / context-rich predictions.
    assert "do not penalise verbosity" in _JUDGE_SYSTEM
    assert "alternatives" in _JUDGE_SYSTEM
