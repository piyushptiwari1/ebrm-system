"""Tests for v0.28 — narrowly-gated reader CoT (router + reader).

Two opt-in features:

1. Aggregation CoT — gate tightened from ``classify_question == "aggregation"``
   to ALSO require ``question_type == "multi-session"``. v0.25 leaked onto
   temporal "how many days" questions and dropped temporal-reasoning by 6 pt.
2. Temporal-ordering CoT — new structured template (CANDIDATES / ORDERED /
   ANSWER) for ``temporal-reasoning`` questions with ordering cues.

These tests cover the router predicates and the reader's gating logic
without making any live LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.router import (
    classify_question,
    is_multi_session_aggregation,
    is_temporal_ordering,
)

# ---------------------------------------------------------------------------
# Router predicates
# ---------------------------------------------------------------------------


class TestIsMultiSessionAggregation:
    def test_fires_on_multi_session_with_aggregation_cue(self):
        assert is_multi_session_aggregation(
            "How many model kits have I worked on or bought?", "multi-session"
        )

    def test_does_not_fire_on_temporal_with_aggregation_cue(self):
        # The v0.25 leak — must be False now.
        assert not is_multi_session_aggregation(
            "How many days between the start of project A and project B?",
            "temporal-reasoning",
        )
        assert (
            classify_question("How many days between A and B?", "temporal-reasoning")
            == "aggregation"
        )

    def test_does_not_fire_without_aggregation_cue(self):
        assert not is_multi_session_aggregation(
            "What did I tell my therapist about my brother?", "multi-session"
        )

    def test_does_not_fire_on_other_question_types(self):
        for qt in (
            "knowledge-update",
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
        ):
            assert not is_multi_session_aggregation("How many books have I bought this year?", qt)


class TestIsTemporalOrdering:
    @pytest.mark.parametrize(
        "q",
        [
            "Which event happened first, the purchase or the malfunction?",
            "Who did I meet first, Mark and Sarah or Tom?",
            "Which show did I start watching first, Crown or GoT?",
            "How many charity events did I participate in before the Run for the Cure event?",
            "Who became a parent first, Rachel or Alex?",
            "Which event did I participate in first, the volleyball league or the 5K?",
        ],
    )
    def test_fires_on_temporal_ordering_questions(self, q):
        assert is_temporal_ordering(q, "temporal-reasoning")

    def test_does_not_fire_on_non_temporal_type(self):
        assert not is_temporal_ordering("Who did I meet first, Mark or Tom?", "multi-session")

    def test_does_not_fire_without_ordering_cue(self):
        assert not is_temporal_ordering(
            "How long ago did I start my job at NovaTech?", "temporal-reasoning"
        )


# ---------------------------------------------------------------------------
# Reader gating
# ---------------------------------------------------------------------------


pytest.importorskip("openai", reason="AzureOpenAIReader requires openai SDK")


@pytest.fixture
def reader_env(monkeypatch):
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")


def _episode(question: str, question_type: str) -> OfficialEpisode:
    turn = OfficialTurn(
        session_id="s0",
        session_idx=0,
        turn_idx=0,
        role="user",
        content="hello",
        session_date="2024-01-01",
        has_answer=False,
    )
    return OfficialEpisode(
        question_id="q",
        question_type=question_type,
        question=question,
        answer="x",
        question_date="2024-02-01",
        turns=(turn,),
        answer_session_ids=(),
        is_abstention=False,
    )


def _stub_reader(monkeypatch, response: str):
    """Patch AzureOpenAI to return a stub client, capturing the user prompt."""
    captured = {"user": None, "calls": 0}

    def _create(**kw):
        captured["calls"] += 1
        for m in kw["messages"]:
            if m["role"] == "user":
                captured["user"] = m["content"]
        msg = MagicMock()
        msg.content = response
        choice = MagicMock()
        choice.message = msg
        rsp = MagicMock()
        rsp.choices = [choice]
        return rsp

    class _Stub:
        def __init__(self, *a, **kw):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = _create

    import openai

    monkeypatch.setattr(openai, "AzureOpenAI", _Stub)
    return captured


def test_aggregation_cot_fires_only_on_multi_session(monkeypatch, reader_env):
    captured = _stub_reader(monkeypatch, "ITEMS:\n1. a\n2. b\nTOTAL: 2\nANSWER: 2")
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(aggregation_cot=True)
    ep = _episode("How many books have I bought?", "multi-session")
    out = r.read(ep, list(ep.turns))
    assert "ITEMS:" in (captured["user"] or "")
    # ANSWER extraction returns just the number
    assert out == "2"


def test_aggregation_cot_does_not_fire_on_temporal(monkeypatch, reader_env):
    captured = _stub_reader(monkeypatch, "Some prose answer.")
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(aggregation_cot=True)
    ep = _episode("How many days between A and B?", "temporal-reasoning")
    out = r.read(ep, list(ep.turns))
    # Standard template — no ITEMS: header in user prompt
    assert "ITEMS:" not in (captured["user"] or "")
    assert out == "Some prose answer."


def test_temporal_ordering_cot_fires_on_ordering_temporal(monkeypatch, reader_env):
    captured = _stub_reader(
        monkeypatch,
        "CANDIDATES:\n- A: 2023/01/05\n- B: 2023/02/10\nORDERED:\n- A\n- B\nANSWER: A",
    )
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(temporal_ordering_cot=True)
    ep = _episode("Which event happened first, A or B?", "temporal-reasoning")
    out = r.read(ep, list(ep.turns))
    assert "CANDIDATES:" in (captured["user"] or "")
    assert "ORDERED:" in (captured["user"] or "")
    assert out == "A"


def test_temporal_ordering_cot_does_not_fire_when_aggregation_wins(monkeypatch, reader_env):
    """Aggregation gate must take precedence over ordering when both fire."""
    captured = _stub_reader(monkeypatch, "ITEMS:\n1. x\nTOTAL: 1\nANSWER: 1")
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(aggregation_cot=True, temporal_ordering_cot=True)
    # multi-session aggregation question — must use ITEMS template, not
    # ordering, even though "before" appears.
    ep = _episode(
        "How many charity events did I attend before the marathon?",
        "multi-session",
    )
    out = r.read(ep, list(ep.turns))
    assert "ITEMS:" in (captured["user"] or "")
    assert "CANDIDATES:" not in (captured["user"] or "")
    assert out == "1"


def test_both_flags_off_uses_standard_template(monkeypatch, reader_env):
    captured = _stub_reader(monkeypatch, "Answer text.")
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(aggregation_cot=False, temporal_ordering_cot=False)
    ep = _episode("How many books?", "multi-session")
    out = r.read(ep, list(ep.turns))
    user = captured["user"] or ""
    assert "ITEMS:" not in user
    assert "CANDIDATES:" not in user
    assert out == "Answer text."


def test_temporal_ordering_does_not_fire_on_non_temporal(monkeypatch, reader_env):
    captured = _stub_reader(monkeypatch, "Plain answer.")
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(temporal_ordering_cot=True)
    ep = _episode("Which show did I watch first, Crown or GoT?", "single-session-user")
    out = r.read(ep, list(ep.turns))
    assert "CANDIDATES:" not in (captured["user"] or "")
    assert out == "Plain answer."
