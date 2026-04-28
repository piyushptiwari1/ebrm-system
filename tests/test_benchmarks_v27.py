"""Tests for v0.27 — Multi-Query Retrieval (router-gated MQR).

Covers:
  * ``MultiQueryRetriever`` RRF-fuses candidates returned for each
    rewritten query and substitutes the question into the per-query
    sub-episode.
  * ``AzureOpenAIQueryRewriter`` cache + fallback contract (no live API
    calls — the openai client is monkeypatched).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.retrieval.base import Retriever, ScoredTurn
from benchmarks.retrieval.multi_query import MultiQueryRetriever


def _episode(turn_contents: list[str], question: str) -> OfficialEpisode:
    turns = [
        OfficialTurn(
            session_id=f"s{i}",
            session_idx=0,
            turn_idx=i,
            role="user",
            content=c,
            session_date="2024-01-01",
            has_answer=False,
        )
        for i, c in enumerate(turn_contents)
    ]
    return OfficialEpisode(
        question_id="q1",
        question_type="multi-session",
        question=question,
        answer="x",
        question_date="2024-02-01",
        turns=tuple(turns),
        answer_session_ids=(),
        is_abstention=False,
    )


# ---------------------------------------------------------------------------
# MultiQueryRetriever
# ---------------------------------------------------------------------------


@dataclass
class _StubRewriter:
    """Returns the original plus a deterministic set of alternates."""

    alternates: list[str]
    name: str = "stub-rewriter"

    def rewrite(self, question: str, question_type: str) -> list[str]:
        return [question, *self.alternates]


@dataclass
class _RecordingRetriever:
    """Returns turns whose order depends on which question was passed in.

    Useful to verify per-query substitution actually happens.
    """

    by_query: dict[str, list[int]]
    seen: list[str]
    name: str = "recording"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        self.seen.append(episode.question)
        order = self.by_query.get(episode.question, [])
        out: list[ScoredTurn] = []
        for rank, idx in enumerate(order[:top_k]):
            turn = episode.turns[idx]
            out.append(ScoredTurn(turn=turn, score=1.0 - rank * 0.1))
        return out


def test_multi_query_protocol():
    base = _RecordingRetriever(by_query={}, seen=[])
    rewriter = _StubRewriter(alternates=["alt-1", "alt-2"])
    mqr = MultiQueryRetriever(base=base, rewriter=rewriter)
    assert isinstance(mqr, Retriever)
    assert "multi-query" in mqr.name
    assert "stub-rewriter" in mqr.name


def test_multi_query_substitutes_question_per_rewrite():
    ep = _episode(["a", "b", "c"], question="orig")
    base = _RecordingRetriever(
        by_query={"orig": [0], "alt-1": [1], "alt-2": [2]}, seen=[]
    )
    rewriter = _StubRewriter(alternates=["alt-1", "alt-2"])
    mqr = MultiQueryRetriever(base=base, rewriter=rewriter, per_query_k=5)
    out = mqr.retrieve(ep, top_k=3)
    # Each query was sent to the base retriever exactly once.
    assert sorted(base.seen) == sorted(["orig", "alt-1", "alt-2"])
    # All three turns surface (one per query, RRF dedupes).
    assert len(out) == 3
    surfaced = {st.turn.turn_idx for st in out}
    assert surfaced == {0, 1, 2}


def test_multi_query_rrf_ranks_overlap_first():
    ep = _episode(["a", "b", "c"], question="orig")
    # Turn 0 appears first in 2/3 queries → should outrank turn 2 which only
    # appears (rank 1) in the third query.
    base = _RecordingRetriever(
        by_query={
            "orig": [0, 1, 2],
            "alt-1": [0, 2, 1],
            "alt-2": [1, 0, 2],
        },
        seen=[],
    )
    rewriter = _StubRewriter(alternates=["alt-1", "alt-2"])
    mqr = MultiQueryRetriever(base=base, rewriter=rewriter, per_query_k=5)
    out = mqr.retrieve(ep, top_k=3)
    assert out[0].turn.turn_idx == 0


def test_multi_query_empty_rewriter_falls_back_to_original():
    @dataclass
    class _EmptyRewriter:
        name: str = "empty"

        def rewrite(self, question: str, question_type: str) -> list[str]:
            return []

    ep = _episode(["a", "b"], question="orig")
    base = _RecordingRetriever(by_query={"orig": [0, 1]}, seen=[])
    mqr = MultiQueryRetriever(base=base, rewriter=_EmptyRewriter())
    out = mqr.retrieve(ep, top_k=2)
    assert base.seen == ["orig"]
    assert len(out) == 2


def test_multi_query_empty_episode_returns_empty():
    ep = _episode([], "orig")
    base = _RecordingRetriever(by_query={}, seen=[])
    rewriter = _StubRewriter(alternates=["a"])
    mqr = MultiQueryRetriever(base=base, rewriter=rewriter)
    assert mqr.retrieve(ep, top_k=5) == []
    # Rewriter must not be invoked when there is nothing to retrieve.
    assert base.seen == []


# ---------------------------------------------------------------------------
# AzureOpenAIQueryRewriter
# ---------------------------------------------------------------------------


pytest.importorskip("openai", reason="AzureOpenAIQueryRewriter requires openai SDK")


@pytest.fixture
def rewriter_env(monkeypatch):
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")


def _patch_client(monkeypatch, response_content: str | None) -> dict:
    """Replace AzureOpenAI with a stub returning ``response_content``."""
    calls = {"n": 0, "last": None}

    class _Resp:
        def __init__(self, content):
            self.choices = [
                type("C", (), {"message": type("M", (), {"content": content})})()
            ]

    class _Chat:
        def __init__(self, content):
            self._content = content
            self.completions = self

        def create(self, **kwargs):
            calls["n"] += 1
            calls["last"] = kwargs
            return _Resp(self._content)

    class _Stub:
        def __init__(self, *a, **kw):
            self.chat = _Chat(response_content)

    import openai

    monkeypatch.setattr(openai, "AzureOpenAI", _Stub)
    return calls


def test_query_rewriter_returns_original_first(monkeypatch, rewriter_env, tmp_path: Path):
    _patch_client(
        monkeypatch,
        json.dumps({"queries": ["alpha", "beta", "gamma"]}),
    )
    from benchmarks.query_rewrite import AzureOpenAIQueryRewriter

    r = AzureOpenAIQueryRewriter(cache_dir=tmp_path)
    out = r.rewrite("orig question", "multi-session")
    assert out[0] == "orig question"
    assert out[1:] == ["alpha", "beta", "gamma"]


def test_query_rewriter_falls_back_on_bad_json(monkeypatch, rewriter_env, tmp_path: Path):
    _patch_client(monkeypatch, "not a json")
    from benchmarks.query_rewrite import AzureOpenAIQueryRewriter

    r = AzureOpenAIQueryRewriter(cache_dir=tmp_path, max_retries=1)
    out = r.rewrite("orig", "aggregation")
    assert out == ["orig"]


def test_query_rewriter_caches(monkeypatch, rewriter_env, tmp_path: Path):
    calls = _patch_client(
        monkeypatch,
        json.dumps({"queries": ["a", "b", "c"]}),
    )
    from benchmarks.query_rewrite import AzureOpenAIQueryRewriter

    r = AzureOpenAIQueryRewriter(cache_dir=tmp_path)
    first = r.rewrite("same q", "multi-session")
    second = r.rewrite("same q", "multi-session")
    assert first == second
    assert calls["n"] == 1  # second call hit the cache


def test_query_rewriter_dedupes_against_original(monkeypatch, rewriter_env, tmp_path: Path):
    _patch_client(
        monkeypatch,
        json.dumps({"queries": ["orig", "alpha", "alpha"]}),
    )
    from benchmarks.query_rewrite import AzureOpenAIQueryRewriter

    r = AzureOpenAIQueryRewriter(cache_dir=tmp_path)
    out = r.rewrite("orig", "multi-session")
    assert out == ["orig", "alpha"]


def test_query_rewriter_protocol_compatibility(monkeypatch, rewriter_env, tmp_path: Path):
    _patch_client(monkeypatch, json.dumps({"queries": []}))
    from benchmarks.query_rewrite import AzureOpenAIQueryRewriter

    r = AzureOpenAIQueryRewriter(cache_dir=tmp_path)
    # Structural Protocol — duck typing.
    assert hasattr(r, "rewrite")
    assert callable(r.rewrite)
    assert hasattr(r, "name")
