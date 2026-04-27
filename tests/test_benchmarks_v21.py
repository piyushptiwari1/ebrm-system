"""Tests for v0.21 LLM fusion reranker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openai", reason="LLMFusionReranker needs openai")

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.retrieval.base import Retriever, ScoredTurn


def _turn(content: str, idx: int = 0, date: str = "2024/01/01 (Mon) 10:00") -> OfficialTurn:
    return OfficialTurn(
        session_id=f"s{idx}",
        session_idx=idx,
        turn_idx=0,
        role="user",
        content=content,
        session_date=date,
        has_answer=False,
    )


def _ep() -> OfficialEpisode:
    return OfficialEpisode(
        question_id="q1",
        question_type="multi-session",
        question="What car does Alice own?",
        answer="Tesla",
        question_date="2024/06/01 (Sat) 12:00",
        turns=(),
        answer_session_ids=("s0",),
        is_abstention=False,
    )


class _Fixed:
    name = "fixed"

    def __init__(self, picks: list[ScoredTurn]) -> None:
        self._picks = picks

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        return list(self._picks[:top_k])


class _FakeOpenAI:
    """Minimal stand-in for openai.AzureOpenAI used in unit tests."""

    def __init__(self, ranking: list[int] | None = None, raise_n: int = 0) -> None:
        self._ranking = ranking
        self._raise_n = raise_n
        self.calls = 0

        outer = self

        class _Choice:
            def __init__(self, content: str) -> None:
                class _Msg:
                    def __init__(self, c: str) -> None:
                        self.content = c

                self.message = _Msg(content)

        class _Resp:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]

        class _Completions:
            def create(self, **kwargs: Any) -> _Resp:
                outer.calls += 1
                if outer.calls <= outer._raise_n:
                    raise RuntimeError("simulated transient")
                payload = {"ranking": outer._ranking} if outer._ranking is not None else {}
                return _Resp(json.dumps(payload))

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()


@pytest.fixture(autouse=True)
def _azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.example.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-02-01")


def _patch_client(monkeypatch: pytest.MonkeyPatch, fake: _FakeOpenAI) -> None:
    import openai

    monkeypatch.setattr(openai, "AzureOpenAI", lambda **_: fake)


def test_fusion_reranks_in_returned_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.fusion import LLMFusionReranker

    base = _Fixed(
        [
            ScoredTurn(_turn("ford pinto", 0), 0.9),
            ScoredTurn(_turn("alice drives a tesla", 1), 0.8),
            ScoredTurn(_turn("weather is nice", 2), 0.7),
        ]
    )
    fake = _FakeOpenAI(ranking=[1, 0, 2])
    _patch_client(monkeypatch, fake)

    reranker = LLMFusionReranker(base=base, candidate_k=3, cache_dir=tmp_path)
    out = reranker.retrieve(_ep(), top_k=2)

    assert [s.turn.content for s in out] == ["alice drives a tesla", "ford pinto"]
    assert fake.calls == 1
    assert isinstance(reranker, Retriever)


def test_fusion_uses_disk_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from benchmarks.fusion import LLMFusionReranker

    base = _Fixed(
        [
            ScoredTurn(_turn("a", 0), 0.5),
            ScoredTurn(_turn("b", 1), 0.4),
        ]
    )
    fake = _FakeOpenAI(ranking=[1, 0])
    _patch_client(monkeypatch, fake)

    reranker = LLMFusionReranker(base=base, candidate_k=2, cache_dir=tmp_path)
    reranker.retrieve(_ep(), top_k=2)
    reranker.retrieve(_ep(), top_k=2)
    assert fake.calls == 1


def test_fusion_handles_malformed_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.fusion import LLMFusionReranker

    base = _Fixed(
        [
            ScoredTurn(_turn("a", 0), 0.5),
            ScoredTurn(_turn("b", 1), 0.4),
        ]
    )
    fake = _FakeOpenAI(ranking=None)
    _patch_client(monkeypatch, fake)

    reranker = LLMFusionReranker(base=base, candidate_k=2, cache_dir=tmp_path)
    out = reranker.retrieve(_ep(), top_k=2)
    assert [s.turn.content for s in out] == ["a", "b"]


def test_fusion_sanitizes_invalid_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.fusion import LLMFusionReranker

    base = _Fixed(
        [
            ScoredTurn(_turn("a", 0), 0.5),
            ScoredTurn(_turn("b", 1), 0.4),
            ScoredTurn(_turn("c", 2), 0.3),
        ]
    )
    fake = _FakeOpenAI(ranking=[2, 7, "foo", 2, 0])  # type: ignore[list-item]
    _patch_client(monkeypatch, fake)

    reranker = LLMFusionReranker(base=base, candidate_k=3, cache_dir=tmp_path)
    out = reranker.retrieve(_ep(), top_k=3)
    # 2, 0 valid + missing index 1 backfilled.
    assert [s.turn.content for s in out] == ["c", "a", "b"]


def test_fusion_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.fusion import llm_fusion

    monkeypatch.setattr(llm_fusion.time, "sleep", lambda *_: None)
    base = _Fixed(
        [
            ScoredTurn(_turn("a", 0), 0.5),
            ScoredTurn(_turn("b", 1), 0.4),
        ]
    )
    fake = _FakeOpenAI(ranking=[1, 0], raise_n=2)
    _patch_client(monkeypatch, fake)

    reranker = llm_fusion.LLMFusionReranker(base=base, candidate_k=2, cache_dir=tmp_path)
    out = reranker.retrieve(_ep(), top_k=2)
    assert [s.turn.content for s in out] == ["b", "a"]
    assert fake.calls == 3


def test_fusion_passthrough_for_empty_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.fusion import LLMFusionReranker

    fake = _FakeOpenAI(ranking=[])
    _patch_client(monkeypatch, fake)
    base = _Fixed([])
    reranker = LLMFusionReranker(base=base, candidate_k=5, cache_dir=tmp_path)
    assert reranker.retrieve(_ep(), top_k=5) == []
    assert fake.calls == 0


def test_fusion_passthrough_for_single_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.fusion import LLMFusionReranker

    fake = _FakeOpenAI(ranking=[0])
    _patch_client(monkeypatch, fake)
    base = _Fixed([ScoredTurn(_turn("only", 0), 0.5)])
    reranker = LLMFusionReranker(base=base, candidate_k=5, cache_dir=tmp_path)
    out = reranker.retrieve(_ep(), top_k=5)
    assert len(out) == 1
    assert fake.calls == 0


def test_fusion_name_property(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from benchmarks.fusion import LLMFusionReranker

    _patch_client(monkeypatch, _FakeOpenAI(ranking=[]))
    base = _Fixed([])
    reranker = LLMFusionReranker(base=base, candidate_k=10, cache_dir=tmp_path)
    assert "llm-fusion" in reranker.name
    assert "fixed" in reranker.name
