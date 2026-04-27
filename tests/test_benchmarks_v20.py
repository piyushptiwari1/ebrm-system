"""Tests for v0.20 temporal + entity rerankers."""

from __future__ import annotations

from datetime import datetime

import pytest
from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.entity import EntityReranker, extract_entities
from benchmarks.retrieval.base import Retriever, ScoredTurn
from benchmarks.temporal import TemporalReranker, parse_lme_date, seconds_between


def _turn(date: str, content: str, t_idx: int = 0) -> OfficialTurn:
    return OfficialTurn(
        session_id=f"s{t_idx}",
        session_idx=0,
        turn_idx=t_idx,
        role="user",
        content=content,
        session_date=date,
        has_answer=False,
    )


def _ep(question: str, qdate: str, turns: list[OfficialTurn]) -> OfficialEpisode:
    return OfficialEpisode(
        question_id="q",
        question_type="temporal-reasoning",
        question=question,
        answer="x",
        question_date=qdate,
        turns=tuple(turns),
        answer_session_ids=("s0",),
        is_abstention=False,
    )


class _Fixed:
    name = "fixed"

    def __init__(self, picks: list[ScoredTurn]) -> None:
        self._picks = picks

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        return list(self._picks[:top_k])


# --- date parsing ---------------------------------------------------------


def test_parse_lme_date_full_format() -> None:
    assert parse_lme_date("2024/05/16 (Thu) 10:30") == datetime(2024, 5, 16, 10, 30)


def test_parse_lme_date_no_weekday() -> None:
    assert parse_lme_date("2024/05/16 10:30") == datetime(2024, 5, 16, 10, 30)


def test_parse_lme_date_date_only() -> None:
    assert parse_lme_date("2024/05/16") == datetime(2024, 5, 16, 0, 0)


def test_parse_lme_date_garbage_returns_none() -> None:
    assert parse_lme_date("not a date") is None
    assert parse_lme_date("") is None
    assert parse_lme_date(None) is None  # type: ignore[arg-type]
    assert parse_lme_date("2024/13/40") is None  # invalid month


def test_seconds_between_handles_missing() -> None:
    assert seconds_between(None, "2024/05/16") is None
    assert seconds_between("2024/05/16", "2024/05/17") == pytest.approx(86400.0)


# --- TemporalReranker ----------------------------------------------------


def test_temporal_reranker_prefers_recent_when_tied() -> None:
    qdate = "2024/05/16 (Thu) 10:00"
    near = _turn("2024/05/16 (Thu) 09:00", "near", 0)
    far = _turn("2023/01/01 (Sun) 09:00", "far", 1)
    base = _Fixed(
        [
            ScoredTurn(turn=far, score=0.9),
            ScoredTurn(turn=near, score=0.9),
        ]
    )
    rr = TemporalReranker(base, alpha=0.5, decay_days=30.0, candidate_k=2)
    out = rr.retrieve(_ep("when?", qdate, [near, far]), top_k=2)
    assert out[0].turn.content == "near"
    assert out[1].turn.content == "far"


def test_temporal_reranker_unparseable_dates_passthrough() -> None:
    near = _turn("not a date", "x")
    base = _Fixed([ScoredTurn(turn=near, score=0.5)])
    rr = TemporalReranker(base, alpha=1.0)
    [hit] = rr.retrieve(_ep("?", "also bad", [near]), top_k=1)
    assert hit.turn.content == "x"


def test_temporal_reranker_empty() -> None:
    rr = TemporalReranker(_Fixed([]))
    assert rr.retrieve(_ep("?", "2024/05/16", []), top_k=5) == []


def test_temporal_reranker_is_a_retriever() -> None:
    assert isinstance(TemporalReranker(_Fixed([])), Retriever)


# --- entity extraction ---------------------------------------------------


def test_extract_entities_quoted_proper_acronym_number() -> None:
    q = 'When did the user mention "the project Aurora" with NASA in 2024?'
    ents = extract_entities(q)
    assert "the project aurora" in ents
    assert "nasa" in ents
    assert "2024" in ents


def test_extract_entities_skips_starter_words() -> None:
    ents = extract_entities("What did I say about Aurora?")
    assert "what" not in ents
    assert "i" not in ents
    assert "aurora" in ents


def test_extract_entities_empty() -> None:
    assert extract_entities("") == []
    assert extract_entities("how are you?") == []


# --- EntityReranker ------------------------------------------------------


def test_entity_reranker_boosts_matching_turn() -> None:
    qdate = "2024/05/16 10:00"
    hit_a = _turn("2024/05/15 10:00", "I worked on Aurora yesterday", 0)
    hit_b = _turn("2024/05/15 10:00", "Random small talk", 1)
    base = _Fixed(
        [
            ScoredTurn(turn=hit_b, score=0.9),
            ScoredTurn(turn=hit_a, score=0.9),
        ]
    )
    rr = EntityReranker(base, alpha=0.5, candidate_k=2)
    ep = _ep("What did I say about Aurora?", qdate, [hit_a, hit_b])
    [first, _] = rr.retrieve(ep, top_k=2)
    assert "Aurora" in first.turn.content


def test_entity_reranker_passthrough_when_no_entities() -> None:
    qdate = "2024/05/16 10:00"
    a = _turn("2024/05/15 10:00", "anything", 0)
    base = _Fixed([ScoredTurn(turn=a, score=0.5)])
    rr = EntityReranker(base, alpha=0.9)
    [hit] = rr.retrieve(_ep("how are you", qdate, [a]), top_k=1)
    assert hit.score == 0.5


def test_entity_reranker_empty() -> None:
    rr = EntityReranker(_Fixed([]))
    assert rr.retrieve(_ep("?", "2024/05/16", []), top_k=5) == []


def test_entity_reranker_is_a_retriever() -> None:
    assert isinstance(EntityReranker(_Fixed([])), Retriever)
