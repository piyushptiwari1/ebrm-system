"""Tests for NeighborExpander (v0.19)."""

from __future__ import annotations

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.retrieval import NeighborExpander, ScoredTurn
from benchmarks.retrieval.base import Retriever


def _turn(s_idx: int, t_idx: int, content: str) -> OfficialTurn:
    return OfficialTurn(
        session_id=f"s{s_idx}",
        session_idx=s_idx,
        turn_idx=t_idx,
        role="user",
        content=content,
        session_date="2025/01/01 (Wed) 09:00",
        has_answer=False,
    )


def _ep(turns: list[OfficialTurn]) -> OfficialEpisode:
    return OfficialEpisode(
        question_id="q",
        question_type="multi-session",
        question="?",
        answer="x",
        question_date="2025/01/01 (Wed) 09:00",
        turns=tuple(turns),
        answer_session_ids=("s0",),
        is_abstention=False,
    )


class _FixedRetriever:
    name = "fixed"

    def __init__(self, picks: list[OfficialTurn]) -> None:
        self._picks = picks

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        return [ScoredTurn(turn=t, score=1.0 / (i + 1)) for i, t in enumerate(self._picks[:top_k])]


def test_neighbor_expander_is_a_retriever() -> None:
    base = _FixedRetriever([])
    exp = NeighborExpander(base, window=1)
    assert isinstance(exp, Retriever)
    assert "neighbors" in exp.name


def test_neighbor_window_zero_passthrough() -> None:
    turns = [_turn(0, i, f"t{i}") for i in range(5)]
    ep = _ep(turns)
    base = _FixedRetriever([turns[2]])
    exp = NeighborExpander(base, window=0)
    out = exp.retrieve(ep, top_k=1)
    assert [s.turn.turn_idx for s in out] == [2]


def test_neighbor_expansion_adds_neighbors_in_order_no_dupes() -> None:
    turns = [_turn(0, i, f"t{i}") for i in range(5)]
    ep = _ep(turns)
    # Pick t2 and t3 — they share neighbor t2/t3, must dedupe.
    base = _FixedRetriever([turns[2], turns[3]])
    exp = NeighborExpander(base, window=1)
    out = exp.retrieve(ep, top_k=2)
    indices = [s.turn.turn_idx for s in out]
    # First the top hit (2) then its neighbors (1, 3) in turn order;
    # then top-2 (3) and its neighbor (4). t2/t3 already seen → skipped.
    assert indices == [2, 1, 3, 4]


def test_neighbor_expansion_respects_session_boundary() -> None:
    turns = [_turn(0, 0, "a"), _turn(0, 1, "b"), _turn(1, 0, "c"), _turn(1, 1, "d")]
    ep = _ep(turns)
    base = _FixedRetriever([turns[1]])  # s0 turn 1 — neighbor s1.0 must NOT be pulled
    exp = NeighborExpander(base, window=1)
    out = exp.retrieve(ep, top_k=1)
    sids = {s.turn.session_id for s in out}
    assert sids == {"s0"}


def test_neighbor_expansion_keeps_memory_namespace_separate() -> None:
    """Memory turns (session_id ending ::mem) must not pull in raw neighbours."""
    raw = [_turn(0, i, f"t{i}") for i in range(3)]
    mem = OfficialTurn(
        session_id="s0::mem",
        session_idx=0,
        turn_idx=3,
        role="memory",
        content="distilled",
        session_date="2025/01/01 (Wed) 09:00",
        has_answer=False,
    )
    ep = _ep([*raw, mem])
    base = _FixedRetriever([mem])
    exp = NeighborExpander(base, window=1)
    out = exp.retrieve(ep, top_k=1)
    sids = {s.turn.session_id for s in out}
    assert sids == {"s0::mem"}
