"""Tests for v0.18 retrieval components.

Covers BM25Retriever, DenseRetriever, RRFRetriever and the public Retriever
Protocol contract. The cross-encoder reranker is exercised by a mocked
sentence_transformers stub injected via sys.modules.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.embedders.hash import HashEmbedder
from benchmarks.retrieval import DenseRetriever, RRFRetriever
from benchmarks.retrieval.base import Retriever, ScoredTurn

bm25 = pytest.importorskip("rank_bm25", reason="BM25Retriever needs rank-bm25")
from benchmarks.retrieval import BM25Retriever  # noqa: E402


def _episode(turn_contents: list[str], question: str) -> OfficialEpisode:
    turns = [
        OfficialTurn(
            session_id=f"s{i}",
            session_idx=0,
            turn_idx=i,
            role="user" if i % 2 == 0 else "assistant",
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
        turns=turns,
        answer_session_ids=[],
        is_abstention=False,
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_retriever_protocol_runtime_checkable():
    r = DenseRetriever(HashEmbedder(dim=64))
    assert isinstance(r, Retriever)
    assert isinstance(BM25Retriever(), Retriever)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


class TestDenseRetriever:
    def test_returns_top_k_in_descending_score(self):
        ep = _episode(
            ["The capital of France is Paris.", "Banana milkshake recipe.", "Paris weather today."],
            question="Where is Paris?",
        )
        r = DenseRetriever(HashEmbedder(dim=128, seed=0))
        out = r.retrieve(ep, top_k=2)
        assert len(out) == 2
        assert all(isinstance(s, ScoredTurn) for s in out)
        assert out[0].score >= out[1].score

    def test_empty_episode(self):
        ep = _episode([], "anything")
        r = DenseRetriever(HashEmbedder(dim=64))
        assert r.retrieve(ep, top_k=5) == []

    def test_top_k_caps_at_haystack_size(self):
        ep = _episode(["a", "b"], "a")
        r = DenseRetriever(HashEmbedder(dim=64))
        assert len(r.retrieve(ep, top_k=10)) == 2

    def test_name_includes_embedder(self):
        r = DenseRetriever(HashEmbedder(dim=64))
        assert "dense" in r.name and "hash" in r.name


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class TestBM25Retriever:
    def test_lexical_match_wins(self):
        ep = _episode(
            [
                "I bought a yellow umbrella yesterday.",
                "The weather forecast says rain.",
                "Tropical fruits include mango and pineapple.",
            ],
            question="umbrella yellow",
        )
        r = BM25Retriever()
        out = r.retrieve(ep, top_k=1)
        assert len(out) == 1
        assert "umbrella" in out[0].turn.content.lower()

    def test_empty_episode(self):
        r = BM25Retriever()
        assert r.retrieve(_episode([], "q"), top_k=3) == []

    def test_handles_empty_token_corpus(self):
        # Whitespace-only / punctuation-only turns get a placeholder so
        # rank-bm25 doesn't blow up.
        ep = _episode(["   ", "...", "actual content"], "content")
        r = BM25Retriever()
        out = r.retrieve(ep, top_k=2)
        assert len(out) == 2

    def test_name(self):
        assert BM25Retriever().name == "bm25"


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


class _FakeRetriever:
    """Returns a fixed ranking — useful for deterministic RRF tests."""

    def __init__(self, name: str, indices: list[int]):
        self.name = name
        self._indices = indices

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        out = []
        for rank, idx in enumerate(self._indices[:top_k], start=1):
            # Fake score = inverse rank; RRF should ignore this.
            out.append(ScoredTurn(episode.turns[idx], 1.0 / rank))
        return out


class TestRRFRetriever:
    def test_combines_two_retrievers(self):
        ep = _episode(["a", "b", "c", "d"], "q")
        # A ranks: 0,1,2,3   B ranks: 3,2,1,0
        # turn 0 -> 1/(60+1) + 1/(60+4) ; turn 3 same flipped.
        # turn 1 -> 1/(60+2) + 1/(60+3) ; turn 2 same.
        # turns 0/3 win over 1/2.
        a = _FakeRetriever("A", [0, 1, 2, 3])
        b = _FakeRetriever("B", [3, 2, 1, 0])
        rrf = RRFRetriever([a, b], rrf_k=60, per_retriever_k=4)
        out = rrf.retrieve(ep, top_k=2)
        ids = sorted(st.turn.turn_idx for st in out)
        assert ids == [0, 3]

    def test_dedupes_when_both_retrievers_agree(self):
        ep = _episode(["a", "b", "c"], "q")
        a = _FakeRetriever("A", [0, 1, 2])
        b = _FakeRetriever("B", [0, 1, 2])
        rrf = RRFRetriever([a, b], rrf_k=60, per_retriever_k=3)
        out = rrf.retrieve(ep, top_k=3)
        # Three unique turns, agreement boosts top-1's score.
        assert len({st.turn.turn_idx for st in out}) == 3
        assert out[0].turn.turn_idx == 0

    def test_empty_episode(self):
        rrf = RRFRetriever([_FakeRetriever("A", [])])
        assert rrf.retrieve(_episode([], "q"), top_k=5) == []

    def test_requires_at_least_one_retriever(self):
        with pytest.raises(ValueError):
            RRFRetriever([])

    def test_name_includes_components(self):
        a = _FakeRetriever("foo", [0])
        b = _FakeRetriever("bar", [0])
        assert RRFRetriever([a, b]).name == "rrf(foo,bar)"


# ---------------------------------------------------------------------------
# Reranker (mocked sentence_transformers)
# ---------------------------------------------------------------------------


def _ensure_fake_st() -> None:
    """Inject a stub ``sentence_transformers`` so reranker import succeeds offline."""
    if "sentence_transformers" in sys.modules:
        return
    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = MagicMock()  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = mod


class TestCrossEncoderReranker:
    def test_reorders_candidates_by_cross_encoder_score(self):
        _ensure_fake_st()
        from benchmarks.retrieval.reranker import CrossEncoderReranker

        ep = _episode(["a", "b", "c"], "q")

        # The base retriever returns [0,1,2] with strictly decreasing scores
        # (so the dense baseline would pick 0 first).
        base = _FakeRetriever("base", [0, 1, 2])

        # Mock CrossEncoder.predict so that turn index 2 wins.
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "sentence_transformers.CrossEncoder"
        ) as ce_cls:
            ce_inst = MagicMock()
            ce_inst.predict.return_value = np.array([0.1, 0.2, 0.9])
            ce_cls.return_value = ce_inst

            r = CrossEncoderReranker(base, candidate_k=3)
            out = r.retrieve(ep, top_k=2)

        assert out[0].turn.turn_idx == 2
        assert out[0].score == pytest.approx(0.9)
        assert out[1].turn.turn_idx == 1

    def test_empty_candidates_passthrough(self):
        _ensure_fake_st()
        from benchmarks.retrieval.reranker import CrossEncoderReranker

        ep = _episode([], "q")
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "sentence_transformers.CrossEncoder"
        ) as ce_cls:
            ce_cls.return_value = MagicMock()
            r = CrossEncoderReranker(_FakeRetriever("base", []), candidate_k=10)
            assert r.retrieve(ep, top_k=5) == []
