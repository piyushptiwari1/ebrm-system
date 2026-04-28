"""Multi-query retrieval via RRF over LLM-rewritten queries (v0.27).

For multi-session and aggregation questions, a single phrasing of the
question often misses turns that use synonyms, paraphrases, or describe
the underlying entity differently. We delegate query rewriting to an
LLM (``QueryRewriter``), run the base retriever once per rewritten
query (with the question field of the episode swapped in), and RRF-fuse
the candidate lists into a single ranking.

This module is a thin wrapper — gating ("apply only for question_type
in {multi-session, aggregation}") is the runner's responsibility, so
this stays cheap and predictable inside unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.query_rewrite.base import QueryRewriter
from benchmarks.retrieval.base import Retriever, ScoredTurn


@dataclass
class MultiQueryRetriever:
    """Run ``base.retrieve`` once per LLM-rewritten query and RRF-fuse.

    The base retriever sees the same episode each time, with only the
    ``question`` field substituted to the rewritten query so any
    downstream embedder / BM25 sees the alternative phrasing.
    """

    base: Retriever
    rewriter: QueryRewriter
    rrf_k: int = 60
    per_query_k: int | None = None

    @property
    def name(self) -> str:
        return f"multi-query({self.base.name},rewriter={self.rewriter.name})"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        if not episode.turns:
            return []
        per_k = self.per_query_k if self.per_query_k is not None else top_k * 4
        queries = self.rewriter.rewrite(episode.question, episode.question_type)
        if not queries:
            queries = [episode.question]

        rrf_score: dict[tuple[str, int, int], float] = {}
        repr_turn: dict[tuple[str, int, int], OfficialTurn] = {}
        for q in queries:
            sub_episode = replace(episode, question=q)
            for rank, st in enumerate(
                self.base.retrieve(sub_episode, top_k=per_k), start=1
            ):
                key = (st.turn.session_id, st.turn.session_idx, st.turn.turn_idx)
                rrf_score[key] = rrf_score.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                repr_turn.setdefault(key, st.turn)

        ranked = sorted(rrf_score.items(), key=lambda kv: kv[1], reverse=True)
        return [ScoredTurn(repr_turn[key], score) for key, score in ranked[:top_k]]


__all__ = ["MultiQueryRetriever"]
