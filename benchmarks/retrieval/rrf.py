"""Reciprocal Rank Fusion (RRF) over multiple retrievers.

RRF is the de-facto fusion of choice for hybrid (BM25 + dense) retrieval
because it's parameter-free w.r.t. score scales — it only sees ranks.

    RRF_score(t) = sum_r  1 / (k + rank_r(t))

Default ``k = 60`` follows Cormack et al. (2009) and reproduces the
TREC-style hybrid baseline used in the LongMemEval paper.
"""

from __future__ import annotations

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.retrieval.base import Retriever, ScoredTurn


class RRFRetriever:
    """Combine an arbitrary number of retrievers via RRF.

    Each underlying retriever is queried with its own ``per_retriever_k``
    (default: ``top_k * 4``) and the union of returned turns is then
    re-ranked by the sum of reciprocal ranks across retrievers.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        rrf_k: int = 60,
        per_retriever_k: int | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("RRFRetriever requires at least one retriever")
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._per_retriever_k = per_retriever_k
        self.name = "rrf(" + ",".join(r.name for r in retrievers) + ")"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        if not episode.turns:
            return []
        per_k = self._per_retriever_k if self._per_retriever_k is not None else top_k * 4
        # rrf_score keyed by (session_id, session_idx, turn_idx) so we
        # collapse identical turns surfaced by multiple retrievers.
        rrf_score: dict[tuple[str, int, int], float] = {}
        repr_turn: dict[tuple[str, int, int], OfficialTurn] = {}
        for retr in self._retrievers:
            for rank, st in enumerate(retr.retrieve(episode, top_k=per_k), start=1):
                key = (st.turn.session_id, st.turn.session_idx, st.turn.turn_idx)
                rrf_score[key] = rrf_score.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                repr_turn.setdefault(key, st.turn)
        ranked = sorted(rrf_score.items(), key=lambda kv: kv[1], reverse=True)
        out: list[ScoredTurn] = []
        for key, score in ranked[:top_k]:
            out.append(ScoredTurn(repr_turn[key], score))
        return out


__all__ = ["RRFRetriever"]
