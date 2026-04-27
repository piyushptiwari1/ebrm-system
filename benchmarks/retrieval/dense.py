"""Dense retriever: cosine top-k over an :class:`Embedder`."""

from __future__ import annotations

import numpy as np

from benchmarks.datasets.longmemeval_official import OfficialEpisode
from benchmarks.embedders.base import Embedder
from benchmarks.retrieval.base import ScoredTurn


class DenseRetriever:
    """Cosine top-k retriever over a pluggable :class:`Embedder`.

    The embedder is expected to return L2-normalised vectors so that the
    inner product equals cosine similarity (this is the contract of every
    embedder shipped in :mod:`benchmarks.embedders`).
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self.name = f"dense({embedder.name})"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        if not episode.turns:
            return []
        turn_vecs = self._embedder.embed([t.content for t in episode.turns])
        q_vec = self._embedder.embed([episode.question])[0]
        scores = turn_vecs @ q_vec  # cosine on normalised vectors
        k = min(top_k, len(scores))
        # argpartition -> exact top-k (unsorted), then sort that k-slice.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [ScoredTurn(episode.turns[int(i)], float(scores[int(i)])) for i in top_idx]


__all__ = ["DenseRetriever"]
