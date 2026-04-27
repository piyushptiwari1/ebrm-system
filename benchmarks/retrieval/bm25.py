"""BM25 lexical retriever using ``rank-bm25``.

Per-episode index is built lazily on each call; the BM25Okapi structure
is cheap to build for the per-episode haystacks LongMemEval gives us
(<= a few thousand turns), so we trade a tiny amount of CPU for full
isolation between episodes.
"""

from __future__ import annotations

import re

import numpy as np

from benchmarks.datasets.longmemeval_official import OfficialEpisode
from benchmarks.retrieval.base import ScoredTurn

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer used by BM25.

    Matches the tokenizer assumed by the LongMemEval paper and by most
    open-source BM25 baselines. We keep digits because dates, prices and
    numbers are common targets in the dataset.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Per-episode BM25 retriever using ``rank-bm25``."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BM25Retriever requires `rank-bm25`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        if not episode.turns:
            return []
        from rank_bm25 import BM25Okapi

        tokenised = [_tokenize(t.content) for t in episode.turns]
        # rank-bm25 fails on a corpus that's entirely empty after tokenisation,
        # so guard with a single placeholder token.
        tokenised = [toks if toks else ["__empty__"] for toks in tokenised]
        bm25 = BM25Okapi(tokenised, k1=self._k1, b=self._b)
        q_tokens = _tokenize(episode.question) or ["__empty__"]
        scores = bm25.get_scores(q_tokens)
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [ScoredTurn(episode.turns[int(i)], float(scores[int(i)])) for i in top_idx]


__all__ = ["BM25Retriever"]
