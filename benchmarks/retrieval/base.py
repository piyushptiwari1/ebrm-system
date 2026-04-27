"""Retriever Protocol used by the LongMemEval pipeline.

A retriever maps an :class:`OfficialEpisode` to a ranked list of turns
together with their scores. The pipeline never looks at the absolute
score, only the rank — but we keep the score for diagnostics and for
the cross-encoder reranker which consumes (turn, score) tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn


@dataclass(frozen=True)
class ScoredTurn:
    """A retrieved turn together with its retriever score."""

    turn: OfficialTurn
    score: float


@runtime_checkable
class Retriever(Protocol):
    """Protocol shared by dense, BM25, RRF and reranker retrievers."""

    name: str

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        """Return up to ``top_k`` turns ranked by retriever-specific score."""
        ...


__all__ = ["Retriever", "ScoredTurn"]
