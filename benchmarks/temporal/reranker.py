"""Temporal-aware reranker.

Wraps any :class:`Retriever` and reorders its hits by combining the
original score with a temporal proximity score derived from the
question_date and each turn's session_date. Useful for the
*temporal-reasoning* split where many turns have similar semantic
similarity but only one is at the right point in time.

The combined score is

    final = (1 - alpha) * semantic_norm + alpha * recency_score

where ``semantic_norm`` is the original score linearly normalised to
[0, 1] within the retrieved set, and ``recency_score`` is

    recency_score = exp(-days_apart / decay_days)

Both pieces are bounded in [0, 1]. ``alpha`` defaults to 0.3 (semantic
dominates, time provides a tie-breaker).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmarks.datasets.longmemeval_official import OfficialEpisode
from benchmarks.retrieval.base import Retriever, ScoredTurn
from benchmarks.temporal.dates import seconds_between

_DAY = 86400.0


@dataclass
class TemporalReranker:
    """Combine semantic score with a recency boost relative to question_date."""

    base: Retriever
    alpha: float = 0.3
    decay_days: float = 30.0
    candidate_k: int = 20

    @property
    def name(self) -> str:
        return f"temporal({self.base.name},alpha={self.alpha},decay={self.decay_days}d)"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        # Pull more candidates than we need so the reranker has room.
        k = max(top_k, self.candidate_k)
        candidates = self.base.retrieve(episode, top_k=k)
        if not candidates:
            return []

        # Normalise base scores into [0, 1] within the candidate set.
        scores = [c.score for c in candidates]
        s_min, s_max = min(scores), max(scores)
        rng = s_max - s_min
        sem_norm = [1.0] * len(candidates) if rng <= 1e-12 else [(s - s_min) / rng for s in scores]

        rescored: list[ScoredTurn] = []
        for cand, sem in zip(candidates, sem_norm, strict=True):
            secs = seconds_between(episode.question_date, cand.turn.session_date)
            if secs is None:
                # Date unparseable — fall back to pure semantic score.
                final = sem
            else:
                days = secs / _DAY
                recency = math.exp(-days / max(self.decay_days, 1e-6))
                final = (1.0 - self.alpha) * sem + self.alpha * recency
            rescored.append(ScoredTurn(turn=cand.turn, score=final))

        rescored.sort(key=lambda st: st.score, reverse=True)
        return rescored[:top_k]


__all__ = ["TemporalReranker"]
