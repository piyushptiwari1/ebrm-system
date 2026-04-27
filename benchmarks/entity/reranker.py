"""Entity-aware reranker — boost candidates mentioning question entities."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.datasets.longmemeval_official import OfficialEpisode
from benchmarks.entity.extractor import extract_entities
from benchmarks.retrieval.base import Retriever, ScoredTurn


@dataclass
class EntityReranker:
    """Up-weight retrieved turns whose content mentions entities from the
    question.

    The combined score is::

        final = (1 - alpha) * sem_norm + alpha * entity_score

    where ``entity_score = min(1.0, n_matches / max(1, n_entities))``.
    When the question has no detectable entities, the wrapper falls
    through to the base ranking unchanged.
    """

    base: Retriever
    alpha: float = 0.25
    candidate_k: int = 20

    @property
    def name(self) -> str:
        return f"entity({self.base.name},alpha={self.alpha})"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        k = max(top_k, self.candidate_k)
        candidates = self.base.retrieve(episode, top_k=k)
        if not candidates:
            return []

        entities = extract_entities(episode.question)
        if not entities:
            return candidates[:top_k]

        scores = [c.score for c in candidates]
        s_min, s_max = min(scores), max(scores)
        rng = s_max - s_min
        sem_norm = [1.0] * len(candidates) if rng <= 1e-12 else [(s - s_min) / rng for s in scores]

        n_ent = len(entities)
        rescored: list[ScoredTurn] = []
        for cand, sem in zip(candidates, sem_norm, strict=True):
            content = cand.turn.content.lower()
            matches = sum(1 for e in entities if e in content)
            entity_score = min(1.0, matches / max(1, n_ent))
            final = (1.0 - self.alpha) * sem + self.alpha * entity_score
            rescored.append(ScoredTurn(turn=cand.turn, score=final))

        rescored.sort(key=lambda st: st.score, reverse=True)
        return rescored[:top_k]


__all__ = ["EntityReranker"]
