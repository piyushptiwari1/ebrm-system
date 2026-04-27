"""Protocol + dataclasses for LLM memory extraction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn


@dataclass(frozen=True)
class ExtractedMemory:
    """A single atomic memory distilled from one or more chat turns.

    Attributes
    ----------
    text:
        Self-contained statement of the fact (e.g. "User's daughter Lily
        has a peanut allergy."). Should make sense without surrounding
        context — that's the whole point of extraction.
    session_id:
        ID of the session it was extracted from.
    session_idx:
        Index of the session within the episode haystack.
    session_date:
        Original session date string ("YYYY/MM/DD (Day) HH:MM").
    role:
        ``"user"``, ``"assistant"``, or ``"memory"`` (the latter when the
        extractor merges info from both sides).
    source_turn_indices:
        Tuple of original turn indices within the source session that
        the memory was derived from. May be empty if the extractor does
        not track provenance.
    """

    text: str
    session_id: str
    session_idx: int
    session_date: str
    role: str
    source_turn_indices: tuple[int, ...] = ()


@runtime_checkable
class MemoryExtractor(Protocol):
    """Distill an episode's haystack into atomic ``ExtractedMemory`` items."""

    name: str

    def extract(self, episode: OfficialEpisode) -> list[ExtractedMemory]: ...


def memories_to_episode(
    episode: OfficialEpisode, memories: Iterable[ExtractedMemory]
) -> OfficialEpisode:
    """Wrap a list of extracted memories as a synthetic ``OfficialEpisode``.

    The resulting episode has the same ``question`` / ``answer`` /
    ``question_date`` etc. as ``episode`` but its ``turns`` are the
    extracted memories. ``turn_idx`` is reassigned to be the memory's
    position in the (stable) input order.

    This lets every existing retriever (dense, BM25, RRF, reranker) and
    the reader work unchanged on extracted memories.
    """
    new_turns = tuple(
        OfficialTurn(
            session_id=m.session_id,
            session_idx=m.session_idx,
            turn_idx=i,
            role=m.role,
            content=m.text,
            session_date=m.session_date,
            has_answer=False,
        )
        for i, m in enumerate(memories)
    )
    return replace(episode, turns=new_turns)


__all__ = [
    "ExtractedMemory",
    "MemoryExtractor",
    "memories_to_episode",
]
