"""Post-retrieval neighbor expansion.

Given a list of retrieved turns, optionally expand each with its ±N
neighbors in the same session. This is a cheap recall boost for
multi-turn answers (multi-session, knowledge-update, single-session-user)
without adding any retrieval cost.

We deduplicate by (session_id, turn_idx) and preserve the original
relevance order — neighbors of the top-1 hit go in immediately after,
then neighbors of the top-2 hit, and so on. ``ScoredTurn.score`` is
copied from the parent hit so downstream consumers (the reader) still
get a meaningful ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.retrieval.base import Retriever, ScoredTurn


def _key(t: OfficialTurn) -> tuple[str, int, int]:
    return (t.session_id, t.session_idx, t.turn_idx)


@dataclass
class NeighborExpander:
    """Wrap a retriever; after retrieval, add ±``window`` same-session turns.

    Memory-augmented episodes use ``session_id="<sid>::mem"`` for memory
    turns, so memories never pull in raw turns and vice-versa — the
    neighbour rule keys on the literal ``session_id``.
    """

    base: Retriever
    window: int = 1

    @property
    def name(self) -> str:
        return f"neighbors({self.base.name},w={self.window})"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        # Ask the base retriever for top_k hits; we keep that exact set
        # plus their neighbours. We do NOT expand top_k itself — the
        # reader budget is set by the caller.
        hits = self.base.retrieve(episode, top_k=top_k)
        if self.window <= 0 or not hits:
            return hits

        # Index turns by session for O(1) neighbor lookup.
        by_session: dict[str, list[OfficialTurn]] = {}
        for t in episode.turns:
            by_session.setdefault(t.session_id, []).append(t)
        for turns in by_session.values():
            turns.sort(key=lambda t: t.turn_idx)

        seen: set[tuple[str, int, int]] = set()
        out: list[ScoredTurn] = []
        for hit in hits:
            for st in self._with_neighbors(hit, by_session):
                k = _key(st.turn)
                if k in seen:
                    continue
                seen.add(k)
                out.append(st)
        return out

    def _with_neighbors(
        self, hit: ScoredTurn, by_session: dict[str, list[OfficialTurn]]
    ) -> list[ScoredTurn]:
        session_turns = by_session.get(hit.turn.session_id, [])
        if not session_turns:
            return [hit]
        # Find the hit's index within its session; fall back to turn_idx
        # comparison if it isn't found (shouldn't happen, but be safe).
        try:
            pos = next(i for i, t in enumerate(session_turns) if t.turn_idx == hit.turn.turn_idx)
        except StopIteration:
            return [hit]
        lo = max(0, pos - self.window)
        hi = min(len(session_turns), pos + self.window + 1)
        # Emit hit first, then neighbours in turn-order.
        out: list[ScoredTurn] = [hit]
        for i in range(lo, hi):
            if i == pos:
                continue
            out.append(ScoredTurn(turn=session_turns[i], score=hit.score))
        return out


__all__ = ["NeighborExpander"]
