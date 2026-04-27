"""Loader for the official LongMemEval dataset (xiaowu0162/longmemeval-cleaned).

The official schema is **different** from our synthetic harness:

```
{
  "question_id": "gpt4_<hash>",   # ends with "_abs" for abstention questions
  "question_type": "single-session-user" | ... | "single-session-preference",
  "question": "...",
  "answer": "...",                # free-text gold answer
  "question_date": "YYYY/MM/DD (Day) HH:MM",
  "haystack_session_ids": [...],
  "haystack_dates": [...],
  "haystack_sessions": [[ {"role": "user|assistant", "content": "...",
                           "has_answer": true?}, ... ], ...],
  "answer_session_ids": [...]      # ground-truth session ids
}
```

This module converts each episode into an :class:`OfficialEpisode`, which is
a flat list of :class:`OfficialTurn` items plus the original question/answer.
The reader/retrieval modules consume these.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# All seven question types in the cleaned dataset (six "kinds" + abstention).
ALL_QUESTION_TYPES_OFFICIAL: tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


@dataclass(frozen=True)
class OfficialTurn:
    """A single chat turn from the haystack."""

    session_id: str
    session_idx: int
    turn_idx: int
    role: str  # "user" or "assistant"
    content: str
    session_date: str  # raw "YYYY/MM/DD (Day) HH:MM"
    has_answer: bool


@dataclass(frozen=True)
class OfficialEpisode:
    """One LongMemEval question with its full chat-history haystack."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    turns: tuple[OfficialTurn, ...]
    answer_session_ids: tuple[str, ...]
    is_abstention: bool

    @property
    def base_question_type(self) -> str:
        """Question type without the ``_abs`` suffix logic — abstention is
        flagged separately via :attr:`is_abstention`."""
        return self.question_type


def _flatten_episode(raw: dict[str, object]) -> OfficialEpisode:
    sessions = raw["haystack_sessions"]
    session_ids = raw["haystack_session_ids"]
    dates = raw["haystack_dates"]
    if not (
        isinstance(sessions, list) and isinstance(session_ids, list) and isinstance(dates, list)
    ):
        raise ValueError(f"malformed episode {raw.get('question_id')!r}")
    if not (len(sessions) == len(session_ids) == len(dates)):
        raise ValueError(
            f"length mismatch in episode {raw.get('question_id')!r}: "
            f"sessions={len(sessions)} ids={len(session_ids)} dates={len(dates)}"
        )

    turns: list[OfficialTurn] = []
    for s_idx, (session, sid, sdate) in enumerate(zip(sessions, session_ids, dates, strict=True)):
        if not isinstance(session, list):
            raise ValueError(f"session must be a list, got {type(session).__name__}")
        for t_idx, turn in enumerate(session):
            if not isinstance(turn, dict):
                raise ValueError(f"turn must be a dict, got {type(turn).__name__}")
            turns.append(
                OfficialTurn(
                    session_id=str(sid),
                    session_idx=s_idx,
                    turn_idx=t_idx,
                    role=str(turn.get("role", "")),
                    content=str(turn.get("content", "")),
                    session_date=str(sdate),
                    has_answer=bool(turn.get("has_answer", False)),
                )
            )

    qid = str(raw["question_id"])
    qtype = str(raw["question_type"])
    if qtype not in ALL_QUESTION_TYPES_OFFICIAL:
        raise ValueError(
            f"unknown question_type {qtype!r} in {qid!r}; "
            f"expected one of {ALL_QUESTION_TYPES_OFFICIAL}"
        )

    answer_sids = raw.get("answer_session_ids") or []
    if not isinstance(answer_sids, list):
        raise ValueError(f"answer_session_ids must be a list in {qid!r}")

    return OfficialEpisode(
        question_id=qid,
        question_type=qtype,
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        question_date=str(raw["question_date"]),
        turns=tuple(turns),
        answer_session_ids=tuple(str(s) for s in answer_sids),
        is_abstention=qid.endswith("_abs"),
    )


def load_longmemeval_official(path: str | Path) -> list[OfficialEpisode]:
    """Load the official ``longmemeval_*.json`` (oracle / S / M).

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError``
    on schema violations (with the offending question_id surfaced).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LongMemEval dataset not found: {p}")
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {p}, got {type(data).__name__}")
    return [_flatten_episode(d) for d in data]


def episodes_iter_question_types(eps: Iterable[OfficialEpisode]) -> dict[str, int]:
    """Helper for audit: count episodes per question_type."""
    out: dict[str, int] = {}
    for e in eps:
        out[e.question_type] = out.get(e.question_type, 0) + 1
    return out


__all__ = [
    "ALL_QUESTION_TYPES_OFFICIAL",
    "OfficialEpisode",
    "OfficialTurn",
    "episodes_iter_question_types",
    "load_longmemeval_official",
]
