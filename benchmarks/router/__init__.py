"""Heuristic question classifier for per-type top_k routing (v0.24).

LongMemEval ships a ground-truth ``question_type``, but for aggregation we
also need to know whether the question wants a count/sum spanning many
sessions (which benefits from a larger ``top_k``) or a single fact lookup
(which is better served by a focused ``top_k=5``). This module returns a
short routing tag that the runner maps to a top_k.
"""

from __future__ import annotations

# Aggregation cues — questions whose gold answer is a count, sum, or
# enumeration that may span more sessions than the default top_k=5/10.
_AGGREGATION_CUES: tuple[str, ...] = (
    "how many",
    "how much",
    "how often",
    "total",
    "in total",
    "altogether",
    "combined",
    "all of the",
    "list all",
    "enumerate",
    "sum of",
    "count of",
)

# Recommendation cues — questions where the reader must commit to a
# personalised suggestion grounded in stated preferences.
_RECOMMEND_CUES: tuple[str, ...] = (
    "can you suggest",
    "can you recommend",
    "could you suggest",
    "could you recommend",
    "what would you suggest",
    "what would you recommend",
    "what should i",
    "any suggestions",
    "any recommendations",
    "give me suggestions",
    "give me recommendations",
)

# Ordering cues — questions whose gold answer is "which of these came
# first / most recently". Used by the v0.28 temporal-ordering CoT reader
# (gated additionally by ``question_type == "temporal-reasoning"``).
_ORDERING_CUES: tuple[str, ...] = (
    "happened first",
    "came first",
    "did i meet first",
    "did i first",
    "started first",
    "did first",
    "happened more recent",
    "happened most recent",
    "happened later",
    "happened earlier",
    "earliest",
    "most recently",
    " before the ",
    " before i ",
    " after the ",
    " after i ",
    "which came",
    "who became",
    "which event happened",
    "which show did i",
    "which did i ",
    "which event did i",
)


def classify_question(question: str, question_type: str) -> str:
    """Return one of ``"aggregation"``, ``"temporal"``, ``"recommendation"``,
    or ``"general"`` based on the question text and known type.

    The classification is conservative: it only fires on clear surface cues
    so the default ``"general"`` path catches everything else.
    """
    q = question.lower()
    if any(c in q for c in _AGGREGATION_CUES):
        return "aggregation"
    if any(c in q for c in _RECOMMEND_CUES):
        return "recommendation"
    if question_type == "temporal-reasoning":
        return "temporal"
    return "general"


def top_k_for(tag: str, *, default: int) -> int:
    """Map a routing tag to a top_k.

    - ``aggregation`` → 20 (need to see all items before counting)
    - ``temporal``    → 5  (extra excerpts hurt chronology in v0.23 ablation)
    - ``recommendation`` / ``general`` → ``default`` (typically 10)
    """
    if tag == "aggregation":
        return max(default, 20)
    if tag == "temporal":
        return min(default, 5)
    return default


def is_multi_session_aggregation(question: str, question_type: str) -> bool:
    """v0.28 — fire aggregation CoT only when BOTH conditions hold.

    The v0.25 attempt fired on ``classify_question == "aggregation"`` alone,
    which leaked onto temporal questions like "how many days between X and
    Y" and dropped temporal-reasoning by 6 pt. Tightening the gate to also
    require the LongMemEval ground-truth question_type ``multi-session``
    eliminates that leak.
    """
    return (
        question_type == "multi-session"
        and classify_question(question, question_type) == "aggregation"
    )


def is_temporal_ordering(question: str, question_type: str) -> bool:
    """v0.28 — fire temporal-ordering CoT only on temporal-reasoning Qs
    that ask "which happened first / most recently / earliest", etc.

    Surface cues alone would over-fire; pairing them with the dataset's
    ground-truth ``temporal-reasoning`` type keeps gating tight.
    """
    if question_type != "temporal-reasoning":
        return False
    q = question.lower()
    return any(c in q for c in _ORDERING_CUES)


__all__ = [
    "classify_question",
    "is_multi_session_aggregation",
    "is_temporal_ordering",
    "top_k_for",
]
