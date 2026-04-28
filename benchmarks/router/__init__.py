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


__all__ = ["classify_question", "top_k_for"]
