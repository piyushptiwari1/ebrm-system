"""Temporal helpers and a temporal-aware retriever wrapper.

LongMemEval session_date strings look like ``"2024/05/16 (Thu) 10:30"``.
This module parses them into :class:`datetime` and provides:

- :func:`parse_lme_date` — robust parser for the official format.
- :class:`TemporalReranker` — a retriever wrapper that reorders the base
  retriever's hits by combining the original semantic score with a
  recency / proximity score relative to the question_date.

Recency boost is critical for the *temporal-reasoning* split: the
question is typically of the form "When did the user say X?" or
"What did the user mention last week?", where the right turn is one of
many candidates with similar semantic similarity but very different
distances in time from the question.
"""

from benchmarks.temporal.dates import (
    parse_lme_date,
    seconds_between,
)
from benchmarks.temporal.reranker import TemporalReranker

__all__ = [
    "TemporalReranker",
    "parse_lme_date",
    "seconds_between",
]
