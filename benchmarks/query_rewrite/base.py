"""Protocol for query rewriters."""

from __future__ import annotations

from typing import Protocol


class QueryRewriter(Protocol):
    """Generate alternative phrasings of a question.

    Implementations MUST always include the original ``question`` as the
    first element of the returned list, and MUST never raise — fall back
    to ``[question]`` on any backend error.
    """

    name: str

    def rewrite(self, question: str, question_type: str) -> list[str]: ...


__all__ = ["QueryRewriter"]
