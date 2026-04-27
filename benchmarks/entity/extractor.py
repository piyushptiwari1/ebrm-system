"""Regex-based entity extractor for LongMemEval questions.

We deliberately avoid spaCy / NLTK to keep the benchmark dependency
footprint identical to v0.19. The extractor recovers four useful
classes of literal that questions tend to anchor on:

1. **Quoted phrases**: ``"the project Aurora"`` → ``the project Aurora``.
2. **Capitalised proper-noun runs** (≥ 1 word, not at sentence start
   alone): names, places, products. We exclude common stop-starts.
3. **Standalone numbers and dates**: ``42``, ``2024``, ``$300``, ``5kg``.
4. **All-caps acronyms**: ``NASA``, ``GPT``.

Returns a list of lower-cased substrings for substring matching against
turn content. Empty string queries return [] (callers must guard).
"""

from __future__ import annotations

import re

# Stop-words that often start sentences but are not entities by themselves.
_SENTENCE_STARTERS = frozenset(
    {
        "What",
        "When",
        "Where",
        "Why",
        "How",
        "Who",
        "Which",
        "I",
        "We",
        "You",
        "They",
        "He",
        "She",
        "It",
        "The",
        "A",
        "An",
        "Did",
        "Does",
        "Do",
        "Is",
        "Are",
        "Was",
        "Were",
        "Has",
        "Have",
        "Can",
        "Could",
        "Should",
        "Would",
        "Will",
        "Tell",
        "Please",
        "On",
        "In",
        "At",
        "Of",
        "For",
        "By",
        "With",
        "From",
        "To",
    }
)

_QUOTED = re.compile(r"[\"\u201c\u201d']([^\"\u201c\u201d']{1,80})[\"\u201c\u201d']")
_PROPER = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_NUMBER = re.compile(r"\b\$?\d+(?:[.,]\d+)?(?:[a-zA-Z]{1,3})?\b")


def _normalise(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        s2 = s.strip().lower()
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        out.append(s2)
    return out


def extract_entities(text: str) -> list[str]:
    """Extract entity substrings from a question.

    Returns lower-cased substrings, deduplicated, in extraction order.
    """
    if not text:
        return []

    quoted = [m.group(1) for m in _QUOTED.finditer(text)]

    # Strip quoted segments before proper-noun extraction so we don't
    # double-count their internals.
    stripped = _QUOTED.sub(" ", text)

    proper: list[str] = []
    for m in _PROPER.finditer(stripped):
        token = m.group(0)
        # Skip when the only token is a sentence-starter like "What".
        first = token.split()[0]
        if len(token.split()) == 1 and first in _SENTENCE_STARTERS:
            continue
        proper.append(token)

    acronyms = [m.group(0) for m in _ACRONYM.finditer(stripped)]
    numbers = [m.group(0) for m in _NUMBER.finditer(stripped)]

    return _normalise([*quoted, *proper, *acronyms, *numbers])


__all__ = ["extract_entities"]
