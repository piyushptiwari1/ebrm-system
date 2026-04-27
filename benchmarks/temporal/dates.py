"""Date parsing for LongMemEval timestamps."""

from __future__ import annotations

import re
from datetime import datetime

# Examples accepted:
#   "2024/05/16 (Thu) 10:30"
#   "2024/05/16 10:30"
#   "2024/05/16"
_DATE_RE = re.compile(
    r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})"
    r"(?:\s*\([A-Za-z]{3}\))?"
    r"(?:\s+(\d{1,2}):(\d{2}))?"
    r"\s*$"
)


def parse_lme_date(s: str) -> datetime | None:
    """Parse a LongMemEval-format timestamp into ``datetime``.

    Returns ``None`` on malformed input rather than raising — a few
    episodes in the official dataset have empty / partial dates and we
    must keep them in the corpus instead of crashing the whole run.
    """
    if not isinstance(s, str):
        return None
    m = _DATE_RE.match(s)
    if not m:
        return None
    year, month, day, hour, minute = m.groups()
    try:
        return datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hour) if hour is not None else 0,
            minute=int(minute) if minute is not None else 0,
        )
    except ValueError:
        return None


def seconds_between(a: str | None, b: str | None) -> float | None:
    """Absolute distance in seconds between two LME timestamps; ``None``
    if either side is unparseable."""
    if a is None or b is None:
        return None
    da, db = parse_lme_date(a), parse_lme_date(b)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds())


__all__ = ["parse_lme_date", "seconds_between"]
