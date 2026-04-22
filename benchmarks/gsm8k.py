"""GSM8K adapter.

Loads GSM8K via `datasets` lazily (optional dep). Provides a Benchmark
implementation that yields `BenchmarkExample`s with parsed numeric answers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from benchmarks.runner import BenchmarkExample

_ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


class GSM8K:
    """GSM8K test-set adapter. Requires `datasets` installed."""

    name = "gsm8k"

    def __init__(self, split: str = "test", limit: int | None = None) -> None:
        self.split = split
        self.limit = limit

    def examples(self) -> Iterable[BenchmarkExample]:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "GSM8K benchmark requires `datasets`. "
                "Install with: pip install 'ebrm-system[benchmark]'"
            ) from exc

        ds = load_dataset("gsm8k", "main", split=self.split)
        for i, row in enumerate(ds):
            if self.limit is not None and i >= self.limit:
                break
            m = _ANSWER_RE.search(str(row["answer"]))
            if not m:
                continue
            yield BenchmarkExample(
                id=f"gsm8k-{self.split}-{i}",
                query=str(row["question"]),
                expected=float(m.group(1)),
            )
