"""Benchmark harness.

Defines an abstract `Benchmark` interface; adapters live next to it (e.g. gsm8k.py).
Results are written as JSON for trend tracking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    query: str
    expected: object
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class BenchmarkRunResult:
    benchmark: str
    total: int
    correct: int
    accuracy: float
    latency_s: float
    details: list[dict[str, object]] = field(default_factory=list)


class Benchmark(Protocol):
    name: str

    def examples(self) -> Iterable[BenchmarkExample]: ...


def run_benchmark(
    benchmark: Benchmark,
    solver: Callable[[str], object],
    grader: Callable[[object, object], bool],
    output_path: Path | None = None,
) -> BenchmarkRunResult:
    """Run a benchmark over all examples and record correctness + latency."""
    start = time.perf_counter()
    total = 0
    correct = 0
    details: list[dict[str, object]] = []
    for ex in benchmark.examples():
        total += 1
        try:
            got = solver(ex.query)
            ok = bool(grader(got, ex.expected))
        except Exception as exc:  # pragma: no cover - defensive
            got = f"<error: {exc}>"
            ok = False
        if ok:
            correct += 1
        details.append(
            {
                "id": ex.id,
                "expected": str(ex.expected),
                "got": str(got),
                "correct": ok,
            }
        )

    latency = time.perf_counter() - start
    accuracy = correct / total if total else 0.0
    result = BenchmarkRunResult(
        benchmark=benchmark.name,
        total=total,
        correct=correct,
        accuracy=accuracy,
        latency_s=latency,
        details=details,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    return result
