"""Benchmark harness.

Defines an abstract `Benchmark` interface; adapters live next to it (e.g. gsm8k.py).
Results are written as JSON for trend tracking; optionally streamed to Trackio
(https://github.com/gradio-app/trackio) for embeddable dashboards.
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
    *,
    trackio_project: str | None = None,
    trackio_config: dict[str, object] | None = None,
    trackio_space_id: str | None = None,
) -> BenchmarkRunResult:
    """Run a benchmark over all examples and record correctness + latency.

    If ``trackio_project`` is provided, metrics are streamed to a Trackio run
    (running accuracy, cumulative latency, per-example correctness).
    Install via ``pip install 'ebrm-system[benchmark]'`` to get trackio.
    """
    tracker = _open_tracker(
        project=trackio_project,
        config={
            "benchmark": benchmark.name,
            **(trackio_config or {}),
        },
        space_id=trackio_space_id,
    )

    start = time.perf_counter()
    total = 0
    correct = 0
    details: list[dict[str, object]] = []
    for ex in benchmark.examples():
        total += 1
        ex_start = time.perf_counter()
        try:
            got = solver(ex.query)
            ok = bool(grader(got, ex.expected))
        except Exception as exc:  # pragma: no cover - defensive
            got = f"<error: {exc}>"
            ok = False
        ex_latency = time.perf_counter() - ex_start
        if ok:
            correct += 1
        details.append(
            {
                "id": ex.id,
                "expected": str(ex.expected),
                "got": str(got),
                "correct": ok,
                "latency_s": ex_latency,
            }
        )
        if tracker is not None:
            tracker.log(
                {
                    "step": total,
                    "correct": int(ok),
                    "running_accuracy": correct / total,
                    "example_latency_s": ex_latency,
                    "cumulative_latency_s": time.perf_counter() - start,
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

    if tracker is not None:
        tracker.finish()

    return result


class _Tracker(Protocol):
    def log(self, metrics: dict[str, object]) -> None: ...
    def finish(self) -> None: ...


def _open_tracker(
    project: str | None,
    config: dict[str, object],
    space_id: str | None,
) -> _Tracker | None:
    """Best-effort Trackio opener; returns None if trackio is unavailable."""
    if project is None:
        return None
    try:
        import trackio  # type: ignore[import-not-found]
    except ImportError:
        return None

    init_kwargs: dict[str, object] = {"project": project, "config": config}
    if space_id is not None:
        init_kwargs["space_id"] = space_id
    trackio.init(**init_kwargs)

    class _TrackioAdapter:
        def log(self, metrics: dict[str, object]) -> None:
            trackio.log(metrics)

        def finish(self) -> None:
            trackio.finish()

    return _TrackioAdapter()
