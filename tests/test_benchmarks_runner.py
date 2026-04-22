"""Tests for the benchmark runner (including Trackio integration)."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterable
from pathlib import Path

from benchmarks.runner import BenchmarkExample, run_benchmark


class _FakeBenchmark:
    name = "fake"

    def examples(self) -> Iterable[BenchmarkExample]:
        yield BenchmarkExample(id="1", query="2+2", expected=4)
        yield BenchmarkExample(id="2", query="3+3", expected=6)
        yield BenchmarkExample(id="3", query="0", expected=1)  # wrong on purpose


def _solver(q: str) -> int:
    return sum(int(x) for x in q.split("+")) if "+" in q else int(q)


def _grader(got: object, expected: object) -> bool:
    return got == expected


def test_runner_counts_correct_and_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    result = run_benchmark(_FakeBenchmark(), _solver, _grader, output_path=out)
    assert result.total == 3
    assert result.correct == 2
    assert result.accuracy == 2 / 3
    payload = json.loads(out.read_text())
    assert payload["benchmark"] == "fake"
    assert len(payload["details"]) == 3


def test_runner_catches_solver_errors() -> None:
    def bad_solver(_: str) -> int:
        raise RuntimeError("boom")

    result = run_benchmark(_FakeBenchmark(), bad_solver, _grader)
    assert result.correct == 0
    assert all(not d["correct"] for d in result.details)


def test_runner_trackio_integration(monkeypatch) -> None:
    """Runner should call trackio.init/log/finish when project is set."""
    calls: dict[str, list[object]] = {"init": [], "log": [], "finish": []}

    fake = types.SimpleNamespace(
        init=lambda **kw: calls["init"].append(kw),
        log=lambda metrics: calls["log"].append(metrics),
        finish=lambda: calls["finish"].append(True),
    )
    monkeypatch.setitem(sys.modules, "trackio", fake)

    result = run_benchmark(
        _FakeBenchmark(),
        _solver,
        _grader,
        trackio_project="ebrm-system-test",
        trackio_config={"model": "unit-test"},
    )

    assert result.total == 3
    assert len(calls["init"]) == 1
    assert calls["init"][0]["project"] == "ebrm-system-test"
    assert calls["init"][0]["config"]["benchmark"] == "fake"
    assert calls["init"][0]["config"]["model"] == "unit-test"
    assert len(calls["log"]) == 3
    assert calls["log"][0]["step"] == 1
    assert "running_accuracy" in calls["log"][0]
    assert len(calls["finish"]) == 1


def test_runner_without_trackio_when_project_none() -> None:
    """No project -> no trackio import attempted."""
    result = run_benchmark(_FakeBenchmark(), _solver, _grader)
    assert result.total == 3


def test_runner_silently_skips_when_trackio_missing(monkeypatch) -> None:
    """Missing trackio should not break the run."""
    # Block trackio import
    monkeypatch.setitem(sys.modules, "trackio", None)
    result = run_benchmark(
        _FakeBenchmark(),
        _solver,
        _grader,
        trackio_project="should-not-break",
    )
    assert result.total == 3
