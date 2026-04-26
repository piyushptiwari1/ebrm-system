"""Tests for the verify-plan CLI subcommand (DRI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ebrm_system.cli import _load_diagram_from_json, app

runner = CliRunner()


def _write(tmp_path: Path, name: str, payload: object) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_load_diagram_from_json(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "diagram.json",
        {
            "morphisms": [
                {"name": "u", "src": "x", "dst": "y", "op": "upper"},
                {"name": "s", "src": "y", "dst": "z", "op": "strip"},
            ]
        },
    )
    d = _load_diagram_from_json(p)
    assert [m.name for m in d.morphisms] == ["u", "s"]
    assert d.compose(["u", "s"], "  hi  ") == "HI"


def test_load_diagram_unsupported_op_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "diagram.json",
        {"morphisms": [{"name": "x", "src": "a", "dst": "b", "op": "exec_arbitrary"}]},
    )
    with pytest.raises(ValueError, match="unsupported op"):
        _load_diagram_from_json(p)


def test_verify_plan_pass(tmp_path: Path) -> None:
    diagram = _write(
        tmp_path,
        "diagram.json",
        {
            "morphisms": [
                {"name": "strip", "src": "raw", "dst": "trim", "op": "strip"},
                {"name": "lower", "src": "trim", "dst": "norm", "op": "lower"},
                {"name": "norm", "src": "raw", "dst": "norm", "op": "lower"},
            ]
        },
    )
    candidate = _write(
        tmp_path,
        "candidate.json",
        {"initial": "HELLO", "paths": [["strip", "lower"], ["norm"]]},
    )
    result = runner.invoke(app, ["verify-plan", str(diagram), str(candidate)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.stdout
    assert "commutes" in result.stdout


def test_verify_plan_fail(tmp_path: Path) -> None:
    diagram = _write(
        tmp_path,
        "diagram.json",
        {
            "morphisms": [
                {"name": "up", "src": "x", "dst": "y", "op": "upper"},
                {"name": "low", "src": "x", "dst": "y", "op": "lower"},
            ]
        },
    )
    candidate = _write(
        tmp_path,
        "candidate.json",
        {"initial": "Hello", "paths": [["up"], ["low"]]},
    )
    result = runner.invoke(app, ["verify-plan", str(diagram), str(candidate)])
    assert result.exit_code == 0
    assert "FAIL" in result.stdout
