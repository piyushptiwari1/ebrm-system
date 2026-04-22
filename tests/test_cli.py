"""Tests for the CLI surface (smoke tests via Typer's CliRunner)."""

from __future__ import annotations

from typer.testing import CliRunner

from ebrm_system.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "ebrm-system" in result.stdout


def test_classify_arithmetic() -> None:
    result = runner.invoke(app, ["classify", "2 + 2"])
    assert result.exit_code == 0
    assert "arithmetic" in result.stdout.lower()


def test_verify_sympy_match() -> None:
    result = runner.invoke(app, ["verify", "4", "4"])
    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_verify_sympy_mismatch() -> None:
    result = runner.invoke(app, ["verify", "4", "5"])
    assert result.exit_code == 0
    assert "FAIL" in result.stdout
