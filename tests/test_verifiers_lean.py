"""Tests for the Lean 4 verifier (subprocess-mocked)."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from ebrm_system.verifiers.lean import LeanVerifier


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_lean_verifier_no_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ebrm_system.verifiers.lean.shutil.which", lambda _b: None)
    v = LeanVerifier()
    result = v.check("theorem t : 1 + 1 = 2 := by rfl")
    assert not result.verified
    assert "not found" in result.reason
    assert result.evidence == {"installed": False}


def test_lean_verifier_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ebrm_system.verifiers.lean.shutil.which", lambda _b: "/usr/bin/lean")

    def fake_run(*_a: Any, **_kw: Any) -> _FakeProc:
        return _FakeProc(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    v = LeanVerifier()
    result = v.check("theorem t : 1 + 1 = 2 := by rfl")
    assert result.verified
    assert result.confidence == 1.0
    assert result.evidence["returncode"] == 0


def test_lean_verifier_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ebrm_system.verifiers.lean.shutil.which", lambda _b: "/usr/bin/lean")

    def fake_run(*_a: Any, **_kw: Any) -> _FakeProc:
        return _FakeProc(returncode=1, stdout="", stderr="error: type mismatch")

    monkeypatch.setattr(subprocess, "run", fake_run)
    v = LeanVerifier()
    result = v.check("theorem t : 1 + 1 = 3 := by rfl")
    assert not result.verified
    assert result.evidence["returncode"] == 1
    assert "type mismatch" in result.evidence["stderr"]


def test_lean_verifier_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ebrm_system.verifiers.lean.shutil.which", lambda _b: "/usr/bin/lean")

    def fake_run(*_a: Any, **_kw: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="lean", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    v = LeanVerifier(timeout_s=1.0)
    result = v.check("theorem t : True := by sorry")
    assert not result.verified
    assert "timed out" in result.reason
