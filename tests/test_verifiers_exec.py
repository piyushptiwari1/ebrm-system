"""Tests for ExecVerifier (sandboxed subprocess)."""

from __future__ import annotations

from ebrm_system.verifiers import ExecVerifier


def test_stdout_match(exec_verifier: ExecVerifier) -> None:
    result = exec_verifier.check("print(2 + 2)", {"expected_stdout": "4"})
    assert result.verified is True


def test_stdout_mismatch(exec_verifier: ExecVerifier) -> None:
    result = exec_verifier.check("print(2 + 2)", {"expected_stdout": "5"})
    assert result.verified is False


def test_non_zero_exit_rejected(exec_verifier: ExecVerifier) -> None:
    result = exec_verifier.check("raise SystemExit(1)", {"expected_stdout": ""})
    assert result.verified is False


def test_syntax_error_rejected(exec_verifier: ExecVerifier) -> None:
    result = exec_verifier.check("def :", {"expected_stdout": ""})
    assert result.verified is False


def test_timeout_enforced() -> None:
    quick = ExecVerifier(timeout_s=0.5)
    result = quick.check("import time; time.sleep(5)", {"expected_stdout": ""})
    assert result.verified is False
    assert "timeout" in (result.reason or "")


def test_non_string_candidate(exec_verifier: ExecVerifier) -> None:
    result = exec_verifier.check(42, {"expected_stdout": "42"})
    assert result.verified is False


def test_json_output_match(exec_verifier: ExecVerifier) -> None:
    result = exec_verifier.check(
        "import json; print(json.dumps({'a': 1, 'b': [2, 3]}))",
        {"expected_json": {"a": 1, "b": [2, 3]}},
    )
    assert result.verified is True


def test_json_output_mismatch(exec_verifier: ExecVerifier) -> None:
    result = exec_verifier.check("print('{\"a\": 1}')", {"expected_json": {"a": 2}})
    assert result.verified is False
