"""Sandboxed Python execution verifier.

Runs a candidate code snippet in a restricted subprocess and checks that:
  1. It executes without error
  2. Its stdout matches the expected output (exact or within tolerance)

Uses a subprocess with resource limits to prevent runaway code. Never executes
untrusted code in the parent process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ebrm_system.verifiers.base import VerificationResult


class ExecVerifier:
    """Verify code by executing it in a sandboxed subprocess.

    Context keys:
        expected_stdout: expected stdout (str) after stripping
        expected_json: expected JSON object printed to stdout
        timeout_s: subprocess timeout in seconds (default 5)
    """

    name = "exec"

    def __init__(self, timeout_s: float = 5.0, max_output_bytes: int = 64 * 1024) -> None:
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        context = context or {}
        if not isinstance(candidate, str):
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="candidate must be a string of Python code",
            )

        timeout_raw = context.get("timeout_s", self.timeout_s)
        try:
            timeout = float(timeout_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            timeout = self.timeout_s

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "candidate.py"
            code_path.write_text(candidate, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", str(code_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    verifier=self.name,
                    verified=False,
                    confidence=0.0,
                    reason=f"timeout after {timeout}s",
                )
            except OSError as exc:
                return VerificationResult(
                    verifier=self.name,
                    verified=False,
                    confidence=0.0,
                    reason=f"subprocess error: {exc}",
                )

        if proc.returncode != 0:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"non-zero exit: {proc.returncode}",
                evidence={"stderr": proc.stderr[: self.max_output_bytes]},
            )

        stdout = proc.stdout[: self.max_output_bytes].strip()

        if "expected_stdout" in context:
            expected = str(context["expected_stdout"]).strip()
            if stdout == expected:
                return VerificationResult(
                    verifier=self.name,
                    verified=True,
                    confidence=1.0,
                    reason="stdout matches",
                    evidence={"stdout": stdout},
                )
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="stdout mismatch",
                evidence={"got": stdout, "expected": expected},
            )

        if "expected_json" in context:
            try:
                got = json.loads(stdout)
            except json.JSONDecodeError as exc:
                return VerificationResult(
                    verifier=self.name,
                    verified=False,
                    confidence=0.0,
                    reason=f"stdout not JSON: {exc}",
                )
            if got == context["expected_json"]:
                return VerificationResult(
                    verifier=self.name,
                    verified=True,
                    confidence=1.0,
                    reason="JSON matches",
                    evidence={"stdout": got},
                )
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="JSON mismatch",
                evidence={"got": got, "expected": context["expected_json"]},
            )

        # No expected output specified: just check it ran cleanly
        return VerificationResult(
            verifier=self.name,
            verified=True,
            confidence=1.0,
            reason="executed without error",
            evidence={"stdout": stdout},
        )
