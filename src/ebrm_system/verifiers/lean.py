"""Lean 4 theorem-prover verifier.

Wraps the ``lean`` binary as a ``Verifier`` so EBRM's ``math`` intent lane
can hard-check candidate proofs. Follows the 2025/26 SOTA pattern set by
DeepSeek-Prover-V2, Leanabell-Prover-V2 and BFS-Prover-V2: the LM proposes
a Lean tactic block, Lean disposes (or accepts).

Design choices:

    * Subprocess-based — works with any Lean 4 / mathlib4 install without
      embedding native bindings.
    * Graceful degradation — if ``lean`` is not on PATH the verifier returns
      an honest "not installed" rejection rather than crashing the chain.
    * Header-aware — accepts an optional ``header`` (e.g. mathlib imports +
      open namespaces) prepended to every candidate; this is what every Lean
      benchmark (MiniF2F, ProofNet) does.
    * Timeout — Lean elaboration can hang on adversarial input. Default
      30 s, configurable.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ebrm_system.verifiers.base import VerificationResult

DEFAULT_HEADER = "import Mathlib\n\n"
DEFAULT_TIMEOUT_S = 30.0


class LeanVerifier:
    """Verifier that elaborates a candidate Lean 4 file via the ``lean`` CLI."""

    name = "lean"

    def __init__(
        self,
        lean_bin: str = "lean",
        header: str = DEFAULT_HEADER,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.lean_bin = lean_bin
        self.header = header
        self.timeout_s = timeout_s

    def _resolve_binary(self) -> str | None:
        return shutil.which(self.lean_bin)

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        del context  # Lean header is provided via constructor; context unused for now.
        if not isinstance(candidate, str):
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="LeanVerifier expects a string candidate",
                evidence={"got_type": type(candidate).__name__},
            )
        binary = self._resolve_binary()
        if binary is None:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"lean binary {self.lean_bin!r} not found on PATH",
                evidence={"installed": False},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidate.lean"
            path.write_text(self.header + candidate, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [binary, str(path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    verifier=self.name,
                    verified=False,
                    confidence=0.0,
                    reason=f"lean elaboration timed out after {self.timeout_s}s",
                    evidence={"timeout_s": self.timeout_s},
                )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        # Lean returns 0 and empty stdout/stderr on a clean elaboration.
        if proc.returncode == 0 and not stdout and not stderr:
            return VerificationResult(
                verifier=self.name,
                verified=True,
                confidence=1.0,
                reason="lean elaborated without errors",
                evidence={"returncode": 0},
            )
        return VerificationResult(
            verifier=self.name,
            verified=False,
            confidence=0.0,
            reason="lean reported errors",
            evidence={
                "returncode": proc.returncode,
                "stdout": stdout[:2000],
                "stderr": stderr[:2000],
            },
        )
