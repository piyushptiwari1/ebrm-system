"""Regex/schema verifier for format validation."""

from __future__ import annotations

import re

from ebrm_system.verifiers.base import VerificationResult


class RegexVerifier:
    """Verify that a string answer matches a regex pattern.

    Context keys:
        pattern: regex string
        flags: optional re flags int (default 0)
    """

    name = "regex"

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        context = context or {}
        pattern = context.get("pattern")
        if not isinstance(pattern, str):
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="no pattern in context",
            )
        if not isinstance(candidate, str):
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="candidate must be a string",
            )

        flags_raw = context.get("flags", 0)
        flags = int(flags_raw) if isinstance(flags_raw, int) else 0
        try:
            match = re.fullmatch(pattern, candidate, flags)
        except re.error as exc:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"invalid regex: {exc}",
            )
        if match is None:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="does not match pattern",
            )
        return VerificationResult(
            verifier=self.name,
            verified=True,
            confidence=1.0,
            reason="matches pattern",
            evidence={"groups": match.groups()},
        )
