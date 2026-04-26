"""Unit tests for ``ebrm_system.core.refinement``."""

from __future__ import annotations

import pytest

from ebrm_system.core.refinement import (
    RefinementConfig,
    build_refined_question,
    collect_critiques,
    should_refine,
)
from ebrm_system.verifiers.base import VerificationResult


def _vr(name: str, ok: bool, reason: str = "") -> VerificationResult:
    return VerificationResult(
        verifier=name,
        verified=ok,
        confidence=1.0 if ok else 0.0,
        reason=reason,
    )


class TestRefinementConfig:
    def test_defaults_disable_refinement(self) -> None:
        cfg = RefinementConfig()
        assert cfg.max_rounds == 0

    def test_rejects_negative_rounds(self) -> None:
        with pytest.raises(ValueError, match="max_rounds"):
            RefinementConfig(max_rounds=-1)

    def test_rejects_out_of_range_threshold(self) -> None:
        with pytest.raises(ValueError, match="trigger_threshold"):
            RefinementConfig(trigger_threshold=1.5)

    def test_rejects_zero_critiques(self) -> None:
        with pytest.raises(ValueError, match="max_critiques"):
            RefinementConfig(max_critiques=0)


class TestCollectCritiques:
    def test_skips_passes_and_empty_reasons(self) -> None:
        results = [
            [_vr("sympy", True, "ok")],
            [_vr("regex", False, "")],
            [_vr("dri", False, "diagram does not commute")],
        ]
        out = collect_critiques(results)
        assert out == ["[dri] diagram does not commute"]

    def test_dedupes_identical_reasons(self) -> None:
        results = [
            [_vr("sympy", False, "x must be positive")],
            [_vr("sympy", False, "x must be positive")],
            [_vr("sympy", False, "y must be integer")],
        ]
        out = collect_critiques(results)
        assert out == [
            "[sympy] x must be positive",
            "[sympy] y must be integer",
        ]

    def test_caps_at_max_critiques(self) -> None:
        results = [[_vr("v", False, f"reason-{i}")] for i in range(10)]
        out = collect_critiques(results, max_critiques=3)
        assert len(out) == 3
        assert out[0] == "[v] reason-0"


class TestShouldRefine:
    def test_disabled_when_max_rounds_zero(self) -> None:
        cfg = RefinementConfig(max_rounds=0)
        assert not should_refine(0.0, ["[v] r"], cfg)

    def test_disabled_when_no_critiques(self) -> None:
        cfg = RefinementConfig(max_rounds=2)
        assert not should_refine(0.0, [], cfg)

    def test_triggers_below_threshold(self) -> None:
        cfg = RefinementConfig(max_rounds=1, trigger_threshold=0.5)
        assert should_refine(0.4, ["[v] r"], cfg)
        assert not should_refine(0.6, ["[v] r"], cfg)


class TestBuildRefinedQuestion:
    def test_returns_original_when_no_critiques(self) -> None:
        cfg = RefinementConfig(max_rounds=1)
        assert build_refined_question("Q?", [], cfg) == "Q?"

    def test_includes_critiques_as_bullets(self) -> None:
        cfg = RefinementConfig(max_rounds=1)
        out = build_refined_question("What is 2+2?", ["[v] off-by-one", "[v2] sign"], cfg)
        assert "What is 2+2?" in out
        assert "- [v] off-by-one" in out
        assert "- [v2] sign" in out
        assert "Re-examine" in out

    def test_respects_max_critiques(self) -> None:
        cfg = RefinementConfig(max_rounds=1, max_critiques=2)
        out = build_refined_question("Q?", ["[a] c1", "[a] c2", "[a] c3"], cfg)
        assert "c1" in out and "c2" in out
        assert "c3" not in out
