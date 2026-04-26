"""Tests for the DRI commutativity verifier."""

from __future__ import annotations

import json

import numpy as np

from ebrm_system.verifiers.dri import (
    Diagram,
    DRIVerifier,
    ExactMorphism,
    VectorMorphism,
    commutes,
)


def _build_data_pipeline_diagram() -> Diagram:
    d = Diagram()
    d.add(ExactMorphism("clean", "raw", "cleaned", lambda x: x.strip().lower()))
    d.add(ExactMorphism("tokenize", "cleaned", "tokens", lambda x: x.split()))
    d.add(ExactMorphism("e2e", "raw", "tokens", lambda x: x.strip().lower().split()))
    return d


def test_commutes_exact_true() -> None:
    d = _build_data_pipeline_diagram()
    ok, outputs = commutes(d, [["clean", "tokenize"], ["e2e"]], "  Hello World  ")
    assert ok
    assert outputs[0] == outputs[1] == ["hello", "world"]


def test_commutes_exact_false() -> None:
    d = Diagram()
    d.add(ExactMorphism("up", "x", "y", str.upper))
    d.add(ExactMorphism("low", "x", "y", str.lower))
    ok, _ = commutes(d, [["up"], ["low"]], "Hello")
    assert not ok


def test_commutes_vector_with_threshold() -> None:
    d = Diagram()
    d.add(VectorMorphism("a", "x", "y", lambda v: v * 2.0))
    d.add(VectorMorphism("b", "x", "y", lambda v: v * 2.0 + 1e-4))
    ok, _ = commutes(d, [["a"], ["b"]], np.ones(8, dtype=np.float32), cosine_threshold=0.99)
    assert ok


def test_diagram_chain_validation() -> None:
    d = Diagram()
    d.add(ExactMorphism("f", "a", "b", lambda x: x))
    d.add(ExactMorphism("g", "c", "d", lambda x: x))
    import pytest

    with pytest.raises(ValueError, match="src"):
        d.compose(["f", "g"], "x")


def test_dri_verifier_happy_path() -> None:
    d = _build_data_pipeline_diagram()
    v = DRIVerifier()
    candidate = json.dumps(
        {"initial": "  Hello World  ", "paths": [["clean", "tokenize"], ["e2e"]]}
    )
    result = v.check(candidate, context={"diagram": d})
    assert result.verified
    assert result.confidence == 1.0


def test_dri_verifier_missing_diagram() -> None:
    v = DRIVerifier()
    result = v.check("{}")
    assert not result.verified
    assert "diagram" in result.reason


def test_dri_verifier_bad_json() -> None:
    v = DRIVerifier()
    result = v.check("not json", context={"diagram": Diagram()})
    assert not result.verified


def test_dri_verifier_unknown_morphism() -> None:
    v = DRIVerifier()
    candidate = json.dumps({"initial": "x", "paths": [["nope"], ["nada"]]})
    result = v.check(candidate, context={"diagram": Diagram()})
    assert not result.verified
    assert "composition failed" in result.reason
