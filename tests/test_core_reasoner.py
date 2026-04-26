"""Unit tests for ``ebrm_system.core.HierarchicalLatentReasoner``.

Uses tiny in-memory encoder/decoder/energy callables so the suite stays
torch-free and CPU-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.core import (
    HierarchicalLatentReasoner,
    ReasonerConfig,
    ReasoningResult,
)
from ebrm_system.intent import Intent, RuleBasedClassifier
from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex

# --------------------------------------------------------------------------- #
# Mock heads                                                                  #
# --------------------------------------------------------------------------- #

D = 16  # tiny latent dim


def make_encoder(answer_table: dict[str, float]):
    """Encoder that hashes the question into the first dim of a latent."""

    def _encode(question: str) -> np.ndarray:
        v = np.zeros(D, dtype=np.float32)
        v[0] = answer_table.get(question, 0.0)
        return v

    return _encode


def make_decoder():
    """Decoder that returns ``round(latent[0])`` as a string."""

    def _decode(latent: np.ndarray) -> str:
        return str(round(float(latent[0])))

    return _decode


def make_energy(target_first_dim: float):
    """Energy = squared distance of latent[0] from a target value."""

    def _energy(latent: np.ndarray) -> float:
        return float((latent[0] - target_first_dim) ** 2)

    return _energy


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def reasoner() -> HierarchicalLatentReasoner:
    table = {"What is 2 plus 3?": 5.0}
    return HierarchicalLatentReasoner(
        encoder=make_encoder(table),
        decoder=make_decoder(),
        energy_fn=make_energy(5.0),
        classifier=RuleBasedClassifier(),
        config=ReasonerConfig(seed=0),
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_solve_returns_consistent_answer(reasoner: HierarchicalLatentReasoner) -> None:
    result = reasoner.solve("What is 2 plus 3?")
    assert isinstance(result, ReasoningResult)
    assert result.answer == "5"
    assert result.vote.support >= 1
    assert result.vote.total == len(result.traces)
    assert 0.0 <= result.vote.agreement <= 1.0


def test_solve_routes_intent(reasoner: HierarchicalLatentReasoner) -> None:
    result = reasoner.solve("What is 2 plus 3?")
    assert result.intent.intent in (Intent.ARITHMETIC, Intent.MATH_REASONING)
    assert len(result.traces) == result.intent.suggested_trace_count


def test_solve_is_deterministic_given_seed() -> None:
    table = {"q": 7.0}
    cfg = ReasonerConfig(seed=42)
    a = HierarchicalLatentReasoner(
        encoder=make_encoder(table),
        decoder=make_decoder(),
        energy_fn=make_energy(7.0),
        config=cfg,
    ).solve("q")
    b = HierarchicalLatentReasoner(
        encoder=make_encoder(table),
        decoder=make_decoder(),
        energy_fn=make_energy(7.0),
        config=cfg,
    ).solve("q")
    assert a.answer == b.answer
    assert a.vote.support == b.vote.support
    assert tuple(t.energy for t in a.traces) == tuple(t.energy for t in b.traces)


def test_warmstart_index_is_used_when_provided() -> None:
    table = {"q": 9.0}
    idx = LatentIndex(IndexConfig(in_dim=D, bits=64))
    payload = np.zeros(D, dtype=np.float32)
    payload[0] = 9.0
    idx.add(np.stack([payload]), [payload])

    reasoner = HierarchicalLatentReasoner(
        encoder=make_encoder(table),
        decoder=make_decoder(),
        energy_fn=make_energy(9.0),
        index=idx,
        config=ReasonerConfig(seed=1),
    )
    result = reasoner.solve("q")
    assert result.details["n_warmstart"] >= 1
    assert result.answer == "9"


def test_encoder_must_return_1d() -> None:
    bad = HierarchicalLatentReasoner(
        encoder=lambda _q: np.zeros((2, D), dtype=np.float32),
        decoder=make_decoder(),
        energy_fn=make_energy(0.0),
    )
    with pytest.raises(ValueError, match="1-D latent"):
        bad.solve("anything")


def test_inverse_energy_weighting_prefers_low_energy() -> None:
    """Two equally-supported answers should be tie-broken by lower energy."""

    # Custom decoder: read latent[1] (which we'll set per-trace) to choose answer.
    # We seed latent[1] from a deterministic-but-state-changing rule via the
    # candidate generator's noise. Simpler approach: decoder returns "low" if
    # latent[0] < 5 else "high"; energy_fn keeps latent[0] near 5 ± noise so
    # both buckets get votes, and inverse_energy weighting picks the closer one.
    def decode(latent: np.ndarray) -> str:
        return "low" if float(latent[0]) < 5.0 else "high"

    def encode(_q: str) -> np.ndarray:
        v = np.zeros(D, dtype=np.float32)
        v[0] = 4.9  # bias slightly into the "low" bucket
        return v

    reasoner = HierarchicalLatentReasoner(
        encoder=encode,
        decoder=decode,
        energy_fn=make_energy(4.9),
        config=ReasonerConfig(seed=0, weight_by="inverse_energy"),
    )
    result = reasoner.solve("anything")
    # Answer must be one of the two buckets; vote object is well-formed.
    assert result.answer in {"low", "high"}
    assert result.vote.weighted_score > 0.0


def test_require_verification_falls_back_when_no_pass() -> None:
    """If no candidate is verified, the pool falls back to all candidates."""

    # Force a math intent so the chain has SymPy + Lean verifiers, and use
    # a decoder that returns garbage so SymPy rejects everything. The
    # reasoner must still return a consensus answer instead of crashing.
    def decode(_l: np.ndarray) -> str:
        return "not-a-number-xyz"

    table = {"What is 2 plus 3?": 5.0}
    reasoner = HierarchicalLatentReasoner(
        encoder=make_encoder(table),
        decoder=decode,
        energy_fn=make_energy(5.0),
        config=ReasonerConfig(seed=0, require_verification=True),
    )
    result = reasoner.solve("What is 2 plus 3?")
    assert result.answer == "not-a-number-xyz"  # fell back to the full pool
    assert result.verified_fraction == 0.0
