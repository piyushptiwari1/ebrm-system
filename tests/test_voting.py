"""Tests for SelfConsistencyVoter."""

from __future__ import annotations

import pytest

from ebrm_system.voting import Candidate, SelfConsistencyVoter


def test_exact_majority(voter_exact: SelfConsistencyVoter) -> None:
    candidates = [
        Candidate(answer="A"),
        Candidate(answer="A"),
        Candidate(answer="B"),
    ]
    result = voter_exact.vote(candidates)
    assert result.answer == "A"
    assert result.support == 2
    assert result.total == 3
    assert result.agreement == pytest.approx(2 / 3)
    assert result.runner_up == "B"
    assert result.runner_up_support == 1


def test_numerical_bucketing() -> None:
    voter = SelfConsistencyVoter(numerical=True, tolerance=0.1)
    candidates = [
        Candidate(answer=3.14),
        Candidate(answer=3.15),
        Candidate(answer=2.71),
    ]
    result = voter.vote(candidates)
    # 3.14 and 3.15 should share a bucket at tolerance=0.1
    assert result.support == 2


def test_confidence_weighted() -> None:
    voter = SelfConsistencyVoter(weight_by="confidence")
    candidates = [
        Candidate(answer="A", confidence=0.1),
        Candidate(answer="A", confidence=0.1),
        Candidate(answer="B", confidence=0.9),
    ]
    result = voter.vote(candidates)
    # B has higher total weight despite lower count
    assert result.answer == "B"


def test_inverse_energy_weighted() -> None:
    voter = SelfConsistencyVoter(weight_by="inverse_energy")
    candidates = [
        Candidate(answer="A", energy=5.0),  # high energy = low weight
        Candidate(answer="A", energy=5.0),
        Candidate(answer="B", energy=-5.0),  # low energy = high weight
    ]
    result = voter.vote(candidates)
    assert result.answer == "B"


def test_unanimous() -> None:
    voter = SelfConsistencyVoter()
    candidates = [Candidate(answer="X") for _ in range(5)]
    result = voter.vote(candidates)
    assert result.answer == "X"
    assert result.agreement == 1.0
    assert result.runner_up is None


def test_empty_candidates_raises() -> None:
    voter = SelfConsistencyVoter()
    with pytest.raises(ValueError):
        voter.vote([])


def test_invalid_weight_by() -> None:
    with pytest.raises(ValueError):
        SelfConsistencyVoter(weight_by="bogus")


def test_candidate_validates_confidence() -> None:
    with pytest.raises(ValueError):
        Candidate(answer="x", confidence=1.5)


def test_mixed_types_exact() -> None:
    voter = SelfConsistencyVoter(numerical=False)
    candidates = [
        Candidate(answer=1),
        Candidate(answer="1"),
        Candidate(answer=1),
    ]
    result = voter.vote(candidates)
    assert result.answer == 1
    assert result.support == 2
