"""Shared pytest fixtures."""

import pytest

from ebrm_system.intent import RuleBasedClassifier
from ebrm_system.verifiers import (
    ExecVerifier,
    RegexVerifier,
    SymPyVerifier,
    VerifierChain,
)
from ebrm_system.voting import SelfConsistencyVoter


@pytest.fixture
def sympy_verifier() -> SymPyVerifier:
    return SymPyVerifier()


@pytest.fixture
def exec_verifier() -> ExecVerifier:
    return ExecVerifier(timeout_s=3.0)


@pytest.fixture
def regex_verifier() -> RegexVerifier:
    return RegexVerifier()


@pytest.fixture
def chain(
    sympy_verifier: SymPyVerifier,
    regex_verifier: RegexVerifier,
) -> VerifierChain:
    return VerifierChain([regex_verifier, sympy_verifier])


@pytest.fixture
def classifier() -> RuleBasedClassifier:
    return RuleBasedClassifier()


@pytest.fixture
def voter_exact() -> SelfConsistencyVoter:
    return SelfConsistencyVoter(numerical=False)


@pytest.fixture
def voter_numerical() -> SelfConsistencyVoter:
    return SelfConsistencyVoter(numerical=True, tolerance=0.01)
