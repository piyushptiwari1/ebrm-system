"""Tests for the GSM8K verifier-guided benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from benchmarks.gsm8k_verifier import (
    GSM8KVerifierBench,
    _majority_vote,
    parse_numeric_answer,
)


@dataclass
class _Ex:
    id: str
    query: str
    expected: float


# ---------------------------------------------------------- parse_numeric_answer


def test_parse_hash_format() -> None:
    assert parse_numeric_answer("Steps...\n#### 42") == 42.0


def test_parse_phrase_format() -> None:
    assert parse_numeric_answer("Reasoning... The answer is 7.") == 7.0


def test_parse_with_dollar_and_commas() -> None:
    assert parse_numeric_answer("The final answer is $1,234.5") == 1234.5


def test_parse_negative() -> None:
    assert parse_numeric_answer("Answer: -3") == -3.0


def test_parse_fallback_last_number() -> None:
    assert parse_numeric_answer("She has 3 apples and 5 pears, total is 8") == 8.0


def test_parse_returns_none_when_no_number() -> None:
    assert parse_numeric_answer("I cannot solve this") is None


# ---------------------------------------------------------- _majority_vote


def test_majority_vote_basic() -> None:
    assert _majority_vote([1.0, 2.0, 1.0, 1.0]) == 1.0


def test_majority_vote_breaks_tie_by_first_seen() -> None:
    assert _majority_vote([2.0, 3.0, 2.0, 3.0]) == 2.0


def test_majority_vote_skips_none() -> None:
    assert _majority_vote([None, 5.0, 5.0, None]) == 5.0


def test_majority_vote_all_none() -> None:
    assert _majority_vote([None, None]) is None


# ---------------------------------------------------------- GSM8KVerifierBench


def test_bench_without_scorer_falls_back_to_single() -> None:
    """No scorer → ebrm column equals single column."""

    def gen(_q: str, n: int) -> list[str]:
        return ["The answer is 1." for _ in range(n)]

    bench = GSM8KVerifierBench(generator=gen, scorer=None, n_candidates=4)
    result = bench.run(examples=[_Ex("a", "Q?", 1.0), _Ex("b", "Q?", 2.0)])
    assert result.total == 2
    assert result.single_correct == 1
    assert result.ebrm_correct == 1  # mirrors single when scorer is None


def test_bench_majority_beats_single_when_outliers_present() -> None:
    seq = iter(
        [
            # First example: noisy first sample, two correct, one wrong.
            ["The answer is 99.", "The answer is 5.", "The answer is 5.", "The answer is 7."],
        ]
    )

    def gen(_q: str, _n: int) -> list[str]:
        return next(seq)

    bench = GSM8KVerifierBench(generator=gen, scorer=None, n_candidates=4)
    result = bench.run(examples=[_Ex("a", "Q?", 5.0)])
    assert result.single_correct == 0
    assert result.majority_correct == 1


def test_bench_uses_scorer_to_pick_candidate() -> None:
    """Scorer with a fixed energy preference picks the right candidate."""

    class _StubScorer:
        def select_best(self, _q: str, candidates: list[str]):
            from ebrm_system.verifiers.ebrm_scorer import EBRMSelection

            # Prefer the one that contains "answer is 5".
            for i, c in enumerate(candidates):
                if "answer is 5" in c:
                    return EBRMSelection(
                        index=i, candidate=c, energy=-1.0, all_energies=(0.0,) * len(candidates)
                    )
            return EBRMSelection(
                index=0,
                candidate=candidates[0],
                energy=0.0,
                all_energies=(0.0,) * len(candidates),
            )

    def gen(_q: str, _n: int) -> list[str]:
        return ["The answer is 99.", "The answer is 5.", "The answer is 7."]

    bench = GSM8KVerifierBench(generator=gen, scorer=_StubScorer(), n_candidates=3)
    result = bench.run(examples=[_Ex("a", "Q?", 5.0)])
    assert result.ebrm_correct == 1
    assert result.single_correct == 0


def test_bench_result_summary_and_accuracies() -> None:
    def gen(_q: str, _n: int) -> list[str]:
        return ["The answer is 5."] * 3

    bench = GSM8KVerifierBench(generator=gen, scorer=None, n_candidates=3)
    result = bench.run(examples=[_Ex("a", "Q?", 5.0), _Ex("b", "Q?", 6.0), _Ex("c", "Q?", 5.0)])
    assert result.total == 3
    assert result.single_acc == pytest.approx(2 / 3)
    assert result.majority_acc == pytest.approx(2 / 3)
    summary = result.summary()
    assert "single=" in summary and "majority@3=" in summary and "ebrm@3=" in summary


def test_bench_limit_truncates_iterable() -> None:
    def gen(_q: str, _n: int) -> list[str]:
        return ["The answer is 1."]

    bench = GSM8KVerifierBench(generator=gen, scorer=None, n_candidates=1)
    result = bench.run(examples=[_Ex(f"x{i}", "Q?", 1.0) for i in range(10)], limit=3)
    assert result.total == 3
