"""Unit tests for ``ebrm_system.reward.prm_data``."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ebrm_system.core import (
    HierarchicalLatentReasoner,
    ReasonerConfig,
    ReasoningResult,
)
from ebrm_system.intent import Intent, IntentPrediction
from ebrm_system.reward.prm_data import make_records, write_jsonl
from ebrm_system.verifiers.base import VerificationResult
from ebrm_system.voting.voter import VoteResult

D = 8


def _encoder(_: str) -> np.ndarray:
    v = np.zeros(D, dtype=np.float32)
    v[0] = 5.0
    return v


def _decoder(latent: np.ndarray) -> str:
    return str(round(float(latent[0])))


def _energy(latent: np.ndarray) -> float:
    return float((latent[0] - 5.0) ** 2)


@pytest.fixture
def reasoner() -> HierarchicalLatentReasoner:
    return HierarchicalLatentReasoner(
        encoder=_encoder,
        decoder=_decoder,
        energy_fn=_energy,
        config=ReasonerConfig(seed=42),
    )


class TestMakeRecords:
    def test_one_record_per_trace(self, reasoner: HierarchicalLatentReasoner) -> None:
        result = reasoner.solve("compute 2+3")
        records = make_records("compute 2+3", result)
        assert len(records) == len(result.traces)

    def test_strong_label_matches_voted_answer(self, reasoner: HierarchicalLatentReasoner) -> None:
        result = reasoner.solve("compute 2+3")
        records = make_records("compute 2+3", result)
        for r in records:
            assert r.strong_label == str(result.answer)

    def test_agreement_flag_consistent(self, reasoner: HierarchicalLatentReasoner) -> None:
        result = reasoner.solve("compute 2+3")
        records = make_records("compute 2+3", result)
        for r in records:
            assert r.agreement == (r.candidate_answer == r.strong_label)

    def test_carries_question_and_intent(self, reasoner: HierarchicalLatentReasoner) -> None:
        result = reasoner.solve("compute 2+3")
        records = make_records("compute 2+3", result)
        for r in records:
            assert r.question == "compute 2+3"
            assert r.intent  # non-empty
            assert 0.0 <= r.difficulty <= 1.0


class TestWriteJsonl:
    def test_writes_one_line_per_record(
        self, tmp_path, reasoner: HierarchicalLatentReasoner
    ) -> None:
        result = reasoner.solve("compute 2+3")
        records = make_records("compute 2+3", result)
        out = tmp_path / "prm.jsonl"
        n = write_jsonl(records, out)
        assert n == len(records)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n
        for line in lines:
            obj = json.loads(line)
            assert "question" in obj
            assert "candidate_answer" in obj
            assert "strong_label" in obj
            assert "agreement" in obj

    def test_appends_across_calls(self, tmp_path, reasoner: HierarchicalLatentReasoner) -> None:
        result = reasoner.solve("compute 2+3")
        records = make_records("q", result)
        out = tmp_path / "prm.jsonl"
        write_jsonl(records, out)
        write_jsonl(records, out)
        assert len(out.read_text().splitlines()) == 2 * len(records)

    def test_creates_parent_dir(self, tmp_path, reasoner: HierarchicalLatentReasoner) -> None:
        result = reasoner.solve("compute 2+3")
        records = make_records("q", result)
        out = tmp_path / "deep" / "nested" / "prm.jsonl"
        write_jsonl(records, out)
        assert out.exists()


class TestPRMRecord:
    def test_carries_verifier_evidence(self) -> None:
        # Synthesize a minimal result with one trace to test serialization shape
        from ebrm_system.core.reasoner import TraceItem

        vr = VerificationResult(
            verifier="dri", verified=False, confidence=0.0, reason="bad morphism"
        )
        trace = TraceItem(
            latent=np.zeros(4, dtype=np.float32),
            answer="42",
            energy=0.1,
            seed=0,
            warmstart=False,
            verified=False,
            verifier_results=(vr,),
        )
        result = ReasoningResult(
            answer="43",
            intent=IntentPrediction(
                intent=Intent.MATH_REASONING,
                difficulty=0.3,
                reasoning="x",
                suggested_trace_count=1,
                suggested_langevin_steps=10,
                suggested_restarts=1,
            ),
            vote=VoteResult(
                answer="43",
                support=1,
                total=1,
                agreement=1.0,
                weighted_score=1.0,
            ),
            traces=(trace,),
            verified_fraction=0.0,
        )
        records = make_records("Q", result)
        assert len(records) == 1
        assert records[0].verifier_results[0]["verifier"] == "dri"
        assert records[0].verifier_results[0]["verified"] is False
        assert records[0].agreement is False  # "42" != "43"
