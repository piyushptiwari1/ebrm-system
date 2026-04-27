"""Tests for v0.17 embedders and official-dataset loader."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _ensure_fake_openai() -> None:
    """Inject a stub ``openai`` module so judge/embedder/reader can import.

    The real ``openai`` package is replaced by a minimal stand-in; tests that
    need to mock its behaviour use ``unittest.mock.patch`` against the stub.
    Idempotent.
    """
    if "openai" in sys.modules and getattr(sys.modules["openai"], "_ebrm_stub", False):
        return
    stub = types.ModuleType("openai")
    stub.AzureOpenAI = MagicMock()  # type: ignore[attr-defined]
    stub._ebrm_stub = True  # type: ignore[attr-defined]
    sys.modules["openai"] = stub


_ensure_fake_openai()

from benchmarks.datasets import (  # noqa: E402
    ALL_QUESTION_TYPES_OFFICIAL,
    OfficialEpisode,
    OfficialTurn,
    episodes_iter_question_types,
    load_longmemeval_official,
)
from benchmarks.embedders import Embedder, HashEmbedder  # noqa: E402
from benchmarks.judges.azure_llm import (  # noqa: E402
    AzureOpenAIJudge,
    JudgeVerdict,
    is_abstention_response,
)

# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------


class TestHashEmbedder:
    def test_protocol_compliance(self) -> None:
        emb = HashEmbedder(dim=16)
        assert isinstance(emb, Embedder)
        assert emb.dim == 16
        assert emb.name == "hash-projection"

    def test_unit_norm_and_shape(self) -> None:
        emb = HashEmbedder(dim=32)
        out = emb.embed(["foo", "bar", "baz"])
        assert out.shape == (3, 32)
        assert out.dtype == np.float32
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_deterministic(self) -> None:
        emb = HashEmbedder(dim=8, seed=7)
        a = emb.embed(["hello"])
        b = emb.embed(["hello"])
        np.testing.assert_array_equal(a, b)

    def test_seed_changes_output(self) -> None:
        a = HashEmbedder(dim=8, seed=1).embed(["hello"])
        b = HashEmbedder(dim=8, seed=2).embed(["hello"])
        assert not np.array_equal(a, b)

    def test_invalid_dim(self) -> None:
        with pytest.raises(ValueError, match="dim must be positive"):
            HashEmbedder(dim=0)


# ---------------------------------------------------------------------------
# Official LongMemEval loader
# ---------------------------------------------------------------------------


def _fake_episode(qid: str = "q1", qtype: str = "single-session-user") -> dict:
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": "Q?",
        "answer": "A.",
        "question_date": "2024/01/02 (Tue) 09:00",
        "haystack_session_ids": ["s0", "s1"],
        "haystack_dates": ["2024/01/01 (Mon) 09:00", "2024/01/01 (Mon) 10:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "u1", "has_answer": True},
                {"role": "assistant", "content": "a1"},
            ],
            [{"role": "user", "content": "u2"}],
        ],
        "answer_session_ids": ["s0"],
    }


class TestLoadOfficial:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "lme.json"
        path.write_text(
            json.dumps([_fake_episode("q1"), _fake_episode("q2_abs")]), encoding="utf-8"
        )
        eps = load_longmemeval_official(path)
        assert len(eps) == 2
        assert isinstance(eps[0], OfficialEpisode)
        assert eps[0].question_id == "q1"
        assert eps[0].is_abstention is False
        assert eps[1].is_abstention is True
        assert len(eps[0].turns) == 3
        first = eps[0].turns[0]
        assert isinstance(first, OfficialTurn)
        assert first.session_id == "s0"
        assert first.has_answer is True

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_longmemeval_official(tmp_path / "nope.json")

    def test_unknown_question_type_rejected(self, tmp_path: Path) -> None:
        bad = _fake_episode()
        bad["question_type"] = "made-up"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([bad]), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown question_type"):
            load_longmemeval_official(path)

    def test_length_mismatch_rejected(self, tmp_path: Path) -> None:
        bad = _fake_episode()
        bad["haystack_dates"] = ["only one"]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([bad]), encoding="utf-8")
        with pytest.raises(ValueError, match="length mismatch"):
            load_longmemeval_official(path)

    def test_question_type_counter(self) -> None:
        eps = [
            OfficialEpisode(
                question_id="x",
                question_type="multi-session",
                question="q",
                answer="a",
                question_date="d",
                turns=(),
                answer_session_ids=(),
                is_abstention=False,
            ),
            OfficialEpisode(
                question_id="y",
                question_type="multi-session",
                question="q",
                answer="a",
                question_date="d",
                turns=(),
                answer_session_ids=(),
                is_abstention=False,
            ),
            OfficialEpisode(
                question_id="z",
                question_type="temporal-reasoning",
                question="q",
                answer="a",
                question_date="d",
                turns=(),
                answer_session_ids=(),
                is_abstention=False,
            ),
        ]
        assert episodes_iter_question_types(eps) == {
            "multi-session": 2,
            "temporal-reasoning": 1,
        }

    def test_all_question_types_constant(self) -> None:
        assert "single-session-preference" in ALL_QUESTION_TYPES_OFFICIAL
        assert len(ALL_QUESTION_TYPES_OFFICIAL) == 6


# ---------------------------------------------------------------------------
# Abstention detector + Azure judge (mocked)
# ---------------------------------------------------------------------------


class TestIsAbstentionResponse:
    @pytest.mark.parametrize(
        "text",
        [
            "I don't know",
            "I do not know.",
            "I don't recall.",
            "I have no record of that.",
            "Not sure.",
            "Cannot determine the answer.",
            "There is no mention of that.",
            "I'm unable to find this information.",
            "",
        ],
    )
    def test_positive_detection(self, text: str) -> None:
        assert is_abstention_response(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is 42.",
            "Paris is the capital of France.",
            "On March 15th, the user said hello.",
        ],
    )
    def test_negative_detection(self, text: str) -> None:
        assert is_abstention_response(text) is False


class TestAzureOpenAIJudge:
    @pytest.fixture
    def env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.example.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
        monkeypatch.setenv("AZURE_API_VERSION", "2025-01-01-preview")
        monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")

    def test_abstention_short_circuits_no_api_call(self, env: None, tmp_path: Path) -> None:
        with patch("openai.AzureOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            judge = AzureOpenAIJudge(cache_dir=str(tmp_path))
            v = judge.judge(
                question="?",
                question_type="single-session-user",
                gold="N/A",
                pred="I don't know",
                is_abstention=True,
            )
            assert v.correct is True
            assert v.raw == "abstain-detector"
            mock_cls.return_value.chat.completions.create.assert_not_called()

    def test_correct_grading_via_mock(self, env: None, tmp_path: Path) -> None:
        with patch("openai.AzureOpenAI") as mock_cls:
            inst = MagicMock()
            inst.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="1"))]
            )
            mock_cls.return_value = inst
            judge = AzureOpenAIJudge(cache_dir=str(tmp_path))
            v = judge.judge(
                question="What is the capital of France?",
                question_type="single-session-user",
                gold="Paris",
                pred="The capital is Paris.",
                is_abstention=False,
            )
            assert v.correct is True
            inst.chat.completions.create.assert_called_once()

    def test_judge_cache_round_trip(self, env: None, tmp_path: Path) -> None:
        with patch("openai.AzureOpenAI") as mock_cls:
            inst = MagicMock()
            inst.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="0"))]
            )
            mock_cls.return_value = inst
            judge = AzureOpenAIJudge(cache_dir=str(tmp_path))
            kw: dict[str, object] = {
                "question": "?",
                "question_type": "multi-session",
                "gold": "g",
                "pred": "p",
                "is_abstention": False,
            }
            v1 = judge.judge(**kw)  # type: ignore[arg-type]
            v2 = judge.judge(**kw)  # type: ignore[arg-type]
            assert v1.correct is False
            assert v2.correct is False
            # Second call must come from disk cache, not API.
            assert inst.chat.completions.create.call_count == 1

    def test_abstention_mismatch_marks_wrong(self, env: None, tmp_path: Path) -> None:
        # Non-abstention question but model said "I don't know" → wrong without API.
        with patch("openai.AzureOpenAI") as mock_cls:
            inst = MagicMock()
            mock_cls.return_value = inst
            judge = AzureOpenAIJudge(cache_dir=str(tmp_path))
            v = judge.judge(
                question="What is X?",
                question_type="single-session-user",
                gold="42",
                pred="I don't know.",
                is_abstention=False,
            )
            assert v.correct is False
            inst.chat.completions.create.assert_not_called()


def test_judge_verdict_repr() -> None:
    v = JudgeVerdict(correct=True, raw="1")
    assert v.correct is True
    assert v.raw == "1"
