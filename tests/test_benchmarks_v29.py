"""Tests for v0.29 — public ``ebrm_system.longmem`` API + self-consistency reader.

v0.29 ships two changes:

1. Public facade: ``ebrm_system.longmem`` exposes ``LongMemPipeline`` so
   ``pip install ebrm-system`` users can run the LongMemEval-tuned stack
   without copy-pasting the benchmark scripts.
2. Self-consistency reader: ``AzureOpenAIReader(n_samples=N)`` samples
   the reader N times (single API call with ``n=N``) at
   ``sc_temperature`` and majority-votes on the final answer.

Tests stub the OpenAI client so no live API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn

# ---------------------------------------------------------------------------
# Self-consistency primitives
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_lowercases_and_strips_punct(self):
        from benchmarks.reader.azure_llm import _normalize_answer

        assert _normalize_answer("Hello, World!") == "hello world"

    def test_collapses_idk_variants(self):
        from benchmarks.reader.azure_llm import _normalize_answer

        for v in ("I don't know.", "I dont know", "i don't know", "I DON'T KNOW!!"):
            assert _normalize_answer(v) == "i don't know"


class TestMajorityVote:
    def test_single_input_returned_as_is(self):
        from benchmarks.reader.azure_llm import _majority_vote

        assert _majority_vote(["foo"]) == "foo"

    def test_majority_wins(self):
        from benchmarks.reader.azure_llm import _majority_vote

        assert _majority_vote(["3", "3", "5"]) == "3"

    def test_punctuation_insensitive(self):
        from benchmarks.reader.azure_llm import _majority_vote

        # "3" and "3." normalize to the same key
        out = _majority_vote(["3.", "3", "five"])
        assert out in ("3.", "3")

    def test_idk_loses_tie_to_substantive_answer(self):
        from benchmarks.reader.azure_llm import _majority_vote

        # 1 IDK, 1 "blue" → blue wins
        assert _majority_vote(["I don't know.", "blue"]) == "blue"

    def test_all_idk_returns_idk(self):
        from benchmarks.reader.azure_llm import _majority_vote

        assert _majority_vote(["I don't know.", "I dont know", "i don't know"]) == "I don't know."

    def test_empty_returns_empty(self):
        from benchmarks.reader.azure_llm import _majority_vote

        assert _majority_vote([]) == ""


# ---------------------------------------------------------------------------
# AzureOpenAIReader self-consistency end-to-end
# ---------------------------------------------------------------------------


pytest.importorskip("openai", reason="AzureOpenAIReader requires openai SDK")


@pytest.fixture
def reader_env(monkeypatch):
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-02-15-preview")


def _episode(question: str, question_type: str = "multi-session") -> OfficialEpisode:
    turn = OfficialTurn(
        session_id="s0",
        session_idx=0,
        turn_idx=0,
        role="user",
        content="hello",
        session_date="2024-01-01",
        has_answer=False,
    )
    return OfficialEpisode(
        question_id="q",
        question_type=question_type,
        question=question,
        answer="x",
        question_date="2024-02-01",
        turns=(turn,),
        answer_session_ids=(),
        is_abstention=False,
    )


def _stub_multi_choice(monkeypatch, choices: list[str]):
    """Stub AzureOpenAI to return ``choices`` as N message contents."""
    captured: dict = {"n": None, "temperature": None, "calls": 0}

    def _create(**kw):
        captured["n"] = kw.get("n")
        captured["temperature"] = kw.get("temperature")
        captured["calls"] += 1
        rsp = MagicMock()
        rsp.choices = []
        for text in choices:
            msg = MagicMock()
            msg.content = text
            ch = MagicMock()
            ch.message = msg
            rsp.choices.append(ch)
        return rsp

    class _Stub:
        def __init__(self, *a, **kw):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = _create

    import openai

    monkeypatch.setattr(openai, "AzureOpenAI", _Stub)
    return captured


def test_n_samples_default_is_one_call_one_choice(monkeypatch, reader_env):
    captured = _stub_multi_choice(monkeypatch, ["the answer"])
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader()
    out = r.read(_episode("anything?"), list(_episode("anything?").turns))
    assert out == "the answer"
    assert captured["n"] == 1
    assert captured["temperature"] == 0.0


def test_n_samples_three_majority_vote(monkeypatch, reader_env):
    captured = _stub_multi_choice(monkeypatch, ["3", "3", "5"])
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(n_samples=3, sc_temperature=0.4)
    out = r.read(_episode("how many?"), list(_episode("how many?").turns))
    assert out == "3"
    assert captured["n"] == 3
    assert captured["temperature"] == 0.4


def test_n_samples_with_aggregation_cot_votes_on_extracted_answer(monkeypatch, reader_env):
    # All three CoT samples → ITEMS/TOTAL/ANSWER blocks with answers 4, 4, 7.
    samples = [
        "ITEMS:\n1. a\n2. b\n3. c\n4. d\nTOTAL: 4\nANSWER: 4",
        "ITEMS:\n1. a\n2. b\n3. c\n4. d\nTOTAL: 4\nANSWER: 4",
        "ITEMS:\n1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n7. g\nTOTAL: 7\nANSWER: 7",
    ]
    captured = _stub_multi_choice(monkeypatch, samples)
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    r = AzureOpenAIReader(aggregation_cot=True, n_samples=3)
    out = r.read(_episode("how many?", "multi-session"), [])
    assert out == "4"
    assert captured["n"] == 3


def test_n_samples_zero_or_negative_raises(reader_env):
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    with pytest.raises(ValueError, match="n_samples"):
        AzureOpenAIReader(n_samples=0)


def test_reader_name_includes_sc_marker(reader_env):
    from benchmarks.reader.azure_llm import AzureOpenAIReader

    assert "sc" not in AzureOpenAIReader().name
    assert AzureOpenAIReader(n_samples=3).name.endswith("-sc3")


# ---------------------------------------------------------------------------
# Public facade: ebrm_system.longmem
# ---------------------------------------------------------------------------


class TestPublicFacadeImports:
    def test_top_level_imports(self):
        from ebrm_system.longmem import (
            LongMemAnswer,
            LongMemPipeline,
            LongMemSession,
            LongMemTurn,
        )

        assert LongMemPipeline.__name__ == "LongMemPipeline"
        assert LongMemTurn.__name__ == "LongMemTurn"
        assert LongMemSession.__name__ == "LongMemSession"
        assert LongMemAnswer.__name__ == "LongMemAnswer"

    def test_all_exports(self):
        import ebrm_system.longmem as m

        assert set(m.__all__) == {
            "LongMemAnswer",
            "LongMemPipeline",
            "LongMemSession",
            "LongMemTurn",
        }


class TestLongMemDataclasses:
    def test_turn_is_frozen(self):
        from ebrm_system.longmem import LongMemTurn

        t = LongMemTurn(role="user", content="hi")
        with pytest.raises(AttributeError):
            t.role = "assistant"  # type: ignore[misc]

    def test_session_from_dicts(self):
        from ebrm_system.longmem import LongMemSession

        s = LongMemSession.from_dicts(
            "s1",
            "2024-03-12 09:30",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        )
        assert s.session_id == "s1"
        assert len(s.turns) == 2
        assert s.turns[0].role == "user"
        assert s.turns[1].content == "hello"


class TestLongMemPipelineWithStubs:
    """End-to-end happy path with stub retriever + stub reader (no live API)."""

    def _stub_pipeline(self):
        from ebrm_system.longmem import LongMemPipeline

        class _StubScored:
            def __init__(self, turn):
                self.turn = turn
                self.score = 1.0

        class _StubRetriever:
            name = "stub-retriever"

            def __init__(self):
                self.last_top_k = None

            def retrieve(self, episode, *, top_k):
                self.last_top_k = top_k
                # Return all turns wrapped as ScoredTurn-likes.
                return [_StubScored(t) for t in episode.turns]

        class _StubReader:
            def __init__(self):
                self.last_turns = None

            def read(self, episode, retrieved_turns):
                self.last_turns = retrieved_turns
                return f"answered {len(retrieved_turns)} turns"

        retr = _StubRetriever()
        rdr = _StubReader()
        pipe = LongMemPipeline(retriever=retr, reader=rdr, top_k=4, per_type_top_k=False)
        return pipe, retr, rdr

    def test_empty_haystack_returns_idk(self):
        pipe, _, _ = self._stub_pipeline()
        out = pipe.ask("anything?", today="2024-04-01")
        assert out.answer == "I don't know."
        assert out.retrieved_session_ids == ()
        assert out.n_retrieved == 0

    def test_add_session_and_ask(self):
        pipe, retr, rdr = self._stub_pipeline()
        pipe.add_session(
            "s1",
            "2024-03-12 09:30",
            [
                {"role": "user", "content": "I bought a Trek bike"},
                {"role": "assistant", "content": "Nice choice!"},
            ],
        )
        pipe.add_session(
            "s2",
            "2024-03-20 14:00",
            [{"role": "user", "content": "I rode it to work"}],
        )
        out = pipe.ask("What bike did I buy?", today="2024-04-01")
        assert out.answer == "answered 3 turns"
        assert out.n_retrieved == 3
        # Distinct session ids preserved in retrieval order.
        assert out.retrieved_session_ids == ("s1", "s2")
        # top_k threaded through (per_type_top_k disabled).
        assert retr.last_top_k == 4
        # Reader saw exactly the right number of turns.
        assert len(rdr.last_turns) == 3

    def test_ask_top_k_override(self):
        pipe, retr, _ = self._stub_pipeline()
        pipe.add_session("s1", "2024-01-01", [{"role": "user", "content": "x"}])
        pipe.ask("?", today="2024-01-02", top_k=99)
        assert retr.last_top_k == 99

    def test_reset_clears_haystack(self):
        pipe, _, _ = self._stub_pipeline()
        pipe.add_session("s1", "2024-01-01", [{"role": "user", "content": "x"}])
        assert len(pipe.sessions) == 1
        pipe.reset()
        assert len(pipe.sessions) == 0
