"""Tests for the v0.19 LLM extraction package."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.extraction import (
    ExtractedMemory,
    MemoryExtractor,
    augment_episode_with_memories,
    memories_to_episode,
)


def _ep(turns: list[OfficialTurn]) -> OfficialEpisode:
    return OfficialEpisode(
        question_id="q1",
        question_type="single-session-user",
        question="what?",
        answer="x",
        question_date="2025/01/01 (Wed) 09:00",
        turns=tuple(turns),
        answer_session_ids=("s0",),
        is_abstention=False,
    )


def _turn(s_idx: int, t_idx: int, role: str, content: str) -> OfficialTurn:
    return OfficialTurn(
        session_id=f"s{s_idx}",
        session_idx=s_idx,
        turn_idx=t_idx,
        role=role,
        content=content,
        session_date=f"2025/01/0{s_idx + 1} (Wed) 09:00",
        has_answer=False,
    )


def test_extractor_protocol_is_runtime_checkable() -> None:
    class _Stub:
        name = "stub"

        def extract(self, episode: OfficialEpisode) -> list[ExtractedMemory]:
            return []

    assert isinstance(_Stub(), MemoryExtractor)


def test_memories_to_episode_replaces_turns_preserves_question() -> None:
    ep = _ep([_turn(0, 0, "user", "I love sushi.")])
    memories = [
        ExtractedMemory(
            text="User loves sushi.",
            session_id="s0",
            session_idx=0,
            session_date="2025/01/01 (Wed) 09:00",
            role="user",
        ),
        ExtractedMemory(
            text="User mentioned sushi on 2025/01/01.",
            session_id="s0",
            session_idx=0,
            session_date="2025/01/01 (Wed) 09:00",
            role="memory",
        ),
    ]
    new_ep = memories_to_episode(ep, memories)
    assert new_ep.question == ep.question
    assert new_ep.answer == ep.answer
    assert len(new_ep.turns) == 2
    assert new_ep.turns[0].content == "User loves sushi."
    assert new_ep.turns[0].role == "user"
    assert new_ep.turns[1].turn_idx == 1
    assert new_ep.turns[0].session_id == "s0"


def test_memories_to_episode_empty() -> None:
    ep = _ep([_turn(0, 0, "user", "Hi")])
    new_ep = memories_to_episode(ep, [])
    assert new_ep.turns == ()
    assert new_ep.question == ep.question


def test_augment_episode_with_memories_appends_with_namespaced_session_id() -> None:
    ep = _ep([_turn(0, 0, "user", "I saw a movie."), _turn(0, 1, "assistant", "Cool.")])
    memories = [
        ExtractedMemory(
            text="User saw a movie.",
            session_id="s0",
            session_idx=0,
            session_date="2025/01/01 (Wed) 09:00",
            role="memory",
        )
    ]
    aug = augment_episode_with_memories(ep, memories)
    assert len(aug.turns) == 3
    assert aug.turns[0].session_id == "s0"
    assert aug.turns[1].session_id == "s0"
    assert aug.turns[2].session_id == "s0::mem"
    assert aug.turns[2].content == "User saw a movie."
    assert aug.turns[2].turn_idx == 2


def test_augment_episode_with_memories_empty_passthrough() -> None:
    ep = _ep([_turn(0, 0, "user", "Hi")])
    aug = augment_episode_with_memories(ep, [])
    assert aug.turns == ep.turns


# --- Azure extractor with mocked OpenAI client ----------------------------------


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = SimpleNamespace(content=content)


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = payloads
        self.calls = 0

    def create(self, **_: object) -> _FakeResp:
        idx = self.calls
        self.calls += 1
        return _FakeResp(self._payloads[idx])


class _FakeChat:
    def __init__(self, payloads: list[str]) -> None:
        self.completions = _FakeCompletions(payloads)


class _FakeAzure:
    def __init__(self, payloads: list[str]) -> None:
        self.chat = _FakeChat(payloads)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, payloads: list[str]) -> _FakeAzure:
    fake = _FakeAzure(payloads)

    def _factory(**_: object) -> _FakeAzure:
        return fake

    fake_module = SimpleNamespace(AzureOpenAI=_factory)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "gpt-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_API_VERSION", "v")
    return fake


def test_azure_extractor_parses_json_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.dumps(
        {
            "memories": [
                {"text": "User likes sushi.", "role": "user"},
                {"text": "Assistant recommended ramen.", "role": "assistant"},
                {"text": "", "role": "user"},  # filtered
            ]
        }
    )
    fake = _install_fake_openai(monkeypatch, [payload])
    from benchmarks.extraction.azure_llm import AzureLLMExtractor

    ex = AzureLLMExtractor(cache_dir=tmp_path)
    ep = _ep([_turn(0, 0, "user", "I like sushi"), _turn(0, 1, "assistant", "Try ramen")])
    mems = ex.extract(ep)
    assert len(mems) == 2
    assert mems[0].text == "User likes sushi."
    assert mems[0].role == "user"
    assert mems[1].role == "assistant"
    assert ex.name == "azure-extractor-gpt-test"
    assert fake.chat.completions.calls == 1
    # Second call must hit the cache
    mems2 = ex.extract(ep)
    assert mems2 == mems
    assert fake.chat.completions.calls == 1


def test_azure_extractor_skips_unknown_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.dumps({"memories": [{"text": "Fact.", "role": "robot"}]})
    _install_fake_openai(monkeypatch, [payload])
    from benchmarks.extraction.azure_llm import AzureLLMExtractor

    ex = AzureLLMExtractor(cache_dir=tmp_path)
    ep = _ep([_turn(0, 0, "user", "...")])
    [mem] = ex.extract(ep)
    assert mem.role == "memory"


def test_azure_extractor_handles_bad_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_openai(monkeypatch, ["not json"])
    from benchmarks.extraction.azure_llm import AzureLLMExtractor

    ex = AzureLLMExtractor(cache_dir=tmp_path, max_retries=1)
    ep = _ep([_turn(0, 0, "user", "...")])
    mems = ex.extract(ep)
    assert mems == []


def test_azure_extractor_groups_by_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payloads = [
        json.dumps({"memories": [{"text": "S0 fact.", "role": "user"}]}),
        json.dumps({"memories": [{"text": "S1 fact.", "role": "assistant"}]}),
    ]
    fake = _install_fake_openai(monkeypatch, payloads)
    from benchmarks.extraction.azure_llm import AzureLLMExtractor

    ex = AzureLLMExtractor(cache_dir=tmp_path)
    ep = _ep([_turn(0, 0, "user", "a"), _turn(1, 0, "assistant", "b")])
    mems = ex.extract(ep)
    assert [m.session_idx for m in mems] == [0, 1]
    assert fake.chat.completions.calls == 2


def test_azure_extractor_empty_episode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _install_fake_openai(monkeypatch, [])
    from benchmarks.extraction.azure_llm import AzureLLMExtractor

    ex = AzureLLMExtractor(cache_dir=tmp_path)
    ep = _ep([])
    assert ex.extract(ep) == []
    assert fake.chat.completions.calls == 0
