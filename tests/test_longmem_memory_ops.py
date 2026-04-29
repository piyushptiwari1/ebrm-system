"""Tests for v0.30 Mem0-style memory operations."""

from __future__ import annotations

import json

import pytest

from ebrm_system.longmem.memory_ops import (
    InMemoryStore,
    LLMMemoryExtractor,
    MemoryAction,
    MemoryRecord,
    _parse_actions,
    apply_actions,
)
from ebrm_system.longmem.pipeline import LongMemPipeline, LongMemSession

# ---------------------------------------------------------------------------
# MemoryAction validation
# ---------------------------------------------------------------------------


def test_memory_action_add_requires_content() -> None:
    with pytest.raises(ValueError):
        MemoryAction(op="ADD", content="")


def test_memory_action_update_requires_id() -> None:
    with pytest.raises(ValueError):
        MemoryAction(op="UPDATE", content="x")


def test_memory_action_delete_requires_id() -> None:
    with pytest.raises(ValueError):
        MemoryAction(op="DELETE")


def test_memory_action_noop_is_valid() -> None:
    a = MemoryAction(op="NOOP")
    assert a.op == "NOOP"


# ---------------------------------------------------------------------------
# InMemoryStore CRUD
# ---------------------------------------------------------------------------


def test_in_memory_store_add_dedupes_identical_content() -> None:
    store = InMemoryStore()
    a = store.add("user likes tea")
    b = store.add("user likes tea")
    assert a == b
    assert len(store.list()) == 1


def test_in_memory_store_update_changes_content() -> None:
    store = InMemoryStore()
    mid = store.add("likes tea")
    store.update(mid, "likes coffee")
    [rec] = store.list()
    assert rec.content == "likes coffee"
    assert rec.memory_id == mid


def test_in_memory_store_update_unknown_raises() -> None:
    store = InMemoryStore()
    with pytest.raises(KeyError):
        store.update("does-not-exist", "x")


def test_in_memory_store_delete_idempotent() -> None:
    store = InMemoryStore()
    mid = store.add("x")
    store.delete(mid)
    store.delete(mid)  # second call must not raise
    assert store.list() == []


# ---------------------------------------------------------------------------
# _parse_actions
# ---------------------------------------------------------------------------


def test_parse_actions_plain_json_list() -> None:
    raw = json.dumps([{"op": "ADD", "content": "hello"}])
    assert _parse_actions(raw) == [MemoryAction(op="ADD", content="hello")]


def test_parse_actions_strips_code_fence() -> None:
    raw = "```json\n" + json.dumps([{"op": "NOOP"}]) + "\n```"
    assert _parse_actions(raw) == [MemoryAction(op="NOOP")]


def test_parse_actions_invalid_json_returns_empty() -> None:
    assert _parse_actions("not json at all") == []


def test_parse_actions_skips_malformed_items() -> None:
    raw = json.dumps(
        [
            {"op": "ADD", "content": "good"},
            {"op": "UPDATE"},  # missing id+content -> skipped
            {"op": "BOGUS", "content": "x"},  # bad op -> skipped
            "not a dict",
        ]
    )
    actions = _parse_actions(raw)
    assert actions == [MemoryAction(op="ADD", content="good")]


def test_parse_actions_non_list_returns_empty() -> None:
    assert _parse_actions(json.dumps({"op": "ADD", "content": "x"})) == []


# ---------------------------------------------------------------------------
# apply_actions
# ---------------------------------------------------------------------------


def test_apply_actions_add_update_delete() -> None:
    store = InMemoryStore()
    mid = store.add("old")
    actions = [
        MemoryAction(op="ADD", content="new"),
        MemoryAction(op="UPDATE", memory_id=mid, content="refined"),
        MemoryAction(op="NOOP"),
    ]
    applied = apply_actions(store, actions)
    assert len(applied) == 3
    contents = sorted(r.content for r in store.list())
    assert contents == ["new", "refined"]


def test_apply_actions_delete_unknown_is_idempotent() -> None:
    store = InMemoryStore()
    actions = [MemoryAction(op="DELETE", memory_id="ghost")]
    applied = apply_actions(store, actions)
    # InMemoryStore.delete is idempotent, so the action is reported as applied
    # and the store remains empty.
    assert applied == actions
    assert store.list() == []


def test_apply_actions_skips_update_to_missing_target() -> None:
    """UPDATE to a non-existent id raises KeyError -> action is dropped."""
    store = InMemoryStore()
    actions = [MemoryAction(op="UPDATE", memory_id="ghost", content="x")]
    applied = apply_actions(store, actions)
    assert applied == []


# ---------------------------------------------------------------------------
# LLMMemoryExtractor
# ---------------------------------------------------------------------------


def test_llm_extractor_invokes_chat_and_parses() -> None:
    captured: list[str] = []

    def fake_chat(prompt: str) -> str:
        captured.append(prompt)
        return json.dumps([{"op": "ADD", "content": "user prefers oat milk"}])

    extractor = LLMMemoryExtractor(chat=fake_chat)
    session = LongMemSession.from_dicts(
        "s1", "2026-04-29", [{"role": "user", "content": "I like oat milk"}]
    )
    actions = extractor.extract(session, existing=[])
    assert actions == [MemoryAction(op="ADD", content="user prefers oat milk")]
    assert "I like oat milk" in captured[0]
    assert "(none)" in captured[0]


def test_llm_extractor_serialises_existing_memories_in_prompt() -> None:
    captured: list[str] = []

    def fake_chat(prompt: str) -> str:
        captured.append(prompt)
        return "[]"

    extractor = LLMMemoryExtractor(chat=fake_chat)
    session = LongMemSession.from_dicts("s1", "d", [{"role": "user", "content": "hi"}])
    extractor.extract(
        session,
        existing=[MemoryRecord(memory_id="abc123", content="prior fact")],
    )
    assert "[abc123] prior fact" in captured[0]


# ---------------------------------------------------------------------------
# LongMemPipeline opt-in wiring
# ---------------------------------------------------------------------------


class _StubRetriever:
    name = "stub"

    def retrieve(self, episode, *, top_k):  # type: ignore[no-untyped-def]
        return []


class _StubReader:
    def read(self, episode, retrieved_turns):  # type: ignore[no-untyped-def]
        return "ok"


def _make_pipeline(**kwargs):  # type: ignore[no-untyped-def]
    return LongMemPipeline(retriever=_StubRetriever(), reader=_StubReader(), **kwargs)


def test_pipeline_default_does_not_touch_memory() -> None:
    pipe = _make_pipeline()
    pipe.add_session("s1", "d", [{"role": "user", "content": "hi"}])
    # Default haystack-only behaviour preserved.
    assert len(pipe.sessions) == 1
    assert pipe._last_memory_actions == []


def test_pipeline_optin_runs_extractor_and_applies_actions() -> None:
    store = InMemoryStore()
    calls: list[int] = []

    class _StubExtractor:
        def extract(self, session, existing):  # type: ignore[no-untyped-def]
            calls.append(len(existing))
            return [MemoryAction(op="ADD", content=f"fact-{session.session_id}")]

    pipe = _make_pipeline(memory_store=store, memory_extractor=_StubExtractor())
    pipe.add_session("s1", "d", [{"role": "user", "content": "x"}])
    pipe.add_session("s2", "d", [{"role": "user", "content": "y"}])

    contents = sorted(r.content for r in store.list())
    assert contents == ["fact-s1", "fact-s2"]
    # Extractor saw growing existing-memory list.
    assert calls == [0, 1]
    assert len(pipe._last_memory_actions) == 1


def test_pipeline_optin_requires_both_store_and_extractor() -> None:
    """Setting only one of the pair must NOT trigger the memory path."""
    store = InMemoryStore()
    pipe = _make_pipeline(memory_store=store, memory_extractor=None)
    pipe.add_session("s1", "d", [{"role": "user", "content": "x"}])
    assert store.list() == []
