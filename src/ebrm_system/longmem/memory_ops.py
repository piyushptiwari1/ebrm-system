"""Mem0-style memory operations for the LongMem pipeline (v0.30).

This module adds an **opt-in** structured memory layer on top of the
session-additive haystack. It is intentionally small and orthogonal:

* :class:`MemoryAction` — a frozen record of a single ADD / UPDATE /
  DELETE / NOOP decision produced by an extractor.
* :class:`MemoryStore` — a minimal Protocol any store can satisfy.
* :class:`InMemoryStore` — a dict-backed default with deterministic ids.
* :class:`MemoryExtractor` — a Protocol; the concrete extractor decides
  whether a session changes existing memories. ``LLMMemoryExtractor``
  delegates to any caller-supplied chat callable so this module stays
  free of provider-specific code.

Default :class:`~ebrm_system.longmem.LongMemPipeline` behaviour is
**unchanged** when no store is passed — this is purely additive.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ebrm_system.longmem.pipeline import LongMemSession

MemoryOp = Literal["ADD", "UPDATE", "DELETE", "NOOP"]


@dataclass(frozen=True)
class MemoryAction:
    """A single memory mutation decision."""

    op: MemoryOp
    content: str = ""
    memory_id: str | None = None  # required for UPDATE / DELETE
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op in ("UPDATE", "DELETE") and not self.memory_id:
            raise ValueError(f"{self.op} action requires memory_id")
        if self.op in ("ADD", "UPDATE") and not self.content:
            raise ValueError(f"{self.op} action requires non-empty content")


@dataclass(frozen=True)
class MemoryRecord:
    """A persisted memory entry."""

    memory_id: str
    content: str
    metadata: Mapping[str, str] = field(default_factory=dict)


class MemoryStore(Protocol):
    """Minimal CRUD surface for a memory backend."""

    def add(self, content: str, metadata: Mapping[str, str] | None = None) -> str: ...
    def update(self, memory_id: str, content: str) -> None: ...
    def delete(self, memory_id: str) -> None: ...
    def list(self) -> list[MemoryRecord]: ...


class InMemoryStore:
    """Dict-backed default :class:`MemoryStore`.

    Memory ids are deterministic SHA-1 prefixes of ``content`` (12 chars)
    so identical ADD calls collapse to a single record. Sufficient for
    in-process use; persistence is the caller's job.
    """

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]

    def add(self, content: str, metadata: Mapping[str, str] | None = None) -> str:
        memory_id = self._hash(content)
        self._records[memory_id] = MemoryRecord(
            memory_id=memory_id,
            content=content,
            metadata=dict(metadata or {}),
        )
        return memory_id

    def update(self, memory_id: str, content: str) -> None:
        if memory_id not in self._records:
            raise KeyError(memory_id)
        old = self._records[memory_id]
        self._records[memory_id] = MemoryRecord(
            memory_id=memory_id,
            content=content,
            metadata=old.metadata,
        )

    def delete(self, memory_id: str) -> None:
        self._records.pop(memory_id, None)

    def list(self) -> list[MemoryRecord]:
        return list(self._records.values())


class MemoryExtractor(Protocol):
    """Decide which memory mutations a session implies."""

    def extract(
        self,
        session: LongMemSession,
        existing: list[MemoryRecord],
    ) -> list[MemoryAction]: ...


_DEFAULT_PROMPT = """You maintain a structured long-term memory for an assistant.
Given the new conversation session and the current memories, decide which
memories to ADD, UPDATE, or DELETE. Return a JSON list. Each item must be:

  {"op": "ADD", "content": "..."}
  {"op": "UPDATE", "memory_id": "...", "content": "..."}
  {"op": "DELETE", "memory_id": "..."}
  {"op": "NOOP"}

Rules:
- Only emit UPDATE if the session contradicts or refines an existing memory.
- Only emit DELETE if the session explicitly invalidates a memory.
- Prefer NOOP over speculative ADDs.
- Output ONLY the JSON list, no prose.

CURRENT MEMORIES:
{memories}

NEW SESSION (date={date}, id={session_id}):
{turns}
"""


@dataclass
class LLMMemoryExtractor:
    """LLM-driven extractor backed by a caller-supplied chat callable.

    The ``chat`` callable receives a single user prompt string and must
    return the model's text response. This keeps the extractor free of
    Azure / OpenAI / Ollama specifics — pass any client adapter that
    matches the signature.
    """

    chat: Callable[[str], str]
    prompt_template: str = _DEFAULT_PROMPT

    def extract(
        self,
        session: LongMemSession,
        existing: list[MemoryRecord],
    ) -> list[MemoryAction]:
        memories_block = (
            "\n".join(f"- [{r.memory_id}] {r.content}" for r in existing) or "(none)"
        )
        turns_block = "\n".join(f"{t.role}: {t.content}" for t in session.turns)
        prompt = (
            self.prompt_template.replace("{memories}", memories_block)
            .replace("{date}", session.date)
            .replace("{session_id}", session.session_id)
            .replace("{turns}", turns_block)
        )
        raw = self.chat(prompt)
        return _parse_actions(raw)


def _parse_actions(raw: str) -> list[MemoryAction]:
    """Best-effort JSON parse; strip code fences if present, drop bad items."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip the first fence and any trailing fence.
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        # Drop a leading "json" language tag.
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    actions: list[MemoryAction] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        op = item.get("op")
        if op not in ("ADD", "UPDATE", "DELETE", "NOOP"):
            continue
        try:
            actions.append(
                MemoryAction(
                    op=op,
                    content=str(item.get("content", "")),
                    memory_id=item.get("memory_id"),
                    metadata={
                        str(k): str(v) for k, v in (item.get("metadata") or {}).items()
                    },
                )
            )
        except ValueError:
            continue  # skip malformed action (e.g. UPDATE without memory_id)
    return actions


def apply_actions(store: MemoryStore, actions: list[MemoryAction]) -> list[MemoryAction]:
    """Apply actions to ``store``; return the subset that succeeded."""
    applied: list[MemoryAction] = []
    for action in actions:
        try:
            if action.op == "ADD":
                store.add(action.content, action.metadata)
            elif action.op == "UPDATE":
                assert action.memory_id is not None  # guarded in __post_init__
                store.update(action.memory_id, action.content)
            elif action.op == "DELETE":
                assert action.memory_id is not None
                store.delete(action.memory_id)
            else:
                # NOOP — nothing to do.
                pass
        except (KeyError, ValueError):
            continue
        applied.append(action)
    return applied


__all__ = [
    "InMemoryStore",
    "LLMMemoryExtractor",
    "MemoryAction",
    "MemoryExtractor",
    "MemoryOp",
    "MemoryRecord",
    "MemoryStore",
    "apply_actions",
]
