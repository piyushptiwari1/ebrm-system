"""Azure OpenAI memory extractor.

Distills each session in an episode into a list of atomic, self-contained
``ExtractedMemory`` items via a single LLM call per session. Results are
cached on disk keyed by ``sha256(deployment + session_payload)`` so a full
500-episode LongMemEval run only costs the extraction pass once.

This is the ADD-only variant of the Mem0-v3 extraction step. It does not
yet do consolidation (UPDATE/DELETE) — those land in v0.20.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.extraction.base import ExtractedMemory

_EXTRACT_SYSTEM = (
    "You distill chat conversations into atomic, self-contained long-term "
    "memories. Read the supplied chat session and emit a JSON object of "
    'the exact form: {"memories": [{"text": "...", "role": "user|assistant|memory"}]}.\n'
    "Rules:\n"
    "- One fact per memory. Each memory must make sense in isolation.\n"
    "- Preserve dates, numbers, names, and units exactly as written.\n"
    "- Resolve pronouns to concrete referents inside the memory text.\n"
    "- Skip greetings, small talk, and content-free filler.\n"
    '- Use role="user" for facts about the user, role="assistant" for '
    'facts the assistant stated, role="memory" for synthesised summaries.\n'
    "- Output VALID JSON only — no prose, no markdown, no trailing commas."
)


def _digest(deployment: str, payload: str) -> str:
    return hashlib.sha256(f"{deployment}\x00{payload}".encode()).hexdigest()


def _format_session(turns: list[OfficialTurn], session_date: str) -> str:
    head = f"Session date: {session_date}\n"
    body = "\n".join(f"[{t.role}] {t.content.strip()}" for t in turns)
    return head + body


class AzureLLMExtractor:
    """Single-pass ADD-only memory extractor over Azure OpenAI."""

    def __init__(
        self,
        *,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        cache_dir: str | Path | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
        max_retries: int = 4,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AzureLLMExtractor requires `openai`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        self._deployment = deployment or os.environ["AZURE_DEPLOYMENT_NAME"]
        self._client = AzureOpenAI(
            azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version or os.environ["AZURE_API_VERSION"],
        )
        self.name = f"azure-extractor-{self._deployment}"
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, payload: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{_digest(self._deployment, payload)}.json"

    def _call_api(self, session_text: str) -> list[dict]:
        for attempt in range(self._max_retries):
            try:
                rsp = self._client.chat.completions.create(
                    model=self._deployment,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _EXTRACT_SYSTEM},
                        {"role": "user", "content": session_text},
                    ],
                )
                content = (rsp.choices[0].message.content or "").strip()
                obj = json.loads(content) if content else {}
                mems = obj.get("memories", [])
                if not isinstance(mems, list):
                    return []
                return [m for m in mems if isinstance(m, dict) and m.get("text")]
            except Exception:
                if attempt == self._max_retries - 1:
                    # On hard failure return empty memories — fall back to
                    # whatever raw turns the runner already had.
                    return []
                time.sleep(min(2**attempt, 30))
        return []

    def _extract_session(
        self,
        turns: list[OfficialTurn],
        session_id: str,
        session_idx: int,
        session_date: str,
    ) -> list[ExtractedMemory]:
        if not turns:
            return []
        payload = _format_session(turns, session_date)
        cache_path = self._cache_path(payload)
        raw_mems: list[dict]
        if cache_path is not None and cache_path.exists():
            try:
                raw_mems = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                raw_mems = self._call_api(payload)
                cache_path.write_text(json.dumps(raw_mems, ensure_ascii=False), encoding="utf-8")
        else:
            raw_mems = self._call_api(payload)
            if cache_path is not None:
                cache_path.write_text(json.dumps(raw_mems, ensure_ascii=False), encoding="utf-8")

        out: list[ExtractedMemory] = []
        for m in raw_mems:
            text = str(m.get("text", "")).strip()
            if not text:
                continue
            role = str(m.get("role", "memory")).strip() or "memory"
            if role not in {"user", "assistant", "memory"}:
                role = "memory"
            out.append(
                ExtractedMemory(
                    text=text,
                    session_id=session_id,
                    session_idx=session_idx,
                    session_date=session_date,
                    role=role,
                )
            )
        return out

    def extract(self, episode: OfficialEpisode) -> list[ExtractedMemory]:
        # Group turns by session_idx in stable order.
        by_session: dict[int, list[OfficialTurn]] = {}
        for t in episode.turns:
            by_session.setdefault(t.session_idx, []).append(t)

        memories: list[ExtractedMemory] = []
        for s_idx in sorted(by_session.keys()):
            session_turns = by_session[s_idx]
            sample = session_turns[0]
            memories.extend(
                self._extract_session(
                    session_turns,
                    session_id=sample.session_id,
                    session_idx=s_idx,
                    session_date=sample.session_date,
                )
            )
        return memories


__all__ = ["AzureLLMExtractor"]
