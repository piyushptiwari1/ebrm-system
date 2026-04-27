"""LLM-based fusion reranker over Azure OpenAI."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.datasets.longmemeval_official import OfficialEpisode
from benchmarks.retrieval.base import Retriever, ScoredTurn

_FUSION_SYSTEM = (
    "You are a memory-retrieval reranker. You will receive a user question, "
    "the date the question was asked, and a numbered list of candidate "
    "chat-history excerpts (each tagged with its own session date). "
    "Re-rank the candidates from most useful to least useful for answering "
    "the question, considering simultaneously:\n"
    "  - semantic relevance to the question;\n"
    "  - temporal fit (recency to the question_date when the question is "
    "    time-sensitive; the most recent contradicting fact wins for "
    "    knowledge-update questions);\n"
    "  - entity overlap (people, places, dates, numbers, quoted phrases).\n"
    'Reply with a single JSON object: {"ranking": [i0, i1, ...]} where each '
    "i is a 0-based index into the candidate list, no duplicates, all "
    "candidates listed. Output VALID JSON only — no prose, no markdown."
)


def _digest(deployment: str, payload: str) -> str:
    return hashlib.sha256(f"{deployment}\x00{payload}".encode()).hexdigest()


def _format_candidates(candidates: list[ScoredTurn]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates):
        text = c.turn.content.strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "…"
        lines.append(f"[{i}] [{c.turn.session_date}] [{c.turn.role}] {text}")
    return "\n".join(lines)


def _build_user_prompt(
    question: str, question_date: str, question_type: str, candidates: list[ScoredTurn]
) -> str:
    return (
        f"question_date: {question_date}\n"
        f"question_type: {question_type}\n"
        f"question: {question}\n\n"
        f"Candidates ({len(candidates)}):\n{_format_candidates(candidates)}\n\n"
        'Return JSON: {"ranking": [...]}'
    )


@dataclass
class LLMFusionReranker:
    """Wrap a base retriever and rerank its top-N candidates via an LLM."""

    base: Retriever
    deployment: str | None = None
    candidate_k: int = 20
    max_tokens: int = 400
    temperature: float = 0.0
    max_retries: int = 4
    cache_dir: Path | None = None
    _client: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLMFusionReranker requires `openai`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        self.deployment = self.deployment or os.environ["AZURE_DEPLOYMENT_NAME"]
        self._client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_API_VERSION"],
        )
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return f"llm-fusion({self.base.name},k={self.candidate_k},d={self.deployment})"

    def _cache_path(self, payload: str) -> Path | None:
        if self.cache_dir is None:
            return None
        assert self.deployment is not None
        return self.cache_dir / f"{_digest(self.deployment, payload)}.json"

    def _call_api(self, user_prompt: str, n_candidates: int) -> list[int]:
        assert self._client is not None
        for attempt in range(self.max_retries):
            try:
                rsp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.deployment,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _FUSION_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = (rsp.choices[0].message.content or "").strip()
                obj = json.loads(content) if content else {}
                ranking = obj.get("ranking", [])
                if not isinstance(ranking, list):
                    return list(range(n_candidates))
                # Sanitize: keep only valid, unique, in-range indices.
                seen: set[int] = set()
                clean: list[int] = []
                for idx in ranking:
                    if not isinstance(idx, int):
                        continue
                    if 0 <= idx < n_candidates and idx not in seen:
                        seen.add(idx)
                        clean.append(idx)
                # Backfill any missing indices in original order.
                for j in range(n_candidates):
                    if j not in seen:
                        clean.append(j)
                return clean
            except Exception:
                if attempt == self.max_retries - 1:
                    return list(range(n_candidates))
                time.sleep(min(2**attempt, 30))
        return list(range(n_candidates))

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        k = max(top_k, self.candidate_k)
        candidates = self.base.retrieve(episode, top_k=k)
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates[:top_k]

        user_prompt = _build_user_prompt(
            episode.question, episode.question_date, episode.question_type, candidates
        )
        cache_path = self._cache_path(user_prompt)
        ranking: list[int]
        if cache_path is not None and cache_path.exists():
            try:
                ranking = json.loads(cache_path.read_text(encoding="utf-8"))
                if not (
                    isinstance(ranking, list)
                    and all(isinstance(i, int) for i in ranking)
                    and len(ranking) == len(candidates)
                ):
                    raise ValueError("bad cache")
            except Exception:
                ranking = self._call_api(user_prompt, len(candidates))
                cache_path.write_text(json.dumps(ranking), encoding="utf-8")
        else:
            ranking = self._call_api(user_prompt, len(candidates))
            if cache_path is not None:
                cache_path.write_text(json.dumps(ranking), encoding="utf-8")

        # Reorder candidates by ranking; assign descending synthetic scores.
        n = len(candidates)
        out: list[ScoredTurn] = []
        for rank, idx in enumerate(ranking):
            cand = candidates[idx]
            score = (n - rank) / n  # 1.0 → ~0
            out.append(ScoredTurn(turn=cand.turn, score=score))
        return out[:top_k]


__all__ = ["LLMFusionReranker"]
