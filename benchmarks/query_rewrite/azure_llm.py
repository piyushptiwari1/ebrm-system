"""Azure OpenAI multi-query rewriter with disk caching."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

_REWRITE_SYSTEM = (
    "You rewrite a single user question into multiple alternative search "
    "queries that, taken together, will recall every relevant past chat "
    "turn from a long conversation history.\n"
    "Goals: cover paraphrases, decompose multi-part questions, and "
    "surface the underlying entities/topics so semantic+lexical retrieval "
    "can match turns that use different wording than the question.\n"
    "Rules:\n"
    "  - Generate exactly 3 alternatives. Do NOT repeat the original.\n"
    "  - Each alternative must be a self-contained search query "
    "(declarative or interrogative), not a chain-of-thought.\n"
    "  - For 'how many / how often / total' questions, produce sub-queries "
    "that target each likely instance type or time window separately.\n"
    "  - For 'across all sessions' / multi-session questions, produce "
    "sub-queries that target distinct facets (events, preferences, "
    "decisions, actions taken).\n"
    "  - Preserve named entities, numbers, and dates verbatim.\n"
    'Reply with VALID JSON only: {"queries": ["q1", "q2", "q3"]} '
    "— no prose, no markdown, no explanations."
)


def _digest(deployment: str, payload: str) -> str:
    return hashlib.sha256(f"{deployment}\x00{payload}".encode()).hexdigest()


def _build_user_prompt(question: str, question_type: str) -> str:
    return (
        f"question_type: {question_type}\n"
        f"question: {question}\n\n"
        'Return JSON: {"queries": ["...", "...", "..."]}'
    )


@dataclass
class AzureOpenAIQueryRewriter:
    """LLM-based query rewriter using Azure OpenAI chat completion.

    Returns ``[original_question, rewrite_1, rewrite_2, rewrite_3]`` on
    success, ``[original_question]`` on any backend failure.
    """

    deployment: str | None = None
    n_alternatives: int = 3
    max_tokens: int = 300
    temperature: float = 0.0
    max_retries: int = 4
    cache_dir: Path | None = None
    _client: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AzureOpenAIQueryRewriter requires `openai`. "
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
        return f"azure-query-rewriter(d={self.deployment},n={self.n_alternatives})"

    def _cache_path(self, payload: str) -> Path | None:
        if self.cache_dir is None:
            return None
        assert self.deployment is not None
        return self.cache_dir / f"{_digest(self.deployment, payload)}.json"

    def _call_api(self, user_prompt: str) -> list[str]:
        assert self._client is not None
        for attempt in range(self.max_retries):
            try:
                rsp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.deployment,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _REWRITE_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = (rsp.choices[0].message.content or "").strip()
                obj = json.loads(content) if content else {}
                queries = obj.get("queries", [])
                if not isinstance(queries, list):
                    return []
                clean: list[str] = []
                for q in queries:
                    if isinstance(q, str):
                        s = q.strip()
                        if s and s not in clean:
                            clean.append(s)
                return clean[: self.n_alternatives]
            except Exception:
                if attempt == self.max_retries - 1:
                    return []
                time.sleep(min(2**attempt, 30))
        return []

    def rewrite(self, question: str, question_type: str) -> list[str]:
        question = (question or "").strip()
        if not question:
            return []
        user_prompt = _build_user_prompt(question, question_type)
        cache_path = self._cache_path(user_prompt)
        alternatives: list[str]
        if cache_path is not None and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if not (
                    isinstance(cached, list)
                    and all(isinstance(s, str) for s in cached)
                ):
                    raise ValueError("bad cache")
                alternatives = cached
            except Exception:
                alternatives = self._call_api(user_prompt)
                cache_path.write_text(
                    json.dumps(alternatives, ensure_ascii=False), encoding="utf-8"
                )
        else:
            alternatives = self._call_api(user_prompt)
            if cache_path is not None:
                cache_path.write_text(
                    json.dumps(alternatives, ensure_ascii=False), encoding="utf-8"
                )
        # Always lead with the original question, dedupe alternatives.
        out: list[str] = [question]
        for q in alternatives:
            if q and q not in out:
                out.append(q)
        return out


__all__ = ["AzureOpenAIQueryRewriter"]
