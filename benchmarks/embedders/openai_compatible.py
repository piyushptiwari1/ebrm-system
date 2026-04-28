"""OpenAI-compatible embedder for any provider exposing the OpenAI API.

Works with: OpenAI, Ollama, vLLM, llama.cpp server, OpenRouter, Together,
Groq, Anyscale, LM Studio, Mistral, DeepInfra, and any other server that
speaks ``POST /v1/embeddings`` in OpenAI shape.

For Azure OpenAI specifically, prefer :class:`AzureOpenAIEmbedder` —
it's the SOTA-validated path and uses Azure's distinct auth model.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np


def _digest(model: str, base: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{base}\x00{text}".encode()).hexdigest()


class OpenAICompatibleEmbedder:
    """Embed texts with any OpenAI-compatible endpoint, with disk cache + retry.

    Parameters
    ----------
    model:
        The embedding model name, e.g. ``"text-embedding-3-small"`` for
        OpenAI, ``"nomic-embed-text"`` for Ollama, ``"BAAI/bge-m3"`` for
        a vLLM-hosted model.
    base_url:
        OpenAI-compatible endpoint. Examples:
        ``"https://api.openai.com/v1"`` (default),
        ``"http://localhost:11434/v1"`` (Ollama),
        ``"http://localhost:8000/v1"`` (vLLM / llama.cpp),
        ``"https://openrouter.ai/api/v1"``.
    api_key:
        API key. Falls back to ``OPENAI_API_KEY``. Pass ``"not-needed"``
        for local servers that don't require auth.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 100,
        max_retries: int = 6,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAICompatibleEmbedder requires `openai`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=timeout,
        )
        # Stable provider tag for cache namespacing — strip scheme + path.
        host = self._base_url.split("//", 1)[-1].split("/", 1)[0]
        self.name = f"oai-{host}-{model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            v = self._call_api(["x"])
            self._dim = int(v.shape[1])
        return self._dim

    def _cache_path(self, text: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{_digest(self._model, self._base_url, text)}.npy"

    def _call_api(self, texts: list[str]) -> np.ndarray:
        for attempt in range(self._max_retries):
            try:
                resp = self._client.embeddings.create(model=self._model, input=texts)
                vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                return vecs / norms
            except Exception:  # pragma: no cover
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 30))
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        to_fetch_idx: list[int] = []
        to_fetch_text: list[str] = []
        for i, t in enumerate(texts):
            cp = self._cache_path(t)
            if cp is not None and cp.exists():
                results[i] = np.load(cp)
            else:
                to_fetch_idx.append(i)
                to_fetch_text.append(t)

        for start in range(0, len(to_fetch_text), self._batch_size):
            batch_texts = to_fetch_text[start : start + self._batch_size]
            batch_idx = to_fetch_idx[start : start + self._batch_size]
            vecs = self._call_api(batch_texts)
            for j, idx in enumerate(batch_idx):
                results[idx] = vecs[j]
                cp = self._cache_path(texts[idx])
                if cp is not None:
                    np.save(cp, vecs[j])

        out = np.stack([r for r in results if r is not None], axis=0)  # type: ignore[misc]
        if self._dim is None:
            self._dim = int(out.shape[1])
        return out.astype(np.float32, copy=False)


__all__ = ["OpenAICompatibleEmbedder"]
