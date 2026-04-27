"""Azure OpenAI embedder with on-disk caching, batching, and retry.

Reads credentials from the standard env vars (set in ``~/.ebrm.env``):

- ``AZURE_OPENAI_ENDPOINT``
- ``AZURE_OPENAI_API_KEY``
- ``AZURE_API_VERSION``
- ``AZURE_EMBEDDING_MODEL_NAME`` (deployment name, e.g. ``text-embedding-3-small-2``)

Caching is essential because embedding the LongMemEval ``S`` split touches
~50k unique strings. The cache is keyed by ``sha256(model + text)`` and
stored as ``.npy`` files under ``cache_dir``.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from openai import AzureOpenAI


def _digest(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class AzureOpenAIEmbedder:
    """Embed texts with an Azure OpenAI deployment, with disk cache + retry."""

    def __init__(
        self,
        *,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 100,
        max_retries: int = 6,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AzureOpenAIEmbedder requires `openai`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        self._deployment = deployment or os.environ["AZURE_EMBEDDING_MODEL_NAME"]
        self._endpoint = endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]
        self._api_key = api_key or os.environ["AZURE_OPENAI_API_KEY"]
        self._api_version = api_version or os.environ["AZURE_API_VERSION"]
        self._client: AzureOpenAI = AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )
        self.name = f"azure-{self._deployment}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Lazy-detect dim by embedding a single token (cached after first call).
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
        return self._cache_dir / f"{_digest(self._deployment, text)}.npy"

    def _call_api(self, texts: list[str]) -> np.ndarray:
        for attempt in range(self._max_retries):
            try:
                resp = self._client.embeddings.create(model=self._deployment, input=texts)
                vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
                # Azure ada-3 / text-embedding-3-* are L2-normalised already,
                # but we re-normalise defensively.
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                return vecs / norms
            except Exception as exc:  # pragma: no cover
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 30))
                _ = exc
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            assert True
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
