"""Local SentenceTransformer embedder (BGE / GTE / E5 / Qwen).

Defaults to ``BAAI/bge-large-en-v1.5`` (1024-dim, strong on retrieval) since
the user authorised "utilize the maximum". Smaller alternatives:

- ``BAAI/bge-small-en-v1.5`` — 384-dim, CPU-friendly
- ``BAAI/bge-base-en-v1.5`` — 768-dim
- ``Alibaba-NLP/gte-Qwen2-7B-instruct`` — official LongMemEval ``flat-gte`` baseline (needs GPU)

Imported lazily so the harness still loads when ``sentence-transformers``
is not installed (it is an optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Embed texts with a HuggingFace SentenceTransformer model.

    Output rows are L2-normalised by ``encode(..., normalize_embeddings=True)``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        *,
        device: str | None = None,
        batch_size: int = 64,
        query_prompt: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "SentenceTransformerEmbedder requires `sentence-transformers`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=device)
        self.name = f"st-{model_name.replace('/', '__')}"
        self._batch_size = batch_size
        self._query_prompt = query_prompt
        # SentenceTransformer exposes the embedding dimensionality.
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prompts = [self._query_prompt + t if self._query_prompt else t for t in texts]
        arr = self._model.encode(
            prompts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)
