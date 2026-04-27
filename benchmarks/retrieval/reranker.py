"""Cross-encoder reranker — reorders an existing candidate list.

The reranker is *not* a retriever — it does not see the full haystack.
It takes the union of candidates returned by an upstream retriever
(typically RRF over BM25 + dense) and reorders them by a cross-encoder
relevance score, which is the highest-precision retrieval signal short
of running the LLM reader directly.

Default model: ``BAAI/bge-reranker-v2-m3`` — multilingual, 568M params,
state-of-the-art on MTEB reranking as of 2024-2025.
"""

from __future__ import annotations

from benchmarks.datasets.longmemeval_official import OfficialEpisode
from benchmarks.retrieval.base import Retriever, ScoredTurn


class CrossEncoderReranker:
    """Wrap a base retriever with a cross-encoder reranking pass.

    Parameters
    ----------
    base
        Upstream retriever that produces the candidate pool.
    model_name
        HuggingFace model id of a sentence-transformers CrossEncoder.
    candidate_k
        How many candidates to ask the base retriever for. The reranker
        reorders these and returns the requested ``top_k``.
    batch_size
        Cross-encoder batch size (passed to ``predict``).
    """

    def __init__(
        self,
        base: Retriever,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        candidate_k: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CrossEncoderReranker requires `sentence-transformers`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        self._base = base
        self._model_name = model_name
        self._candidate_k = candidate_k
        self._batch_size = batch_size
        # device=None lets sentence-transformers auto-pick (cuda if available).
        self._model = CrossEncoder(model_name, device=device)
        self.name = f"rerank({base.name},{model_name})"

    def retrieve(self, episode: OfficialEpisode, *, top_k: int) -> list[ScoredTurn]:
        candidates = self._base.retrieve(episode, top_k=self._candidate_k)
        if not candidates:
            return []
        pairs = [(episode.question, st.turn.content) for st in candidates]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [ScoredTurn(st.turn, float(score)) for st, score in ranked[:top_k]]


__all__ = ["CrossEncoderReranker"]
