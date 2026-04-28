"""LLM-based multi-signal fusion reranker (v0.21).

Replaces hand-tuned linear blends (temporal/entity, v0.20) with a single
gpt-4o-mini call that ranks the top-N bge candidates jointly considering
semantic, temporal and entity relevance. Disk-cached on
``sha256(deployment + question + question_date + candidate_texts)``.
"""

from benchmarks.fusion.llm_fusion import LLMFusionReranker

__all__ = ["LLMFusionReranker"]
