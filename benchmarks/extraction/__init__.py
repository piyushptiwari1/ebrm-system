"""LLM-based memory extraction (Mem0-v3-style).

Distills raw chat sessions into atomic, self-contained "memories" that can
be retrieved more reliably than long, noisy multi-turn dialogue. The
``Retriever`` Protocol is unchanged — extracted memories are exposed as
synthetic ``OfficialTurn`` instances so the rest of the pipeline (BM25,
dense, RRF, reranker, reader) keeps working untouched.
"""

from benchmarks.extraction.azure_llm import AzureLLMExtractor
from benchmarks.extraction.base import (
    ExtractedMemory,
    MemoryExtractor,
    memories_to_episode,
)

__all__ = [
    "AzureLLMExtractor",
    "ExtractedMemory",
    "MemoryExtractor",
    "memories_to_episode",
]
