"""Public long-term memory pipeline for ``ebrm-system``.

This package is the **user-facing** entry point to the LongMemEval-tuned
retrieval + reader pipeline that drives ``benchmarks/`` internally. It
gives you the same hybrid (BM25+dense+RRF) + cross-encoder + LLM-fusion
+ aggregation-CoT stack that scores 77.4 % on LongMemEval oracle, behind
a five-line Python API and an ``ebrm-system longmem`` CLI subcommand.

Quick start
-----------

    from ebrm_system.longmem import LongMemPipeline

    pipe = LongMemPipeline.from_default()
    pipe.add_session(
        session_id="s1",
        date="2024-03-12 09:30",
        turns=[
            {"role": "user",      "content": "I just bought a new gravel bike."},
            {"role": "assistant", "content": "Nice, what model?"},
            {"role": "user",      "content": "A Trek Checkpoint SL5."},
        ],
    )
    answer = pipe.ask("What bike did I buy?", today="2024-04-01 10:00")

The default constructor wires up the same components used by the
benchmark runner — **Azure OpenAI** for embeddings, the LLM-fusion
reranker, and the reader. You can swap any layer by passing your own
``Retriever``/``Reader`` to :class:`LongMemPipeline`.

Components are lazy-imported, so plain ``pip install ebrm-system`` is
enough to import this module — the optional ``[embedders]`` extra is
only required when you actually call the Azure-backed pipeline.
"""

from __future__ import annotations

from ebrm_system.longmem.pipeline import (
    LongMemAnswer,
    LongMemPipeline,
    LongMemSession,
    LongMemTurn,
)

__all__ = [
    "LongMemAnswer",
    "LongMemPipeline",
    "LongMemSession",
    "LongMemTurn",
]
