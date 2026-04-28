"""Public long-term memory pipeline for ``ebrm-system``.

This package is the **user-facing** entry point to the LongMemEval-tuned
retrieval + reader pipeline that drives ``benchmarks/`` internally. It
gives you the same hybrid (BM25+dense+RRF) + cross-encoder + LLM-fusion
+ aggregation-CoT stack that scores 77.4 % on LongMemEval oracle, behind
a five-line Python API and an ``ebrm-system longmem`` CLI subcommand.

Quick start (Azure — SOTA-validated path)
-----------------------------------------

    from ebrm_system.longmem import LongMemPipeline

    pipe = LongMemPipeline.from_default()  # needs AZURE_OPENAI_* env vars
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

Other providers
---------------

The pipeline works with **any OpenAI-compatible endpoint** — OpenAI,
Ollama, vLLM, llama.cpp server, OpenRouter, Together, Groq, Anyscale,
LM Studio, Mistral, DeepInfra, etc.::

    LongMemPipeline.from_openai(api_key=...)               # OpenAI proper
    LongMemPipeline.from_ollama()                          # local, no key
    LongMemPipeline.from_openrouter(chat_model="anthropic/claude-3.5-sonnet")
    LongMemPipeline.from_provider(                         # fully custom
        chat_model="...", embed_model="...",
        base_url="http://your-vllm:8000/v1", api_key="...",
    )

Only :meth:`LongMemPipeline.from_default` (Azure) is benchmark-validated
at 77.4 % oracle; other providers will score differently depending on
the chat / embedding model you pick.

Components are lazy-imported, so plain ``pip install ebrm-system`` is
enough to import this module — the optional ``[embedders]`` extra is
only required when you actually run the pipeline.
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
