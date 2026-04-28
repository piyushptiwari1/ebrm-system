"""OpenAI-compatible reader for any provider exposing the OpenAI API.

Works with: OpenAI, Ollama, vLLM, llama.cpp server, OpenRouter, Together,
Groq, Anyscale, LM Studio, Mistral, DeepInfra, and any other server that
speaks ``POST /v1/chat/completions`` in OpenAI shape.

Reuses 100 % of :class:`AzureOpenAIReader`'s prompt logic, gating, and
self-consistency voting via subclassing — only the underlying chat
client differs.
"""

from __future__ import annotations

import os

from benchmarks.reader.azure_llm import AzureOpenAIReader


class OpenAICompatibleReader(AzureOpenAIReader):
    """LongMemEval reader backed by any OpenAI-compatible chat endpoint.

    Parameters
    ----------
    model:
        The chat model name, e.g. ``"gpt-4o-mini"`` for OpenAI,
        ``"llama3.1:8b"`` for Ollama, ``"mistralai/Mistral-7B-Instruct"``
        for a vLLM-hosted model.
    base_url:
        OpenAI-compatible endpoint. Defaults to OpenAI proper.
    api_key:
        API key. Falls back to ``OPENAI_API_KEY``. Pass ``"not-needed"``
        for local servers (Ollama, llama.cpp) that ignore auth.
    aggregation_cot, temporal_ordering_cot, n_samples, sc_temperature:
        See :class:`AzureOpenAIReader` — identical semantics.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
        aggregation_cot: bool = False,
        temporal_ordering_cot: bool = False,
        n_samples: int = 1,
        sc_temperature: float = 0.5,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAICompatibleReader requires `openai`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")

        # Bypass AzureOpenAIReader.__init__ (it requires AZURE_* env vars);
        # we mirror its attribute setup with OpenAI-compatible client.
        self._deployment = model
        self._client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
            timeout=timeout,
        )
        host = base_url.rstrip("/").split("//", 1)[-1].split("/", 1)[0]
        self.name = f"oai-reader-{host}-{model}"
        if n_samples > 1:
            self.name += f"-sc{n_samples}"
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._aggregation_cot = aggregation_cot
        self._temporal_ordering_cot = temporal_ordering_cot
        self._n_samples = n_samples
        self._sc_temperature = sc_temperature


__all__ = ["OpenAICompatibleReader"]
