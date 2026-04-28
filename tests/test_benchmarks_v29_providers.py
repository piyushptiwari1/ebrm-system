"""Tests for v0.29 multi-provider support (any OpenAI-compatible endpoint).

Adds two generic classes that work with OpenAI, Ollama, vLLM, llama.cpp,
OpenRouter, Together, Groq, etc. — anything speaking the OpenAI HTTP API.

All tests stub the ``openai.OpenAI`` client; no live API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn

pytest.importorskip("openai", reason="OpenAI SDK required")
pytest.importorskip("rank_bm25", reason="BM25Retriever needs rank-bm25")


def _episode(question: str = "anything?") -> OfficialEpisode:
    turn = OfficialTurn(
        session_id="s0",
        session_idx=0,
        turn_idx=0,
        role="user",
        content="hello",
        session_date="2024-01-01",
        has_answer=False,
    )
    return OfficialEpisode(
        question_id="q",
        question_type="multi-session",
        question=question,
        answer="x",
        question_date="2024-02-01",
        turns=(turn,),
        answer_session_ids=(),
        is_abstention=False,
    )


def _stub_openai(monkeypatch, *, choices: list[str] | None = None, embed_dim: int = 8):
    """Replace ``openai.OpenAI`` with a stub capturing init + call args."""
    captured: dict = {
        "init_kwargs": None,
        "chat_kwargs": None,
        "embed_kwargs": None,
    }

    def _chat_create(**kw):
        captured["chat_kwargs"] = kw
        rsp = MagicMock()
        rsp.choices = []
        for text in choices or ["stub answer"]:
            msg = MagicMock()
            msg.content = text
            ch = MagicMock()
            ch.message = msg
            rsp.choices.append(ch)
        return rsp

    def _embed_create(**kw):
        captured["embed_kwargs"] = kw
        n = len(kw["input"])
        rsp = MagicMock()
        rsp.data = []
        for i in range(n):
            d = MagicMock()
            # Simple deterministic vector per input.
            d.embedding = [float(i + 1)] * embed_dim
            rsp.data.append(d)
        return rsp

    class _Stub:
        def __init__(self, **kw):
            captured["init_kwargs"] = kw
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = _chat_create
            self.embeddings = MagicMock()
            self.embeddings.create = _embed_create

    import openai

    monkeypatch.setattr(openai, "OpenAI", _Stub, raising=False)
    return captured


# ---------------------------------------------------------------------------
# OpenAICompatibleReader
# ---------------------------------------------------------------------------


class TestOpenAICompatibleReader:
    def test_init_uses_base_url_and_api_key(self, monkeypatch):
        cap = _stub_openai(monkeypatch)
        from benchmarks.reader.openai_compatible import OpenAICompatibleReader

        OpenAICompatibleReader(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        assert cap["init_kwargs"]["base_url"] == "https://api.openai.com/v1"
        assert cap["init_kwargs"]["api_key"] == "sk-test"

    def test_falls_back_to_env_then_not_needed(self, monkeypatch):
        cap = _stub_openai(monkeypatch)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from benchmarks.reader.openai_compatible import OpenAICompatibleReader

        OpenAICompatibleReader(model="m", base_url="http://localhost:11434/v1")
        # Ollama-style: when no key set, defaults to "not-needed".
        assert cap["init_kwargs"]["api_key"] == "not-needed"

    def test_reads_openai_api_key_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        cap = _stub_openai(monkeypatch)
        from benchmarks.reader.openai_compatible import OpenAICompatibleReader

        OpenAICompatibleReader(model="m")
        assert cap["init_kwargs"]["api_key"] == "sk-from-env"

    def test_name_includes_host_and_model_and_sc_marker(self, monkeypatch):
        _stub_openai(monkeypatch)
        from benchmarks.reader.openai_compatible import OpenAICompatibleReader

        r = OpenAICompatibleReader(model="llama3.1:8b", base_url="http://localhost:11434/v1")
        assert "localhost:11434" in r.name
        assert "llama3.1:8b" in r.name
        assert "sc" not in r.name
        r2 = OpenAICompatibleReader(model="m", n_samples=3)
        assert r2.name.endswith("-sc3")

    def test_read_returns_majority_vote(self, monkeypatch):
        cap = _stub_openai(monkeypatch, choices=["7", "7", "9"])
        from benchmarks.reader.openai_compatible import OpenAICompatibleReader

        r = OpenAICompatibleReader(model="m", n_samples=3, sc_temperature=0.4)
        ep = _episode("how many?")
        out = r.read(ep, list(ep.turns))
        assert out == "7"
        assert cap["chat_kwargs"]["n"] == 3
        assert cap["chat_kwargs"]["temperature"] == 0.4
        assert cap["chat_kwargs"]["model"] == "m"

    def test_n_samples_zero_raises(self, monkeypatch):
        _stub_openai(monkeypatch)
        from benchmarks.reader.openai_compatible import OpenAICompatibleReader

        with pytest.raises(ValueError, match="n_samples"):
            OpenAICompatibleReader(model="m", n_samples=0)


# ---------------------------------------------------------------------------
# OpenAICompatibleEmbedder
# ---------------------------------------------------------------------------


class TestOpenAICompatibleEmbedder:
    def test_embed_returns_normalised_vectors(self, monkeypatch, tmp_path):
        _stub_openai(monkeypatch, embed_dim=4)
        from benchmarks.embedders.openai_compatible import OpenAICompatibleEmbedder

        e = OpenAICompatibleEmbedder(
            model="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            cache_dir=tmp_path,
        )
        v = e.embed(["hello", "world"])
        assert v.shape == (2, 4)
        # Each vector is L2-normalised → norm ≈ 1.
        import numpy as np

        norms = np.linalg.norm(v, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_cache_short_circuits_second_call(self, monkeypatch, tmp_path):
        cap = _stub_openai(monkeypatch, embed_dim=4)
        from benchmarks.embedders.openai_compatible import OpenAICompatibleEmbedder

        e = OpenAICompatibleEmbedder(
            model="m",
            api_key="k",
            cache_dir=tmp_path,
        )
        e.embed(["foo"])
        cap["embed_kwargs"] = None
        # Second call for the same text should hit cache, no API.
        e.embed(["foo"])
        assert cap["embed_kwargs"] is None

    def test_name_includes_host_and_model(self, monkeypatch):
        _stub_openai(monkeypatch)
        from benchmarks.embedders.openai_compatible import OpenAICompatibleEmbedder

        e = OpenAICompatibleEmbedder(
            model="nomic-embed-text",
            base_url="http://localhost:11434/v1",
        )
        assert "localhost:11434" in e.name
        assert "nomic-embed-text" in e.name


# ---------------------------------------------------------------------------
# LongMemPipeline classmethods (from_openai / from_ollama / from_openrouter
# / from_provider)
# ---------------------------------------------------------------------------


class TestPipelineProviderClassmethods:
    def test_from_openai_uses_openai_base_url(self, monkeypatch, tmp_path):
        cap = _stub_openai(monkeypatch)
        # Avoid hitting sentence-transformers / cross-encoder (heavy).
        from ebrm_system.longmem import LongMemPipeline

        pipe = LongMemPipeline.from_openai(
            chat_model="gpt-4o-mini",
            embed_model="text-embedding-3-small",
            api_key="sk-test",
            cache_dir=tmp_path,
            neighbor_window=0,
            fusion_rerank=False,
            reranker="none",
        )
        # Reader was constructed → init kwargs captured.
        assert cap["init_kwargs"]["base_url"] == "https://api.openai.com/v1"
        assert cap["init_kwargs"]["api_key"] == "sk-test"
        # Pipeline composition is the OpenAI-compatible reader.
        assert "openai.com" in pipe.reader.name

    def test_from_ollama_no_key_local_url(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cap = _stub_openai(monkeypatch)
        from ebrm_system.longmem import LongMemPipeline

        pipe = LongMemPipeline.from_ollama(
            chat_model="llama3.1:8b",
            embed_model="nomic-embed-text",
            cache_dir=tmp_path,
            neighbor_window=0,
            fusion_rerank=False,
            reranker="none",
        )
        assert cap["init_kwargs"]["base_url"] == "http://localhost:11434/v1"
        assert cap["init_kwargs"]["api_key"] == "not-needed"
        assert "localhost:11434" in pipe.reader.name

    def test_from_openrouter_reads_env_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-sk-test")
        cap = _stub_openai(monkeypatch)
        from ebrm_system.longmem import LongMemPipeline

        LongMemPipeline.from_openrouter(
            chat_model="anthropic/claude-3.5-sonnet",
            cache_dir=tmp_path,
            neighbor_window=0,
            fusion_rerank=False,
            reranker="none",
        )
        assert cap["init_kwargs"]["base_url"] == "https://openrouter.ai/api/v1"
        assert cap["init_kwargs"]["api_key"] == "or-sk-test"

    def test_from_provider_custom_url(self, monkeypatch, tmp_path):
        cap = _stub_openai(monkeypatch)
        from ebrm_system.longmem import LongMemPipeline

        LongMemPipeline.from_provider(
            chat_model="any-model",
            embed_model="any-embed",
            base_url="http://my-vllm:8000/v1",
            api_key="custom-key",
            cache_dir=tmp_path,
            neighbor_window=0,
            fusion_rerank=False,
            reranker="none",
        )
        assert cap["init_kwargs"]["base_url"] == "http://my-vllm:8000/v1"
        assert cap["init_kwargs"]["api_key"] == "custom-key"

    def test_from_provider_invalid_reranker(self, monkeypatch, tmp_path):
        _stub_openai(monkeypatch)
        from ebrm_system.longmem import LongMemPipeline

        with pytest.raises(ValueError, match="unknown reranker"):
            LongMemPipeline.from_provider(
                chat_model="m",
                embed_model="e",
                cache_dir=tmp_path,
                reranker="invalid-x",
                neighbor_window=0,
                fusion_rerank=False,
            )
