"""LongMem pipeline implementation — wraps ``benchmarks/`` behind a clean API."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class LongMemTurn:
    """A single chat turn fed into the pipeline."""

    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class LongMemSession:
    """A single conversation session — many turns sharing a date."""

    session_id: str
    date: str  # any string parseable as a date; pipeline preserves verbatim
    turns: tuple[LongMemTurn, ...]

    @classmethod
    def from_dicts(
        cls,
        session_id: str,
        date: str,
        turns: Iterable[Mapping[str, str]],
    ) -> LongMemSession:
        return cls(
            session_id=session_id,
            date=date,
            turns=tuple(LongMemTurn(role=str(t["role"]), content=str(t["content"])) for t in turns),
        )


@dataclass(frozen=True)
class LongMemAnswer:
    """Result of :meth:`LongMemPipeline.ask`.

    Attributes
    ----------
    answer:
        The reader's final answer string.
    retrieved_session_ids:
        Distinct session ids whose turns were surfaced and shown to the
        reader, in retrieval order.
    n_retrieved:
        Number of individual chat turns shown to the reader.
    """

    answer: str
    retrieved_session_ids: tuple[str, ...]
    n_retrieved: int


# ---------------------------------------------------------------------------
# Reader / Retriever protocols (re-exported to keep public surface clean)
# ---------------------------------------------------------------------------


class _ReaderLike(Protocol):
    def read(self, episode: Any, retrieved_turns: list[Any]) -> str: ...


class _RetrieverLike(Protocol):
    name: str

    def retrieve(self, episode: Any, *, top_k: int) -> list[Any]: ...


# ---------------------------------------------------------------------------
# LongMemPipeline
# ---------------------------------------------------------------------------


@dataclass
class LongMemPipeline:
    """Public facade over the LongMemEval-tuned retrieval + reader stack.

    Use :meth:`from_default` to build the same pipeline that scores 77.4 %
    on LongMemEval oracle (Azure embeddings + BM25 + dense + RRF + BGE
    cross-encoder + LLM-fusion reranker + neighbor expansion + Azure
    reader). Pass your own ``retriever`` / ``reader`` to swap layers.

    The pipeline is **session-additive**: each call to :meth:`add_session`
    appends turns to the in-memory haystack, and each :meth:`ask` rebuilds
    the retrieval episode from the cumulative haystack. There is no
    background process / disk persistence — the pipeline is intentionally
    a thin orchestration layer; persistence is the caller's job.
    """

    retriever: _RetrieverLike
    reader: _ReaderLike
    top_k: int = 10
    per_type_top_k: bool = True
    _sessions: list[LongMemSession] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_default(
        cls,
        *,
        top_k: int = 10,
        cache_dir: str | Path = "cache",
        neighbor_window: int = 1,
        fusion_rerank: bool = True,
        aggregation_cot: bool = True,
        reranker: str = "bge",
        embedder: str = "azure",
        n_samples: int = 1,
        sc_temperature: float = 0.5,
    ) -> LongMemPipeline:
        """Build the recommended default pipeline.

        Mirrors the v0.28 default benchmark runner. Requires the
        ``[embedders]`` extra (``pip install 'ebrm-system[embedders]'``)
        and the Azure OpenAI environment variables
        ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
        ``AZURE_API_VERSION``, ``AZURE_DEPLOYMENT_NAME``,
        ``AZURE_EMBEDDING_MODEL_NAME``.

        Set ``n_samples > 1`` to enable v0.29 self-consistency: the reader
        is sampled N times at ``sc_temperature`` in a single API call and
        the majority answer wins. ``n_samples=3`` is the recommended
        balance of cost vs. quality.
        """
        from benchmarks.embedders.azure_openai import AzureOpenAIEmbedder
        from benchmarks.fusion import LLMFusionReranker
        from benchmarks.reader.azure_llm import AzureOpenAIReader
        from benchmarks.retrieval import (
            BM25Retriever,
            DenseRetriever,
            NeighborExpander,
            RRFRetriever,
        )

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        if embedder != "azure":
            raise ValueError(
                f"from_default only supports embedder='azure'; got {embedder!r}. "
                "Construct LongMemPipeline directly with your own retriever."
            )

        retr: Any = RRFRetriever(
            [
                DenseRetriever(AzureOpenAIEmbedder(cache_dir=cache_path / "embed")),  # type: ignore[arg-type]
                BM25Retriever(),
            ]
        )
        if reranker == "bge":
            from benchmarks.retrieval.reranker import CrossEncoderReranker

            retr = CrossEncoderReranker(retr, candidate_k=20)
        elif reranker != "none":
            raise ValueError(f"unknown reranker {reranker!r}")
        if fusion_rerank:
            retr = LLMFusionReranker(base=retr, candidate_k=20, cache_dir=cache_path / "fusion")
        if neighbor_window > 0:
            retr = NeighborExpander(retr, window=neighbor_window)

        rdr = AzureOpenAIReader(
            aggregation_cot=aggregation_cot,
            n_samples=n_samples,
            sc_temperature=sc_temperature,
        )
        return cls(retriever=retr, reader=rdr, top_k=top_k)

    # ------------------------------------------------------------------
    # Multi-provider convenience constructors (v0.29.1+)
    # ------------------------------------------------------------------

    @classmethod
    def from_provider(
        cls,
        *,
        chat_model: str,
        embed_model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        top_k: int = 10,
        cache_dir: str | Path = "cache",
        neighbor_window: int = 1,
        fusion_rerank: bool = False,
        aggregation_cot: bool = True,
        reranker: str = "none",
        n_samples: int = 1,
        sc_temperature: float = 0.5,
        timeout: float = 60.0,
    ) -> LongMemPipeline:
        """Build a pipeline against any OpenAI-compatible provider.

        Works with **OpenAI, Ollama, vLLM, llama.cpp server, OpenRouter,
        Together, Groq, Anyscale, LM Studio, Mistral, DeepInfra**, and any
        other server speaking the OpenAI HTTP API. For Azure OpenAI,
        prefer :meth:`from_default` (it uses Azure-specific auth and is
        the SOTA-validated path).

        Defaults are deliberately conservative: ``fusion_rerank=False``
        and ``reranker="none"`` keep latency / cost low for non-SOTA
        backends. Turn them on if your provider can sustain the extra
        chat / cross-encoder load.

        Examples
        --------
        OpenAI proper::

            LongMemPipeline.from_provider(
                chat_model="gpt-4o-mini",
                embed_model="text-embedding-3-small",
                api_key=os.environ["OPENAI_API_KEY"],
            )

        Ollama (local, no key needed)::

            LongMemPipeline.from_ollama(
                chat_model="llama3.1:8b",
                embed_model="nomic-embed-text",
            )

        OpenRouter::

            LongMemPipeline.from_provider(
                chat_model="anthropic/claude-3.5-sonnet",
                embed_model="text-embedding-3-small",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        """
        from benchmarks.embedders.openai_compatible import OpenAICompatibleEmbedder
        from benchmarks.reader.openai_compatible import OpenAICompatibleReader
        from benchmarks.retrieval import (
            BM25Retriever,
            DenseRetriever,
            NeighborExpander,
            RRFRetriever,
        )

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        embedder_obj = OpenAICompatibleEmbedder(
            model=embed_model,
            base_url=base_url,
            api_key=api_key,
            cache_dir=cache_path / "embed",
            timeout=timeout,
        )
        retr: Any = RRFRetriever([DenseRetriever(embedder_obj), BM25Retriever()])  # type: ignore[arg-type]
        if reranker == "bge":
            from benchmarks.retrieval.reranker import CrossEncoderReranker

            retr = CrossEncoderReranker(retr, candidate_k=20)
        elif reranker != "none":
            raise ValueError(f"unknown reranker {reranker!r}")
        if fusion_rerank:
            from benchmarks.fusion import LLMFusionReranker

            # NOTE: fusion uses the Azure-tuned LLMFusionReranker; for
            # non-Azure providers it falls back to text-similarity scoring.
            retr = LLMFusionReranker(base=retr, candidate_k=20, cache_dir=cache_path / "fusion")
        if neighbor_window > 0:
            retr = NeighborExpander(retr, window=neighbor_window)

        rdr = OpenAICompatibleReader(
            model=chat_model,
            base_url=base_url,
            api_key=api_key,
            aggregation_cot=aggregation_cot,
            n_samples=n_samples,
            sc_temperature=sc_temperature,
            timeout=timeout,
        )
        return cls(retriever=retr, reader=rdr, top_k=top_k)

    @classmethod
    def from_openai(
        cls,
        *,
        chat_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> LongMemPipeline:
        """Build against OpenAI proper. Reads ``OPENAI_API_KEY`` if unset."""
        return cls.from_provider(
            chat_model=chat_model,
            embed_model=embed_model,
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            **kwargs,
        )

    @classmethod
    def from_ollama(
        cls,
        *,
        chat_model: str = "llama3.1:8b",
        embed_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> LongMemPipeline:
        """Build against a local Ollama instance. No API key required.

        Default models assume ``ollama pull llama3.1:8b`` and
        ``ollama pull nomic-embed-text``. Timeout defaults to 5 min
        because local CPU inference can be slow on n>1 sampling.
        """
        return cls.from_provider(
            chat_model=chat_model,
            embed_model=embed_model,
            base_url=base_url,
            api_key="not-needed",
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def from_openrouter(
        cls,
        *,
        chat_model: str,
        embed_model: str = "text-embedding-3-small",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> LongMemPipeline:
        """Build against OpenRouter. Reads ``OPENROUTER_API_KEY`` if unset.

        Note: OpenRouter does not host embedding models — pass an
        ``embed_model`` from a provider that does (or use a different
        ``embed_base_url`` by constructing :class:`OpenAICompatibleEmbedder`
        directly).
        """
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        return cls.from_provider(
            chat_model=chat_model,
            embed_model=embed_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def add_session(
        self,
        session_id: str,
        date: str,
        turns: Iterable[Mapping[str, str]],
    ) -> None:
        """Append a session of chat turns to the haystack."""
        self._sessions.append(LongMemSession.from_dicts(session_id, date, turns))

    def add_sessions(self, sessions: Sequence[LongMemSession]) -> None:
        """Append multiple pre-built :class:`LongMemSession` objects."""
        self._sessions.extend(sessions)

    def reset(self) -> None:
        """Drop the in-memory haystack."""
        self._sessions.clear()

    @property
    def sessions(self) -> tuple[LongMemSession, ...]:
        """Read-only view of the current haystack."""
        return tuple(self._sessions)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        today: str | None = None,
        question_type: str = "multi-session",
        top_k: int | None = None,
    ) -> LongMemAnswer:
        """Answer ``question`` against the current haystack.

        Parameters
        ----------
        question:
            The user's question, free-form.
        today:
            "Today's" date passed to the reader as the temporal anchor for
            phrases like "last week" / "two weeks ago". Defaults to the
            most recent session date in the haystack.
        question_type:
            One of the LongMemEval question types
            (``"multi-session"``, ``"temporal-reasoning"``,
            ``"knowledge-update"``, ``"single-session-user"``,
            ``"single-session-assistant"``,
            ``"single-session-preference"``). Affects per-type top_k
            routing and CoT gating. ``"multi-session"`` is the safest
            default.
        top_k:
            Override the pipeline's default top_k for this question.
        """
        if not self._sessions:
            return LongMemAnswer(
                answer="I don't know.",
                retrieved_session_ids=(),
                n_retrieved=0,
            )

        from benchmarks.datasets.longmemeval_official import (
            OfficialEpisode,
            OfficialTurn,
        )
        from benchmarks.router import classify_question, top_k_for

        # Build a synthetic OfficialEpisode from the haystack.
        turn_objs: list[OfficialTurn] = []
        for s_idx, sess in enumerate(self._sessions):
            for t_idx, turn in enumerate(sess.turns):
                turn_objs.append(
                    OfficialTurn(
                        session_id=sess.session_id,
                        session_idx=s_idx,
                        turn_idx=t_idx,
                        role=turn.role,
                        content=turn.content,
                        session_date=sess.date,
                        has_answer=False,
                    )
                )
        question_date = today or (self._sessions[-1].date if self._sessions else "")
        episode = OfficialEpisode(
            question_id="user",
            question_type=question_type,
            question=question,
            answer="",
            question_date=question_date,
            turns=tuple(turn_objs),
            answer_session_ids=(),
            is_abstention=False,
        )

        effective_top_k = top_k if top_k is not None else self.top_k
        if self.per_type_top_k:
            tag = classify_question(question, question_type)
            effective_top_k = top_k_for(tag, default=effective_top_k)

        retrieved_scored = self.retriever.retrieve(episode, top_k=effective_top_k)
        retrieved_turns = [st.turn for st in retrieved_scored]
        answer = self.reader.read(episode, retrieved_turns)

        # Preserve order, dedupe.
        seen: set[str] = set()
        ordered_ids: list[str] = []
        for t in retrieved_turns:
            if t.session_id not in seen:
                seen.add(t.session_id)
                ordered_ids.append(t.session_id)

        return LongMemAnswer(
            answer=answer,
            retrieved_session_ids=tuple(ordered_ids),
            n_retrieved=len(retrieved_turns),
        )


__all__ = [
    "LongMemAnswer",
    "LongMemPipeline",
    "LongMemSession",
    "LongMemTurn",
]
