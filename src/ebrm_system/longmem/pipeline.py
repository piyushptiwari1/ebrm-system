"""LongMem pipeline implementation — wraps ``benchmarks/`` behind a clean API."""

from __future__ import annotations

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
                DenseRetriever(AzureOpenAIEmbedder(cache_dir=cache_path / "embed")),
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
