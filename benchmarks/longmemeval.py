"""LongMemEval-style benchmark harness for tiered latent memory.

LongMemEval (`Wu et al., 2024 <https://arxiv.org/abs/2410.10813>`_) tests an
agent's ability to recall facts from long multi-session conversation
histories. The benchmark has five question types:

    * single-session-user      — recall a fact stated once, mid-session
    * single-session-assistant — recall a fact the agent itself stated
    * multi-session            — combine facts spread across sessions
    * temporal-reasoning       — reason about *when* something was said
    * knowledge-update         — handle a fact superseded by a later one

The full LongMemEval dataset is gated; this harness ships a *synthetic*
generator that produces structurally identical episodes so the EBRM
:class:`~ebrm_system.memory.TieredMemory` can be validated locally and
in CI. Drop the real dataset in via :func:`load_longmemeval_jsonl` when
you have access — same record schema.

Usage
-----

    from ebrm_system.benchmarks.longmemeval import (
        run_longmemeval, synth_longmemeval,
    )
    from ebrm_system.memory import TieredMemory, TieredMemoryConfig

    mem = TieredMemory(TieredMemoryConfig(in_dim=64))
    result = run_longmemeval(synth_longmemeval(seed=0, num_episodes=20), mem)
    print(result.accuracy_by_type)

The harness is torch-free. The "embedding" is a deterministic hash-based
projection so results are reproducible without a real model — drop in your
projector when running for science.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ebrm_system.memory import TieredMemory, TieredMemoryConfig

QuestionType = Literal[
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

ALL_QUESTION_TYPES: tuple[QuestionType, ...] = (
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


@dataclass(frozen=True)
class MemoryFact:
    """A single statement that should be stored in memory."""

    text: str
    session: int
    speaker: Literal["user", "assistant"]
    """Who originally uttered the fact."""
    superseded_by: int | None = None
    """Index of a later fact that updates this one (knowledge-update)."""


@dataclass(frozen=True)
class LongMemEpisode:
    """One LongMemEval episode."""

    id: str
    facts: tuple[MemoryFact, ...]
    question: str
    answer: str
    question_type: QuestionType


@dataclass
class EpisodeResult:
    episode_id: str
    question_type: QuestionType
    correct: bool
    retrieved_top_k: int
    expected_answer: str


@dataclass
class LongMemRunResult:
    total: int
    correct: int
    accuracy: float
    accuracy_by_type: dict[str, float]
    per_type_counts: dict[str, int]
    details: list[EpisodeResult] = field(default_factory=list)


# --------------------------------------------------------------------------
# Embedding (hash-based, deterministic, torch-free)
# --------------------------------------------------------------------------


def hash_embed(text: str, dim: int = 64, seed: int = 0) -> NDArray[np.float32]:
    """Map a string to a unit-norm latent via a seeded hash.

    Pure numpy + hashlib. Good enough for harness self-tests; replace with
    a real projector for science runs.
    """
    digest = hashlib.sha256(f"{seed}::{text}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


# --------------------------------------------------------------------------
# Synthetic episode generator
# --------------------------------------------------------------------------


_SUBJECTS = (
    "Alice",
    "Bob",
    "Carol",
    "Dave",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
)
_HOBBIES = (
    "rock climbing",
    "violin",
    "chess",
    "kitesurfing",
    "pottery",
    "fencing",
    "astronomy",
    "baking",
)
_CITIES = (
    "Lisbon",
    "Kyoto",
    "Reykjavík",
    "Cape Town",
    "Montreal",
    "Tbilisi",
    "Hanoi",
    "Helsinki",
)


def _make_single_session_user(rng: random.Random, eid: int) -> LongMemEpisode:
    subj = rng.choice(_SUBJECTS)
    hobby = rng.choice(_HOBBIES)
    facts = (MemoryFact(text=f"{subj} likes {hobby}.", session=0, speaker="user"),)
    return LongMemEpisode(
        id=f"sssu-{eid}",
        facts=facts,
        question=f"What does {subj} like?",
        answer=hobby,
        question_type="single-session-user",
    )


def _make_single_session_assistant(rng: random.Random, eid: int) -> LongMemEpisode:
    subj = rng.choice(_SUBJECTS)
    city = rng.choice(_CITIES)
    facts = (
        MemoryFact(
            text=f"I told {subj} the best coffee is in {city}.",
            session=0,
            speaker="assistant",
        ),
    )
    return LongMemEpisode(
        id=f"sssa-{eid}",
        facts=facts,
        question=f"Where did I say {subj} should go for coffee?",
        answer=city,
        question_type="single-session-assistant",
    )


def _make_multi_session(rng: random.Random, eid: int) -> LongMemEpisode:
    subj = rng.choice(_SUBJECTS)
    hobby = rng.choice(_HOBBIES)
    city = rng.choice(_CITIES)
    facts = (
        MemoryFact(text=f"{subj} lives in {city}.", session=0, speaker="user"),
        MemoryFact(text=f"{subj} took up {hobby}.", session=2, speaker="user"),
    )
    return LongMemEpisode(
        id=f"ms-{eid}",
        facts=facts,
        question=f"What does {subj} do in {city}?",
        answer=hobby,
        question_type="multi-session",
    )


def _make_temporal(rng: random.Random, eid: int) -> LongMemEpisode:
    subj = rng.choice(_SUBJECTS)
    hobby1 = rng.choice(_HOBBIES)
    hobby2 = rng.choice([h for h in _HOBBIES if h != hobby1])
    facts = (
        MemoryFact(text=f"{subj} used to do {hobby1}.", session=0, speaker="user"),
        MemoryFact(text=f"{subj} now does {hobby2}.", session=3, speaker="user"),
    )
    return LongMemEpisode(
        id=f"tr-{eid}",
        facts=facts,
        question=f"What did {subj} *first* take up?",
        answer=hobby1,
        question_type="temporal-reasoning",
    )


def _make_knowledge_update(rng: random.Random, eid: int) -> LongMemEpisode:
    subj = rng.choice(_SUBJECTS)
    city1 = rng.choice(_CITIES)
    city2 = rng.choice([c for c in _CITIES if c != city1])
    facts = (
        MemoryFact(
            text=f"{subj} lives in {city1}.",
            session=0,
            speaker="user",
            superseded_by=1,
        ),
        MemoryFact(text=f"{subj} moved to {city2}.", session=4, speaker="user"),
    )
    return LongMemEpisode(
        id=f"ku-{eid}",
        facts=facts,
        question=f"Where does {subj} currently live?",
        answer=city2,
        question_type="knowledge-update",
    )


_GENERATORS = {
    "single-session-user": _make_single_session_user,
    "single-session-assistant": _make_single_session_assistant,
    "multi-session": _make_multi_session,
    "temporal-reasoning": _make_temporal,
    "knowledge-update": _make_knowledge_update,
}


def synth_longmemeval(
    *,
    seed: int = 0,
    num_episodes: int = 25,
    types: Sequence[QuestionType] = ALL_QUESTION_TYPES,
) -> list[LongMemEpisode]:
    """Generate a balanced synthetic LongMemEval-style dataset.

    Episodes are split across the requested ``types`` as evenly as possible.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if not types:
        raise ValueError("types must be non-empty")
    rng = random.Random(seed)
    out: list[LongMemEpisode] = []
    for i in range(num_episodes):
        qt = types[i % len(types)]
        out.append(_GENERATORS[qt](rng, i))
    return out


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _grader(answer: str, retrieved: list[tuple[float, object]]) -> bool:
    """Pass iff the expected answer appears in any retrieved payload's text."""
    target = answer.lower()
    for _sim, payload in retrieved:
        text = _payload_text(payload)
        if text and target in text.lower():
            return True
    return False


def _payload_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, MemoryFact):
        return payload.text
    if isinstance(payload, dict) and "text" in payload:
        return str(payload["text"])
    return ""


def run_longmemeval(
    episodes: Iterable[LongMemEpisode],
    memory: TieredMemory,
    *,
    embed_dim: int | None = None,
    embed_seed: int = 0,
    top_k: int = 5,
) -> LongMemRunResult:
    """Run the harness over ``episodes`` against a (possibly empty) ``memory``.

    Each episode:
        1. Writes its facts into ``memory`` (skipping superseded ones for
           knowledge-update episodes — the *current* state should win).
        2. Embeds the question and runs ``memory.search(query, k=top_k)``.
        3. Grades by substring match of the expected answer in any retrieved
           fact's text.

    The TieredMemory should ideally be the size you plan to deploy with;
    eviction and promotion behaviour are part of what's being measured.
    """
    dim = embed_dim if embed_dim is not None else memory.config.in_dim
    if dim != memory.config.in_dim:
        raise ValueError(
            f"embed_dim={dim} does not match memory.config.in_dim={memory.config.in_dim}"
        )
    correct = 0
    total = 0
    by_type_correct: dict[str, int] = dict.fromkeys(ALL_QUESTION_TYPES, 0)
    by_type_total: dict[str, int] = dict.fromkeys(ALL_QUESTION_TYPES, 0)
    details: list[EpisodeResult] = []

    for episode in episodes:
        # 1) Write facts. Skip superseded facts so knowledge-update reflects
        # the latest known state — exactly what a well-tuned memory should
        # produce after promotion + summarization in a real run.
        active = [f for f in episode.facts if f.superseded_by is None]
        if active:
            latents = np.stack([hash_embed(f.text, dim=dim, seed=embed_seed) for f in active])
            memory.add(latents, list(active))

        # 2) Retrieve.
        q_latent = hash_embed(episode.question, dim=dim, seed=embed_seed)
        retrieved = memory.search(q_latent, k=top_k)

        # 3) Grade.
        ok = _grader(episode.answer, retrieved)
        total += 1
        by_type_total[episode.question_type] += 1
        if ok:
            correct += 1
            by_type_correct[episode.question_type] += 1
        details.append(
            EpisodeResult(
                episode_id=episode.id,
                question_type=episode.question_type,
                correct=ok,
                retrieved_top_k=len(retrieved),
                expected_answer=episode.answer,
            )
        )

    accuracy_by_type = {
        qt: (by_type_correct[qt] / by_type_total[qt] if by_type_total[qt] else 0.0)
        for qt in ALL_QUESTION_TYPES
    }
    return LongMemRunResult(
        total=total,
        correct=correct,
        accuracy=(correct / total) if total else 0.0,
        accuracy_by_type=accuracy_by_type,
        per_type_counts=by_type_total,
        details=details,
    )


def default_memory(in_dim: int = 64) -> TieredMemory:
    """A reasonably-tuned :class:`TieredMemory` for LongMemEval runs."""
    from ebrm_system.memory import TierConfig

    return TieredMemory(
        TieredMemoryConfig(
            in_dim=in_dim,
            working=TierConfig(max_size=64, promote_after_hits=2),
            episodic=TierConfig(max_size=512, promote_after_hits=3),
            semantic=TierConfig(max_size=4096),
        )
    )


# --------------------------------------------------------------------------
# Real-dataset adapter (JSONL) + result serialization
# --------------------------------------------------------------------------


def load_longmemeval_jsonl(path: str | Path) -> list[LongMemEpisode]:
    """Load LongMemEval episodes from a JSONL file.

    Each line must be a JSON object with at least::

        {
          "id": "<str>",
          "question": "<str>",
          "answer": "<str>",
          "question_type": "<one of ALL_QUESTION_TYPES>",
          "facts": [
            {
              "text": "<str>",
              "session": <int>,
              "speaker": "user" | "assistant",
              "superseded_by": <int|null>   # optional
            },
            ...
          ]
        }

    Unknown extra fields are ignored. Validation is strict: missing
    required keys or unknown question types raise ``ValueError`` with the
    offending line number, so dataset-format issues fail loud at load time
    rather than silently distorting results.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LongMemEval JSONL not found: {p}")
    episodes: list[LongMemEpisode] = []
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{line_no}: invalid JSON ({exc})") from exc
            try:
                episodes.append(_parse_episode(obj))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{p}:{line_no}: {exc}") from exc
    return episodes


def _parse_episode(obj: dict[str, object]) -> LongMemEpisode:
    for required in ("id", "question", "answer", "question_type", "facts"):
        if required not in obj:
            raise KeyError(f"missing required field {required!r}")
    qt = obj["question_type"]
    if qt not in ALL_QUESTION_TYPES:
        raise ValueError(f"question_type {qt!r} not in {list(ALL_QUESTION_TYPES)}")
    raw_facts = obj["facts"]
    if not isinstance(raw_facts, list):
        raise ValueError("facts must be a list")
    facts: list[MemoryFact] = []
    for i, raw_fact in enumerate(raw_facts):
        if not isinstance(raw_fact, dict):
            raise ValueError(f"facts[{i}] must be an object")
        for required in ("text", "session", "speaker"):
            if required not in raw_fact:
                raise KeyError(f"facts[{i}] missing field {required!r}")
        speaker = raw_fact["speaker"]
        if speaker not in ("user", "assistant"):
            raise ValueError(f"facts[{i}].speaker must be 'user' or 'assistant', got {speaker!r}")
        facts.append(
            MemoryFact(
                text=str(raw_fact["text"]),
                session=int(raw_fact["session"]),
                speaker=speaker,  # type: ignore[arg-type]
                superseded_by=(
                    int(raw_fact["superseded_by"])
                    if raw_fact.get("superseded_by") is not None
                    else None
                ),
            )
        )
    return LongMemEpisode(
        id=str(obj["id"]),
        facts=tuple(facts),
        question=str(obj["question"]),
        answer=str(obj["answer"]),
        question_type=qt,  # type: ignore[arg-type]
    )


def write_results_json(
    result: LongMemRunResult,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Persist a run result as JSON.

    The schema is stable: top-level keys are ``total``, ``correct``,
    ``accuracy``, ``accuracy_by_type``, ``per_type_counts``, ``details``,
    plus an optional ``metadata`` block (e.g. memory config, embed dim,
    code version) for reproducibility.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "total": result.total,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "accuracy_by_type": result.accuracy_by_type,
        "per_type_counts": result.per_type_counts,
        "details": [
            {
                "episode_id": d.episode_id,
                "question_type": d.question_type,
                "correct": d.correct,
                "retrieved_top_k": d.retrieved_top_k,
                "expected_answer": d.expected_answer,
            }
            for d in result.details
        ],
    }
    if metadata:
        payload["metadata"] = metadata
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ALL_QUESTION_TYPES",
    "EpisodeResult",
    "LongMemEpisode",
    "LongMemRunResult",
    "MemoryFact",
    "QuestionType",
    "default_memory",
    "hash_embed",
    "load_longmemeval_jsonl",
    "run_longmemeval",
    "synth_longmemeval",
    "write_results_json",
]
