"""GSM8K verifier-guided re-ranking benchmark.

Compares three test-time strategies on the GSM8K test split:

1. **single**: one greedy completion (temperature 0).
2. **majority@N**: sample N completions, majority-vote the parsed numeric answer.
3. **ebrm@N**: sample N completions, pick the candidate with lowest EBRM energy.

The candidate generator is any callable ``(question: str, n: int) -> list[str]``.
A reference implementation backed by an OpenAI-compatible Chat Completions API
is provided in :func:`make_chat_completions_generator`.

Usage
-----
::

    from ebrm_system.verifiers import EBRMScorer
    from benchmarks.gsm8k_verifier import (
        GSM8KVerifierBench, make_chat_completions_generator,
    )

    scorer = EBRMScorer.from_pretrained()
    gen = make_chat_completions_generator(model="gpt-4o-mini", temperature=0.7)
    bench = GSM8KVerifierBench(generator=gen, scorer=scorer, n_candidates=8)
    result = bench.run(limit=200)
    print(result)

The harness is **not** auto-registered with the LongMemEval runner; it has
distinct semantics (reasoning rather than retrieval) and a different result
shape (three accuracy numbers per row).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ebrm_system.verifiers.ebrm_scorer import EBRMScorer

# Greedy heuristics used by GSM8K research code:
#   * "#### <number>" — official format.
#   * "The answer is <number>" — common chat-tuned suffix.
#   * Else: last number on the last non-empty line.
_ANSWER_HASH_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)\s*\$?(-?[\d,]+\.?\d*)",
    re.IGNORECASE,
)
_ANY_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_numeric_answer(text: str) -> float | None:
    """Best-effort numeric extraction. Returns ``None`` if nothing parses."""
    for pat in (_ANSWER_HASH_RE, _ANSWER_PHRASE_RE):
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    # Fallback: last number anywhere in the response.
    matches = _ANY_NUMBER_RE.findall(text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def _majority_vote(answers: list[float | None]) -> float | None:
    """Round-and-bucket majority vote, breaking ties by first occurrence."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts: dict[float, int] = {}
    order: list[float] = []
    for a in valid:
        if a not in counts:
            order.append(a)
        counts[a] = counts.get(a, 0) + 1
    best_count = max(counts.values())
    for a in order:
        if counts[a] == best_count:
            return a
    return None  # unreachable


# ----------------------------------------------------------------- generator


class CandidateGenerator(Protocol):
    """Callable that returns ``n`` independent candidate solutions for a question."""

    def __call__(self, question: str, n: int) -> list[str]: ...


def make_chat_completions_generator(
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_key: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
) -> CandidateGenerator:
    """Return a generator backed by the OpenAI Python SDK.

    Works with any OpenAI-compatible endpoint (vLLM, Ollama with the OpenAI
    shim, OpenRouter, Azure OpenAI). Requires ``pip install openai``.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "make_chat_completions_generator requires openai. "
            "Install with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_msg = system_prompt or (
        "You are a careful math tutor. Solve the problem step by step. "
        "On the last line write exactly: 'The answer is <number>.'"
    )

    def _gen(question: str, n: int) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": question},
                ],
            )
            out.append(resp.choices[0].message.content or "")
        return out

    return _gen


# ----------------------------------------------------------------- bench


@dataclass
class GSM8KVerifierResult:
    total: int
    single_correct: int
    majority_correct: int
    ebrm_correct: int
    n_candidates: int
    rows: list[dict[str, object]] = field(default_factory=list)

    @property
    def single_acc(self) -> float:
        return self.single_correct / self.total if self.total else 0.0

    @property
    def majority_acc(self) -> float:
        return self.majority_correct / self.total if self.total else 0.0

    @property
    def ebrm_acc(self) -> float:
        return self.ebrm_correct / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"GSM8K verifier-bench (n={self.total}, candidates={self.n_candidates}): "
            f"single={self.single_acc:.3f}  "
            f"majority@{self.n_candidates}={self.majority_acc:.3f}  "
            f"ebrm@{self.n_candidates}={self.ebrm_acc:.3f}"
        )


@dataclass
class GSM8KVerifierBench:
    """Three-way accuracy comparison over GSM8K test."""

    generator: CandidateGenerator
    scorer: EBRMScorer | None = None
    n_candidates: int = 8
    tolerance: float = 1e-3

    def run(
        self,
        *,
        examples: Iterable[object] | None = None,
        limit: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> GSM8KVerifierResult:
        """Run the benchmark.

        ``examples`` may be any iterable of objects with ``query`` and
        ``expected`` attributes (e.g. :class:`benchmarks.runner.BenchmarkExample`).
        Defaults to the full GSM8K test split via :class:`benchmarks.gsm8k.GSM8K`.
        """
        if examples is None:
            from benchmarks.gsm8k import GSM8K

            examples = GSM8K(split="test", limit=limit).examples()
        elif limit is not None:
            examples = list(examples)[:limit]

        result = GSM8KVerifierResult(
            total=0,
            single_correct=0,
            majority_correct=0,
            ebrm_correct=0,
            n_candidates=self.n_candidates,
        )

        for i, ex in enumerate(examples):
            query = str(ex.query)  # type: ignore[attr-defined]
            expected = float(ex.expected)  # type: ignore[attr-defined]

            candidates = self.generator(query, self.n_candidates)
            if not candidates:
                continue
            parsed = [parse_numeric_answer(c) for c in candidates]

            single_ans = parsed[0]
            majority_ans = _majority_vote(parsed)

            if self.scorer is not None:
                selection = self.scorer.select_best(query, candidates)
                ebrm_ans = parsed[selection.index]
            else:
                ebrm_ans = single_ans

            result.total += 1
            if single_ans is not None and abs(single_ans - expected) < self.tolerance:
                result.single_correct += 1
            if majority_ans is not None and abs(majority_ans - expected) < self.tolerance:
                result.majority_correct += 1
            if ebrm_ans is not None and abs(ebrm_ans - expected) < self.tolerance:
                result.ebrm_correct += 1

            result.rows.append(
                {
                    "id": getattr(ex, "id", f"ex-{i}"),
                    "expected": expected,
                    "single": single_ans,
                    "majority": majority_ans,
                    "ebrm": ebrm_ans,
                    "parsed": parsed,
                }
            )
            if progress is not None:
                progress(i + 1, result.total)

        return result
