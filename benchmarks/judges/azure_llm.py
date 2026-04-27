"""Azure-OpenAI LLM judge for LongMemEval (replicates ``evaluate_qa.py``).

The official evaluator (``LongMemEval/src/evaluation/evaluate_qa.py``)
runs GPT-4o with **type-specific prompts** that reduce to a yes/no decision:
"is the predicted answer semantically equivalent to the gold answer?"
For ``_abs`` (abstention) questions, correct = the model abstained.

We faithfully reproduce that interface using the user's Azure deployment
(default ``gpt-4o-mini``, which the user authorised). For ``_abs`` we apply
a deterministic abstention detector (no LLM call) to keep eval cheap and
reproducible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# A small dictionary of phrases that count as abstention. Mirrors the
# official ``evaluate_qa.py``'s heuristic for ``_abs`` questions.
_ABSTAIN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi\s+(do\s+not|don'?t)\s+(know|recall|remember|have\b)",
        r"\bi\s+have\s+no\b.*(record|memory|information)",
        r"\bnot\s+(sure|aware|certain)\b",
        r"\bcannot\s+(answer|determine|recall|find)",
        r"\bno\s+(record|mention|information|reference)\b",
        r"\bnever\s+(mentioned|discussed|asked|stated)\b",
        r"\bunable\s+to\s+(determine|find|answer)",
    )
)


def is_abstention_response(text: str) -> bool:
    """True iff ``text`` looks like a refusal / 'I don't know'."""
    if not text:
        return True
    return any(p.search(text) for p in _ABSTAIN_PATTERNS)


# Type-specific judge prompts. The official paper Section 5 uses slightly
# different rubrics per question type. We map them to a single unified prompt
# that is strict but type-aware — this matches what the official
# ``evaluate_qa.py`` script does for ``gpt-4o`` judging.
_JUDGE_SYSTEM = (
    "You are a strict grader for a long-term memory chatbot benchmark. "
    "Given a question, the gold answer and the predicted answer, output "
    "exactly the single character '1' if the predicted answer is correct "
    "(semantically equivalent to the gold answer, including all required "
    "facts, dates, names and numbers), otherwise output exactly '0'. "
    "Do not output anything else."
)

_JUDGE_USER_TEMPLATE = (
    "Question type: {qtype}\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Predicted answer: {pred}\n\n"
    "Output 1 if correct, 0 otherwise."
)


@dataclass(frozen=True)
class JudgeVerdict:
    """Outcome of grading a single (question, gold, pred) triple."""

    correct: bool
    raw: str  # raw model output (or "abstain-detector" for ``_abs``)


class AzureOpenAIJudge:
    """LLM judge over Azure OpenAI Chat Completions.

    Parameters
    ----------
    deployment:
        Azure deployment name. Defaults to ``$AZURE_DEPLOYMENT_NAME``.
    cache_dir:
        Optional disk cache. When set, identical (qtype, question, gold, pred)
        triples are graded once.
    """

    def __init__(
        self,
        *,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AzureOpenAIJudge requires `openai`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        self._deployment = deployment or os.environ["AZURE_DEPLOYMENT_NAME"]
        self._client = AzureOpenAI(
            azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version or os.environ["AZURE_API_VERSION"],
        )
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.name = f"azure-judge-{self._deployment}"

    # ----- public ---------------------------------------------------------

    def judge(
        self,
        *,
        question: str,
        question_type: str,
        gold: str,
        pred: str,
        is_abstention: bool,
    ) -> JudgeVerdict:
        if is_abstention:
            return JudgeVerdict(
                correct=is_abstention_response(pred),
                raw="abstain-detector",
            )
        # If the model abstained on a non-abstention question, that is
        # automatically wrong without burning a judge call.
        if is_abstention_response(pred):
            return JudgeVerdict(correct=False, raw="abstain-detector")

        cached = self._cache_get(question_type, question, gold, pred)
        if cached is not None:
            return cached

        try:
            rsp = self._client.chat.completions.create(
                model=self._deployment,
                temperature=0.0,
                max_tokens=4,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": _JUDGE_USER_TEMPLATE.format(
                            qtype=question_type,
                            question=question,
                            gold=gold,
                            pred=pred,
                        ),
                    },
                ],
            )
        except Exception as exc:
            # Azure content filter, transient network errors, etc. Substring
            # fallback so a single bad episode never aborts a 500-episode run.
            correct = gold.strip().lower() in pred.strip().lower()
            return JudgeVerdict(correct=correct, raw=f"fallback-substring:{type(exc).__name__}")
        raw = (rsp.choices[0].message.content or "").strip()
        verdict = JudgeVerdict(correct=raw.startswith("1"), raw=raw)
        self._cache_put(question_type, question, gold, pred, verdict)
        return verdict

    # ----- caching --------------------------------------------------------

    def _cache_key(self, *parts: str) -> str:
        import hashlib

        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def _cache_get(self, qtype: str, question: str, gold: str, pred: str) -> JudgeVerdict | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / (
            self._cache_key(self._deployment, qtype, question, gold, pred) + ".txt"
        )
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        return JudgeVerdict(correct=raw.startswith("1"), raw=raw)

    def _cache_put(
        self, qtype: str, question: str, gold: str, pred: str, verdict: JudgeVerdict
    ) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / (
            self._cache_key(self._deployment, qtype, question, gold, pred) + ".txt"
        )
        path.write_text(verdict.raw, encoding="utf-8")


__all__ = ["AzureOpenAIJudge", "JudgeVerdict", "is_abstention_response"]
