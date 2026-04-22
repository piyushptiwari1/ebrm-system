"""Intent & difficulty classifier.

Routes incoming queries to the right reasoning path. The baseline is a fast
rule-based classifier; a neural classifier can be swapped in later via the
same Classifier protocol.

Intent categories:
    arithmetic      — single-expression calculation
    math_reasoning  — multi-step math (GSM8K-style)
    code            — code generation/debugging
    factual         — knowledge lookup
    creative        — open-ended generation
    dialogue        — conversational turn
    unknown         — unclassified

Difficulty is a float in [0, 1]. Used to set Langevin step budget and number
of parallel traces. Cheap to compute; the reasoner can override it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Intent(str, Enum):
    ARITHMETIC = "arithmetic"
    MATH_REASONING = "math_reasoning"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    DIALOGUE = "dialogue"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentPrediction:
    """Classifier output: intent + difficulty + compute budget hint."""

    intent: Intent
    difficulty: float  # 0.0 = trivial, 1.0 = hard
    suggested_langevin_steps: int
    suggested_restarts: int
    suggested_trace_count: int
    reasoning: str  # human-readable why

    def __post_init__(self) -> None:
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"difficulty must be in [0,1], got {self.difficulty}")
        if self.suggested_langevin_steps < 1:
            raise ValueError("suggested_langevin_steps must be >= 1")


class Classifier(Protocol):
    def classify(self, query: str) -> IntentPrediction: ...


# --- Rule-based baseline ---

_CODE_KEYWORDS = (
    r"\bdef\s+\w+|\bclass\s+\w+|\bimport\s+\w+|\bfunction\b|\bbug\b|\bdebug\b|"
    r"\berror\b|\btraceback\b|\bpython\b|\bjavascript\b|\btypescript\b|\brust\b|"
    r"\bwrite\s+a\s+(function|program|script)\b|\brefactor\b"
)
_MATH_KEYWORDS = (
    r"\bsolve\b|\bcompute\b|\bcalculate\b|\bprove\b|\bintegral\b|\bderivative\b|"
    r"\bequation\b|\bsum\b|\bproduct\b|\baverage\b|\bpercent\b|\bratio\b"
)
_CREATIVE_KEYWORDS = (
    r"\bwrite\s+a\s+(story|poem|essay|article|blog)\b|\bimagine\b|\binvent\b|"
    r"\bmake\s+up\b|\bcompose\b|\bnarrate\b"
)
_FACTUAL_KEYWORDS = (
    r"\bwho\s+is\b|\bwhat\s+is\b|\bwhen\s+was\b|\bwhere\s+is\b|\bwhy\s+did\b|"
    r"\bexplain\b|\bdefine\b|\bdescribe\b"
)
_DIALOGUE_KEYWORDS = r"\b(hi|hello|hey|thanks|thank you|sorry|ok|okay|yes|no)\b"
_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


class RuleBasedClassifier:
    """Fast rule-based intent classifier. No model load required."""

    name = "rule_based"

    def classify(self, query: str) -> IntentPrediction:
        q = query.strip().lower()
        if not q:
            return IntentPrediction(
                intent=Intent.UNKNOWN,
                difficulty=0.0,
                suggested_langevin_steps=50,
                suggested_restarts=1,
                suggested_trace_count=1,
                reasoning="empty query",
            )

        num_count = len(_NUMBER_RE.findall(q))
        word_count = len(q.split())

        # Code first — code queries often contain math keywords too
        if re.search(_CODE_KEYWORDS, q):
            difficulty = min(1.0, 0.3 + word_count / 300)
            return self._pack(
                Intent.CODE,
                difficulty,
                f"matched code keywords; length={word_count}",
            )

        # Arithmetic: short, number-heavy, no narrative
        if num_count >= 1 and word_count <= 20 and not re.search(_MATH_KEYWORDS, q):
            # Still math_reasoning if there's a "problem" structure
            if re.search(r"\?$", q) and word_count > 8:
                return self._pack(
                    Intent.MATH_REASONING,
                    min(1.0, 0.4 + num_count / 10),
                    "math word problem (short)",
                )
            return self._pack(
                Intent.ARITHMETIC,
                min(1.0, 0.1 + num_count / 15),
                "short numeric expression",
            )

        if re.search(_MATH_KEYWORDS, q) or num_count >= 2:
            difficulty = min(1.0, 0.4 + num_count / 8 + word_count / 400)
            return self._pack(
                Intent.MATH_REASONING,
                difficulty,
                f"math reasoning; numbers={num_count}, words={word_count}",
            )

        if re.search(_CREATIVE_KEYWORDS, q):
            return self._pack(
                Intent.CREATIVE,
                0.5,
                "matched creative keywords",
            )

        if re.search(_FACTUAL_KEYWORDS, q):
            return self._pack(
                Intent.FACTUAL,
                0.3,
                "matched factual keywords",
            )

        if re.search(_DIALOGUE_KEYWORDS, q) and word_count <= 6:
            return self._pack(
                Intent.DIALOGUE,
                0.05,
                "short dialogue phrase",
            )

        return self._pack(
            Intent.UNKNOWN,
            0.5,
            "no rule matched",
        )

    @staticmethod
    def _pack(intent: Intent, difficulty: float, reasoning: str) -> IntentPrediction:
        # Compute budget policy: log-scale with difficulty
        steps = int(50 + 1950 * difficulty)  # 50 -> 2000
        restarts = int(1 + 9 * difficulty)  # 1 -> 10
        traces = int(1 + 15 * difficulty)  # 1 -> 16 parallel traces
        return IntentPrediction(
            intent=intent,
            difficulty=difficulty,
            suggested_langevin_steps=steps,
            suggested_restarts=restarts,
            suggested_trace_count=traces,
            reasoning=reasoning,
        )
