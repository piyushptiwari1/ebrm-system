"""Self-consistency voting across parallel reasoning traces.

Given K candidate answers (possibly with confidence / energy scores), return
the consensus answer plus agreement metrics. Supports:

- Exact majority vote (discrete answers)
- Numerical bucket vote (continuous answers, tolerance-based)
- Weighted vote (by confidence or 1/energy)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """A single reasoning trace output."""

    answer: object
    confidence: float = 1.0
    energy: float | None = None
    trace_id: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


@dataclass(frozen=True)
class VoteResult:
    """Consensus result."""

    answer: object
    support: int  # how many candidates voted for this answer
    total: int  # total candidates
    agreement: float  # support / total
    weighted_score: float  # sum of weights for winning bucket
    runner_up: object | None = None
    runner_up_support: int = 0
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class _Bucket:
    """Internal voting bucket."""

    representative: object
    support: int = 0
    weighted: float = 0.0


class SelfConsistencyVoter:
    """Aggregate multiple reasoning traces into a single consensus answer.

    Args:
        numerical: if True, bucket numeric answers by `tolerance`; else exact match
        tolerance: numeric bucket width (absolute)
        weight_by: 'uniform' | 'confidence' | 'inverse_energy'
    """

    def __init__(
        self,
        numerical: bool = False,
        tolerance: float = 1e-6,
        weight_by: str = "uniform",
    ) -> None:
        if weight_by not in {"uniform", "confidence", "inverse_energy"}:
            raise ValueError(f"invalid weight_by: {weight_by}")
        self.numerical = numerical
        self.tolerance = tolerance
        self.weight_by = weight_by

    def vote(self, candidates: Sequence[Candidate]) -> VoteResult:
        if not candidates:
            raise ValueError("cannot vote on empty candidate list")

        weights = [self._weight(c) for c in candidates]

        if self.numerical:
            buckets = self._bucket_numeric(candidates, weights)
        else:
            buckets = self._bucket_exact(candidates, weights)

        # Sort by (weighted_score desc, support desc)
        ranked = sorted(
            buckets.items(),
            key=lambda kv: (kv[1].weighted, kv[1].support),
            reverse=True,
        )

        _top_key, top = ranked[0]
        runner_up_answer: object | None = None
        runner_up_support = 0
        if len(ranked) > 1:
            runner_up_answer = ranked[1][1].representative
            runner_up_support = ranked[1][1].support

        total = len(candidates)
        return VoteResult(
            answer=top.representative,
            support=top.support,
            total=total,
            agreement=top.support / total,
            weighted_score=top.weighted,
            runner_up=runner_up_answer,
            runner_up_support=runner_up_support,
            details={"bucket_count": len(buckets), "weight_by": self.weight_by},
        )

    def _weight(self, c: Candidate) -> float:
        if self.weight_by == "uniform":
            return 1.0
        if self.weight_by == "confidence":
            return max(c.confidence, 1e-9)
        if self.weight_by == "inverse_energy":
            if c.energy is None:
                return 1.0
            return 1.0 / (1.0 + math.exp(c.energy))  # sigmoid of -energy
        return 1.0

    def _bucket_exact(
        self, candidates: Sequence[Candidate], weights: list[float]
    ) -> dict[object, _Bucket]:
        buckets: dict[object, _Bucket] = {}
        for c, w in zip(candidates, weights, strict=True):
            key = c.answer
            bucket = buckets.get(key)
            if bucket is None:
                bucket = _Bucket(representative=c.answer)
                buckets[key] = bucket
            bucket.support += 1
            bucket.weighted += w
        return buckets

    def _bucket_numeric(
        self, candidates: Sequence[Candidate], weights: list[float]
    ) -> dict[object, _Bucket]:
        buckets: dict[object, _Bucket] = {}
        for c, w in zip(candidates, weights, strict=True):
            key: object
            try:
                val = float(c.answer)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # Fall back to exact bucketing for non-numeric
                key = c.answer
            else:
                # Round to nearest tolerance bucket
                key = val if self.tolerance <= 0 else round(val / self.tolerance) * self.tolerance
            bucket = buckets.get(key)
            if bucket is None:
                bucket = _Bucket(representative=c.answer)
                buckets[key] = bucket
            bucket.support += 1
            bucket.weighted += w
        return buckets
