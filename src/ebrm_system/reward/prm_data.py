"""Free Process-Reward-Model (PRM) training data from existing reasoning traces.

Implements the Athena-PRM / ThinkPRM trick: when a *weak* policy (e.g. greedy
decode) and a *strong* policy (e.g. 32-way self-consistency) **agree** on a
candidate's correctness, that agreement is a reliable pseudo-label for
process supervision — no human annotation, no Monte-Carlo rollouts.

References:
    - "The Lessons of Developing Process Reward Models in Mathematical
      Reasoning" (ACL Findings 2025, arXiv:2501.07301).
    - Athena-PRM: "Enhancing Multimodal Reasoning with Data-efficient
      Process Reward Models" (OpenReview YyCgWQFGtL, AMD ROCm blog 2026-01).

This module turns every successful invocation of
:meth:`HierarchicalLatentReasoner.solve` into a JSONL line suitable for
fine-tuning a generative PRM (e.g. ThinkPRM) without any extra inference.

Pure-Python, deterministic, no torch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebrm_system.core import ReasoningResult


@dataclass(frozen=True)
class PRMRecord:
    """One PRM training example derived from a reasoning trace.

    A record carries pseudo-labels at two levels of confidence:

    * ``strong_label`` — the majority vote's chosen answer (highest-quality
      signal we have without ground truth).
    * ``agreement`` — True iff the candidate's answer equals the strong
      label. These are the records ThinkPRM/Athena-PRM treats as reliable.
    """

    question: str
    candidate_answer: str
    strong_label: str
    """The aggregated answer (self-consistency winner)."""
    agreement: bool
    """True iff candidate matches the strong label."""
    energy: float
    verified: bool
    """Whether this candidate passed the full verifier chain."""
    verifier_results: tuple[dict[str, object], ...] = field(default_factory=tuple)
    intent: str = ""
    difficulty: float = 0.0
    timestamp: float = 0.0


def make_records(
    question: str,
    result: ReasoningResult,
) -> list[PRMRecord]:
    """Materialize one PRMRecord per trace in ``result``.

    The ``strong_label`` is the voted answer. ``agreement`` flags the subset
    of traces that match it — those are the high-confidence positives that
    a downstream PRM trainer should up-weight.

    No filtering is done here; the caller decides how to balance the
    dataset (typical recipe: keep all agreement=True plus a stratified
    sample of agreement=False as negatives).
    """
    strong = str(result.answer)
    now = time.time()
    out: list[PRMRecord] = []
    for trace in result.traces:
        verifier_dump = tuple(
            {
                "verifier": vr.verifier,
                "verified": bool(vr.verified),
                "confidence": float(vr.confidence),
                "reason": vr.reason,
            }
            for vr in trace.verifier_results
        )
        out.append(
            PRMRecord(
                question=question,
                candidate_answer=trace.answer,
                strong_label=strong,
                agreement=(trace.answer == strong),
                energy=float(trace.energy),
                verified=bool(trace.verified),
                verifier_results=verifier_dump,
                intent=str(result.intent.intent.value),
                difficulty=float(result.intent.difficulty),
                timestamp=now,
            )
        )
    return out


def write_jsonl(records: Iterable[PRMRecord], path: str | Path) -> int:
    """Append PRM records as JSONL to ``path``.

    The file is opened in append mode so multiple solve() calls can dump
    into the same dataset. Returns the number of records written.

    Parent directories are created as needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")
            n += 1
    return n


__all__ = ["PRMRecord", "make_records", "write_jsonl"]
