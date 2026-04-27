"""Production runner for the OFFICIAL LongMemEval dataset.

Pipeline (v0.17, dense-only):
1. Load ``longmemeval_*.json`` via ``benchmarks.datasets.load_longmemeval_official``.
2. For each episode, embed every turn (cached) and the question.
3. Cosine top-K retrieval over turns.
4. Feed retrieved turns + question to ``AzureOpenAIReader``.
5. Grade with ``AzureOpenAIJudge`` (abstention via deterministic detector).
6. Write a stable JSON with full metadata to ``benchmarks/results/``.

Subsequent releases (v0.18+) add BM25, reranking, extraction, time-aware
filtering, etc., but keep this CLI shape.

Example:
    AZURE_OPENAI_API_KEY=... \\
    python scripts/run_longmemeval_official.py \\
        --dataset data/longmemeval/longmemeval_oracle.json \\
        --embedder azure --top-k 5 \\
        --out benchmarks/results/longmemeval-oracle-v0.17.0.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasets import (  # noqa: E402
    OfficialEpisode,
    OfficialTurn,
    episodes_iter_question_types,
    load_longmemeval_official,
)
from benchmarks.embedders.hash import HashEmbedder  # noqa: E402

from ebrm_system import __version__ as ebrm_version  # noqa: E402


def _build_embedder(name: str, *, cache_dir: Path):
    if name == "hash":
        return HashEmbedder(dim=384)
    if name == "bge":
        from benchmarks.embedders.sentence_transformer import (
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(
            model_name=os.environ.get("EBRM_BGE_MODEL", "BAAI/bge-large-en-v1.5")
        )
    if name == "azure":
        from benchmarks.embedders.azure_openai import AzureOpenAIEmbedder

        return AzureOpenAIEmbedder(cache_dir=cache_dir / "embed")
    raise ValueError(f"unknown embedder {name!r}")


def _topk_turns(
    embedder,
    episode: OfficialEpisode,
    *,
    top_k: int,
) -> list[OfficialTurn]:
    if not episode.turns:
        return []
    turn_texts = [t.content for t in episode.turns]
    turn_vecs = embedder.embed(turn_texts)
    q_vec = embedder.embed([episode.question])[0]
    # Vectors are L2-normalised — cosine == dot.
    scores = turn_vecs @ q_vec
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [episode.turns[int(i)] for i in top_idx]


def _aggregate_summary(per_q: list[dict], episodes: list[OfficialEpisode]) -> dict:
    total = len(per_q)
    correct = sum(1 for r in per_q if r["correct"])
    by_type: dict[str, dict[str, int]] = {}
    for r, ep in zip(per_q, episodes, strict=True):
        bucket = by_type.setdefault(
            ep.question_type, {"correct": 0, "total": 0, "abs_correct": 0, "abs_total": 0}
        )
        bucket["total"] += 1
        if r["correct"]:
            bucket["correct"] += 1
        if ep.is_abstention:
            bucket["abs_total"] += 1
            if r["correct"]:
                bucket["abs_correct"] += 1
    accuracy_by_type = {qt: round(b["correct"] / b["total"], 4) for qt, b in by_type.items()}
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_by_type": accuracy_by_type,
        "per_type_counts": {qt: b["total"] for qt, b in by_type.items()},
        "abstention_total": sum(b["abs_total"] for b in by_type.values()),
        "abstention_correct": sum(b["abs_correct"] for b in by_type.values()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset", required=True, help="Path to longmemeval_*.json")
    p.add_argument(
        "--embedder",
        choices=["hash", "bge", "azure"],
        default="azure",
        help="Embedder backend (default: azure).",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap number of episodes (smoke testing). Default = all.",
    )
    p.add_argument(
        "--reader",
        choices=["azure", "none"],
        default="azure",
        help="LLM reader backend (default: azure).",
    )
    p.add_argument(
        "--judge",
        choices=["azure", "none"],
        default="azure",
        help="LLM judge backend (default: azure).",
    )
    p.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "cache"),
        help="Disk cache for embeddings and judge calls (default: ./cache).",
    )
    p.add_argument(
        "--out",
        default=None,
        help=("Output JSON path. Default: benchmarks/results/longmemeval-{split}-v{version}.json"),
    )
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset}", flush=True)
    episodes = load_longmemeval_official(args.dataset)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    print(
        f"[load] {len(episodes)} episodes, types: {episodes_iter_question_types(episodes)}",
        flush=True,
    )

    print(f"[embed] backend={args.embedder}", flush=True)
    embedder = _build_embedder(args.embedder, cache_dir=cache_dir)

    reader = None
    if args.reader == "azure":
        from benchmarks.reader.azure_llm import AzureOpenAIReader

        reader = AzureOpenAIReader()

    judge = None
    if args.judge == "azure":
        from benchmarks.judges.azure_llm import AzureOpenAIJudge

        judge = AzureOpenAIJudge(cache_dir=str(cache_dir / "judge"))

    per_q: list[dict] = []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        retrieved = _topk_turns(embedder, ep, top_k=args.top_k)
        if reader is not None:
            pred = reader.read(ep, retrieved)
        else:
            pred = retrieved[0].content if retrieved else ""
        if judge is not None:
            verdict = judge.judge(
                question=ep.question,
                question_type=ep.question_type,
                gold=ep.answer,
                pred=pred,
                is_abstention=ep.is_abstention,
            )
            correct = verdict.correct
            judge_raw = verdict.raw
        else:
            # Naive substring match — only useful for offline smoke tests.
            correct = ep.answer.strip().lower() in pred.lower()
            judge_raw = "substring"
        per_q.append(
            {
                "question_id": ep.question_id,
                "question_type": ep.question_type,
                "is_abstention": ep.is_abstention,
                "gold": ep.answer,
                "pred": pred,
                "correct": correct,
                "judge_raw": judge_raw,
                "retrieved_session_ids": [t.session_id for t in retrieved],
                "retrieved_n": len(retrieved),
            }
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(episodes):
            elapsed = time.time() - t0
            running = sum(1 for r in per_q if r["correct"]) / len(per_q)
            print(
                f"[run] {i + 1}/{len(episodes)}  acc={running:.3f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = _aggregate_summary(per_q, episodes)
    elapsed_total = time.time() - t0

    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        split = Path(args.dataset).stem.replace("longmemeval_", "")
        out_path = (
            REPO_ROOT / "benchmarks" / "results" / f"longmemeval-{split}-v{ebrm_version}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        **summary,
        "details": per_q,
        "metadata": {
            "ebrm_system_version": ebrm_version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dataset": str(args.dataset),
            "embedder": getattr(embedder, "name", args.embedder),
            "embedder_dim": getattr(embedder, "dim", None),
            "reader": getattr(reader, "name", args.reader),
            "judge": getattr(judge, "name", args.judge),
            "top_k": args.top_k,
            "max_episodes": args.max_episodes,
            "elapsed_seconds": round(elapsed_total, 2),
            "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(
        f"\n=== Summary ===\n"
        f"Total: {summary['total']}  Correct: {summary['correct']}  "
        f"Accuracy: {summary['accuracy']}\n"
        f"By type: {summary['accuracy_by_type']}\n"
        f"Wrote: {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
