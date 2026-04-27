#!/usr/bin/env python3
"""Run the LongMemEval-style harness against TieredMemory and persist results.

Usage:

    # Synthetic dataset (default 200 episodes, deterministic):
    python scripts/run_longmemeval.py

    # Real LongMemEval JSONL (download separately, gated dataset):
    python scripts/run_longmemeval.py --jsonl path/to/longmemeval.jsonl

    # Custom output and memory dim:
    python scripts/run_longmemeval.py --num 500 --dim 128 \
        --out benchmarks/results/longmemeval-custom.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a plain script from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.longmemeval import (  # noqa: E402
    default_memory,
    load_longmemeval_jsonl,
    run_longmemeval,
    synth_longmemeval,
    write_results_json,
)

from ebrm_system import __version__ as ebrm_version  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Path to a real LongMemEval JSONL file. If omitted, uses the "
        "deterministic synthetic generator.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=200,
        help="Synthetic episode count (ignored when --jsonl is set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Synthetic generator seed (ignored when --jsonl is set).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Latent / embedding dimensionality.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k passed to memory.search per question.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "results" / f"longmemeval-v{ebrm_version}.json",
        help="Where to write the results JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.jsonl is not None:
        episodes = load_longmemeval_jsonl(args.jsonl)
        source = f"jsonl:{args.jsonl}"
    else:
        episodes = synth_longmemeval(seed=args.seed, num_episodes=args.num)
        source = f"synthetic(seed={args.seed},num={args.num})"

    memory = default_memory(in_dim=args.dim)

    print(f"Running LongMemEval: {len(episodes)} episodes, dim={args.dim}, source={source}")
    result = run_longmemeval(episodes, memory, embed_dim=args.dim, top_k=args.top_k)

    metadata = {
        "ebrm_system_version": ebrm_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source": source,
        "embed_dim": args.dim,
        "top_k": args.top_k,
        "memory_config": {
            "in_dim": memory.config.in_dim,
            "bits": memory.config.bits,
            "working_max_size": memory.config.working.max_size,
            "episodic_max_size": memory.config.episodic.max_size,
            "semantic_max_size": memory.config.semantic.max_size,
        },
        "memory_stats_after_run": memory.stats(),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_results_json(result, args.out, metadata=metadata)

    print()
    print(f"Total: {result.total} | Correct: {result.correct} | Accuracy: {result.accuracy:.3f}")
    print("Per-type accuracy:")
    for qt, acc in result.accuracy_by_type.items():
        n = result.per_type_counts[qt]
        print(f"  {qt:32s} {acc:.3f}  ({n} episodes)")
    print()
    print(f"Memory stats: {memory.stats()}")
    print(f"Wrote results to {args.out}")
    print()
    print(
        json.dumps(
            {"summary": {"accuracy": result.accuracy, "by_type": result.accuracy_by_type}}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
