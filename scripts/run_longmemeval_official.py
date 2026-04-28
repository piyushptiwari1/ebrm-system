"""Production runner for the OFFICIAL LongMemEval dataset.

Pipeline (v0.19, hybrid retrieval + optional LLM extraction + reranker):
1. Load ``longmemeval_*.json`` via ``benchmarks.datasets.load_longmemeval_official``.
2. Optionally distill each session into atomic memories with
   ``--extraction azure`` (Mem0-v3-style ADD-only, cached on disk).
3. Build the requested retriever:
     - ``dense``  — pure dense top-k over the chosen embedder.
     - ``bm25``   — pure BM25 over a per-episode index.
     - ``hybrid`` — RRF fusion of dense + BM25 (default).
   Optionally wrap with ``--reranker bge`` to add a cross-encoder pass.
4. Feed retrieved turns + question to ``AzureOpenAIReader``.
5. Grade with ``AzureOpenAIJudge`` (abstention via deterministic detector).
6. Write a stable JSON with full metadata to ``benchmarks/results/``.

Example:
    AZURE_OPENAI_API_KEY=... \\
    python scripts/run_longmemeval_official.py \\
        --dataset data/longmemeval/longmemeval_oracle.json \\
        --embedder azure --retriever hybrid --reranker bge \\
        --top-k 5 \\
        --out benchmarks/results/longmemeval-oracle-v0.18.0.json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.datasets import (  # noqa: E402
    OfficialEpisode,
    episodes_iter_question_types,
    load_longmemeval_official,
)
from benchmarks.embedders.hash import HashEmbedder  # noqa: E402
from benchmarks.entity import EntityReranker  # noqa: E402
from benchmarks.extraction import (  # noqa: E402
    ExtractedMemory,
    MemoryExtractor,
    augment_episode_with_memories,
)
from benchmarks.fusion import LLMFusionReranker  # noqa: E402
from benchmarks.retrieval import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    NeighborExpander,
    Retriever,
    RRFRetriever,
    ScoredTurn,
)
from benchmarks.router import classify_question, top_k_for  # noqa: E402
from benchmarks.temporal import TemporalReranker  # noqa: E402

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


def _build_retriever(
    *,
    retriever_name: str,
    embedder_name: str,
    cache_dir: Path,
    rrf_k: int,
    per_retriever_k: int,
) -> Retriever:
    if retriever_name == "dense":
        return DenseRetriever(_build_embedder(embedder_name, cache_dir=cache_dir))
    if retriever_name == "bm25":
        return BM25Retriever()
    if retriever_name == "hybrid":
        return RRFRetriever(
            [
                DenseRetriever(_build_embedder(embedder_name, cache_dir=cache_dir)),
                BM25Retriever(),
            ],
            rrf_k=rrf_k,
            per_retriever_k=per_retriever_k,
        )
    raise ValueError(f"unknown retriever {retriever_name!r}")


def _maybe_wrap_reranker(
    base: Retriever,
    *,
    reranker_name: str,
    candidate_k: int,
) -> Retriever:
    if reranker_name == "none":
        return base
    if reranker_name == "bge":
        from benchmarks.retrieval.reranker import CrossEncoderReranker

        model = os.environ.get("EBRM_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        return CrossEncoderReranker(base, model_name=model, candidate_k=candidate_k)
    raise ValueError(f"unknown reranker {reranker_name!r}")


def _build_extractor(name: str, *, cache_dir: Path) -> MemoryExtractor | None:
    if name == "none":
        return None
    if name == "azure":
        from benchmarks.extraction import AzureLLMExtractor

        return AzureLLMExtractor(cache_dir=cache_dir / "extract")
    raise ValueError(f"unknown extractor {name!r}")


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
        help="Embedder backend used by dense / hybrid retrievers (default: azure).",
    )
    p.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="Retriever (default: hybrid = RRF over dense + BM25).",
    )
    p.add_argument(
        "--reranker",
        choices=["none", "bge"],
        default="none",
        help="Optional cross-encoder reranker on top of the retriever (default: none).",
    )
    p.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant used by hybrid (default: 60, per Cormack et al. 2009).",
    )
    p.add_argument(
        "--per-retriever-k",
        type=int,
        default=20,
        help="How many candidates each base retriever returns inside hybrid.",
    )
    p.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=20,
        help="Number of candidates fed to the cross-encoder reranker.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--aggregation-cot",
        action="store_true",
        help=(
            "For aggregation questions, use a list-then-count CoT reader "
            "prompt that enumerates items before reporting the total. v0.28 "
            "tightens the gate to question_type=='multi-session' (was "
            "leaking onto temporal in v0.25)."
        ),
    )
    p.add_argument(
        "--temporal-ordering-cot",
        action="store_true",
        help=(
            "v0.28: For temporal-reasoning questions that ask 'which "
            "happened first / most recently / before / after', use a "
            "structured CoT reader (CANDIDATES / ORDERED / ANSWER) that "
            "extracts session_dates per candidate and orders them "
            "deterministically — defeats the v0.24 reader bug where it "
            "did correct date analysis then concluded the wrong order."
        ),
    )
    p.add_argument(
        "--per-type-top-k",
        action="store_true",
        help=(
            "Route top_k by question class: aggregation=max(top_k,20), "
            "temporal=min(top_k,5), other=top_k. Improves multi-session "
            "counting without hurting temporal chronology (v0.24)."
        ),
    )
    p.add_argument(
        "--reader-n-samples",
        type=int,
        default=1,
        help=(
            "v0.29: self-consistency reader. n>1 samples the reader N times "
            "at sc_temperature in a single API call and majority-votes on "
            "the final answer. Default 1 (off). 3 is the recommended setting."
        ),
    )
    p.add_argument(
        "--reader-sc-temperature",
        type=float,
        default=0.5,
        help="Sampling temperature when --reader-n-samples > 1 (default 0.5).",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap number of episodes (smoke testing). Default = all.",
    )
    p.add_argument(
        "--temporal-rerank",
        action="store_true",
        help="Apply temporal reranker (combine semantic + recency to question_date).",
    )
    p.add_argument(
        "--temporal-alpha",
        type=float,
        default=0.3,
        help="Weight on the temporal score (default 0.3).",
    )
    p.add_argument(
        "--temporal-decay-days",
        type=float,
        default=30.0,
        help="Recency decay scale in days (default 30).",
    )
    p.add_argument(
        "--entity-rerank",
        action="store_true",
        help="Apply entity reranker (boost turns mentioning question entities).",
    )
    p.add_argument(
        "--entity-alpha",
        type=float,
        default=0.25,
        help="Weight on the entity-match score (default 0.25).",
    )
    p.add_argument(
        "--fusion-rerank",
        action="store_true",
        help=(
            "Apply LLM-based multi-signal fusion reranker (gpt-4o-mini ranks "
            "top-N candidates jointly considering semantic + temporal + entity)."
        ),
    )
    p.add_argument(
        "--fusion-candidate-k",
        type=int,
        default=20,
        help="Number of candidates fed to the fusion reranker (default 20).",
    )
    p.add_argument(
        "--neighbor-window",
        type=int,
        default=0,
        help=(
            "After retrieval, also include +/-N same-session neighbors of "
            "each hit (cheap recall boost; default: 0 = off)."
        ),
    )
    p.add_argument(
        "--extraction",
        choices=["none", "azure"],
        default="none",
        help=(
            "Distill sessions into atomic memories before retrieval. "
            "`azure` uses gpt-4o-mini (cached on disk). Default: none."
        ),
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

    print(
        f"[retriever] {args.retriever}  embedder={args.embedder}  reranker={args.reranker}",
        flush=True,
    )
    base_retriever = _build_retriever(
        retriever_name=args.retriever,
        embedder_name=args.embedder,
        cache_dir=cache_dir,
        rrf_k=args.rrf_k,
        per_retriever_k=args.per_retriever_k,
    )
    retriever = _maybe_wrap_reranker(
        base_retriever,
        reranker_name=args.reranker,
        candidate_k=args.rerank_candidate_k,
    )
    if args.entity_rerank:
        retriever = EntityReranker(
            retriever, alpha=args.entity_alpha, candidate_k=args.rerank_candidate_k
        )
        print(f"[entity] alpha={args.entity_alpha}", flush=True)
    if args.temporal_rerank:
        retriever = TemporalReranker(
            retriever,
            alpha=args.temporal_alpha,
            decay_days=args.temporal_decay_days,
            candidate_k=args.rerank_candidate_k,
        )
        print(
            f"[temporal] alpha={args.temporal_alpha} decay={args.temporal_decay_days}d",
            flush=True,
        )
    if args.fusion_rerank:
        retriever = LLMFusionReranker(
            base=retriever,
            candidate_k=args.fusion_candidate_k,
            cache_dir=cache_dir / "fusion",
        )
        print(
            f"[fusion] candidate_k={args.fusion_candidate_k}",
            flush=True,
        )
    if args.neighbor_window > 0:
        retriever = NeighborExpander(retriever, window=args.neighbor_window)
        print(f"[neighbors] window={args.neighbor_window}", flush=True)
    extractor = _build_extractor(args.extraction, cache_dir=cache_dir)
    if extractor is not None:
        print(f"[extractor] {extractor.name} (additive corpus)", flush=True)

    reader = None
    if args.reader == "azure":
        from benchmarks.reader.azure_llm import AzureOpenAIReader

        reader = AzureOpenAIReader(
            aggregation_cot=args.aggregation_cot,
            temporal_ordering_cot=args.temporal_ordering_cot,
            n_samples=args.reader_n_samples,
            sc_temperature=args.reader_sc_temperature,
        )

    judge = None
    if args.judge == "azure":
        from benchmarks.judges.azure_llm import AzureOpenAIJudge

        judge = AzureOpenAIJudge(cache_dir=str(cache_dir / "judge"))

    per_q: list[dict] = []
    t0 = time.time()
    extracted_counts: list[int] = []
    for i, ep in enumerate(episodes):
        retrieval_episode = ep
        if extractor is not None:
            memories: list[ExtractedMemory] = extractor.extract(ep)
            extracted_counts.append(len(memories))
            if memories:
                retrieval_episode = augment_episode_with_memories(ep, memories)
        if args.per_type_top_k:
            tag = classify_question(ep.question, ep.question_type)
            this_top_k = top_k_for(tag, default=args.top_k)
        else:
            this_top_k = args.top_k
        retrieved_scored: list[ScoredTurn] = retriever.retrieve(retrieval_episode, top_k=this_top_k)
        retrieved = [st.turn for st in retrieved_scored]
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
                "retrieval_scores": [round(st.score, 4) for st in retrieved_scored],
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
            "embedder": args.embedder,
            "retriever": retriever.name,
            "reader": getattr(reader, "name", args.reader),
            "judge": getattr(judge, "name", args.judge),
            "top_k": args.top_k,
            "per_type_top_k": args.per_type_top_k,
            "aggregation_cot": args.aggregation_cot,
            "temporal_ordering_cot": args.temporal_ordering_cot,
            "reader_n_samples": args.reader_n_samples,
            "reader_sc_temperature": args.reader_sc_temperature,
            "rrf_k": args.rrf_k,
            "per_retriever_k": args.per_retriever_k,
            "rerank_candidate_k": args.rerank_candidate_k,
            "max_episodes": args.max_episodes,
            "extraction": args.extraction,
            "extractor": getattr(extractor, "name", args.extraction),
            "extraction_mode": "additive" if args.extraction != "none" else "off",
            "neighbor_window": args.neighbor_window,
            "temporal_rerank": args.temporal_rerank,
            "temporal_alpha": args.temporal_alpha,
            "temporal_decay_days": args.temporal_decay_days,
            "entity_rerank": args.entity_rerank,
            "entity_alpha": args.entity_alpha,
            "fusion_rerank": args.fusion_rerank,
            "fusion_candidate_k": args.fusion_candidate_k,
            "avg_memories_per_episode": (
                round(sum(extracted_counts) / len(extracted_counts), 2)
                if extracted_counts
                else None
            ),
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
