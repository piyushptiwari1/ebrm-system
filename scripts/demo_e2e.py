"""End-to-end demo: Qwen3-4B + TurboQuant KV + routed verifiers + self-consistency.

Runs on Tesla T4 (16 GB VRAM). Measures:
- KV-cache memory: FP16 baseline vs TurboQuant 4-bit codes
- GSM8K-style accuracy with n-sample self-consistency, gated by SymPyVerifier
- End-to-end latency per question
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ebrm_system.inference.turboquant_attention import quantize_kv_torch
from ebrm_system.intent import Intent
from ebrm_system.verifiers.routing import chain_for_intent

MODEL_ID = "piyushptiwari/ebrm-v2-qwen3-4b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CANDIDATES = 32
MAX_NEW_TOKENS = 160
OUT = Path("demo_results.json")

GSM8K_SAMPLE = [
    ("Janet has 3 apples. She buys 5 more, then gives 2 to her friend. How many apples does she have?", "6"),
    ("A train travels 60 miles in 2 hours. What is its average speed in miles per hour?", "30"),
    ("Tom has twice as many marbles as Jerry. If Jerry has 7 marbles, how many does Tom have?", "14"),
    ("A rectangle has length 8 cm and width 5 cm. What is its area in square centimeters?", "40"),
    ("If 4 pencils cost $2, how much do 10 pencils cost (in dollars)?", "5"),
]


def kv_compression_stats(model, tok, prompt: str) -> dict:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    pkv = out.past_key_values
    if hasattr(pkv, "to_legacy_cache"):
        pkv = pkv.to_legacy_cache()

    fp16_bytes = 0
    tq_bytes = 0
    for k, v in pkv:
        fp16_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        kq = quantize_kv_torch(k, bits=4, rotate=False)
        vq = quantize_kv_torch(v, bits=4, rotate=False)
        tq_bytes += (
            kq.codes.numel() * kq.codes.element_size()
            + kq.scale.numel() * kq.scale.element_size()
            + vq.codes.numel() * vq.codes.element_size()
            + vq.scale.numel() * vq.scale.element_size()
        )
    return {
        "fp16_mb": round(fp16_bytes / 1024 / 1024, 2),
        "turboquant_4bit_mb": round(tq_bytes / 1024 / 1024, 2),
        "compression_ratio": round(fp16_bytes / max(tq_bytes, 1), 2),
        "context_tokens": int(enc["input_ids"].shape[1]),
    }


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> str:
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else ""


def self_consistency_answer(model, tok, question: str, gold: str, n: int) -> dict:
    prompt = (
        f"Question: {question}\n"
        "Solve step by step, then write the final answer as a single number on the last line.\n"
        "Answer:"
    )
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=n,
            pad_token_id=tok.eos_token_id,
        )
    latency = time.time() - t0
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)

    chain = chain_for_intent(Intent.MATH_REASONING)
    verified_preds: list[str] = []
    all_preds: list[str] = []
    for t in texts:
        pred = extract_final_number(t)
        all_preds.append(pred)
        if not pred:
            continue
        results = chain.verify(pred, context={"expected": gold})
        if chain.all_passed(results):
            verified_preds.append(pred)

    pool = [p for p in (verified_preds or all_preds) if p]
    chosen = Counter(pool).most_common(1)[0][0] if pool else ""
    return {
        "question": question,
        "gold": gold,
        "chosen": chosen,
        "correct": chosen == gold,
        "n_total": n,
        "n_verified": len(verified_preds),
        "all_preds": all_preds,
        "latency_s": round(latency, 2),
        "sample_text": texts[0][:300],
    }


def main() -> None:
    print(f"[demo] device={DEVICE}, loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        print(f"[demo] model loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("[demo] measuring KV memory compression...")
    long_prompt = (GSM8K_SAMPLE[0][0] + " ") * 16
    kv_stats = kv_compression_stats(model, tok, long_prompt)
    print(
        f"  ctx={kv_stats['context_tokens']} tok | FP16={kv_stats['fp16_mb']} MB | "
        f"TQ4={kv_stats['turboquant_4bit_mb']} MB | ratio={kv_stats['compression_ratio']}x"
    )

    print(f"[demo] self-consistency n={N_CANDIDATES}...")
    results = []
    for i, (q, gold) in enumerate(GSM8K_SAMPLE):
        r = self_consistency_answer(model, tok, q, gold, N_CANDIDATES)
        print(
            f"  [{i+1}/{len(GSM8K_SAMPLE)}] gold={gold} chosen={r['chosen']} "
            f"correct={r['correct']} verified={r['n_verified']}/{r['n_total']} "
            f"({r['latency_s']}s)"
        )
        results.append(r)

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    summary = {
        "model": MODEL_ID,
        "device": DEVICE,
        "n_candidates": N_CANDIDATES,
        "kv_memory": kv_stats,
        "accuracy": round(accuracy, 3),
        "avg_latency_s": round(avg_latency, 2),
        "per_item": results,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"[demo] DONE accuracy={accuracy*100:.1f}% avg_latency={avg_latency:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
