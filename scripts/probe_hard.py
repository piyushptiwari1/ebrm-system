"""Probe: GSM8K-Hard / adversarial reasoning items where frontier LLMs are reported to fail.

Sources for the adversarial set (paraphrased to avoid memorization):
- GSM-Hard (Gao et al. 2023): GSM8K with very large numbers
- GSM-Symbolic (Apple, 2024): templated GSM8K with red-herring distractors
- Trick-arithmetic items widely reported to break GPT-4 / Claude / Gemini

We run Qwen3-4B with two configurations and report side-by-side accuracy:

  A. baseline    : 1 sample, greedy
  B. ebrm-system : 32 samples, self-consistency majority vote (no SymPy gating
                   since these items have non-trivial answers; we just count votes)
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CANDIDATES = 32
MAX_NEW_TOKENS = 384  # hard problems need longer chains
OUT = Path("hard_results.json")

# (question, gold, source-tag) — paraphrased to avoid contamination
HARD_ITEMS = [
    # 1. GSM-Hard: very large numbers (LLMs often slip on the arithmetic)
    (
        "A factory produces 14,357 widgets per day. After 23 days of production, "
        "they ship 198,432 widgets to a buyer. How many widgets remain in stock?",
        str(14357 * 23 - 198432),  # 130,779
        "gsm-hard",
    ),
    # 2. Red herring (GSM-Symbolic style): irrelevant detail planted mid-problem
    (
        "Liam has 5 apples. He gives 2 to his sister, who is 7 years old and "
        "lives next door. He then buys 3 more apples and eats 1. "
        "How many apples does Liam have now?",
        "5",  # 5-2+3-1 = 5
        "red-herring",
    ),
    # 3. Order-of-operations trap
    (
        "A store sells pens in packs of 12 for $8 and pencils in packs of 20 for $5. "
        "If I buy 3 pen packs and 4 pencil packs, then return 1 pen pack for a refund, "
        "how much did I spend in total (in dollars)?",
        str(3 * 8 + 4 * 5 - 8),  # 36
        "ooo-trap",
    ),
    # 4. Negative-result sanity check (LLMs sometimes refuse to give negatives)
    (
        "A submarine is at depth 250 m below sea level. It rises 80 m, then dives "
        "another 410 m. What is its final depth in metres below sea level "
        "(use a positive number for depth below sea level)?",
        str(250 - 80 + 410),  # 580
        "sign",
    ),
    # 5. Apple "GSM-Symbolic" classic kiwi problem (reformulated)
    (
        "Oliver picks 44 pears on Friday and 58 pears on Saturday. On Sunday he "
        "picks twice as many pears as on Friday, but five of the pears on Sunday "
        "are smaller than average. How many pears does Oliver have in total?",
        str(44 + 58 + 2 * 44),  # 190 — Apple paper showed GPT-4o etc. subtract the 5
        "apple-symbolic",
    ),
    # 6. Compound percentage trap
    (
        "An item costs $200. It is discounted by 25%, then a 10% sales tax is "
        "applied to the discounted price. What is the final price in dollars?",
        str(round(200 * 0.75 * 1.10)),  # 165
        "percent",
    ),
    # 7. Counting with overlap (inclusion-exclusion)
    (
        "In a class of 40 students, 22 play soccer, 18 play basketball, and 7 play "
        "both. How many students play neither sport?",
        str(40 - (22 + 18 - 7)),  # 7
        "set",
    ),
    # 8. Multi-step rate problem
    (
        "Tap A fills a tank in 6 hours. Tap B fills the same tank in 4 hours. "
        "If both taps are open together, how many hours does it take to fill "
        "the tank? (Round to two decimal places.)",
        f"{1 / (1 / 6 + 1 / 4):.2f}",  # 2.40
        "rate",
    ),
]

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_final_number(text: str) -> str:
    nums = _NUM_RE.findall(text)
    if not nums:
        return ""
    n = nums[-1].replace(",", "")
    # normalize "5.00" -> "5", but keep "2.40"
    if "." in n:
        try:
            f = float(n)
            if f.is_integer():
                return str(int(f))
        except ValueError:
            pass
    return n


def normalize(answer: str) -> str:
    a = answer.replace(",", "").strip()
    if "." in a:
        try:
            f = float(a)
            if f.is_integer():
                return str(int(f))
            return f"{f:.2f}".rstrip("0").rstrip(".")
        except ValueError:
            pass
    return a


def generate(model, tok, prompt: str, n: int) -> tuple[list[str], float]:
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=(n > 1),
            temperature=0.7 if n > 1 else 1.0,
            top_p=0.95,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=n,
            pad_token_id=tok.eos_token_id,
        )
    latency = time.time() - t0
    gen = out[:, enc["input_ids"].shape[1] :]
    return tok.batch_decode(gen, skip_special_tokens=True), latency


def main() -> None:
    print(f"[hard] loading {MODEL_ID} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map=DEVICE)
    model.eval()

    results = []
    base_correct = 0
    ebrm_correct = 0

    for i, (q, gold, tag) in enumerate(HARD_ITEMS):
        prompt = (
            f"Question: {q}\n"
            "Solve step by step. End with a line of the form '#### <number>'.\n"
            "Answer:"
        )

        # A. baseline: 1 sample, low-temperature greedy-ish
        b_texts, b_lat = generate(model, tok, prompt, 1)
        b_pred = normalize(extract_final_number(b_texts[0]))
        b_ok = b_pred == normalize(gold)
        base_correct += int(b_ok)

        # B. ebrm: 32-sample self-consistency vote
        e_texts, e_lat = generate(model, tok, prompt, N_CANDIDATES)
        votes = [normalize(extract_final_number(t)) for t in e_texts]
        votes_nz = [v for v in votes if v]
        e_pred = Counter(votes_nz).most_common(1)[0][0] if votes_nz else ""
        e_ok = e_pred == normalize(gold)
        ebrm_correct += int(e_ok)

        print(
            f"[{i + 1}/{len(HARD_ITEMS)}] {tag:>14}  gold={gold:<10} "
            f"base={b_pred:<10} {'✓' if b_ok else '✗'}   "
            f"ebrm32={e_pred:<10} {'✓' if e_ok else '✗'}   "
            f"({b_lat:.1f}s / {e_lat:.1f}s)"
        )
        results.append(
            {
                "tag": tag,
                "question": q,
                "gold": gold,
                "baseline_pred": b_pred,
                "baseline_correct": b_ok,
                "baseline_latency_s": round(b_lat, 2),
                "ebrm_pred": e_pred,
                "ebrm_correct": e_ok,
                "ebrm_latency_s": round(e_lat, 2),
                "ebrm_vote_distribution": dict(Counter(votes_nz).most_common(5)),
            }
        )

    n = len(HARD_ITEMS)
    summary = {
        "model": MODEL_ID,
        "n_items": n,
        "n_candidates": N_CANDIDATES,
        "baseline_accuracy": round(base_correct / n, 3),
        "ebrm_accuracy": round(ebrm_correct / n, 3),
        "lift": round((ebrm_correct - base_correct) / n, 3),
        "per_item": results,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(
        f"\n[hard] DONE  baseline={base_correct}/{n}={summary['baseline_accuracy'] * 100:.0f}%  "
        f"ebrm32={ebrm_correct}/{n}={summary['ebrm_accuracy'] * 100:.0f}%  "
        f"lift={summary['lift'] * 100:+.0f} pts  ->  {OUT}"
    )


if __name__ == "__main__":
    main()
