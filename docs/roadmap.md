# EBRM-System Roadmap

This page maps every shipped release of `ebrm-system` to the research papers
that motivated it. Use it as the single onboarding-grade reference for "why
does this module exist?"

Each entry follows the format:

> **vX.Y.0** — *short title*
> Module(s): `ebrm_system.<path>`
> Reference(s): paper title (arXiv ID).
> One-line design note.

---

## Foundational releases

### v0.1.0 — Project skeleton
Module(s): `ebrm_system.__init__`
Initial scaffold: typed Python project, hatchling build, Apache-2.0 license.

### v0.3.0 — Verifier plane, inference modules, reward, CLI
Module(s): `ebrm_system.verifiers`, `ebrm_system.inference`,
`ebrm_system.reward`, `ebrm_system.cli`
The four pillars of EBRM as separable planes. Each verifier is a hard,
side-effect-free check; the inference plane generates candidates; the reward
plane provides PRM signals; the CLI exposes the lot.

### v0.4.0 — `ebrm_system.core` stable surface
Module(s): `ebrm_system.core.HierarchicalLatentReasoner`
The orchestrator that wires `encoder → generate → decode → verify → vote`
into a single deterministic pipeline. This is the public API consumed by all
later releases.

---

## Sampling & memory releases

### v0.5.0 — Adaptive halt + bounded `LatentIndex`
Module(s): `ebrm_system.inference.halt`, `ebrm_system.reward.qjl_index`
Reference(s): adaptive computation time literature.
Stops Langevin sampling once energy plateaus; evicts cold latents from the
warm-start index to keep memory bounded under long-running deployments.

### v0.6.0 — Verification-and-refinement, difficulty-adaptive compute, PRM data
Module(s): `ebrm_system.core.refinement`, `ebrm_system.core.compute_profile`,
`ebrm_system.reward.prm_data`
Reference(s): "Winning Gold at IMO 2025 with a Model-Agnostic
Verification-and-Refinement Pipeline" (arXiv:2507.15855).
On verifier rejection, augment the question with the rejection reason and
re-run the pipeline. The same mechanism that pushed silver-medal LLMs to
IMO-2025 gold.

### v0.7.0 — Diverse Verifier Tree Search + pluggable PRM verifier
Module(s): `ebrm_system.inference.diverse_selector`,
`ebrm_system.verifiers.prm`
Reference(s): DVTS (Diverse Verifier Tree Search) and PRM literature.
Cluster-prune the candidate pool before voting so that voting cannot be
hijacked by near-duplicate samples; PRM then scores each candidate.

### v0.8.0 — Coconut-style latent recursion
Module(s): `ebrm_system.inference.latent_recursion`
Reference(s): Coconut (arXiv:2412.06769), recurrent-depth transformers.
Iteratively re-feed the encoder output through a small fixed transform
before sampling — the latent equivalent of chain-of-thought.

### v0.9.0 — Three-tier latent memory
Module(s): `ebrm_system.memory.tiered`
Reference(s): cognitive-architecture memory hierarchies (working / episodic
/ semantic).
A bounded working buffer that promotes frequently retrieved items to
episodic, and consolidates stable episodics to semantic.

### v0.10.0 — ReST-MCTS\* re-ranking
Module(s): `ebrm_system.inference.mcts`
Reference(s): ReST-MCTS\* (arXiv:2406.03816), AlphaProof.
UCB1-driven Monte Carlo Tree Search over candidate latents using a value
function (default: energy-rank, override-able). Re-orders the voting pool.

### v0.11.0 — Episodic write-back
Module(s): `ebrm_system.core.HierarchicalLatentReasoner._maybe_write_back`
Reference(s): Letta closed-loop memory.
After a successful solve, write the winning latent back to the index so the
next call can warm-start from it.

### v0.12.0 — PRM-guided refinement
Module(s): `ebrm_system.core.refinement` + `ebrm_system.verifiers.prm`
Closes the loop between v0.6 (refinement) and v0.7 (PRM): use PRM scores to
decide *which* rejection reasons to surface in the augmented prompt.

---

## Evaluation & reproducibility releases

### v0.13.0 — LongMemEval-style benchmark harness
Module(s): `benchmarks.longmemeval`
Reference(s): LongMemEval (long-context memory evaluation).
End-to-end harness over the five canonical LongMemEval question types
(single-session-user, single-session-assistant, multi-session,
temporal-reasoning, knowledge-update). Ships a deterministic synthetic
generator so the harness is runnable without the real dataset.

### v0.14.0 — Real LongMemEval pipeline
Module(s): `benchmarks.longmemeval.load_longmemeval_jsonl`,
`benchmarks.longmemeval.write_results_json`, `scripts/run_longmemeval.py`
Strict JSONL adapter for the official dataset format, stable JSON results
schema, production CLI runner with full reproducibility metadata, and the
**first published baseline** at
`benchmarks/results/longmemeval-v0.14.0.json` (35% accuracy on 200
synthetic episodes with the hash-projection embedder). This is the floor
that future trained projectors must beat.

### v0.15.0 — MCTS-seeded refinement
Module(s): `ebrm_system.core.refinement.RefinementConfig.use_mcts_seed`
Reference(s): combines arXiv:2507.15855 (refinement) with arXiv:2406.03816
(ReST-MCTS\*).
When opted in, refinement rounds beyond round 0 seed the Langevin sampler
from the MCTS top-1 latent of the accumulated trace pool, focusing extra
compute on the most promising region of latent space identified so far.

### v0.16.0 — This roadmap page
Module(s): `docs/roadmap.md`
Onboarding-grade single-page mapping of every shipped feature to the
research paper that motivated it.

---

## Looking ahead

Future releases will publish:

- **Trained projectors** — replace the hash-projection embedder with a small
  trained projector and publish a new baseline that beats 35%.
- **Real LongMemEval results** — once the official dataset is downloaded,
  rerun `scripts/run_longmemeval.py --jsonl <path>` and publish a
  `longmemeval-vX.Y.0.json` against the real questions.
- **PRM training pipeline** — turn `ebrm_system.reward.prm_data` into a full
  training loop with a published checkpoint.

Each future release will follow the same protocol: real artifacts, real
metadata, no placeholders. See the
[CHANGELOG](https://github.com/piyushptiwari1/ebrm-system/blob/main/CHANGELOG.md)
for the authoritative version history.
