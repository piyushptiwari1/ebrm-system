# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.17.0] - 2026-04-28

### Added — real semantics for LongMemEval (Phase 1 of 90%+ plan)

- **`benchmarks/embedders/`** package — pluggable `Embedder` Protocol with
  three production-grade implementations:
  - `HashEmbedder` (deterministic, 128-dim, no deps; the previous baseline)
  - `SentenceTransformerEmbedder` (default `BAAI/bge-large-en-v1.5`,
    1024-dim, lazy `sentence-transformers` import)
  - `AzureOpenAIEmbedder` (default `text-embedding-3-small`, 1536-dim,
    sha256-keyed disk cache, batched, exponential-backoff retry)
- **`benchmarks/datasets/longmemeval_official.py`** — strict loader for the
  official LongMemEval dataset (`OfficialEpisode` / `OfficialTurn` schema),
  covering all six question types including `single-session-preference` and
  the 30 abstention questions (`*_abs`).
- **`benchmarks/judges/azure_llm.py`** — `AzureOpenAIJudge` reproducing the
  paper's evaluation methodology: type-aware unified prompt, deterministic
  abstention short-circuit, sha256-keyed disk cache, deterministic verdicts.
- **`benchmarks/reader/azure_llm.py`** — `AzureOpenAIReader` that composes
  free-text answers from retrieved turns with explicit abstention guidance.
- **`scripts/run_longmemeval_official.py`** — full-stack runner for the
  official dataset with `--embedder {hash,bge,azure}`, `--reader`,
  `--judge`, `--top-k`, `--max-episodes`, `--cache-dir`. Emits a versioned
  results JSON with embedder name + dim, reader/judge IDs, per-type and
  abstention accuracy, elapsed seconds, and full provenance.
- **`pyproject.toml`** — new optional-deps group `embedders`
  (`openai>=1.50`, `sentence-transformers>=3.0`).
- **`tests/test_benchmarks_v17.py`** — 28 new tests (HashEmbedder,
  official loader, abstention detector, judge with mocked Azure).

### Notes

- **Measured LongMemEval accuracy (oracle, n=500)**: **50.8 %** overall
  (254/500 correct), up from the no-semantics 35 % baseline.
- Per-type breakdown: single-session-user 85.7 %, single-session-assistant
  83.9 %, knowledge-update 66.7 %, temporal-reasoning 37.6 %,
  multi-session 33.1 %, single-session-preference 3.3 % (judge is strict
  on subjective preferences — addressed in v0.21).
- Abstention bucket: 26/30 = **86.7 %** correct (the model knows what it
  doesn't know).
- Configuration: `text-embedding-3-small` (1536-dim) + `gpt-4o-mini`
  reader + `gpt-4o-mini` LLM-judge, top-k=5, full 500-episode oracle
  split. Wall time 881.9 s (≈14.7 min, fully cached on resume).
- Results JSON checked in at
  [benchmarks/results/longmemeval-oracle-v0.17.0.json](benchmarks/results/longmemeval-oracle-v0.17.0.json).
- This release replaces the no-semantics 35 % floor with real embeddings,
  a real reader, and a real judge — the foundation for v0.18 (hybrid
  retrieval), v0.19 (LLM extraction), v0.20 (temporal/entity), v0.21
  (fusion + abstention), v0.22 (docs).
- 322 tests pass, ruff/mypy clean, 95 % coverage.

## [0.16.0] - 2026-04-27

### Added

- **`docs/roadmap.md`** — single-page onboarding-grade reference mapping
  every shipped release (v0.1 → v0.16) to the research papers that
  motivated it. Linked from the MkDocs site nav.

### Notes

- Documentation-only release; no code changes.
- 294 tests pass, mypy/ruff clean.

## [0.15.0] - 2026-04-27

### Added

- **MCTS-seeded refinement** (`RefinementConfig.use_mcts_seed: bool = False`):
  When enabled and `ReasonerConfig.mcts` is configured, refinement rounds
  beyond round 0 seed the Langevin sampler from the MCTS top-1 latent of
  the accumulated trace pool — instead of re-encoding only the augmented
  question. This focuses extra compute on the most promising region of
  latent space identified so far.
- New helper `HierarchicalLatentReasoner._mcts_top1_latent(traces)` returns
  the MCTS top-ranked latent or `None` when MCTS is disabled.
- `_reason_once` now accepts an optional `seed_latent` kwarg that overrides
  the encoder when provided. Round 0 still always uses the encoder.

### Notes

- Default behaviour unchanged: `use_mcts_seed` defaults to `False`.
- 294 tests pass, 95% coverage, ruff/mypy clean.

## [0.14.0] - 2026-04-27

### Added

- **Real LongMemEval JSONL adapter** (`benchmarks.longmemeval.load_longmemeval_jsonl`):
  strict JSONL loader for the official LongMemEval dataset format. Validates
  required fields (`id`, `question`, `answer`, `question_type`, `facts`),
  enforces the five canonical `question_type` values, validates each fact's
  `speaker` (`user`/`assistant`), and supports optional `superseded_by` for
  knowledge-update episodes. Errors include the source path and line number.
- **Stable results serializer** (`benchmarks.longmemeval.write_results_json`):
  writes a stable JSON schema (`total`, `correct`, `accuracy`,
  `accuracy_by_type`, `per_type_counts`, `details`, optional `metadata`) and
  auto-creates parent directories.
- **Production CLI runner** (`scripts/run_longmemeval.py`): argparse-driven
  entry point with `--jsonl`, `--num`, `--seed`, `--dim`, `--top-k`, `--out`.
  Captures full reproducibility metadata (ebrm-system version, Python
  version, platform, source, embed_dim, top_k, memory config, post-run
  memory stats, UTC timestamp).
- **First published baseline**: `benchmarks/results/longmemeval-v0.14.0.json`
  records 200-episode synthetic-harness results with the hash-projection
  embedder. This is the floor that future trained projectors must beat.

### Notes

- Python 3.10 compatibility verified: the runner uses
  `datetime.now(timezone.utc)` (not the 3.11+ `datetime.UTC` alias).

## [0.13.0] - 2026-04-27

### Added

- **LongMemEval-style benchmark harness** (`benchmarks.longmemeval`):
  end-to-end validation of `TieredMemory` on the five canonical question
  types from LongMemEval (single-session-user, single-session-assistant,
  multi-session, temporal-reasoning, knowledge-update). Ships a
  deterministic synthetic generator (`synth_longmemeval`) so the harness
  runs in CI without the gated dataset; drop in real episodes via the
  `LongMemEpisode` schema when you have access.
  - `MemoryFact` schema with `superseded_by` to model knowledge updates.
  - `LongMemRunResult` reports per-type accuracy, useful for tuning
    promotion thresholds and TTLs.
  - `default_memory()` helper returns a sensibly-tuned `TieredMemory`
    for science runs.
  - `hash_embed()` torch-free deterministic projector for harness self-
    tests; swap in your real projector when running for science.
- 14 new tests; suite now 279 tests, 96% coverage.

## [0.12.0] - 2026-04-27

### Added

- **PRM-guided refinement** — closes the loop between v0.6 (refinement)
  and v0.7 (pluggable PRM verifier):
  - New `extra_verifiers` constructor kwarg on
    `HierarchicalLatentReasoner`. Verifiers passed here are *appended*
    to the intent-routed chain so PRM rejections (or any custom check)
    flow into the existing refinement-and-critique loop without
    overwriting hard verifiers.
  - The reasoner now passes `{"question": question}` as verifier
    context, so PRM-style verifiers receive both the question and the
    candidate answer (matches the `ScalarPRMFn` / `GenerativePRMFn`
    contracts in `verifiers/prm.py`).
- 3 new tests; suite now 265 tests, 95% coverage.

## [0.11.0] - 2026-04-27

### Added

- **Episodic write-back to the index** (Letta-style closed-loop learning):
  new `ReasonerConfig.learn_from_solves` flag. When enabled, every
  successful `solve()` pushes the winning trace's latent into
  `self.index` via the duck-typed `add(latents, payloads)` method.
  Compatible with both `LatentIndex` and `TieredMemory` so subsequent
  solves can warm-start from prior wins. `False` by default to preserve
  existing read-only behaviour. Audit info surfaced at
  `ReasoningResult.details["memory_write"]` (`written`, `kind`,
  `energy`, `answer`).
- 5 new tests; suite now 262 tests, 95% coverage.

## [0.10.0] - 2026-04-27

### Added

- **ReST-MCTS\*-style search** (`ebrm_system.inference.mcts`): UCB1-driven
  Monte Carlo Tree Search over the candidate pool, re-ranking under a
  fixed `num_simulations` value-function budget. Tree shape is shallow
  (root → DVTS-style clusters → candidates) and pure-Python + NumPy.
  Disabled by default; opt in via `ReasonerConfig.mcts = MCTSConfig(...)`.
  The default value function maps each candidate's energy into `[0, 1]`;
  pass `mcts_value_fn=...` at construction to plug in a PRM-guided value
  (ThinkPRM, Athena-PRM, or any callable returning `[0, 1]`).
  Audit (`simulations_run`, `top_visits`, `top_values`, `pool_size`) is
  surfaced in `ReasoningResult.details["mcts"]`.
  References: ReST-MCTS\* (arXiv:2406.03816), AlphaProof.
- 14 new tests (8 module + 3 integration + reuse). Suite now 257 tests,
  96% coverage.

## [0.9.0] - 2026-04-27

### Added

- **Three-tier latent memory** (`ebrm_system.memory.tiered`): a
  Hindsight/Letta-style stack of working → episodic → semantic
  `LatentIndex` instances with hit-count promotion, TTL expiry, capacity
  eviction, and an optional summarization hook on eviction
  (`Summarizer` callback). Drop-in compatible with the existing
  `LatentIndex` duck-typed interface, so it slots straight into
  `generate_candidates(..., index=tiered_memory)` for QJL warm-start.
  References: Hindsight (LongMemEval 91.4%), Letta/MemGPT.
- 14 new tests; suite now 243 tests, 95% coverage.

## [0.8.0] - 2026-04-27

### Added

- **Coconut-style latent recursion**
  (`ebrm_system.inference.latent_recursion`): iterates the seed latent
  through a caller-supplied `StepFn` before candidate generation, with
  optional plateau-based early halting via the existing `HaltPolicy`.
  Default step is a torch-free finite-difference energy descent
  (`gradient_step`). Disabled by default; opt in via
  `ReasonerConfig.latent_recursion = RecursionConfig(max_steps=K)`.
  Audit info (`steps_run`, `halted_early`, `energy_start`, `energy_end`)
  is surfaced in `ReasoningResult.details["latent_recursion"]`.
  References: Coconut (arXiv:2412.06769), recurrent-depth (OpenReview
  D6o6Bwtq7h), LatentSeek.
- New constructor kwarg `recursion_step_fn` on `HierarchicalLatentReasoner`
  for plugging in a torch-backed recurrent block.
- 17 new tests; suite now 229 tests, 95% coverage.

## [0.7.0] - 2026-04-27

### Added

- **Diverse Verifier Tree Search-style selection**
  (`ebrm_system.inference.diverse_selector`): clusters candidate latents
  by greedy farthest-first traversal, keeps the lowest-energy member of
  each cluster, and votes only over the survivors. Disabled by default;
  set `ReasonerConfig.diverse_selection = DiverseSelectionConfig(...)`
  to enable. Reference: HuggingFaceH4 "Scaling test-time compute"
  (DVTS).
- **Pluggable PRM verifier** (`ebrm_system.verifiers.prm`):
  `ScalarPRMVerifier` wraps `(question, answer) -> float` plus a
  threshold; `GenerativePRMVerifier` wraps
  `(question, answer) -> PRMVerdict` for richer step-level feedback
  (the ThinkPRM / Athena-PRM contract). The generative verifier carries
  its rejection reasoning into `VerificationResult.reason`, which the
  v0.6 refinement loop folds back into the next prompt automatically.
- `ReasoningResult.details["diverse_selection"]` reports
  `{input, survivors}` whenever DVTS is enabled.

## [0.6.0] - 2026-04-27

### Added

- **Verification-and-refinement loop** (`ebrm_system.core.refinement`):
  when the verifier chain rejects candidates, the rejection reasons are
  collected as critiques, the question is re-rendered with those
  critiques appended, and reasoning is run again. Verified candidates
  from all rounds are pooled before voting. Disabled by default
  (`RefinementConfig(max_rounds=0)`); set `max_rounds >= 1` to enable.
  This is the IMO-2025-gold mechanism from arXiv:2507.15855.
- **Difficulty-adaptive compute profile**
  (`ebrm_system.core.compute_profile`): new `ComputeProfile` enum
  (`ECONOMY`, `BALANCED`, `MAX_QUALITY`) plus `scale_budget()`. Economy
  collapses easy questions (`difficulty < 0.3`) to N=1 greedy decoding;
  Max-Quality doubles candidates and Langevin steps for hard questions
  (`difficulty >= 0.5`). Balanced is a no-op. Wired into
  `ReasonerConfig.compute_profile`.
- **Free PRM training data** (`ebrm_system.reward.prm_data`):
  `PRMRecord`, `make_records()`, `write_jsonl()`. Implements the
  Athena-PRM / ThinkPRM weak-vs-strong agreement labelling: every
  `solve()` call can dump per-candidate JSONL records with a
  `strong_label` (the voted answer) and an `agreement` flag — reliable
  pseudo-labels for fine-tuning a generative PRM with no human
  annotation. References ACL 2025 "Lessons of PRMs" and Athena-PRM.
- `ReasoningResult.details` now carries `compute_profile`, the actual
  scaled `budget`, and `refinement_rounds` for full audit.

### Changed

- `HierarchicalLatentReasoner.solve()` is now a thin orchestration loop
  over a per-round helper `_reason_once()` for cleanly separating
  encode→generate→decode→verify from the new pooling/refinement logic.

## [0.5.0] - 2026-04-27

### Added

- **Adaptive halt for Langevin candidate generation**
  (`ebrm_system.inference.halt`): new `HaltPolicy` Protocol with
  `NeverHalt` (default, preserves prior behaviour) and `PlateauHalt` —
  stops a trajectory when energy variance over a rolling window falls
  below a threshold. Free compute savings on easy problems; no model
  changes. `Candidate.steps_run` now records the actual step count.
- **Bounded `LatentIndex` with eviction**
  (`ebrm_system.reward.qjl_index`): `IndexConfig.max_size` and
  `evict_policy = "lru" | "fifo"`. LRU touches entries on `search()`
  hits; both policies are O(evicted) per insert. `add()` now returns the
  number of evicted entries. Backwards-compatible: `max_size=None`
  (default) keeps the index unbounded.
- 14 new tests (6 halt + 8 eviction). Suite is now 141 tests, 94%
  coverage. Lint and mypy clean.

### Why these landed now

Both are agent-loop hygiene improvements that block real deployments:

- Without adaptive halt, every easy question pays for the worst-case
  Langevin budget.
- Without bounded `LatentIndex`, a long-running agent's warm-start
  cache grows until OOM.

## [0.4.0] - 2026-04-27

### Added

- **`ebrm_system.core` is now stable** — `HierarchicalLatentReasoner`
  composes intent routing → encoder → Langevin candidate generation
  (with optional QJL warm-start) → decoder → verifier chain → weighted
  self-consistency vote into a single `solve(question)` call. Encoder,
  decoder, and energy callables are injected so the module is
  torch-optional and unit-testable on CPU.
- `ReasonerConfig` knobs: `weight_by`, `numerical_tolerance`,
  `require_verification`, `seed`. Compute budget (Langevin steps, restart
  count, candidate count) is auto-selected from the intent classifier.
- `ReasoningResult` carries a tuple of `TraceItem`s (latent, decoded
  answer, energy, seed, warm-start flag, verifier results) for full
  audit. `verified_fraction` reports the share of candidates passing the
  hard chain.
- 7 new unit tests in `tests/test_core_reasoner.py` covering deterministic
  seeding, intent routing, QJL warm-start, encoder shape validation, and
  verification fallback. Total suite: 127 tests, 94% coverage.

### Changed

- README component table: `core` is now ✅ stable. The only remaining
  change vs v0.3.0 is the new `core` row.

## [0.3.0] - 2026-04-27

### Added

- **Verifier plane (per-intent hard checks)**
  - `verifiers.lean.LeanVerifier` — Lean 4 subprocess wrapper for the math
    lane. Gracefully degrades when `lean` is not on PATH.
  - `verifiers.dri.DRIVerifier` — diagram-driven commutativity checker for the
    advice/plan lane. Pure-Python `Diagram` / `ExactMorphism` /
    `VectorMorphism` types.
  - `verifiers.routing.chain_for_intent()` and `advice_chain()` —
    intent-routed `VerifierChain` factory.

- **Inference modules**
  - `inference.candidates` — multi-seed Langevin generator with optional
    `LatentIndex` warm-start retrieval.
  - `inference.qjl` — 1-bit Johnson-Lindenstrauss projector (~96× compression,
    < 0.1 cosine error at m=2048).
  - `inference.turboquant_kv` — numpy reference for TurboQuant-style
    Hadamard-rotated 2/4/8-bit KV-cache compression.
  - `inference.torch_langevin` — autograd-based Langevin step (torch-optional).
  - `inference.turboquant_attention` — torch SDPA over compressed KV
    (torch-optional, torch.compile-friendly).

- **Reward**
  - `reward.qjl_index.LatentIndex` — bit-code latent ANN index.

- **CLI**
  - `ebrm-system verify-routed <query> <candidate>` — intent-routed verification.
  - `ebrm-system verify-plan <diagram.json> <candidate.json>` — DRI plan check.

### Internals

- 120 tests (up from 53) · 93 % branch coverage · ruff strict + format ·
  mypy strict.

## [Unreleased]

### Added
- Trackio integration in `benchmarks/runner.py` — pass `trackio_project="..."` to stream per-example correctness, running accuracy, and latency to a local or HF-Spaces-hosted [Trackio](https://github.com/gradio-app/trackio) dashboard. Opt-in, silently degrades when trackio is not installed.
- `trackio>=0.2` added to the `[benchmark]` extra.
- `tests/test_benchmarks_runner.py` covering happy path, solver errors, trackio integration (via monkeypatched module), and missing-trackio fallback.
- `pythonpath = ["."]` in pytest config so `benchmarks/` imports resolve in tests.

## [0.1.0] - 2026-04-22

### Added
- Initial public scaffold.
- `ebrm_system.intent` — `Intent`, `IntentPrediction`, `Classifier` Protocol, `RuleBasedClassifier` with compute-budget policy.
- `ebrm_system.verifiers` — `VerificationResult`, `Verifier` Protocol, `VerifierChain`; concrete `SymPyVerifier`, `ExecVerifier` (sandboxed subprocess), `RegexVerifier`.
- `ebrm_system.voting` — `Candidate`, `VoteResult`, `SelfConsistencyVoter` with uniform / confidence / inverse-energy weighting and exact / numerical bucketing.
- Typer CLI: `version`, `classify`, `verify`.
- Pytest suite across verifiers, intent classifier, voter, and CLI.
- CI: lint (ruff), type-check (mypy strict), tests on Python 3.11 / 3.12 / 3.13, sdist + wheel build.
- Docs (MkDocs-material): architecture, intent, verifiers, voting, API reference.
- ADR-001 documenting the two-repo split (`ebrm` model vs `ebrm-system` pipeline).
- Benchmark harness + GSM8K adapter.
- Default YAML config.
- Apache-2.0 license.
