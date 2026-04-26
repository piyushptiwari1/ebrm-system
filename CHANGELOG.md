# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
