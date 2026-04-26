# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-27

### Added

- **Verifier plane (per-intent hard checks)**
  - `verifiers.lean.LeanVerifier` — Lean 4 subprocess wrapper for the math lane.
    Gracefully degrades when `lean` is not on PATH.
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
  - `reward.qjl_index.LatentIndex` — bit-code latent ANN index for warm-start
    retrieval.

- **CLI**
  - `ebrm-system verify-routed <query> <candidate>` — classify, then run the
    intent-routed verifier chain.
  - `ebrm-system verify-plan <diagram.json> <candidate.json>` — DRI
    commutativity checks from the command line, restricted to safe stateless
    string ops.

### Changed

- `Verifier` protocol now exported from `ebrm_system.verifiers` package root,
  alongside `Diagram`, `ExactMorphism`, `Morphism`, `VectorMorphism`,
  `commutes`, `chain_for_intent`, `advice_chain`.

### Internals

- 120 tests (up from 53) · 93 % branch coverage · ruff strict + format clean ·
  mypy strict clean.

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
