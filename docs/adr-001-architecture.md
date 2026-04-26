# ADR-001: Six-Layer Reasoning Pipeline, Separated from the Model

- **Status**: Accepted
- **Date**: 2026-03-28
- **Deciders**: @piyushptiwari1

## Context

We have two artefacts that evolve on different cadences and serve different audiences:

1. **The model** (trained weights + training code) — the research artefact. Published as [`ebrm`](https://github.com/piyushptiwari1/ebrm). Cite-able, fixed once trained, versioned by checkpoint.
2. **The system** — the production pipeline that wraps *any* base reasoner with intent routing, adaptive compute, verifiers, and voting. Published as `ebrm-system`. Ships on a fast cadence, should be installable via `pip`, should run with or without the v2 model.

Keeping them in one repository forced tight coupling: breaking changes in the pipeline churned the paper repo; retraining pinned the pipeline version. Splitting them lets each evolve independently and keeps the model repo citable.

## Decision

**Two repositories, one pipeline architecture.**

- `ebrm` — model, training loop, paper reference, Hugging Face release.
- `ebrm-system` (this repo) — installable pipeline with the following six layers, each behind a `typing.Protocol`:

  1. **Intent / difficulty classifier** → emits `IntentPrediction` with suggested Langevin steps, restarts, and trace count.
  2. **Hierarchical latent reasoner** → inner latent-thought loop (Coconut-inspired). Swappable with any `nn.Module`.
  3. **Adaptive Langevin** → test-time compute scaled by difficulty; K parallel traces.
  4. **Process reward model** → stepwise energy → per-trace confidence.
  5. **External verifier bridge** → `SymPyVerifier`, `ExecVerifier` (sandboxed subprocess), `RegexVerifier`, composed via `VerifierChain` that short-circuits on rejection.
  6. **Self-consistency voter** → `SelfConsistencyVoter` with `uniform | confidence | inverse_energy` weighting and exact / numeric bucketing.

The pipeline is CPU-testable end-to-end *without* the model — every layer has a deterministic rule-based / algorithmic implementation so tests don't require GPUs.

## Consequences

**Positive**
- Fast CI (no GPU / no model download for 95 % of tests).
- Clean citation: paper → model repo; deployment → system repo.
- Pipeline can wrap third-party reasoners (Qwen, Llama, DeepSeek) with the same verifier + voting layers.
- Each Protocol is independently swappable.

**Negative**
- Two repos to maintain; cross-repo docs must be kept in sync.
- The v2 model release must pin a compatible `ebrm-system` minor version.

**Mitigations**
- Documentation tables show which `ebrm-system` versions match which `ebrm` checkpoints.
- Pipeline exports a `Reasoner` Protocol so model implementations can be version-gated at import time.

## Alternatives considered

- **Monorepo.** Rejected: churn, heavier CI, weaker citability.
- **Ship pipeline as part of model package.** Rejected: forces GPU install for users who only want verifiers / voting.
- **No pipeline, only a notebook.** Rejected: users cannot `pip install` a notebook.
