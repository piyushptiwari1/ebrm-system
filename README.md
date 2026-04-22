# ebrm-system

[![CI](https://github.com/piyushptiwari1/ebrm-system/actions/workflows/ci.yml/badge.svg)](https://github.com/piyushptiwari1/ebrm-system/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/ebrm-system.svg)](https://pypi.org/project/ebrm-system/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](LICENSE)

> **Energy-Based Reasoning Machine — the system.**
> A production reasoning pipeline: intent routing, adaptive test-time compute, energy-based scoring, external verifier bridge, and self-consistency voting.

This repository is the **system layer** on top of the [ebrm](https://github.com/piyushptiwari1/ebrm) model (research / paper reference). `ebrm-system` is the framework you deploy — `ebrm` is the model you cite.

---

## Why this exists

Modern reasoning LLMs are strong but unverifiable: they emit plausible chains that are hard to check mechanically. `ebrm-system` wraps any base reasoner with a pipeline that makes answers **auditable, budget-aware, and consistency-checked**.

## Architecture

```
query
  │
  ▼
┌──────────────────────────┐
│ 1. Intent Classifier     │   rule-based or neural; emits difficulty + budget
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐
│ 2. Hierarchical Reasoner │   latent-thought inner loop (Coconut-inspired)
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐
│ 3. Adaptive Langevin     │   steps scale with difficulty; K parallel traces
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐
│ 4. Process Reward Model  │   stepwise energy → trace confidence
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐
│ 5. External Verifier     │   SymPy / sandboxed exec / regex — mechanical check
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐
│ 6. Self-Consistency Vote │   weighted by confidence or 1/energy
└──────────────────────────┘
  │
  ▼
 answer + audit trail
```

Every stage is a swappable component behind a Protocol. The verifier layer never hallucinates — it only confirms what SymPy / Python / regex can mechanically check.

## Install

```bash
pip install ebrm-system
```

From source:

```bash
git clone https://github.com/piyushptiwari1/ebrm-system
cd ebrm-system
pip install -e ".[dev]"
```

## Quick start

```bash
# See what the intent router thinks about a query
ebrm-system classify "Solve: 3x + 7 = 22"

# Verify an answer mechanically
ebrm-system verify "x**2 + 2*x + 1" "(x+1)**2"
```

Python API:

```python
from ebrm_system.intent import RuleBasedClassifier
from ebrm_system.verifiers import SymPyVerifier, VerifierChain
from ebrm_system.voting import Candidate, SelfConsistencyVoter

clf = RuleBasedClassifier()
pred = clf.classify("Solve: 3x + 7 = 22")
# pred.suggested_langevin_steps, pred.suggested_trace_count, ...

chain = VerifierChain([SymPyVerifier()])
results = chain.verify("5", {"expected": "5"})
assert chain.all_passed(results)

voter = SelfConsistencyVoter(numerical=True, tolerance=0.01, weight_by="inverse_energy")
result = voter.vote([
    Candidate(answer=5.0, energy=-2.0),
    Candidate(answer=5.0, energy=-1.5),
    Candidate(answer=4.0, energy= 3.0),
])
# result.answer == 5.0, weighted by low energy
```

## Components

| Module | Status | Purpose |
| --- | --- | --- |
| `ebrm_system.intent` | ✅ stable | Intent + difficulty + compute budget |
| `ebrm_system.verifiers` | ✅ stable | SymPy / exec sandbox / regex verifiers |
| `ebrm_system.voting` | ✅ stable | Self-consistency with weighted bucketing |
| `ebrm_system.core` | 🚧 WIP | Hierarchical latent reasoner |
| `ebrm_system.reward` | 🚧 WIP | Process reward model |
| `ebrm_system.inference` | 🚧 WIP | Adaptive Langevin orchestrator |

## Development

```bash
pip install -e ".[dev]"
pytest                           # run tests
ruff check .                     # lint
mypy src                         # type-check
pre-commit install               # optional hooks
```

CI runs lint + type + test on Python 3.10/3.11/3.12/3.13. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Design principles

1. **Mechanical over mystical** — verifiers confirm with SymPy / exec / regex; never an LLM grading an LLM.
2. **Budget-aware** — easy queries don't pay for hard-query compute. Intent routing controls Langevin steps, restarts, and trace count.
3. **Audit-first** — every candidate carries its trace, energy, and verifier evidence.
4. **Swappable** — everything is a Protocol. Swap the rule-based classifier for a neural one; swap SymPy for Z3; drop in your own voter.

## Citation

If you use this system in academic work, please cite the model paper:

```bibtex
@software{ebrm_system_2026,
  author  = {Tiwari, Piyush},
  title   = {ebrm-system: An Energy-Based Reasoning Machine pipeline},
  year    = {2026},
  url     = {https://github.com/piyushptiwari1/ebrm-system}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
