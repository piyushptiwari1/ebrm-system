# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.0] - TBD

- First PyPI release.
