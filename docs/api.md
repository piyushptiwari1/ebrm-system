# API reference

## `ebrm_system.intent`

- `Intent` — enum of routing categories.
- `IntentPrediction` — `{intent, difficulty, suggested_langevin_steps, suggested_restarts, suggested_trace_count, reasoning}`.
- `Classifier` — Protocol: `classify(query: str) -> IntentPrediction`.
- `RuleBasedClassifier` — baseline rule-based classifier.

## `ebrm_system.verifiers`

- `VerificationResult` — `{verifier, verified, confidence, reason, evidence}`.
- `Verifier` — Protocol: `check(candidate, context) -> VerificationResult`.
- `VerifierChain` — runs verifiers in order, short-circuits on rejection.
- `SymPyVerifier` — symbolic + numeric equality.
- `ExecVerifier` — sandboxed subprocess exec + stdout/JSON check.
- `RegexVerifier` — `re.fullmatch` on a string candidate.

## `ebrm_system.voting`

- `Candidate` — `{answer, confidence, energy, trace_id}`.
- `VoteResult` — `{answer, support, total, agreement, weighted_score, runner_up, runner_up_support, details}`.
- `SelfConsistencyVoter` — weighted bucket-vote.

## `ebrm_system.core`, `ebrm_system.reward`, `ebrm_system.inference`

🚧 In progress. Interfaces will be released before the 0.2 line.
