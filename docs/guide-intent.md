# Intent & routing

The first layer maps a query to an intent, a difficulty in `[0,1]`, and a compute budget.

## Quick example

```python
from ebrm_system.intent import RuleBasedClassifier

clf = RuleBasedClassifier()
pred = clf.classify("Solve: 3x + 7 = 22")

print(pred.intent)                       # Intent.MATH_REASONING
print(pred.suggested_langevin_steps)     # scales with difficulty
print(pred.suggested_trace_count)        # K parallel traces
```

## Budget policy

- `steps`: 50 → 2000 (linear in difficulty)
- `restarts`: 1 → 10
- `trace_count`: 1 → 16

## Swap in a neural classifier

Any object implementing `Classifier.classify(query: str) -> IntentPrediction` works.
