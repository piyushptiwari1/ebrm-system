# Self-consistency voting

Given K candidate answers, elect a consensus. Supports exact and numerical bucketing, with uniform / confidence / inverse-energy weighting.

```python
from ebrm_system.voting import Candidate, SelfConsistencyVoter

voter = SelfConsistencyVoter(
    numerical=True,
    tolerance=0.01,
    weight_by="inverse_energy",
)
result = voter.vote([
    Candidate(answer=5.0, energy=-2.0),
    Candidate(answer=5.0, energy=-1.5),
    Candidate(answer=4.0, energy= 3.0),
])

print(result.answer)         # 5.0
print(result.agreement)      # 2/3
print(result.runner_up)      # 4.0
```

## Weighting modes

| `weight_by` | Behaviour |
| --- | --- |
| `uniform` | one vote per candidate |
| `confidence` | weighted by `Candidate.confidence` |
| `inverse_energy` | weighted by `sigmoid(-energy)` |

## Bucketing

- `numerical=False` → exact key (`c.answer`)
- `numerical=True` → round to nearest `tolerance`
