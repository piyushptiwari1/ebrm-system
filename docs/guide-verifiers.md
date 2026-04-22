# Verifiers

Verifiers are **mechanical** — they confirm with SymPy, Python, or regex. They never ask an LLM to grade an LLM.

## SymPy

```python
from ebrm_system.verifiers import SymPyVerifier

v = SymPyVerifier()
r = v.check("(x+1)**2", {"expected": "x**2 + 2*x + 1"})
assert r.verified
```

## Exec (sandboxed)

```python
from ebrm_system.verifiers import ExecVerifier

v = ExecVerifier(timeout_s=3.0)
r = v.check("print(2+2)", {"expected_stdout": "4"})
assert r.verified
```

The subprocess is launched with `-I -S` (isolated, no site imports), with a timeout, in a temp cwd.

## Regex

```python
from ebrm_system.verifiers import RegexVerifier

v = RegexVerifier()
r = v.check("abc123", {"pattern": r"[a-z]+\d+"})
assert r.verified
```

## Chaining

```python
from ebrm_system.verifiers import VerifierChain, RegexVerifier, SymPyVerifier

chain = VerifierChain([RegexVerifier(), SymPyVerifier()])
results = chain.verify("42", {"pattern": r"\d+", "expected": 42})
assert chain.all_passed(results)
```

The chain short-circuits on the first rejection.
