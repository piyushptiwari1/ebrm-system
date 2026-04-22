# Contributing to ebrm-system

Thanks for your interest. This project ships as a production pipeline — contributions that keep it clean, typed, tested, and CPU-runnable are very welcome.

## Dev setup

```bash
git clone https://github.com/piyushptiwari1/ebrm-system
cd ebrm-system
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Ground rules

- **Protocols over inheritance.** Every layer is a `typing.Protocol`. Keep it that way.
- **CPU-testable.** Tests must not require a GPU or model download.
- **Mechanical verification only.** Verifiers confirm with SymPy / Python / regex. No LLM-as-judge.
- **Keep commits small and scoped.** One PR = one coherent change.

## Before you open a PR

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
```

CI enforces all of the above on Python 3.11 / 3.12 / 3.13.

## Adding a new verifier

1. Create `src/ebrm_system/verifiers/<name>_verifier.py` implementing the `Verifier` Protocol.
2. Export it from `src/ebrm_system/verifiers/__init__.py`.
3. Add tests in `tests/test_verifiers_<name>.py` covering: pass, fail, malformed input, missing context key.
4. Update `docs/guide-verifiers.md` and `docs/api.md`.

## Commit style

Conventional Commits are preferred:

```
feat(intent): add neural classifier wrapper
fix(verifier): handle sympy SympifyError on bad input
docs: expand voting guide with inverse-energy example
```

## Licensing

By contributing, you agree your contribution is licensed under Apache-2.0 (see [LICENSE](LICENSE)).
