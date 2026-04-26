"""Diagram-Driven Relational Intelligence (DRI) commutativity verifier.

For the ``advice`` / ``plan`` intent lane there is no formal language like
Lean, but plans *can* still be modelled as small categories: nodes are
states, morphisms are transitions, and a plan is a path through the
diagram. A plan is **valid** when the diagram commutes — i.e. every pair of
paths between the same source and target produces the same final state.

This module implements a minimal, dependency-free commutativity checker
suitable for plans of up to a few dozen nodes.

Worked example:

    Nodes: {raw_data, cleaned, model, deployed_model}
    Morphisms:
        clean : raw_data → cleaned
        train : cleaned  → model
        deploy: model    → deployed_model
        e2e   : raw_data → deployed_model     (claimed shortcut)

    The diagram commutes iff   deploy ∘ train ∘ clean  ≡  e2e.

The checker accepts:

    * ``ExactMorphism`` — pure-Python callable on JSON-serializable state
      (compared by ``==``).
    * ``VectorMorphism`` — callable on numpy arrays (compared by cosine
      similarity ≥ ``cosine_threshold``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ebrm_system.verifiers.base import VerificationResult


class Morphism(Protocol):
    """A morphism src → dst."""

    name: str
    src: str
    dst: str

    def apply(self, x: Any) -> Any: ...


@dataclass(frozen=True)
class ExactMorphism:
    """Morphism on JSON-comparable state, checked by structural equality."""

    name: str
    src: str
    dst: str
    fn: Callable[[Any], Any]

    def apply(self, x: Any) -> Any:
        return self.fn(x)


@dataclass(frozen=True)
class VectorMorphism:
    """Morphism on numpy arrays, checked by cosine similarity."""

    name: str
    src: str
    dst: str
    fn: Callable[[NDArray[np.float32]], NDArray[np.float32]]

    def apply(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.fn(x)


@dataclass
class Diagram:
    """A small category: nodes + labelled morphisms."""

    morphisms: list[Morphism] = field(default_factory=list)

    def add(self, m: Morphism) -> None:
        self.morphisms.append(m)

    def by_name(self, name: str) -> Morphism:
        for m in self.morphisms:
            if m.name == name:
                return m
        raise KeyError(f"morphism {name!r} not in diagram")

    def compose(self, names: Iterable[str], x: Any) -> Any:
        """Apply morphisms in order to x. Validates src/dst chaining."""
        names = list(names)
        if not names:
            return x
        last_dst: str | None = None
        out = x
        for name in names:
            m = self.by_name(name)
            if last_dst is not None and m.src != last_dst:
                raise ValueError(
                    f"morphism {name!r} src={m.src!r} does not match previous dst={last_dst!r}"
                )
            out = m.apply(out)
            last_dst = m.dst
        return out


def _equal_exact(a: Any, b: Any) -> bool:
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )
    except (TypeError, ValueError):
        return bool(a == b)


def _cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def commutes(
    diagram: Diagram,
    paths: list[list[str]],
    initial: Any,
    cosine_threshold: float = 0.985,
) -> tuple[bool, list[Any]]:
    """Check whether all listed paths produce the same final value from ``initial``.

    Returns (verdict, list_of_outputs).

    For numpy arrays, equality means cosine ≥ ``cosine_threshold``.
    For everything else, JSON-structural equality.
    """
    if len(paths) < 2:
        raise ValueError("need at least 2 paths to check commutativity")
    outputs = [diagram.compose(p, initial) for p in paths]
    ref = outputs[0]
    if isinstance(ref, np.ndarray):
        ok = all(
            isinstance(o, np.ndarray) and _cosine(ref, o) >= cosine_threshold for o in outputs[1:]
        )
    else:
        ok = all(_equal_exact(ref, o) for o in outputs[1:])
    return ok, outputs


class DRIVerifier:
    """Verifier that checks commutativity of a candidate plan diagram.

    The candidate is the JSON description::

        {
          "initial": <any JSON value>,
          "paths":   [["m1", "m2"], ["m3"]]
        }

    The diagram and morphisms are supplied via ``context["diagram"]``. This
    keeps the verifier stateless across calls and lets users compose multiple
    DRIVerifiers (one per agent) into a chain.
    """

    name = "dri"

    def __init__(self, cosine_threshold: float = 0.985) -> None:
        self.cosine_threshold = cosine_threshold

    def check(
        self, candidate: object, context: dict[str, object] | None = None
    ) -> VerificationResult:
        if context is None or "diagram" not in context:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="DRIVerifier requires context['diagram']",
                evidence={},
            )
        diagram = context["diagram"]
        if not isinstance(diagram, Diagram):
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="context['diagram'] must be a Diagram",
                evidence={"got_type": type(diagram).__name__},
            )
        if not isinstance(candidate, str):
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason="DRIVerifier expects a JSON string candidate",
                evidence={"got_type": type(candidate).__name__},
            )

        try:
            payload = json.loads(candidate)
            paths = payload["paths"]
            initial = payload["initial"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"could not parse candidate: {exc}",
                evidence={},
            )

        try:
            ok, outputs = commutes(diagram, paths, initial, self.cosine_threshold)
        except (KeyError, ValueError) as exc:
            return VerificationResult(
                verifier=self.name,
                verified=False,
                confidence=0.0,
                reason=f"diagram composition failed: {exc}",
                evidence={"paths": paths},
            )

        return VerificationResult(
            verifier=self.name,
            verified=ok,
            confidence=1.0 if ok else 0.0,
            reason="diagram commutes" if ok else "diagram does not commute",
            evidence={"num_paths": len(paths), "outputs_summary": str(outputs)[:500]},
        )
