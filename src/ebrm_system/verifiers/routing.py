"""Intent-routed verifier chains.

Maps an :class:`Intent` to the appropriate :class:`VerifierChain` so the
pipeline can apply *hard* checks per lane:

    arithmetic / math_reasoning  -> SymPy + Lean (if available)
    code                         -> Exec + Regex
    factual                      -> Regex (citation/format)
    creative                     -> none (EBRM soft score only)
    dialogue / unknown           -> none

Lean and DRI verifiers degrade gracefully when their backends are absent,
so it is safe to include them unconditionally in the math/advice chains.
"""

from __future__ import annotations

from ebrm_system.intent import Intent
from ebrm_system.verifiers.base import Verifier, VerifierChain
from ebrm_system.verifiers.dri import DRIVerifier
from ebrm_system.verifiers.exec_verifier import ExecVerifier
from ebrm_system.verifiers.lean import LeanVerifier
from ebrm_system.verifiers.regex_verifier import RegexVerifier
from ebrm_system.verifiers.sympy_verifier import SymPyVerifier


def chain_for_intent(intent: Intent) -> VerifierChain:
    """Return the canonical verifier chain for a given intent."""
    verifiers: list[Verifier] = []
    if intent in (Intent.ARITHMETIC, Intent.MATH_REASONING):
        verifiers = [SymPyVerifier(), LeanVerifier()]
    elif intent == Intent.CODE:
        verifiers = [ExecVerifier(), RegexVerifier()]
    elif intent == Intent.FACTUAL:
        verifiers = [RegexVerifier()]
    # creative / dialogue / unknown -> empty chain (EBRM soft score only)
    return VerifierChain(verifiers)


def advice_chain() -> VerifierChain:
    """Verifier chain for structured advice/plan candidates (DRI)."""
    return VerifierChain([DRIVerifier()])
