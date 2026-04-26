"""Core reasoning primitives — hierarchical latent reasoner.

The :class:`HierarchicalLatentReasoner` composes the rest of `ebrm-system`
(intent routing, candidate generation, verifiers, voting) into a single
end-to-end pipeline. The encoder/decoder/energy callables are injected so
this module is torch-optional and unit-testable on CPU.

For the real EBRM v2 weights, load them from
    https://huggingface.co/piyushptiwari/ebrm-v2-qwen3-4b
and adapt them to the :data:`EncoderFn`, :data:`DecoderFn`, :data:`EnergyFn`
contracts.
"""

from ebrm_system.core.compute_profile import ComputeProfile, ScaledBudget, scale_budget
from ebrm_system.core.reasoner import (
    DecoderFn,
    EncoderFn,
    EnergyFn,
    HierarchicalLatentReasoner,
    LatentCandidate,
    ReasonerConfig,
    ReasoningResult,
    TraceItem,
)
from ebrm_system.core.refinement import (
    RefinementConfig,
    build_refined_question,
    collect_critiques,
    should_refine,
)
from ebrm_system.inference.diverse_selector import (
    DiverseSelectionConfig,
    select_diverse,
)
from ebrm_system.inference.latent_recursion import (
    RecursionConfig,
    RecursionResult,
    recurse_latent,
)

__all__ = [
    "ComputeProfile",
    "DecoderFn",
    "DiverseSelectionConfig",
    "EncoderFn",
    "EnergyFn",
    "HierarchicalLatentReasoner",
    "LatentCandidate",
    "ReasonerConfig",
    "ReasoningResult",
    "RecursionConfig",
    "RecursionResult",
    "RefinementConfig",
    "ScaledBudget",
    "TraceItem",
    "build_refined_question",
    "collect_critiques",
    "recurse_latent",
    "scale_budget",
    "select_diverse",
    "should_refine",
]
