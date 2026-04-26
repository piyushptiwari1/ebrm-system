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

__all__ = [
    "DecoderFn",
    "EncoderFn",
    "EnergyFn",
    "HierarchicalLatentReasoner",
    "LatentCandidate",
    "ReasonerConfig",
    "ReasoningResult",
    "TraceItem",
]
