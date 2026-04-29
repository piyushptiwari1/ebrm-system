"""Core reasoning primitives — hierarchical latent reasoner.

.. note::
   This subpackage is **experimental** (see :mod:`ebrm_system._experimental`).
   Its API may change between minor releases without a deprecation cycle.

The :class:`HierarchicalLatentReasoner` composes the rest of `ebrm-system`
(intent routing, candidate generation, verifiers, voting) into a single
end-to-end pipeline. The encoder/decoder/energy callables are injected so
this module is torch-optional and unit-testable on CPU.

For the real EBRM v2 weights, load them from
    https://huggingface.co/piyushptiwari/ebrm-v2-qwen3-4b
and adapt them to the :data:`EncoderFn`, :data:`DecoderFn`, :data:`EnergyFn`
contracts.
"""

from ebrm_system._experimental import _warn_experimental

_warn_experimental("core")

from ebrm_system.core.compute_profile import (  # noqa: E402
    ComputeProfile,
    ScaledBudget,
    scale_budget,
)
from ebrm_system.core.reasoner import (  # noqa: E402
    DecoderFn,
    EncoderFn,
    EnergyFn,
    HierarchicalLatentReasoner,
    LatentCandidate,
    ReasonerConfig,
    ReasoningResult,
    TraceItem,
)
from ebrm_system.core.refinement import (  # noqa: E402
    RefinementConfig,
    build_refined_question,
    collect_critiques,
    should_refine,
)
from ebrm_system.inference.diverse_selector import (  # noqa: E402
    DiverseSelectionConfig,
    select_diverse,
)
from ebrm_system.inference.latent_recursion import (  # noqa: E402
    RecursionConfig,
    RecursionResult,
    recurse_latent,
)
from ebrm_system.inference.mcts import (  # noqa: E402
    MCTSConfig,
    MCTSResult,
    mcts_select,
)

__all__ = [
    "ComputeProfile",
    "DecoderFn",
    "DiverseSelectionConfig",
    "EncoderFn",
    "EnergyFn",
    "HierarchicalLatentReasoner",
    "LatentCandidate",
    "MCTSConfig",
    "MCTSResult",
    "ReasonerConfig",
    "ReasoningResult",
    "RecursionConfig",
    "RecursionResult",
    "RefinementConfig",
    "ScaledBudget",
    "TraceItem",
    "build_refined_question",
    "collect_critiques",
    "mcts_select",
    "recurse_latent",
    "scale_budget",
    "select_diverse",
    "should_refine",
]
