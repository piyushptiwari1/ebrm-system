"""Inference orchestration — adaptive Langevin with budget controller.

.. note::
   This subpackage is **experimental** (see :mod:`ebrm_system._experimental`).
   Its API may change between minor releases without a deprecation cycle.

v3 modules:
    * ``candidates``     — multi-seed Langevin candidate generator (numpy).
    * ``qjl``            — 1-bit Johnson-Lindenstrauss projector.
    * ``turboquant_kv``  — TurboQuant-style KV-cache compressor.
    * ``torch_langevin`` — autograd-based Langevin step (torch-optional;
                           import directly when torch is available).
"""

from ebrm_system._experimental import _warn_experimental

_warn_experimental("inference")

from ebrm_system.inference.candidates import (  # noqa: E402
    Candidate,
    CandidateConfig,
    generate_candidates,
    langevin_step,
)
from ebrm_system.inference.diverse_selector import (  # noqa: E402
    DiverseSelectionConfig,
    select_diverse,
)
from ebrm_system.inference.latent_recursion import (  # noqa: E402
    RecursionConfig,
    RecursionResult,
    gradient_step,
    recurse_latent,
)
from ebrm_system.inference.mcts import (  # noqa: E402
    MCTSConfig,
    MCTSResult,
    mcts_select,
)
from ebrm_system.inference.qjl import QJLConfig, QJLProjector  # noqa: E402
from ebrm_system.inference.turboquant_kv import (  # noqa: E402
    CompressedKV,
    KVCacheCompressor,
    KVQuantConfig,
)

__all__ = [
    "Candidate",
    "CandidateConfig",
    "CompressedKV",
    "DiverseSelectionConfig",
    "KVCacheCompressor",
    "KVQuantConfig",
    "MCTSConfig",
    "MCTSResult",
    "QJLConfig",
    "QJLProjector",
    "RecursionConfig",
    "RecursionResult",
    "generate_candidates",
    "gradient_step",
    "langevin_step",
    "mcts_select",
    "recurse_latent",
    "select_diverse",
]
