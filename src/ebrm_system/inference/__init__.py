"""Inference orchestration — adaptive Langevin with budget controller.

v3 modules:
    * ``candidates``     — multi-seed Langevin candidate generator (numpy).
    * ``qjl``            — 1-bit Johnson-Lindenstrauss projector.
    * ``turboquant_kv``  — TurboQuant-style KV-cache compressor.
    * ``torch_langevin`` — autograd-based Langevin step (torch-optional;
                           import directly when torch is available).
"""

from ebrm_system.inference.candidates import (
    Candidate,
    CandidateConfig,
    generate_candidates,
    langevin_step,
)
from ebrm_system.inference.diverse_selector import (
    DiverseSelectionConfig,
    select_diverse,
)
from ebrm_system.inference.qjl import QJLConfig, QJLProjector
from ebrm_system.inference.turboquant_kv import (
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
    "QJLConfig",
    "QJLProjector",
    "generate_candidates",
    "langevin_step",
    "select_diverse",
]
