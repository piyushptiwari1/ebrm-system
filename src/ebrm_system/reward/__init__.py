"""Process Reward Model — scores each step of a reasoning trace.

.. note::
   This subpackage is **experimental** (see :mod:`ebrm_system._experimental`).
   Its API may change between minor releases without a deprecation cycle.

v3 modules:
    * ``qjl_index`` — quantized latent index for warm-start retrieval.
    * ``prm_data`` — derive PRM training data from solve() traces
      (Athena-PRM weak/strong agreement labelling).
"""

from ebrm_system._experimental import _warn_experimental

_warn_experimental("reward")

from ebrm_system.reward.prm_data import PRMRecord, make_records, write_jsonl  # noqa: E402
from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex  # noqa: E402

__all__ = [
    "IndexConfig",
    "LatentIndex",
    "PRMRecord",
    "make_records",
    "write_jsonl",
]
