"""Process Reward Model — scores each step of a reasoning trace.

v3 modules:
    * ``qjl_index`` — quantized latent index for warm-start retrieval.
    * ``prm_data`` — derive PRM training data from solve() traces
      (Athena-PRM weak/strong agreement labelling).
"""

from ebrm_system.reward.prm_data import PRMRecord, make_records, write_jsonl
from ebrm_system.reward.qjl_index import IndexConfig, LatentIndex

__all__ = [
    "IndexConfig",
    "LatentIndex",
    "PRMRecord",
    "make_records",
    "write_jsonl",
]
