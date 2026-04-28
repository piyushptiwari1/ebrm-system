"""EBRM System — production reasoning pipeline.

A full reasoning system built on the Energy-Based Reasoning Model (EBRM) v2
architecture. Adds:

- Intent + difficulty classification (routing)
- Hierarchical latent reasoning (Coconut-inspired thought sequences)
- Adaptive test-time compute (difficulty-aware Langevin budget)
- Process reward model (step-level scoring)
- External symbolic verification (SymPy, Python exec, regex)
- Self-consistency voting (majority vote over parallel traces)

For the original EBRM v2 reference implementation (paper code), see:
    https://github.com/piyushptiwari1/ebrm
"""

__version__ = "0.26.0"
__author__ = "Piyush Tiwari"
__license__ = "Apache-2.0"
