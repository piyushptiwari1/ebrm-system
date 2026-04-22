"""Core reasoning primitives — hierarchical latents, energy functions.

This module will host the v3 reasoning core once EBRM v2 training completes.
The v2 reference implementation is available at:
    https://github.com/piyushptiwari1/ebrm

Planned components:
    HierarchicalLatentReasoner  — thought sequence s_0 ... s_T
    StepwiseEnergy              — per-step energy with consistency term
    AdaptiveLangevin            — difficulty-aware compute budget
"""
