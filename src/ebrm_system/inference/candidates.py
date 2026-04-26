"""Multi-seed candidate generator for self-consistency scoring.

Generates N candidate latent states from the same problem encoding by running
independent stochastic processes (Langevin or Gaussian noise injection) from
different random seeds. Candidates are returned as a fixed-shape numpy array
so downstream verifiers, voters, and the QJL latent index can operate on
them without depending on torch.

The actual EBRM model is plugged in via a callable (``encode_fn``) so this
module stays torch-optional and unit-testable on CPU with mocks.

Warm-start retrieval
--------------------
Optionally, a :class:`~ebrm_system.reward.qjl_index.LatentIndex` can be passed
in. When supplied, the first ``warmstart_k`` candidates are seeded from the
top-k QJL-nearest cached solution latents instead of fresh Gaussian
perturbations. This dramatically reduces the number of Langevin steps needed
to reach a low-energy attractor for problems that resemble previously seen
ones — the core idea behind retrieval-augmented self-consistency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ebrm_system.reward.qjl_index import LatentIndex

LatentT = NDArray[np.float32]
EnergyFn = Callable[[LatentT], float]


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for the candidate generator."""

    num_candidates: int = 32
    num_steps: int = 50
    step_size: float = 0.01
    noise_scale: float = 0.05
    seed: int = 0
    warmstart_k: int = 0
    """How many of ``num_candidates`` to seed from the latent index (if any).

    Must satisfy 0 <= warmstart_k <= num_candidates. Set to 0 to disable
    retrieval and fall back to pure Gaussian perturbation.
    """

    def __post_init__(self) -> None:
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if self.warmstart_k < 0 or self.warmstart_k > self.num_candidates:
            raise ValueError("warmstart_k must be in [0, num_candidates]")


@dataclass(frozen=True)
class Candidate:
    """A single latent candidate with its scalar energy."""

    latent: LatentT
    energy: float
    seed: int
    warmstart: bool = False
    """True if this candidate was seeded from the QJL latent index."""


def langevin_step(
    s: LatentT,
    energy_fn: EnergyFn,
    step_size: float,
    noise_scale: float,
    rng: np.random.Generator,
    eps: float = 1e-3,
) -> LatentT:
    """One step of Langevin dynamics with finite-difference gradient.

    Used here because ``energy_fn`` is treated as a black box (no autograd).
    For the real EBRM model in v3.1, replace with a torch-native step that
    uses ``torch.autograd.grad``.
    """
    grad = np.empty_like(s)
    e0 = energy_fn(s)
    for i in range(s.shape[0]):
        s_pert = s.copy()
        s_pert[i] += eps
        grad[i] = (energy_fn(s_pert) - e0) / eps
    noise = rng.standard_normal(s.shape, dtype=np.float32) * noise_scale
    out: NDArray[np.float32] = (s - step_size * grad + noise).astype(np.float32, copy=False)
    return out


def generate_candidates(
    seed_latent: LatentT,
    energy_fn: EnergyFn,
    config: CandidateConfig | None = None,
    index: LatentIndex | None = None,
) -> list[Candidate]:
    """Generate ``num_candidates`` independent low-energy latents.

    Each candidate starts from ``seed_latent`` plus a fresh Gaussian
    perturbation, then runs ``num_steps`` of Langevin dynamics.

    If ``index`` is provided and ``config.warmstart_k > 0``, the first
    ``warmstart_k`` candidates are seeded from the top-k QJL-nearest cached
    latents (whose payloads must themselves be ``np.ndarray`` latents).
    """
    cfg = config or CandidateConfig()
    base_rng = np.random.default_rng(cfg.seed)
    candidates: list[Candidate] = []

    warm_seeds: list[LatentT] = []
    if cfg.warmstart_k > 0 and index is not None and len(index) > 0:
        results = index.search(seed_latent, k=cfg.warmstart_k)
        for _sim, payload in results:
            if isinstance(payload, np.ndarray) and payload.shape == seed_latent.shape:
                warm_seeds.append(payload.astype(np.float32, copy=False))

    for k in range(cfg.num_candidates):
        sub_seed = int(base_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        warm = k < len(warm_seeds)
        base = warm_seeds[k] if warm else seed_latent
        s = base + rng.standard_normal(seed_latent.shape, dtype=np.float32) * cfg.noise_scale
        for _ in range(cfg.num_steps):
            s = langevin_step(s, energy_fn, cfg.step_size, cfg.noise_scale, rng)
        candidates.append(
            Candidate(
                latent=s,
                energy=float(energy_fn(s)),
                seed=sub_seed,
                warmstart=warm,
            )
        )

    return sorted(candidates, key=lambda c: c.energy)
