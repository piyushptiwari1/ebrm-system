"""Coconut-style latent recursion for EBRM seed latents.

Why this exists
---------------
`Coconut <https://arxiv.org/abs/2412.06769>`_ feeds the last hidden state of
an LLM directly back as the next input embedding, skipping the
detokenize→retokenize round trip that costs accuracy on multi-step
reasoning. The recurrent-depth approach
(`OpenReview D6o6Bwtq7h <https://openreview.net/forum?id=D6o6Bwtq7h>`_)
shows the same trick scales test-time compute by *unrolling* a recurrent
block — a 3.5B model with iterated depth matches a 50B model on reasoning.
LatentSeek (`bigai-nlco.github.io/LatentSeek
<https://bigai-nlco.github.io/LatentSeek/>`_) does the same in latent
space at inference time.

EBRM is already a latent-reasoning system: encoder → projector → Langevin
candidates → decoder. This module slots a Coconut-style recurrence between
``encoder`` and ``generate_candidates``: the seed latent is iterated
``max_steps`` times by a caller-supplied ``step_fn`` (typically a small
energy-gradient descent or a learned recurrent block) until either the
budget is exhausted or the energy has plateaued.

The recursion is pure-Python + NumPy, deterministic given a seed, and
torch-optional. Nothing here calls a torch model directly — bring your
own ``step_fn``. The native `inference.halt.PlateauHalt` is reused as the
stopping criterion.

Default ``step_fn``: :func:`gradient_step`, a finite-difference energy
descent that needs nothing but the existing ``EnergyFn`` callable. This
makes the feature usable today on the CPU mock pipeline; production
users will swap in a torch-backed step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ebrm_system.inference.halt import HaltPolicy, NeverHalt

LatentT = NDArray[np.float32]
EnergyFn = Callable[[LatentT], float]
StepFn = Callable[[LatentT, int], LatentT]
"""``(latent, step_index) -> next_latent``. Must be deterministic for a given
input. Implementations may close over a torch module, an EBRM block, or a
plain function — this module never inspects them."""


@dataclass(frozen=True)
class RecursionConfig:
    """Knobs for :func:`recurse_latent`."""

    max_steps: int = 0
    """Maximum number of recurrence steps. ``0`` disables the recursion."""

    step_size: float = 0.1
    """Used by :func:`gradient_step` only — the descent rate."""

    fd_eps: float = 1e-3
    """Finite-difference epsilon used by :func:`gradient_step`."""

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {self.max_steps}")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {self.step_size}")
        if self.fd_eps <= 0:
            raise ValueError(f"fd_eps must be > 0, got {self.fd_eps}")


@dataclass(frozen=True)
class RecursionResult:
    """Output of :func:`recurse_latent`."""

    latent: LatentT
    """The final latent (same shape as the seed)."""

    steps_run: int
    """Number of recurrence steps actually performed (<= ``max_steps``)."""

    energy_trajectory: tuple[float, ...]
    """Energy at each step (length ``steps_run + 1`` if energy_fn was
    provided, else empty)."""

    halted_early: bool
    """True iff the halt policy fired before ``max_steps``."""


def gradient_step(
    energy_fn: EnergyFn,
    *,
    step_size: float = 0.1,
    fd_eps: float = 1e-3,
) -> StepFn:
    """Build a finite-difference energy-descent :data:`StepFn`.

    The returned callable ignores its ``step_index`` argument and applies a
    single gradient-descent step on the energy surface using central finite
    differences. Pure-Python + NumPy; no torch.
    """

    def _step(latent: LatentT, _step: int) -> LatentT:
        z = np.asarray(latent, dtype=np.float32)
        grad = np.zeros_like(z)
        for i in range(z.shape[0]):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += fd_eps
            z_minus[i] -= fd_eps
            grad[i] = (energy_fn(z_plus) - energy_fn(z_minus)) / (2.0 * fd_eps)
        return (z - step_size * grad).astype(np.float32)

    return _step


def recurse_latent(
    seed: LatentT,
    step_fn: StepFn,
    *,
    config: RecursionConfig | None = None,
    energy_fn: EnergyFn | None = None,
    halt_policy: HaltPolicy | None = None,
) -> RecursionResult:
    """Run Coconut-style latent recursion on a seed latent.

    Parameters
    ----------
    seed
        Starting latent (1-D ``float32`` array).
    step_fn
        :data:`StepFn` to apply at each iteration.
    config
        Optional :class:`RecursionConfig`. ``max_steps == 0`` short-circuits
        the recursion and returns the seed unchanged.
    energy_fn
        Optional energy callable. When provided, the trajectory is recorded
        and the halt policy can use it.
    halt_policy
        Optional early-stop policy (default: :class:`NeverHalt`). The policy
        is reset inside the function so callers can reuse the same instance.

    Returns
    -------
    RecursionResult
        Final latent + audit trail.

    Notes
    -----
    The function makes no assumption about the structure of ``step_fn``; it
    is fully agnostic. This is what keeps the module torch-optional — a
    production caller wraps the recurrent block in a thin ``StepFn``
    closure.
    """
    cfg = config or RecursionConfig()
    halt = halt_policy or NeverHalt()
    halt.reset()

    z = np.asarray(seed, dtype=np.float32).copy()
    if z.ndim != 1:
        raise ValueError(f"seed must be 1-D, got shape {z.shape}")
    if cfg.max_steps == 0:
        return RecursionResult(latent=z, steps_run=0, energy_trajectory=(), halted_early=False)

    traj: list[float] = []
    if energy_fn is not None:
        traj.append(float(energy_fn(z)))

    halted_early = False
    steps_run = 0
    for step in range(cfg.max_steps):
        z = step_fn(z, step)
        z = np.asarray(z, dtype=np.float32)
        steps_run += 1
        if energy_fn is not None:
            e = float(energy_fn(z))
            traj.append(e)
            if halt.should_halt(step, e):
                halted_early = True
                break

    return RecursionResult(
        latent=z,
        steps_run=steps_run,
        energy_trajectory=tuple(traj),
        halted_early=halted_early,
    )


__all__ = [
    "EnergyFn",
    "RecursionConfig",
    "RecursionResult",
    "StepFn",
    "gradient_step",
    "recurse_latent",
]
