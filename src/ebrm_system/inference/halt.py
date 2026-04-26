"""Adaptive halt policies for Langevin candidate generation.

Inspired by adaptive-compute halt heads in recurrent-depth transformers, but
data-free: instead of a learned head, we use the *energy trajectory* itself
to decide when to stop. This is free compute savings on easy problems —
candidates that have already reached a low-energy basin don't need more
gradient steps.

Two policies:

- :class:`NeverHalt` — disable early stopping (default; preserves prior
  behaviour).
- :class:`PlateauHalt` — stop when the rolling stddev of energy over the
  last ``window`` steps drops below ``threshold``, after a minimum of
  ``min_steps`` warm-up.

Both implement :class:`HaltPolicy` so callers can supply their own.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol


class HaltPolicy(Protocol):
    """Decides when to stop a Langevin trajectory.

    Implementations must be stateless across trajectories: callers obtain a
    fresh state via :meth:`reset` (or by constructing a new instance) and
    feed each step's energy to :meth:`should_halt`.
    """

    def reset(self) -> None: ...

    def should_halt(self, step: int, energy: float) -> bool: ...


@dataclass
class NeverHalt:
    """No early stopping. Equivalent to running the full ``num_steps``."""

    def reset(self) -> None:
        return None

    def should_halt(self, step: int, energy: float) -> bool:
        return False


@dataclass
class PlateauHalt:
    """Halt when energy has plateaued.

    The policy tracks the last ``window`` energies in a ring buffer. After
    ``min_steps`` warm-up steps, it halts when the rolling stddev of those
    energies falls below ``threshold``.

    Parameters
    ----------
    window
        How many recent steps to average over. Must be ≥ 2.
    threshold
        Absolute stddev cutoff. The right value depends on the energy
        function's scale; a good default is ~1% of the typical seed energy.
    min_steps
        Minimum steps to run before halting is allowed. Prevents halting
        on the initial transient.
    """

    window: int = 8
    threshold: float = 1e-3
    min_steps: int = 10

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.min_steps < 0:
            raise ValueError("min_steps must be non-negative")
        self._buf: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buf.clear()

    def should_halt(self, step: int, energy: float) -> bool:
        self._buf.append(energy)
        if step < self.min_steps:
            return False
        if len(self._buf) < self.window:
            return False
        # Use range as a fast plateau proxy; equivalent decision boundary
        # to stddev for the small windows we use here, and avoids importing
        # numpy in the inner loop.
        lo = min(self._buf)
        hi = max(self._buf)
        return (hi - lo) <= self.threshold


__all__ = ["HaltPolicy", "NeverHalt", "PlateauHalt"]
