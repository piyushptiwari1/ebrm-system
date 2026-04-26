"""Torch-native Langevin step using true autograd gradients.

The numpy implementation in :mod:`ebrm_system.inference.candidates` uses
finite differences, which costs ``2 * d`` energy evaluations per step.
For the real EBRM model where ``energy_fn`` is a neural network, that is
prohibitively slow. This module provides a torch backend that:

    1. Treats ``energy_fn`` as a differentiable callable
       ``Tensor -> Tensor (scalar)``.
    2. Computes ``grad_s E(s)`` exactly via ``torch.autograd.grad``.
    3. Adds Gaussian noise with the same scale convention as the numpy
       version, so behaviour is consistent across backends.

It is **torch-optional** — importing this module fails with a clear
:class:`ImportError` if torch is missing, but the rest of
``ebrm_system.inference`` keeps working on numpy alone.

Device selection: pure-torch, no explicit ``.cuda()`` calls. The caller
moves their model and seed latent to the right device; this module
preserves whatever device/dtype the input tensor is on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "ebrm_system.inference.torch_langevin requires torch. "
        "Install with: pip install 'ebrm-system[model]'"
    ) from exc


TorchEnergyFn = Callable[["Tensor"], "Tensor"]


@dataclass(frozen=True)
class TorchCandidate:
    """A single torch latent candidate with its scalar energy."""

    latent: Tensor
    energy: float
    seed: int


def torch_langevin_step(
    s: Tensor,
    energy_fn: TorchEnergyFn,
    step_size: float,
    noise_scale: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """One step of Langevin dynamics with autograd gradient.

    Parameters
    ----------
    s
        Current latent. Any shape, any float dtype, any device.
    energy_fn
        Differentiable mapping ``Tensor -> Tensor`` returning a 0-dim tensor.
    step_size
        Gradient descent step (η in the Langevin SDE).
    noise_scale
        Stddev of the Gaussian noise added each step.
    generator
        Optional :class:`torch.Generator` for reproducibility. Must live on
        the same device as ``s`` (torch convention).
    """
    s_in = s.detach().requires_grad_(True)
    e = energy_fn(s_in)
    if e.ndim != 0:
        raise ValueError(f"energy_fn must return a 0-dim tensor, got shape {tuple(e.shape)}")
    (grad,) = torch.autograd.grad(e, s_in, create_graph=False)
    noise = torch.randn(s.shape, generator=generator, device=s.device, dtype=s.dtype)
    return (s_in.detach() - step_size * grad + noise_scale * noise).detach()


def generate_torch_candidates(
    seed_latent: Tensor,
    energy_fn: TorchEnergyFn,
    num_candidates: int = 32,
    num_steps: int = 50,
    step_size: float = 0.01,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> list[TorchCandidate]:
    """Generate ``num_candidates`` low-energy latents using torch autograd.

    Each candidate is a fresh trajectory from a perturbed copy of ``seed_latent``.
    Results are sorted ascending by energy. Reproducible given ``seed``.
    """
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    device = seed_latent.device
    dtype = seed_latent.dtype

    # Master generator on the right device for distinct per-candidate seeds.
    master_gen = torch.Generator(device=device).manual_seed(seed)
    sub_seeds: list[int] = []
    with torch.no_grad():
        for _ in range(num_candidates):
            sub_seeds.append(int(torch.randint(0, 2**31 - 1, (1,), generator=master_gen).item()))

    final: list[TorchCandidate] = []
    for sub_seed in sub_seeds:
        gen = torch.Generator(device=device).manual_seed(sub_seed)
        with torch.no_grad():
            s = seed_latent.detach() + noise_scale * torch.randn(
                seed_latent.shape, generator=gen, device=device, dtype=dtype
            )
        for _ in range(num_steps):
            s = torch_langevin_step(s, energy_fn, step_size, noise_scale, generator=gen)
        with torch.no_grad():
            energy = float(energy_fn(s).item())
        final.append(TorchCandidate(latent=s.detach(), energy=energy, seed=sub_seed))

    return sorted(final, key=lambda c: c.energy)
