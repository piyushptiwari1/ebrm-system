"""Vendored EBRM v2 architecture (encoder-free).

This module mirrors the trainable heads from the upstream training repo
(``piyushptiwari/ebrm`` → ``ebrm/model.py``) so that checkpoints saved as
``ebrm_inference.pt`` (and uploaded to ``piyushptiwari/ebrm-v2-qwen3-4b``)
load cleanly via ``state_dict``.

Only the modules required for **scoring** are re-implemented here:

* :class:`WeightedPooler` — 4-head attention pool over encoder hidden states.
* :class:`GatedProjector` — encoder hidden_dim → 768-d latent.
* :class:`CrossAttentionEnergy` — ``(solution, problem) → scalar`` energy.

The base encoder (Qwen3-4B 4-bit) is loaded separately via
``transformers.AutoModelForCausalLM`` inside :class:`EBRMScorer`. The
``answer_decoder`` and ``confidence_head`` are *not* needed for the
verifier path and are skipped on load.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import spectral_norm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ebrm_system.verifiers._ebrm_arch requires torch. "
        "Install with: pip install 'ebrm-system[model]'"
    ) from exc


class WeightedPooler(nn.Module):
    """Multi-head attention-weighted pooling over encoder hidden states."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1, bias=False) for _ in range(num_heads)]
        )
        self.combine = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_list = []
        for head in self.attention_heads:
            weights = head(hidden_states).squeeze(-1)
            weights = weights.masked_fill(~attention_mask.bool(), float("-inf"))
            weights = F.softmax(weights, dim=-1)
            pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
            pooled_list.append(pooled)
        combined = torch.cat(pooled_list, dim=-1)
        return self.norm(self.combine(combined))


class GatedProjector(nn.Module):
    """Gated MLP projector with residual connection."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        intermediate = latent_dim * 3
        self.up = nn.Linear(hidden_dim, intermediate)
        self.gate = nn.Linear(hidden_dim, intermediate)
        self.down = nn.Linear(intermediate, latent_dim)
        self.norm1 = nn.LayerNorm(intermediate)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.residual = (
            nn.Linear(hidden_dim, latent_dim) if hidden_dim != latent_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.silu(self.gate(x)) * self.up(x))
        out = self.down(h)
        return self.norm2(out + self.residual(x))


class CrossAttentionEnergy(nn.Module):
    """Multi-component energy: cross-attn + local + global L2 + bilinear."""

    def __init__(
        self, latent_dim: int, num_heads: int = 8, hidden_dim: int = 384
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.cross_norm = nn.LayerNorm(latent_dim)
        self.compatibility = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )
        self.local_energy = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )
        self.bilinear = nn.Bilinear(latent_dim, latent_dim, 1)
        self.lambda_cross = nn.Parameter(torch.tensor(1.0))
        self.lambda_local = nn.Parameter(torch.tensor(0.5))
        self.lambda_global = nn.Parameter(torch.tensor(0.3))
        self.lambda_bilinear = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, solution_state: torch.Tensor, problem_state: torch.Tensor
    ) -> torch.Tensor:
        sol_seq = solution_state.unsqueeze(1)
        prob_seq = problem_state.unsqueeze(1)
        cross_out, _ = self.cross_attn(sol_seq, prob_seq, prob_seq)
        cross_out = self.cross_norm(cross_out + sol_seq).squeeze(1)
        e_cross = self.compatibility(cross_out).squeeze(-1)
        e_local = self.local_energy(solution_state).squeeze(-1)
        e_global = (solution_state - problem_state).pow(2).mean(dim=-1)
        e_bilinear = self.bilinear(solution_state, problem_state).squeeze(-1)
        return (
            self.lambda_cross * e_cross
            + self.lambda_local * e_local
            + self.lambda_global * e_global
            + self.lambda_bilinear * e_bilinear
        )
