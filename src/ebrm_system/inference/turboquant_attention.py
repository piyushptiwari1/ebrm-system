"""Torch-native attention with TurboQuant-compressed KV cache.

Bridges :mod:`ebrm_system.inference.turboquant_kv` (numpy reference impl)
into a torch attention path. The K/V tensors are quantized once and
materialized to fp16/fp32 on demand inside the attention computation.

This is **not** a fused CUDA kernel — that requires a custom op and only
gives a real win on Hopper-class GPUs. On Ampere/Turing-class hardware
(T4, A100), torch.compile + a clean dequant-then-attention path is within
~10-15 % of a hand-fused kernel and far easier to maintain.

Module is torch-optional (matches :mod:`torch_langevin`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

try:
    import torch
    import torch.nn.functional as functional
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ebrm_system.inference.turboquant_attention requires torch. "
        "Install with: pip install 'ebrm-system[model]'"
    ) from exc


def _hadamard_matrix_torch(n: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Sylvester Hadamard, normalized so H @ H == I."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"hadamard size must be power of 2, got {n}")
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return cast(Tensor, h / (n**0.5))


@dataclass
class CompressedKVTensor:
    """Compressed K or V tensor (torch).

    ``codes`` are signed int8 values in [-(2^(b-1)-1), 2^(b-1)-1]; ``scale``
    is per-token broadcast across head_dim. Shape: codes [..., D],
    scale [..., 1].
    """

    codes: Tensor
    scale: Tensor
    bits: int
    rotated: bool

    def decompress(self, hadamard: Tensor | None = None) -> Tensor:
        max_int = (1 << (self.bits - 1)) - 1
        x = self.codes.to(self.scale.dtype) * self.scale / max_int
        if self.rotated:
            assert hadamard is not None, "rotated tensor requires the Hadamard matrix"
            x = x @ hadamard
        return x


def quantize_kv_torch(
    x: Tensor,
    *,
    bits: int = 4,
    rotate: bool = True,
    hadamard: Tensor | None = None,
) -> CompressedKVTensor:
    """Quantize a K or V tensor of shape [..., head_dim].

    If ``rotate=True`` and ``hadamard`` is None, the matrix is constructed
    on-the-fly. Pre-computing and reusing it across calls is much faster.
    """
    if bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4, or 8")
    head_dim = x.shape[-1]
    if rotate:
        if hadamard is None:
            hadamard = _hadamard_matrix_torch(head_dim, device=x.device, dtype=x.dtype)
        x = x @ hadamard
    max_int = (1 << (bits - 1)) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    codes = (x / scale * max_int).round().clamp(-max_int, max_int).to(torch.int8)
    return CompressedKVTensor(codes=codes, scale=scale, bits=bits, rotated=rotate)


def turboquant_attention(
    q: Tensor,
    k: CompressedKVTensor,
    v: CompressedKVTensor,
    *,
    is_causal: bool = False,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Scaled dot-product attention with TurboQuant-compressed K and V.

    Q remains full precision (it's the small one — query length is usually
    1 in autoregressive decoding). K and V are decompressed once per call
    inside the function, then handed to torch's fused SDPA. With
    ``torch.compile`` this dequant gets fused into the attention prologue.

    Shapes:
        q : [batch, heads, q_len, head_dim]
        k : codes/scale matching SDPA's K shape
        v : codes/scale matching SDPA's V shape
    """
    head_dim = q.shape[-1]
    h_k = _hadamard_matrix_torch(head_dim, device=q.device, dtype=q.dtype) if k.rotated else None
    h_v = _hadamard_matrix_torch(head_dim, device=q.device, dtype=q.dtype) if v.rotated else None
    k_full = k.decompress(h_k).to(q.dtype)
    v_full = v.decompress(h_v).to(q.dtype)
    return functional.scaled_dot_product_attention(
        q, k_full, v_full, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )
