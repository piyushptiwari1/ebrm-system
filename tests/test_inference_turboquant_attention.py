"""Tests for torch TurboQuant attention."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ebrm_system.inference.turboquant_attention import (  # noqa: E402
    _hadamard_matrix_torch,
    quantize_kv_torch,
    turboquant_attention,
)


def test_hadamard_matrix_torch_orthonormal() -> None:
    h = _hadamard_matrix_torch(64, device=torch.device("cpu"), dtype=torch.float32)
    eye = torch.eye(64, dtype=torch.float32)
    torch.testing.assert_close(h @ h, eye, atol=1e-5, rtol=1e-5)


def test_hadamard_matrix_torch_requires_power_of_two() -> None:
    with pytest.raises(ValueError):
        _hadamard_matrix_torch(48, device=torch.device("cpu"), dtype=torch.float32)


def test_quantize_kv_torch_round_trip_4bit() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 64)  # batch, heads, seq, head_dim
    c = quantize_kv_torch(x, bits=4, rotate=True)
    h = _hadamard_matrix_torch(64, device=x.device, dtype=x.dtype)
    x_hat = c.decompress(h)
    err = (x - x_hat).norm() / x.norm()
    assert err.item() < 0.15


def test_quantize_kv_torch_8bit_better_than_4bit() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 64)
    c4 = quantize_kv_torch(x, bits=4, rotate=False)
    c8 = quantize_kv_torch(x, bits=8, rotate=False)
    e4 = (x - c4.decompress()).norm() / x.norm()
    e8 = (x - c8.decompress()).norm() / x.norm()
    assert e8 < e4


def test_quantize_kv_torch_invalid_bits() -> None:
    with pytest.raises(ValueError):
        quantize_kv_torch(torch.zeros(4, 8), bits=3)


def test_turboquant_attention_matches_full_precision_within_tol() -> None:
    """Compressed-KV attention should be close to full-precision SDPA."""
    torch.manual_seed(0)
    b, h, s, d = 1, 2, 16, 32
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)

    # Full-precision reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # 8-bit Turboquant path (high precision -> tight tolerance)
    k_q = quantize_kv_torch(k, bits=8, rotate=True)
    v_q = quantize_kv_torch(v, bits=8, rotate=True)
    out = turboquant_attention(q, k_q, v_q)

    assert out.shape == ref.shape
    rel = (out - ref).norm() / ref.norm()
    assert rel.item() < 0.1, f"relative attention error {rel.item():.4f}"


def test_turboquant_attention_causal_masking() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 1, 8, 16)
    k = torch.randn(1, 1, 8, 16)
    v = torch.randn(1, 1, 8, 16)
    k_q = quantize_kv_torch(k, bits=8, rotate=False)
    v_q = quantize_kv_torch(v, bits=8, rotate=False)
    out = turboquant_attention(q, k_q, v_q, is_causal=True)
    assert out.shape == (1, 1, 8, 16)


def test_turboquant_attention_preserves_dtype() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8, dtype=torch.float64)
    k = torch.randn(1, 1, 4, 8, dtype=torch.float64)
    v = torch.randn(1, 1, 4, 8, dtype=torch.float64)
    k_q = quantize_kv_torch(k, bits=8, rotate=False)
    v_q = quantize_kv_torch(v, bits=8, rotate=False)
    out = turboquant_attention(q, k_q, v_q)
    assert out.dtype == torch.float64
