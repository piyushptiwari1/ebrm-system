"""Tests for TurboQuant-style KV-cache compressor."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.inference.turboquant_kv import (
    CompressedKV,
    KVCacheCompressor,
    KVQuantConfig,
    _hadamard_matrix,
)


def test_kv_quant_config_validation() -> None:
    with pytest.raises(ValueError, match="bits"):
        KVQuantConfig(bits=3)


def test_hadamard_orthogonality() -> None:
    h = _hadamard_matrix(64)
    assert h.shape == (64, 64)
    np.testing.assert_allclose(h @ h, np.eye(64), atol=1e-5)


def test_hadamard_requires_power_of_two() -> None:
    with pytest.raises(ValueError):
        _hadamard_matrix(48)


def test_compress_decompress_roundtrip_4bit() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 16, 64)).astype(np.float32)  # batch, heads, seq, dim
    comp = KVCacheCompressor(KVQuantConfig(bits=4, rotate=True))
    err = comp.round_trip_error(x)
    assert err < 0.15, f"4-bit round-trip error too large: {err}"


def test_compress_decompress_roundtrip_8bit_better_than_4bit() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 64)).astype(np.float32)
    err4 = KVCacheCompressor(KVQuantConfig(bits=4)).round_trip_error(x)
    err8 = KVCacheCompressor(KVQuantConfig(bits=8)).round_trip_error(x)
    assert err8 < err4


def test_compressed_kv_dataclass_fields() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    comp = KVCacheCompressor(KVQuantConfig(bits=4))
    c = comp.compress(x)
    assert isinstance(c, CompressedKV)
    assert c.codes.dtype == np.int8
    assert c.scale.shape == (4, 1)
    assert c.bits == 4
    assert c.compression_ratio == pytest.approx(4.0)


def test_compress_without_rotation() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    comp = KVCacheCompressor(KVQuantConfig(bits=8, rotate=False))
    err = comp.round_trip_error(x)
    assert err < 0.1
