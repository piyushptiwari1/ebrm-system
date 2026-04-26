"""Tests for QJL projector."""

from __future__ import annotations

import numpy as np
import pytest

from ebrm_system.inference.qjl import QJLConfig, QJLProjector


def test_qjl_config_validation() -> None:
    with pytest.raises(ValueError, match="in_dim"):
        QJLConfig(in_dim=0)
    with pytest.raises(ValueError, match="out_bits"):
        QJLConfig(in_dim=64, out_bits=7)


def test_qjl_project_shape() -> None:
    proj = QJLProjector(QJLConfig(in_dim=64, out_bits=128, seed=0))
    x = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    code = proj.project(x)
    assert code.shape == (16,)  # 128 bits / 8
    assert code.dtype == np.uint8


def test_qjl_batch_matches_per_vector() -> None:
    proj = QJLProjector(QJLConfig(in_dim=32, out_bits=64, seed=42))
    rng = np.random.default_rng(0)
    x_batch = rng.standard_normal((10, 32)).astype(np.float32)
    batch = proj.project_batch(x_batch)
    for i in range(10):
        np.testing.assert_array_equal(batch[i], proj.project(x_batch[i]))


def test_qjl_distance_preservation() -> None:
    """Cosine estimate should match true cosine within ~0.1 at m=2048."""
    proj = QJLProjector(QJLConfig(in_dim=128, out_bits=2048, seed=0))
    rng = np.random.default_rng(1)
    a = rng.standard_normal(128).astype(np.float32)
    b = rng.standard_normal(128).astype(np.float32)
    true_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    est = proj.estimate_cosine(proj.project(a), proj.project(b))
    assert abs(true_cos - est) < 0.1


def test_qjl_self_similarity_is_one() -> None:
    proj = QJLProjector(QJLConfig(in_dim=64, out_bits=512, seed=0))
    x = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    code = proj.project(x)
    assert proj.estimate_cosine(code, code) == pytest.approx(1.0)


def test_qjl_wrong_shape_raises() -> None:
    proj = QJLProjector(QJLConfig(in_dim=64, out_bits=128))
    with pytest.raises(ValueError):
        proj.project(np.zeros(32, dtype=np.float32))
    with pytest.raises(ValueError):
        proj.project_batch(np.zeros((4, 32), dtype=np.float32))
