"""Tests for the LongMemEval-style harness (v0.13)."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.longmemeval import (
    ALL_QUESTION_TYPES,
    LongMemEpisode,
    MemoryFact,
    default_memory,
    hash_embed,
    run_longmemeval,
    synth_longmemeval,
)

from ebrm_system.memory import TieredMemory, TieredMemoryConfig


class TestHashEmbed:
    def test_unit_norm(self) -> None:
        v = hash_embed("hello", dim=32)
        assert v.shape == (32,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_deterministic(self) -> None:
        a = hash_embed("hello world", dim=64, seed=7)
        b = hash_embed("hello world", dim=64, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_seed_changes_output(self) -> None:
        a = hash_embed("hello", dim=32, seed=0)
        b = hash_embed("hello", dim=32, seed=1)
        assert not np.array_equal(a, b)


class TestSynthGenerator:
    def test_balanced_across_types(self) -> None:
        eps = synth_longmemeval(seed=0, num_episodes=10)
        assert len(eps) == 10
        types = {ep.question_type for ep in eps}
        assert types == set(ALL_QUESTION_TYPES)

    def test_deterministic_given_seed(self) -> None:
        a = synth_longmemeval(seed=42, num_episodes=8)
        b = synth_longmemeval(seed=42, num_episodes=8)
        assert [ep.id for ep in a] == [ep.id for ep in b]
        assert [ep.answer for ep in a] == [ep.answer for ep in b]

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="num_episodes"):
            synth_longmemeval(seed=0, num_episodes=0)

    def test_rejects_empty_types(self) -> None:
        with pytest.raises(ValueError, match="types"):
            synth_longmemeval(seed=0, num_episodes=1, types=())

    def test_filtering_to_one_type(self) -> None:
        eps = synth_longmemeval(seed=0, num_episodes=5, types=("multi-session",))
        assert all(ep.question_type == "multi-session" for ep in eps)


class TestRunLongMemEval:
    def test_dim_mismatch_raises(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=64))
        with pytest.raises(ValueError, match="embed_dim"):
            run_longmemeval([], mem, embed_dim=32)

    def test_empty_episodes(self) -> None:
        mem = TieredMemory(TieredMemoryConfig(in_dim=32))
        result = run_longmemeval([], mem, embed_dim=32)
        assert result.total == 0
        assert result.correct == 0
        assert result.accuracy == 0.0

    def test_runs_and_reports_per_type(self) -> None:
        mem = default_memory(in_dim=64)
        eps = synth_longmemeval(seed=0, num_episodes=10)
        result = run_longmemeval(eps, mem, embed_dim=64)
        assert result.total == 10
        assert sum(result.per_type_counts.values()) == 10
        # Either everyone passed or accuracy bands are bounded.
        assert 0.0 <= result.accuracy <= 1.0
        for qt in ALL_QUESTION_TYPES:
            assert 0.0 <= result.accuracy_by_type[qt] <= 1.0

    def test_perfect_recall_on_trivial_episode(self) -> None:
        """One episode, one fact, exact-text question — must recall."""
        ep = LongMemEpisode(
            id="t-0",
            facts=(
                MemoryFact(
                    text="Alice likes rock climbing.",
                    session=0,
                    speaker="user",
                ),
            ),
            question="Alice likes rock climbing.",
            answer="rock climbing",
            question_type="single-session-user",
        )
        mem = TieredMemory(TieredMemoryConfig(in_dim=64))
        result = run_longmemeval([ep], mem, embed_dim=64)
        assert result.correct == 1
        assert result.accuracy == 1.0

    def test_knowledge_update_skips_superseded(self) -> None:
        """A superseded fact should not be written; the later fact should win."""
        ep = LongMemEpisode(
            id="ku-0",
            facts=(
                MemoryFact(
                    text="Bob lives in Lisbon.",
                    session=0,
                    speaker="user",
                    superseded_by=1,
                ),
                MemoryFact(text="Bob moved to Kyoto.", session=4, speaker="user"),
            ),
            question="Where does Bob currently live?",
            answer="Kyoto",
            question_type="knowledge-update",
        )
        mem = TieredMemory(TieredMemoryConfig(in_dim=64))
        result = run_longmemeval([ep], mem, embed_dim=64)
        assert result.correct == 1
        assert len(mem) == 1  # only the active fact written

    def test_default_memory_returns_tiered(self) -> None:
        mem = default_memory(in_dim=128)
        assert mem.config.in_dim == 128
        assert isinstance(mem, TieredMemory)
