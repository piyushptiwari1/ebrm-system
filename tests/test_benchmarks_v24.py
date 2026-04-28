"""Tests for v0.24 question router."""

from __future__ import annotations

from benchmarks.router import classify_question, top_k_for


def test_classify_aggregation() -> None:
    assert classify_question("How many model kits have I bought?", "multi-session") == "aggregation"
    assert classify_question("How much total did I spend?", "multi-session") == "aggregation"
    assert classify_question("List all the books I read.", "multi-session") == "aggregation"


def test_classify_recommendation() -> None:
    assert (
        classify_question("Can you suggest a hotel in Miami?", "single-session-preference")
        == "recommendation"
    )
    assert (
        classify_question("What should I watch tonight?", "single-session-preference")
        == "recommendation"
    )


def test_classify_temporal_falls_through_to_type() -> None:
    assert (
        classify_question("Which happened first, the dentist or the doctor?", "temporal-reasoning")
        == "temporal"
    )


def test_classify_general_default() -> None:
    assert classify_question("What car do I drive?", "single-session-user") == "general"
    assert classify_question("Who is my manager?", "multi-session") == "general"


def test_classify_aggregation_overrides_temporal_type() -> None:
    # An aggregation cue takes precedence over the question_type tag —
    # the runner needs more candidates regardless.
    assert (
        classify_question("How many days passed between the two events?", "temporal-reasoning")
        == "aggregation"
    )


def test_top_k_for_aggregation_floor_20() -> None:
    assert top_k_for("aggregation", default=5) == 20
    assert top_k_for("aggregation", default=25) == 25


def test_top_k_for_temporal_ceiling_5() -> None:
    assert top_k_for("temporal", default=10) == 5
    assert top_k_for("temporal", default=3) == 3


def test_top_k_for_general_uses_default() -> None:
    assert top_k_for("general", default=10) == 10
    assert top_k_for("recommendation", default=10) == 10
