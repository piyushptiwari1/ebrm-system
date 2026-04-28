"""Tests for v0.26 reader/router upgrades."""

from __future__ import annotations

from benchmarks.reader.azure_llm import _READER_SYSTEM
from benchmarks.router import _RECOMMEND_CUES, classify_question


def test_reader_system_has_latest_wins_rule() -> None:
    assert "MOST RECENT" in _READER_SYSTEM
    assert "stale" in _READER_SYSTEM


def test_reader_system_has_abstention_safety_rule() -> None:
    assert "Abstention safety" in _READER_SYSTEM
    assert "I don't know" in _READER_SYSTEM


def test_reader_system_has_examples_for_latest_wins() -> None:
    # Concrete examples teach the model the rule
    assert "gravel bike" in _READER_SYSTEM
    assert "25:50" in _READER_SYSTEM


def test_router_recognises_any_tips() -> None:
    assert (
        classify_question(
            "My kitchen is a mess. Any tips for keeping it clean?",
            "single-session-preference",
        )
        == "recommendation"
    )


def test_router_recognises_what_do_you_think() -> None:
    assert (
        classify_question(
            "I'm trying to decide whether to buy a NAS now. What do you think?",
            "single-session-preference",
        )
        == "recommendation"
    )


def test_router_recognises_could_there_be_a_reason() -> None:
    assert (
        classify_question(
            "My bike performs better on Sunday rides. Could there be a reason?",
            "single-session-preference",
        )
        == "recommendation"
    )


def test_router_recognises_im_looking_for() -> None:
    assert (
        classify_question(
            "I'm looking for a new pair of running shoes.",
            "single-session-preference",
        )
        == "recommendation"
    )


def test_router_aggregation_still_wins_over_recommendation() -> None:
    # Aggregation cue must take priority
    assert (
        classify_question(
            "How many tips have you given me, and what should I do?",
            "single-session-preference",
        )
        == "aggregation"
    )


def test_recommendation_cue_list_grew() -> None:
    # Sanity: we kept the v0.23 cues and added new ones
    assert "can you recommend" in _RECOMMEND_CUES
    assert "any tips" in _RECOMMEND_CUES
    assert "what do you think" in _RECOMMEND_CUES
