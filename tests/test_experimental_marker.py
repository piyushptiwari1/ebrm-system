"""Tests for the experimental-API marker."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from ebrm_system._experimental import (
    _WARNED,
    EBRMExperimentalWarning,
    _warn_experimental,
)


@pytest.fixture(autouse=True)
def _reset_warned() -> None:
    """Each test starts with a clean ``_WARNED`` set so it can re-trigger."""
    _WARNED.clear()


def test_marker_is_user_warning_subclass() -> None:
    assert issubclass(EBRMExperimentalWarning, UserWarning)


def test_warn_experimental_emits_once_per_module() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EBRMExperimentalWarning)
        _warn_experimental("foo")
        _warn_experimental("foo")
        _warn_experimental("bar")
    msgs = [w for w in caught if issubclass(w.category, EBRMExperimentalWarning)]
    assert len(msgs) == 2
    assert "foo" in str(msgs[0].message)
    assert "bar" in str(msgs[1].message)


def test_warn_experimental_message_mentions_policy() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EBRMExperimentalWarning)
        _warn_experimental("baz")
    assert "experimental" in str(caught[0].message).lower()
    assert "_experimental" in str(caught[0].message)


@pytest.mark.parametrize(
    "module_name",
    ["ebrm_system.core", "ebrm_system.inference", "ebrm_system.memory", "ebrm_system.reward"],
)
def test_experimental_packages_warn_on_import(module_name: str) -> None:
    # Force a fresh import to re-trigger the package __init__.
    for mod in list(sys.modules):
        if mod == module_name or mod.startswith(module_name + "."):
            del sys.modules[mod]
    _WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EBRMExperimentalWarning)
        importlib.import_module(module_name)
    matched = [w for w in caught if issubclass(w.category, EBRMExperimentalWarning)]
    assert len(matched) >= 1
    assert module_name.split(".", 1)[1] in str(matched[0].message)


@pytest.mark.parametrize(
    "module_name",
    ["ebrm_system.verifiers", "ebrm_system.voting", "ebrm_system.intent", "ebrm_system.longmem"],
)
def test_stable_packages_do_not_warn(module_name: str) -> None:
    for mod in list(sys.modules):
        if mod == module_name or mod.startswith(module_name + "."):
            del sys.modules[mod]
    _WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EBRMExperimentalWarning)
        importlib.import_module(module_name)
    matched = [w for w in caught if issubclass(w.category, EBRMExperimentalWarning)]
    assert matched == []


def test_silencing_filter_works() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("ignore", category=EBRMExperimentalWarning)
        _warn_experimental("silenced")
    assert all(not issubclass(w.category, EBRMExperimentalWarning) for w in caught)
