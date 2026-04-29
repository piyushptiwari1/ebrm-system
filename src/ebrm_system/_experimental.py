"""Experimental-API marker for provisional modules.

Following PEP 411 conventions, certain ebrm-system subpackages are marked
*experimental*: their public API may change between minor releases without
a deprecation cycle, and they are not part of the stability guarantees that
apply to the rest of the package.

Importing an experimental module emits :class:`EBRMExperimentalWarning`
once per module per process. The warning is intentionally a subclass of
``UserWarning`` (not ``DeprecationWarning``) so that it surfaces by default
rather than being filtered.

Stable surface (v0.30):
  * :mod:`ebrm_system.verifiers`
  * :mod:`ebrm_system.voting`
  * :mod:`ebrm_system.intent`
  * :mod:`ebrm_system.longmem`

Experimental surface (v0.30):
  * :mod:`ebrm_system.core`
  * :mod:`ebrm_system.inference`
  * :mod:`ebrm_system.memory`
  * :mod:`ebrm_system.reward`

Silencing the warning::

    import warnings
    from ebrm_system._experimental import EBRMExperimentalWarning
    warnings.filterwarnings("ignore", category=EBRMExperimentalWarning)

The set of experimental modules is reviewed every minor release. A module
graduates to stable when (a) it has a benchmark-validated win on a
published baseline and (b) its public API has been unchanged for one full
release cycle.
"""

from __future__ import annotations

import warnings


class EBRMExperimentalWarning(UserWarning):
    """Issued when an experimental ebrm-system module is imported.

    See :mod:`ebrm_system._experimental` for policy.
    """


_WARNED: set[str] = set()


def _warn_experimental(module_name: str) -> None:
    """Emit :class:`EBRMExperimentalWarning` once per ``module_name``.

    Safe to call from a package ``__init__.py``. Subsequent imports in the
    same process are silent so we don't spam users who legitimately depend
    on a module.
    """
    if module_name in _WARNED:
        return
    _WARNED.add(module_name)
    warnings.warn(
        (
            f"ebrm_system.{module_name} is an experimental module: "
            "its API may change between minor releases without notice. "
            "See ebrm_system._experimental for the stability policy."
        ),
        category=EBRMExperimentalWarning,
        stacklevel=3,
    )
