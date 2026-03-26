"""
Centralized deferred imports for exam-conductor modules.

The exam-conductor directory uses a hyphen which is not a valid Python
identifier.  All ExamPen route files must use ``importlib.import_module``
to load subpackages.  This helper provides a single ``load(dotted_path)``
function and pre-built accessors so each route file doesn't repeat the
importlib boilerplate.

Usage::

    from api.v1._exampen_imports import load_exampen

    mod = load_exampen("pcr.storage")
    SolutionRepository = mod.SolutionRepository
"""

from __future__ import annotations

import importlib
from typing import Any


def load_exampen(subpath: str) -> Any:
    """Import a sub-module from the exam-conductor package.

    Parameters
    ----------
    subpath : str
        Dotted path relative to exam-conductor, e.g. ``"pcr.storage"``
        or ``"llm_gate.gate"``.

    Returns
    -------
    module
        The imported module object.

    Raises
    ------
    ImportError
        If the exam-conductor package or the requested sub-module is
        not available.
    """
    return importlib.import_module(f"exam-conductor.{subpath}")
