"""Loader for the organizers' official scorer (``track3/evaluate.py``).

We deliberately *delegate* all scoring to the challenge-provided evaluator instead
of maintaining a parallel implementation — that guarantees our local numbers match
the leaderboard and can never silently drift. The scorer ships with the TAR dataset;
we keep the canonical copy at ``track3/evaluate.py`` (the single source every code
path references). To adopt an updated scorer, replace that one file.

It is loaded by path (rather than ``import track3.evaluate``) to preserve the
drop-in semantics and the per-path cache.
"""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from types import ModuleType

# the canonical scorer lives alongside this module, inside the track3 package.
DEFAULT_EVALUATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py")


@lru_cache(maxsize=4)
def load_official(path: str = DEFAULT_EVALUATE_PATH) -> ModuleType:
    """Import and cache the official evaluate.py module from ``path``."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Official scorer not found at {path}. Place the challenge's "
            "evaluate.py at track3/evaluate.py (it ships with the TAR dataset).")
    spec = importlib.util.spec_from_file_location("tar_official_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
