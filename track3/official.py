"""Loader for the organizers' official scorer (``track3/evaluate.py``)."""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from types import ModuleType

DEFAULT_EVALUATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py")


@lru_cache(maxsize=4)
def load_official(path: str = DEFAULT_EVALUATE_PATH) -> ModuleType:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Official scorer not found at {path}. Place the challenge's "
            "evaluate.py at track3/evaluate.py (it ships with the TAR dataset).")
    spec = importlib.util.spec_from_file_location("tar_official_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
