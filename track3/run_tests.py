"""Minimal CPU test runner: runs every ``test_*`` function in track3/test_*.py.

Run:  python -m track3.run_tests [substring ...]
"""
from __future__ import annotations

import glob
import importlib
import os
import sys
import traceback


def main() -> None:
    filters = sys.argv[1:]
    here = os.path.dirname(__file__)
    passed = failed = 0
    failures: list[str] = []
    for path in sorted(glob.glob(os.path.join(here, "test_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if filters and not any(f in name for f in filters):
            continue
        mod = importlib.import_module(f"track3.{name}")
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except Exception:
                failed += 1
                failures.append(f"{name}.{attr}")
                print(f"FAIL {name}.{attr}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
