#!/usr/bin/env python3
"""Check that track3.tasks extractors agree with the official track3/evaluate.py.

Run: python -m track3.check_eval
"""
import os
import sys

from track3.official import load_official
from track3 import tasks as T

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_JSON = os.path.join(_REPO_ROOT, "data", "test", "test.json")
_EXAMPLE_SUB = os.path.join(_REPO_ROOT, "data", "test", "submission.example.csv")


CASES = [
    "Yes", "No", "Yes.", "No, the vehicle does not turn.", "yes it does",
    "It is clearly no.", "A", "A)", "A) Rear-end collision", "(B)",
    "The answer is C.", "D. Rollover", "maybe", "", "E) none", "I think B) side",
    "no anomaly is present", "Yes - a collision occurs",
]


def main() -> int:
    o = load_official()
    bad = 0
    for c in CASES:
        if T.extract_yesno(c) != o._extract_yesno(c):
            print(f"  yesno mismatch {c!r}: ours={T.extract_yesno(c)} official={o._extract_yesno(c)}")
            bad += 1
        if T.extract_letter(c) != o._extract_letter(c):
            print(f"  letter mismatch {c!r}: ours={T.extract_letter(c)} official={o._extract_letter(c)}")
            bad += 1

    # the official validator must accept the shape make_submission produces
    res = o.evaluate(_TEST_JSON, _EXAMPLE_SUB, allow_missing=True)
    ok_validate = res.get("mode") == "validate"

    print(f"extractor agreement: {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    print(f"official validate(example): {'OK' if ok_validate else 'FAIL'}")
    if bad or not ok_validate:
        return 1
    print("\n==> eval consistency: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
