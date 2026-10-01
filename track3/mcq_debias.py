"""MCQ permutation debias: pool first-token letter scores over cyclic rotations of the options.

Only the max_tokens=1 scoring pass sees permuted questions; generation prompts stay verbatim.
"""
from __future__ import annotations

import re

# One option per line: "A) text" (test) or "A. text" (train).
_OPTION_LINE = re.compile(r"^(\s*)([A-D])([).])(\s*)(.+?)\s*$", re.MULTILINE)


def permuted_question(question: str, shift: int):
    """Rotate option texts by ``shift`` slots.

    Returns (question, {slot_letter: original_letter}), or None unless the question
    has exactly the 4 options A-D.
    """
    ms = list(_OPTION_LINE.finditer(question or ""))
    letters = [m.group(2) for m in ms]
    if len(ms) != 4 or sorted(letters) != ["A", "B", "C", "D"]:
        return None
    shift %= 4
    texts = [m.group(5) for m in ms]
    mapping = {}
    out, last = [], 0
    for i, m in enumerate(ms):
        j = (i + shift) % 4
        mapping[letters[i]] = letters[j]
        out.append(question[last:m.start(5)])
        out.append(texts[j])
        last = m.end(5)
    out.append(question[last:])
    return "".join(out), mapping


def merge_permuted_votes(dists: list[dict], mappings: list[dict]) -> dict:
    """Average per-permutation {slot_letter: prob} back in original letters.

    Empty dists (no logprobs) are skipped; returns {} when nothing scored.
    """
    score: dict[str, float] = {}
    n = 0
    for dist, mapping in zip(dists, mappings):
        if not dist:
            continue
        n += 1
        for slot, p in dist.items():
            orig = mapping.get(slot)
            if orig is not None:
                score[orig] = score.get(orig, 0.0) + p
    return {k: v / n for k, v in score.items()} if n else {}
