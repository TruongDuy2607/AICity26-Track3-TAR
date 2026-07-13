"""MCQ permutation debias (PROVE side lever, PROVE.md §1).

LLM multiple-choice decisions carry position/letter bias: the same option text can
score differently depending on which letter slot it occupies. The W1 first-token
scoring pass is therefore repeated under R cyclic rotations of the option TEXTS,
per-permutation letter distributions are pooled OVER OPTION TEXT, and the average is
re-expressed in the original letters. Only the max_tokens=1 SCORING pass ever sees a
permuted question — generation passes always use the verbatim SFT prompt (the F6
register lesson), and the result plugs into the existing ``votes`` field so the
mcq<->mcq_openended pooling and the conditioned-explanation pass are unchanged.

Pure python (no torch/ms-swift) — unit-tested on the CPU dev box
(track3/test_mcq_debias.py); track3.infer consumes it behind ``--mcq-permute``.
"""
from __future__ import annotations

import re

# One option per line, "A) text" (test) or "A. text" (train) — same surface form
# structural.question_options parses.
_OPTION_LINE = re.compile(r"^(\s*)([A-D])([).])(\s*)(.+?)\s*$", re.MULTILINE)


def permuted_question(question: str, shift: int):
    """The question with option TEXTS cyclically rotated by ``shift`` slots, plus the
    slot->original-letter mapping ({slot_letter: original_letter of the text now in
    that slot}). Letter markers and separators stay in place; only the texts move.

    Returns None when the question does not carry exactly 4 well-formed distinct
    option lines (malformed items silently fall back to single-pass scoring).
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
    """Average the per-permutation letter distributions back in ORIGINAL letters.

    ``dists[k]`` is the first-token {slot_letter: prob} for permutation k;
    ``mappings[k]`` its {slot_letter: original_letter}. Empty dists (backend gave no
    logprobs for that pass) are skipped so a partial failure degrades to the
    permutations that did score. Returns {} when nothing scored.
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
