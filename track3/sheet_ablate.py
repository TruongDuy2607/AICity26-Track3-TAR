"""Evidence-Sheet ablation transforms (paper §5.2 controlled analysis).

The whole point of these ablations is that they change **exactly one variable** —
the content of the Cross-Question Evidence Sheet — while every other part of the
inference chain (render template, length calibration, candidate pool, MBR select,
official grader) stays byte-identical to the shipped test pipeline. That isolation
is what makes a delta on the held-out split attributable to the sheet mechanism
rather than to a prompt/format change.

A transform is applied in ``text_dossier.run`` right after the per-clip
``EvidenceSheet`` is assembled (``clip_facts``) and before any rendering. It mutates
``clip["facts"]`` in place and returns a per-clip log (used by the override /
propagation analysis for E4). Every transform is a pure function of the assembled
sheets + a seed — no torch, no model, CPU-unit-testable.

Transform grammar (``--sheet-transform``):

  ``""`` / ``identity`` / ``full``   no change (B1 — ours).
  ``none``                            strip ALL content (B0 — no-evidence control).
  ``drop:FIELD``                      empty one field (leave-one-field-out, E2b).
  ``keep:FIELD``                      keep only one field (additive probe).
  ``swap``                            replace every clip's content with a DIFFERENT
                                      clip's sheet (B3 — wrong-but-plausible content).
  ``corrupt:FIELD[:P]``               inject a wrong-but-plausible value into FIELD
                                      with probability P (default 1.0), borrowed from
                                      a donor clip so register/length are preserved
                                      (E4 — override-vs-propagation).

FIELD is one of: event, window, cast, scene, cause, consequence, observations.
The question-derived fields (event, window, cast) are exact at test time; corrupting
them is only meaningful as a stress test and is logged as such.
"""
from __future__ import annotations

import dataclasses
import random

from track3.evidence import SCENE_GROUPS, EvidenceSheet, corrupt_scene

# Fields a transform may touch. ``event``/``window``/``cast`` are question-derived
# (exact at test); ``scene``/``cause``/``consequence``/``observations`` are the
# cross-question, prediction-derived fields the sheet's name refers to.
CONTENT_FIELDS = ("event", "window", "cast", "scene", "cause", "consequence",
                  "observations")
CROSS_QUESTION_FIELDS = ("scene", "cause", "consequence", "observations")

# How each field name maps onto the EvidenceSheet dataclass attribute + its "empty"
# value (dropping a field == setting it to empty, which drops its prompt line).
_FIELD_ATTR = {
    "event": ("event_phrase", ""),
    "window": ("window", None),
    "cast": ("cast", ()),
    "scene": ("scene", ()),
    "cause": ("cause", ""),
    "consequence": ("consequence", ""),
    "observations": ("observations", []),
}


def _empty_like(ev: EvidenceSheet) -> EvidenceSheet:
    """A sheet with the same video_id but no content (renders header + question only)."""
    return EvidenceSheet(video_id=ev.video_id, event_phrase="")


def _set_field(ev: EvidenceSheet, field: str, value) -> EvidenceSheet:
    attr, _ = _FIELD_ATTR[field]
    return dataclasses.replace(ev, **{attr: value})


def _get_field(ev: EvidenceSheet, field: str):
    attr, _ = _FIELD_ATTR[field]
    return getattr(ev, attr)


def _drop_field(ev: EvidenceSheet, field: str) -> EvidenceSheet:
    attr, empty = _FIELD_ATTR[field]
    return dataclasses.replace(ev, **{attr: empty})


def _parse_spec(spec: str) -> tuple[str, str, float]:
    """('mode', field, prob) from a transform spec string."""
    spec = (spec or "").strip()
    if spec in ("", "identity", "full"):
        return "identity", "", 1.0
    head, _, tail = spec.partition(":")
    head = head.lower()
    if head in ("none",):
        return "none", "", 1.0
    if head in ("drop", "keep"):
        if tail not in CONTENT_FIELDS:
            raise ValueError(f"{head}: unknown field {tail!r}; pick from {CONTENT_FIELDS}")
        return head, tail, 1.0
    if head == "swap":
        return "swap", "", 1.0
    if head == "corrupt":
        field, _, p = tail.partition(":")
        if field not in CONTENT_FIELDS:
            raise ValueError(f"corrupt: unknown field {field!r}; pick from {CONTENT_FIELDS}")
        prob = float(p) if p else 1.0
        return "corrupt", field, prob
    raise ValueError(f"unknown sheet-transform spec {spec!r}")


def _donor_index(i: int, n: int, rng: random.Random) -> int:
    """A clip index != i (for borrowing a wrong-but-plausible value)."""
    if n <= 1:
        return i
    j = rng.randrange(n - 1)
    return j if j < i else j + 1


def _corrupt_one(ev: EvidenceSheet, donor: EvidenceSheet, field: str,
                 rng: random.Random) -> tuple[EvidenceSheet, dict | None]:
    """Inject a wrong value into ``field`` of ``ev`` borrowed from ``donor`` (or,
    for scene, an in-group swap). Returns (new_sheet, log|None). ``log`` records the
    truth (original) and the injected value so the propagation analysis can tell an
    override (injected absent) from a leak (injected present)."""
    orig = _get_field(ev, field)
    if field == "scene":
        new_scene, changed = corrupt_scene(ev.scene, rng, p_scene=1.0)
        if not changed:
            return ev, None
        return _set_field(ev, "scene", new_scene), {
            "field": "scene", "original": list(orig), "injected": list(new_scene)}
    # borrow the donor's value for this field; only inject when it is non-empty and
    # actually differs from the truth (otherwise there is nothing wrong to inject).
    borrowed = _get_field(donor, field)
    if not borrowed or borrowed == orig:
        return ev, None
    return _set_field(ev, field, borrowed), {
        "field": field,
        "original": list(orig) if isinstance(orig, (list, tuple)) else orig,
        "injected": list(borrowed) if isinstance(borrowed, (list, tuple)) else borrowed}


def apply(clips: list[dict], spec: str, seed: int = 0) -> list[dict]:
    """Apply the transform ``spec`` to ``clips`` in place; return a per-clip log.

    ``clips`` is text_dossier's clip list — each dict carries ``"facts"``
    (EvidenceSheet) and ``"vid"``. ``swap``/``corrupt`` borrow across clips, so the
    whole list is transformed together under a single seeded RNG (deterministic).
    """
    mode, field, prob = _parse_spec(spec)
    if mode == "identity" or not clips:
        return []
    rng = random.Random(f"sheet-ablate:{spec}:{seed}")
    n = len(clips)
    log: list[dict] = []

    if mode == "none":
        for c in clips:
            c["facts"] = _empty_like(c["facts"])
        return [{"video_id": c["vid"], "mode": "none"} for c in clips]

    if mode == "drop":
        for c in clips:
            c["facts"] = _drop_field(c["facts"], field)
        return [{"video_id": c["vid"], "mode": "drop", "field": field} for c in clips]

    if mode == "keep":
        for c in clips:
            ev = c["facts"]
            kept = _empty_like(ev)
            c["facts"] = _set_field(kept, field, _get_field(ev, field))
        return [{"video_id": c["vid"], "mode": "keep", "field": field} for c in clips]

    if mode == "swap":
        # a fixed derangement-ish shift: clip i gets clip (i+off)'s content, off != 0.
        originals = [c["facts"] for c in clips]
        off = 1 + rng.randrange(max(1, n - 1)) if n > 1 else 0
        for i, c in enumerate(clips):
            donor = originals[(i + off) % n]
            c["facts"] = dataclasses.replace(donor, video_id=c["vid"])
            log.append({"video_id": c["vid"], "mode": "swap",
                        "donor_video_id": donor.video_id})
        return log

    if mode == "corrupt":
        originals = [c["facts"] for c in clips]
        for i, c in enumerate(clips):
            if rng.random() >= prob:
                continue
            donor = originals[_donor_index(i, n, rng)]
            new_ev, rec = _corrupt_one(c["facts"], donor, field, rng)
            if rec is not None:
                c["facts"] = dataclasses.replace(new_ev, corrupted=True)
                rec["video_id"] = c["vid"]
                rec["mode"] = "corrupt"
                log.append(rec)
        return log

    raise ValueError(f"unhandled mode {mode!r}")
