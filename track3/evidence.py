"""Evidence Sheet: fact-sheet schema, render prompt and question/GT mining helpers.

Imported verbatim by build_render_sft (train) and text_dossier --direct (test) so the
render prompt is identical on both sides.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field

# Tasks whose reference answers echo the question's [X,Y] window.
WINDOW_TASKS = frozenset({"temporal_description", "causal_linkage"})

SYS_RENDER = (
    "You are an expert traffic-surveillance video analyst. Answer the question using the "
    "verified facts AND the video. The video is the ground truth: treat the facts as hints "
    "and correct or ignore any fact that conflicts with what you see. Be concise, factual "
    "and specific; respond with exactly the requested content and nothing else."
)
FACT_HEADERS = {
    "video-priority": ("Verified facts about this clip (hints — the video is the ground "
                       "truth; correct any fact that conflicts with what you see):"),
    "trust": ("Verified facts about this clip (use them; do not contradict them):"),
}


@dataclass
class EvidenceSheet:
    """Per-clip facts; empty fields drop their prompt line."""
    video_id: str = ""
    event_phrase: str = "the collision or anomaly"
    window: tuple[str, str] | None = None
    cast: tuple[str, ...] = ()          # vehicle/agent descriptors (question-mined, exact)
    scene: tuple[str, ...] = ()         # scene attributes (SD-GT-mined / probed)
    cause: str = ""                     # chosen root-cause/fault mcq option text
    consequence: str = ""               # chosen outcome/sequence mcq option text
    observations: list[str] = field(default_factory=list)   # bcq_oe one-liners
    corrupted: bool = False             # train-side noise bookkeeping (stats/tests)


def render_evidence_prompt(task: str, question: str, ev, with_video: bool = True,
                           fact_policy: str = "video-priority") -> str:
    """Canonical fact-sheet prompt. ``ev`` is duck-typed; v1 sheets render the v1 prompt."""
    lines = ["<video>"] if with_video else []
    lines += [FACT_HEADERS.get(fact_policy, FACT_HEADERS["video-priority"]),
              f"- Key event: {ev.event_phrase}."]
    if ev.window:
        lines.append(f"- It occurs between {ev.window[0]} and {ev.window[1]}.")
    cast = getattr(ev, "cast", ())
    if cast:
        lines.append("- Vehicles involved: " + "; ".join(cast) + ".")
    scene = getattr(ev, "scene", ())
    if scene:
        lines.append("- Scene: " + ", ".join(scene) + ".")
    cause = getattr(ev, "cause", "")
    if cause:
        lines.append(f"- Root cause: {cause}")
    if ev.consequence:
        lines.append(f"- Likely consequence: {ev.consequence}")
    for obs in ev.observations:
        lines.append(f"- Observation: {obs}")
    lines += ["", question.strip()]
    if task in WINDOW_TASKS and ev.window:
        lines.append(f"Refer to the moments {ev.window[0]} and {ev.window[1]} "
                     "explicitly in your answer.")
    return "\n".join(lines)


# Cast mining: vehicle/agent descriptors from the question texts.
_COLORS = (r"(?:white|black|red|blue|silver|gr[ae]y|green|yellow|orange|brown|gold|"
           r"beige|maroon|dark|light)")
_VEHICLES = (r"(?:van|SUV|sedan|truck|car|bus|motorcycle|motorbike|scooter|"
             r"pickup(?:\s+truck)?|taxi|hatchback|minivan|semi(?:-truck)?|trailer)")
_CAST = re.compile(
    rf"\b((?:{_COLORS}(?:\s+and\s+{_COLORS})?\s+){_VEHICLES}|pedestrians?|cyclists?|"
    rf"motorcyclists?|bicycle)\b", re.IGNORECASE)


def extract_cast(questions, limit: int = 6) -> tuple[str, ...]:
    """Unique vehicle/agent descriptors over the clip's questions, first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for q in questions:
        for m in _CAST.finditer(q or ""):
            ent = re.sub(r"\s+", " ", m.group(1)).strip().lower()
            if ent not in seen:
                seen.add(ent)
                out.append(ent)
            if len(out) >= limit:
                return tuple(out)
    return tuple(out)


_CAUSE_STEM = re.compile(r"root cause|at fault|fault attribution|why did|caused the",
                         re.IGNORECASE)


def classify_stem(question: str) -> str:
    """'cause' (root-cause/fault stem) or 'outcome' (sequence/type/consequence stem)."""
    stem = (question or "").split("\n")[0]
    return "cause" if _CAUSE_STEM.search(stem) else "outcome"


# Scene attributes: mined from SD GT (train) or probed (test). At most one value
# per group enters the sheet.
SCENE_GROUPS: dict[str, tuple[str, ...]] = {
    "time": ("daytime", "nighttime"),
    "setting": ("an intersection", "a highway", "a city street", "a rural road"),
    "weather": ("clear weather", "rainy weather", "snowy weather", "foggy weather"),
}

_SCENE_MINE: dict[tuple[str, str], re.Pattern] = {
    ("time", "daytime"): re.compile(r"\b(?:daytime|day[- ]?light|during the day|broad daylight)\b", re.I),
    ("time", "nighttime"): re.compile(r"\b(?:night|nighttime|after dark)\b", re.I),
    ("setting", "an intersection"): re.compile(r"\b(?:intersection|junction|crossroads?)\b", re.I),
    ("setting", "a highway"): re.compile(r"\b(?:highway|freeway|motorway|expressway|interstate)\b", re.I),
    ("setting", "a city street"): re.compile(r"\b(?:city street|urban (?:road|street)|downtown)\b", re.I),
    ("setting", "a rural road"): re.compile(r"\b(?:rural|country(?:side)? road)\b", re.I),
    ("weather", "clear weather"): re.compile(r"\b(?:clear (?:weather|conditions|day|sky)|dry road)\b", re.I),
    ("weather", "rainy weather"): re.compile(r"\b(?:rain\w*|wet road|drizzle)\b", re.I),
    ("weather", "snowy weather"): re.compile(r"\b(?:snow\w*|icy)\b", re.I),
    ("weather", "foggy weather"): re.compile(r"\b(?:fog\w*|misty|hazy)\b", re.I),
}

# Test-time probes: P(first token == "Yes") per (group, value).
SCENE_PROBE_QUESTIONS: dict[tuple[str, str], str] = {
    ("time", "daytime"): "Is this video recorded during daylight?",
    ("time", "nighttime"): "Is this video recorded at night?",
    ("setting", "an intersection"): "Does the video show a road intersection or junction?",
    ("setting", "a highway"): "Does the video show a highway or freeway?",
    ("setting", "a city street"): "Does the video show a city street?",
    ("setting", "a rural road"): "Does the video show a rural road?",
    ("weather", "clear weather"): "Is the weather in the video clear and the road dry?",
    ("weather", "rainy weather"): "Is it raining or is the road wet in the video?",
    ("weather", "snowy weather"): "Is there snow or ice visible in the video?",
    ("weather", "foggy weather"): "Is the scene foggy or hazy in the video?",
}


def mine_scene_attributes(sd_answer: str) -> tuple[str, ...]:
    """Scene attributes from a scene_description GT; ambiguous groups are skipped."""
    out = []
    for group, values in SCENE_GROUPS.items():
        hits = [v for v in values if _SCENE_MINE[(group, v)].search(sd_answer or "")]
        if len(hits) == 1:
            out.append(hits[0])
    return tuple(out)


def pick_scene_from_probes(scores: dict[tuple[str, str], float],
                           min_p: float = 0.5) -> tuple[str, ...]:
    """Per group, the argmax probe value if its P(Yes) >= ``min_p``."""
    out = []
    for group, values in SCENE_GROUPS.items():
        scored = [(scores.get((group, v), 0.0), v) for v in values]
        best_p, best_v = max(scored)
        if best_p >= min_p:
            out.append(best_v)
    return tuple(out)


def corrupt_scene(scene: tuple[str, ...], rng, p_scene: float) -> tuple[tuple[str, ...], bool]:
    """With prob ``p_scene`` swap one attribute for another value of its group.

    Returns (scene, corrupted).
    """
    if not scene or rng.random() >= p_scene:
        return scene, False
    idxs = list(range(len(scene)))
    rng.shuffle(idxs)
    for idx in idxs:
        for values in SCENE_GROUPS.values():
            if scene[idx] in values:
                alts = [v for v in values if v != scene[idx]]
                if alts:
                    out = list(scene)
                    out[idx] = rng.choice(alts)
                    return tuple(out), True
                break
    return scene, False


def jitter_evidence(ev: EvidenceSheet, rng) -> EvidenceSheet:
    """Copy of ``ev`` with one optional fact dropped (decorrelates MBR candidates)."""
    slots: list[str] = []
    if getattr(ev, "cause", ""):
        slots.append("cause")
    if ev.consequence:
        slots.append("consequence")
    if getattr(ev, "scene", ()):
        slots.append("scene")
    if getattr(ev, "cast", ()):
        slots.append("cast")
    slots += [f"obs:{i}" for i in range(len(ev.observations))]
    if not slots:
        return ev
    drop = rng.choice(slots)
    kw: dict = {}
    if drop == "cause":
        kw["cause"] = ""
    elif drop == "consequence":
        kw["consequence"] = ""
    elif drop == "scene":
        kw["scene"] = ()
    elif drop == "cast":
        kw["cast"] = ()
    else:
        i = int(drop.split(":")[1])
        kw["observations"] = ev.observations[:i] + ev.observations[i + 1:]
    return dataclasses.replace(ev, **kw)
