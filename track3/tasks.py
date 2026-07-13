"""Task registry — the single source of truth for the 10 TAR sub-tasks.

``build_dataset.py`` (training targets), ``infer.py`` / ``make_submission.py``
(prediction formatting) and ``metrics.py`` / ``eval_local.py`` (scoring) all read
from here, so adding or re-tuning a task is a one-line change in ``TASKS``.

A task is identified by its ``key`` which equals the TAR annotation file stem,
e.g. ``bcq`` ↔ ``bcq.json``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- metric families (must mirror the official evaluate.py) -------------------
METRIC_ACC_YESNO = "acc_yesno"   # regex Yes/No accuracy
METRIC_ACC_LETTER = "acc_letter"  # regex A-D accuracy
METRIC_IOU = "iou"               # mean IoU over {start,end}
METRIC_BERTSCORE = "bertscore"   # BERTScore-F1 roberta-large, rescaled

# --- output policies (how the assistant target is shaped) ---------------------
POLICY_ANSWER_ONLY = "answer_only"      # bare token, e.g. "Yes" / "A"
POLICY_ANSWER_TEXT = "answer_text"      # free-form graded text
POLICY_JSON_INTERVAL = "json_interval"  # {"start":"MM:SS","end":"MM:SS"}

VIDEO_TAG = "<video>"

# System prompts priming the model per task family. Kept short and factual so the
# model's answer register matches the concise, grounded TAR reference style.
_SYS_BASE = (
    "You are an expert traffic-surveillance video analyst for an anomaly "
    "reasoning system. Watch the video carefully and answer grounded in the "
    "visual evidence."
)
_SYS_CONCISE = _SYS_BASE + " Be concise, factual, and specific."
_SYS_STRICT = _SYS_BASE + " Respond with exactly the requested format and nothing else."
_SYS_ACC = _SYS_STRICT
# Temporal localization is graded by IoU over wall-clock MM:SS, so the model must
# anchor the event to the frame timestamps it is shown (see method.md §3.3).
_SYS_TEMPORAL = (
    _SYS_BASE + " You are shown video frames sampled at known timestamps together "
    "with the total duration. Find when the queried event begins and ends by "
    "reasoning about which timestamped frames bound it, then respond with exactly "
    '{"start":"MM:SS","end":"MM:SS"} and nothing else. Both times must lie within '
    "the video duration and start must not exceed end."
)


@dataclass
class TaskSpec:
    key: str
    group: str                       # basic | scene | temporal
    metric: str
    policy: str
    system: str
    # Parse raw model output -> the gradable / submission string for this task.
    parse: Callable[[str], str]
    # Format the *submission* string (e.g. fence temporal JSON). Defaults to parse.
    format_submission: Optional[Callable[[str], str]] = None
    items_per_video: int = 1         # bcq* carry 2 (a Yes and a No)
    cot_policy: str = "none"         # none | hidden | inline  (see method.md §3.2)
    # Time-sensitive tasks (temporal_localization) need an explicit, dense time
    # axis: they always sample timestamped `extract` frames at a denser budget,
    # independent of the global frames-mode (see frame_plan + method.md §3.3).
    time_sensitive: bool = False

    def submission(self, raw: str) -> str:
        """Map any raw model output to the EXACT string the official grader parses.

        This is the single format-enforcement choke-point (see ``enforce_format``):
        accuracy tasks collapse to their canonical token, temporal to a fenced JSON
        interval, open-ended text is cleaned but kept verbatim. Idempotent, so it is
        safe to apply at infer time *and* again in make_submission. An explicit
        ``format_submission`` override (e.g. temporal's fence) takes precedence.
        """
        if self.format_submission is not None:
            return self.format_submission(raw)
        return enforce_format(self, raw)


# ---------------------------------------------------------------------------
# Extraction — mirrors track3/evaluate.py so our submission formatting and
# self-consistency voting agree exactly with how the organizers grade.
# (The authoritative copies live in the official scorer; these are kept
#  byte-compatible and covered by a self-test in track3/check_eval.py.)
# ---------------------------------------------------------------------------
_TIME = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)")

# Reasoning ("thinking") that must be stripped before grading/submission. Qwen3.5
# emits `<think>\n...\n</think>\n\n<answer>`; the think content (long CoT) would
# wreck BERTScore and can flip the Yes/No / letter regex if left in.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def strip_reasoning(text) -> str:
    """Remove ``<think>...</think>`` reasoning, keeping only the final answer.

    - Complete ``<think>...</think>`` blocks are dropped (text after them kept).
    - If only a closing ``</think>`` survives (the opening was streamed as a
      separate ``reasoning_content``), keep the tail after the last ``</think>``.
    - A dangling, unclosed ``<think>`` (truncated output) leaves its text but the
      lone tag is removed — so the temporal extractor can still mine timestamps.
    """
    if text is None:
        return ""
    s = str(text)
    s = _THINK_BLOCK.sub("", s)
    if _THINK_CLOSE.search(s):
        s = _THINK_CLOSE.split(s)[-1]
    s = _THINK_TAG.sub("", s)
    return s.strip()


def extract_yesno(text) -> Optional[str]:
    """'yes' / 'no' / None — leading token preferred, else first occurrence."""
    if text is None or not str(text).strip():
        return None
    s = str(text).strip().lower()
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def extract_letter(text) -> Optional[str]:
    """Single choice letter (upper) / None — mirrors official _extract_letter."""
    if text is None or not str(text).strip():
        return None
    s = str(text).strip()
    m = re.match(r"^\(?([A-Za-z])\)?[).\s,:]", s)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"[A-Da-d]", s):
        return s.upper()
    m = re.search(r"\b([A-D])\b", s)
    return m.group(1) if m else None


def parse_timestamp(ts) -> float:
    """MM:SS / HH:MM:SS / float-seconds -> seconds (mirrors official)."""
    parts = str(ts).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def _canon_ts(text) -> Optional[str]:
    """First timestamp in ``text`` -> canonical zero-padded ``MM:SS[.ff]``.

    Normalizes every form (MM:SS, HH:MM:SS, fractional) to the **MM:SS** shape the
    prompt actually requests, so training targets and submissions are format-
    consistent (e.g. ``00:00:13`` -> ``00:13``). Fractional seconds are preserved —
    ~20% of temporal GT intervals are <3s where dropping ``.ff`` distorts the IoU.
    The grader's ``_parse_timestamp`` reads seconds either way, so this never changes
    a score; it only removes the HH:MM:SS/MM:SS inconsistency the auto-labels carry.
    """
    m = _TIME.search(str(text))
    if not m:
        return None
    h, mm, ss = m.groups()
    total = (int(h) if h is not None else 0) * 3600 + int(mm) * 60 + float(ss)
    minutes, seconds = int(total // 60), total % 60
    if "." in ss:
        return f"{minutes:02d}:{seconds:05.2f}"   # MM:SS.ff
    return f"{minutes:02d}:{int(round(seconds)):02d}"


def parse_interval(text: str) -> str:
    """Canonicalize any model/GT text to a {"start":"MM:SS","end":"MM:SS"} string.

    Used for (a) canonicalizing training targets and (b) repairing submissions.
    Always returns valid JSON so the temporal task is never unparseable. Models
    routinely emit the ``{start,end}`` object *inline in prose* (unfenced), which
    the official ``_extract_json`` misses (it only reads a ```json fence or a
    whole-string JSON) — so we mine the embedded object first, then fall back to
    the first two timestamps anywhere in the text.
    """
    text = strip_reasoning(text)
    start = end = None
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
            if "start" in obj and "end" in obj:
                start, end = str(obj["start"]), str(obj["end"])
                break
        except json.JSONDecodeError:
            continue
    if start is None or end is None:
        times = [m.group(0) for m in _TIME.finditer(text)]
        if len(times) >= 2:
            start, end = times[0], times[1]
        elif len(times) == 1:
            start = end = times[0]
        else:
            start, end = "00:00", "00:00"
    start = _canon_ts(start) or "00:00"
    end = _canon_ts(end) or start
    return json.dumps({"start": start, "end": end})


def parse_text(text: str) -> str:
    """Submission form for raw-text answers: strip whitespace + any leaked tags.

    The organizers extract the gradable token from raw model output themselves
    (Yes/No, letter, BERTScore over the whole string), so for every task *except*
    temporal_localization we submit the model's text essentially verbatim — but
    with any ``<think>...</think>`` reasoning removed first (it is not the answer).
    """
    t = strip_reasoning(text)
    t = re.sub(r"</?(?:reason)>", "", t, flags=re.IGNORECASE).strip()
    return t


def fence_interval(text: str) -> str:
    """Submission form for temporal_localization: a parseable fenced ```json block."""
    return "```json\n" + parse_interval(text) + "\n```"


# ---------------------------------------------------------------------------
# Strict format enforcement — the deterministic guarantee that every prediction
# we emit is EXACTLY what the official grader can parse, so compliance never
# depends on the model's prose discipline. Each branch mirrors the matching
# extractor in track3/evaluate.py.
# ---------------------------------------------------------------------------
# Non-empty placeholder for an open-ended answer the model failed to produce
# (e.g. output truncated inside an unclosed <think>). BERTScore needs a real
# token; this scores ~0 but never crashes or yields an empty/unparseable row.
_OPEN_FALLBACK = "unknown"


def enforce_yesno(text: str) -> str:
    """bcq -> exactly 'Yes' or 'No' (grader reads the leading Yes/No token)."""
    v = extract_yesno(strip_reasoning(text))
    return v.capitalize() if v else "No"


_MCQ_MARK = re.compile(r"\(?([A-Da-d])(?:\)|\.|:)")   # "C)" / "C." / "C:" / "(C)"
_MCQ_WORD = re.compile(r"\b([A-Da-d])\b")


def enforce_letter(text: str) -> str:
    """mcq -> exactly one of A/B/C/D, emitted bare so the grader can't misread it.

    The official extractor is positional ("first letter-ish token"), so on a
    rambling answer like "I think the answer is (C)" it returns 'I'. Since *we*
    control the submitted string, we resolve the real option — prefer an explicit
    option marker (C)/C., else the last standalone A-D, else default 'A' — and
    emit just that letter, which the grader then reads unambiguously.
    """
    s = strip_reasoning(text)
    v = extract_letter(s)
    if v in ("A", "B", "C", "D"):
        return v
    for pat in (_MCQ_MARK, _MCQ_WORD):
        m = pat.findall(s)
        if m:
            return m[-1].upper()
    return "A"


def enforce_text(text: str) -> str:
    """Open-ended (BERTScore) -> cleaned text, guaranteed non-empty."""
    t = parse_text(text)
    return t if t else _OPEN_FALLBACK


def enforce_format(spec: "TaskSpec", raw: str) -> str:
    """Coerce ``raw`` to the grader-exact form for ``spec``'s metric. Idempotent."""
    if spec.metric == METRIC_ACC_YESNO:
        return enforce_yesno(raw)
    if spec.metric == METRIC_ACC_LETTER:
        return enforce_letter(raw)
    if spec.metric == METRIC_IOU:
        return fence_interval(raw)
    return enforce_text(raw)


def check_parseable(spec: "TaskSpec", submission_text: str) -> bool:
    """True iff ``submission_text`` parses for ``spec``'s metric (grader's view).

    Mirrors track3/evaluate.py's ``_check_parseable`` so the format-validation gate
    in infer.py agrees byte-for-byte with how the organizers validate a CSV.
    """
    if spec.metric == METRIC_ACC_YESNO:
        return extract_yesno(submission_text) is not None
    if spec.metric == METRIC_ACC_LETTER:
        return extract_letter(submission_text) is not None
    if spec.metric == METRIC_IOU:
        obj = extract_interval_obj(submission_text)
        return obj is not None and "start" in obj and "end" in obj
    return bool(submission_text and submission_text.strip())


def extract_interval_obj(text: str):
    """Parse a submission's temporal JSON the way the official _extract_json does
    (fenced ```json block or whole-string JSON), returning the dict or None."""
    if text is None or not str(text).strip():
        return None
    s = str(text).strip()
    m = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj[0]
            return obj
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def vote_token(spec: "TaskSpec", text: str) -> str:
    """Reduce a sampled output to its canonical answer token (for voting).

    Falls back to a deterministic guess when nothing parses, so a vote is always
    cast (better expected score than abstaining on an unparseable sample)."""
    text = strip_reasoning(text)  # never vote on tokens inside the <think> trace
    if spec.metric == METRIC_ACC_YESNO:
        v = extract_yesno(text)
        return v.capitalize() if v else "No"
    if spec.metric == METRIC_ACC_LETTER:
        v = extract_letter(text)
        return v if v else "A"
    return parse_text(text)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TASKS: dict[str, TaskSpec] = {
    "bcq": TaskSpec(
        "bcq", "basic", METRIC_ACC_YESNO, POLICY_ANSWER_ONLY,
        _SYS_ACC, parse_text, items_per_video=2),
    "bcq_openended": TaskSpec(
        "bcq_openended", "basic", METRIC_BERTSCORE, POLICY_ANSWER_TEXT,
        _SYS_CONCISE, parse_text, items_per_video=2),
    "mcq": TaskSpec(
        "mcq", "basic", METRIC_ACC_LETTER, POLICY_ANSWER_ONLY,
        _SYS_ACC, parse_text),
    "mcq_openended": TaskSpec(
        "mcq_openended", "basic", METRIC_BERTSCORE, POLICY_ANSWER_TEXT,
        _SYS_CONCISE, parse_text),
    "open_qa": TaskSpec(
        "open_qa", "basic", METRIC_BERTSCORE, POLICY_ANSWER_TEXT,
        _SYS_CONCISE, parse_text),
    "scene_description": TaskSpec(
        "scene_description", "scene", METRIC_BERTSCORE, POLICY_ANSWER_TEXT,
        _SYS_CONCISE, parse_text),
    "video_summarization": TaskSpec(
        "video_summarization", "scene", METRIC_BERTSCORE, POLICY_ANSWER_TEXT,
        _SYS_CONCISE, parse_text),
    "temporal_localization": TaskSpec(
        "temporal_localization", "temporal", METRIC_IOU, POLICY_JSON_INTERVAL,
        _SYS_TEMPORAL, parse_interval, format_submission=fence_interval,
        time_sensitive=True),
    "temporal_description": TaskSpec(
        "temporal_description", "temporal", METRIC_BERTSCORE, POLICY_ANSWER_TEXT,
        _SYS_CONCISE, parse_text),
    "causal_linkage": TaskSpec(
        "causal_linkage", "temporal", METRIC_BERTSCORE, POLICY_ANSWER_TEXT,
        _SYS_CONCISE, parse_text),
}

ALL_TASK_KEYS = list(TASKS.keys())


def get_task(key: str) -> TaskSpec:
    if key not in TASKS:
        raise KeyError(f"Unknown task '{key}'. Known: {ALL_TASK_KEYS}")
    return TASKS[key]


# Heuristic fallback when a test item lacks an explicit `task` field.
def infer_task_key(item: dict) -> str:
    for fld in ("task", "task_type", "type"):
        if item.get(fld) in TASKS:
            return item[fld]
    q = (item.get("question") or "").lower()
    if "start and end" in q or "mm:ss" in q or "when does" in q:
        return "temporal_localization"
    if "yes or no" in q:
        return "bcq"
    if re.search(r"\n\s*a\)", q) or re.search(r"\n\s*a\.", q):
        return "mcq"
    if "summar" in q:
        return "video_summarization"
    if "caus" in q or "what caused" in q:
        return "causal_linkage"
    if "describe the scene" in q or "scene" in q:
        return "scene_description"
    return "open_qa"


def frame_plan(
    spec: TaskSpec,
    base_mode: str,
    base_frames: int,
    base_side: int,
    temporal_frames: Optional[int] = None,
    temporal_side: Optional[int] = None,
) -> tuple[str, int, int]:
    """Per-task ``(frames_mode, num_frames, max_side)`` for video sampling.

    Single source of truth shared by build_dataset and infer so train- and
    test-time framing never diverge.

    Time-sensitive tasks (temporal_localization) **always use native ``video``
    mode**, never pre-extracted image lists. Measured finding (see memory
    ``track3-status``): handing Qwen3-VL a list of JPEGs stamps them with a
    *default* fps (~2) in ``ms-swift/.../qwen.py:replace_tag``, which corrupts the
    model's native time-aligned position IDs and makes the timestamp hint a no-op
    (mIoU stuck at 0.186). Native video preserves real fps/duration metadata; the
    IoU lever is then driven by the *global frame budget* (FPS_MAX_FRAMES) plus a
    duration hint in the prompt. We still surface a (denser) temporal frame count
    so callers can raise FPS_MAX_FRAMES for time-sensitive items if desired.
    """
    if spec.time_sensitive:
        return "video", (temporal_frames or base_frames), (temporal_side or base_side)
    return base_mode, base_frames, base_side


def build_user_prompt(spec: TaskSpec, question: str, video_hint: str = "") -> str:
    """User turn = <video> + optional frame/time hint + the verbatim question."""
    parts = [VIDEO_TAG]
    if video_hint:
        parts.append(video_hint)
    parts.append(question.strip())
    return "\n".join(parts)
