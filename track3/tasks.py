"""Task registry: the single source of truth for the 10 TAR sub-tasks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# Metric families (mirror the official evaluate.py).
METRIC_ACC_YESNO = "acc_yesno"   # regex Yes/No accuracy
METRIC_ACC_LETTER = "acc_letter"  # regex A-D accuracy
METRIC_IOU = "iou"               # mean IoU over {start,end}
METRIC_BERTSCORE = "bertscore"   # BERTScore-F1 roberta-large, rescaled

# Output policies (shape of the assistant target).
POLICY_ANSWER_ONLY = "answer_only"      # bare token, e.g. "Yes" / "A"
POLICY_ANSWER_TEXT = "answer_text"      # free-form graded text
POLICY_JSON_INTERVAL = "json_interval"  # {"start":"MM:SS","end":"MM:SS"}

VIDEO_TAG = "<video>"

_SYS_BASE = (
    "You are an expert traffic-surveillance video analyst for an anomaly "
    "reasoning system. Watch the video carefully and answer grounded in the "
    "visual evidence."
)
_SYS_CONCISE = _SYS_BASE + " Be concise, factual, and specific."
_SYS_STRICT = _SYS_BASE + " Respond with exactly the requested format and nothing else."
_SYS_ACC = _SYS_STRICT
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
    parse: Callable[[str], str]
    format_submission: Optional[Callable[[str], str]] = None
    items_per_video: int = 1         # bcq* carry 2 (a Yes and a No)
    cot_policy: str = "none"         # none | hidden | inline
    time_sensitive: bool = False     # see frame_plan

    def submission(self, raw: str) -> str:
        """Map raw model output to the exact string the official grader parses (idempotent)."""
        if self.format_submission is not None:
            return self.format_submission(raw)
        return enforce_format(self, raw)


# Extractors mirror track3/evaluate.py (checked by track3/check_eval.py).
_TIME = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)")

# <think>...</think> reasoning must be stripped before grading/voting.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def strip_reasoning(text) -> str:
    """Drop complete, close-only or dangling ``<think>`` reasoning; keep the final answer."""
    if text is None:
        return ""
    s = str(text)
    s = _THINK_BLOCK.sub("", s)
    if _THINK_CLOSE.search(s):
        s = _THINK_CLOSE.split(s)[-1]
    s = _THINK_TAG.sub("", s)
    return s.strip()


def extract_yesno(text) -> Optional[str]:
    if text is None or not str(text).strip():
        return None
    s = str(text).strip().lower()
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def extract_letter(text) -> Optional[str]:
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
    parts = str(ts).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def _canon_ts(text) -> Optional[str]:
    """First timestamp in ``text`` -> zero-padded ``MM:SS[.ff]`` (fractional seconds kept)."""
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
    """Canonicalize text to a {"start","end"} JSON string (always valid JSON).

    Mines an inline {start,end} object first, then the first two timestamps.
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
    """Raw-text submission: strip whitespace, reasoning and leaked tags."""
    t = strip_reasoning(text)
    t = re.sub(r"</?(?:reason)>", "", t, flags=re.IGNORECASE).strip()
    return t


def fence_interval(text: str) -> str:
    """Submission form for temporal_localization: a parseable fenced ```json block."""
    return "```json\n" + parse_interval(text) + "\n```"


# Strict format enforcement: each branch mirrors an extractor in track3/evaluate.py.
# Placeholder for an empty open-ended answer (BERTScore needs a non-empty string).
_OPEN_FALLBACK = "unknown"


def enforce_yesno(text: str) -> str:
    """bcq -> exactly 'Yes' or 'No' (grader reads the leading Yes/No token)."""
    v = extract_yesno(strip_reasoning(text))
    return v.capitalize() if v else "No"


_MCQ_MARK = re.compile(r"\(?([A-Da-d])(?:\)|\.|:)")   # "C)" / "C." / "C:" / "(C)"
_MCQ_WORD = re.compile(r"\b([A-Da-d])\b")


def enforce_letter(text: str) -> str:
    """mcq -> one bare A-D letter.

    The official extractor is positional ("I think ... (C)" reads as 'I'), so resolve
    an explicit marker, else the last standalone A-D, else 'A'.
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
    """Open-ended -> cleaned, non-empty text."""
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
    """True iff ``submission_text`` parses for ``spec``'s metric (mirrors official _check_parseable)."""
    if spec.metric == METRIC_ACC_YESNO:
        return extract_yesno(submission_text) is not None
    if spec.metric == METRIC_ACC_LETTER:
        return extract_letter(submission_text) is not None
    if spec.metric == METRIC_IOU:
        obj = extract_interval_obj(submission_text)
        return obj is not None and "start" in obj and "end" in obj
    return bool(submission_text and submission_text.strip())


def extract_interval_obj(text: str):
    """Parse temporal JSON like the official _extract_json; dict or None."""
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
    """Canonical answer token for voting; deterministic guess when nothing parses."""
    text = strip_reasoning(text)
    if spec.metric == METRIC_ACC_YESNO:
        v = extract_yesno(text)
        return v.capitalize() if v else "No"
    if spec.metric == METRIC_ACC_LETTER:
        v = extract_letter(text)
        return v if v else "A"
    return parse_text(text)


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


# Heuristic fallback when an item has no explicit task field.
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
    """Per-task ``(frames_mode, num_frames, max_side)``, shared by build_dataset and infer.

    Time-sensitive tasks always use native ``video`` mode: an image list gets a default
    fps in ms-swift, which breaks Qwen3-VL's time-aligned position IDs.
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
