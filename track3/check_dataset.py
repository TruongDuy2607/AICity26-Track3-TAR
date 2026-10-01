"""Validate the ms-swift jsonl produced by ``build_dataset.py`` before training.

Checks message structure, <video> tag count, media existence, label format, task
coverage and train/val video leakage. Exits non-zero on any hard error.

Run: ``python -m track3.check_dataset --data-dir data``
     ``python -m track3.check_dataset --files data/train.jsonl --no-media``  # schema only
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from track3.tasks import (
    METRIC_ACC_LETTER,
    METRIC_ACC_YESNO,
    METRIC_IOU,
    POLICY_JSON_INTERVAL,
    VIDEO_TAG,
    ALL_TASK_KEYS,
    TASKS,
    extract_letter,
    extract_yesno,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data",
                   help="Dir holding train.jsonl / val.jsonl (used if --files omitted).")
    p.add_argument("--files", nargs="*", default=None,
                   help="Explicit jsonl files to check (overrides --data-dir).")
    p.add_argument("--no-media", action="store_true",
                   help="Skip on-disk media existence checks (schema-only).")
    p.add_argument("--max-errors", type=int, default=20,
                   help="How many example errors to print per file.")
    p.add_argument("--max-missing-frac", type=float, default=0.0,
                   help="Tolerated fraction of records with missing media (0 = none).")
    return p.parse_args()


class Report:
    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.errors: list[str] = []
        self.task_counts: dict[str, int] = defaultdict(int)
        self.missing_media = 0
        self.bad_label = 0
        self.video_ids: set[str] = set()

    def err(self, line_no: int, msg: str) -> None:
        self.errors.append(f"  [{self.name}:L{line_no}] {msg}")


def _check_messages(rec: dict) -> str | None:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "missing/empty 'messages'"
    roles = [m.get("role") for m in msgs]
    if "assistant" not in roles:
        return "no assistant turn"
    asst = next(m for m in msgs if m.get("role") == "assistant")
    if not (asst.get("content") or "").strip():
        return "empty assistant target"
    user = next((m for m in msgs if m.get("role") == "user"), None)
    if user is None:
        return "no user turn"
    return None


def _video_entries(rec: dict) -> list:
    return rec.get("videos") or rec.get("video") or []


def _check_video_tags(rec: dict) -> str | None:
    user = next((m for m in rec["messages"] if m.get("role") == "user"), {})
    n_tags = (user.get("content") or "").count(VIDEO_TAG)
    n_videos = len(_video_entries(rec))
    if n_tags != n_videos:
        return f"<video> tag count ({n_tags}) != #videos ({n_videos})"
    return None


def _check_media(rec: dict) -> str | None:
    for v in _video_entries(rec):
        if isinstance(v, list):  # frames-mode=extract: list of frame paths
            if not v:
                return "empty frame list"
            missing = [p for p in v if not os.path.exists(p)]
            if missing:
                return f"{len(missing)}/{len(v)} frames missing (e.g. {missing[0]})"
        else:                    # frames-mode=video: a single file path
            if not (str(v).startswith(("http://", "https://", "data:")) or os.path.exists(v)):
                return f"video not found: {v}"
    return None


def _ts_seconds(value) -> float | None:
    """Timestamp -> seconds via the grader's parser; None if unparseable."""
    from track3.tasks import parse_timestamp
    try:
        s = str(value or "").strip()
        return parse_timestamp(s) if s else None
    except (ValueError, TypeError):
        return None


def _check_label(task_key: str, target: str) -> str | None:
    spec = TASKS.get(task_key)
    if spec is None:
        return f"unknown task '{task_key}'"
    metric = spec.metric
    if metric == METRIC_ACC_YESNO:
        if extract_yesno(target) is None:
            return f"bcq target has no extractable Yes/No: {target!r}"
    elif metric == METRIC_ACC_LETTER:
        if extract_letter(target) is None:
            return f"mcq target has no extractable letter: {target!r}"
    elif metric == METRIC_IOU or spec.policy == POLICY_JSON_INTERVAL:
        # target may be inline CoT ("reason ... {json}"): extract the embedded interval
        obj = _extract_interval_obj(target)
        if obj is None:
            return f"no parseable start/end interval in target: {target!r}"
        s, e = _ts_seconds(obj.get("start")), _ts_seconds(obj.get("end"))
        if s is None or e is None:
            return f"interval start/end not a valid timestamp: {target!r}"
        if s > e:
            return f"interval start>end: {target!r}"
    return None


def _extract_interval_obj(target: str) -> dict | None:
    """Find the {start,end} JSON object embedded in a (possibly CoT) target."""
    import re
    for m in re.finditer(r"\{[^{}]*\}", str(target)):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "start" in obj and "end" in obj:
            return obj
    return None


def check_file(path: str, args: argparse.Namespace) -> Report:
    rep = Report(os.path.basename(path))
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rep.n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as ex:
                rep.err(i, f"invalid json: {ex}")
                continue

            msg_err = _check_messages(rec)
            if msg_err:
                rep.err(i, msg_err)
                continue

            task = rec.get("task", "?")
            rep.task_counts[task] += 1
            if rec.get("video_id"):
                rep.video_ids.add(rec["video_id"])

            for err in (_check_video_tags(rec),):
                if err:
                    rep.err(i, err)

            if not args.no_media:
                m = _check_media(rec)
                if m:
                    rep.missing_media += 1
                    rep.err(i, m)

            target = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
            lbl = _check_label(task, target)
            if lbl:
                rep.bad_label += 1
                rep.err(i, lbl)
    return rep


def main() -> int:
    args = parse_args()
    files = args.files or [os.path.join(args.data_dir, f) for f in ("train.jsonl", "val.jsonl")]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("ERROR: no jsonl files found to check.")
        return 2

    reports = [check_file(f, args) for f in files]

    print(f"\n{'file':16s} {'records':>8s} {'missing':>8s} {'bad_lbl':>8s} {'errors':>8s}")
    hard_fail = False
    for rep in reports:
        print(f"{rep.name:16s} {rep.n:>8d} {rep.missing_media:>8d} "
              f"{rep.bad_label:>8d} {len(rep.errors):>8d}")
        miss_frac = rep.missing_media / rep.n if rep.n else 0.0
        if rep.n == 0 or rep.bad_label > 0 or miss_frac > args.max_missing_frac:
            hard_fail = True

    print(f"\n{'task':24s}" + "".join(f"{r.name:>14s}" for r in reports))
    for task in ALL_TASK_KEYS:
        row = "".join(f"{r.task_counts.get(task, 0):>14d}" for r in reports)
        flag = "" if any(r.task_counts.get(task, 0) for r in reports) else "  <- MISSING"
        print(f"{task:24s}{row}{flag}")
        if not any(r.task_counts.get(task, 0) for r in reports):
            hard_fail = True

    if len(reports) >= 2:
        leak = reports[0].video_ids & reports[1].video_ids
        if leak:
            hard_fail = True
            print(f"\nLEAKAGE: {len(leak)} video_id(s) in both "
                  f"{reports[0].name} and {reports[1].name} (e.g. {sorted(leak)[:3]})")
        else:
            print(f"\nNo video-id leakage between {reports[0].name} and {reports[1].name}.")

    for rep in reports:
        if rep.errors:
            print(f"\nFirst errors in {rep.name}:")
            print("\n".join(rep.errors[: args.max_errors]))
            if len(rep.errors) > args.max_errors:
                print(f"  ... and {len(rep.errors) - args.max_errors} more")

    status = "FAIL" if hard_fail else "OK"
    print(f"\n==> dataset check: {status}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
