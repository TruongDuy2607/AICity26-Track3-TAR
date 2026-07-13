"""Build ms-swift training jsonl from the TAR annotation files.

Reads the 10 TAR task files (``bcq.json`` … ``causal_linkage.json``), resolves
each ``video_id`` to a media path, applies the per-task prompt template + output
policy from :mod:`track3.tasks`, balances tasks, splits by *video* (no leakage),
and writes ``train.jsonl`` / ``val.jsonl`` in the ms-swift messages format:

    {"messages": [{"role":"system",...},{"role":"user","content":"<video>..."},
                  {"role":"assistant","content":"<target>"}],
     "videos": ["/abs/video.mp4"]}                 # frames-mode=video
     "videos": [["/f0.jpg", "/f1.jpg", ...]]        # frames-mode=extract

Run: ``python -m track3.build_dataset --help``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict

from track3 import frames as frame_utils
from track3.tasks import (
    ALL_TASK_KEYS,
    POLICY_ANSWER_ONLY,
    POLICY_JSON_INTERVAL,
    TASKS,
    build_user_prompt,
    frame_plan,
    get_task,
    parse_interval,
    parse_timestamp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann-dir", required=True,
                   help="Dir containing the TAR <task>.json annotation files.")
    p.add_argument("--videos-root", required=True,
                   help="Root that <video_id> relative paths resolve against.")
    p.add_argument("--out-dir", default="data", help="Where to write jsonl files.")
    p.add_argument("--tasks", nargs="*", default=ALL_TASK_KEYS,
                   help="Subset of task keys to include.")
    p.add_argument("--frames-mode", choices=["video", "extract"], default="video")
    p.add_argument("--frames-root", default="data/frames",
                   help="Cache dir for extracted frames (frames-mode=extract).")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-side", type=int, default=448)
    p.add_argument("--temporal-num-frames", type=int, default=0,
                   help="Frame budget for time-sensitive tasks (temporal); "
                        "0 = same as --num-frames. Denser frames raise IoU.")
    p.add_argument("--temporal-max-side", type=int, default=0,
                   help="Max side for temporal frames (0 = same as --max-side).")
    p.add_argument("--filter-temporal", action="store_true",
                   help="Drop temporal_localization items with degenerate/reversed "
                        "intervals (auto-label noise that caps mIoU).")
    p.add_argument("--temporal-cot", action="store_true",
                   help="Train temporal_localization with an inline timestamp "
                        "chain-of-thought mined from the `reasoning` field, then the "
                        "JSON interval (the IoU grader extracts only the trailing "
                        "JSON, so the reason is free supervision; see phase2_survey).")
    p.add_argument("--max-per-task", type=int, default=0,
                   help="Cap items per task (0 = no cap). Balances the metric.")
    p.add_argument("--val-ratio", type=float, default=0.02,
                   help="Fraction of *videos* held out for local eval.")
    p.add_argument("--skip-missing", action="store_true",
                   help="Drop items whose video file is absent.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_items(ann_dir: str, task_key: str) -> tuple[list[dict], str | None]:
    path = os.path.join(ann_dir, f"{task_key}.json")
    if not os.path.exists(path):
        return [], None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("items", []), data.get("media_root")
    return data, None  # tolerate a bare list


def resolve_video(video_id: str, videos_root: str, media_root: str | None) -> str:
    root = media_root or videos_root
    if os.path.isabs(video_id):
        return video_id
    return os.path.normpath(os.path.join(root, video_id))


def build_target(spec, item: dict, temporal_cot: bool = False) -> str:
    answer = (item.get("answer") or "").strip()
    if spec.policy == POLICY_JSON_INTERVAL:
        interval = parse_interval(answer)  # canonical {"start":"MM:SS","end":"MM:SS"}
        if temporal_cot or spec.cot_policy == "inline":
            # Surface the timestamp reasoning the grader ignores but that teaches
            # the model to anchor the interval (the auto-labels embed precise mm:ss
            # in `reasoning`, e.g. "the truck loses control at 00:54.20"). The IoU
            # grader extracts the trailing JSON, so the lead-in reason is free
            # supervision (method.md §3.2; data finding in docs/phase2_survey.md).
            reason = _temporal_cot(item.get("reasoning", ""))
            if reason:
                return f"{reason}\n{interval}"
        return interval
    if spec.policy == POLICY_ANSWER_ONLY and spec.cot_policy == "none":
        return answer
    # POLICY_ANSWER_TEXT (and any inline-CoT variants) use the answer verbatim.
    return answer


# Sub-second timestamps the auto-labeller embeds in `reasoning` (e.g. "00:54.20").
_TS = re.compile(r"\b(\d{1,2}:\d{2}(?:\.\d+)?)\b")


def _temporal_cot(reasoning: str, max_chars: int = 400) -> str:
    """Short, timestamp-bearing chain-of-thought for temporal_localization.

    Keeps only the leading sentences of the noisy auto-generated `reasoning` that
    actually mention a timestamp (the signal for IoU), capped in length so the
    target stays concise. Returns "" when no usable timestamped reasoning exists.
    """
    reasoning = (reasoning or "").strip()
    if not reasoning or not _TS.search(reasoning):
        return ""
    kept, total = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", reasoning):
        if not _TS.search(sent):
            continue
        kept.append(sent.strip())
        total += len(sent)
        if total >= max_chars:
            break
    return " ".join(kept)[:max_chars].strip()


def _scorer_gt_answer(task_key: str, answer: str) -> str:
    """Normalize a held-out GT answer to the surface form the official scorer wants.

    ``track3/evaluate.py:_gt_letter`` asserts mcq GT matches ``^([A-Za-z])\\)`` (an
    "X)" prefix), but the auto-generated TAR train labels store a *bare* letter
    ("B"). Normalize bare mcq letters to "B)" so local eval on the held-out *train*
    split doesn't crash and matches how the human-curated test GT (already "X) …")
    is graded. All other tasks pass through unchanged.
    """
    a = (answer or "").strip()
    if task_key == "mcq" and re.fullmatch(r"[A-Za-z]", a):
        return f"{a})"
    return answer


def item_index(video_id: str, task_key: str, question: str) -> str:
    """Stable 16-hex id for an item (mirrors the test.json item_index shape)."""
    h = hashlib.blake2b(f"{video_id}|{task_key}|{question}".encode("utf-8"), digest_size=8)
    return h.hexdigest()


def make_record(spec, item, video_path, frames_mode, frame_res,
                video_hint="", temporal_cot=False):
    videos_field: object
    if frames_mode == "extract":
        videos_field = [frame_res.paths]
        if spec.policy == POLICY_JSON_INTERVAL and not video_hint:
            video_hint = frame_utils.timestamp_hint(frame_res.timestamps, frame_res.duration)
    else:
        videos_field = [video_path]

    vid = item.get("video_id", "")
    user = build_user_prompt(spec, item.get("question", ""), video_hint)
    target = build_target(spec, item, temporal_cot=temporal_cot)
    return {
        "messages": [
            {"role": "system", "content": spec.system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "videos": videos_field,
        # bookkeeping (ignored by ms-swift; used for eval grouping / submission join):
        "item_index": item_index(vid, spec.key, item.get("question", "")),
        "task": spec.key,
        "task_type": spec.key,
        "video_id": vid,
        "answer": item.get("answer", ""),  # raw GT answer for local scoring
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) decide the held-out video set from the union of all video_ids.
    all_video_ids: set[str] = set()
    per_task_items: dict[str, list[dict]] = {}
    loaded: dict[str, int] = {}
    for key in args.tasks:
        items, media_root = load_items(args.ann_dir, key)
        per_task_items[key] = items
        per_task_items.setdefault(f"__media__{key}", media_root)  # stash media_root
        loaded[key] = len(items)
        for it in items:
            all_video_ids.add(it.get("video_id", ""))

    if not all_video_ids:
        present = sorted(os.listdir(args.ann_dir)) if os.path.isdir(args.ann_dir) else None
        raise SystemExit(
            "ERROR: no annotation items found — 0 videos.\n"
            f"  --ann-dir : {args.ann_dir}\n"
            f"  exists?   : {os.path.isdir(args.ann_dir)}\n"
            f"  expected  : <ann-dir>/<task>.json for {args.tasks}\n"
            f"  found in dir: {present}\n"
            "Fix ANN_DIR in configs/common.sh (it should point at the TAR 'train/' "
            "folder containing bcq.json, mcq.json, ...) and rerun.")

    # held-out videos: round(ratio*N), clamped to [0, N]; >=1 when ratio>0 and we have videos.
    k = min(len(all_video_ids), round(len(all_video_ids) * args.val_ratio))
    if args.val_ratio > 0:
        k = max(1, k)
    val_videos = set(rng.sample(sorted(all_video_ids), k=k)) if k > 0 else set()

    # 2) build records, with per-task caps and frame caching per video.
    # Cache key includes the frame budget so temporal's denser extract and any
    # base-mode extract of the same video don't collide.
    frame_cache: dict[tuple, frame_utils.FrameResult] = {}
    stats = defaultdict(lambda: {"train": 0, "val": 0, "missing": 0, "filtered": 0})
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    val_gt_items: list[dict] = []  # tao-vl-reason-v1.0 items for the official scorer

    with open(train_path, "w", encoding="utf-8") as ftr, \
         open(val_path, "w", encoding="utf-8") as fval:
        for key in args.tasks:
            spec = get_task(key)
            items = list(per_task_items[key])
            media_root = per_task_items.get(f"__media__{key}")
            rng.shuffle(items)
            if args.max_per_task > 0:
                items = _cap_balanced(items, args.max_per_task)
            for it in items:
                vid = it.get("video_id", "")
                if (spec.key == "temporal_localization" and args.filter_temporal
                        and not _valid_interval(it.get("answer", ""))):
                    stats[key]["filtered"] += 1
                    continue
                vpath = resolve_video(vid, args.videos_root, media_root)
                if not os.path.exists(vpath):
                    if args.skip_missing:
                        stats[key]["missing"] += 1
                        continue  # else: keep path; useful for dry-runs

                # per-task framing: temporal gets denser timestamped extract frames.
                eff_mode, eff_frames, eff_side = frame_plan(
                    spec, args.frames_mode, args.num_frames, args.max_side,
                    args.temporal_num_frames or None, args.temporal_max_side or None)
                frame_res = None
                if eff_mode == "extract" and os.path.exists(vpath):
                    ck = (vid, eff_frames, eff_side)
                    if ck not in frame_cache:
                        out_dir = os.path.join(
                            args.frames_root, f"{_safe(vid)}_n{eff_frames}_s{eff_side}")
                        frame_cache[ck] = frame_utils.extract_uniform_frames(
                            vpath, out_dir, eff_frames, eff_side)
                    frame_res = frame_cache[ck]
                elif eff_mode == "extract":
                    eff_mode = "video"  # dry-run without the video file: fall back

                # Native-video temporal items carry a duration hint (mirrors infer)
                # so the model has the wall-clock axis it needs to emit MM:SS.
                video_hint = ""
                if (eff_mode == "video" and spec.policy == POLICY_JSON_INTERVAL
                        and os.path.exists(vpath)):
                    video_hint = frame_utils.duration_hint(
                        frame_utils.video_duration(vpath))

                rec = make_record(spec, it, vpath, eff_mode, frame_res,
                                  video_hint=video_hint, temporal_cot=args.temporal_cot)
                split = "val" if vid in val_videos else "train"
                (fval if split == "val" else ftr).write(
                    json.dumps(rec, ensure_ascii=False) + "\n")
                stats[key][split] += 1
                if split == "val":
                    val_gt_items.append({
                        "item_index": rec["item_index"],
                        "task_type": spec.key,
                        "video_id": vid,
                        "question": it.get("question", ""),
                        # GT in the surface form the official scorer expects
                        # (mcq _gt_letter needs an "X)" prefix; train labels are bare).
                        "answer": _scorer_gt_answer(spec.key, it.get("answer", "")),
                    })

    val_gt_path = _write_val_gt(args.out_dir, val_gt_items)
    _report(stats, train_path, val_path)
    print(f"Wrote held-out GT for the official scorer: {val_gt_path} "
          f"({len(val_gt_items)} items)")


def _write_val_gt(out_dir: str, items: list[dict]) -> str:
    """Emit val_gt.json in tao-vl-reason-v1.0 form (consumed by track3/evaluate.py)."""
    doc = {
        "format": "tao-vl-reason-v1.0",
        "metadata": {"type": "annotation", "task": "tar_val_holdout",
                     "description": "Held-out split for local scoring.",
                     "license": "CC-BY-4.0"},
        "media_root": None,
        "items": items,
    }
    path = os.path.join(out_dir, "val_gt.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return path


def _cap_balanced(items: list[dict], cap: int) -> list[dict]:
    """Cap items at ``cap`` while preserving Yes/No balance when present."""
    if len(items) <= cap:
        return items
    yes = [i for i in items if (i.get("answer", "").strip().lower() == "yes")]
    no = [i for i in items if (i.get("answer", "").strip().lower() == "no")]
    if yes and no:  # bcq: keep an even Yes/No split
        half = cap // 2
        return yes[:half] + no[:cap - half]
    return items[:cap]


def _valid_interval(answer: str) -> bool:
    """True if a temporal label is a usable, non-degenerate interval.

    ``parse_interval`` always yields valid JSON (00:00/00:00 on failure), so we
    test the parsed seconds: drop unparseable, zero-length, or reversed
    (end < start) intervals — auto-generated noise that caps mIoU regardless of
    the model's prediction.
    """
    try:
        obj = json.loads(parse_interval(answer))
        return parse_timestamp(obj["end"]) > parse_timestamp(obj["start"])
    except Exception:
        return False


def _safe(video_id: str) -> str:
    return frame_utils.safe_name(video_id)


def _report(stats, train_path, val_path) -> None:
    print(f"\n{'task':24s} {'train':>8s} {'val':>6s} {'missing':>8s} {'filtered':>9s}")
    tot = {"train": 0, "val": 0, "missing": 0, "filtered": 0}
    for key in sorted(stats):
        s = stats[key]
        print(f"{key:24s} {s['train']:>8d} {s['val']:>6d} {s['missing']:>8d} "
              f"{s['filtered']:>9d}")
        for k in tot:
            tot[k] += s[k]
    print(f"{'TOTAL':24s} {tot['train']:>8d} {tot['val']:>6d} {tot['missing']:>8d} "
          f"{tot['filtered']:>9d}")
    print(f"\nWrote {train_path} and {val_path}")


if __name__ == "__main__":
    main()
