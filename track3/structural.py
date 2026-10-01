"""Structural post-processor: rule-based fixes on (test.json, predictions jsonl).

Each rule is flag-gated and keeps the model prediction when its precondition is missing.

  --temporal-prior   temporal_localization := the [X,Y] window of the same video's
                     temporal_description question (fallback: causal_linkage window).
  --bcq-pairing      a video's two bcq items are {Yes, No}: when both agree, flip the
                     lower-confidence one (by the ``votes`` margin).

Run::

    python -m track3.structural --test-json data/test/test.json \
        --pred preds/test_pred.jsonl --out preds/test_pred.struct.jsonl --all
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from track3.tasks import extract_yesno, parse_timestamp

# MM:SS / MM:SS.ff / HH:MM:SS timestamps in TAR questions.
_TS = re.compile(r"\d{1,2}:\d{2}(?:\.\d+)?(?::\d{2}(?:\.\d+)?)?")
# MM:SS:ff typo: a colon used as the fractional separator, e.g. "00:08:78".
_COLON3 = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)$")


def normalize_ts(ts: str) -> str:
    """Repair the MM:SS:ff typo to MM:SS.ff (clips are < ~60 s, so d:dd:dd > 60 s is the typo)."""
    ts = str(ts).strip()
    m = _COLON3.match(ts)
    if m and parse_timestamp(ts) > 60.0:
        return f"{m.group(1)}:{m.group(2)}.{m.group(3)}"
    return ts


def resolve_video(videos_root: str, vid: str) -> str:
    """Absolute path for a video_id under videos_root (absolute ids pass through)."""
    return vid if os.path.isabs(vid) else os.path.join(videos_root, vid)


def question_window(question: str) -> tuple[str, str] | None:
    """The [X,Y] timestamps of a td/cl question (typo-repaired, ordered), or None."""
    ts = _TS.findall(question or "")
    if len(ts) < 2:
        return None
    ts = sorted((normalize_ts(t) for t in ts[:2]), key=parse_timestamp)
    return ts[0], ts[1]


def question_stem(question: str) -> str:
    """First question line, lowercased (bcq <-> bcq_openended join key)."""
    return (question or "").strip().splitlines()[0].strip().lower()


_OPTION = re.compile(r"^\s*([A-D])[).]\s*(.+?)\s*$", re.MULTILINE)


def question_options(question: str, normalize: bool = True) -> dict[str, str]:
    """{letter: option text}; ``normalize`` lowercases and strips punctuation."""
    out = {}
    for m in _OPTION.finditer(question or ""):
        text = m.group(2).strip()
        out[m.group(1)] = re.sub(r"\W+", " ", text).strip().lower() if normalize else text
    return out


def _vote_margin(rec: dict) -> float:
    """Winner-minus-runner-up weight of the ``votes`` field ({token: weight}), else 0."""
    votes = rec.get("votes") or {}
    counts = sorted(votes.values(), reverse=True)
    if len(counts) >= 2:
        return float(counts[0] - counts[1])
    return float(counts[0]) if counts else 0.0


def apply_temporal_prior(items_by_video: dict, preds: dict,
                         override: dict | None = None) -> int:
    """Replace temporal_localization predictions with the td/cl question window.

    ``override`` (item_index -> {"start","end"}) takes precedence per item.
    """
    override = override or {}
    changed = 0
    for vid, tasks in items_by_video.items():
        window = None
        for src in ("temporal_description", "causal_linkage"):
            for it in tasks.get(src, ()):
                window = question_window(it["question"])
                if window:
                    break
            if window:
                break
        for it in tasks.get("temporal_localization", ()):
            rec = preds.get(it["item_index"])
            if rec is None:
                continue
            chosen = override.get(it["item_index"])
            if chosen is not None:
                start, end = chosen["start"], chosen["end"]
            elif window is not None:
                start, end = window
            else:
                continue  # keep the model's prediction
            interval = json.dumps({"start": start, "end": end})
            rec["prediction"] = f"```json\n{interval}\n```"
            rec["structural"] = "temporal_override" if chosen is not None else "temporal_prior"
            changed += 1
    return changed


def _retoken_open(rec: dict, token: str) -> int:
    """Force an open-ended answer to open with ``token`` (keep the explanation)."""
    text = (rec.get("prediction") or "").strip()
    lead = re.match(r"^(yes|no|[A-D])\b[.,:)]?\s*", text, re.IGNORECASE)
    rest = text[lead.end():] if lead else text
    new = f"{token}. {rest}".strip()
    if new != text:
        rec["prediction"] = new
        rec["structural"] = "bcq_pairing"
        return 1
    return 0


def apply_bcq_pairing(items_by_video: dict, preds: dict) -> int:
    """When a video's two bcq answers agree, flip the lower-confidence one."""
    changed = 0
    for vid, tasks in items_by_video.items():
        for key in ("bcq", "bcq_openended"):
            pair = [preds[it["item_index"]] for it in tasks.get(key, ())
                    if it["item_index"] in preds]
            if len(pair) != 2:
                continue
            toks = [extract_yesno(r["prediction"]) for r in pair]
            if None in toks or toks[0] != toks[1]:
                continue
            conf = [_vote_margin(r) for r in pair]
            weak = pair[0] if conf[0] <= conf[1] else pair[1]
            flipped = "No" if toks[0] == "yes" else "Yes"
            if key == "bcq":
                weak["prediction"] = flipped
                changed += 1
            else:
                changed += _retoken_open(weak, flipped)
            weak["structural"] = "bcq_pairing"
    return changed


def apply_text_override(preds: dict, override: dict) -> int:
    """Replace predictions with external text (item_index -> text); absent items are kept."""
    changed = 0
    for idx, text in override.items():
        rec = preds.get(str(idx))
        if rec is None or not text:
            continue
        rec["prediction"] = text
        rec["structural"] = "text_override"
        changed += 1
    return changed


def load_items_by_video(test_json: str) -> dict:
    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data
    by_video: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_video[it["video_id"]][it.get("task_type") or it.get("task")].append(it)
    return by_video


def _load_temporal_override(path: str) -> dict:
    """{item_index: {"start","end"}} from a temporal-override jsonl, or {}."""
    if not path:
        return {}
    out = {}
    for rec in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()):
        out[str(rec["item_index"])] = {"start": rec["start"], "end": rec["end"]}
    return out


def _load_text_override(path: str) -> dict:
    """{item_index: prediction text} from a text-override jsonl, or {}."""
    if not path:
        return {}
    out = {}
    for rec in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()):
        out[str(rec["item_index"])] = rec.get("prediction") or rec.get("text") or ""
    return out


def run(test_json: str, pred_path: str, out_path: str, rules: dict[str, bool],
        temporal_override: str = "", text_override: str = "") -> dict:
    items_by_video = load_items_by_video(test_json)
    preds: dict[str, dict] = {}
    order: list[str] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[str(rec["item_index"])] = rec
                order.append(str(rec["item_index"]))

    report = {}
    # text override first, so the rules below see the final text
    if text_override:
        report["text_override"] = apply_text_override(
            preds, _load_text_override(text_override))
    if rules.get("temporal_prior"):
        report["temporal_prior"] = apply_temporal_prior(
            items_by_video, preds, _load_temporal_override(temporal_override))
    if rules.get("bcq_pairing"):
        report["bcq_pairing"] = apply_bcq_pairing(items_by_video, preds)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx in order:
            rec = dict(preds[idx])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--pred", required=True, help="predictions jsonl from infer.py")
    ap.add_argument("--out", required=True, help="revised predictions jsonl")
    ap.add_argument("--temporal-prior", action="store_true")
    ap.add_argument("--bcq-pairing", action="store_true")
    ap.add_argument("--all", action="store_true", help="enable every rule")
    ap.add_argument("--temporal-override", default="",
                    help="jsonl with (item_index,start,end) to feed rule 4.1 a "
                         "per-item temporal window from an external method.")
    ap.add_argument("--text-override", default="",
                    help="jsonl with (item_index,prediction) to replace open-ended "
                         "text answers from an external method (track3.text_dossier). "
                         "Applied regardless of the rule flags; partial-safe.")
    args = ap.parse_args()

    rules = {k: bool(args.all or getattr(args, k)) for k in
             ("temporal_prior", "bcq_pairing")}
    if not any(rules.values()) and not args.text_override:
        ap.error("enable at least one rule (or --all), or pass --text-override")
    report = run(args.test_json, args.pred, args.out, rules,
                 temporal_override=args.temporal_override,
                 text_override=args.text_override)
    for rule, n in report.items():
        print(f"[structural] {rule:16s} changed {n} prediction(s)")
    print(f"[structural] wrote {args.out}")


if __name__ == "__main__":
    main()
