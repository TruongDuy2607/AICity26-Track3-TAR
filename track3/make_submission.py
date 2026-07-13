"""Turn raw predictions into the official submission CSV.

Reads the predictions jsonl from :mod:`track3.infer` and emits ``item_index,prediction``
with each prediction shaped to what the grader expects. Per the official
``track3/evaluate.py`` the grader extracts the gradable token from raw model text
itself, so we submit the text essentially verbatim — *except* temporal_localization,
which we repair into a parseable fenced ```json block. 960 rows expected.

Run: ``python -m track3.make_submission --pred preds/test_pred.jsonl --out submissions/submission.csv``
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from track3.tasks import get_task, infer_task_key


def write_submission(pred_path: str, out_path: str, test_json: str = "") -> int:
    """Build the submission CSV. Returns the number of rows written."""
    preds: dict[str, dict] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[str(r["item_index"])] = r

    order = _test_order(test_json) if test_json else list(preds.keys())

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["item_index", "prediction"])
        for idx in order:
            r = preds.get(str(idx))
            if r is None:
                w.writerow([idx, ""])  # never drop a required row
                continue
            task = r.get("task") or r.get("task_type") or infer_task_key(r)
            spec = get_task(task)
            w.writerow([idx, spec.submission(r.get("prediction", ""))])
    return len(order)


def _test_order(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data
    return [str(it.get("item_index") or it.get("index") or it.get("id")) for it in items]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="predictions jsonl from infer.py")
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--test-json", default="",
                    help="Optional test.json to enforce row order / completeness.")
    args = ap.parse_args()
    n = write_submission(args.pred, args.out, args.test_json)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
