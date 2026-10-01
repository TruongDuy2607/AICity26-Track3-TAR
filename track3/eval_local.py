"""Local evaluation on a held-out split, scored by the official ``track3/evaluate.py``.

  --gt val_gt.json --pred preds/val_pred.jsonl   build a submission from raw preds, then score
  --gt val_gt.json --submission my_sub.csv       score an existing submission CSV

Run: ``python -m track3.eval_local --gt data/val_gt.json --pred preds/val_pred.jsonl``
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from track3.make_submission import write_submission
from track3.official import load_official


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True,
                    help="Held-out GT in tao-vl-reason-v1.0 format with real answers.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", help="Raw predictions jsonl from infer.py.")
    src.add_argument("--submission", help="An already-built submission CSV.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Score the remainder when some GT items lack a prediction.")
    ap.add_argument("--out", default="", help="Optional json file for the metrics.")
    args = ap.parse_args()

    official = load_official()

    tmp = None
    if args.submission:
        sub_csv = args.submission
    else:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        tmp.close()
        n = write_submission(args.pred, tmp.name, test_json=args.gt)
        print(f"[eval] built submission from {args.pred}: {n} rows")
        sub_csv = tmp.name

    result = official.evaluate(args.gt, sub_csv, allow_missing=args.allow_missing)

    print()
    if result.get("mode") == "score":
        metrics = result["metrics"]
        _print_metrics(metrics)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            print(f"[eval] wrote {args.out}")
    else:
        print(f"[eval] validation-only ({result.get('reason')}). "
              "Pass a GT with real answers (data/val_gt.json) to get scores.")

    if tmp is not None:
        os.unlink(tmp.name)


def _print_metrics(metrics: dict) -> None:
    mean = metrics.get("mean")
    print(f"{'metric':36s} {'score':>8s}")
    print("-" * 46)
    for k in sorted(metrics):
        if k == "mean":
            continue
        print(f"{k:36s} {metrics[k]:>8.4f}")
    print("-" * 46)
    if mean is not None:
        print(f"{'OVERALL (unweighted mean)':36s} {mean:>8.4f}")


if __name__ == "__main__":
    main()
