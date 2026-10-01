"""Restore the untrimmed temporal_description MBR picks.

For each TD item take the longest pool candidate that extends the base row's text
(both opener-stripped); otherwise keep the base row.

- pipeline mode: rewrite the TD rows of the MBR text-override jsonl::

    python -m track3.td_untrim --override-jsonl preds/text_override.jsonl \
        --candidates preds/candidates.jsonl --out-jsonl preds/text_override.untrim.jsonl

- A/B mode: derive a CSV variant from a base submission::

    python -m track3.td_untrim --test-json data/test/test.json \
        --base-csv submissions/32b-top2/submission-subR.csv \
        --candidates preds/union/candidates.union.jsonl \
        --out-csv submissions/32b-top2/submission-tdLong.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics

TASK = "temporal_description"
PREFIX_CHARS = 120     # prefix-match anchor length
MIN_EXTRA_CHARS = 20   # minimum gain to accept a longer candidate

_OPENER = re.compile(
    r"^Between \d{1,2}:\d{2}(?::\d{1,2}|\.\d{1,2})? and "
    r"\d{1,2}:\d{2}(?::\d{1,2}|\.\d{1,2})?,\s*")


def strip_opener(text: str) -> str:
    """Drop a leading "Between X and Y, " window phrase."""
    s = _OPENER.sub("", (text or "").strip())
    return s[:1].upper() + s[1:] if s else s


def recover(base_text: str, candidates: list[str]) -> str:
    """Longest opener-stripped candidate extending the stripped base text, else the stripped base."""
    base = strip_opener(base_text)
    anchor = base[:PREFIX_CHARS]
    if not anchor:
        return base_text
    hits = [c for c in (strip_opener(c) for c in candidates) if c.startswith(anchor)]
    best = max(hits, key=len, default="")
    return best if len(best) >= len(base) + MIN_EXTRA_CHARS else base


def apply(base: dict[str, str], td_indices: list[str],
          pool: dict[str, list[str]]) -> dict[str, str]:
    out = dict(base)
    for idx in td_indices:
        out[idx] = recover(base[idx], pool.get(idx, []))
    return out


def apply_jsonl(rows: list[dict], pool: dict[str, list[str]]) -> list[dict]:
    """Untrim the TD rows of a text-override jsonl; changed rows get a ``+untrim`` source."""
    out = []
    for row in rows:
        row = dict(row)
        if row.get("task") == TASK:
            new = recover(row.get("prediction", ""), pool.get(row["item_index"], []))
            if new != row.get("prediction"):
                row["prediction"] = new
                row["source"] = f"{row.get('source', 'mbr')}+untrim"
        out.append(row)
    return out


def _load_pool(path: str) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("task") == TASK:
                pool[rec["item_index"]] = rec.get("candidates", [])
    return pool


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True,
                   help="candidates jsonl holding the untrimmed renders.")
    p.add_argument("--override-jsonl", default="",
                   help="pipeline mode: text-override jsonl whose TD rows to untrim.")
    p.add_argument("--out-jsonl", default="",
                   help="pipeline mode output (required with --override-jsonl).")
    p.add_argument("--test-json", default="data/test/test.json")
    p.add_argument("--base-csv", default="",
                   help="A/B mode: scored base submission CSV.")
    p.add_argument("--out-csv", default="",
                   help="A/B mode output (required with --base-csv).")
    a = p.parse_args()

    pool = _load_pool(a.candidates)

    # pipeline mode: jsonl -> jsonl
    if a.override_jsonl:
        assert a.out_jsonl, "--out-jsonl is required with --override-jsonl"
        with open(a.override_jsonl, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        out_rows = apply_jsonl(rows, pool)
        changed = sum(1 for b, o in zip(rows, out_rows) if b != o)
        with open(a.out_jsonl, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        td_lens = [len(r.get("prediction", "")) for r in out_rows if r.get("task") == TASK]
        print(f"[td_untrim] {changed} TD row(s) extended"
              + (f", TD len median {statistics.median(td_lens):.0f}" if td_lens else "")
              + f" -> {a.out_jsonl}")
        return

    # A/B mode: submission CSV -> CSV
    assert a.base_csv and a.out_csv, "need --base-csv + --out-csv (or --override-jsonl)"
    data = json.load(open(a.test_json, encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    td_idx = [it["item_index"] for it in items if it.get("task_type") == TASK]

    with open(a.base_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    base = {r["item_index"]: r["prediction"] for r in rows}
    order = [r["item_index"] for r in rows]

    out = apply(base, td_idx, pool)

    # hard gates: full index set, TD-only diff, no empties
    assert set(out) == set(base) == {it["item_index"] for it in items}, "index mismatch"
    changed = [k for k in base if out[k] != base[k]]
    assert all(k in set(td_idx) for k in changed), "a non-TD row changed"
    assert all((v or "").strip() for v in out.values()), "empty prediction produced"

    with open(a.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_index", "prediction"])
        for idx in order:
            w.writerow([idx, out[idx]])

    lens_b = [len(base[i]) for i in td_idx]
    lens_o = [len(out[i]) for i in td_idx]
    print(f"[td_untrim] {len(changed)}/{len(td_idx)} TD rows extended, len median "
          f"{statistics.median(lens_b):.0f} -> {statistics.median(lens_o):.0f} "
          f"-> {a.out_csv}")


if __name__ == "__main__":
    main()
