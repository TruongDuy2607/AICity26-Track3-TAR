"""Analysis + aggregation for the §5.2 controlled ablations (scripts/ablations/).

Pure post-processing over jsonl/json the heavy GPU stages already wrote — no torch,
no model, no bert-score — so it runs on a CPU dev box. Subcommands:

  reliability   closed-form P_theta calibration: per-bin accuracy vs confidence,
                ECE, and accuracy at each margin gate (E1b). Reads the ``votes``
                first-token distribution in val_pred.jsonl against val_gt.json.
  selfverify    aggregate the raw per-claim self-verification distribution
                (claim_verify --claim-probs-out): mean P(Yes) and the fraction of
                the generator's OWN claims its verifier rejects (E1a, generator side).
  gt-preds      build the oracle sheet inputs from GT (E2 B2): a predictions jsonl
                (answer-as-prediction) + a scene jsonl (mined from GT scene_description).
  override      override-vs-propagation rate for an injected wrong fact (E4): reads
                the sheets dump (text_dossier --sheets-out) + the override predictions.
  tabulate      merge several eval_local metric jsons into one markdown table with a
                narrative-only mean row (the paper's Table-2 mean definition).

Run: ``python -m track3.ablate_report <subcommand> ...`` (see each --help).
"""
from __future__ import annotations

import argparse
import json
import os


# ---------------------------------------------------------------------------
# shared IO
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", data) if isinstance(data, dict) else data


# The nine graded task types (temporal_localization mIoU is excluded from Mean).
NARRATIVE_TYPES = ("bcq_openended", "mcq_openended", "open_qa", "causal_linkage",
                   "scene_description", "temporal_description", "video_summarization")
CLOSED_TYPES = ("bcq", "mcq")
GRADED_METRICS = (
    "bcq_accuracy", "mcq_accuracy",
    *(f"{t}_bertscore_f1" for t in NARRATIVE_TYPES),
)


# ---------------------------------------------------------------------------
# E1b — closed-form reliability / ECE
# ---------------------------------------------------------------------------

def _dist_top2(votes: dict) -> tuple[str, float, float]:
    """(top_token, p1, p2) from a first-token distribution, renormalised to sum 1."""
    if not votes:
        return "", 0.0, 0.0
    tot = sum(votes.values()) or 1.0
    ordered = sorted(((k, v / tot) for k, v in votes.items()), key=lambda kv: -kv[1])
    top, p1 = ordered[0]
    p2 = ordered[1][1] if len(ordered) > 1 else 0.0
    return top, p1, p2


def reliability(pred_path: str, gt_path: str, nbins: int, plot: str = "") -> dict:
    from track3.official import load_official
    o = load_official()
    gt = {str(it.get("item_index")): it for it in _load_items(gt_path)}
    rows = []  # (conf, margin, correct)
    per_task_hits: dict[str, list[int]] = {"bcq": [], "mcq": []}
    for rec in _load_jsonl(pred_path):
        task = rec.get("task")
        if task not in ("bcq", "mcq"):
            continue
        votes = rec.get("votes")
        gi = gt.get(str(rec["item_index"]))
        if not votes or gi is None or not str(gi.get("answer", "")).strip():
            continue
        top, p1, p2 = _dist_top2(votes)
        try:
            if task == "bcq":
                correct = int(top.lower() == o._gt_yesno(gi["answer"]))
            else:
                correct = int(top.upper() == o._gt_letter(gi["answer"]))
        except AssertionError:
            continue
        rows.append((p1, p1 - p2, correct))
        per_task_hits[task].append(correct)

    if not rows:
        raise SystemExit("[reliability] no scored closed-form items — did infer run "
                         "with --closed-scoring logprob (writes 'votes')?")

    # equal-width confidence bins + ECE.
    bins = [[] for _ in range(nbins)]
    for conf, _m, correct in rows:
        b = min(nbins - 1, int(conf * nbins))
        bins[b].append((conf, correct))
    n = len(rows)
    ece, table = 0.0, []
    for b, bucket in enumerate(bins):
        if not bucket:
            table.append({"bin": b, "n": 0, "conf": None, "acc": None})
            continue
        conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(k for _, k in bucket) / len(bucket)
        ece += len(bucket) / n * abs(acc - conf)
        table.append({"bin": b, "range": [b / nbins, (b + 1) / nbins],
                      "n": len(bucket), "conf": round(conf, 4), "acc": round(acc, 4)})

    # accuracy at margin gates (the safe-fallback gate signal).
    gates = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    gate_tbl = []
    for g in gates:
        kept = [(c, k) for c, m, k in rows if m >= g]
        gate_tbl.append({"gate": g, "coverage": round(len(kept) / n, 4),
                         "acc": round(sum(k for _, k in kept) / len(kept), 4) if kept else None})

    result = {
        "n": n, "ece": round(ece, 4),
        "acc": {t: round(sum(v) / len(v), 4) for t, v in per_task_hits.items() if v},
        "bins": table, "margin_gates": gate_tbl,
    }
    _print_reliability(result)
    if plot:
        _plot_reliability(table, result["ece"], plot)
    return result


def _print_reliability(r: dict) -> None:
    print(f"\n[reliability] closed-form P_theta over {r['n']} item(s)  ECE={r['ece']}")
    print(f"  accuracy: " + ", ".join(f"{k}={v}" for k, v in r["acc"].items()))
    print(f"  {'conf-bin':>12} {'n':>5} {'conf':>7} {'acc':>7}")
    for b in r["bins"]:
        if not b.get("n"):
            continue
        lo, hi = b["range"]
        print(f"  [{lo:.2f},{hi:.2f}) {b['n']:>5} {b['conf']:>7.4f} {b['acc']:>7.4f}")
    print(f"  {'margin>=':>12} {'cover':>7} {'acc':>7}")
    for g in r["margin_gates"]:
        acc = "  n/a" if g["acc"] is None else f"{g['acc']:.4f}"
        print(f"  {g['gate']:>12} {g['coverage']:>7.4f} {acc:>7}")


def _plot_reliability(table: list[dict], ece: float, out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # optional dependency
        print(f"[reliability][WARN] matplotlib unavailable ({e}); skipping plot.")
        return
    pts = [(b["conf"], b["acc"]) for b in table if b.get("n")]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#1f77b4", label=f"P_theta (ECE={ece:.3f})")
    ax.set_xlabel("confidence (top first-token prob)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Closed-form reliability")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[reliability] wrote plot -> {out}")


# ---------------------------------------------------------------------------
# E1a — generator-side self-verification distribution
# ---------------------------------------------------------------------------

def selfverify(claim_probs_path: str) -> dict:
    probs = [r["p_yes"] for r in _load_jsonl(claim_probs_path)]
    if not probs:
        raise SystemExit("[selfverify] no claim probs — run claim_verify "
                         "--candidates <narratives> --claim-probs-out ...")
    n = len(probs)
    mean = sum(probs) / n
    frac = lambda t: sum(1 for p in probs if p < t) / n
    hist = [0] * 10
    for p in probs:
        hist[min(9, int(p * 10))] += 1
    res = {"n_claims": n, "mean_p_yes": round(mean, 4),
           "frac_below_0.5": round(frac(0.5), 4),
           "frac_below_0.3": round(frac(0.3), 4),
           "hist_deciles": hist}
    print(f"\n[selfverify] {n} generated claim(s); P_theta endorses them at mean "
          f"P(Yes)={res['mean_p_yes']}")
    print(f"  self-rejected  (P(Yes)<0.5): {res['frac_below_0.5']:.1%}")
    print(f"  strongly rej.  (P(Yes)<0.3): {res['frac_below_0.3']:.1%}")
    print(f"  decile histogram [0..1]: {hist}")
    return res


# ---------------------------------------------------------------------------
# E2 B2 — oracle sheet inputs from GT
# ---------------------------------------------------------------------------

def gt_preds(gt_path: str, preds_out: str, scene_out: str) -> None:
    """Emit (a) a predictions jsonl with the GT answer as the prediction — so
    text_dossier's clip_facts mines the sheet's cause/consequence/observations from
    ground truth; and (b) a scene jsonl mined from the GT scene_description. Together
    they build the ORACLE Evidence Sheet (the pool of what a perfect verifier could
    have supplied)."""
    from track3.evidence import mine_scene_attributes
    items = _load_items(gt_path)
    with open(preds_out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "item_index": it.get("item_index"),
                "video_id": it.get("video_id", ""),
                "task": it.get("task_type") or it.get("task"),
                "prediction": str(it.get("answer", "")),
            }, ensure_ascii=False) + "\n")
    scene_by_vid: dict[str, list] = {}
    for it in items:
        if (it.get("task_type") or it.get("task")) == "scene_description":
            scene_by_vid[it.get("video_id", "")] = list(
                mine_scene_attributes(str(it.get("answer", ""))))
    with open(scene_out, "w", encoding="utf-8") as f:
        for vid, scene in scene_by_vid.items():
            f.write(json.dumps({"video_id": vid, "scene": scene},
                               ensure_ascii=False) + "\n")
    print(f"[gt-preds] wrote {len(items)} GT prediction(s) -> {preds_out}")
    print(f"[gt-preds] wrote {len(scene_by_vid)} GT scene line(s) -> {scene_out}")


def gen_scene(pred_path: str, scene_out: str) -> None:
    """Scene attributes mined from the model's OWN generated scene_description text
    (the free-generation channel) — the G->G counterpart to the calibrated P_theta
    scene probes (E3). Same mining rule, different source, so the sheet fields are
    populated from generation instead of from a calibrated first-token readout."""
    from track3.evidence import mine_scene_attributes
    scene_by_vid: dict[str, list] = {}
    for r in _load_jsonl(pred_path):
        if r.get("task") == "scene_description":
            scene_by_vid[r.get("video_id", "")] = list(
                mine_scene_attributes(str(r.get("prediction", ""))))
    with open(scene_out, "w", encoding="utf-8") as f:
        for vid, scene in scene_by_vid.items():
            f.write(json.dumps({"video_id": vid, "scene": scene},
                               ensure_ascii=False) + "\n")
    print(f"[gen-scene] wrote {len(scene_by_vid)} generated scene line(s) -> {scene_out}")


# ---------------------------------------------------------------------------
# E4 — override vs propagation of an injected wrong fact
# ---------------------------------------------------------------------------

def _tokens(value) -> list[str]:
    """Salient lowercase tokens of an injected value (for substring matching in the
    narrative). Lists (cast/scene) -> their elements; strings -> the whole phrase."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def override_rate(sheets_path: str, override_path: str) -> dict:
    """For every clip where a wrong fact was injected, decide per graded item whether
    the narrative LEAKED the injected value (propagation) or dropped it (override).
    An injected token counts as leaked if it appears verbatim in the output and the
    original truth token does not. Clips with no injection are ignored."""
    injected = {r["video_id"]: r["ablate"] for r in _load_jsonl(sheets_path)
                if r.get("ablate") and r["ablate"].get("mode") == "corrupt"}
    if not injected:
        raise SystemExit("[override] no injected clips in the sheets dump — run "
                         "text_dossier with --sheet-transform corrupt:FIELD --sheets-out.")
    preds = _load_jsonl(override_path)
    n_leak = n_override = n_total = 0
    by_video: dict[str, dict] = {}
    for p in preds:
        vid = p.get("video_id", "")
        rec = injected.get(vid)
        if not rec:
            continue
        text = (p.get("prediction") or "").lower()
        inj_tokens = _tokens(rec.get("injected"))
        orig_tokens = _tokens(rec.get("original"))
        leaked = any(t and t in text for t in inj_tokens)
        n_total += 1
        if leaked:
            n_leak += 1
        else:
            n_override += 1
        v = by_video.setdefault(vid, {"leak": 0, "override": 0,
                                      "field": rec.get("field")})
        v["leak" if leaked else "override"] += 1
    leak_clips = sum(1 for v in by_video.values() if v["leak"] > 0)
    res = {"clips_injected": len(injected),
           "clips_with_narrative": len(by_video),
           "items_scored": n_total,
           "leak_items": n_leak, "override_items": n_override,
           "item_override_rate": round(n_override / n_total, 4) if n_total else None,
           "clips_leaked": leak_clips,
           "clip_leak_rate": round(leak_clips / max(1, len(by_video)), 4)}
    print(f"\n[override] injected wrong fact into {len(injected)} clip(s); "
          f"{len(by_video)} produced a narrative.")
    print(f"  item-level: {n_override}/{n_total} overridden "
          f"({res['item_override_rate']}), {n_leak} leaked.")
    print(f"  clip-level: {leak_clips}/{len(by_video)} clip(s) leaked the wrong fact "
          f"into >=1 narrative  ->  paper's X/N failure count.")
    return res


# ---------------------------------------------------------------------------
# E1a — turn a predictions/override jsonl into a claim_verify candidates file
# ---------------------------------------------------------------------------

def to_candidates(pred_path: str, out_path: str, tasks: tuple[str, ...]) -> None:
    """One-candidate-per-item candidates jsonl (the item's narrative), so
    claim_verify --candidates can probe the generator's OWN claims (E1a). Only the
    narrative tasks are kept — closed decisions have no free-text claims to verify."""
    keep = set(tasks)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in _load_jsonl(pred_path):
            if r.get("task") not in keep:
                continue
            text = (r.get("prediction") or "").strip()
            if not text:
                continue
            f.write(json.dumps({"item_index": r["item_index"],
                                "video_id": r.get("video_id", ""),
                                "task": r.get("task"), "candidates": [text]},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"[to-candidates] wrote {n} candidate(s) -> {out_path}")


# ---------------------------------------------------------------------------
# tabulate several eval_local metric jsons into one markdown table
# ---------------------------------------------------------------------------

def tabulate(labels_paths: list[str]) -> None:
    """``labels_paths`` = ["Label=metrics.json", ...]. Prints a markdown table with
    one column per condition and a Narrative-mean row (mean over the 7 BERTScore
    types) plus the 9-type Graded-mean, matching the paper's Table-2 definition."""
    cols = []
    for lp in labels_paths:
        label, _, path = lp.partition("=")
        with open(path, encoding="utf-8") as f:
            cols.append((label, json.load(f)))

    def _mean(metrics: dict, keys) -> float | None:
        vals = [metrics[k] for k in keys if k in metrics]
        return sum(vals) / len(vals) if vals else None

    rows = list(GRADED_METRICS)
    print("\n| metric | " + " | ".join(l for l, _ in cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for k in rows:
        cells = []
        for _, m in cols:
            cells.append(f"{m[k]:.4f}" if k in m else "—")
        print(f"| {k} | " + " | ".join(cells) + " |")
    nar_keys = [f"{t}_bertscore_f1" for t in NARRATIVE_TYPES]
    for name, keys in (("**Narrative-mean (7)**", nar_keys),
                       ("**Graded-mean (9)**", list(GRADED_METRICS))):
        cells = []
        for _, m in cols:
            v = _mean(m, keys)
            cells.append(f"{v:.4f}" if v is not None else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reliability", help="closed-form calibration + ECE (E1b).")
    r.add_argument("--pred", required=True, help="val_pred.jsonl (with 'votes').")
    r.add_argument("--gt", required=True, help="val_gt.json.")
    r.add_argument("--nbins", type=int, default=10)
    r.add_argument("--plot", default="", help="optional reliability-diagram PNG.")
    r.add_argument("--out", default="", help="optional metrics json.")

    s = sub.add_parser("selfverify", help="generator self-verification dist (E1a).")
    s.add_argument("--claim-probs", required=True,
                   help="claim_verify --claim-probs-out jsonl.")
    s.add_argument("--out", default="")

    g = sub.add_parser("gt-preds", help="oracle sheet inputs from GT (E2 B2).")
    g.add_argument("--gt", required=True)
    g.add_argument("--preds-out", required=True)
    g.add_argument("--scene-out", required=True)

    gs = sub.add_parser("gen-scene", help="scene mined from generated SD (E3 G->G).")
    gs.add_argument("--pred", required=True)
    gs.add_argument("--scene-out", required=True)

    o = sub.add_parser("override", help="override-vs-propagation rate (E4).")
    o.add_argument("--sheets", required=True, help="text_dossier --sheets-out dump.")
    o.add_argument("--override", required=True, help="the resulting override jsonl.")
    o.add_argument("--out", default="")

    c = sub.add_parser("to-candidates", help="preds/override jsonl -> candidates (E1a).")
    c.add_argument("--pred", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tasks", nargs="*", default=list(NARRATIVE_TYPES))

    t = sub.add_parser("tabulate", help="merge eval metric jsons -> markdown table.")
    t.add_argument("cols", nargs="+", help="Label=metrics.json ...")

    a = p.parse_args()
    out = None
    if a.cmd == "reliability":
        out = reliability(a.pred, a.gt, a.nbins, a.plot)
    elif a.cmd == "selfverify":
        out = selfverify(a.claim_probs)
    elif a.cmd == "gt-preds":
        gt_preds(a.gt, a.preds_out, a.scene_out)
    elif a.cmd == "gen-scene":
        gen_scene(a.pred, a.scene_out)
    elif a.cmd == "override":
        out = override_rate(a.sheets, a.override)
    elif a.cmd == "to-candidates":
        to_candidates(a.pred, a.out, tuple(a.tasks))
    elif a.cmd == "tabulate":
        tabulate(a.cols)
    if out is not None and getattr(a, "out", ""):
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[ablate-report] wrote {a.out}")


if __name__ == "__main__":
    main()
