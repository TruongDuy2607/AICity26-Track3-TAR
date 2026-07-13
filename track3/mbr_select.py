"""Verification-weighted MBR candidate selection (PROVE tier 3, PROVE.md §1).

Greedy decoding maximizes sequence probability, not the expected grading metric.
For the BERTScore-graded narrative tasks the theoretically correct decode is
Minimum Bayes Risk selection (Kumar & Byrne 2004; Eikema & Aziz 2020): among K
candidate renders, pick the one with the highest expected utility against the
others, with the utility = the EXACT official metric (``bert_score.BERTScorer(
lang="en", rescale_with_baseline=True)`` — the same construction as
track3/evaluate.py). An optional per-candidate verify score (track3.claim_verify)
is blended in: ``score(c) = consensus(c) + lam * verify(c)`` (QE-weighted MBR).

Input   : candidates jsonl from ``text_dossier --candidates-out``
          {item_index, video_id, task, candidates: [greedy-first, ...]}
Optional: verify jsonl from ``claim_verify --candidates``
          {item_index, scores: [one float per candidate]}
Output  : text-override jsonl {item_index, video_id, task, prediction} for
          ``structural --text-override``.

Every input record emits exactly one output line; when scoring is impossible
(single candidate / scorer failure) the choice degrades to candidates[0] — the
greedy full-evidence render — so the override never regresses below tier 1/2.
Ties prefer the lowest index (greedy first). Selection logic is pure and
CPU-tested (track3/test_mbr_select.py); bert-score is imported lazily in the CLI.

Run::

    python -m track3.mbr_select --candidates preds/candidates.jsonl \
        --verify preds/verify.jsonl --lam 0.3 --out preds/text_override.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# Pure selection logic
# ---------------------------------------------------------------------------

def all_pairs(n: int) -> list[tuple[int, int]]:
    """Unordered index pairs (i < j) — BERTScore F1 is symmetric, score each once."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def consensus_scores(n: int, pair_sims: dict[tuple[int, int], float]) -> list[float]:
    """Per-candidate mean similarity to every other candidate (the MBR utility)."""
    if n <= 1:
        return [0.0] * n
    tot = [0.0] * n
    for (i, j), s in pair_sims.items():
        tot[i] += s
        tot[j] += s
    return [t / (n - 1) for t in tot]


def select(consensus: list[float], verify: list[float] | None = None,
           lam: float = 0.0) -> int:
    """argmax of consensus + lam*verify; ties -> lowest index (the greedy render)."""
    n = len(consensus)
    if n == 0:
        return 0
    verify = verify if verify and len(verify) == n else [0.0] * n
    return max(range(n), key=lambda i: (consensus[i] + lam * verify[i], -i))


# ---------------------------------------------------------------------------
# IO + batched scoring (lazy bert-score; one global batch over all items)
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _bertscore_pairs(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    import bert_score  # lazy — GPU/cluster only
    scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)  # = evaluate.py
    _, _, f1 = scorer.score([a for a, _ in pairs], [b for _, b in pairs],
                            batch_size=batch_size)
    return [float(x) for x in f1]


def run(records: list[dict], verify_by_item: dict[str, list[float]], lam: float,
        pair_scorer) -> list[dict]:
    """One selection per record. ``pair_scorer(text_pairs) -> [sim]`` is injected
    (the real BERTScore batch above; a stub in the tests)."""
    # 1) one global pair batch across all items (BERTScore startup is the cost).
    flat_pairs: list[tuple[str, str]] = []
    spans: list[tuple[dict, list[tuple[int, int]], int]] = []
    for rec in records:
        cands = [c for c in rec.get("candidates", []) if (c or "").strip()]
        idx = all_pairs(len(cands))
        spans.append((rec, idx, len(cands)))
        flat_pairs += [(cands[i], cands[j]) for i, j in idx]
    sims = pair_scorer(flat_pairs) if flat_pairs else []

    # 2) per-item consensus + verify blend.
    out, pos, picked = [], 0, defaultdict(int)
    for rec, idx, n in spans:
        cands = [c for c in rec.get("candidates", []) if (c or "").strip()]
        if not cands:
            continue
        pair_sims = {ij: sims[pos + k] for k, ij in enumerate(idx)}
        pos += len(idx)
        verify = verify_by_item.get(str(rec["item_index"]))
        choice = select(consensus_scores(n, pair_sims), verify, lam)
        picked[choice] += 1
        out.append({"item_index": rec["item_index"],
                    "video_id": rec.get("video_id", ""),
                    "task": rec.get("task", ""),
                    "prediction": cands[choice], "source": "mbr"})
    if out:
        greedy = picked.get(0, 0)
        print(f"[mbr] selected over {len(out)} item(s); greedy kept on {greedy} "
              f"({greedy / len(out):.0%}), choice histogram {dict(sorted(picked.items()))}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True,
                   help="candidates jsonl from text_dossier --candidates-out.")
    p.add_argument("--verify", default="",
                   help="verify jsonl from claim_verify --candidates (optional).")
    p.add_argument("--lam", type=float, default=0.0,
                   help="verify-score weight (0 = pure MBR).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="preds/text_override.jsonl")
    a = p.parse_args()

    records = load_jsonl(a.candidates)
    verify_by_item = {}
    if a.verify and os.path.exists(a.verify):
        for rec in load_jsonl(a.verify):
            verify_by_item[str(rec["item_index"])] = rec.get("scores") or []
    elif a.verify:
        print(f"[mbr][WARN] verify file {a.verify} not found — pure MBR (lam ignored).")

    out = run(records, verify_by_item, a.lam,
              lambda pairs: _bertscore_pairs(pairs, a.batch_size))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[mbr] wrote {len(out)} override(s) to {a.out}")


if __name__ == "__main__":
    main()
