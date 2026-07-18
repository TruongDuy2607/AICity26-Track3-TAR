# EDCR controlled ablations (`scripts/ablations/`)

Support experiments for the paper's **§5.2 "Why it works"** analysis. They show a
reviewer *why* EDCR works — not just the final leaderboard number — and make the
paper general by grounding every claim in a model property (the
verification–generation gap) rather than a leaderboard result.

The leaderboard is closed, so these run on a **held-out split of the training
videos** (`val_gt.json`, ~73 clips / ~870 items, produced by
`scripts/prepare_data.sh` with `VAL_RATIO=0.02`) scored by the **official grader**.

---

## Design invariant — consistency with the test setup

Every ablation runs the **same chain stages as `scripts/prove/phase3_infer.sh`**,
differing only in:

1. **the split** — `val_gt.json` instead of `test.json`, videos under
   `TRAIN_VIDEOS_ROOT` (val is held out of train), same official scorer; and
2. **exactly one toggled variable** — the sheet content (a `--sheet-transform`) or
   the selection rule (`--oracle-gt`).

Everything else — the merged base checkpoint, the render template
(`evidence.render_evidence_prompt`), length calibration, the candidate pool, MBR,
the grader — is **byte-identical to the shipped pipeline**. The whole point is that
a delta is then attributable to that one variable, not to a prompt/format change.
The single integration point is one hook in `text_dossier.run()`
(`track3/sheet_ablate.py`); nothing in the decode path is duplicated.

### ⚠️ How to report these numbers

`val` references are **machine-generated (training-style)**, unlike the
human-curated test set. So report **relative deltas and behavior**, never the
absolute BERTScore as a test estimate. Suggested paper sentence:

> *All analyses in this section run on a held-out split of the training videos with
> machine-generated references; we report them for relative deltas and mechanism
> behavior, not as estimates of the human-curated test score (Table 2).*

---

## Prerequisites

```bash
# 1) build the split (once). Creates data/processed/val_gt.json + val.jsonl.
bash scripts/prepare_data.sh

# 2) the base checkpoint (recorded by phase0). Auto-read from
#    output/prove/BASE_MODEL_PATH, or export MODEL_PATH=<merged dir>.
#    Multi-GPU: export CUDA_VISIBLE_DEVICES / PROFILE as for phase3.

# 3) shared base artifacts on val (idempotent; FORCE=1 to rebuild).
bash scripts/ablations/00_base_val.sh
```

`00_base_val.sh` produces, under `output/ablations/`:

| file | what | used by |
|---|---|---|
| `val_pred.jsonl` | base infer (logprob closed scoring + MCQ debias); the `votes` dists + the no-chain narrative baseline | E1, E2 baseline |
| `val_pred.struct0.jsonl` | structural pairing (paired bcq_oe observations the sheet quotes) | sheet source |
| `scene_probes.jsonl` | `P_theta` scene attributes | sheet source |

Then run any `eN_*.sh`, or `bash scripts/ablations/run_all.sh` for all of them.

---

## The experiments

### E1 — Quantify the verification–generation gap  (`e1_gap.sh`)
Fills the paper's **commented-out `§sec:gap`** — the central premise currently has
no measurement.
- **Verifier side:** `reliability.png` + `e1_reliability.json` — closed-form
  `P_theta` accuracy, **ECE**, and accuracy at each margin gate.
- **Generator side:** `e1_selfverify.json` — `P_theta` asked to verify the
  generator's **own** narrative sentences: mean endorsement `P(Yes)` and the
  fraction of **self-rejected** claims. High closed-form accuracy + low
  self-endorsement = the gap, operationalized.

```bash
bash scripts/ablations/e1_gap.sh
# probe a sheet-conditioned condition instead of the base narratives:
SOURCE=output/ablations/B1-ours.override.jsonl bash scripts/ablations/e1_gap.sh
```

### E2 — Isolate the Cross-Question Evidence Sheet  (`e2_sheet_isolation.sh`)
The paper's **core contribution**, currently confounded in the *cumulative* Table 2.
Clean A/B holding everything but the sheet content fixed → `e2_table.md`:

| condition | sheet | reads as |
|---|---|---|
| `base-nochain` | — (raw model) | honest no-chain baseline |
| `B0-none` | empty (header + question) | no-evidence control |
| `B3-swap` | a **different clip's** sheet | wrong-but-plausible content |
| `B1-ours` | `P_theta` probes | ours |
| `B2-oracle` | **ground-truth** facts | pool ceiling (upper bound) |
| `drop-<field>` | leave-one-field-out | which facts drive the gain |

Key results to quote: **B1 > B0** (the sheet helps), **B3 < B0** (a
plausibly-formatted *wrong* sheet *hurts* → the model uses fact **content**, not
scaffolding — the generality argument), **B2** = headroom, **drop-*** = per-field
attribution.

```bash
bash scripts/ablations/e2_sheet_isolation.sh
DROP_FIELDS="cause scene" bash scripts/ablations/e2_sheet_isolation.sh   # subset
```

### E3 — Direction of information flow  (`e3_flow_direction.sh`)
Defends the thesis *"every mechanism moves information from the verifier to the
generator, and none the other way."* Same sheet fields, different **source
channel**: `P->G` (calibrated first-token readout) vs `G->G` (free-generation:
`--closed-scoring vote` decisions + scene mined from generated text). Expect
**P->G > G->G** → the calibrated channel, not mere conditioning, is load-bearing.
→ `e3_table.md`.

### E4 — Override vs propagation of a wrong fact  (`e4_override.sh`)
Fills the paper's **`X/80` placeholder (§6)** and quantifies Fig. 2. Injects a
known-wrong value (borrowed from another clip, so register/length are preserved)
into one field, then measures how often the narrative **overrides** it vs **leaks**
it. `clip_leak_rate * 80 ≈` the residual-failure count. → `e4_*.json`.

```bash
bash scripts/ablations/e4_override.sh
FIELDS="cast" RATE=1.0 bash scripts/ablations/e4_override.sh   # agent-identity only
```

### E5 — Selection value + pool ceiling  (`e5_mbr_oracle.sh`)
Explains Stage 3's small **+0.0092**. Over one K-candidate pool: `greedy` vs `MBR`
(shipped) vs `oracle` (argmax true BERTScore vs GT — analysis-only). The
`oracle − greedy` gap is pool quality; `MBR − greedy` is what selection recovers of
it. The `[mbr] greedy kept on N%` line is the overturn rate. → `e5_table.md`.

---

## Mapping to the paper (why these are load-bearing)

| Experiment | Paper hole it fills | Claim it turns from assertion → measurement |
|---|---|---|
| **E1** | `§sec:gap` (commented out) | `P_theta` calibrated vs `G_theta` unreliable — the whole premise |
| **E2 B1>B0** | Stage-2 gain confounded in cumulative Table 2 | the sheet's *clean* contribution |
| **E2 B3<B0** | (new) generality | the model uses fact **content**, not prompt scaffolding |
| **E2 drop-*** | leave-one-out table (cut from the draft) | per-field attribution |
| **E3** | thesis "none the other way" | the routing **direction** matters |
| **E4** | `X/80` in §6 + Fig. 2 caption | override is a prompt, not a guarantee — the real leak rate |
| **E5** | Stage 3 "+0.0092" + negative result (ii) | pool ceiling vs same-model selection limit |

Recommended paper edit: add **§5.2 "Why it works: a controlled analysis on a
held-out split"** before the cumulative Table 2, containing E1 (reliability figure +
gap table) and E2 (isolation + leave-one-out). This reframes the paper around a
mechanism grounded in a model property — general beyond this benchmark — and cleanly
separates the two benchmark-specific stages (1, 5) from the transferable ones.

## Outputs

Everything lands in `output/ablations/` (kept out of `preds/`/`submissions/` so it
can never pollute a real submission): `*.metrics.json` per condition, `e{2,3,5}_table.md`,
`e{1,4}_*.json`, and `reliability.png`.

## New/changed code

- `track3/sheet_ablate.py` — pure sheet transforms (the only new mechanism).
- `track3/ablate_report.py` — CPU analysis/aggregation (reliability, selfverify,
  gt-preds, gen-scene, to-candidates, override, tabulate).
- hooks (small, isolated): `text_dossier.py` (`--sheet-transform/--sheets-out`),
  `mbr_select.py` (`--oracle-gt`), `claim_verify.py` (`--claim-probs-out`),
  `evidence.py` (empty event line skip), `scripts/text_dossier.sh` (passthrough).
