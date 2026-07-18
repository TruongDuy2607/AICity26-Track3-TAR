#!/usr/bin/env bash
# E2 — Isolate the Cross-Question Evidence Sheet (paper §5.2, the core contribution).
#
# All conditions are direct-greedy renders that share the SAME template, length
# calibration and grader; the ONLY difference is the sheet content. Closed-task
# predictions are held at the base val_pred values, so they cancel in every delta.
#
#   B0-none    empty sheet (header + question only)        -> no-evidence control
#   B1-ours    full sheet from P_theta probes              -> ours
#   B2-oracle  sheet built from GROUND-TRUTH facts         -> pool ceiling (upper bound)
#   B3-swap    sheet borrowed from a DIFFERENT clip        -> wrong-but-plausible content
#   drop:FIELD leave-one-field-out (cause/scene/...)       -> which facts drive the gain
#
# B3 < B0 is the key generality result: a plausibly-formatted but wrong sheet HURTS,
# proving the model uses the fact CONTENT, not just the added scaffolding/length.
#
# Prereq: bash scripts/ablations/00_base_val.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# --- B1 (ours), B0 (none), B3 (swap) --------------------------------------------
run_sheet_condition "B1-ours" ""
run_sheet_condition "B0-none" "none"
run_sheet_condition "B3-swap" "swap"

# --- B2 (oracle): build the sheet from GT, then render with the SAME transform="" -
echo "==================== [E2] building oracle (GT) sheet inputs ===================="
python -m track3.ablate_report gt-preds --gt "$VAL_GT" \
    --preds-out "$ABL_DIR/gt_preds.jsonl" --scene-out "$ABL_DIR/gt_scene.jsonl"
PRED="$ABL_DIR/gt_preds.jsonl" SCENE_PROBES="$ABL_DIR/gt_scene.jsonl" \
    run_sheet_condition "B2-oracle" ""

# --- leave-one-field-out (E2b) --------------------------------------------------
for FIELD in ${DROP_FIELDS:-cause consequence scene cast observations}; do
    run_sheet_condition "drop-$FIELD" "drop:$FIELD"
done

# --- one table over every condition ---------------------------------------------
echo "==================== [E2] summary ===================="
COLS=("base-nochain=$ABL_DIR/base_nochain.metrics.json"
      "B0-none=$ABL_DIR/B0-none.metrics.json"
      "B3-swap=$ABL_DIR/B3-swap.metrics.json"
      "B1-ours=$ABL_DIR/B1-ours.metrics.json"
      "B2-oracle=$ABL_DIR/B2-oracle.metrics.json")
for FIELD in ${DROP_FIELDS:-cause consequence scene cast observations}; do
    COLS+=("drop-$FIELD=$ABL_DIR/drop-$FIELD.metrics.json")
done
python -m track3.ablate_report tabulate "${COLS[@]}" | tee "$ABL_DIR/e2_table.md"
echo ""
echo "[E2] DONE — table in $ABL_DIR/e2_table.md"
echo "  Read: B1 > B0 (sheet helps), B3 < B0 (wrong content hurts => model uses facts),"
echo "        B2 = ceiling (headroom of the evidence channel), drop-* = per-field gain."
