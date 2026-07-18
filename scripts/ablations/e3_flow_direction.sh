#!/usr/bin/env bash
# E3 — Direction of information flow: P->G vs G->G (paper thesis "none the other way").
#
# Both conditions render the SAME sheet fields through the SAME template; the ONLY
# difference is the CHANNEL the cross-question facts are sourced from:
#
#   P->G (ours)  facts from the CALIBRATED first-token readout P_theta
#                (logprob-pooled MCQ letter + P_theta scene probes)   = B1-ours
#   G->G         facts from the FREE-GENERATION channel
#                (self-consistency 'vote' closed decisions + scene mined from the
#                 model's own generated scene_description)
#
# P->G > G->G shows the calibrated channel is what makes the sheet work, not any
# conditioning — the direction of the routing is load-bearing.
#
# Prereq: bash scripts/ablations/00_base_val.sh  (and B1-ours from e2, or it is
#         recomputed here).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# P->G (ours): reuse B1 if present, else compute it.
if [ ! -f "$ABL_DIR/B1-ours.metrics.json" ]; then
    run_sheet_condition "B1-ours" ""
fi

# --- build the free-generation (G->G) sheet sources -----------------------------
VAL_PRED_VOTE="$ABL_DIR/val_pred_vote.jsonl"
VAL_STRUCT0_VOTE="$ABL_DIR/val_pred_vote.struct0.jsonl"
SCENE_GEN="$ABL_DIR/scene_gen.jsonl"

if [ "${FORCE:-0}" = "1" ] || [ ! -f "$VAL_PRED_VOTE" ]; then
    echo "==================== [E3] infer val with free-generation voting ===================="
    CLOSED_SCORING=vote VOTE_N="${VOTE_N:-5}" PRED_OUT="$VAL_PRED_VOTE" \
    MODEL_PATH="$MODEL_PATH" TEST_JSON="$VAL_GT" VIDEOS_ROOT="$VAL_VIDEOS" \
        bash "$HERE/scripts/infer.sh"
    python -m track3.structural --test-json "$VAL_GT" \
        --pred "$VAL_PRED_VOTE" --out "$VAL_STRUCT0_VOTE" --all
fi
# scene mined from the model's OWN generated scene_description (generation channel).
python -m track3.ablate_report gen-scene --pred "$VAL_PRED" --scene-out "$SCENE_GEN"

# --- render G->G (facts from the generation channel) ----------------------------
PRED="$VAL_STRUCT0_VOTE" SCENE_PROBES="$SCENE_GEN" \
    run_sheet_condition "G2G-flow" ""

# --- compare --------------------------------------------------------------------
echo "==================== [E3] summary ===================="
python -m track3.ablate_report tabulate \
    "base-nochain=$ABL_DIR/base_nochain.metrics.json" \
    "G2G(gen-channel)=$ABL_DIR/G2G-flow.metrics.json" \
    "P2G(ours)=$ABL_DIR/B1-ours.metrics.json" | tee "$ABL_DIR/e3_table.md"
echo ""
echo "[E3] DONE — expect P2G > G2G (the calibrated channel, not mere conditioning, drives the gain)."
