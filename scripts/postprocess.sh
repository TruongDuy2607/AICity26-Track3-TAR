#!/usr/bin/env bash
# Structural post-processing -> submission CSV -> official validation (or scoring).
#
# Usage:
#   bash scripts/postprocess.sh                          # all rules
#   RULES="--temporal-prior" bash scripts/postprocess.sh # one rule
#   PRED_OUT=preds/val_pred.jsonl GT_JSON=data/processed/val_gt.json \
#       bash scripts/postprocess.sh                      # local scoring
#
# Env: PRED_OUT, STRUCT_OUT, SUBMIT_CSV, RULES (default --all),
#      GT_JSON (default $TEST_JSON = validation only; val_gt.json = real scores),
#      TEMPORAL_OVERRIDE / TEXT_OVERRIDE (optional override jsonl).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

PRED_OUT="${PRED_OUT:-$HERE/preds/test_pred.jsonl}"
STRUCT_OUT="${STRUCT_OUT:-${PRED_OUT%.jsonl}.struct.jsonl}"
SUBMIT_CSV="${SUBMIT_CSV:-$SUBMIT_DIR/submission.csv}"
GT_JSON="${GT_JSON:-$TEST_JSON}"
RULES="${RULES:---all}"
OVERRIDE_ARG=()
[ -n "${TEMPORAL_OVERRIDE:-}" ] && OVERRIDE_ARG+=(--temporal-override "$TEMPORAL_OVERRIDE")
[ -n "${TEXT_OVERRIDE:-}" ] && OVERRIDE_ARG+=(--text-override "$TEXT_OVERRIDE")

cd "$HERE"
# 1) structural rules
# shellcheck disable=SC2086  # RULES is an intentional word-split flag list
python -m track3.structural \
    --test-json "$GT_JSON" \
    --pred "$PRED_OUT" \
    --out "$STRUCT_OUT" \
    $RULES "${OVERRIDE_ARG[@]}"

# 2) submission CSV (row order = GT item order)
python -m track3.make_submission \
    --pred "$STRUCT_OUT" \
    --test-json "$GT_JSON" \
    --out "$SUBMIT_CSV"

# 3) official validator (scores when GT_JSON has real answers)
python -m track3.evaluate --gt "$GT_JSON" --submission "$SUBMIT_CSV"

echo "[postprocess] rules='$RULES'"
echo "[postprocess] revised preds: $STRUCT_OUT"
echo "[postprocess] submission:    $SUBMIT_CSV"
