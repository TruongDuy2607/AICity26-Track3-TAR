#!/usr/bin/env bash
# Build the official submission.csv from raw predictions.
# Usage: bash scripts/submit.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

PRED_OUT="${PRED_OUT:-$HERE/preds/test_pred.jsonl}"
SUBMIT_CSV="${SUBMIT_CSV:-$SUBMIT_DIR/submission.csv}"

cd "$HERE"
python -m track3.make_submission \
    --pred "$PRED_OUT" \
    --test-json "$TEST_JSON" \
    --out "$SUBMIT_CSV"

echo "[submit] wrote $SUBMIT_CSV"
