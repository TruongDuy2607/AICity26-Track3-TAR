#!/usr/bin/env bash
# Steps 6-7 of the runbook (method.md §8): structural post-processing -> official
# submission CSV -> format validation against the released (redacted) test.json.
#
# Usage:
#   bash scripts/postprocess.sh                          # all rules (Sub C shape)
#   RULES="--temporal-prior" bash scripts/postprocess.sh # Sub A shape (one lever)
#   PRED_OUT=preds/val_pred.jsonl GT_JSON=data/processed/val_gt.json \
#       bash scripts/postprocess.sh                      # Step-4 local gate
#
# Env knobs:
#   PRED_OUT    input predictions jsonl (default preds/test_pred.jsonl)
#   STRUCT_OUT  revised jsonl            (default ${PRED_OUT%.jsonl}.struct.jsonl)
#   SUBMIT_CSV  submission CSV           (default $SUBMIT_DIR/submission.csv)
#   GT_JSON     GT for validate/score    (default $TEST_JSON — validation only;
#               point it at val_gt.json to get real scores on the held-out split)
#   RULES       structural rule flags    (default --all)
#   TEMPORAL_OVERRIDE  jsonl (item_index,start,end) feeding rule 4.1 a per-item
#               temporal window from an external method (track3.temporal_grounding).
#   TEXT_OVERRIDE      jsonl (item_index,prediction) replacing open-ended text
#               answers from an external method (track3.text_dossier). Both empty
#               by default = the plain model output / td-window leak.
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
# 1) structural rules (method.md §4) — logs how many predictions each rule changed
# shellcheck disable=SC2086  # RULES is an intentional word-split flag list
python -m track3.structural \
    --test-json "$GT_JSON" \
    --pred "$PRED_OUT" \
    --out "$STRUCT_OUT" \
    $RULES "${OVERRIDE_ARG[@]}"

# 2) submission CSV through the format choke-point (row order = GT item order)
python -m track3.make_submission \
    --pred "$STRUCT_OUT" \
    --test-json "$GT_JSON" \
    --out "$SUBMIT_CSV"

# 3) official validator: must report every row parseable before upload.
#    Scores automatically when GT_JSON carries real answers (val_gt.json).
#    Single canonical scorer = track3/evaluate.py (run from repo root via -m).
python -m track3.evaluate --gt "$GT_JSON" --submission "$SUBMIT_CSV"

echo "[postprocess] rules='$RULES'"
echo "[postprocess] revised preds: $STRUCT_OUT"
echo "[postprocess] submission:    $SUBMIT_CSV"
