#!/usr/bin/env bash
# Shared prerequisite for every ablation: produce the base artifacts on the val
# split, using the SAME stages as phase3_infer.sh stages 1-3.
#
#   val_pred.jsonl        base infer (W1 logprob closed scoring + MCQ permute) — the
#                         'votes' first-token distributions drive E1b reliability, and
#                         its narrative predictions ARE the no-chain baseline (B0-raw).
#   val_pred.struct0.jsonl structural --all (paired bcq_oe observations the sheet quotes).
#   scene_probes.jsonl    claim_verify --scene-out (P_theta scene attributes for the sheet).
#
# Run once, then run any eN_*.sh. Re-run with FORCE=1 to rebuild.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
FORCE="${FORCE:-0}"
read -r -a ENG_ARGS <<< "$ENG_ARGS_STR"

# 1) base inference on val (logprob closed scoring + MCQ permutation debias).
if [ "$FORCE" = "1" ] || [ ! -f "$VAL_PRED" ]; then
    echo "==================== [ablate] 1/3 base infer on val ===================="
    MCQ_PERMUTE="${MCQ_PERMUTE:-4}" PRED_OUT="$VAL_PRED" \
    MODEL_PATH="$MODEL_PATH" TEST_JSON="$VAL_GT" VIDEOS_ROOT="$VAL_VIDEOS" \
        bash "$HERE/scripts/infer.sh"
else
    echo "[ablate] reuse $VAL_PRED (FORCE=1 to rebuild)"
fi

# 2) structural pre-pass (bcq pairing -> the observations the evidence sheet quotes).
if [ "$FORCE" = "1" ] || [ ! -f "$VAL_STRUCT0" ]; then
    echo "==================== [ablate] 2/3 structural pre-pass ===================="
    python -m track3.structural --test-json "$VAL_GT" \
        --pred "$VAL_PRED" --out "$VAL_STRUCT0" --all
fi

# 3) scene-attribute probes (P_theta over the closed scene vocabulary).
if [ "$FORCE" = "1" ] || [ ! -f "$VAL_SCENE" ]; then
    echo "==================== [ablate] 3/3 scene probes ===================="
    python -m track3.claim_verify --model "$MODEL_PATH" \
        --videos-root "$VAL_VIDEOS" --test-json "$VAL_GT" \
        --scene-out "$VAL_SCENE" --scene-min-p "${SCENE_MIN_P:-0.5}" \
        --probe-batch "$PROBE_BATCH" "${ENG_ARGS[@]}"
fi

# baseline score (no chain) for reference — the honest B0-raw narrative number.
echo "==================== [ablate] base (no-chain) score ===================="
ablate_eval "$VAL_PRED" "$ABL_DIR/base_nochain.metrics.json"
echo "[ablate] base artifacts ready in $ABL_DIR"
