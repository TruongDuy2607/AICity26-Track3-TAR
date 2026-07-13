#!/usr/bin/env bash

# Run:

# MODEL_PATH=output/prove_base_20260705_075519/v0-20260705-075536 \
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" GPU_MEM_UTIL=0.85 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/prove/k16_narrative.sh

# k16-5task (FINAL_SPRINT.md endgame step 1) — re-select the 5 non-TD narrative
# columns with a DOUBLED candidate pool + verification-weighted MBR: the exact
# mechanism pair behind the two confirmed jumps (official base +0.0053 = verify-MBR;
# td700 +0.0207 TD = bigger/uncapped pool). TD is NOT here — it is owned by the
# td700 override already banked at 0.4975.
#
#   1. text_dossier renders  TASKS=5 narrative, N_SAMPLES=16, LENGTH_TOL=0.8
#        -> preds/candidates.k16.jsonl (+ greedy preds/text_dossier.k16.jsonl)
#   2. claim_verify W1 probes over every candidate
#        -> preds/verify.k16.jsonl
#   3. mbr_select consensus + 0.3*verify
#        -> preds/text_override.k16.jsonl
#
# All outputs are .k16-suffixed — nothing from the scored official run is touched.
# Then, on the dev box, build the submission on the td700 base:
#
#   python -m track3.apply_override \
#       --base-csv submissions/32b-official/submission-td700.csv \
#       --override preds/32b-official/text_override.k16.jsonl \
#       --tasks bcq_openended causal_linkage scene_description open_qa video_summarization \
#       --out-csv submissions/32b-official/submission-k16.csv
#
# Usage (cluster):
#   MODEL_PATH=output/prove_base_20260705_075519/v0-20260705-075536 \
#   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" GPU_MEM_UTIL=0.85 \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/prove/k16_narrative.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/common.sh"
: "${MODEL_PATH:?Set MODEL_PATH=<merged 32B dir>}"
export MODEL_PATH

TASKS="bcq_openended causal_linkage scene_description open_qa video_summarization"
N_SAMPLES="${N_SAMPLES:-16}"
LENGTH_TOL="${LENGTH_TOL:-0.8}"
LAM="${LAM:-0.3}"
CANDS="$HERE/preds/candidates.k16.jsonl"
VERIFY="$HERE/preds/verify.k16.jsonl"
OVERRIDE="$HERE/preds/text_override.k16.jsonl"

NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")

echo "==================== [k16] 1/3 renders (K=$N_SAMPLES, tol=$LENGTH_TOL) ===================="
DIRECT=1 TASKS="$TASKS" LENGTH_TOL="$LENGTH_TOL" \
N_SAMPLES="$N_SAMPLES" SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}" \
PRED="$HERE/preds/test_pred.struct0.jsonl" \
SCENE_PROBES="$HERE/preds/scene_probes.jsonl" \
OUT="$HERE/preds/text_dossier.k16.jsonl" CANDIDATES_OUT="$CANDS" \
    bash "$HERE/scripts/text_dossier.sh"

echo "==================== [k16] 2/3 verify probes ===================="
python -m track3.claim_verify --model "$MODEL_PATH" \
    --videos-root "$TEST_VIDEOS_ROOT" \
    --candidates "$CANDS" --out "$VERIFY" \
    --probe-batch "${PROBE_BATCH:-256}" \
    --tensor-parallel-size "${TENSOR_PARALLEL:-$NGPU}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" --num-frames "$NUM_FRAMES"

echo "==================== [k16] 3/3 verify-weighted MBR (lam=$LAM) ===================="
python -m track3.mbr_select --candidates "$CANDS" \
    --verify "$VERIFY" --lam "$LAM" --out "$OVERRIDE"

echo "[k16] DONE -> $OVERRIDE  (copy candidates.k16 + text_override.k16 to the dev box)"
