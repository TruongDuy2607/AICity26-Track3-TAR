#!/usr/bin/env bash
# k16 verify + MBR (steps 2-3 of k16_narrative.sh, for when the renders already
# exist in preds/candidates.k16.jsonl and the run stopped before verify/MBR).
#
#   1. filter the pool to TASKS            -> preds/candidates.k16.<tag>.jsonl
#   2. claim_verify W1 probes (GPU)        -> preds/verify.k16.jsonl
#   3. mbr_select consensus + LAM*verify   -> preds/text_override.k16.jsonl
#
# Default TASKS = bcq_openended only — the one column with a measured mechanism
# (13% of the pool at the old 126-ch cap); the other 4 narrative tasks are re-roll
# noise and OQA/SUM lead the board. Widen with TASKS="..." if wanted.
#
# Usage (cluster):
#   MODEL_PATH=output/prove_base_20260705_075519/v0-20260705-075536 \
#   GPU_MEM_UTIL=0.85 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/prove/k16_verify.sh
#
# Then on the dev box (submission on the td700 base):
#   python -m track3.apply_override \
#       --base-csv submissions/32b-official/submission-td700.csv \
#       --override preds/32b-official/text_override.k16.jsonl \
#       --tasks bcq_openended \
#       --out-csv submissions/32b-official/submission-k16.csv
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/common.sh"
: "${MODEL_PATH:?Set MODEL_PATH=<merged 32B dir>}"
export MODEL_PATH

TASKS="${TASKS:-bcq_openended}"
LAM="${LAM:-0.3}"
CANDS_ALL="$HERE/preds/candidates.k16.jsonl"
CANDS="$HERE/preds/candidates.k16.filtered.jsonl"
VERIFY="$HERE/preds/verify.k16.jsonl"
OVERRIDE="$HERE/preds/text_override.k16.jsonl"
[ -f "$CANDS_ALL" ] || { echo "ERROR: $CANDS_ALL missing (run k16_narrative.sh first)" >&2; exit 1; }

NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")

echo "==================== [k16v] 1/3 filter pool to: $TASKS ===================="
TASKS="$TASKS" CANDS_ALL="$CANDS_ALL" CANDS="$CANDS" python - <<'PY'
import json, os
tasks = set(os.environ["TASKS"].split())
kept = 0
with open(os.environ["CANDS"], "w", encoding="utf-8") as out:
    for line in open(os.environ["CANDS_ALL"], encoding="utf-8"):
        if json.loads(line).get("task") in tasks:
            out.write(line)
            kept += 1
print(f"[k16v] kept {kept} item(s) for {sorted(tasks)}")
PY

echo "==================== [k16v] 2/3 verify probes ===================="
python -m track3.claim_verify --model "$MODEL_PATH" \
    --videos-root "$TEST_VIDEOS_ROOT" \
    --candidates "$CANDS" --out "$VERIFY" \
    --probe-batch "${PROBE_BATCH:-256}" \
    --tensor-parallel-size "${TENSOR_PARALLEL:-$NGPU}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" --num-frames "$NUM_FRAMES"

echo "==================== [k16v] 3/3 verify-weighted MBR (lam=$LAM) ===================="
python -m track3.mbr_select --candidates "$CANDS" \
    --verify "$VERIFY" --lam "$LAM" --out "$OVERRIDE"

echo "[k16v] DONE -> $OVERRIDE  (copy text_override.k16.jsonl to the dev box)"
