#!/usr/bin/env bash
# E5 — Selection value + pool ceiling (paper Stage 3, +0.0092 small gain explained).
#
# Over the SAME K-candidate pool (same knobs as phase3: K=8, temp=0.8, lam=0.3):
#   greedy   candidates[0] (Stage 2 render)                 -> what MBR starts from
#   MBR      consensus + lam*verify selection (Stage 3)     -> the shipped selection
#   oracle   argmax true BERTScore vs GT (analysis-only)    -> the pool CEILING
#
# greedy < MBR <= oracle separates pool quality (oracle - greedy: how good the best
# candidate is) from selection quality (MBR - greedy: how much MBR recovers of it).
# A small MBR gain with a large oracle gap means the ceiling is real but same-model
# selection cannot reach it — which is exactly why Stage 3 is only +0.0092.
#
# Prereq: bash scripts/ablations/00_base_val.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
read -r -a ENG_ARGS <<< "$ENG_ARGS_STR"

K="${K:-8}"; LAM="${LAM:-0.3}"; SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}"
GREEDY="$ABL_DIR/e5_greedy.jsonl"
CANDS="$ABL_DIR/e5_candidates.jsonl"
VERIFY="$ABL_DIR/e5_verify.jsonl"
MBR="$ABL_DIR/e5_mbr.jsonl"
ORACLE="$ABL_DIR/e5_oracle.jsonl"

# 1) build the K-candidate pool (greedy override + K-1 jittered samples).
echo "==================== [E5] 1/4 candidate pool (K=$K) ===================="
DIRECT=1 TASKS="$NARR_TASKS" \
MODEL_PATH="$MODEL_PATH" TEST_JSON="$VAL_GT" GROUND_VIDEOS_ROOT="$VAL_VIDEOS" \
PRED="$VAL_STRUCT0" SCENE_PROBES="$VAL_SCENE" \
N_SAMPLES="$K" SAMPLE_TEMPERATURE="$SAMPLE_TEMPERATURE" \
OUT="$GREEDY" CANDIDATES_OUT="$CANDS" GEN_BATCH="$GEN_BATCH" \
    bash "$HERE/scripts/text_dossier.sh"

# 2) claim-level verification of every candidate.
echo "==================== [E5] 2/4 claim verify ===================="
python -m track3.claim_verify --model "$MODEL_PATH" --videos-root "$VAL_VIDEOS" \
    --candidates "$CANDS" --out "$VERIFY" --probe-batch "$PROBE_BATCH" "${ENG_ARGS[@]}"

# 3) MBR (shipped) + oracle (ceiling) selection over the same pool.
echo "==================== [E5] 3/4 MBR + oracle selection ===================="
python -m track3.mbr_select --candidates "$CANDS" --verify "$VERIFY" --lam "$LAM" --out "$MBR"
python -m track3.mbr_select --candidates "$CANDS" --oracle-gt "$VAL_GT" --out "$ORACLE"

# 4) score greedy / MBR / oracle overlaid on the base val_pred.
echo "==================== [E5] 4/4 score ===================="
eval_override "$GREEDY" "$ABL_DIR/e5_greedy.metrics.json"
eval_override "$MBR"    "$ABL_DIR/e5_mbr.metrics.json"
eval_override "$ORACLE" "$ABL_DIR/e5_oracle.metrics.json"
python -m track3.ablate_report tabulate \
    "greedy=$ABL_DIR/e5_greedy.metrics.json" \
    "MBR(ours)=$ABL_DIR/e5_mbr.metrics.json" \
    "oracle(ceiling)=$ABL_DIR/e5_oracle.metrics.json" | tee "$ABL_DIR/e5_table.md"
echo ""
echo "[E5] DONE — the '[mbr] greedy kept on N%' line above is the MBR-vs-greedy overturn"
echo "     rate; the oracle column is the pool ceiling."
