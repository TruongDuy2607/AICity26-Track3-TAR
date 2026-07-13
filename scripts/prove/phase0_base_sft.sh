#!/usr/bin/env bash
# PROVE Phase 0 — Base SFT from scratch (only when no previous checkpoint exists).
# Produces the merged base checkpoint that phase3_infer.sh runs the chain on:
#   1. prepare the base multi-task SFT set  (data/processed/{train,val}.jsonl —
#      train-runai.sh auto-runs prepare_data.sh inside the conda env when absent;
#      FORCE_PREPARE=1 rebuilds it)
#   2. LoRA SFT from the foundation model   (the recipe in
#      configs/qwen3vl_32b_lora.sh: Qwen3-VL-32B-Instruct, r16/α32 all-linear,
#      ViT frozen, 2 epochs, lr 2e-4, 32 frames)
#   3. merge the last checkpoint            (vLLM cannot apply Qwen3-VL ViT-LoRA)
#
# The merged path is written to output/prove/BASE_MODEL_PATH so phase3_infer.sh
# picks it up without copy-pasting paths.
#
# Usage:
#   bash scripts/prove/phase0_base_sft.sh
#   PROFILE=a100_80g_4x_32b NUM_EPOCHS=2 ... (any train-runai.sh knob overridable)
set -euo pipefail
# common.sh resolves + exports HERE (repo root) from its own location.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/common.sh"

# Base SFT trains on the plain multi-task set — never the render mix.
export RENDER_SFT=0
# PROFILE owns CUDA_VISIBLE_DEVICES/DDP knobs; train-runai.sh sources it.
export PROFILE="${PROFILE:-a100_80g_4x_32b}"
# Deterministic run dir so this script can find + merge the checkpoint afterwards.
export RUN_DIR="${RUN_DIR:-$HERE/output/prove_base_$(date +%Y%m%d_%H%M%S)}"

echo "[prove-p0] base SFT (S1 recipe) from the foundation model -> $RUN_DIR"
bash "$HERE/scripts/train-runai.sh"

# --- auto-merge the last checkpoint ---------------------------------------------
CKPT="$(find "$RUN_DIR" -maxdepth 3 -type d -name 'checkpoint-*' | sort -V | tail -1)"
[ -n "$CKPT" ] || { echo "[prove-p0] ERROR: no checkpoint under $RUN_DIR" >&2; exit 1; }
MERGED_DIR="${MERGED_DIR:-${CKPT%/}-merged}"
if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "[prove-p0] merging LoRA: $CKPT -> $MERGED_DIR ..."
    swift export --adapters "$CKPT" --merge_lora true --output_dir "$MERGED_DIR"
fi

STATE_DIR="$HERE/output/prove"
mkdir -p "$STATE_DIR"
echo "$MERGED_DIR" > "$STATE_DIR/BASE_MODEL_PATH"

echo ""
echo "[prove-p0] done. Base checkpoint (recorded in $STATE_DIR/BASE_MODEL_PATH):"
echo "    MODEL_PATH=$MERGED_DIR bash scripts/prove/phase3_infer.sh"
