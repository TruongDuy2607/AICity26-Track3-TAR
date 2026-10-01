#!/usr/bin/env bash
# PROVE Phase 0: base LoRA SFT from the foundation model, then merge the last checkpoint.
# The merged path is recorded in output/prove/BASE_MODEL_PATH for phase3_infer.sh.
#
# Usage:
#   bash scripts/prove/phase0_base_sft.sh
#   PROFILE=a100_80g_4x_32b NUM_EPOCHS=2 ... (any train-runai.sh knob overridable)
set -euo pipefail
# common.sh resolves + exports HERE (repo root) from its own location.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/common.sh"

# plain multi-task set, never the render mix
export RENDER_SFT=0
export PROFILE="${PROFILE:-a100_80g_4x_32b}"
# fixed run dir so the checkpoint can be found and merged below
export RUN_DIR="${RUN_DIR:-$HERE/output/prove_base_$(date +%Y%m%d_%H%M%S)}"

echo "[prove-p0] base SFT (S1 recipe) from the foundation model -> $RUN_DIR"
bash "$HERE/scripts/train-runai.sh"

# merge the last checkpoint
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
