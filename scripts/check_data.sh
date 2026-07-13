#!/usr/bin/env bash
# Validate the processed train/val jsonl before training.
# Usage: bash scripts/check_data.sh            # full check (incl. media on disk)
#        NO_MEDIA=1 bash scripts/check_data.sh  # schema-only (no file existence)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

cd "$HERE"
EXTRA=()
[ "${NO_MEDIA:-0}" = "1" ] && EXTRA+=(--no-media)
python -m track3.check_dataset --data-dir "$DATA_DIR" "${EXTRA[@]}" "$@"
