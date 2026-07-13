#!/usr/bin/env bash
# Fetch the TAR dataset: annotations (HF) + test clips (YouTube via official script).
# Train *videos* (~150 GB) come from 8 upstream sources; see the HF README for the
# per-source download instructions and place them under $VIDEOS_ROOT/<source>/...
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/configs/common.sh"

REPO="nvidia/PhysicalAI-Traffic-Anomaly-Reasoning"
mkdir -p "$TAR_ROOT"   # = $TRACK3_ROOT/data by default (see configs/common.sh)

echo "[download] annotations + test manifest from $REPO ..."
huggingface-cli download "$REPO" --repo-type dataset --local-dir "$TAR_ROOT" \
    --include "train/*" "test/*"

echo "[download] test clips via the official download_test_videos.py ..."
if [ -f "$TAR_ROOT/test/download_test_videos.py" ]; then
    ( cd "$TAR_ROOT/test" && python download_test_videos.py )
fi

cat <<EOF

[download] done.
  annotations : $TAR_ROOT/train/<task>.json   -> set ANN_DIR=$TAR_ROOT/train
  test.json   : $TAR_ROOT/test/test.json      -> set TEST_JSON
  NEXT: download the 8 train video sources into \$VIDEOS_ROOT (see HF README),
        then run: bash scripts/prepare_data.sh
EOF
