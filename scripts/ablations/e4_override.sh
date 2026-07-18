#!/usr/bin/env bash
# E4 — Override vs propagation of an injected wrong fact (paper §6, fills 'X/N').
#
# The sheet is consumed ZERO-SHOT, so "correct any conflicting fact" is a prompt,
# not a trained behaviour. We inject a KNOWN-WRONG value (borrowed from another clip,
# so register/length are preserved) into one field at a controlled rate, then measure
# how often the narrative OVERRIDES it (drops the wrong token) vs LEAKS it into the
# output. The clip-level leak count is the paper's residual-failure number.
#
# Prereq: bash scripts/ablations/00_base_val.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Which field to corrupt (agent identity is the Fig.2 story; also try scene / cause).
FIELDS="${FIELDS:-cast scene cause}"
RATE="${RATE:-1.0}"   # inject into every clip that has the field (max signal)

for FIELD in $FIELDS; do
    LABEL="corrupt-$FIELD"
    SHEETS="$ABL_DIR/${LABEL}.sheets.jsonl"
    echo "==================== [E4] inject wrong '$FIELD' (rate=$RATE) ===================="
    SHEETS_OUT="$SHEETS" \
        run_sheet_condition "$LABEL" "corrupt:${FIELD}:${RATE}"
    python -m track3.ablate_report override \
        --sheets "$SHEETS" --override "$ABL_DIR/${LABEL}.override.jsonl" \
        --out "$ABL_DIR/e4_${FIELD}.json"
done

echo ""
echo "[E4] DONE. Per-field override/propagation in $ABL_DIR/e4_*.json"
echo "  'clip_leak_rate' * 80  ~  the paper's X/80 residual-failure count."
echo "  Also compare the corrupt-* metrics to B1-ours: the score drop is the cost of a"
echo "  wrong sheet entry the zero-shot override fails to catch."
