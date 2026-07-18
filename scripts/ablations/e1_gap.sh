#!/usr/bin/env bash
# E1 — Quantify the verification-generation gap (paper §5.2, restores §sec:gap).
#
#   (a) VERIFIER side: closed-form P_theta is calibrated -> reliability diagram + ECE
#       + accuracy at each margin gate (from val_pred 'votes' vs val_gt).
#   (b) GENERATOR side: P_theta, asked to verify the generator's OWN narrative
#       sentences, endorses them far less -> mean self-verification P(Yes) and the
#       fraction of self-rejected claims. The two together ARE the gap: the same
#       weights judge a fact reliably (a) yet state it unreliably in prose (b).
#
# Prereq: bash scripts/ablations/00_base_val.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
read -r -a ENG_ARGS <<< "$ENG_ARGS_STR"
[ -f "$VAL_PRED" ] || { echo "run 00_base_val.sh first"; exit 1; }

# --- (a) closed-form reliability (no model call — reads val_pred votes) ----------
echo "==================== [E1a] closed-form reliability + ECE ===================="
python -m track3.ablate_report reliability --pred "$VAL_PRED" --gt "$VAL_GT" \
    --nbins "${NBINS:-10}" --plot "$ABL_DIR/reliability.png" \
    --out "$ABL_DIR/e1_reliability.json"

# --- (b) generator self-verification gap ----------------------------------------
# Probe the generator's OWN narrative claims. Source = the no-chain base narratives
# (B0). Set SOURCE=<override.jsonl> to instead probe a sheet-conditioned condition.
SOURCE="${SOURCE:-$VAL_PRED}"
CANDS="$ABL_DIR/e1_selfverify_cands.jsonl"
CLAIMS="$ABL_DIR/e1_claim_probs.jsonl"
echo "==================== [E1b] self-verification over generated narratives ===================="
python -m track3.ablate_report to-candidates --pred "$SOURCE" --out "$CANDS" \
    --tasks $NARR_TASKS
python -m track3.claim_verify --model "$MODEL_PATH" --videos-root "$VAL_VIDEOS" \
    --candidates "$CANDS" --out "$ABL_DIR/e1_verify.jsonl" \
    --claim-probs-out "$CLAIMS" --probe-batch "$PROBE_BATCH" "${ENG_ARGS[@]}"
python -m track3.ablate_report selfverify --claim-probs "$CLAIMS" \
    --out "$ABL_DIR/e1_selfverify.json"

echo ""
echo "[E1] DONE. For the paper's §5.2 gap paragraph:"
echo "  - verifier: closed-form accuracy + ECE in $ABL_DIR/e1_reliability.json (+ reliability.png)"
echo "  - generator: mean self-verify P(Yes) + self-reject rate in $ABL_DIR/e1_selfverify.json"
