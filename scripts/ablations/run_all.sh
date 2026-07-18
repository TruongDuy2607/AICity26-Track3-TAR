#!/usr/bin/env bash
# Run the whole §5.2 controlled-analysis suite end to end on the held-out val split.
# Each stage is independent; comment out any you do not need. GPU-heavy stages reuse
# the base artifacts from 00_base_val.sh, so run that first (it is idempotent).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DIR/00_base_val.sh"
bash "$DIR/e1_gap.sh"
bash "$DIR/e2_sheet_isolation.sh"
bash "$DIR/e3_flow_direction.sh"
bash "$DIR/e4_override.sh"
bash "$DIR/e5_mbr_oracle.sh"

echo ""
echo "==================== ALL ABLATIONS DONE ===================="
echo "Tables + metrics under output/ablations/. Paper-ready summaries:"
echo "  E1  e1_reliability.json + reliability.png + e1_selfverify.json"
echo "  E2  e2_table.md   (B0/B1/B2/B3 + leave-one-out)"
echo "  E3  e3_table.md   (P->G vs G->G)"
echo "  E4  e4_*.json     (override vs propagation)"
echo "  E5  e5_table.md   (greedy / MBR / oracle)"
