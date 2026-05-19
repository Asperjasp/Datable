#!/usr/bin/env bash
# Setup cron jobs for automated debate runs
# Usage: bash scripts/setup_cron.sh
# This will add two cron jobs:
#   - 7:30 AM daily: morning debate run
#   - 8:30 PM daily: evening debate run

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_FILE="$REPO_DIR/scripts/debat_zero_cron"

cat > "$CRON_FILE" << EOF
# Debat-Zero Automated Debate Runs
# Morning run: 7:30 AM
30 7 * * * cd $REPO_DIR && bash scripts/run_debates.sh morning >> logs/cron_morning.log 2>&1
# Evening run: 8:30 PM
30 20 * * * cd $REPO_DIR && bash scripts/run_debates.sh evening >> logs/cron_evening.log 2>&1
EOF

# Install cron jobs
crontab -l 2>/dev/null | grep -v "debat-zero" > /tmp/existing_cron || true
cat /tmp/existing_cron "$CRON_FILE" | crontab -

echo "✅ Cron jobs installed:"
echo "  - Morning run: 7:30 AM daily"
echo "  - Evening run: 8:30 PM daily"
echo ""
echo "To verify: crontab -l"
echo "To remove: crontab -r"
