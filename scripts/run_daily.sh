#!/usr/bin/env bash
# Entrypoint for cron. Add to crontab:
#   0 8 * * * /path/to/debat-zero/scripts/run_daily.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"

echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] === Starting daily run ===" | tee -a "$LOG_FILE"

cd "$REPO_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python -m tracker.runner 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] === Done (exit $EXIT_CODE) ===" | tee -a "$LOG_FILE"
exit "$EXIT_CODE"
