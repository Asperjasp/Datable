#!/usr/bin/env bash
# Automated debate runner - runs at 7:30 AM and 8:30 PM daily
# Add to crontab:
#   30 7 * * * /path/to/debat-zero/scripts/run_debates.sh morning
#   30 20 * * * /path/to/debat-zero/scripts/run_debates.sh evening

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
DEBATE_DIR="$REPO_DIR/data/debates"
RUN_TYPE="${1:-morning}"

mkdir -p "$LOG_DIR" "$DEBATE_DIR"

LOG_FILE="$LOG_DIR/debates_$(date +%Y-%m-%d).log"
TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S%z)

echo "[$TIMESTAMP] === Starting $RUN_TYPE debate run ===" | tee -a "$LOG_FILE"

cd "$REPO_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run the daily bias tracker first
echo "[$TIMESTAMP] Running daily bias tracker..." | tee -a "$LOG_FILE"
python -m tracker.runner 2>&1 | tee -a "$LOG_FILE"

# Run preset debates
echo "[$TIMESTAMP] Running preset debates..." | tee -a "$LOG_FILE"

# Define the debates to run
DEBATES=(
    "cepeda:de_la_espriella:salud"
    "cepeda:fajardo:educación"
    "fajardo:de_la_espriella:economía"
    "valencia:cepeda:seguridad"
)

for debate in "${DEBATES[@]}"; do
    IFS=':' read -r candidate_a candidate_b topic <<< "$debate"
    echo "[$TIMESTAMP] Running debate: $candidate_a vs $candidate_b on $topic" | tee -a "$LOG_FILE"
    
    python -c "
import asyncio
from tracker.debate_simulator import run_debate, save_debate_result

async def main():
    result = await run_debate(
        candidate_a_key='$candidate_a',
        candidate_b_key='$candidate_b',
        actor_model_key='qwen_direct',
        judge_model_key='claude',
        topic='$topic',
    )
    path = save_debate_result(result)
    print(f'Debate saved to: {path}')

asyncio.run(main())
" 2>&1 | tee -a "$LOG_FILE"
done

EXIT_CODE=${PIPESTATUS[0]}
echo "[$TIMESTAMP] === Done (exit $EXIT_CODE) ===" | tee -a "$LOG_FILE"
exit "$EXIT_CODE"
