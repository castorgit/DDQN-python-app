#!/usr/bin/env bash
# Convenience wrapper to run experiment_02 (1000-episode config).
# Usage:
#   ./run_experiment_01.sh [OUT_DIR] [-- extra args passed to runner]
# Examples:
#   ./run_experiment_01.sh
#   ./run_experiment_01.sh ./logs
#   PYTHON=/home/jaumemanero/RL/bin/python ./run_experiment_01.sh ./logs -- --some-flag

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_ROOT/experiments/configs/experiment_02.yaml}"
OUT="${1:-$PROJECT_ROOT/logs}"

# echo chosen settings
echo "Project root: $PROJECT_ROOT"
echo "Using python: $PYTHON"
echo "Config: $CONFIG"
echo "Out dir: $OUT"

cd "$PROJECT_ROOT"

# Collect extra args ($2 and beyond) in a POSIX-safe way
EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  i=1
  for a in "$@"; do
    if [ $i -gt 1 ]; then
      EXTRA_ARGS="$EXTRA_ARGS \"$a\""
    fi
    i=$((i + 1))
  done
fi

CMD="\"$PYTHON\" experiments/run_experiment.py --config \"$CONFIG\" --out \"$OUT\" $EXTRA_ARGS"

# Run
eval "$CMD"
