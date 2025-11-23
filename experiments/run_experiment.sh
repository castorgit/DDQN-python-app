#!/usr/bin/env bash
# Use strict flags but avoid `pipefail` to be compatible when invoked with `/bin/sh`.
# Prefer running via `bash run_experiment.sh` or `./run_experiment.sh` so the shebang is honored.
set -eu

# run_experiment.sh
# Convenience wrapper to run the experiment runner from the project.
# Usage:
#   ./run_experiment.sh [CONFIG_PATH] [OUT_DIR] [-- additional args passed to runner]
# Examples:
#   ./run_experiment.sh
#   ./run_experiment.sh experiments/configs/configuration1.yaml ./logs
#   PYTHON=/home/jaumemanero/RL/bin/python ./run_experiment.sh experiments/configs/configuration1.yaml ./logs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Allow overriding python executable via environment
PYTHON="${PYTHON:-python}"

CONFIG="${1:-$PROJECT_ROOT/experiments/configs/configuration_smoke.yaml}"
OUT="${2:-$PROJECT_ROOT/logs}"

echo "Project root: $PROJECT_ROOT"
echo "Using python: $PYTHON"
echo "Config: $CONFIG"
echo "Out dir: $OUT"

cd "$PROJECT_ROOT"
# forward remaining args ($3 and beyond) in a POSIX-compatible way
EXTRA_ARGS=""
if [ "$#" -gt 2 ]; then
	i=1
	for a in "$@"; do
		if [ $i -gt 2 ]; then
			# append each extra arg, properly quoted later via eval
			EXTRA_ARGS="$EXTRA_ARGS \"$a\""
		fi
		i=$((i + 1))
	done
fi

# construct and run the command
CMD="\"$PYTHON\" experiments/run_experiment.py --config \"$CONFIG\" --out \"$OUT\" $EXTRA_ARGS"
eval "$CMD"
