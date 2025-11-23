#!/usr/bin/env bash
# Safe cleanup helper for removing experiment artifacts and caches.
# Run from project root: ./scripts/clean_logs.sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "This will remove experiment output directories and Python caches under:"
echo "  $ROOT/logs"
echo "  $ROOT/experiments/logs"
echo "  __pycache__ directories under src/ and experiments/"
read -p "Proceed and delete these files? [y/N]: " ans
if [ "${ans:-n}" != "y" ]; then
  echo "Abort. Nothing was deleted."
  exit 0
fi

# remove experiment logs and caches
rm -rf "$ROOT/logs" || true
rm -rf "$ROOT/experiments/logs" || true
find "$ROOT" -type d -name '__pycache__' -print0 | xargs -0 rm -rf || true
find "$ROOT" -type f -name '*.pyc' -print0 | xargs -0 rm -f || true

echo "Cleanup complete."
