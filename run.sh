#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

exec python -m scholar_analysis.main
