#!/usr/bin/env bash
# End-to-end reproduction helper (environment + directory layout + dry-run wiring).
# Extend with real retrieval / feature / train steps for your cluster.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
uv run python scripts/utils/check_environment.py --skip-api
uv run python scripts/utils/setup_directories.py
uv run python scripts/01_download_data.py --dry-run 2015 2024
echo "reproduce_paper.sh: base checks done. Add 02/03/04/05 for full reprocess + train on your HPC data."
