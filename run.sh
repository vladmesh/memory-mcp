#!/usr/bin/env bash
# Launch the memory-MCP HTTP daemon. Model loads once at startup and stays warm.
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
exec python -u server.py
