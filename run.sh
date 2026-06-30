#!/usr/bin/env bash
# Launch the memory-MCP HTTP daemon. Model loads once at startup and stays warm.
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
# onnxruntime 1.27 rejects the symlinked HF cache layout ("External data
# path escapes model directory"); disable symlinks so downloads land as
# real files instead.
export HF_HUB_DISABLE_SYMLINKS=1
export HF_HUB_DISABLE_SYMLINKS_DOWNLOAD=1
exec python -u server.py
