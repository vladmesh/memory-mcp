#!/usr/bin/env python3
"""Offline, atomic rebuild of a memory-mcp index from a canon snapshot."""
import argparse
import json
from pathlib import Path

import server


def rebuild(canon, export, target_db, model, dim, document_embed=None, allow_empty=False) -> dict:
    """Public rebuild entry point. ``document_embed`` keeps tests model-free."""
    return server.offline_rebuild(canon, export, target_db, model, dim, document_embed, allow_empty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canon", required=True, type=Path, help="markdown facts root")
    parser.add_argument("--export", required=True, type=Path, help="published export.ndjson snapshot")
    parser.add_argument("--target-db", required=True, type=Path, help="sqlite index to atomically replace")
    parser.add_argument("--model", required=True, help="fastembed model id")
    parser.add_argument("--dim", required=True, type=int, help="embedding vector dimension")
    parser.add_argument("--allow-empty", action="store_true", help="publish an empty index intentionally")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(rebuild(**vars(parse_args())), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1)
