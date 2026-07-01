#!/usr/bin/env python3
"""Rebuild the sqlite index from the markdown canon (panelmem-kb).

Canon is source of truth: one fact = one .md file under `memory/`, with YAML
frontmatter + body ("утверждение + почему"). Scope comes from the first path
component under memory/ (`global` → global, else `project:<dir>`). This wipes
and repopulates both `memories` and `vec_memories` from the canon, so the
sqlite file is a disposable derivative.

Run with the memory-mcp service stopped (single writer to the sqlite file).
"""
import os
from pathlib import Path

import sqlite_vec
import yaml

import server

KB = Path(os.environ.get("PANELMEM_KB", str(Path.home() / "panelmem-kb")))
CANON = KB / "memory"


def scope_for(path: Path) -> str:
    top = path.relative_to(CANON).parts[0]
    return "global" if top == "global" else f"project:{top}"


def parse_fact(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        meta = yaml.safe_load(front) or {}
    tags = meta.get("tags")
    return {
        "scope": scope_for(path),
        "text": body.strip(),
        "tags": ",".join(tags) if tags else None,
        "source": meta.get("source"),
        "created_at": str(meta["created"]) if meta.get("created") else None,
    }


def load_canon() -> list:
    return [parse_fact(md) for md in sorted(CANON.rglob("*.md"))]


def reindex() -> None:
    facts = load_canon()
    conn = server.db()
    conn.execute("DROP TABLE IF EXISTS memories")
    conn.execute("DROP TABLE IF EXISTS vec_memories")
    conn.execute(
        "CREATE TABLE memories("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
        "scope TEXT, tags TEXT, source TEXT, created_at TEXT)"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_memories USING vec0(embedding float[{server.DIM}])"
    )
    for f in facts:
        cur = conn.execute(
            "INSERT INTO memories(text, scope, tags, source, created_at) VALUES (?,?,?,?,?)",
            (f["text"], f["scope"], f["tags"], f["source"], f["created_at"]),
        )
        conn.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, sqlite_vec.serialize_float32(server.embed_doc(f["text"]).tolist())),
        )
    conn.commit()
    conn.close()
    print(f"reindexed {len(facts)} facts from {CANON} at dim={server.DIM} model={server.MODEL}")


if __name__ == "__main__":
    server.embedder()
    reindex()
