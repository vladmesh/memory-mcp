#!/usr/bin/env python3
"""Rebuild vec_memories from memories using the currently configured MEMORY_MODEL/MEMORY_DIM.

Run with the memory-mcp service stopped (single writer to the sqlite file).
"""
import sqlite_vec
import server


def reindex() -> None:
    conn = server.db()
    rows = conn.execute("SELECT id, text FROM memories").fetchall()
    conn.execute("DROP TABLE IF EXISTS vec_memories")
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_memories USING vec0(embedding float[{server.DIM}])"
    )
    for rid, text in rows:
        vec = server.embed_doc(text)
        conn.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            (rid, sqlite_vec.serialize_float32(vec.tolist())),
        )
    conn.commit()
    conn.close()
    print(f"reindexed {len(rows)} memories at dim={server.DIM} model={server.MODEL}")


if __name__ == "__main__":
    server.embedder()
    reindex()
