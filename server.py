"""Memory MCP server — shared semantic memory for any MCP-speaking agent.

SQLite + sqlite-vec for storage/ANN, fastembed (bge-m3, multilingual) for embeddings,
exposed over streamable-HTTP so Claude / Codex / Hermes all share ONE warm instance.
"""

import os
import sqlite3
import datetime
import threading
from pathlib import Path

import numpy as np
import sqlite_vec
from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).parent
DB_PATH = os.environ.get("MEMORY_DB", str(HERE / "memory.db"))
MODEL = os.environ.get("MEMORY_MODEL", "intfloat/multilingual-e5-large")
PORT = int(os.environ.get("MEMORY_PORT", "8077"))
DIM = int(os.environ.get("MEMORY_DIM", "1024"))

_embedder = None
_lock = threading.Lock()


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = TextEmbedding(model_name=MODEL)
    return _embedder


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def embed_doc(text: str) -> np.ndarray:
    return _unit(list(embedder().embed([text]))[0])


def embed_query(text: str) -> np.ndarray:
    return _unit(list(embedder().query_embed(text))[0])


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db() -> None:
    conn = db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
        "tags TEXT, source TEXT, created_at TEXT)"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[{DIM}])"
    )
    conn.commit()
    conn.close()


def add_memory(text: str, tags=None, source=None) -> int:
    conn = db()
    cur = conn.execute(
        "INSERT INTO memories(text, tags, source, created_at) VALUES (?,?,?,?)",
        (text, tags, source, datetime.datetime.utcnow().isoformat()),
    )
    rid = cur.lastrowid
    conn.execute(
        "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
        (rid, sqlite_vec.serialize_float32(embed_doc(text).tolist())),
    )
    conn.commit()
    conn.close()
    return rid


def search_memory(query: str, k: int = 5) -> list:
    conn = db()
    rows = conn.execute(
        "SELECT v.rowid, v.distance, m.text, m.tags, m.source, m.created_at "
        "FROM vec_memories v JOIN memories m ON m.id = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
        (sqlite_vec.serialize_float32(embed_query(query).tolist()), k),
    ).fetchall()
    conn.close()
    # unit vectors → L2 distance d relates to cosine: cos = 1 - d^2/2
    return [
        {
            "id": r[0],
            "score": round(1 - (r[1] ** 2) / 2, 4),
            "text": r[2],
            "tags": r[3],
            "source": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


mcp = FastMCP("memory", host="127.0.0.1", port=PORT)

# Память read-only для агентов: пишет только куратор (markdown-канон → reindex).
# Внутренний add_memory оставлен для миграции/selftest, но через MCP не светится.


@mcp.tool()
def memory_search(query: str, k: int = 5) -> list:
    """Semantic search over shared memory. Returns up to k closest entries with scores."""
    return search_memory(query, k)


@mcp.tool()
def memory_get(id: int) -> dict:
    """Fetch one memory entry by id."""
    conn = db()
    r = conn.execute(
        "SELECT id, text, tags, source, created_at FROM memories WHERE id = ?", (id,)
    ).fetchone()
    conn.close()
    if not r:
        return {"error": "not found", "id": id}
    return {"id": r[0], "text": r[1], "tags": r[2], "source": r[3], "created_at": r[4]}


@mcp.tool()
def memory_list(limit: int = 50) -> list:
    """List recent memory entries (newest first)."""
    conn = db()
    rows = conn.execute(
        "SELECT id, text, tags, source, created_at FROM memories ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "text": r[1], "tags": r[2], "source": r[3], "created_at": r[4]}
        for r in rows
    ]


if __name__ == "__main__":
    init_db()
    embedder()  # warm the model once at startup
    print(f"memory-mcp ready: model={MODEL} dim={DIM} db={DB_PATH} port={PORT}", flush=True)
    mcp.run(transport="streamable-http")
