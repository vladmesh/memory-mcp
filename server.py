"""Memory MCP server — shared semantic memory for any MCP-speaking agent.

SQLite + sqlite-vec for storage/ANN, fastembed (bge-m3, multilingual) for embeddings,
exposed over streamable-HTTP so Claude / Codex / Hermes all share ONE warm instance.
"""

import json
import os
import sqlite3
import datetime
import threading
import time
from pathlib import Path

import numpy as np
import sqlite_vec
import yaml
from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).parent
DB_PATH = os.environ.get("MEMORY_DB", str(HERE / "memory.db"))
MODEL = os.environ.get("MEMORY_MODEL", "intfloat/multilingual-e5-large")
PORT = int(os.environ.get("MEMORY_PORT", "8077"))
DIM = int(os.environ.get("MEMORY_DIM", "1024"))
SEARCH_LOG = os.environ.get("MEMORY_SEARCH_LOG", str(Path(DB_PATH).parent / "search-log.jsonl"))

# Markdown canon (panelmem-kb) is source of truth; this index is derived. The daemon
# owns the sqlite file, so the daemon rebuilds the index in-process — no stop/start,
# no second writer. A watcher thread repolls the canon and rebuilds on change.
KB = Path(os.environ.get("PANELMEM_KB", str(Path.home() / "panelmem-kb")))
CANON = KB / "memory"
WATCH_INTERVAL = float(os.environ.get("MEMORY_WATCH_INTERVAL", "10"))

_embedder = None
_lock = threading.Lock()
_reindex_lock = threading.Lock()
_search_log_lock = threading.Lock()  # own lock: don't couple log I/O to embedder/reindex locks


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
        "scope TEXT, tags TEXT, source TEXT, created_at TEXT)"
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


def normalize_scope(scope: str | None) -> str | None:
    """Accept "global", "project:<dir>" or a bare project dir name ("orca" → "project:orca")."""
    if not scope:
        return None
    scope = scope.strip()
    if scope in ("", "global") or scope.startswith("project:"):
        return scope or None
    return f"project:{scope}"


def search_memory(query: str, k: int = 5, scope: str | None = None) -> list:
    scope = normalize_scope(scope)
    # sqlite-vec KNN can't push a join filter into MATCH, so with a scope we over-fetch
    # and trim post-hoc — fine at canon size (hundreds of facts).
    fetch = max(k * 5, 25) if scope else k
    qvec = sqlite_vec.serialize_float32(embed_query(query).tolist())
    with _reindex_lock:  # don't read while a rebuild is swapping tables
        conn = db()
        rows = conn.execute(
            "SELECT v.rowid, v.distance, m.text, m.scope, m.tags, m.source, m.created_at "
            "FROM vec_memories v JOIN memories m ON m.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (qvec, fetch),
        ).fetchall()
        conn.close()
    if scope:
        rows = [r for r in rows if r[3] == scope][:k]
    # unit vectors → L2 distance d relates to cosine: cos = 1 - d^2/2
    return [
        {
            "id": r[0],
            "score": round(1 - (r[1] ** 2) / 2, 4),
            "text": r[2],
            "scope": r[3],
            "tags": r[4],
            "source": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def log_search(query: str, k: int, results: list,
               scope: str | None = None, caller: str | None = None) -> None:
    """Append one jsonl line per memory_search call. Best-effort telemetry:
    any failure here is swallowed — logging must never break the search.
    caller is self-reported by the agent (role name) — attribution, not auth."""
    try:
        entry = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "query": query,
            "k": k,
            "hits": [{"id": r["id"], "score": r["score"]} for r in results],
        }
        if scope:
            entry["scope"] = scope
        if caller:
            entry["caller"] = caller
        line = json.dumps(entry, ensure_ascii=False)
        with _search_log_lock:  # one write per line so concurrent calls don't interleave
            with open(SEARCH_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
    except Exception:
        pass


# ── Canon → index (daemon-owned reindex) ──────────────────────────────────────
# Kept in sync with reindex.py's parsing; reindex.py is the manual fallback and
# reuses these functions.


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
    if not CANON.is_dir():
        return []
    return [parse_fact(md) for md in sorted(CANON.rglob("*.md"))]


def canon_signature() -> tuple:
    """Cheap change token: (file count, max mtime, total size). Catches add/edit/delete."""
    if not CANON.is_dir():
        return (0, 0.0, 0)
    count = mtime = size = 0
    for md in CANON.rglob("*.md"):
        st = md.stat()
        count += 1
        mtime = max(mtime, st.st_mtime)
        size += st.st_size
    return (count, mtime, size)


def rebuild_index() -> int:
    """Rebuild memories + vec_memories from the canon. Embeds outside the lock; only the
    drop/create/insert runs under _reindex_lock so searches see a brief, consistent swap."""
    facts = load_canon()
    rows = [(f, embed_doc(f["text"])) for f in facts]  # slow part, no lock held
    with _reindex_lock:
        conn = db()
        conn.execute("DROP TABLE IF EXISTS memories")
        conn.execute("DROP TABLE IF EXISTS vec_memories")
        conn.execute(
            "CREATE TABLE memories("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
            "scope TEXT, tags TEXT, source TEXT, created_at TEXT)"
        )
        conn.execute(f"CREATE VIRTUAL TABLE vec_memories USING vec0(embedding float[{DIM}])")
        for f, vec in rows:
            cur = conn.execute(
                "INSERT INTO memories(text, scope, tags, source, created_at) VALUES (?,?,?,?,?)",
                (f["text"], f["scope"], f["tags"], f["source"], f["created_at"]),
            )
            conn.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                (cur.lastrowid, sqlite_vec.serialize_float32(vec.tolist())),
            )
        conn.commit()
        conn.close()
    return len(rows)


def start_canon_watcher() -> None:
    """Background thread: rebuild the index whenever the canon changes on disk."""
    def loop():
        last = canon_signature()
        while True:
            time.sleep(WATCH_INTERVAL)
            try:
                sig = canon_signature()
                if sig != last:
                    n = rebuild_index()
                    last = sig
                    print(f"memory-mcp: canon changed, reindexed {n} facts", flush=True)
            except Exception as e:  # never let the watcher kill the daemon
                print(f"memory-mcp: watcher error: {e}", flush=True)

    threading.Thread(target=loop, name="canon-watcher", daemon=True).start()


mcp = FastMCP("memory", host="127.0.0.1", port=PORT)

# Память read-only для агентов: пишет только куратор (markdown-канон → reindex).
# Внутренний add_memory оставлен для миграции/selftest, но через MCP не светится.


@mcp.tool()
def memory_search(query: str, k: int = 5, scope: str = "", caller: str = "") -> list:
    """Semantic search over shared memory. Returns up to k closest entries with scores.

    scope: optional filter — "global", "project:<dir>" or bare project dir name
    (e.g. "orca", "triggered-agents"). Search your own project's scope first,
    then retry without scope. caller: your role (worker/reviewer/steward/secretary/
    curator) — telemetry only, always pass it."""
    results = search_memory(query, k, scope=scope or None)
    # tool-level: capture real agent queries, not internal calls
    log_search(query, k, results, scope=normalize_scope(scope), caller=caller or None)
    return results


@mcp.tool()
def memory_get(id: int) -> dict:
    """Fetch one memory entry by id."""
    conn = db()
    r = conn.execute(
        "SELECT id, text, scope, tags, source, created_at FROM memories WHERE id = ?", (id,)
    ).fetchone()
    conn.close()
    if not r:
        return {"error": "not found", "id": id}
    return {"id": r[0], "text": r[1], "scope": r[2], "tags": r[3], "source": r[4], "created_at": r[5]}


@mcp.tool()
def memory_list(limit: int = 50) -> list:
    """List recent memory entries (newest first)."""
    conn = db()
    rows = conn.execute(
        "SELECT id, text, scope, tags, source, created_at FROM memories ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "text": r[1], "scope": r[2], "tags": r[3], "source": r[4], "created_at": r[5]}
        for r in rows
    ]


if __name__ == "__main__":
    embedder()  # warm the model once at startup
    n = rebuild_index()  # boot the index straight from the canon (source of truth)
    start_canon_watcher()
    print(
        f"memory-mcp ready: model={MODEL} dim={DIM} db={DB_PATH} port={PORT} "
        f"facts={n} canon={CANON} watch={WATCH_INTERVAL}s",
        flush=True,
    )
    mcp.run(transport="streamable-http")
