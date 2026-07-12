"""Smoke-test canon snapshot rebuild and MCP readiness on temp dirs.

The test patches embeddings with deterministic 4-d vectors, so it exercises
sqlite-vec, rebuild publication, canon parsing and MCP not-ready behavior without
downloading the production embedding model.

Run: python3 selftest.py
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="memory-mcp-selftest-"))
MEMORY_DIR = TMP / "memory"
CANON = MEMORY_DIR / "facts"
EXPORT = MEMORY_DIR / "export.ndjson"
DB = MEMORY_DIR / "index.sqlite"

os.environ["MEMORY_CANON_ROOT"] = str(CANON)
os.environ["MEMORY_CANON_EXPORT"] = str(EXPORT)
os.environ["MEMORY_DB"] = str(DB)
os.environ["MEMORY_DIM"] = "4"
os.environ["MEMORY_SEARCH_LOG"] = str(MEMORY_DIR / "search-log.jsonl")

import numpy as np  # noqa: E402

import server  # noqa: E402
import reindex  # noqa: E402


def configure(canon: Path, export: Path, db: Path) -> None:
    server.CANON = canon
    server.CANON_EXPORT = export
    server.DB_PATH = str(db)
    server.SEARCH_LOG = str(db.parent / "search-log.jsonl")
    server.mark_search_not_ready()


def fake_vec(text: str) -> np.ndarray:
    lower = text.lower()
    if "raise_embed" in lower:
        raise RuntimeError("forced embedding failure")
    vec = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    groups = [
        (("openrouter", "api key", "open_router_key"), 0),
        (("dnd", "world", "симуляция"), 1),
        (("fallback", "rollback", "committed"), 2),
        (("current", "replacement", "6768", "порт"), 3),
    ]
    for needles, idx in groups:
        if any(needle in lower for needle in needles):
            vec[idx] = 1.0
    return server._unit(vec)


server.embed_doc = fake_vec
server.embed_query = fake_vec
server.embedder = lambda: object()


def fact(tags: str, body: str, extra: str = "") -> str:
    return (
        "---\n"
        f"tags: [{tags}]\n"
        "source: selftest\n"
        "created: 2026-07-12\n"
        "pinned: false\n"
        f"{extra}"
        "---\n"
        f"{body}\n"
    )


def write_export(entries: list[tuple[str, str]]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with EXPORT.open("w", encoding="utf-8") as f:
        for fact_id, text in entries:
            obj = {
                "id": fact_id,
                "path": f"{fact_id}.md",
                "text": text,
                "bytes": len(text.encode("utf-8")),
                "mtime": 1783519406,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_export_rebuild_checks() -> None:
    write_export(
        [
            (
                "global/openrouter-key",
                fact("secrets", "OpenRouter API-ключ лежит в project_inspect/.env, переменная open_router_key."),
            ),
            (
                "dnd-simulator/world-sim-scope",
                fact("projects", "dnd-simulator: симуляция мира, политика, поселения, экология."),
            ),
            (
                "global/old-port",
                fact("infra", "Obsolete old port fact with marker old_port_marker."),
            ),
            (
                "global/current-port",
                fact("infra", "Current replacement fact says Orca serve порт 6768.", "supersedes: old-port\n"),
            ),
            (
                "global/deleted-fact",
                fact("trash", "deleted_marker must never be indexed.", "deleted: true\n"),
            ),
        ]
    )

    n = reindex.reindex()
    assert_true(n == 3, f"expected 3 current facts, got {n}")
    parity = server.parity_gate()
    assert_true(parity == {"ok": True, "expected": 3, "indexed": 3}, f"bad parity: {parity}")

    hit = server.search_memory("openrouter api key", k=1)[0]
    assert_true("open_router_key" in hit["text"] and hit["scope"] == "global", "known fact search failed")
    rows = server.memory_list(limit=10)
    text = "\n".join(row["text"] for row in rows)
    assert_true("old_port_marker" not in text, "superseded fact leaked into index")
    assert_true("deleted_marker" not in text, "deleted fact leaked into index")


def run_failed_rebuild_preserves_index() -> None:
    before = server.search_memory("openrouter api key", k=1)[0]["text"]
    write_export(
        [
            ("global/openrouter-key", fact("secrets", "RAISE_EMBED")),
        ]
    )
    try:
        server.rebuild_index()
    except RuntimeError:
        pass
    else:
        raise AssertionError("rebuild unexpectedly succeeded")
    after = server.search_memory("openrouter api key", k=1)[0]["text"]
    assert_true(before == after, "failed rebuild replaced the previous index")


def run_readiness_checks() -> None:
    server.mark_search_not_ready()
    response = server.memory_search("openrouter api key", caller="worker")
    assert_true(response["status"] == "not_ready" and response["retryable"] is True, "not-ready response is not retryable")
    server.mark_search_ready()
    response = server.memory_search("openrouter api key", caller="worker")
    assert_true(isinstance(response, list) and response, "ready search did not return hits")

    write_export(
        [
            (
                "global/openrouter-key",
                fact("secrets", "OpenRouter API-ключ лежит в project_inspect/.env, переменная open_router_key."),
            ),
        ]
    )
    server.mark_search_not_ready()
    assert_true(server.bootstrap_index() == 1 and server.search_ready(), "first bootstrap failed")
    server.mark_search_not_ready()
    assert_true(server.bootstrap_index() == 1 and server.search_ready(), "second bootstrap failed")


def run_git_fallback_checks() -> None:
    legacy = TMP / "panelmem-kb"
    legacy_canon = legacy / "memory"
    legacy_db = TMP / "legacy-index.sqlite"
    path = legacy_canon / "global" / "fallback.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fact("fallback", "fallback committed rollback fact"), encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=legacy, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "selftest@example.com"], cwd=legacy, check=True)
    subprocess.run(["git", "config", "user.name", "selftest"], cwd=legacy, check=True)
    subprocess.run(["git", "add", "memory/global/fallback.md"], cwd=legacy, check=True)
    subprocess.run(["git", "commit", "-m", "seed fallback"], cwd=legacy, check=True, stdout=subprocess.DEVNULL)

    path.write_text(fact("fallback", "dirty fallback fact should be ignored"), encoding="utf-8")
    configure(legacy_canon, legacy / "export.ndjson", legacy_db)
    n = reindex.reindex()
    assert_true(n == 1, f"fallback expected 1 fact, got {n}")
    hit = server.search_memory("fallback rollback", k=1)[0]["text"]
    assert_true("committed rollback" in hit and "dirty" not in hit, "fallback did not read committed HEAD")


try:
    configure(CANON, EXPORT, DB)
    run_export_rebuild_checks()
    run_failed_rebuild_preserves_index()
    run_readiness_checks()
    run_git_fallback_checks()
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print("PASS")
