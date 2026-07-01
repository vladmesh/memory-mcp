"""Smoke test the canon→index path on temp dirs.

Builds a tiny markdown canon in a tempdir, reindexes it into a temp sqlite,
and checks semantic search. Touches neither the production memory.db nor the
real ~/panelmem-kb. Covers frontmatter parsing + reindex + sqlite-vec + search.

Run: .venv/bin/python selftest.py   (exit 0 = pass, 1 = fail)
"""
import os
import shutil
import tempfile
import time
from pathlib import Path

# Point server/reindex at throwaway paths BEFORE importing them — both read
# their env (MEMORY_DB, PANELMEM_KB) at import time.
TMP = Path(tempfile.mkdtemp(prefix="panelmem-selftest-"))
os.environ["PANELMEM_KB"] = str(TMP)
os.environ["MEMORY_DB"] = str(TMP / "index.db")

import server  # noqa: E402
import reindex  # noqa: E402

FACTS = {
    "memory/global/orca-serve-port.md": (
        "---\ntags: [infra, orca]\nsource: selftest\ncreated: 2026-06-30\n---\n"
        "Orca serve крутится headless на порту 6768 (systemd-юнит orca-server).\n\n"
        "Почему: единый оркестратор, к которому цепляются головы.\n"
    ),
    "memory/global/openrouter-key.md": (
        "---\ntags: [secrets]\nsource: selftest\ncreated: 2026-06-30\n---\n"
        "OpenRouter API-ключ лежит в project_inspect/.env, переменная open_router_key.\n"
    ),
    "memory/dnd-simulator/world-sim-scope.md": (
        "---\ntags: [projects]\nsource: selftest\ncreated: 2026-06-30\n---\n"
        "dnd-simulator — симуляция мира: политика, поселения, экология, существа.\n"
    ),
}
for rel, body in FACTS.items():
    p = TMP / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")

t0 = time.time()
server.embedder()
reindex.reindex()
print(f"reindexed {len(FACTS)} facts from temp canon in {time.time()-t0:.1f}s (model={server.MODEL})")

# (query, substring expected in top-1, expected scope)
checks = [
    ("на каком порту оркестратор", "6768", "global"),
    ("где api key от openrouter", "open_router_key", "global"),
    ("что за игра про мир", "dnd-simulator", "project:dnd-simulator"),
]
ok = True
for q, needle, scope in checks:
    hit = server.search_memory(q, k=1)[0]
    good = needle in hit["text"] and hit["scope"] == scope
    ok = ok and good
    print(f"{'ok  ' if good else 'FAIL'} [{hit['score']}] {q!r} -> scope={hit['scope']}")

shutil.rmtree(TMP, ignore_errors=True)
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
