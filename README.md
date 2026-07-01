# memory-mcp

A small semantic-memory server exposed over MCP, so any MCP-speaking agent
(Claude Code, Codex, Gemini CLI, Hermes, ...) shares one warm memory store.

- **Storage / ANN:** SQLite + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) (vectors live in the same file).
- **Embeddings:** [`fastembed`](https://github.com/qdrant/fastembed) on CPU, multilingual model (`intfloat/multilingual-e5-large`, 1024-dim). No GPU, no local LLM.
- **Transport:** streamable-HTTP daemon, model loads once at startup and stays warm, so there is no per-agent cold start.

## Tools

- `memory_add(text, tags?, source?)`
- `memory_search(query, k=5)`
- `memory_get(id)`
- `memory_list(limit=50)`

## Run

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run.sh                     # serves on 127.0.0.1:8077
```

Connect an agent (Claude Code example):

```sh
claude mcp add --scope user --transport http panelmem http://127.0.0.1:8077/mcp
```

> Pick an MCP server name that does not collide with the agent's built-in
> toolsets (e.g. Hermes already has a `memory` toolset, so use `panelmem`).

## Config (env)

| var | default | note |
|-----|---------|------|
| `MEMORY_DB` | `./memory.db` | sqlite file |
| `MEMORY_MODEL` | `intfloat/multilingual-e5-large` | fastembed model id |
| `MEMORY_DIM` | `1024` | must match the model |
| `MEMORY_PORT` | `8077` | HTTP port |
| `MEMORY_SEARCH_LOG` | `<db dir>/search-log.jsonl` | append-only jsonl log of every `memory_search` call |

Changing the model means re-embedding: stop the daemon, set `MEMORY_MODEL`/`MEMORY_DIM`, run `python reindex.py`, start again.

## Deploy

`memory-mcp.service` is a systemd unit template (expects a checkout at
`/home/dev/memory-mcp`). Adjust paths, then `systemctl enable --now memory-mcp`.

## Notes

- `sqlite-vec` KNN needs `WHERE embedding MATCH ? AND k = ?`, not a `LIMIT` after a JOIN.
- The markdown canon is the source of truth; this DB is a derived index and can be rebuilt from it with `reindex.py`.
- **systemd `203/EXEC`:** `run.sh` needs the exec bit (`chmod +x`), else the unit fails on start.
- **onnxruntime 1.27 + HF symlink cache:** fails with `External data path escapes model directory`. Download the model once with `HF_HUB_DISABLE_SYMLINKS=1 HF_HUB_DISABLE_SYMLINKS_DOWNLOAD=1` (real file copies) before starting the daemon.
- **fastembed 0.8.0** switched e5-large to mean-pooling (was CLS) — harmless warning at load, search stays consistent (index and query use the same model). To restore old behaviour, pin `fastembed==0.5.1` or use `add_custom_model`.
