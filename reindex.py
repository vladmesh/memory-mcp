#!/usr/bin/env python3
"""Manual fallback for rebuilding the sqlite index from the configured canon.

In normal operation the daemon rebuilds its own index in-process (see
server.rebuild_index + the canon watcher). This script is the offline path. Run it
with the daemon stopped (single writer to the sqlite file) for a migration, a model/dim
change, or to force a rebuild without the running daemon. Same logic, one implementation.
"""
import server


def reindex() -> int:
    server.embedder()
    return server.rebuild_index()


if __name__ == "__main__":
    n = reindex()
    parity = server.parity_gate()
    print(
        f"reindexed {n} facts from {server.CANON} into {server.DB_PATH} "
        f"at dim={server.DIM} model={server.MODEL}"
    )
    print(
        f"parity: expected={parity['expected']} indexed={parity['indexed']} "
        f"ok={parity['ok']}"
    )
