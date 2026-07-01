#!/usr/bin/env python3
"""Manual fallback for rebuilding the sqlite index from the markdown canon.

In normal operation the daemon rebuilds its own index in-process (see
server.rebuild_index + the canon watcher). This script is the offline path — run it
with the daemon stopped (single writer to the sqlite file) for a migration, a model/dim
change, or to force a rebuild without the running daemon. Same logic, one implementation.
"""
import server

if __name__ == "__main__":
    server.embedder()
    n = server.rebuild_index()
    print(f"reindexed {n} facts from {server.CANON} at dim={server.DIM} model={server.MODEL}")
