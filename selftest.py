"""Direct store self-test (no MCP layer): proves sqlite-vec + bge-m3 + search."""
import time
import server as s

t0 = time.time()
s.init_db()
facts = [
    ("OpenRouter ключ хранится в project_inspect/.env, переменная open_router_key", "secrets"),
    ("Orca serve крутится headless на порту 6768 через systemd unit orca-server", "infra"),
    ("control-panel — приватный репозиторий-секретарь, единая точка входа", "meta"),
    ("Hermes ставится через install.sh, конфиг и память в ~/.hermes", "agents"),
    ("dnd-simulator — симуляция мира: политика, поселения, экология, существа", "projects"),
]
ids = [s.add_memory(t, tags=tag, source="selftest") for t, tag in facts]
print(f"loaded model + indexed {len(ids)} facts in {time.time()-t0:.1f}s (model={s.MODEL})")

for q in ["где лежит api key от openrouter", "на каком порту оркестратор", "что за игра про мир"]:
    hit = s.search_memory(q, k=1)[0]
    print(f"\nQ: {q}\n -> [{hit['score']}] {hit['text'][:75]}")
