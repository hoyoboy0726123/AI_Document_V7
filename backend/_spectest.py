import json, time, httpx

BASE = "http://127.0.0.1:8001/api/v1"
Q = "這份測試計畫的 pressure test 各子項目的測試規格(測試方法與條件)是什麼？"
c = httpx.Client(timeout=httpx.Timeout(600.0))
tok = c.post(BASE + "/auth/login", data={"username": "admin", "password": "Admin@123"}).json()["access_token"]
H = {"Authorization": "Bearer " + tok}

print("########## RAG (/rag/query) ##########", flush=True)
t0 = time.time()
r = c.post(BASE + "/rag/query", headers=H, json={"question": Q, "top_k": 5}).json()
print(f"time={round(time.time()-t0,1)}s sources={len(r.get('sources',[]))}", flush=True)
print(r.get("answer") or "", flush=True)

print("\n########## AGENT (/agent/chat) ##########", flush=True)
t0 = time.time(); final = ""; srcs = 0
with c.stream("POST", BASE + "/agent/chat", headers=H, json={"question": Q, "max_steps": 8}) as resp:
    for line in resp.iter_lines():
        if line and line.startswith("data:"):
            try: d = json.loads(line.split(":", 1)[1].strip())
            except Exception: continue
            if d.get("text"): final = d["text"]
            if d.get("sources"): srcs = len(d["sources"])
print(f"time={round(time.time()-t0,1)}s sources={srcs}", flush=True)
print(final, flush=True)
print("DONE", flush=True)
