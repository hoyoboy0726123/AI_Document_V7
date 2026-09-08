"""確認這台主機的 RAG 回應沒有被 Ollama 反覆重載拖慢。

背景：翻譯檢索詞的呼叫曾經寫死 num_ctx=2048，而回答用 OLLAMA_NUM_CTX；Ollama 把同一個
模型的不同 num_ctx 當成不同 runner，於是每一題中文自然問法都讓主模型卸載再載入兩次
（gemma4:12b 每次 6.5 秒）。修正後兩個呼叫共用 OLLAMA_NUM_CTX。這支腳本用來在任何一台
主機上驗證：問幾題不含規範編號的中文問題，量每題耗時，並統計期間 Ollama 載入 runner 的
次數（正常情況：第一題最多載入一次，之後為 0）。

用法（後端已啟動、在 backend/ 目錄）：
    uv run python scripts/check_rag_latency.py
    uv run python scripts/check_rag_latency.py --base http://127.0.0.1:8001/api/v1 --user admin --password ...
    uv run python scripts/check_rag_latency.py --question "沙塵測試怎麼進行" --question "鹽霧測試要噴多久"

帳密預設讀 backend/.env 的 DEFAULT_ADMIN_USERNAME / DEFAULT_ADMIN_PASSWORD。
Ollama 的 server.log 在 Windows 是 %LOCALAPPDATA%\\Ollama\\server.log；其他平台可用 --ollama-log 指定，
找不到時只靠 `ollama ps` 的 CONTEXT 欄位判斷（同一模型的 CONTEXT 在執行中改變＝重載）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS = [
    "我們的筆電要做沙塵測試，測試是怎麼進行的？要在什麼溫度下吹多久？",
    "鹽霧測試要噴霧多久、乾燥多久？總共要做幾個循環？",
    "淋雨測試的降雨強度和風速是多少？每一面要淋多久？",
]


def read_env(path: str) -> dict:
    out = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def ollama_ps() -> list[tuple[str, str]]:
    """[(model, context)] from `ollama ps`; empty when nothing is loaded or ollama is not on PATH."""
    try:
        txt = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    rows = []
    for line in txt.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5:
            # NAME ID SIZE PROCESSOR CONTEXT UNTIL  (PROCESSOR may be "100% GPU" = two tokens)
            m = re.search(r"\s(\d+)\s+(?:\d+\s+\w+|Forever|Never|Stopping|\d)", line)
            ctx = m.group(1) if m else parts[-3]
            rows.append((parts[0], ctx))
    return rows


def default_ollama_log() -> str | None:
    cands = []
    if os.name == "nt":
        cands.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log"))
    cands += [os.path.expanduser("~/.ollama/logs/server.log"), "/var/log/ollama/server.log"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8001/api/v1")
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--question", action="append", help="可重複；預設三題不含規範編號的中文問題")
    ap.add_argument("--ollama-log", default=default_ollama_log())
    args = ap.parse_args()

    env = read_env(os.path.join(HERE, "..", ".env"))
    user = args.user or env.get("DEFAULT_ADMIN_USERNAME") or "admin"
    password = args.password or env.get("DEFAULT_ADMIN_PASSWORD")
    if not password:
        print("需要密碼：--password 或 backend/.env 的 DEFAULT_ADMIN_PASSWORD", file=sys.stderr)
        return 2
    questions = args.question or DEFAULT_QUESTIONS

    log_pos = None
    if args.ollama_log and os.path.exists(args.ollama_log):
        log_pos = os.path.getsize(args.ollama_log)

    c = httpx.Client(base_url=args.base, timeout=600)
    r = c.post("/auth/login", data={"username": user, "password": password})
    r.raise_for_status()
    h = {"Authorization": "Bearer " + r.json()["access_token"]}

    print(f"設定：OLLAMA_NUM_CTX={env.get('OLLAMA_NUM_CTX', '(未設定→8192)')}  OLLAMA_KEEP_ALIVE={env.get('OLLAMA_KEEP_ALIVE', '(未設定→5m)')}  "
          f"LLM={env.get('OLLAMA_LLM_MODEL', '?')}  RAG_TRANSLATE_MODEL={env.get('RAG_TRANSLATE_MODEL', '(未設定，用主模型)')}")
    before = ollama_ps()
    print("執行前 ollama ps：", before or "（沒有模型在記憶體，第一題會有一次正常載入）")

    states: list[list[tuple[str, str]]] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            s = ollama_ps()
            if not states or s != states[-1]:
                states.append(s)
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    lat = []
    for q in questions:
        t0 = time.perf_counter()
        resp = c.post("/rag/query", json={"question": q, "top_k": 5, "skip_ai_understanding": True}, headers=h)
        dt = time.perf_counter() - t0
        ok = resp.status_code == 200
        ans = (resp.json().get("answer") if ok else resp.text)[:60].replace("\n", " ")
        lat.append(dt)
        print(f"  {dt:6.1f}s  {'ok ' if ok else 'ERR'}  {q[:26]}… → {ans}")
    stop.set()
    th.join(timeout=2)

    # runner loads seen in Ollama's log during the run
    loads = []
    if log_pos is not None:
        with open(args.ollama_log, encoding="utf-8", errors="replace") as f:
            f.seek(log_pos)
            for line in f:
                m = re.search(r"new slot, n_ctx = (\d+)", line)
                if m:
                    loads.append(m.group(1))
    ctx_by_model: dict[str, set] = {}
    for s in states:
        for model, ctx in s:
            ctx_by_model.setdefault(model, set()).add(ctx)

    print(f"\n平均耗時 {sum(lat) / len(lat):.1f} 秒（{len(lat)} 題）")
    if log_pos is not None:
        print(f"期間 Ollama 載入 runner：{len(loads)} 次，n_ctx={loads or '—'}")
    print("期間 ollama ps 看到的 (模型, CONTEXT)：", {m: sorted(v) for m, v in ctx_by_model.items()} or "（ollama 不在 PATH 或沒抓到）")

    # 第一次載入（主模型、嵌入模型各一次）是正常的；同一個 n_ctx 再出現一次就是重載。
    problem = False
    if log_pos is not None and len(loads) > len(set(loads)):
        problem = True
    if any(len(v) > 1 for v in ctx_by_model.values()):
        problem = True
    if problem:
        print("\n結論：主模型在執行中被重載——請確認 backend/.env 的 OLLAMA_NUM_CTX 有設定、程式版本含 commit 1bce0b4，"
              "以及 VRAM 足以同時放下主模型與嵌入模型（`ollama ps` 的模型不應在題目之間消失或改變 CONTEXT）。")
        return 1
    print("\n結論：沒有多餘的重載，翻譯與回答共用同一個 num_ctx；耗時就是檢索與生成本身。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
