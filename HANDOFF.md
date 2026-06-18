# Handoff — RAG / Agent thinking-mode & low-confidence-gate fixes

本次改動聚焦三個檔案，解決「thinking 型 LLM（gemma4）讓 RAG/Agent 變慢甚至逾時 500」以及「Agent 在低信心時短路、不用 KG 工具」兩類問題。所有改動皆在本機（Windows + RTX 5090 + Ollama 0.30.8，LLM=gemma4、embedding=qwen3-embedding:8b）以完整 1089 頁 MIL-STD-810H 純文字 ingest（~2453 chunks）實測過。

## 改了什麼（已 commit）

### 1. `backend/app/services/ollama_client.py` — `think` 永遠明確送出
`chat()` / `chat_stream()` 原本是「只有 `think=True` 才把 `think` 放進 payload」，`think=False`（預設）時**省略**該欄位，於是 Ollama 退回模型預設 —— 而 thinking 型模型（gemma4）預設**思考 ON**。
改成永遠 `payload["think"] = think`。
- 實測 gemma4:12b：thinking 18.8s → `think=false` 1.9s（約 10×），答案等同或更完整。
- 對非思考模型（qwen2.5vl / qwen3-vl-instruct）送 `think:false` 安全（Ollama 回 200）。

### 2. `backend/app/services/ai.py` — RAG 生成關閉思考
`generate_rag_answer()` 與 RAG 串流原本硬寫 `think=True`，使每題 RAG 答案要 60–120s、撞上 `OLLAMA_TIMEOUT=120` → 500。
兩處改為 `think=False`。
- 修正後 RAG `/query` 每題約 10–27s、零 500，跨語言（英文問→中文答）、帶 `[來源N]` 引用。

### 3. `backend/app/services/agent.py` — 移除低信心短路，改為「合成→不足則補充→仍不足才低信心」
**原行為**：Agent 開頭做一次基準 RAG seed 檢索；若 cross-encoder 信心 < `RAG_LOWCONF_CE_THRESHOLD`(0.15) 就**直接回低信心模板、不進 ReAct 迴圈**（註解寫「與 RAG 模式一致」，是從純 RAG 模式抄來的）。
**問題**：版本/取代/引用這類「向量檢索弱、但 KG 工具強」的題型，正好都低於門檻 → KG 工具（`spec_supersedes_chain` 等）永遠沒機會被呼叫。換 12b/26b 都一樣失敗 → 證實是這道閘門、非模型能力。

**新行為**（取代閘門）：
1. seed 檢索照舊（證據一律存進 `rag_evidence`），但**不再因低信心短路**。
2. ReAct 迴圈照常跑（LLM 自由用 KG/結構/RAG 工具，這是第一層補充）。
3. 迴圈後 `_grounded_synthesis()` 合成帶引用答案。
4. **充足性檢查（確定性，不靠 LLM）**：合成不出，或「迴圈完全沒撈到 KG 關聯且 seed 信心低」→ 視為不足。
5. **自動補充一輪**：更廣檢索（top_k 10）+ 對問題中的規範 ID 自動 `spec_references` 展開 → **重新合成**。（這是給「不夠就補」的確定性保證，不依賴 LLM 判斷。）
6. 補過仍**完全無證據**才回低信心模板（真正最後手段）。

同時移除了過程中嘗試過的暫時性 hack（`_looks_like_spec_relation` 豁免清單）——新管線已涵蓋，不需逐題型豁免。

**驗證（gemma4:12b，5 題批次）**：
| 題型 | 工具呼叫 | 結果 |
|---|---|---|
| 版本演進鏈 | spec_lookup, spec_supersedes_chain | ✅ 正確「取代 810G w/Change 1」（修前是 0 工具、fallback）|
| 引用 ASTM/ISO | spec_lookup, spec_references | ✅ |
| 鹽霧程序 | rag_search | ✅ 詳盡 |
| 兩規範間關聯 | spec_lookup×2, spec_references×2 | ✅ 合成出 528.1 與 MIL-STD-167 的取代關係 |
| 語料外（量子電腦抗磁暴）| rag_search×4 | ✅ 誠實「查無」+ 列出實際存在 Method，不幻覺 |

> 行為變更注意：低信心模板現在只在「補過仍零證據」時出現；對「有最接近段落但向量信心低」的題，會給 grounded 的誠實答案（含「查無但最接近…」），不再直接套模板。若團隊偏好保留舊的模板 UX，可在第 6 步加一個「seed 低信心且補後仍無 KG 證據 → 用模板」的分支。

## 環境層調整（**未**進此 commit，需在部署環境另行處理）

- **GPU OCR（rapid 引擎 + onnxruntime-gpu，CUDA 13）**：`.venv` 內放了 `sitecustomize.py`，把 CUDA-13 nvidia wheel 的 DLL 目錄 `nvidia/cu13/bin/x86_64`（onnxruntime 1.27 的 `preload_dlls()` 不會掃這層）加進 DLL 搜尋路徑，否則 CUDA EP 載入失敗、靜默退回 CPU（server tier CPU ≈ 110s/頁，GPU ≈ 10s/頁）。需安裝：`onnxruntime-gpu rapidocr rapid_layout rapid_table` + `--extra-index-url https://pypi.nvidia.com nvidia-cublas nvidia-cudnn-cu13 nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cufft nvidia-curand`，並於 `.env` 設 `OCR_ENGINE=rapid OCR_MODEL_TIER=server OCR_VERSION=PP-OCRv5 OCR_DEVICE=gpu`。
- **SQLite 鎖競爭**：大文件 KG 自動抽取（重寫入）會讓 login/RAG 偶發 500（寫鎖）。本機以 `PRAGMA journal_mode=WAL` + `busy_timeout` 緩解。**建議**在 DB engine 初始化處（`app/database.py`）對 SQLite 連線加上 `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=8000;`，或正式多人使用改 PostgreSQL（`DATABASE_URL`）。此項刻意未進 commit（未以程式路徑測試）。
- 本機 `.env` 的模型選擇（非機密、未提交）：`OLLAMA_LLM_MODEL=gemma4:12b`、`OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct`（注意：`qwen3-vl:8b` 不帶 `-instruct` 是 thinking 版，Ollama 0.30.8 下 `think:false`/`/no_think` 皆無法關閉思考 → 逐頁 90–145s，**請用 `-instruct`**）、`OLLAMA_EMBED_MODEL=qwen3-embedding:8b`。

## 測試/診斷腳本
過程中在 `backend/scripts/` 寫了若干一次性腳本（`ocr_speed.py`、`vision_speed.py`、`agent_batch.py`、`kg_*.py`、`cuda_diag.py` 等），含本機絕對路徑，**未提交**。如需重現基準可參考，但建議重寫成可攜版本。
