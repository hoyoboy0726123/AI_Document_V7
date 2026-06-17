@echo off
cd /d "%~dp0"
echo [Launcher] Starting AI_Document_V3...

REM OLLAMA_NUM_CTX：環境變數優先於 .env，把 context window 拉到 8192（此 GPU 可負荷，
REM gemma4:12b 約佔 9.7GB < 12GB）。OLLAMA_TIMEOUT：12B 模型冷載/換載較久，拉長避免逾時。
start cmd /k "cd backend && set OLLAMA_NUM_CTX=8192&& set OLLAMA_TIMEOUT=300&& .venv\Scripts\python.exe -u -m uvicorn app.main:app --host 127.0.0.1 --port 8001"
start cmd /k "cd frontend && set VITE_API_TARGET=http://127.0.0.1:8001 && npm run dev -- --host --port 5175"

echo [Launcher] Applications started!
echo Backend: http://127.0.0.1:8001
echo Frontend: http://localhost:5175
pause
