@echo off
title TraceCraft Local Dev Launcher
cls
echo ======================================================================
echo           TraceCraft Local Development Launcher (1-Click)
echo ======================================================================
echo.
echo   [1/3] Engine:    http://localhost:8000 (Python FastAPI)
echo   [2/3] Backend:   http://localhost:5000 (Express REST API)
echo   [3/3] Frontend:  http://localhost:3000 (Next.js Studio)
echo.
echo ======================================================================
echo.

cd /d "%~dp0"

echo Starting [1/3] Vectorizer Engine (Port 8000)...
start "TraceCraft - [1] Engine (8000)" cmd /k "title Engine (Port 8000) && cd /d "%~dp0" && .\engine\.venv\Scripts\uvicorn.exe engine.api:app --host 127.0.0.1 --port 8000 --reload"

echo Starting [2/3] Backend API (Port 5000)...
start "TraceCraft - [2] Backend (5000)" cmd /k "title Backend (Port 5000) && cd /d "%~dp0backend" && pnpm dev"

echo Starting [3/3] Frontend (Port 3000)...
start "TraceCraft - [3] Frontend (3000)" cmd /k "title Frontend (Port 3000) && cd /d "%~dp0frontend" && pnpm dev"

echo.
echo ======================================================================
echo   All 3 services are running in separate terminal windows!
echo   Opening Studio in browser: http://localhost:3000/studio ...
echo ======================================================================
echo.

timeout /t 4 >nul
start http://localhost:3000/studio

echo Done! (You can close this launcher window; the 3 service windows will keep running.)
