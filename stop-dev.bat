@echo off
title Stop TraceCraft Local Dev
cls
echo ======================================================================
echo           Stopping TraceCraft Local Dev Services (8000, 5000, 3000)
echo ======================================================================
echo.

echo Freeing Port 8000 (Engine)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul

echo Freeing Port 5000 (Backend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul

echo Freeing Port 3000 (Frontend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul

echo.
echo All services stopped!
pause
