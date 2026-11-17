@echo off
title OralVision AI
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1

set "PY_EXE=venv_fixed\Scripts\python.exe"
if not exist "%PY_EXE%" (
  set "PY_EXE=venv\Scripts\python.exe"
)

if not exist "%PY_EXE%" (
  echo ERROR: Virtual environment not found.
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)

if "%FLASK_SECRET_KEY%"=="" (
  set "FLASK_SECRET_KEY=oralvision-local-dev-secret-key-321"
)

echo ===================================================
echo               OralVision AI Server
echo ===================================================
echo Starting application...
echo The browser will open automatically at http://127.0.0.1:5000
echo Keep this terminal window open while using the application.
echo ===================================================
echo.
"%PY_EXE%" app.py

if errorlevel 1 (
  echo.
  echo Application encountered an error or closed.
  pause
)
