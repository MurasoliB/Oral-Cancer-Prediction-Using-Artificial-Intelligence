@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1
echo Creating a clean Python 3.11 environment for OralVision...
py -3.11 -m venv venv_fixed
if errorlevel 1 (
  echo ERROR: Python 3.11 was not found. Install Python 3.11 x64 and enable the Python Launcher.
  pause
  exit /b 1
)
echo Installing compatible project dependencies...
venv_fixed\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
venv_fixed\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Checking dependency consistency...
venv_fixed\Scripts\python.exe -m pip check
if errorlevel 1 goto :error
echo.
echo Testing bundled model loading...
venv_fixed\Scripts\python.exe scripts\manual_model_check.py
if errorlevel 1 goto :error
echo.
echo Setup complete. Run run_app.bat to start the website.
pause
exit /b 0
:error
echo.
echo Setup did not complete. Copy the error shown above and send it for diagnosis.
pause
exit /b 1
