@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
if exist ".venv311\Scripts\python.exe" (
    ".venv311\Scripts\python.exe" -c "import pandas" >nul 2>&1
    if not errorlevel 1 set "PYTHON=.venv311\Scripts\python.exe"
)
if not defined PYTHON if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import pandas" >nul 2>&1
    if not errorlevel 1 set "PYTHON=.venv\Scripts\python.exe"
)
if not defined PYTHON (
    py -3.11 -c "import pandas" >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -3.11"
)
if not defined PYTHON (
    echo No usable Python environment with pandas was found.
    echo Install stable Python 3.11 and run:
    echo   py -3.11 -m venv .venv311
    echo   .venv311\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Checking port 5001...
powershell -NoProfile -Command "$port = 5001; $processIds = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($processId in $processIds) { $process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $processId); if ($process.CommandLine -notmatch 'dashboard\.py') { Write-Error ('Port ' + $port + ' is occupied by another application (PID ' + $processId + '). It was not stopped.'); exit 1 }; Write-Host ('Stopping existing dashboard process ' + $processId); Stop-Process -Id $processId -Force -ErrorAction Stop }; Start-Sleep -Milliseconds 500; if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { Write-Error ('Port ' + $port + ' is still in use'); exit 1 }"
if errorlevel 1 (
    echo Could not safely free port 5001. Stop its owner or choose another port.
    pause
    exit /b 1
)

%PYTHON% dashboard.py

if errorlevel 1 (
    echo Application stopped with an error.
    pause
    exit /b 1
)
