@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"
set "LOG_DIR=%PROJECT_DIR%tmp\\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "app.py" (
    echo [Auditr] app.py was not found in:
    echo %PROJECT_DIR%
    pause
    exit /b 1
)

set "PYTHON_EXE=python"
if exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
)

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [Auditr] Python is not available. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import streamlit,pandas,numpy,plotly,xgboost" >nul 2>&1
if errorlevel 1 (
    echo [Auditr] Installing missing dependencies...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r "%PROJECT_DIR%requirements.txt"
)

set "RUNNING_PID="
for /f %%P in ('powershell -NoProfile -Command "$p=(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess); if($p){$p}"') do set "RUNNING_PID=%%P"

if defined RUNNING_PID (
    start "" "http://127.0.0.1:8501"
    exit /b 0
)

if exist "%LOG_DIR%\\streamlit.stdout.log" del /f /q "%LOG_DIR%\\streamlit.stdout.log" >nul 2>&1
if exist "%LOG_DIR%\\streamlit.stderr.log" del /f /q "%LOG_DIR%\\streamlit.stderr.log" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -FilePath '%PYTHON_EXE%' -ArgumentList '-m','streamlit','run','app.py','--server.port','8501','--server.headless','true' -WorkingDirectory '%PROJECT_DIR%' -WindowStyle Hidden -RedirectStandardOutput '%LOG_DIR%\\streamlit.stdout.log' -RedirectStandardError '%LOG_DIR%\\streamlit.stderr.log'"

set /a WAIT_COUNT=0
:wait_loop
set /a WAIT_COUNT+=1
set "READY_FLAG="
for /f %%R in ('powershell -NoProfile -Command "if(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue){'1'}"') do set "READY_FLAG=%%R"

if "%READY_FLAG%"=="1" goto launch_browser
if %WAIT_COUNT% GEQ 25 goto launch_failed
timeout /t 1 >nul
goto wait_loop

:launch_browser
start "" "http://127.0.0.1:8501"
exit /b 0

:launch_failed
echo [Auditr] App failed to start on port 8501.
echo Check:
echo   %LOG_DIR%\\streamlit.stderr.log
echo   %LOG_DIR%\\streamlit.stdout.log
pause
exit /b 1
