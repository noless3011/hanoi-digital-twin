@echo off
rem Start the bounding-box labeller (perception\finetune) and open it in the browser.
rem   label.bat                       label data\labeling\hanoi-v1
rem   label.bat data\labeling\other   label another project
rem If the project does not exist yet, frames are sampled from data\ first (100 per camera, pre-labelled).
rem Stop the labeller with Ctrl+C in this window; every edit is already saved.

setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PROJECT=%~1"
if "%PROJECT%"=="" set "PROJECT=data\labeling\hanoi-v1"
set "PORT=8000"

if not exist "%PY%" (
    echo Python environment not found at %PY%. Run setup.ps1 first.
    exit /b 1
)

if not exist "%PROJECT%\items.json" (
    echo No labelling project at %PROJECT%; sampling frames from data\ ...
    "%PY%" perception\finetune\sample_frames.py --project "%PROJECT%" --per-camera 100
    if errorlevel 1 exit /b 1
)

rem Open the browser once the server has had a moment to start (ping waits ~2 s and, unlike timeout, needs no console).
start "" /b cmd /c "ping -n 3 127.0.0.1 >nul & start "" http://localhost:%PORT%"
"%PY%" perception\finetune\label_server.py --project "%PROJECT%" --port %PORT%
