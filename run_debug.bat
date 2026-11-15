@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    if not exist ".python\python.exe" (
        echo [ERRO] Python local nao encontrado em .python\python.exe
        pause
        exit /b 1
    )
    echo Criando ambiente virtual local .venv ...
    ".python\python.exe" -m venv .venv
)

echo === [1/2] Instalando dependencias + YOLO (Modo Debug: SEM stable-diffusion.cpp / GGUF 9B) ===
".venv\Scripts\python.exe" install.py --mode debug
if errorlevel 1 (
    echo [ERRO] Falha na instalacao do Modo Debug.
    pause
    exit /b 1
)

echo === [2/2] Iniciando Gradio UI em Modo Debug ===
".venv\Scripts\python.exe" app.py --debug --host 127.0.0.1 --port 7860
pause
