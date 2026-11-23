@echo off
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
title EcoVision - Servidor en Tiempo Real
echo =======================================================
echo   Iniciando EcoVision en Tiempo Real...
echo   Abriendo navegador en: http://localhost:8000
echo =======================================================
cd /d "%~dp0"
start "" http://localhost:8000
..\.venv\Scripts\python.exe web_app.py
pause
