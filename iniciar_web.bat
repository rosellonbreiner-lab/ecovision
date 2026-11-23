@echo off
title EcoVision - Servidor en Tiempo Real
echo =======================================================
echo   Iniciando EcoVision en Tiempo Real...
echo   Abriendo navegador en: http://localhost:8000
echo =======================================================
cd /d "%~dp0"
start "" http://localhost:8000
..\.venv\Scripts\python.exe web_app.py
pause
