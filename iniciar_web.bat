@echo off
title EcoVision - Servidor en Tiempo Real
echo Iniciando servidor EcoVision...
cd /d "%~dp0"
..\.venv\Scripts\python.exe web_app.py
pause
