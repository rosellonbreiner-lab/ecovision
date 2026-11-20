@echo off
title EcoVision - App Streamlit
echo Iniciando interfaz Streamlit de EcoVision...
cd /d "%~dp0"
..\.venv\Scripts\python.exe -m streamlit run app.py
pause
