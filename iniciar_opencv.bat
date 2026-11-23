@echo off
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
title EcoVision - Visor OpenCV de Camara Local
echo =======================================================
echo   Iniciando Visor de Camara Local EcoVision (OpenCV HUD)
echo =======================================================
cd /d "%~dp0"
set "PYTHONPATH=%~dp0..\.venv\Lib\site-packages;%PYTHONPATH%"
if exist "C:\Python313\python.exe" (
    C:\Python313\python.exe live_cam.py
) else (
    ..\.venv\Scripts\python.exe live_cam.py
)
pause
