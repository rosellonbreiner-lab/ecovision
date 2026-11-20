@echo off
title EcoVision - Visor OpenCV de Camara Local
echo Abriendo ventana HUD de camara en vivo...
cd /d "%~dp0"
..\.venv\Scripts\python.exe live_cam.py
pause
