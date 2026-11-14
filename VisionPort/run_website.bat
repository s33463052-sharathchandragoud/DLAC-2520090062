@echo off
title VisionPort Web Portal
echo ========================================================
echo VisionPort: Intelligent Traffic Monitoring Web Portal
echo ========================================================
echo Starting local evaluation backend on http://localhost:8000 ...
timeout /t 2 /nobreak >nul
start "" http://localhost:8000
python main.py
pause
