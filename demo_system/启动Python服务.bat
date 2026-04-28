@echo off
chcp 65001
title Python Service
set PYTHONIOENCODING=utf-8

cd /d "%~dp0python_service"

echo ========================================
echo   Starting Python Service...
echo   Press Ctrl+C to stop
echo ========================================
echo.

python flask_api.py