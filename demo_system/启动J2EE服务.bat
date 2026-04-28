@echo off
chcp 65001
title J2EE Service
set JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF-8

cd /d "%~dp0j2ee_service"

echo ========================================
echo   Starting J2EE Service...
echo   Press Ctrl+C to stop
echo ========================================
echo.

mvn spring-boot:run