@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Starting Finance AI & 24/7 Market Signal Engine...
echo ==========================================================
echo Current directory: %CD%
py run_local.py
pause
