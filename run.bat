@echo off
rem Pakistan Law web page: ask the statute book as of a date.
rem   run.bat                         serve on 8841 and open a browser tab
rem   run.bat --port 9000             another port
rem   run.bat --no-browser            serve only
rem   run.bat --corpus path\to\file   another corpus (default: PAKLAW_CORPUS, then data\pakistan-code.jsonl.gz)
rem Close this window (or Ctrl+C) to stop it.
title Pakistan Law
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY for %%P in (py.exe) do if not "%%~$PATH:P"=="" set "PY=py"
if not defined PY for %%P in (python.exe) do if not "%%~$PATH:P"=="" set "PY=python"
if not defined PY (
  echo No Python found. Install Python 3.10+ or create .venv first.
  exit /b 1
)
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
"%PY%" -m paklaw.web %*
exit /b %errorlevel%
