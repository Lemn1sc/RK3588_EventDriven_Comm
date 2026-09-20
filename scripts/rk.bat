@echo off
REM Run a Python script with the project's "rk" conda env (cmd / PowerShell).
REM
REM Why this exists: the default `python` on this machine is the Windows Store
REM build, which has no cv2 / torch / ultralytics -- running a script with it
REM fails with "ModuleNotFoundError: No module named 'cv2'".
REM
REM NOTE: this file is intentionally ASCII-only. cmd reads .bat files in the
REM system codepage (GBK on zh-CN Windows), so UTF-8 Chinese text here would be
REM mojibake and get executed as commands.
REM
REM Usage:
REM   scripts\rk.bat scripts\demo_fall_detect.py --batch 5
REM   scripts\rk.bat -c "import cv2; print(cv2.__version__)"
setlocal

if "%RK_PYTHON%"=="" set "RK_PYTHON=%USERPROFILE%\miniconda3\envs\rk\python.exe"

if not exist "%RK_PYTHON%" (
  echo [rk] interpreter not found: %RK_PYTHON% 1>&2
  echo [rk] if the env lives elsewhere, point RK_PYTHON at it: 1>&2
  echo [rk]   set RK_PYTHON=C:\path\to\python.exe 1>&2
  exit /b 1
)

"%RK_PYTHON%" %*
