@echo off
REM 用项目的 rk 环境跑 Python 脚本（cmd / PowerShell 版）。
REM
REM 为什么需要这个：系统默认的 python（Windows 商店版）没有 cv2 / torch /
REM ultralytics，直接 `python scripts\xxx.py` 会报
REM ModuleNotFoundError: No module named 'cv2'。
REM
REM 用法：
REM   scripts\rk.bat scripts\demo_fall_detect.py --batch 5
REM   scripts\rk.bat -c "import cv2; print(cv2.__version__)"
setlocal

if "%RK_PYTHON%"=="" (
  set "RK_PYTHON=%USERPROFILE%\miniconda3\envs\rk\python.exe"
)

if not exist "%RK_PYTHON%" (
  echo 找不到 rk 环境的解释器：%RK_PYTHON% 1>&2
  echo   如果你把环境装在别处，用 RK_PYTHON 指定： 1>&2
  echo     set RK_PYTHON=C:\path\to\python.exe 1>&2
  exit /b 1
)

"%RK_PYTHON%" %*
