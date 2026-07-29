@echo off
REM ── T0.1 一键建环境（Windows）──────────────────────────────
REM 需先装好 Anaconda / Miniconda。双击或在命令行运行本脚本。
REM 完成后：conda activate rk && python scripts\check_env.py

setlocal
echo [1/3] 创建 conda 环境 rk (python=3.10) ...
call conda create -n rk python=3.10 -y || goto :err

echo [2/3] 安装依赖 (含 CUDA 版 torch) ...
call conda run -n rk pip install -r "%~dp0..\requirements.txt" || goto :err

echo [3/3] 环境自检 ...
call conda run -n rk python "%~dp0check_env.py"

echo.
echo 完成。下次使用请先执行:  conda activate rk
goto :eof

:err
echo.
echo 安装失败，请检查 conda 是否已安装并在 PATH 中。
exit /b 1
