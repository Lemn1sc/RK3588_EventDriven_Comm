#!/usr/bin/env bash
# 用项目的 rk 环境跑 Python 脚本（Git Bash 版）。
#
# 为什么需要这个：conda 不在 Git Bash 的 PATH 上，`conda activate rk` 会失败；
# 而系统默认的 python（Windows 商店版 3.14）没有 cv2 / torch / ultralytics，
# 直接 `python scripts/xxx.py` 会报 ModuleNotFoundError: No module named 'cv2'。
#
# 用法：
#   scripts/rk.sh scripts/demo_fall_detect.py --batch 5
#   scripts/rk.sh -c "import cv2; print(cv2.__version__)"
set -euo pipefail

PY="${RK_PYTHON:-/c/Users/Heshuai Xu/miniconda3/envs/rk/python.exe}"

if [ ! -x "$PY" ]; then
  echo "找不到 rk 环境的解释器：$PY" >&2
  echo "  如果你把环境装在别处，用 RK_PYTHON 指定：" >&2
  echo "    RK_PYTHON=/path/to/python.exe scripts/rk.sh <脚本> [参数]" >&2
  exit 1
fi

exec "$PY" "$@"
