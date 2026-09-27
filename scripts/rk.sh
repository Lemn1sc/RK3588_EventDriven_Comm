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

# 自动发现解释器，**不硬编码用户名** —— 这是个公开仓库，别人 clone 下来也该能用。
#
# 两个坑：
#  1. Git Bash 的 $HOME 在有些机器上被改到别处（本机就是 /e/SPB_Data），
#     Windows 用户目录得从 $USERPROFILE 经 cygpath 转换才拿得到。
#  2. 环境可能在 conda 默认位置，也可能在 ProgramData；Windows 下解释器叫
#     python.exe，Linux/macOS 下叫 bin/python。
# 所以：把候选根目录都凑齐，逐个试。
ROOTS=("$HOME")
if [ -n "${USERPROFILE:-}" ]; then
  if command -v cygpath >/dev/null 2>&1; then
    ROOTS+=("$(cygpath -u "$USERPROFILE")")
  fi
  ROOTS+=("${USERPROFILE//\\//}")          # 退路：反斜杠换正斜杠
fi
ROOTS+=(/c/ProgramData /usr/local)

PY=""
for root in "${ROOTS[@]}"; do
  [ -n "$root" ] || continue
  for dist in miniconda3 anaconda3; do
    for exe in "$root/$dist/envs/rk/python.exe" "$root/$dist/envs/rk/bin/python"; do
      [ -x "$exe" ] && PY="$exe" && break 3
    done
  done
done
[ -n "${RK_PYTHON:-}" ] && [ -x "$RK_PYTHON" ] && PY="$RK_PYTHON"   # RK_PYTHON 优先

if [ -z "$PY" ]; then
  echo "找不到 rk 环境的解释器。试过的根目录：" >&2
  for r in "${ROOTS[@]}"; do [ -n "$r" ] && echo "  $r/{miniconda3,anaconda3}/envs/rk/" >&2; done
  echo "  → 用 RK_PYTHON 指定你的解释器：" >&2
  echo "      RK_PYTHON=/path/to/python.exe scripts/rk.sh <脚本> [参数]" >&2
  echo "  → 或参见 docs/04「运行环境」一节。" >&2
  exit 1
fi

exec "$PY" "$@"
