#!/bin/sh
# 组装给控制端/传输端的接口包。
#
# 内容只包含**定义 + 写端参考实现**，不含读端/传输实现：
#   README.md                 交接说明（已定 / 待讨论 / 怎么验证）
#   edge_shm.h                本机共享内存布局（src 里的原文）
#   edge_wire.h               线上报文结构建议稿
#   check_edge_shm_layout.c   布局探针（sizeof/offsetof）
#   shm_contract.md           共享内存使用说明（面向读端，自然语言）
#   reference/edge_shm.py     写端实现，供对照字段语义
#
# 模型文件 yolov8n_232.rknn 单独发送，不打进包（体积大、且可能换版本）。
#
# 用法：sh handoff/make_handoff.sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PKG="$ROOT/handoff/pkg"
STAMP=$(date +%Y%m%d)

rm -rf "$PKG"
mkdir -p "$PKG"

cp "$ROOT/handoff/README.md" "$PKG/"
cp "$ROOT/handoff/edge_wire.h" "$PKG/"
cp "$ROOT/handoff/shm_contract.md" "$PKG/"
cp "$ROOT/src/edge_node/edge_shm.h" "$PKG/"
cp "$ROOT/scripts/check_edge_shm_layout.c" "$PKG/"

# 写端参考实现：给读端对照"写端到底保证了什么"，C 读端不依赖它
mkdir -p "$PKG/reference"
cp "$ROOT/src/edge_node/edge_shm.py" "$PKG/reference/"

OUT="$ROOT/handoff/edge_iface_${STAMP}.tar.gz"
tar -czf "$OUT" -C "$PKG" .
echo "打包完成：$OUT"
echo "内容："
tar -tzf "$OUT"
