#!/bin/sh
# 组装给队友的板端部署包。
# 前置：models/rknn/yolov8n_232.rknn 已转好；
#       deploy/wheels/ 下放了 cp311 aarch64 的 rknn_toolkit_lite2-2.3.2 wheel。
# 用法：sh deploy/make_package.sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PKG="$ROOT/deploy/pkg"
MODEL="${MODEL:-$ROOT/models/rknn/yolov8n_232.rknn}"
BUS="$ROOT/data/recordings/detect/bus.jpg"
WHEEL_DIR="$ROOT/deploy/wheels"

[ -f "$MODEL" ] || { echo "缺少模型：$MODEL"; exit 1; }
[ -f "$BUS" ] || { echo "缺少测试图：$BUS"; exit 1; }
if ! ls "$WHEEL_DIR"/rknn_toolkit_lite2-2.3.2-*aarch64.whl >/dev/null 2>&1; then
    echo "缺少板端 wheel：$WHEEL_DIR/rknn_toolkit_lite2-2.3.2-*aarch64.whl"
    exit 1
fi

rm -rf "$PKG"
mkdir -p "$PKG/wheels"
mkdir -p "$PKG/src/edge_node" "$PKG/src/common"

cp "$ROOT/deploy/README_board.md" "$ROOT/deploy/check_board_env.sh" "$PKG/"
cp "$ROOT/scripts/board_check_rknn.py" \
   "$ROOT/scripts/board_decode_yolo.py" \
   "$ROOT/scripts/board_camera_edge.py" \
   "$ROOT/scripts/demo_edge_shm.py" "$PKG/"
cp "$ROOT/src/edge_node/__init__.py" \
   "$ROOT/src/edge_node/event_tracker.py" \
   "$ROOT/src/edge_node/edge_shm.py" \
   "$ROOT/src/edge_node/edge_shm.h" \
   "$ROOT/src/edge_node/shm_publisher.py" "$PKG/src/edge_node/"
cp "$ROOT/src/common/__init__.py" \
   "$ROOT/src/common/event_vector.py" \
   "$ROOT/src/common/calibration.py" "$PKG/src/common/"
cp "$MODEL" "$PKG/"
cp "$BUS" "$PKG/bus.jpg"
cp "$WHEEL_DIR"/*.whl "$PKG/wheels/"

OUT="$ROOT/deploy/rk3588_deploy_$(date +%Y%m%d).tar.gz"
tar -czf "$OUT" -C "$PKG" .
echo "打包完成：$OUT"
echo "内容："
tar -tzf "$OUT"
