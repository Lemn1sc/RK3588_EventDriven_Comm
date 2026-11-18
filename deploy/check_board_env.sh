#!/bin/sh
# 板端环境自检（只读，不改系统）。
# 用法：sh check_board_env.sh > board_env.txt 2>&1
set -u

echo "===== 1. OS / kernel ====="
uname -a
if [ -r /etc/os-release ]; then
    sed -n '1,6p' /etc/os-release
fi

echo
echo "===== 2. Python / pip / numpy ====="
python3 --version 2>&1
python3 -m pip --version 2>&1
python3 -c 'import numpy; print("numpy", numpy.__version__)' 2>&1
python3 -c 'from rknnlite.api import RKNNLite; print("rknnlite import OK")' 2>&1

echo
echo "===== 3. NPU runtime ====="
for f in /usr/lib/librknnrt.so /usr/lib64/librknnrt.so /usr/bin/rknn_server; do
    if [ -e "$f" ]; then
        echo "-- $f"
        if command -v strings >/dev/null 2>&1; then
            strings "$f" | grep -i "version" | head -3
        else
            echo "   (strings not installed)"
        fi
    fi
done
echo "-- rknpu driver"
cat /sys/kernel/debug/rknpu/version 2>&1
echo "-- /dev/dri"
ls -l /dev/dri 2>&1

echo
echo "===== 4. Camera / V4L2 ====="
if command -v v4l2-ctl >/dev/null 2>&1; then
    v4l2-ctl --list-devices 2>&1
else
    echo "v4l2-ctl not installed (apt install v4l-utils)"
fi
ls -l /dev/video* 2>&1

echo
echo "===== 5. Camera app binaries ====="
for f in /home/camera/cameraApp /home/camera/camera_consumer; do
    if [ -e "$f" ]; then
        echo "-- $f"
        if command -v file >/dev/null 2>&1; then
            file "$f"
        fi
        ldd "$f" 2>&1 | grep -E "not found" || echo "   ldd: no missing libs"
    else
        echo "-- $f not found"
    fi
done

echo
echo "===== 6. Shared memory / socket ====="
ls -l /dev/shm/camera_shm 2>&1
ls -l /tmp/camera_shm.sock 2>&1

echo
echo "===== 7. Disk / model files ====="
df -h /home 2>&1
ls -lh /home/model 2>&1

echo
echo "===== done ====="
