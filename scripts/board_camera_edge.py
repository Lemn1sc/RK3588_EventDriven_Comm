#!/usr/bin/env python3
"""板端摄像头 → RKNN 检测闭环（Python consumer）。

配合 teammates 的 `cameraApp` 使用：

  终端 A: ./cameraApp 640 480 100
  终端 B: python3 board_camera_edge.py --model yolov8n.rknn --stride 2

它做：
  Unix socket (SOCK_SEQPACKET) 收 READY/STREAM_INFO/END
  → 读 /camera_shm 里的 NV12
  → 按 wstride/hstride 处理 stride
  → NV12 → BGR
  → letterbox 640×640 → RGB 0~255 NHWC
  → RKNN 推理 → 9 输出 DFL 解码 + NMS
  → 框映射回摄像头原图 → 打印/保存标注图
  → RELEASE

--stride 2 表示每 2 帧推理一次（模型约 15 FPS，摄像头约 30 FPS）；
未推理的帧仍然要 RELEASE，否则 cameraApp 会阻塞。

用法：
    python3 board_camera_edge.py --model yolov8n.rknn --stride 2
    python3 board_camera_edge.py --model yolov8n.rknn --stride 3 --out-dir camera_out
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import io
import json
import mmap
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

SHM_NAME = "/camera_shm"
SOCKET_PATH = "/tmp/camera_shm.sock"
SHM_MAGIC = 0x43414D31
SHM_VERSION = 1
NV12_FOURCC = 0x3231564E       # V4L2_PIX_FMT_NV12
SLOT_HEADER = 48
MSG_SIZE = 20

SHM_FRAME_READY = 1
SHM_RELEASE = 2
SHM_STREAM_INFO = 3
SHM_END = 4
STATUS_OK = 0
STATUS_BAD_DATA = 1


def _shm_open_readonly(name: str) -> int:
    """ctypes 调 shm_open；Buildroot/glibc 可能把它放在 libc 或 librt。"""
    for libname in ("libc.so.6", "librt.so.1"):
        try:
            lib = ctypes.CDLL(libname, use_errno=True)
            fn = lib.shm_open
        except (OSError, AttributeError):
            continue
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        fn.restype = ctypes.c_int
        fd = fn(name.encode(), os.O_RDONLY, 0)
        if fd >= 0:
            return int(fd)
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, os.strerror(err), name)
    raise OSError(errno.ENOENT, "shm_open failed", name)


def recv_msg(sock: socket.socket):
    data = sock.recv(MSG_SIZE)
    if len(data) != MSG_SIZE:
        raise ConnectionError(f"socket closed / short message: {len(data)} bytes")
    return struct.unpack(">5I", data)


def send_msg(sock: socket.socket, typ, slot_id, frame_id, data_size, status):
    sock.sendall(struct.pack(">5I", typ, slot_id, frame_id, data_size, status))


def slot_offsets(mm, slot_id: int):
    magic, ver, n_slots, cap = struct.unpack_from("<IIII", mm, 0)
    if magic != SHM_MAGIC or ver != SHM_VERSION:
        raise SystemExit(f"共享内存不匹配：magic={magic:#x} version={ver}")
    if not (0 <= slot_id < n_slots):
        raise ValueError(f"slot_id={slot_id} 越界（n_slots={n_slots}）")
    slot_size = SLOT_HEADER + cap
    off = 16 + slot_id * slot_size
    return off, cap


def read_slot_header(mm, off: int):
    # 8×uint32 到 hstride；timestamp_flags 在 32；uint64 在 40；data 在 48
    fid, size, seq, fmt, w, h, wstride, hstride = struct.unpack_from("<8I", mm, off)
    ts_flags = struct.unpack_from("<I", mm, off + 32)[0]
    ts_ns = struct.unpack_from("<Q", mm, off + 40)[0]
    return dict(frame_id=fid, data_size=size, source=seq, fmt=fmt,
                width=w, height=h, wstride=wstride, hstride=hstride,
                ts_flags=ts_flags, ts_ns=ts_ns)


def nv12_to_bgr(mm, off: int, meta: dict, swap_uv: bool = False) -> np.ndarray:
    """按 stride 取 NV12 → BGR。有效宽高是 width/height，存储跨度为 wstride/hstride。

    swap_uv=True 时按 NV21（V/U 顺序相反）解释，用来排查色彩异常。
    """
    import cv2

    w, h = meta["width"], meta["height"]
    ws, hs = meta["wstride"], meta["hstride"]
    if w <= 0 or h <= 0 or ws < w or hs < h or h % 2:
        raise ValueError(f"非法 NV12 布局：{w}x{h} stride={ws}x{hs}")
    required = ws * hs + ws * (hs // 2)
    if meta["data_size"] < required:
        raise ValueError(f"data_size={meta['data_size']} < 需要 {required}")

    data_off = off + SLOT_HEADER
    y = np.frombuffer(mm, np.uint8, count=ws * hs, offset=data_off).reshape(hs, ws)
    uv = np.frombuffer(mm, np.uint8, count=ws * (hs // 2),
                       offset=data_off + ws * hs).reshape(hs // 2, ws)

    compact = np.empty((h * 3 // 2, w), np.uint8)
    compact[:h] = y[:h, :w]
    compact[h:] = uv[:h // 2, :w]
    code = cv2.COLOR_YUV2BGR_NV21 if swap_uv else cv2.COLOR_YUV2BGR_NV12
    return cv2.cvtColor(compact, code)


def rotate_bgr(bgr: np.ndarray, angle: int) -> np.ndarray:
    """按顺时针角度旋转 BGR。用于排查 RGA 旋转方向与物理安装是否匹配。"""
    import cv2

    if angle == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def letterbox_rgb(bgr: np.ndarray, size: int = 640):
    import cv2

    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32)  # BGR→RGB, 0~255
    return rgb[None], (r, left, top)


def decode(outs, conf_thres=0.25, iou_thres=0.7):
    """9 输出 → 640 letterbox 坐标下的 xyxy / conf / cls。"""
    import cv2

    by_hw: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for o in outs:
        a = np.asarray(o)[0] if np.asarray(o).ndim == 4 else np.asarray(o)
        if a.ndim != 3:
            continue
        c, h, w = a.shape
        slot = by_hw.setdefault((h, w), {})
        if c == 64:
            slot["box"] = a
        elif c == 80:
            slot["score"] = a

    boxes, confs, clss = [], [], []
    for (h, _w), slot in sorted(by_hw.items(), key=lambda kv: -kv[0][0]):
        if "box" not in slot or "score" not in slot:
            continue
        stride = 640 // h
        box = slot["box"].reshape(4, 16, h, _w)
        b = np.exp(box - box.max(1, keepdims=True))
        b /= b.sum(1, keepdims=True)
        d = (b * np.arange(16).reshape(1, 16, 1, 1)).sum(1)
        gy, gx = np.meshgrid(np.arange(h), np.arange(_w), indexing="ij")
        score = slot["score"]
        c = score.max(0)
        m = c > conf_thres
        if not m.any():
            continue
        boxes.append(np.stack([
            (gx + .5 - d[0])[m] * stride, (gy + .5 - d[1])[m] * stride,
            (gx + .5 + d[2])[m] * stride, (gy + .5 + d[3])[m] * stride], 1))
        confs.append(c[m])
        clss.append(score.argmax(0)[m])

    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    B = np.concatenate(boxes); C = np.concatenate(confs); K = np.concatenate(clss)
    nms_boxes = np.stack([B[:, 0], B[:, 1], B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]], 1)
    keep = [int(i) for i in cv2.dnn.NMSBoxes(nms_boxes.tolist(), C.tolist(), conf_thres, iou_thres)]
    return B[keep], C[keep], K[keep]


def scale_boxes(B, orig_w, orig_h, size, r, left, top):
    if not len(B):
        return B
    B = B.copy()
    B[:, [0, 2]] = (B[:, [0, 2]] - left) / r
    B[:, [1, 3]] = (B[:, [1, 3]] - top) / r
    B[:, [0, 2]] = np.clip(B[:, [0, 2]], 0, orig_w - 1)
    B[:, [1, 3]] = np.clip(B[:, [1, 3]], 0, orig_h - 1)
    return B


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("yolov8n.rknn"))
    p.add_argument("--stride", type=int, default=2,
                   help="每 N 帧推理一次；其它帧快速 RELEASE")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--core-mask", choices=["auto", "0", "1", "2", "0_1_2"],
                   default="0_1_2")
    p.add_argument("--out-dir", type=Path, default=Path("camera_out"))
    p.add_argument("--save-every", type=int, default=10,
                   help="每 N 个推理帧保存一张标注图；0=不保存")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                   help="在 NV12→BGR 后顺时针旋转，用来排查画面方向")
    p.add_argument("--swap-uv", action="store_true",
                   help="按 NV21 解释 UV，用来排查颜色异常（U/V 反了）")
    p.add_argument("--save-raw", action="store_true",
                   help="保存标注图时同时保存一张未画框的原图")
    p.add_argument("--dump-nv12", type=Path, default=None,
                   help="把第一帧槽数据原样存文件（附 json 元数据），供 PC 端排查")
    args = p.parse_args()

    try:
        from rknnlite.api import RKNNLite
    except Exception as exc:
        raise SystemExit(f"无法 import rknnlite：{exc}")
    if not args.model.exists():
        raise SystemExit(f"模型不存在：{args.model}")

    # 连接 cameraApp
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        sock.connect(SOCKET_PATH)
    except OSError as exc:
        raise SystemExit(f"连接 {SOCKET_PATH} 失败：{exc}\n先启动 ./cameraApp W H N")

    fd = _shm_open_readonly(SHM_NAME)
    st = os.fstat(fd)
    mm = mmap.mmap(fd, st.st_size, access=mmap.ACCESS_READ)
    os.close(fd)

    # STREAM_INFO
    typ, slot_id, total, data_size, status = recv_msg(sock)
    if typ != SHM_STREAM_INFO or total == 0:
        raise SystemExit(f"非法 STREAM_INFO：type={typ} total={total}")
    print(f"[stream] total={total}")

    # 加载模型
    rknn = RKNNLite()
    if rknn.load_rknn(str(args.model)) != 0:
        raise SystemExit("load_rknn 失败")
    masks = {
        "auto": getattr(RKNNLite, "NPU_CORE_AUTO", None),
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2,
    }
    mask = masks[args.core_mask]
    ret = rknn.init_runtime(core_mask=mask) if mask is not None else rknn.init_runtime()
    if ret != 0:
        raise SystemExit(f"init_runtime 失败：{ret}")
    print(f"[model] {args.model} stride={args.stride} core={args.core_mask} "
          f"rotate={args.rotate} swap_uv={args.swap_uv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    received = skipped = inferred = saved = 0
    infer_ms = []
    try:
        while True:
            typ, slot_id, frame_id, msg_size, status = recv_msg(sock)
            if typ == SHM_END:
                if slot_id or frame_id or msg_size or status:
                    print("[!] END 字段非零")
                break
            if typ != SHM_FRAME_READY:
                raise SystemExit(f"未知消息 type={typ}")

            ok = True
            try:
                off, cap = slot_offsets(mm, slot_id)
                meta = read_slot_header(mm, off)
                if (meta["frame_id"] != frame_id or meta["data_size"] != msg_size
                        or msg_size == 0 or msg_size > cap
                        or meta["fmt"] != NV12_FOURCC):
                    raise ValueError(f"READY/槽不匹配：{meta}")
                received += 1

                # 只在需要推理的帧上做 NV12→BGR；其它帧直接归还。
                if frame_id % max(1, args.stride) == 0:
                    t0 = time.perf_counter()
                    if args.dump_nv12 is not None and inferred == 0:
                        args.dump_nv12.parent.mkdir(parents=True, exist_ok=True)
                        start = off + SLOT_HEADER
                        args.dump_nv12.write_bytes(bytes(mm[start:start + meta["data_size"]]))
                        args.dump_nv12.with_suffix(".json").write_text(
                            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                        print(f"[dump] {args.dump_nv12} ({meta['data_size']} B) + "
                              f"{args.dump_nv12.with_suffix('.json')}")
                    bgr = nv12_to_bgr(mm, off, meta, args.swap_uv)
                    if args.rotate:
                        bgr = rotate_bgr(bgr, args.rotate)
                    rot_h, rot_w = bgr.shape[:2]
                    if inferred == 0:
                        mean_bgr = bgr.mean(axis=(0, 1))
                        print(f"[img] shape={rot_w}x{rot_h} "
                              f"mean B/G/R={mean_bgr[0]:.1f}/{mean_bgr[1]:.1f}/"
                              f"{mean_bgr[2]:.1f} min={bgr.min()} max={bgr.max()} "
                              f"shm={meta['width']}x{meta['height']} "
                              f"stride={meta['wstride']}x{meta['hstride']}")
                    rgb, (r, left, top) = letterbox_rgb(bgr, 640)
                    outs = rknn.inference(inputs=[rgb])
                    B, C, K = decode(outs, args.conf, args.iou)
                    B = scale_boxes(B, rot_w, rot_h, 640, r, left, top)
                    t_ms = (time.perf_counter() - t0) * 1000.0
                    infer_ms.append(t_ms)
                    inferred += 1
                    print(f"[infer] frame={frame_id} boxes={len(B)} "
                          f"conf={C.max() if len(C) else 0:.3f} "
                          f"total={t_ms:.1f}ms size={rot_w}x{rot_h}")
                    for i, (b, c, k) in enumerate(zip(B, C, K)):
                        if i >= 8:
                            break
                        x1, y1, x2, y2 = (int(v) for v in b)
                        print(f"   cls={int(k)} conf={c:.3f} xyxy=({x1},{y1},{x2},{y2})")
                    if args.save_every and (inferred - 1) % args.save_every == 0:
                        import cv2
                        raw = bgr.copy()
                        for b, c, k in zip(B, C, K):
                            x1, y1, x2, y2 = (int(v) for v in b)
                            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(bgr, f"{int(k)}:{c:.2f}", (x1, max(15, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        out = args.out_dir / f"frame_{frame_id:06d}.jpg"
                        cv2.imwrite(str(out), bgr)
                        if args.save_raw:
                            cv2.imwrite(str(args.out_dir / f"raw_{frame_id:06d}.jpg"), raw)
                        saved += 1
            except Exception as exc:
                ok = False
                skipped += 1
                print(f"[!] frame={frame_id} 处理失败：{type(exc).__name__}: {exc}")

            send_msg(sock, SHM_RELEASE, slot_id, frame_id, msg_size,
                     STATUS_OK if ok else STATUS_BAD_DATA)
    finally:
        try:
            rknn.release()
        except Exception:
            pass
        mm.close()
        sock.close()

    ms = np.array(infer_ms) if infer_ms else np.array([0.0])
    print(f"\n[END] total={total} received={received} bad={skipped} "
          f"inferred={inferred} saved={saved}")
    print(f"[latency] infer+pre+decode mean={ms.mean():.1f}ms "
          f"median={np.median(ms):.1f}ms")
    return 0 if received + skipped == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
