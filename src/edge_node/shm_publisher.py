"""单写端发布器：POSIX SHM + flock；不依赖 numpy/RKNN。

默认固定使用 v1 的 480x640 BGR 布局。backing_path 仅供主机文件映射测试。
不自动删除已有对象；正常 close 标记 EXITED 并 unlink，已映射的读端仍可读到尾帧。
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import math
import mmap
import os
import struct
import time
from dataclasses import replace

from .edge_shm import (
    DEFAULT_LAYOUT, SHM_NAME, LOCK_PATH, REC_SIZE, IMG_HDR_SIZE, IMG_SLOT_NONE,
    Header, FrameRecord, FrameImageMeta, FrameFlags, ModelFlags, Detection, MAX_DET,
)


def _shm_function(name):
    for library in ("libc.so.6", "librt.so.1"):
        try:
            function = getattr(ctypes.CDLL(library, use_errno=True), name)
        except (OSError, AttributeError):
            continue
        if name == "shm_open":
            function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        else:
            function.argtypes = [ctypes.c_char_p]
        function.restype = ctypes.c_int
        return function
    raise RuntimeError("POSIX shared memory API unavailable")


def _shm_unlink(name):
    if _shm_function("shm_unlink")(os.fsencode(name)) < 0:
        error = ctypes.get_errno()
        if error != errno.ENOENT:
            raise OSError(error, os.strerror(error), name)


def make_detection_record(frame_id, boxes, confs, classes, width, height,
                          event_vector=None, capture_ts_ns=0, infer_ms=0.0):
    """像素 xyxy -> 原图归一化记录；event_vector 由现有检测事件模块提供。

    不生成 EventTracker 事件：best_event_id=0，reliability=-1，事件表保持空。
    """
    if width <= 0 or height <= 0 or not (len(boxes) == len(confs) == len(classes)):
        raise ValueError("invalid detection dimensions/counts")
    detections = []
    for box, confidence, cls in zip(boxes, confs, classes):
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in (*box, confidence)):
            raise ValueError("invalid detection values")
        if not 0 <= float(confidence) <= 1 or not 0 <= int(cls) <= 65535:
            raise ValueError("invalid detection class/confidence")
        x1, y1, x2, y2 = (float(x) for x in box)
        if not 0 <= x1 <= x2 <= width or not 0 <= y1 <= y2 <= height:
            raise ValueError("detection outside original image")
        detections.append(Detection(int(cls), float(confidence),
                                    x1 / width, y1 / height, x2 / width, y2 / height))
    people = sum(d.cls == 0 for d in detections)
    if len(detections) > 65535:
        raise ValueError("too many detections for v1")
    detections.sort(key=lambda d: (d.cls != 0, -d.conf))
    if event_vector is None and people:
        primary = detections[0]
        ev = (1.0, primary.conf, *primary.bbox, 1.0, 0.0)
    else:
        ev = tuple(float(x) for x in event_vector) if event_vector is not None else (0.0,) * 8
    if len(ev) != 8 or not all(math.isfinite(x) for x in ev):
        raise ValueError("event_vector must contain 8 finite values")
    record = FrameRecord(0, int(frame_id), int(capture_ts_ns), 0,
                         infer_ms=float(infer_ms), flags=int(FrameFlags.INFERRED),
                         n_person=people, det_total=len(detections),
                         dets=tuple(detections[:MAX_DET]), event_vector=ev)
    if not people:
        record.flags |= FrameFlags.NO_PERSON
    if capture_ts_ns:
        record.flags |= FrameFlags.TS_FROM_CAMERA
    if ev[6] >= 0.5:
        x1, y1, x2, y2 = ev[2:6]
        record.cx, record.cy = (x1 + x2) / 2, (y1 + y2) / 2
        record.w, record.h = max(0, x2 - x1), max(0, y2 - y1)
        record.area_ratio = record.w * record.h
        record.aspect = record.h * height / max(record.w * width, 1e-6)
    return record


class ShmPublisher:
    def __init__(self, shm_name=SHM_NAME, lock_path=LOCK_PATH, backing_path=None):
        if not shm_name.startswith("/") or "/" in shm_name[1:]:
            raise ValueError("shm_name must be /name")
        self.shm_name = shm_name
        self.lock_path = os.fspath(lock_path)
        self.backing_path = os.fspath(backing_path) if backing_path is not None else None
        self.layout = DEFAULT_LAYOUT
        self.header = Header.for_layout()
        self.mm = None
        self._fd = self._lock_fd = self._owner_fd = None
        self._created = False

    def open(self):
        if self.mm is not None:
            raise RuntimeError("publisher already open")
        try:
            # 生命周期锁与短时数据锁分开；锁文件保留，避免替换 inode 破坏互斥。
            self._owner_fd = os.open(self.lock_path + ".writer", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self._owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
                if self.backing_path is not None:
                    self._fd = os.open(self.backing_path, flags, 0o600)
                else:
                    self._fd = _shm_function("shm_open")(os.fsencode(self.shm_name), flags, 0o600)
                    if self._fd < 0:
                        error = ctypes.get_errno()
                        self._fd = None
                        raise OSError(error, os.strerror(error), self.shm_name)
                self._created = True
                os.ftruncate(self._fd, self.layout.total_size)
                self.mm = mmap.mmap(self._fd, self.layout.total_size)
                # 新建文件/SHM 的扩展区域为零，事件表初始为空。
                self.header = Header.for_layout()
                self.header.flags = ModelFlags.READY
                self.header.last_update_ts_ns = time.monotonic_ns()
                self.header.pack_into(self.mm)
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        except BaseException:
            self.close()
            raise
        return self

    def _unlink(self):
        if self.backing_path is not None:
            try:
                os.unlink(self.backing_path)
            except FileNotFoundError:
                pass
        else:
            _shm_unlink(self.shm_name)

    @classmethod
    def clean(cls, shm_name=SHM_NAME, lock_path=LOCK_PATH, backing_path=None):
        """显式清理崩溃残留；活跃写端持有生命周期锁时拒绝清理。"""
        probe = cls(shm_name, lock_path, backing_path)
        owner = os.open(probe.lock_path + ".writer", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            probe._unlink()
        finally:
            os.close(owner)

    def update_counters(self, frames_ready, frames_inferred):
        if self.mm is None:
            raise RuntimeError("publisher not open")
        if not self.header.frames_ready <= frames_ready < 2**64 or not self.header.frames_inferred <= frames_inferred <= frames_ready:
            raise ValueError("invalid/nonmonotonic counters")
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            self.header.frames_ready = frames_ready
            self.header.frames_inferred = frames_inferred
            self.header.pack_into(self.mm)
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def publish(self, record, bgr=None, started_ns=None, error_code=0):
        """发布一次推理尝试；失败帧用 BAD、无图。返回发布号，不修改调用方 record。

        bgr 必须为连续 uint8 480x640x3，或相同大小的原始字节 buffer。
        先在私有缓冲区打包，参数错误不会覆盖 SHM 中的旧图/记录。
        """
        if self.mm is None:
            raise RuntimeError("publisher not open")
        if self.header.pub_seq == 2**64 - 1:
            raise OverflowError("publication sequence exhausted")
        if record.flags & FrameFlags.BAD_FRAME and bgr is not None:
            raise ValueError("BAD frame must not carry an image")
        raw = None
        if bgr is not None:
            view = memoryview(bgr)
            if not view.c_contiguous or view.itemsize != 1 or view.format != "B":
                raise ValueError("BGR must be contiguous uint8")
            if view.ndim != 1 and view.shape != (self.layout.img_h, self.layout.img_w, 3):
                raise ValueError("v1 requires BGR shape (640,480,3)")
            raw = view.cast("B")
            if len(raw) != self.layout.img_size:
                raise ValueError("v1 requires 480x640 BGR")
        floats = [record.infer_ms, *record.event_vector, record.aspect, record.cx, record.cy,
                  record.w, record.h, record.area_ratio, record.best_conf, record.best_reliability]
        floats += [x for d in record.dets for x in (d.conf, *d.bbox)]
        if len(record.event_vector) != 8 or not all(math.isfinite(float(x)) for x in floats):
            raise ValueError("record contains invalid floats")
        slot = self.header.img_seq % self.layout.n_img if raw is not None else IMG_SLOT_NONE
        now = time.monotonic_ns()
        if started_ns is not None and not 0 <= started_ns <= now:
            raise ValueError("invalid processing start timestamp")
        current = replace(record, seq=self.header.pub_seq + 1, publish_ts_ns=now, img_slot=slot,
                          infer_ms=(now - started_ns) / 1e6 if started_ns is not None else record.infer_ms)
        packed = bytearray(REC_SIZE)
        current.pack_into(packed)
        image_meta = bytearray(IMG_HDR_SIZE)
        if raw is not None:
            FrameImageMeta(self.header.img_seq + 1, current.frame_id, current.capture_ts_ns,
                           self.layout.img_w, self.layout.img_h).pack_into(image_meta)
        next_header = replace(self.header, pub_seq=current.seq, last_update_ts_ns=now,
                              frames_bad=self.header.frames_bad + bool(current.flags & FrameFlags.BAD_FRAME),
                              last_error=int(error_code))
        next_header.frames_inferred = max(next_header.frames_inferred, current.seq)
        next_header.frames_ready = max(next_header.frames_ready, next_header.frames_inferred)
        if raw is not None:
            next_header.img_seq += 1
            next_header.img_overwrites += self.header.img_seq >= self.layout.n_img
        header_bytes = bytearray(128)
        next_header.pack_into(header_bytes)  # 计数等范围也在写 SHM 前检查。
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            if raw is not None:
                offset = self.layout.img_offset(slot)
                self.mm[offset:offset + IMG_HDR_SIZE] = image_meta
                self.mm[offset + IMG_HDR_SIZE:offset + self.layout.img_slot_size] = raw
            offset = self.layout.rec_offset(current.seq)
            # 图片复制和等待数据锁也属于取帧到发布的处理时间。
            committed_ns = time.monotonic_ns()
            struct.pack_into("<Q", packed, 24, committed_ns)
            if started_ns is not None:
                struct.pack_into("<f", packed, 32, (committed_ns - started_ns) / 1e6)
            next_header.last_update_ts_ns = committed_ns
            struct.pack_into("<Q", header_bytes, 120, committed_ns)
            self.mm[offset:offset + REC_SIZE] = packed
            # 同锁下更新所有计数，pub_seq 最后提交。
            struct.pack_into("<Q", header_bytes, 64, self.header.pub_seq)
            self.mm[:128] = header_bytes
            self.header = next_header
            struct.pack_into("<Q", self.mm, 64, current.seq)
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        return current.seq

    def close(self):
        try:
            if self.mm is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
                    self.header.flags = ModelFlags.EXITED
                    self.header.pack_into(self.mm)
                finally:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    self.mm.close()
                    self.mm = None
        finally:
            try:
                if self._created:
                    self._unlink()
                    self._created = False
            finally:
                for attr in ("_fd", "_lock_fd", "_owner_fd"):
                    fd = getattr(self, attr)
                    if fd is not None:
                        os.close(fd)
                        setattr(self, attr, None)

    def __enter__(self):
        return self.open()

    def __exit__(self, *_exc):
        self.close()
