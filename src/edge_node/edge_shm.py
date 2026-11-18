"""模型 ↔ 传输 共享内存的字节布局（纯标准库）。

契约文档：**docs/10_模型传输共享内存契约.md**。本文件是该文档的可执行版本。

── 为什么单独一个模块、而且不依赖 numpy ──

写端（模型进程）和读端（GRU / 传输进程）必须用**同一份**结构体定义：偏移错一个字节，
读到的就是垃圾，而且**不会报错**。所以这里只做三件事：

1. 定义常量与布局（`Layout`），并给出 `rec_offset` / `ev_offset` / `img_offset`；
2. 定义四个结构体（Header / FrameRecord / EventSnapshot / FrameImageMeta），
   都带 `pack_into` / `unpack_from`；
3. 在 import 时用 `struct.calcsize` 断言尺寸 —— 布局漂移第一时间炸出来。

刻意只用标准库：读端可能用纯 Python 实现；图像数据以 `memoryview` 暴露，需要 numpy
视图时调用方自己 `np.frombuffer(mm, np.uint8, count=img_size, offset=...)`。
这样这个文件也可以原样拷给 C++ 侧当参照（字段顺序即结构体顺序）。

── 字段偏移从哪来 ──

代码里**没有手写的偏移数字**，全部由格式串推导（`_REC_PREFIX.size` 等）。
文档里的偏移表与这里的格式串是同一份事实；改一边改另一边，否则 `check_edge_shm.py` 会炸。

── 约定（详见 docs/10 §7）──

- 时间：`CLOCK_MONOTONIC` 纳秒；
- 坐标：归一化到**旋转后的摄像头原图**（当前 480×640），不是 letterbox 的 640×640；
- 禁止 NaN / Inf；无框 → `dets=()`，无事件 → `best_event_id=0` 且 `best_*` 清零；
- `n_det` 由 `len(dets)` 推导，不是独立字段 —— 从根上消除"计数与实际不符"。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Sequence

__all__ = [
    "SHM_NAME", "LOCK_PATH", "MAGIC", "VERSION",
    "HDR_SIZE", "REC_SIZE", "EV_SIZE", "IMG_HDR_SIZE", "MAX_DET", "IMG_SLOT_NONE",
    "N_REC", "N_EV", "N_IMG", "DEFAULT_IMG_W", "DEFAULT_IMG_H",
    "FrameFlags", "ModelFlags",
    "Layout", "DEFAULT_LAYOUT",
    "Header", "Detection", "FrameRecord", "EventSnapshot", "FrameImageMeta",
    "validate_header", "image_payload",
]

# ── 常量 ──────────────────────────────────────────────────────────────

SHM_NAME = "/edge_infer_shm"
LOCK_PATH = "/tmp/edge_infer.lock"

MAGIC = 0x45444731          # "EDG1"
VERSION = 1

HDR_SIZE = 128
REC_SIZE = 512
EV_SIZE = 192
IMG_HDR_SIZE = 32

N_REC = 64
N_EV = 16
N_IMG = 8

MAX_DET = 6
IMG_SLOT_NONE = 0xFFFF

DEFAULT_IMG_W = 480
DEFAULT_IMG_H = 640

# kind / state / urgency 是 EventKind / EventState / Urgency 的数值：
#   kind   0=空槽, 1=PERSON, 2=PERSON_ABNORMAL, 3=TRACK_LOST,
#          4=SUSPECTED_FALL, 5=FALL_CONFIRMED, 6=BACKGROUND
#   state  0=CANDIDATE, 1=ACTIVE, 2=CONFIRMED, 3=EXPIRED
#   urgency 0=BACKGROUND, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL
# 这里不 import event_tracker：本模块要被传输端原样复用，不该拖上层依赖。


class FrameFlags(IntFlag):
    INFERRED = 1 << 0
    BAD_FRAME = 1 << 1
    NO_PERSON = 1 << 2
    TS_FROM_CAMERA = 1 << 3
    TRACKER_ERROR = 1 << 4


class ModelFlags(IntFlag):
    READY = 1 << 0
    EXITED = 1 << 1


# ── 格式串（唯一事实来源）────────────────────────────────────────────

_DET_FMT = "<HHfffff"                                  # cls, pad, conf, x1, y1, x2, y2
_REC_PREFIX_FMT = "<QQQQfIHHHH" "8f" "6f" "I" "4B" "ff" "Q"
_EV_PREFIX_FMT = "<QI4BQQQfffI4fII"
_IMG_FMT = "<QQQII"
_HDR_FMT = "<IIQ" + "I" * 10 + "II" + "Q" * 6 + "II" + "Q"

# 完整的 REC 格式串（含 dets 与 reserved），仅供尺寸断言与交叉核对。
REC_FMT = _REC_PREFIX_FMT + _DET_FMT[1:] * MAX_DET + str(REC_SIZE - 272) + "x"

_DET = struct.Struct(_DET_FMT)
_REC_PREFIX = struct.Struct(_REC_PREFIX_FMT)
_EV_PREFIX = struct.Struct(_EV_PREFIX_FMT)
_IMG = struct.Struct(_IMG_FMT)
_HDR = struct.Struct(_HDR_FMT)

assert _DET.size == 24, _DET.size
assert _REC_PREFIX.size == 128, _REC_PREFIX.size
assert struct.calcsize(REC_FMT) == REC_SIZE, struct.calcsize(REC_FMT)
assert _EV_PREFIX.size == 80, _EV_PREFIX.size
assert _IMG.size == IMG_HDR_SIZE, _IMG.size
assert _HDR.size == HDR_SIZE, _HDR.size

DET_OFF = _REC_PREFIX.size                  # dets 在记录内的偏移（128）
REC_RESERVED_OFF = DET_OFF + MAX_DET * _DET.size   # 272
EV_RESERVED_OFF = _EV_PREFIX.size           # 80

_ZERO_DET = (0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
_ZERO_REC_RESERVED = bytes(REC_SIZE - REC_RESERVED_OFF)
_ZERO_EV_RESERVED = bytes(EV_SIZE - EV_RESERVED_OFF)


def _align64(n: int) -> int:
    return (n + 63) & ~63


# ── 布局 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Layout:
    """各段的起始偏移与总长。默认值即 docs/10 §1 的定稿布局。"""

    n_rec: int = N_REC
    n_ev: int = N_EV
    n_img: int = N_IMG
    img_w: int = DEFAULT_IMG_W
    img_h: int = DEFAULT_IMG_H

    @property
    def off_rec(self) -> int:
        return HDR_SIZE

    @property
    def off_ev(self) -> int:
        return self.off_rec + self.n_rec * REC_SIZE

    @property
    def off_img(self) -> int:
        return _align64(self.off_ev + self.n_ev * EV_SIZE)

    @property
    def img_stride(self) -> int:
        return self.img_w * 3

    @property
    def img_size(self) -> int:
        return self.img_stride * self.img_h

    @property
    def img_slot_size(self) -> int:
        return IMG_HDR_SIZE + self.img_size

    @property
    def total_size(self) -> int:
        return self.off_img + self.n_img * self.img_slot_size

    # ── 序号 → 偏移 ──
    # 注意三者的"取模口径"不同：记录用 (seq-1)，事件和帧环直接用自身计数。

    def rec_offset(self, seq: int) -> int:
        """第 seq 条记录（从 1 开始）所在槽位的偏移。"""
        if seq < 1:
            raise ValueError(f"seq 从 1 开始，收到 {seq}")
        return self.off_rec + ((seq - 1) % self.n_rec) * REC_SIZE

    def ev_offset(self, event_id: int) -> int:
        """事件 id 对应槽位；event_id=0 也能算，但语义上是空槽。"""
        if event_id < 0:
            raise ValueError(f"event_id 不能为负：{event_id}")
        return self.off_ev + (event_id % self.n_ev) * EV_SIZE

    def img_offset(self, slot: int) -> int:
        """帧环槽位索引（= img_seq % n_img）。"""
        if not 0 <= slot < self.n_img:
            raise ValueError(f"slot={slot} 越界（n_img={self.n_img}）")
        return self.off_img + slot * self.img_slot_size

    @classmethod
    def from_header(cls, header: "Header") -> "Layout":
        return cls(n_rec=header.n_rec, n_ev=header.n_ev, n_img=header.n_img,
                   img_w=header.img_w, img_h=header.img_h)


DEFAULT_LAYOUT = Layout()


# ── Header ────────────────────────────────────────────────────────────


@dataclass
class Header:
    magic: int = MAGIC
    version: int = VERSION
    total_size: int = 0
    n_rec: int = N_REC
    rec_size: int = REC_SIZE
    n_ev: int = N_EV
    ev_size: int = EV_SIZE
    n_img: int = N_IMG
    img_w: int = DEFAULT_IMG_W
    img_h: int = DEFAULT_IMG_H
    img_stride: int = DEFAULT_IMG_W * 3
    img_size: int = DEFAULT_IMG_W * DEFAULT_IMG_H * 3
    flags: int = 0
    pad0: int = 0
    pad1: int = 0
    pub_seq: int = 0
    img_seq: int = 0
    frames_ready: int = 0
    frames_inferred: int = 0
    frames_bad: int = 0
    img_overwrites: int = 0
    last_error: int = 0
    pad2: int = 0
    last_update_ts_ns: int = 0

    @classmethod
    def for_layout(cls, layout: Layout = DEFAULT_LAYOUT) -> "Header":
        """写端启动时用：把尺寸字段一次填对。"""
        return cls(
            total_size=layout.total_size,
            n_rec=layout.n_rec, rec_size=REC_SIZE,
            n_ev=layout.n_ev, ev_size=EV_SIZE,
            n_img=layout.n_img,
            img_w=layout.img_w, img_h=layout.img_h,
            img_stride=layout.img_stride, img_size=layout.img_size,
        )

    @property
    def layout(self) -> Layout:
        return Layout.from_header(self)

    def pack_into(self, buf, offset: int = 0) -> None:
        _HDR.pack_into(
            buf, offset,
            self.magic, self.version, self.total_size,
            self.n_rec, self.rec_size, self.n_ev, self.ev_size, self.n_img,
            self.img_w, self.img_h, self.img_stride, self.img_size,
            int(self.flags), self.pad0, self.pad1,
            self.pub_seq, self.img_seq, self.frames_ready, self.frames_inferred,
            self.frames_bad, self.img_overwrites,
            self.last_error, self.pad2, self.last_update_ts_ns,
        )

    @classmethod
    def unpack_from(cls, buf, offset: int = 0) -> "Header":
        v = _HDR.unpack_from(buf, offset)
        return cls(
            magic=v[0], version=v[1], total_size=v[2],
            n_rec=v[3], rec_size=v[4], n_ev=v[5], ev_size=v[6], n_img=v[7],
            img_w=v[8], img_h=v[9], img_stride=v[10], img_size=v[11],
            flags=v[12], pad0=v[13], pad1=v[14],
            pub_seq=v[15], img_seq=v[16], frames_ready=v[17], frames_inferred=v[18],
            frames_bad=v[19], img_overwrites=v[20],
            last_error=v[21], pad2=v[22], last_update_ts_ns=v[23],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "magic": f"{self.magic:#x}", "version": self.version,
            "total_size": self.total_size, "n_rec": self.n_rec, "n_ev": self.n_ev,
            "n_img": self.n_img, "img": f"{self.img_w}x{self.img_h}",
            "flags": int(self.flags), "pub_seq": self.pub_seq, "img_seq": self.img_seq,
            "frames_ready": self.frames_ready, "frames_inferred": self.frames_inferred,
            "frames_bad": self.frames_bad, "img_overwrites": self.img_overwrites,
            "last_error": self.last_error,
            "last_update_ts_ns": self.last_update_ts_ns,
        }


def validate_header(header: Header) -> list[str]:
    """读端启动校验。返回问题列表，空列表表示可以开始读。"""
    errs: list[str] = []
    if header.magic != MAGIC:
        errs.append(f"magic={header.magic:#x}，期望 {MAGIC:#x}")
    if header.version != VERSION:
        errs.append(f"version={header.version}，期望 {VERSION}")
    if header.rec_size != REC_SIZE:
        errs.append(f"rec_size={header.rec_size}，期望 {REC_SIZE}")
    if header.ev_size != EV_SIZE:
        errs.append(f"ev_size={header.ev_size}，期望 {EV_SIZE}")
    if header.n_rec != N_REC:
        errs.append(f"n_rec={header.n_rec}，期望 {N_REC}")
    if header.n_ev != N_EV:
        errs.append(f"n_ev={header.n_ev}，期望 {N_EV}")
    expect = Layout(header.n_rec, header.n_ev, header.n_img,
                    header.img_w, header.img_h).total_size
    if header.total_size != expect:
        errs.append(f"total_size={header.total_size}，按 header 参数应为 {expect}")
    return errs


# ── Detection ─────────────────────────────────────────────────────────


@dataclass
class Detection:
    cls: int = 0
    conf: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def _args(self) -> tuple:
        """喂给 _DET.pack_into 的参数元组。"""
        return (int(self.cls), 0, self.conf, self.x1, self.y1, self.x2, self.y2)

    def pack_into(self, buf, offset: int) -> None:
        _DET.pack_into(buf, offset, *self._args())

    @classmethod
    def unpack_from(cls, buf, offset: int) -> "Detection":
        _cls, _pad, conf, x1, y1, x2, y2 = _DET.unpack_from(buf, offset)
        return cls(cls=_cls, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2)


# ── FrameRecord ───────────────────────────────────────────────────────


@dataclass
class FrameRecord:
    """一帧推理结果。`n_det` 由 `len(dets)` 推导，不作为独立字段。"""

    seq: int
    frame_id: int
    capture_ts_ns: int
    publish_ts_ns: int
    infer_ms: float = 0.0
    flags: int = 0
    n_person: int = 0
    det_total: int = 0
    img_slot: int = IMG_SLOT_NONE
    event_vector: Sequence[float] = (0.0,) * 8
    aspect: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    area_ratio: float = 0.0
    best_event_id: int = 0
    best_kind: int = 0
    best_state: int = 0
    best_urgency: int = 0
    flags2: int = 0
    best_conf: float = 0.0
    best_reliability: float = -1.0
    kind_first_ts_ns: int = 0
    dets: Sequence[Detection] = ()

    @property
    def n_det(self) -> int:
        return len(self.dets)

    def pack_into(self, buf, offset: int = 0) -> None:
        dets = list(self.dets)
        if len(dets) > MAX_DET:
            raise ValueError(f"dets 最多 {MAX_DET} 个，收到 {len(dets)}")
        ev = list(self.event_vector)[:8]
        ev += [0.0] * (8 - len(ev))          # 长度不对时补零，绝不让帧循环崩

        _REC_PREFIX.pack_into(
            buf, offset,
            self.seq, self.frame_id, self.capture_ts_ns, self.publish_ts_ns,
            self.infer_ms, int(self.flags), len(dets),
            int(self.n_person), int(self.det_total), int(self.img_slot),
            *ev,
            self.aspect, self.cx, self.cy, self.w, self.h, self.area_ratio,
            int(self.best_event_id), int(self.best_kind), int(self.best_state),
            int(self.best_urgency), int(self.flags2),
            self.best_conf, self.best_reliability, self.kind_first_ts_ns,
        )
        for i in range(MAX_DET):
            _DET.pack_into(buf, offset + DET_OFF + i * _DET.size,
                           *(dets[i]._args() if i < len(dets) else _ZERO_DET))
        # 未使用的 det 槽和 reserved 必须清零，否则会残留上一条记录的数据。
        buf[offset + REC_RESERVED_OFF:offset + REC_SIZE] = _ZERO_REC_RESERVED

    @classmethod
    def unpack_from(cls, buf, offset: int = 0) -> "FrameRecord":
        v = _REC_PREFIX.unpack_from(buf, offset)
        n_det = v[6]
        if n_det > MAX_DET:
            raise ValueError(f"记录里的 n_det={n_det} 超过 {MAX_DET}，布局可能不匹配")
        dets = tuple(Detection.unpack_from(buf, offset + DET_OFF + i * _DET.size)
                     for i in range(n_det))
        return cls(
            seq=v[0], frame_id=v[1], capture_ts_ns=v[2], publish_ts_ns=v[3],
            infer_ms=v[4], flags=v[5], n_person=v[7], det_total=v[8], img_slot=v[9],
            event_vector=tuple(v[10:18]),
            aspect=v[18], cx=v[19], cy=v[20], w=v[21], h=v[22], area_ratio=v[23],
            best_event_id=v[24], best_kind=v[25], best_state=v[26],
            best_urgency=v[27], flags2=v[28], best_conf=v[29],
            best_reliability=v[30], kind_first_ts_ns=v[31],
            dets=dets,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "frame_id": self.frame_id,
            "capture_ts_ns": self.capture_ts_ns, "publish_ts_ns": self.publish_ts_ns,
            "infer_ms": round(self.infer_ms, 3), "flags": int(self.flags),
            "n_det": self.n_det, "n_person": self.n_person,
            "det_total": self.det_total, "img_slot": self.img_slot,
            "event_vector": [round(x, 4) for x in self.event_vector],
            "primary": [round(x, 4) for x in
                        (self.aspect, self.cx, self.cy, self.w, self.h, self.area_ratio)],
            "best": {"event_id": self.best_event_id, "kind": self.best_kind,
                     "state": self.best_state, "urgency": self.best_urgency,
                     "conf": round(self.best_conf, 4),
                     "reliability": round(self.best_reliability, 4)},
            "kind_first_ts_ns": self.kind_first_ts_ns,
            "dets": [{"cls": d.cls, "conf": round(d.conf, 3),
                      "bbox": [round(x, 4) for x in d.bbox]} for d in self.dets],
        }


# ── EventSnapshot ─────────────────────────────────────────────────────


@dataclass
class EventSnapshot:
    """活跃事件快照。kind=0 表示空槽；槽位 = event_id % n_ev。"""

    seq: int = 0
    event_id: int = 0
    kind: int = 0
    state: int = 0
    urgency: int = 0
    start_ts_ns: int = 0
    last_seen_ts_ns: int = 0
    kind_first_ts_ns: int = 0
    confidence: float = 0.0
    peak_confidence: float = 0.0
    reliability: float = -1.0
    hits: int = 0
    bbox: Sequence[float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def is_empty(self) -> bool:
        return self.kind == 0

    def pack_into(self, buf, offset: int = 0) -> None:
        b = list(self.bbox)[:4]
        b += [0.0] * (4 - len(b))
        _EV_PREFIX.pack_into(
            buf, offset,
            self.seq, int(self.event_id),
            int(self.kind), int(self.state), int(self.urgency), 0,
            self.start_ts_ns, self.last_seen_ts_ns, self.kind_first_ts_ns,
            self.confidence, self.peak_confidence, self.reliability,
            int(self.hits), *b, 0, 0,
        )
        buf[offset + EV_RESERVED_OFF:offset + EV_SIZE] = _ZERO_EV_RESERVED

    @classmethod
    def unpack_from(cls, buf, offset: int = 0) -> "EventSnapshot":
        v = _EV_PREFIX.unpack_from(buf, offset)
        return cls(
            seq=v[0], event_id=v[1], kind=v[2], state=v[3], urgency=v[4],
            start_ts_ns=v[6], last_seen_ts_ns=v[7], kind_first_ts_ns=v[8],
            confidence=v[9], peak_confidence=v[10], reliability=v[11],
            hits=v[12], bbox=tuple(v[13:17]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "kind": self.kind, "state": self.state,
            "urgency": self.urgency, "start_ts_ns": self.start_ts_ns,
            "last_seen_ts_ns": self.last_seen_ts_ns,
            "kind_first_ts_ns": self.kind_first_ts_ns,
            "confidence": round(self.confidence, 4),
            "peak_confidence": round(self.peak_confidence, 4),
            "reliability": round(self.reliability, 4),
            "hits": self.hits, "bbox": [round(x, 4) for x in self.bbox],
        }


# ── FrameImageMeta ────────────────────────────────────────────────────


@dataclass
class FrameImageMeta:
    """帧环槽位的 32 B 头；FHR 负载紧随其后，长度 = header.img_size。"""

    seq: int = 0
    frame_id: int = 0
    capture_ts_ns: int = 0
    w: int = DEFAULT_IMG_W
    h: int = DEFAULT_IMG_H

    def pack_into(self, buf, offset: int = 0) -> None:
        _IMG.pack_into(buf, offset, self.seq, self.frame_id,
                       self.capture_ts_ns, self.w, self.h)

    @classmethod
    def unpack_from(cls, buf, offset: int = 0) -> "FrameImageMeta":
        seq, frame_id, ts, w, h = _IMG.unpack_from(buf, offset)
        return cls(seq=seq, frame_id=frame_id, capture_ts_ns=ts, w=w, h=h)


def image_payload(buf, layout: Layout, slot: int) -> memoryview:
    """返回帧环槽位里 BGR 数据的**零拷贝**视图（H×W×3 的 uint8）。

    需要 numpy 时：`np.frombuffer(view, np.uint8).reshape(h, w, 3)`。
    """
    off = layout.img_offset(slot) + IMG_HDR_SIZE
    return memoryview(buf)[off:off + layout.img_size]
