#!/usr/bin/env python
"""共享内存契约自检（纯 Python，不依赖 numpy / cv2 / 板子）。

对应 docs/10_模型传输共享内存契约.md。它验证的不是"代码能跑"，而是**字节对齐**：

1. 所有结构体尺寸 = 128 / 512 / 192 / 32 / 24；
2. 文档 §1 里的段偏移（128 / 32896 / 35968 / 7409024）与代码一致；
3. 每个字段在**文档写明的偏移**上 —— 用独立的 `struct.unpack_from` 交叉验证，
   等价于"队友的 C++ 读端按文档偏移直接读"；
4. 打包 → 解包 → 逐字段还原；
5. 未使用的 det 槽和 reserved 必须清零（否则会残留上一条记录）；
6. 环的取模口径：记录用 (seq-1)，事件/帧环用自身 id；
7. Header 校验能拦住 magic/version/尺寸不对。

用法：
    python scripts/check_edge_shm.py
"""
from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from edge_node import edge_shm as E  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"✓ {name}")
    else:
        FAILED.append(name)
        print(f"✗ {name}" + (f"\n    {detail}" if detail else ""))


def approx(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


# ── 1. 尺寸 ───────────────────────────────────────────────────────────

def check_sizes() -> None:
    check("Header = 128 B", E._HDR.size == E.HDR_SIZE, f"{E._HDR.size}")
    check("FrameRecord = 512 B", struct.calcsize(E.REC_FMT) == E.REC_SIZE,
          f"{struct.calcsize(E.REC_FMT)}")
    check("EventSnapshot = 192 B", E._EV_PREFIX.size + 112 == E.EV_SIZE)
    check("FrameImageMeta = 32 B", E._IMG.size == E.IMG_HDR_SIZE, f"{E._IMG.size}")
    check("Detection = 24 B", E._DET.size == 24, f"{E._DET.size}")
    check("dets 起点 = 128", E.DET_OFF == 128, f"{E.DET_OFF}")
    check("reserved 起点 = 272", E.REC_RESERVED_OFF == 272, f"{E.REC_RESERVED_OFF}")


# ── 2. 布局 ───────────────────────────────────────────────────────────

def check_layout() -> None:
    L = E.DEFAULT_LAYOUT
    check("off_rec = 128", L.off_rec == 128, f"{L.off_rec}")
    check("off_ev = 32896", L.off_ev == 32896, f"{L.off_ev}")
    check("off_img = 35968", L.off_img == 35968, f"{L.off_img}")
    check("img_size = 921600", L.img_size == 480 * 640 * 3, f"{L.img_size}")
    check("img_slot_size = 921632", L.img_slot_size == 32 + 921600, f"{L.img_slot_size}")
    check("total_size = 7409024", L.total_size == 7409024, f"{L.total_size}")

    check("rec: seq=1 → 槽 0", L.rec_offset(1) == L.off_rec)
    check("rec: seq=65 → 回到槽 0", L.rec_offset(65) == L.rec_offset(1))
    check("rec: seq=2 与 1 不重叠",
          L.rec_offset(2) - L.rec_offset(1) == E.REC_SIZE)
    check("ev: id=17 → 槽 1", L.ev_offset(17) == L.off_ev + E.EV_SIZE)
    check("ev: id=0 → 槽 0", L.ev_offset(0) == L.off_ev)
    check("img: slot=7 是最后一槽",
          L.img_offset(7) + L.img_slot_size == L.total_size)
    try:
        L.img_offset(8)
        check("img: slot 越界应报错", False)
    except ValueError:
        check("img: slot 越界应报错", True)


# ── 3. Header 往返 ────────────────────────────────────────────────────

def check_header() -> None:
    h = E.Header.for_layout()
    h.pub_seq, h.img_seq = 41, 42
    h.frames_ready, h.frames_inferred = 100, 50
    h.frames_bad, h.img_overwrites = 1, 3
    h.last_update_ts_ns = 123456789
    h.flags = int(E.ModelFlags.READY)

    buf = bytearray(E.HDR_SIZE)
    h.pack_into(buf)
    h2 = E.Header.unpack_from(buf)

    check("Header 往返字段一致",
          h2.pub_seq == 41 and h2.img_seq == 42 and h2.frames_bad == 1
          and h2.last_update_ts_ns == 123456789 and h2.flags == 1)
    check("Header 尺寸字段反推布局一致", h2.layout.total_size == E.DEFAULT_LAYOUT.total_size)
    check("validate_header 接受正常头", E.validate_header(h2) == [])

    # 文档偏移交叉验证：pub_seq 在 64，last_update_ts_ns 在 120
    check("pub_seq 在偏移 64",
          struct.unpack_from("<Q", buf, 64)[0] == 41)
    check("last_update_ts_ns 在偏移 120",
          struct.unpack_from("<Q", buf, 120)[0] == 123456789)

    bad = E.Header.unpack_from(buf)
    bad.magic = 0
    bad.version = 99
    errs = E.validate_header(bad)
    check("validate_header 拦住 magic/version", len(errs) >= 2, str(errs))


# ── 4. FrameRecord 往返 + 文档偏移 ────────────────────────────────────

def make_record(seq: int = 7) -> E.FrameRecord:
    return E.FrameRecord(
        seq=seq, frame_id=1234, capture_ts_ns=1_000_000_000,
        publish_ts_ns=1_068_000_000, infer_ms=68.0,
        flags=int(E.FrameFlags.INFERRED | E.FrameFlags.TS_FROM_CAMERA),
        n_person=2, det_total=3, img_slot=3,
        event_vector=(4.0, 0.5, 0.25, 0.125, 0.75, 0.875, 1.0, 0.625),
        aspect=0.5, cx=0.25, cy=0.75, w=0.125, h=0.5, area_ratio=0.0625,
        best_event_id=7, best_kind=4, best_state=1, best_urgency=3,
        best_conf=0.5, best_reliability=0.25, kind_first_ts_ns=1_050_000_000,
        dets=(
            E.Detection(cls=0, conf=0.5, x1=0.125, y1=0.25, x2=0.375, y2=0.75),
            E.Detection(cls=0, conf=0.25, x1=0.5, y1=0.125, x2=0.75, y2=0.5),
        ),
    )


def check_record() -> None:
    rec = make_record()
    buf = bytearray(E.REC_SIZE)
    rec.pack_into(buf)
    got = E.FrameRecord.unpack_from(buf)

    check("FrameRecord 往返：标量", got.seq == 7 and got.frame_id == 1234
          and got.capture_ts_ns == 1_000_000_000 and got.publish_ts_ns == 1_068_000_000
          and approx(got.infer_ms, 68.0) and got.flags == 9)   # INFERRED|TS_FROM_CAMERA
    check("FrameRecord 往返：n_det 由 dets 推导", got.n_det == 2, f"{got.n_det}")
    check("FrameRecord 往返：事件向量",
          all(approx(a, b) for a, b in zip(got.event_vector, rec.event_vector)))
    check("FrameRecord 往返：主目标特征",
          all(approx(a, b) for a, b in zip(
              (got.aspect, got.cx, got.cy, got.w, got.h, got.area_ratio),
              (rec.aspect, rec.cx, rec.cy, rec.w, rec.h, rec.area_ratio))))
    check("FrameRecord 往返：事件上下文",
          got.best_event_id == 7 and got.best_kind == 4 and got.best_state == 1
          and got.best_urgency == 3 and approx(got.best_conf, 0.5)
          and approx(got.best_reliability, 0.25) and got.kind_first_ts_ns == 1_050_000_000)
    check("FrameRecord 往返：检测框",
          len(got.dets) == 2 and got.dets[0].cls == 0 and approx(got.dets[1].conf, 0.25)
          and approx(got.dets[0].x2, 0.375))

    # 文档偏移交叉验证 —— 等价于队友的 C++ 读端按文档直接读
    check("img_slot 在偏移 46", struct.unpack_from("<H", buf, 46)[0] == 3)
    check("best_kind 在偏移 108", struct.unpack_from("<B", buf, 108)[0] == 4)
    check("best_conf 在偏移 112", approx(struct.unpack_from("<f", buf, 112)[0], 0.5))
    check("kind_first_ts_ns 在偏移 120",
          struct.unpack_from("<Q", buf, 120)[0] == 1_050_000_000)
    check("dets[0].cls 在偏移 128", struct.unpack_from("<H", buf, 128)[0] == 0)
    check("dets[0].conf 在偏移 132", approx(struct.unpack_from("<f", buf, 132)[0], 0.5))
    check("dets[1] 在偏移 152（24 B 步长）",
          approx(struct.unpack_from("<f", buf, 152 + 4)[0], 0.25))


def check_record_zeroing() -> None:
    """未使用的 det 槽与 reserved 必须清零，不能残留上一条记录。"""
    buf = bytearray(b"\xAA" * E.REC_SIZE)
    E.FrameRecord(seq=1, frame_id=1, capture_ts_ns=0, publish_ts_ns=0,
                  dets=(E.Detection(cls=0, conf=0.5),)).pack_into(buf)

    tail = buf[E.DET_OFF + 24: E.REC_RESERVED_OFF]
    check("未使用的 det 槽已清零", all(b == 0 for b in tail))
    check("reserved 已清零（240 B）",
          E.REC_SIZE - E.REC_RESERVED_OFF == 240
          and all(b == 0 for b in buf[E.REC_RESERVED_OFF:]))

    # 空 dets → n_det=0，全部 det 槽为零
    buf2 = bytearray(b"\xAA" * E.REC_SIZE)
    E.FrameRecord(seq=1, frame_id=1, capture_ts_ns=0, publish_ts_ns=0).pack_into(buf2)
    got = E.FrameRecord.unpack_from(buf2)
    check("空 dets → n_det=0 且解包不报错", got.n_det == 0)
    check("空 dets 时 det 区全零", all(b == 0 for b in buf2[E.DET_OFF:E.REC_RESERVED_OFF]))

    # 超过 6 个必须报错
    try:
        E.FrameRecord(seq=1, frame_id=1, capture_ts_ns=0, publish_ts_ns=0,
                      dets=[E.Detection()] * 7).pack_into(bytearray(E.REC_SIZE))
        check("dets > 6 应报错", False)
    except ValueError:
        check("dets > 6 应报错", True)


# ── 5. EventSnapshot ──────────────────────────────────────────────────

def check_event() -> None:
    ev = E.EventSnapshot(
        seq=3, event_id=17, kind=4, state=1, urgency=3,
        start_ts_ns=1_000_000_000, last_seen_ts_ns=1_100_000_000,
        kind_first_ts_ns=1_050_000_000, confidence=0.5, peak_confidence=0.75,
        reliability=0.25, hits=12, bbox=(0.125, 0.25, 0.375, 0.5),
    )
    buf = bytearray(b"\xAA" * E.EV_SIZE)
    ev.pack_into(buf)
    got = E.EventSnapshot.unpack_from(buf)

    check("EventSnapshot 往返",
          got.event_id == 17 and got.kind == 4 and got.state == 1
          and got.urgency == 3 and got.kind_first_ts_ns == 1_050_000_000
          and approx(got.peak_confidence, 0.75) and got.hits == 12
          and all(approx(a, b) for a, b in zip(got.bbox, ev.bbox)))
    check("Event 空槽判定（kind=0）", E.EventSnapshot().is_empty)
    check("Event 非空槽", not got.is_empty)
    check("Event reserved 已清零（112 B）",
          all(b == 0 for b in buf[E.EV_RESERVED_OFF:]))

    # 文档偏移：kind 在 12，kind_first_ts_ns 在 32，bbox 在 56
    check("event.kind 在偏移 12", struct.unpack_from("<B", buf, 12)[0] == 4)
    check("event.kind_first_ts_ns 在偏移 32",
          struct.unpack_from("<Q", buf, 32)[0] == 1_050_000_000)
    check("event.bbox[0] 在偏移 56", approx(struct.unpack_from("<f", buf, 56)[0], 0.125))


# ── 6. FrameImage ─────────────────────────────────────────────────────

def check_image() -> None:
    L = E.DEFAULT_LAYOUT
    shm = bytearray(L.total_size)

    meta = E.FrameImageMeta(seq=42, frame_id=1234,
                            capture_ts_ns=1_000_000_000, w=480, h=640)
    slot = 3
    off = L.img_offset(slot)
    meta.pack_into(shm, off)

    # 写一点可辨识的像素（只写首尾，不真的填 921 KB）
    view = E.image_payload(shm, L, slot)
    view[0], view[1], view[2] = 10, 20, 30
    view[-3], view[-2], view[-1] = 40, 50, 60

    got = E.FrameImageMeta.unpack_from(shm, off)
    check("FrameImageMeta 往返",
          got.seq == 42 and got.frame_id == 1234
          and got.capture_ts_ns == 1_000_000_000 and got.w == 480 and got.h == 640)
    check("image_payload 长度 = img_size", len(view) == L.img_size, f"{len(view)}")
    check("image_payload 可写且定位正确",
          view[0] == 10 and view[1] == 20 and view[2] == 30
          and view[-3] == 40 and view[-2] == 50 and view[-1] == 60)
    check("帧环首尾相接：slot 3 负载不越界",
          L.img_offset(slot) + E.IMG_HDR_SIZE + L.img_size
          <= L.img_offset(slot + 1))

    # 文档偏移：frame_id 在 +8，w 在 +24
    check("image.frame_id 在 +8", struct.unpack_from("<Q", shm, off + 8)[0] == 1234)
    check("image.w 在 +24", struct.unpack_from("<I", shm, off + 24)[0] == 480)


# ── 7. 全内存模拟：写端写、读端按 seq 读 ─────────────────────────────

def check_ring_roundtrip() -> None:
    L = E.DEFAULT_LAYOUT
    shm = bytearray(L.total_size)
    h = E.Header.for_layout(L)
    h.pack_into(shm)

    # 写 3 条，读端只拿最新的
    for i in (1, 2, 3):
        rec = make_record(seq=i)
        rec.frame_id = 1000 + i
        rec.pack_into(shm, L.rec_offset(i))
        h.pub_seq = i
        h.pack_into(shm)

    h2 = E.Header.unpack_from(shm)
    latest = E.FrameRecord.unpack_from(shm, L.rec_offset(h2.pub_seq))
    check("读端按 pub_seq 拿到最新记录", latest.seq == 3 and latest.frame_id == 1003)

    # 落后 2 条时仍能按 seq 逐条读回
    seqs = [E.FrameRecord.unpack_from(shm, L.rec_offset(s)).seq
            for s in range(1, h2.pub_seq + 1)]
    check("读端可以补读落后的记录", seqs == [1, 2, 3], str(seqs))

    # 环写满后覆盖：seq=1 与 seq=1+n_rec 落在同一槽
    rec = make_record(seq=1 + E.N_REC)
    rec.pack_into(shm, L.rec_offset(1 + E.N_REC))
    same_slot = E.FrameRecord.unpack_from(shm, L.rec_offset(1))
    check("环覆盖后旧序号读到的是新内容", same_slot.seq == 1 + E.N_REC)


# ── 8. C 头交叉验证（跨语言 ABI）─────────────────────────────────────

def check_c_header() -> None:
    """编译 src/edge_node/edge_shm.h，逐项对比 sizeof/offsetof 与 Python 布局。

    这一步等价于"队友的 C 读端按文档偏移直接读"——两边数字一致，ABI 才算对齐。
    本机没有 C 编译器时跳过，不判失败。
    """
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        print("… 跳过 C 头交叉验证（本机没有 gcc / cc）")
        return

    src = ROOT / "scripts" / "check_edge_shm_layout.c"
    inc = ROOT / "src" / "edge_node"
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / ("layout.exe" if os.name == "nt" else "layout")
        build = subprocess.run(
            [cc, "-std=c11", "-Wall", "-Wextra", "-Wshadow", "-Wconversion",
             "-Wsign-conversion", "-Werror", f"-I{inc}", str(src), "-o", str(exe)],
            capture_output=True, text=True)
        if build.returncode != 0:
            check("C 头编译（_Static_assert + 严格告警）", False,
                  (build.stderr or build.stdout).strip()[:1200])
            return
        check("C 头编译（_Static_assert + 严格告警）", True)
        run = subprocess.run([str(exe)], capture_output=True, text=True)
    if run.returncode != 0:
        check("C 布局探针运行", False, (run.stderr or "").strip()[:600])
        return

    got: dict[str, int] = {}
    for line in run.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            got[k.strip()] = int(v)

    L = E.DEFAULT_LAYOUT
    expected = {
        "edge_shm_header.size": E.HDR_SIZE,
        "edge_frame_record.size": E.REC_SIZE,
        "edge_detection.size": 24,
        "edge_event_snapshot.size": E.EV_SIZE,
        "edge_image_meta.size": E.IMG_HDR_SIZE,
        "edge_frame_record.img_slot": 46,
        "edge_frame_record.event_vector": 48,
        "edge_frame_record.best_kind": 108,
        "edge_frame_record.best_conf": 112,
        "edge_frame_record.kind_first_ts_ns": 120,
        "edge_frame_record.dets": E.DET_OFF,
        "edge_frame_record.reserved": E.REC_RESERVED_OFF,
        "edge_event_snapshot.kind": 12,
        "edge_event_snapshot.kind_first_ts_ns": 32,
        "edge_event_snapshot.bbox": 56,
        "edge_shm_header.pub_seq": 64,
        "edge_shm_header.last_update_ts_ns": 120,
        "off_rec": L.off_rec,
        "off_ev": L.off_ev,
        "off_img": L.off_img,
        "img_size": L.img_size,
        "img_slot_size": L.img_slot_size,
        "total_size": L.total_size,
        "rec_offset.1": L.rec_offset(1),
        "rec_offset.65": L.rec_offset(65),
        "ev_offset.17": L.ev_offset(17),
        "img_offset.7": L.img_offset(7),
    }
    missing = sorted(set(expected) - set(got))
    check("C 探针输出项齐全", not missing, f"缺少 {missing}")
    mismatch = [f"{k}: C={got.get(k)} / Py={v}"
                for k, v in expected.items() if got.get(k) != v]
    check("C 头与 Python 布局逐项一致", not mismatch, "; ".join(mismatch))


# ── main ──────────────────────────────────────────────────────────────

def main() -> int:
    print("== 尺寸与格式串 ==")
    check_sizes()
    print("\n== 布局与偏移 ==")
    check_layout()
    print("\n== Header ==")
    check_header()
    print("\n== FrameRecord ==")
    check_record()
    check_record_zeroing()
    print("\n== EventSnapshot ==")
    check_event()
    print("\n== FrameImage ==")
    check_image()
    print("\n== 环形读写 ==")
    check_ring_roundtrip()
    print("\n== C 头交叉验证 ==")
    check_c_header()

    print("\n" + "=" * 56)
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败：")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("✓ 全部通过：共享内存布局与 docs/10 一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
