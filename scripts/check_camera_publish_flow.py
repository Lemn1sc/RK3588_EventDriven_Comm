#!/usr/bin/env python3
"""用摄像头/RKNN替身检查真实 run_camera 的发布和清理路径，不测推理精度。"""
import ast
from contextlib import ExitStack, redirect_stdout
import io
import mmap
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from edge_node.edge_shm import DEFAULT_LAYOUT, FrameRecord, FrameFlags
from edge_node.shm_publisher import ShmPublisher, make_detection_record


class Array(list):
    def __truediv__(self, _other):
        return self

    def tolist(self):
        return list(self)

    def mean(self):
        return sum(self) / len(self)

    def max(self):
        return max(self)


class Bgr(bytearray):
    shape = (640, 480, 3)

    def mean(self, **_kwargs):
        return [17, 17, 17]

    def min(self):
        return 17

    def max(self):
        return 17


def check(directory, fail_init=False, wrong_shape=False):
    released, publications, objects, models = [], [], [], []
    active = {"frame": 0}
    messages = [(3, 0, 6, 0, 0)] + [(1, (i - 1) % 3, i, 10, 0) for i in range(1, 7)] + [(4, 0, 0, 0, 0)]
    output = directory / "output.bin"
    lock = directory / "output.lock"
    camera_file = directory / "camera.bin"
    camera_file.write_bytes(bytes(32))

    class Socket:
        closed = False

        def connect(self, _path):
            pass

        def close(self):
            self.closed = True

    sock = Socket()

    class Mapping:
        closed = False

        def close(self):
            self.closed = True

    mapping = Mapping()

    class Runtime:
        NPU_CORE_0, NPU_CORE_1, NPU_CORE_2, NPU_CORE_0_1_2 = 1, 2, 4, 7

        def __init__(self):
            self.closed, self.calls = False, 0
            models.append(self)

        def load_rknn(self, _path):
            return 0

        def init_runtime(self, **_kwargs):
            return -1 if fail_init else 0

        def inference(self, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected inference failure")
            return []

        def release(self):
            self.closed = True

    class Publisher(ShmPublisher):
        def __init__(self):
            super().__init__(backing_path=output, lock_path=lock)
            objects.append(self)

        def publish(self, *args, **kwargs):
            seq = super().publish(*args, **kwargs)
            publications.append(FrameRecord.unpack_from(self.mm, self.layout.rec_offset(seq)))
            return seq

    def recv(_sock):
        message = messages.pop(0)
        if message[0] == 1:
            active["frame"] = message[2]
        return message

    image = Bgr(bytes([17]) * DEFAULT_LAYOUT.img_size)
    if wrong_shape:
        image.shape = (480, 640, 3)
    api = ModuleType("rknnlite.api")
    api.RKNNLite = Runtime
    previous = {key: sys.modules.get(key) for key in ("rknnlite", "rknnlite.api")}
    sys.modules.update({"rknnlite": ModuleType("rknnlite"), "rknnlite.api": api})
    source = ast.parse((ROOT / "scripts/board_camera_edge.py").read_text())
    # 编译真正的主循环；只替换外设/数组计算边界，不复制主循环的逻辑。
    subset = ast.Module(body=[node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                              and node.name in ("run_camera", "OutputLayoutError")], type_ignores=[])
    namespace = dict(
        ExitStack=ExitStack, socket=SimpleNamespace(socket=lambda *_a: sock, AF_UNIX=1, SOCK_SEQPACKET=5),
        mmap=SimpleNamespace(mmap=lambda *_a, **_k: mapping, ACCESS_READ=mmap.ACCESS_READ), os=os,
        _shm_open_readonly=lambda _name: os.open(camera_file, os.O_RDONLY),
        recv_msg=recv, send_msg=lambda _s, *message: released.append(message),
        SHM_NAME="/camera_shm", SOCKET_PATH="test", SHM_STREAM_INFO=3, SHM_FRAME_READY=1,
        SHM_END=4, SHM_RELEASE=2, STATUS_OK=0, STATUS_BAD_DATA=1, NV12_FOURCC=123,
        ShmPublisher=Publisher, FrameRecord=FrameRecord, FrameFlags=FrameFlags, DEFAULT_LAYOUT=DEFAULT_LAYOUT,
        slot_offsets=lambda *_a: (0, 1024),
        read_slot_header=lambda *_a: dict(frame_id=active["frame"], data_size=10, fmt=123,
                                         ts_flags=0x2000, ts_ns=1234, wstride=480, hstride=640,
                                         width=480, height=640),
        nv12_to_bgr=lambda *_a: image, letterbox_rgb=lambda *_a: (image, (1, 0, 0)),
        decode=lambda *_a: (Array([[48, 64, 240, 576]]), Array([0.9]), Array([0])),
        scale_boxes=lambda b, *_a: b,
        event_from_detections=lambda *_a: SimpleNamespace(to_array=lambda: Array([1, 0.9, 0.1, 0.1, 0.5, 0.9, 1, 0])),
        make_detection_record=make_detection_record, time=time,
        np=SimpleNamespace(array=lambda data, **_k: Array(data), float32=float,
                           median=lambda data: sorted(data)[len(data) // 2]),
    )
    exec(compile(subset, "board_camera_edge.py", "exec"), namespace)
    args = SimpleNamespace(model=camera_file, stride=2, save_every=0, conf=0.25, iou=0.7, core_mask="0_1_2",
                           publish_shm=True, rotate=0, swap_uv=False, out_dir=directory,
                           dump_nv12=None)
    try:
        log = io.StringIO()
        with redirect_stdout(log):
            try:
                result = namespace["run_camera"](args)
            except SystemExit:
                assert fail_init
            except namespace["OutputLayoutError"]:
                assert wrong_shape
            else:
                assert not fail_init and not wrong_shape and result == 1
        assert sock.closed and mapping.closed and models[0].closed
        assert not output.exists()
        if fail_init:
            assert not objects and not released
        elif wrong_shape:
            assert len(released) == 2 and released[-1][-1] == 1
        else:
            assert [r.seq for r in publications] == [1, 2, 3]
            assert [r.frame_id for r in publications] == [2, 4, 6]
            assert publications[1].flags & FrameFlags.BAD_FRAME
            assert publications[1].img_slot == 0xffff and not publications[1].dets
            assert publications[0].capture_ts_ns == 1234 and publications[0].n_person == 1, log.getvalue()
            assert len(released) == 6
            assert [r[2] for r in released] == list(range(1, 7))
            assert [r[-1] for r in released] == [0, 0, 0, 1, 0, 0]
            assert objects[0].header.frames_ready == 6 and objects[0].header.frames_inferred == 3
            assert objects[0].header.frames_bad == 1
    finally:
        for key, old in previous.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old


def main():
    with tempfile.TemporaryDirectory(prefix="camera-publish-flow-") as directory:
        path = Path(directory)
        check(path)
        print("PASS: real camera loop, stride=2, inference failure -> BAD/no-image, exactly one RELEASE per READY")
        check(path, fail_init=True)
        print("PASS: runtime initialization failure closes camera mapping/socket/runtime")
        check(path, wrong_shape=True)
        print("PASS: wrong output shape aborts after RELEASE and cleans publisher")
    print("Camera flow checks passed (external devices/arrays mocked; no RKNN inference performed).")


if __name__ == "__main__":
    main()
