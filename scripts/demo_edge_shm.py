#!/usr/bin/env python3
"""不需要摄像头/RKNN/numpy的共享内存发布演示；发布合成 BGR 和检测框。"""
import argparse
from pathlib import Path
import signal
import sys
import time

script_dir = Path(__file__).resolve().parent
source_dir = script_dir / "src" if (script_dir / "src").is_dir() else script_dir.parent / "src"
sys.path.insert(0, str(source_dir))
from edge_node.shm_publisher import ShmPublisher, make_detection_record
from edge_node.edge_shm import DEFAULT_LAYOUT, FrameRecord, FrameFlags, SHM_NAME, LOCK_PATH


def stop(_signo, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200, help="0 表示持续发布")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bad-every", type=int, default=0, help="每 N 次发布无图 BAD 记录，0 关闭")
    parser.add_argument("--shm-name", default=SHM_NAME)
    parser.add_argument("--lock-path", default=LOCK_PATH)
    parser.add_argument("--shm-file", type=Path, help="主机文件映射测试，代替 POSIX SHM")
    parser.add_argument("--clean", action="store_true", help="显式清理崩溃残留，活跃写端存在则拒绝")
    args = parser.parse_args()
    if args.count < 0 or not 1 <= args.fps <= 120 or args.bad_every < 0:
        parser.error("count/bad-every 必须 >=0，fps 为 1..120")
    options = dict(shm_name=args.shm_name, lock_path=args.lock_path, backing_path=args.shm_file)
    if args.clean:
        ShmPublisher.clean(**options)
        print("Cleaned inactive output object; lock files retained")
        return 0
    signal.signal(signal.SIGTERM, stop)
    try:
        with ShmPublisher(**options) as publisher:
            sequence = 0
            print("SYNTHETIC shared-memory demo; no real model inference", flush=True)
            while args.count == 0 or sequence < args.count:
                started = time.monotonic_ns()
                sequence += 1
                frame_id = sequence * 2  # 模拟 stride=2，发布号仍连续。
                publisher.update_counters(frame_id, sequence)
                if args.bad_every and sequence % args.bad_every == 0:
                    record = FrameRecord(0, frame_id, 0, 0, flags=int(FrameFlags.BAD_FRAME))
                    image = None
                else:
                    record = make_detection_record(
                        frame_id, [[48, 64, 240, 576]], [0.9], [0], 480, 640,
                        [1, 0.9, 0.1, 0.1, 0.5, 0.9, 1, 0], started)
                    image = bytes([sequence % 251]) * DEFAULT_LAYOUT.img_size
                publisher.publish(record, image, started_ns=started, error_code=1 if image is None else 0)
                if sequence == 1 or sequence % args.fps == 0:
                    print(f"[publish] seq={sequence} frame_id={frame_id} image={image is not None}", flush=True)
                remaining = 1 / args.fps - (time.monotonic_ns() - started) / 1e9
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("Stopped; output shared memory cleaned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
