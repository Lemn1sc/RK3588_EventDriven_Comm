import argparse
import csv
import json
import socket
import threading
import time
from pathlib import Path
from typing import Iterable

from event_rule_engine import FrameSample, RuleEventEngine


BASE_DIR = Path(__file__).resolve().parent


def build_synthetic_stream(fps: float = 10.0) -> Iterable[FrameSample]:
    step = 1.0 / fps
    frame_id = 0
    timestamp = 0.0

    def emit(person_present: bool, bbox, in_danger_zone: bool, count: int):
        nonlocal frame_id, timestamp
        for i in range(count):
            yield FrameSample(
                frame_id=frame_id,
                timestamp=timestamp,
                person_present=person_present,
                bbox=bbox,
                in_danger_zone=in_danger_zone,
            )
            frame_id += 1
            timestamp += step

    # No person
    yield from emit(False, None, False, 15)

    # Standing normally, aspect ratio stays stable
    for i in range(20):
        dx = (i % 3) - 1
        dy = ((i + 1) % 3) - 1
        bbox = (100 + dx, 60 + dy, 180 + dx, 250 + dy)
        yield from emit(True, bbox, False, 1)

    # Sudden posture change
    yield from emit(True, (95, 95, 245, 215), False, 1)

    # Static after fall
    for i in range(35):
        dx = (i % 2)
        dy = ((i + 1) % 2)
        bbox = (92 + dx, 112 + dy, 248 + dx, 228 + dy)
        yield from emit(True, bbox, False, 1)

    # Danger zone entry
    for i in range(10):
        bbox = (320, 80, 420, 220)
        yield from emit(True, bbox, True, 1)


def write_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def send_packet(sock: socket.socket, payload: dict) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def run_edge(host: str, port: int, log_path: Path, delay: float = 0.03) -> None:
    engine = RuleEventEngine()
    sent_reasons = set()
    client = None

    try:
        client = socket.create_connection((host, port), timeout=2.0)
        print(f"[EDGE] connected to {host}:{port}")
    except OSError as exc:
        print(f"[EDGE] pc server not reachable, continue without send: {exc}")

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_id",
                "timestamp",
                "person_present",
                "bbox",
                "in_danger_zone",
                "trigger",
                "reason",
                "stable",
                "motion_px",
                "aspect_ratio",
                "static_seconds",
            ]
        )

        for sample in build_synthetic_stream():
            decision = engine.update(sample)
            static_seconds = decision.meta.get("static_seconds", 0.0)
            writer.writerow(
                [
                    sample.frame_id,
                    f"{sample.timestamp:.2f}",
                    int(sample.person_present),
                    "" if sample.bbox is None else list(sample.bbox),
                    int(sample.in_danger_zone),
                    int(decision.trigger),
                    decision.reason,
                    int(decision.stable),
                    f"{decision.motion_px:.2f}",
                    f"{decision.aspect_ratio:.3f}",
                    f"{static_seconds:.2f}",
                ]
            )

            if not sample.person_present:
                sent_reasons.clear()

            if decision.trigger and decision.reason not in sent_reasons:
                payload = {
                    "frame_id": sample.frame_id,
                    "timestamp": sample.timestamp,
                    "person_present": sample.person_present,
                    "bbox": sample.bbox,
                    "in_danger_zone": sample.in_danger_zone,
                    "trigger": decision.trigger,
                    "reason": decision.reason,
                    "stable": decision.stable,
                    "motion_px": round(decision.motion_px, 2),
                    "aspect_ratio": round(decision.aspect_ratio, 3),
                    "meta": decision.meta,
                }
                print(f"[EDGE] trigger -> {decision.reason}, send to PC")
                if client is not None:
                    try:
                        send_packet(client, payload)
                    except OSError as exc:
                        print(f"[EDGE] send failed: {exc}")
                        client.close()
                        client = None
                sent_reasons.add(decision.reason)
            else:
                print(
                    f"[EDGE] frame={sample.frame_id:03d} reason={decision.reason:<14} "
                    f"stable={int(decision.stable)} motion={decision.motion_px:5.2f}"
                )

            time.sleep(delay)

    if client is not None:
        client.close()


def precise_pose_stub(payload: dict) -> dict:
    reason = payload.get("reason", "")
    if reason == "fall_suspicious":
        return {"result": "suspected_fall", "pose": "lying_or_low"}
    if reason == "danger_zone":
        return {"result": "danger_area", "pose": "needs_check"}
    if reason == "static_too_long":
        return {"result": "long_static", "pose": "possibly_motionless"}
    return {"result": "unknown", "pose": "unknown"}


def server_loop(host: str, port: int, log_path: Path, stop_event: threading.Event) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.settimeout(0.5)
        print(f"[PC] listening on {host}:{port}")

        conn = None
        buffer = b""
        while not stop_event.is_set():
            if conn is None:
                try:
                    conn, addr = server.accept()
                    conn.settimeout(0.5)
                    buffer = b""
                    print(f"[PC] client connected: {addr[0]}:{addr[1]}")
                except socket.timeout:
                    continue

            try:
                chunk = conn.recv(4096)
                if not chunk:
                    print("[PC] client disconnected")
                    conn.close()
                    conn = None
                    buffer = b""
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    if not raw.strip():
                        continue
                    payload = json.loads(raw.decode("utf-8"))
                    write_jsonl(log_path, payload)
                    refined = precise_pose_stub(payload)
                    print(
                        f"[PC] recv reason={payload.get('reason')} "
                        f"frame={payload.get('frame_id')} -> {refined['result']}"
                    )
            except socket.timeout:
                continue
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[PC] connection error: {exc}")
                if conn is not None:
                    conn.close()
                conn = None
                buffer = b""

        if conn is not None:
            conn.close()


def run_server(host: str, port: int, log_path: Path) -> None:
    stop_event = threading.Event()
    try:
        server_loop(host, port, log_path, stop_event)
    except KeyboardInterrupt:
        stop_event.set()


def run_demo(host: str, port: int, edge_log: Path, pc_log: Path) -> None:
    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=server_loop,
        args=(host, port, pc_log, stop_event),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)
    run_edge(host, port, edge_log)
    stop_event.set()
    server_thread.join(timeout=1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["server", "edge", "demo"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--edge-log", default=str(BASE_DIR / "edge_frame_log.csv"))
    parser.add_argument("--pc-log", default=str(BASE_DIR / "pc_event_log.jsonl"))
    args = parser.parse_args()

    edge_log = Path(args.edge_log)
    pc_log = Path(args.pc_log)

    if args.role == "server":
        run_server(args.host, args.port, pc_log)
    elif args.role == "edge":
        run_edge(args.host, args.port, edge_log)
    else:
        run_demo(args.host, args.port, edge_log, pc_log)


if __name__ == "__main__":
    main()
