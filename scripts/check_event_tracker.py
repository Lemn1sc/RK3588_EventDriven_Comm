#!/usr/bin/env python
"""事件形成器自检（纯 Python，不依赖 numpy / cv2 / torch）。

用合成序列覆盖事件层最容易写错的几件事：

1. 同一个人跨帧 → 一个事件，不重复；
2. 短断档 → 同一事件；长断档 → 关闭旧事件、开新事件；
3. 两个人 → 两个事件，不因同帧而合并；
4. PERSON → SUSPECTED_FALL → FALL_CONFIRMED 只升不降、保留同一 id；
5. TRACK_LOST 能作为独立事件（把“检测器失效”变成信号）；
6. TTL / 断档关闭，且 end 取最后一次证据时间，不是关闭时间；
7. 反向 / 越界 bbox 归一化；
8. 快照可 JSON 序列化。

用法：
    python scripts/check_event_tracker.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from edge_node.event_tracker import (  # noqa: E402
    EventKind, EventState, EventTracker, EventTrackerConfig, Evidence, UpdateAction, Urgency,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"✓ {name}")
    else:
        FAILED.append(name)
        print(f"✗ {name}" + (f"\n    {detail}" if detail else ""))


def ev(kind: EventKind, conf: float = 0.8, bbox=(0.2, 0.2, 0.4, 0.6),
       source: str = "detector", reliability: float | None = None,
       urgency: Urgency | None = None) -> Evidence:
    return Evidence(kind=kind, confidence=conf, bbox=bbox, source=source,
                    reliability=reliability, urgency=urgency)


def feed(tracker: EventTracker, rows: list[tuple[int, list[Evidence]]]) -> list:
    out = []
    for frame, evidences in rows:
        out.extend(tracker.update(frame, frame / 30.0, evidences))
    return out


def test_single_track() -> None:
    t = EventTracker(EventTrackerConfig(min_hits=2))
    ups = feed(t, [(i, [ev(EventKind.PERSON)]) for i in range(5)])
    check("同一目标 5 帧 → 1 个事件", len(t.events) == 1, f"实际 {len(t.events)}")
    e = next(iter(t.events.values()))
    check("命中 5 次", e.hits == 5, f"实际 {e.hits}")
    check("candidate 升为 active", e.state == EventState.ACTIVE, f"实际 {e.state.name}")
    starts = [u for u in ups if u.action == UpdateAction.START]
    check("只发一次 START", len(starts) == 1, f"实际 {len(starts)}")


def test_short_gap() -> None:
    t = EventTracker(EventTrackerConfig(max_gap_frames=3, min_hits=2))
    feed(t, [(0, [ev(EventKind.PERSON)]), (1, [ev(EventKind.PERSON)]),
             (3, [ev(EventKind.PERSON)])])   # 断 1 帧
    check("短断档仍是同一事件", len(t.events) == 1, f"实际 {len(t.events)}")
    e = next(iter(t.events.values()))
    check("短断档命中数累加", e.hits == 3, f"实际 {e.hits}")


def test_long_gap() -> None:
    t = EventTracker(EventTrackerConfig(max_gap_frames=3, min_hits=1))
    ups = feed(t, [(0, [ev(EventKind.PERSON)]), (1, [ev(EventKind.PERSON)]),
                   (10, [ev(EventKind.PERSON)])])
    check("长断档 → 两个事件", len(t.events) == 2, f"实际 {len(t.events)}")
    old = t.events[1]
    check("旧事件已关闭", old.state == EventState.EXPIRED, f"实际 {old.state.name}")
    check("旧事件 end 是最后证据帧（1）", old.last_seen_frame == 1,
          f"实际 {old.last_seen_frame}")
    check("关闭发生在断档被发现的帧（10）", old.closed_frame == 10,
          f"实际 {old.closed_frame}")
    closes = [u for u in ups if u.action == UpdateAction.CLOSE]
    check("发出一条 CLOSE", len(closes) == 1, f"实际 {len(closes)}")


def test_two_people() -> None:
    t = EventTracker(EventTrackerConfig(min_hits=1))
    a = (0.05, 0.10, 0.20, 0.50)
    b = (0.70, 0.10, 0.85, 0.50)
    feed(t, [(0, [ev(EventKind.PERSON, bbox=a), ev(EventKind.PERSON, bbox=b)])])
    check("同帧两个远处目标 → 两个事件", len(t.events) == 2, f"实际 {len(t.events)}")


def test_escalation() -> None:
    t = EventTracker(EventTrackerConfig(min_hits=2))
    ups = feed(t, [
        (0, [ev(EventKind.PERSON, 0.9)]),
        (1, [ev(EventKind.SUSPECTED_FALL, 0.4)]),
        (2, [ev(EventKind.FALL_CONFIRMED, 0.9, source="remote_pose")]),
    ])
    check("升级不新建事件", len(t.events) == 1, f"实际 {len(t.events)}")
    e = next(iter(t.events.values()))
    check("最终类型 FALL_CONFIRMED", e.kind == EventKind.FALL_CONFIRMED, f"实际 {e.kind.name}")
    check("最终状态 CONFIRMED", e.state == EventState.CONFIRMED, f"实际 {e.state.name}")
    check("紧急度 CRITICAL", e.suggested_urgency == Urgency.CRITICAL,
          f"实际 {e.suggested_urgency.name}")
    check("START 只有一次", len([u for u in ups if u.action == UpdateAction.START]) == 1)
    d = e.to_dict()
    check("记录 PERSON 首次出现帧", d["kind_first_frame"].get("PERSON") == 0,
          str(d["kind_first_frame"]))
    check("记录 SUSPECTED_FALL 首次告警帧",
          d["kind_first_frame"].get("SUSPECTED_FALL") == 1,
          str(d["kind_first_frame"]))
    check("记录 FALL_CONFIRMED 首次帧",
          d["kind_first_frame"].get("FALL_CONFIRMED") == 2,
          str(d["kind_first_frame"]))
    check("记录各类型首次 confidence",
          d["kind_confidence"].get("PERSON") == 0.9
          and d["kind_confidence"].get("SUSPECTED_FALL") == 0.4
          and d["kind_confidence"].get("FALL_CONFIRMED") == 0.9,
          str(d["kind_confidence"]))


def test_track_lost() -> None:
    t = EventTracker(EventTrackerConfig(min_hits=2))
    feed(t, [
        (0, [ev(EventKind.PERSON)]),
        (1, [ev(EventKind.PERSON)]),
        (2, [ev(EventKind.TRACK_LOST, 0.2, source="track_lost")]),
    ])
    e = next(iter(t.events.values()))
    check("TRACK_LOST 升级同一事件", len(t.events) == 1 and e.kind == EventKind.TRACK_LOST,
          f"实际 {len(t.events)} / {e.kind.name}")
    check("TRACK_LOST → MEDIUM", e.suggested_urgency == Urgency.MEDIUM,
          f"实际 {e.suggested_urgency.name}")
    check("TRACK_LOST 保留了位置", e.bbox is not None)
    check("记录 TRACK_LOST 首次出现帧",
          e.to_dict()["kind_first_frame"].get("TRACK_LOST") == 2,
          str(e.to_dict()["kind_first_frame"]))
    check("记录 TRACK_LOST 自己的分数",
          e.to_dict()["kind_confidence"].get("TRACK_LOST") == 0.2,
          str(e.to_dict()["kind_confidence"]))


def test_urgency_override() -> None:
    t = EventTracker(EventTrackerConfig(min_hits=1))
    t.update(0, 0.0, [ev(EventKind.TRACK_LOST, 0.5, source="track_lost",
                         urgency=Urgency.HIGH)])
    e = next(iter(t.events.values()))
    check("来源可把 TRACK_LOST 建议为 HIGH", e.suggested_urgency == Urgency.HIGH,
          f"实际 {e.suggested_urgency.name}")

    # 后续低优先证据不能降级
    t.update(1, 1 / 30.0, [ev(EventKind.TRACK_LOST, 0.2, source="track_lost",
                              urgency=Urgency.MEDIUM)])
    check("紧急度只升不降", e.suggested_urgency == Urgency.HIGH,
          f"实际 {e.suggested_urgency.name}")

    t.update(2, 2 / 30.0, [ev(EventKind.FALL_CONFIRMED, 0.9, source="remote_pose")])
    check("确认后升到 CRITICAL", e.suggested_urgency == Urgency.CRITICAL,
          f"实际 {e.suggested_urgency.name}")


def test_ttl() -> None:
    t = EventTracker(EventTrackerConfig(max_gap_frames=100, ttl_frames=5, min_hits=1))
    ups = feed(t, [(0, [ev(EventKind.PERSON)]), (10, [])])
    e = next(iter(t.events.values()))
    check("TTL 到期关闭", e.state == EventState.EXPIRED, f"实际 {e.state.name}")
    check("TTL 关闭原因是 ttl",
          any(u.action == UpdateAction.CLOSE and "ttl" in u.reason for u in ups))


def test_bbox_and_reliability() -> None:
    e = Evidence(kind=EventKind.PERSON, confidence=1.7,
                 bbox=(0.5, 0.6, 0.1, 0.2), reliability=1.3)
    check("反向 bbox 被排序", e.bbox == (0.1, 0.2, 0.5, 0.6), f"实际 {e.bbox}")
    check("confidence 裁剪到 1", e.confidence == 1.0, f"实际 {e.confidence}")
    check("reliability 裁剪到 1", e.reliability == 1.0, f"实际 {e.reliability}")

    t = EventTracker(EventTrackerConfig(min_hits=1))
    t.update(0, 0.0, [ev(EventKind.SUSPECTED_FALL, 0.3, reliability=0.42)])
    d = next(iter(t.events.values())).to_dict()
    check("reliability 透传到快照", d["reliability"] == 0.42, f"实际 {d['reliability']}")


def test_json() -> None:
    t = EventTracker(EventTrackerConfig(min_hits=1))
    ups = feed(t, [(0, [ev(EventKind.SUSPECTED_FALL, 0.3, reliability=0.42)])])
    ups += t.close_all(1, 1 / 30.0)
    try:
        s = json.dumps([u.to_dict() for u in ups], ensure_ascii=False)
        ok = "SUSPECTED_FALL" in s and "suggested_urgency" in s
    except TypeError as exc:
        ok = False
        s = str(exc)
    check("快照可 JSON 序列化", ok, s[:120])


def main() -> int:
    for fn in (test_single_track, test_short_gap, test_long_gap, test_two_people,
               test_escalation, test_track_lost, test_urgency_override, test_ttl,
               test_bbox_and_reliability, test_json):
        print(f"\n── {fn.__name__} ──")
        try:
            fn()
        except Exception as exc:  # 单条测试炸掉时继续跑后面的，最后统一报失败
            FAILED.append(fn.__name__)
            print(f"✗ {fn.__name__} 抛异常：{type(exc).__name__}: {exc}")

    print()
    if FAILED:
        print(f"失败 {len(FAILED)} 项：{FAILED}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
