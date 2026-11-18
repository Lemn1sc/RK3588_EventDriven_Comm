"""事件形成器 · 把逐帧证据聚合成带生命周期的事件。

── 为什么需要这一层 ──

现有脚本输出的是**逐帧行**（`has_event / confidence / bbox / severity ...`），但通信
控制端要调度的不是"帧"，而是"事件"：

- 一次跌倒横跨几十帧，控制端只需要知道"事件 7 开始了、现在置信度多少、还在不在"；
- 人躺地后检测器失效，逐帧行只留下 `has_event=0`，和"画面本来就没人"无法区分；
  事件层可以把它表示成 `TRACK_LOST` / `SUSPECTED_FALL` 的**持续事件**；
- 调度需要 `start_ts / suggested_urgency / reliability / payload`，这些都属于事件。

本模块刻意不依赖 numpy / cv2 / torch：纯 Python + 标准库，任何环境可裸跑，
`scripts/check_event_tracker.py` 用合成序列做回归测试。

── 事件生命周期 ──

    candidate ──(命中 >= min_hits)──> active ──(收到 FALL_CONFIRMED)──> confirmed
        │                                  │
        └────── gap > max_gap / age > ttl ──┴──> expired

- `candidate`：只出现 1 帧的疑似事件（可能是噪声）；
- `active`：连续命中，已经值得控制端关注；
- `confirmed`：远端 / 高置信确认（由调用方发 `FALL_CONFIRMED` 证据）；
- `expired`：证据断档超过 `max_gap_frames`，或事件寿命超过 `ttl_frames`。

“证据断档”用**帧号差**而不是“连续未命中计数”判断 —— 帧表可能跳过未标注区间，
帧号会跳变，计数会把一次长间隔误当成很多帧。见 `EventTracker.update()`。

── 紧急度只是“建议” ──

事件层给 `suggested_urgency`，最终是否传、何时传、传多大，由控制端结合信道状态决定。
不要把调度决策塞进事件层，否则事件层会重新变成一个隐式的通信控制器。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Mapping

__all__ = [
    "EventKind", "Urgency", "EventState", "UpdateAction",
    "Evidence", "Event", "EventUpdate", "EventTrackerConfig", "EventTracker",
]


class EventKind(IntEnum):
    """事件类型。数值只用于协议编码，语义见名称。"""
    PERSON = 1            # 检测到人，姿态/行为正常
    PERSON_ABNORMAL = 2   # 人停留超时、姿态异常等（还未到跌倒疑似）
    TRACK_LOST = 3        # 之前跟踪到的人突然消失（可能是走出画面，也可能是跌倒）
    SUSPECTED_FALL = 4    # 疑似跌倒（低置信躺姿、轨迹消失+下沉、前景大 blob 等）
    FALL_CONFIRMED = 5    # 远端姿态 / 人工确认跌倒
    BACKGROUND = 6        # 心跳、遥测等背景事件


class Urgency(IntEnum):
    """控制端调度用的紧急度。事件层只给建议值。"""
    BACKGROUND = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EventState(IntEnum):
    CANDIDATE = 0
    ACTIVE = 1
    CONFIRMED = 2
    EXPIRED = 3


class UpdateAction(IntEnum):
    START = 0
    UPDATE = 1
    CLOSE = 2


# 事件类型 → 建议紧急度。原则：**漏报代价高的类型给高紧急度**，
# 不按检测置信度打折 —— 低置信躺姿恰恰是最需要远端确认的情况。
_KIND_URGENCY: dict[EventKind, Urgency] = {
    EventKind.PERSON: Urgency.LOW,
    EventKind.PERSON_ABNORMAL: Urgency.MEDIUM,
    EventKind.TRACK_LOST: Urgency.MEDIUM,
    EventKind.SUSPECTED_FALL: Urgency.HIGH,
    EventKind.FALL_CONFIRMED: Urgency.CRITICAL,
    EventKind.BACKGROUND: Urgency.BACKGROUND,
}

# 升级顺序：只升不降。一个事件从 PERSON 演化到 SUSPECTED_FALL 时保留同一个 id。
_KIND_RANK: dict[EventKind, int] = {
    EventKind.BACKGROUND: 0,
    EventKind.PERSON: 1,
    EventKind.PERSON_ABNORMAL: 2,
    EventKind.TRACK_LOST: 3,
    EventKind.SUSPECTED_FALL: 4,
    EventKind.FALL_CONFIRMED: 5,
}

# 这些类型都描述“人”，允许互相匹配（同一个人从正常到疑似跌倒应升级同一个事件）。
_PERSON_KINDS = frozenset({
    EventKind.PERSON, EventKind.PERSON_ABNORMAL, EventKind.TRACK_LOST,
    EventKind.SUSPECTED_FALL, EventKind.FALL_CONFIRMED,
})


def _norm_bbox(bbox) -> tuple[float, float, float, float] | None:
    """归一化 xyxy：排序、裁剪到 [0,1]。非法输入返回 None。"""
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(v == v for v in (x1, y1, x2, y2)):   # NaN
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
    y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
    return (x1, y1, x2, y2)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _center_dist(a, b) -> float | None:
    if a is None or b is None:
        return None
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


@dataclass(frozen=True)
class Evidence:
    """一帧里的一条证据。由检测器 / 前景 / 轨迹管理 / 远端确认等来源产生。

    `confidence` 是来源自己的置信度；`reliability` 是**校准后**的“该事件为真”的概率
    （可选）。事件层不负责校准，但会把它透传给控制端。
    """
    kind: EventKind
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    source: str = "detector"
    reliability: float | None = None
    # 可选：来源建议的紧急度。事件层只升不降；例如高分的 TRACK_LOST 可以建议 HIGH，
    # 而不用假装它已经是 SUSPECTED_FALL。控制端仍按信道/队列做最终调度。
    urgency: Urgency | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen dataclass 用 object.__setattr__ 归一化；不归一化的话下游要处理
        # 反向框 / 越界框，很容易在 IoU 匹配上出现“看起来对、其实匹配错人”。
        object.__setattr__(self, "confidence", float(min(1.0, max(0.0, self.confidence))))
        object.__setattr__(self, "bbox", _norm_bbox(self.bbox))
        if self.reliability is not None:
            object.__setattr__(self, "reliability",
                               float(min(1.0, max(0.0, self.reliability))))
        if self.urgency is not None:
            object.__setattr__(self, "urgency", Urgency(int(self.urgency)))


@dataclass
class Event:
    """一个持续存在的事件。id 在同一个 tracker 内单调递增。"""
    id: int
    kind: EventKind
    state: EventState
    start_frame: int
    start_ts: float
    last_seen_frame: int
    last_seen_ts: float
    confidence: float
    peak_conf: float
    bbox: tuple[float, float, float, float] | None
    sources: set[str] = field(default_factory=set)
    hits: int = 1
    missed: int = 0
    reliability: float | None = None
    suggested_urgency: Urgency = Urgency.LOW
    closed_frame: int | None = None
    # 每种类型第一次出现的帧 / 时间。用于区分“事件从人出现就开始了”和
    # “事件在某一帧才升级成告警”—— 把前者当提前量会虚报预警时间。
    kind_first_frame: dict[int, int] = field(default_factory=dict)
    kind_first_ts: dict[int, float] = field(default_factory=dict)
    # 每种类型第一次出现时的 confidence。事件最终 confidence 是“最后一次证据”的，
    # 一个 PERSON 事件升级成 SUSPECTED_FALL 后，最终 confidence 会被后续检测覆盖；
    # 想复盘“TRACK_LOST 当时打了多少分”必须看这里。
    kind_confidence: dict[int, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def observe(self, frame_index: int, timestamp: float, ev: Evidence,
                min_hits: int) -> tuple[tuple[EventState, EventKind], tuple[EventState, EventKind]]:
        """吸收一条证据，返回 (更新前状态, 更新后状态)，供调用方判断是否发 UPDATE。"""
        before = (self.state, self.kind)
        # 记录“本帧证据类型”第一次出现的时刻。注意这里记的是 ev.kind 而不是
        # self.kind：一个 PERSON 事件在第 30 帧收到 SUSPECTED_FALL 证据时，
        # 我们要的是第 30 帧（告警起点），不是第 0 帧（人出现）。
        self.kind_first_frame.setdefault(int(ev.kind), int(frame_index))
        self.kind_first_ts.setdefault(int(ev.kind), float(timestamp))
        self.kind_confidence.setdefault(int(ev.kind), float(ev.confidence))
        # 保留**最近一条**证据的解释字段（dy/d_aspect/score/reasons…），
        # 方便控制端和调试看到“为什么这个事件被升级/触发”。
        # 用替换而不是 update：否则旧证据的 score 会残留，和当前证据对不上。
        self.meta = dict(ev.meta)
        self.last_seen_frame = int(frame_index)
        self.last_seen_ts = float(timestamp)
        self.confidence = float(ev.confidence)
        self.peak_conf = max(self.peak_conf, float(ev.confidence))
        self.missed = 0
        self.hits += 1
        self.sources.add(ev.source)
        if ev.bbox is not None:
            self.bbox = ev.bbox
        if ev.reliability is not None:
            self.reliability = float(ev.reliability)

        # 只升不降：PERSON → SUSPECTED_FALL 时保留同一事件，避免“同一次跌倒
        # 被拆成一个人事件 + 一个跌倒事件”造成重复计数。
        if _KIND_RANK[ev.kind] > _KIND_RANK[self.kind]:
            self.kind = ev.kind
            self.suggested_urgency = _KIND_URGENCY[ev.kind]

        # 来源可以只调“紧急度”而不改类型。只升不降：后续低分证据不会把
        # 一个已经 HIGH 的 TRACK_LOST 降回 MEDIUM。
        if ev.urgency is not None:
            self.suggested_urgency = Urgency(
                max(int(self.suggested_urgency), int(ev.urgency)))

        if self.kind == EventKind.FALL_CONFIRMED:
            self.state = EventState.CONFIRMED
        elif self.state == EventState.CANDIDATE and self.hits >= min_hits:
            self.state = EventState.ACTIVE
        return before, (self.state, self.kind)

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化的快照。字段名同时给协议用的数值 id 和人读的名称。"""
        return {
            "event_id": int(self.id),
            "kind": self.kind.name,
            "kind_id": int(self.kind),
            "state": self.state.name,
            "state_id": int(self.state),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.last_seen_frame),
            "start_ts": round(float(self.start_ts), 6),
            "end_ts": round(float(self.last_seen_ts), 6),
            "duration_s": round(max(0.0, float(self.last_seen_ts - self.start_ts)), 6),
            "confidence": round(float(self.confidence), 6),
            "peak_confidence": round(float(self.peak_conf), 6),
            "reliability": (None if self.reliability is None
                            else round(float(self.reliability), 6)),
            "suggested_urgency": self.suggested_urgency.name,
            "urgency_id": int(self.suggested_urgency),
            "bbox": (None if self.bbox is None
                     else [round(float(v), 6) for v in self.bbox]),
            "sources": sorted(self.sources),
            "hits": int(self.hits),
            "missed_frames": int(self.missed),
            "closed_frame": (None if self.closed_frame is None else int(self.closed_frame)),
            "kind_first_frame": {EventKind(k).name: int(v)
                                 for k, v in sorted(self.kind_first_frame.items())},
            "kind_first_ts": {EventKind(k).name: round(float(v), 6)
                              for k, v in sorted(self.kind_first_ts.items())},
            "kind_confidence": {EventKind(k).name: round(float(v), 6)
                                for k, v in sorted(self.kind_confidence.items())},
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class EventUpdate:
    """事件流的一条消息。控制端消费 start / update / close，而不是逐帧行。"""
    action: UpdateAction
    event: dict[str, Any]
    frame_index: int
    timestamp: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.name,
            "action_id": int(self.action),
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 6),
            "reason": self.reason,
            "event": self.event,
        }


@dataclass(frozen=True)
class EventTrackerConfig:
    """事件形成参数。帧数单位与输入帧率绑定（URFD 30fps / MCF ~25fps）。"""
    max_gap_frames: int = 30      # 证据断档超过该帧数 → 事件关闭
    ttl_frames: int = 300         # 事件最长寿命（约 10s @30fps），防止陈旧告警堆积
    min_hits: int = 2             # candidate → active 所需命中数
    match_iou: float = 0.10       # 同一个人跨帧匹配的最低 IoU
    match_center_dist: float = 0.20  # 无 IoU 时的质心距离上限（归一化坐标）
    update_every: int = 0         # >0 时每 N 帧对 active 事件补一条 UPDATE（0=仅状态变化）


class EventTracker:
    """把逐帧证据流聚合成事件流。

    典型用法::

        tracker = EventTracker()
        for frame in frames:
            updates = tracker.update(frame.index, frame.timestamp, frame.evidences)
            for u in updates:
                send_to_controller(u.to_dict())
        for u in tracker.close_all(last_index, last_ts):
            send_to_controller(u.to_dict())
    """

    def __init__(self, config: EventTrackerConfig | None = None):
        self.config = config or EventTrackerConfig()
        self.events: dict[int, Event] = {}
        self._next_id = 1

    # ── 匹配 ──────────────────────────────────────────────────────

    def _match_score(self, event: Event, ev: Evidence) -> float:
        """返回 >=0 的匹配分；-1 表示不匹配。分数越大越可能属于同一目标。"""
        if not (event.kind == ev.kind or
                (event.kind in _PERSON_KINDS and ev.kind in _PERSON_KINDS)):
            return -1.0
        if event.bbox is None or ev.bbox is None:
            # 没有位置信息时只能弱匹配：给一个很小的正分，避免抢掉有位置的候选。
            return 0.01
        iou = _iou(event.bbox, ev.bbox)
        if iou >= self.config.match_iou:
            return iou
        dist = _center_dist(event.bbox, ev.bbox)
        if dist is not None and dist <= self.config.match_center_dist:
            # 距离越近分越高；上限略高于 0，保证仍低于任何正 IoU。
            return 0.5 * (1.0 - dist / max(self.config.match_center_dist, 1e-9))
        return -1.0

    def _open_events(self) -> list[Event]:
        return [e for e in self.events.values() if e.state != EventState.EXPIRED]

    # ── 主入口 ────────────────────────────────────────────────────

    def update(self, frame_index: int, timestamp: float,
               evidences: Iterable[Evidence]) -> list[EventUpdate]:
        """处理一帧。返回本帧产生的 START / UPDATE / CLOSE 消息。"""
        frame_index = int(frame_index)
        timestamp = float(timestamp)
        updates: list[EventUpdate] = []
        used: set[int] = set()

        # 先关闭“已经断档”的事件，再拿本帧证据去匹配。
        #
        # 为什么必须先关：匹配只看空间（IoU/质心），不看时间。如果帧号从 1 直接跳到
        # 10（帧表跳过未标注区间，或视频有丢帧），同一位置的证据会被错误地接回
        # 8 帧前的老事件 —— 看起来像“一直有人”，其实是两段不同的事。实测这条
        # 会把 MCF 里相隔很远的两个标注区间合并成一个事件。
        stale = [e for e in self._open_events()
                 if (frame_index - e.last_seen_frame > self.config.max_gap_frames
                     or frame_index - e.start_frame > self.config.ttl_frames)]
        for event in sorted(stale, key=lambda e: e.id):
            gap = frame_index - event.last_seen_frame
            age = frame_index - event.start_frame
            if gap > self.config.max_gap_frames:
                self._close(event, frame_index, timestamp, updates,
                            reason=f"gap {gap}>{self.config.max_gap_frames}")
            else:
                self._close(event, frame_index, timestamp, updates,
                            reason=f"ttl {age}>{self.config.ttl_frames}")

        # 高紧急度 / 高置信证据先匹配，避免 PERSON 抢走本该升级的 SUSPECTED_FALL。
        ordered = sorted(evidences,
                         key=lambda e: (-_KIND_RANK[e.kind], -e.confidence))

        for ev in ordered:
            best: Event | None = None
            best_score = -1.0
            for event in self._open_events():
                if event.id in used:
                    continue
                score = self._match_score(event, ev)
                if score > best_score:
                    best, best_score = event, score

            if best is not None and best_score >= 0.0:
                before, after = best.observe(frame_index, timestamp, ev, self.config.min_hits)
                used.add(best.id)
                if before != after:
                    updates.append(EventUpdate(
                        UpdateAction.UPDATE, best.to_dict(), frame_index, timestamp,
                        reason=f"{before[0].name}/{before[1].name} → "
                               f"{after[0].name}/{after[1].name}"))
                elif self.config.update_every > 0 and frame_index % self.config.update_every == 0:
                    updates.append(EventUpdate(
                        UpdateAction.UPDATE, best.to_dict(), frame_index, timestamp,
                        reason="periodic"))
            else:
                event = Event(
                    id=self._next_id,
                    kind=ev.kind,
                    state=(EventState.CONFIRMED if ev.kind == EventKind.FALL_CONFIRMED
                           else (EventState.ACTIVE if self.config.min_hits <= 1
                                 else EventState.CANDIDATE)),
                    start_frame=frame_index,
                    start_ts=timestamp,
                    last_seen_frame=frame_index,
                    last_seen_ts=timestamp,
                    confidence=float(ev.confidence),
                    peak_conf=float(ev.confidence),
                    bbox=ev.bbox,
                    sources={ev.source},
                    hits=1,
                    missed=0,
                    reliability=ev.reliability,
                    suggested_urgency=(ev.urgency if ev.urgency is not None
                                       else _KIND_URGENCY[ev.kind]),
                    kind_first_frame={int(ev.kind): frame_index},
                    kind_first_ts={int(ev.kind): timestamp},
                    kind_confidence={int(ev.kind): float(ev.confidence)},
                    meta=dict(ev.meta),
                )
                self._next_id += 1
                self.events[event.id] = event
                used.add(event.id)
                updates.append(EventUpdate(
                    UpdateAction.START, event.to_dict(), frame_index, timestamp,
                    reason=f"new {ev.kind.name} from {ev.source}"))

        # 未在本帧获得证据的事件：按**帧号差**判断断档。
        for event in self._open_events():
            if event.id in used:
                continue
            event.missed += 1
            gap = frame_index - event.last_seen_frame
            age = frame_index - event.start_frame
            if gap > self.config.max_gap_frames:
                self._close(event, frame_index, timestamp, updates,
                            reason=f"gap {gap}>{self.config.max_gap_frames}")
            elif age > self.config.ttl_frames:
                self._close(event, frame_index, timestamp, updates,
                            reason=f"ttl {age}>{self.config.ttl_frames}")
            elif self.config.update_every > 0 and frame_index % self.config.update_every == 0:
                updates.append(EventUpdate(
                    UpdateAction.UPDATE, event.to_dict(), frame_index, timestamp,
                    reason="periodic"))

        return updates

    def _close(self, event: Event, frame_index: int, timestamp: float,
               updates: list[EventUpdate], reason: str) -> None:
        event.state = EventState.EXPIRED
        event.closed_frame = int(frame_index)
        updates.append(EventUpdate(UpdateAction.CLOSE, event.to_dict(),
                                   frame_index, timestamp, reason=reason))

    def close_all(self, frame_index: int, timestamp: float) -> list[EventUpdate]:
        """输入流结束 / 视频读完时关闭所有未结束事件。"""
        updates: list[EventUpdate] = []
        for event in self._open_events():
            self._close(event, frame_index, timestamp, updates, reason="end_of_stream")
        return updates

    def snapshots(self, include_expired: bool = True) -> list[dict[str, Any]]:
        items = [e for e in self.events.values()
                 if include_expired or e.state != EventState.EXPIRED]
        return [e.to_dict() for e in sorted(items, key=lambda x: x.id)]

    def active_snapshots(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in sorted(self.events.values(), key=lambda x: x.id)
                if e.state in (EventState.CANDIDATE, EventState.ACTIVE, EventState.CONFIRMED)]

    def __len__(self) -> int:
        return len(self.events)
