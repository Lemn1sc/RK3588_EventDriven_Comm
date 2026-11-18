"""edge_node · 事件端（事件形成 / 证据 / 载荷）。

刻意**不在这里 import 具体模块**：`payload.py` 依赖 cv2，而事件形成器是纯 Python。
调用方按需 `from edge_node.event_tracker import ...` 或 `from edge_node.payload import ...`，
这样纯 numpy/Python 的自检不需要拉起 OpenCV。

事件向量（8 维）仍由 `common.event_vector` 定义；本包只做**事件级**封装：
把逐帧证据组织成带生命周期的事件，供通信控制端调度。
"""
from .event_tracker import (
    Event,
    EventKind,
    EventState,
    EventTracker,
    EventTrackerConfig,
    EventUpdate,
    Evidence,
    UpdateAction,
    Urgency,
)

__all__ = [
    "Event", "EventKind", "EventState", "EventTracker", "EventTrackerConfig",
    "EventUpdate", "Evidence", "UpdateAction", "Urgency",
]
