"""common · 全项目共享的数据契约（T0.3）。

已固化：
- event_vector: 8 维事件向量（事件端产出，GRU 消费）
待定（与协作者讨论后补）：
- 状态向量（12 维，GRU 输入）
- 通信协议（UDP 包头格式）
"""
from .event_vector import (
    EVENT_DIM,
    NONE_EVENT,
    EventType,
    EventVector,
    build_event,
    event_from_detections,
    event_from_result,
)

__all__ = [
    "EVENT_DIM", "EventType", "EventVector", "NONE_EVENT",
    "build_event", "event_from_detections", "event_from_result",
]
