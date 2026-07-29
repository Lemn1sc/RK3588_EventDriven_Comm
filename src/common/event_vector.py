"""事件向量（8 维）· 数据契约核心（T0.3）。

事件端（yolov8n 检测）每帧输出一个定长 8 维向量，**不做任何通信决策**；
下游 GRU 控制端消费该向量的时序做卸载/传输决策。定义见 `docs/02_技术方案.md`。

布局（float32, shape=(8,)）：
    [0] event_type  事件类型枚举（见 EventType，0=无事件）
    [1] confidence  主目标检测置信度 0~1
    [2] bbox_x1     归一化左上 x
    [3] bbox_y1     归一化左上 y
    [4] bbox_x2     归一化右下 x
    [5] bbox_y2     归一化右下 y
    [6] has_event   是否有目标（二值 0/1）
    [7] severity    综合严重程度 0~1（置信度 × 高宽比异常度）

约定：一帧只表征「最严重的主目标」。多人场景取 severity 最高者；
高宽比突变 / 停留超时 / 位置异常等时序判定不在此层，属 GRU 职责。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

__all__ = [
    "EVENT_DIM", "EventType", "EventVector", "NONE_EVENT",
    "IDX_EVENT_TYPE", "IDX_CONFIDENCE", "IDX_BBOX", "IDX_HAS_EVENT", "IDX_SEVERITY",
    "COCO_PERSON", "aspect_abnormality", "build_event",
    "event_from_detections", "event_from_result",
]

EVENT_DIM = 8

# ── 向量索引布局（下游按名取值，勿用魔法数字）──
IDX_EVENT_TYPE = 0
IDX_CONFIDENCE = 1
IDX_BBOX = slice(2, 6)  # x1, y1, x2, y2（归一化）
IDX_HAS_EVENT = 6
IDX_SEVERITY = 7


class EventType(IntEnum):
    """事件类型枚举（对应组合3 跌倒检测的边缘端可见状态）。"""
    NONE = 0       # 无事件 / 画面无人 → 不通信
    PERSON = 1     # 检测到人，姿态正常 → 不通信
    SUSPECTED = 2  # 疑似异常（高宽比偏低等）→ 建议卸载确认


# COCO 中 person 的类别号（yolov8n 检测跌倒只关心此类）
COCO_PERSON = 0

# 高宽比阈值：站立 h/w 约 2~3，躺卧/跌倒 < 1（实测见 docs/07）。
# 低于此值视为几何异常，用于合成 severity 与判定 SUSPECTED。
ASPECT_UPRIGHT = 1.5   # ≥ 该值视为直立，异常度 0
ASPECT_LYING = 0.7     # ≤ 该值视为完全躺卧，异常度 1
SEVERITY_SUSPECT = 0.5  # severity ≥ 该值升级为 SUSPECTED


@dataclass
class EventVector:
    """8 维事件向量的结构化视图。to_array() 得定长 np.float32 用于传输/入网。"""
    event_type: EventType = EventType.NONE
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # 归一化 x1y1x2y2
    has_event: bool = False
    severity: float = 0.0

    def to_array(self) -> np.ndarray:
        v = np.zeros(EVENT_DIM, dtype=np.float32)
        v[IDX_EVENT_TYPE] = float(int(self.event_type))
        v[IDX_CONFIDENCE] = self.confidence
        v[IDX_BBOX] = self.bbox
        v[IDX_HAS_EVENT] = 1.0 if self.has_event else 0.0
        v[IDX_SEVERITY] = self.severity
        return v

    @classmethod
    def from_array(cls, v: np.ndarray) -> "EventVector":
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] != EVENT_DIM:
            raise ValueError(f"事件向量维度应为 {EVENT_DIM}，实得 {v.shape[0]}")
        return cls(
            event_type=EventType(int(round(float(v[IDX_EVENT_TYPE])))),
            confidence=float(v[IDX_CONFIDENCE]),
            bbox=tuple(float(x) for x in v[IDX_BBOX]),  # type: ignore[arg-type]
            has_event=bool(v[IDX_HAS_EVENT] >= 0.5),
            severity=float(v[IDX_SEVERITY]),
        )


# 全零向量：无人帧的规范表示，供边缘端「无事件」时直接发送
NONE_EVENT = EventVector()


def aspect_abnormality(x1: float, y1: float, x2: float, y2: float) -> float:
    """由归一化框算高宽比异常度 0~1。直立→0，躺卧→1（线性插值）。"""
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 0.0)
    ratio = h / w  # 高宽比：站立大，躺卧小
    if ratio >= ASPECT_UPRIGHT:
        return 0.0
    if ratio <= ASPECT_LYING:
        return 1.0
    return (ASPECT_UPRIGHT - ratio) / (ASPECT_UPRIGHT - ASPECT_LYING)


def build_event(bbox_xyxyn, confidence: float) -> EventVector:
    """由单个人体框（归一化 xyxy）+ 置信度构造事件向量。

    severity = 置信度 × 高宽比异常度：正常站立≈0，躺卧/跌倒且高置信→高。
    低置信度自动打折，避免鬼影框误报。
    """
    x1, y1, x2, y2 = (float(c) for c in bbox_xyxyn)
    conf = float(confidence)
    abn = aspect_abnormality(x1, y1, x2, y2)
    severity = conf * abn
    etype = EventType.SUSPECTED if severity >= SEVERITY_SUSPECT else EventType.PERSON
    return EventVector(
        event_type=etype,
        confidence=conf,
        bbox=(x1, y1, x2, y2),
        has_event=True,
        severity=severity,
    )


def event_from_detections(boxes_xyxyn, confs, classes, conf_thres: float = 0.25) -> EventVector:
    """由一帧的检测数组构造事件向量（框架无关，只吃 numpy）。

    参数
        boxes_xyxyn: (N,4) 归一化 xyxy 框
        confs:       (N,)  置信度
        classes:     (N,)  COCO 类别号
        conf_thres:  过滤阈值，低于此值的框忽略
    规则：过滤出 person 且过阈值的框，取 severity 最高者为主目标；无则 NONE_EVENT。
    """
    boxes = np.asarray(boxes_xyxyn, dtype=np.float32).reshape(-1, 4)
    confs = np.asarray(confs, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes).reshape(-1).astype(int)
    if boxes.shape[0] == 0:
        return NONE_EVENT

    best: EventVector | None = None
    for i in range(boxes.shape[0]):
        if classes[i] != COCO_PERSON or confs[i] < conf_thres:
            continue
        ev = build_event(boxes[i], confs[i])
        if best is None or ev.severity > best.severity:
            best = ev
    return best if best is not None else NONE_EVENT


def event_from_result(result, conf_thres: float = 0.25) -> EventVector:
    """从 ultralytics Results 对象直接构造（供 edge_node 便捷调用）。

    仅取 boxes.xyxyn/conf/cls；无框时返回 NONE_EVENT。不硬依赖 ultralytics。
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return NONE_EVENT
    xyxyn = boxes.xyxyn.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    return event_from_detections(xyxyn, confs, classes, conf_thres)
