from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


BBox = Tuple[float, float, float, float]


@dataclass
class FrameSample:
    frame_id: int
    timestamp: float
    person_present: bool
    bbox: Optional[BBox] = None
    in_danger_zone: bool = False


@dataclass
class Decision:
    trigger: bool
    reason: str
    stable: bool = False
    motion_px: float = 0.0
    aspect_ratio: float = 0.0
    meta: Dict[str, float] = field(default_factory=dict)


class RuleEventEngine:
    def __init__(
        self,
        stable_ratio_tol: float = 0.08,
        static_move_px: float = 6.0,
        static_seconds: float = 3.0,
        fall_width_gain: float = 0.25,
        fall_height_drop: float = 0.20,
        history_size: int = 8,
    ) -> None:
        self.stable_ratio_tol = stable_ratio_tol
        self.static_move_px = static_move_px
        self.static_seconds = static_seconds
        self.fall_width_gain = fall_width_gain
        self.fall_height_drop = fall_height_drop
        self.history_size = history_size
        self.reset()

    def reset(self) -> None:
        self.prev_bbox: Optional[BBox] = None
        self.prev_center: Optional[Tuple[float, float]] = None
        self.static_start_time: Optional[float] = None
        self.aspect_history: List[float] = []
        self.center_history: List[Tuple[float, float]] = []

    @staticmethod
    def _bbox_stats(bbox: BBox) -> Tuple[float, float, float, float, float]:
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        ratio = width / height if height > 0 else 0.0
        return width, height, cx, cy, ratio

    def update(self, sample: FrameSample) -> Decision:
        if not sample.person_present or sample.bbox is None:
            self.reset()
            return Decision(trigger=False, reason="no_person")

        width, height, cx, cy, ratio = self._bbox_stats(sample.bbox)
        if width <= 0 or height <= 0:
            self.reset()
            return Decision(trigger=False, reason="invalid_bbox")

        motion_px = 0.0
        if self.prev_center is not None:
            motion_px = math.hypot(cx - self.prev_center[0], cy - self.prev_center[1])

        self.aspect_history.append(ratio)
        self.center_history.append((cx, cy))
        self.aspect_history = self.aspect_history[-self.history_size :]
        self.center_history = self.center_history[-self.history_size :]

        stable = False
        if len(self.aspect_history) >= 2:
            stable = (max(self.aspect_history) - min(self.aspect_history)) <= self.stable_ratio_tol

        if sample.in_danger_zone:
            self._remember(sample.bbox, (cx, cy))
            return Decision(
                trigger=True,
                reason="danger_zone",
                stable=stable,
                motion_px=motion_px,
                aspect_ratio=ratio,
                meta={"width": width, "height": height},
            )

        if self.prev_bbox is not None:
            prev_width, prev_height, _, _, _ = self._bbox_stats(self.prev_bbox)
            width_gain = (width - prev_width) / prev_width if prev_width > 0 else 0.0
            height_drop = (prev_height - height) / prev_height if prev_height > 0 else 0.0
            if width_gain >= self.fall_width_gain and height_drop >= self.fall_height_drop:
                self._remember(sample.bbox, (cx, cy))
                return Decision(
                    trigger=True,
                    reason="fall_suspicious",
                    stable=stable,
                    motion_px=motion_px,
                    aspect_ratio=ratio,
                    meta={
                        "width_gain": width_gain,
                        "height_drop": height_drop,
                        "width": width,
                        "height": height,
                    },
                )

        if motion_px <= self.static_move_px:
            if self.static_start_time is None:
                self.static_start_time = sample.timestamp
        else:
            self.static_start_time = sample.timestamp

        static_elapsed = 0.0
        if self.static_start_time is not None:
            static_elapsed = sample.timestamp - self.static_start_time

        self._remember(sample.bbox, (cx, cy))

        if static_elapsed >= self.static_seconds:
            return Decision(
                trigger=True,
                reason="static_too_long",
                stable=stable,
                motion_px=motion_px,
                aspect_ratio=ratio,
                meta={"static_seconds": static_elapsed},
            )

        return Decision(
            trigger=False,
            reason="stable_normal" if stable else "normal",
            stable=stable,
            motion_px=motion_px,
            aspect_ratio=ratio,
            meta={"static_seconds": static_elapsed},
        )

    def _remember(self, bbox: BBox, center: Tuple[float, float]) -> None:
        self.prev_bbox = bbox
        self.prev_center = center
