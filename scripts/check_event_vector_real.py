"""T0.3 · 事件向量对「真实模型输出」的验证（需 ultralytics，跑在 rk 环境）。

`check_event_vector.py` 用合成 numpy 数组验契约本身，不依赖 ultralytics；
本脚本补上另一半：`event_from_result()` 直接吃真实 Results 对象，
确认 boxes.xyxyn/conf/cls 的取值路径、dtype、归一化范围都对得上。

用法：python scripts/check_event_vector_real.py
     看到 "真实输出验证 OK ✓" 即通过。
"""
import io
import sys
from pathlib import Path

import numpy as np
import torch

# Windows 控制台默认 GBK，打不出 ✓。用 isinstance 而非 hasattr：
# typeshed 把 sys.stdout 标注为 TextIO（无 reconfigure），isinstance 能让类型检查器收窄。
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import ASSETS, YOLO  # noqa: E402
from ultralytics.engine.results import Boxes, Results  # noqa: E402

from common.event_vector import (  # noqa: E402
    EVENT_DIM, EventType, EventVector,
    event_from_detections, event_from_result,
)

MODEL = YOLO(str(ROOT / "models" / "pt" / "yolov8n.pt"))


def predict(source) -> Results:
    return MODEL.predict(source, verbose=False)[0]


def to_numpy(x) -> np.ndarray:
    """统一转 numpy。

    ultralytics 把 Boxes.xyxyn 等属性标注为 NDArray，但运行时按来源可能是
    torch 张量（本项目实测 torch.float32，可能在 GPU 上）。两种都吃，
    顺带把 GPU 张量搬回 CPU —— 这也是 event_from_result() 内部做的事。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def boxes_of(r: Results) -> Boxes:
    """取 r.boxes 并断言非 None。

    Results.boxes 标注为 Boxes | None（pose/seg 等任务可能没有该字段），
    检测模型必有；显式断言既满足类型检查，也能在模型选错时立刻报错。
    """
    assert r.boxes is not None, "检测模型的 Results.boxes 不该为 None，请确认加载的是 yolov8n.pt"
    return r.boxes


def dump(r: Results, tag: str) -> None:
    """打印一帧里每个框的几何量，便于人工核对判定是否合理。"""
    b_all = boxes_of(r)
    xyxyn = to_numpy(b_all.xyxyn)
    confs = to_numpy(b_all.conf)
    classes = to_numpy(b_all.cls)
    print(f"  {tag}: {len(b_all)} 框")
    for i in range(len(b_all)):
        b = xyxyn[i]
        cid = int(classes[i])
        hw = (b[3] - b[1]) / max(b[2] - b[0], 1e-6)
        print(f"    cls={cid}({MODEL.names[cid]}) conf={float(confs[i]):.3f} h/w={hw:.2f}")


def check_upright() -> Results:
    """bus.jpg：4 个直立行人 + 公交车 → 主目标 PERSON，severity≈0，不该触发卸载。"""
    r = predict(str(ASSETS / "bus.jpg"))
    dump(r, "bus.jpg")
    ev = event_from_result(r)
    arr = ev.to_array()
    assert arr.shape == (EVENT_DIM,) and arr.dtype == np.float32
    assert ev.event_type == EventType.PERSON, ev.event_type
    assert ev.has_event is True
    assert ev.severity < 0.1, ev.severity
    # 归一化框必须落在 0~1，否则说明误取了像素坐标 xyxy
    assert all(0.0 <= c <= 1.0 for c in ev.bbox), ev.bbox
    # 往返一致：真实数据也要能过序列化
    assert EventVector.from_array(arr).event_type == EventType.PERSON
    print(f"    → {ev.event_type.name} severity={ev.severity:.3f} bbox 归一化 ✓")
    print("  [1] 真实 Results 取值路径 + 直立场景不误报 ✓")
    return r


def check_consistency(r: Results) -> None:
    """同一帧走 Results 路径与走 numpy 路径必须得到完全一致的向量。"""
    b_all = boxes_of(r)
    from_result = event_from_result(r).to_array()
    from_numpy = event_from_detections(
        to_numpy(b_all.xyxyn),
        to_numpy(b_all.conf),
        to_numpy(b_all.cls),
    ).to_array()
    assert np.allclose(from_result, from_numpy), (from_result, from_numpy)
    print("  [2] Results 路径与 numpy 路径结果一致 ✓")


def check_suspected() -> None:
    """zidane.jpg：一个 h/w≈0.9 的扁框人 + 一个直立人 + 一条领带(cls 27)。

    验三件事：扁框人被选为主目标并升级 SUSPECTED、非 person 类被忽略、
    多人时按 severity 而非置信度挑选（直立那位 conf 只低 0.002）。
    """
    r = predict(str(ASSETS / "zidane.jpg"))
    dump(r, "zidane.jpg")
    ev = event_from_result(r)
    assert ev.event_type == EventType.SUSPECTED, ev.event_type
    assert ev.severity >= 0.5, ev.severity
    # 主目标必须是那个扁框（h/w<1），不是领带也不是直立者
    h_w = (ev.bbox[3] - ev.bbox[1]) / max(ev.bbox[2] - ev.bbox[0], 1e-6)
    assert h_w < 1.0, h_w
    print(f"    → {ev.event_type.name} severity={ev.severity:.3f} 主目标 h/w={h_w:.2f} ✓")
    print("  [3] 真实扁框场景升级 SUSPECTED + 忽略非 person 类 ✓")


def check_empty() -> None:
    """纯色空帧：无框 → NONE_EVENT（全零向量），边缘端据此不通信。"""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    r = predict(blank)
    n = len(boxes_of(r))
    assert n == 0, n
    ev = event_from_result(r)
    assert ev.event_type == EventType.NONE and ev.has_event is False
    assert np.allclose(ev.to_array(), np.zeros(EVENT_DIM, dtype=np.float32))
    print("  [4] 空帧 → NONE_EVENT 全零向量 ✓")


if __name__ == "__main__":
    print("模型: models/pt/yolov8n.pt  数据源: ultralytics 自带 bus.jpg / zidane.jpg\n")
    bus = check_upright()
    check_consistency(bus)
    check_suspected()
    check_empty()
    print("\n真实输出验证 OK ✓")
