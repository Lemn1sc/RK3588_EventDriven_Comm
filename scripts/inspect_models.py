"""T0.2 · 跑通检测 + 姿态两个模型，看清输出结构。

目的：把 yolov8n（检测）和 yolov8n-pose（姿态）的原始输出结构打印出来，
为 T0.3 设计 8 维事件向量 / 姿态判定提供依据。默认用 ultralytics 自带 bus.jpg。
用法：python scripts/inspect_models.py  [可选: 图片/视频路径]
"""
import sys
from pathlib import Path

from ultralytics import ASSETS
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PT_DIR = ROOT / "models" / "pt"
PT_DIR.mkdir(parents=True, exist_ok=True)

# COCO 17 关键点顺序（yolov8-pose 固定输出顺序，T0.3 会引用）
COCO_KPTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SOURCE = sys.argv[1] if len(sys.argv) > 1 else str(ASSETS / "bus.jpg")


def line(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def inspect_detect():
    line("检测模型 yolov8n · 事件端角色（本地/边缘）")
    model = YOLO(str(PT_DIR / "yolov8n.pt"))
    r = model.predict(SOURCE, verbose=False, save=True, exist_ok=True,
                      project=str(ROOT / "data" / "recordings"), name="detect")[0]
    b = r.boxes
    if b is None:
        print("未检测到任何目标框。")
        return model.names
    print(f"图像尺寸 (h,w): {r.orig_shape}")
    print(f"检测到目标数: {len(b)}")
    print(f"boxes.xyxy   shape={tuple(b.xyxy.shape)}  像素坐标 x1,y1,x2,y2")
    print(f"boxes.xywhn  shape={tuple(b.xywhn.shape)}  归一化中心+宽高")
    print(f"boxes.conf   shape={tuple(b.conf.shape)}  置信度")
    print(f"boxes.cls    shape={tuple(b.cls.shape)}   类别编号")
    print("\n逐目标（前 5 个）：")
    for i in range(min(5, len(b))):
        cid = int(b.cls[i])
        xyxy = b.xyxy[i].cpu().numpy().round(1)
        xywhn = b.xywhn[i].cpu().numpy().round(3)
        print(f"  #{i} 类别={cid}({model.names[cid]}) conf={float(b.conf[i]):.3f} "
              f"xyxy={xyxy.tolist()} 高宽比={float(xywhn[3]/max(xywhn[2],1e-6)):.2f}")
    print(f"\n保存可视化 → data/recordings/detect/")
    return model.names


def inspect_pose():
    line("姿态模型 yolov8n-pose · 远端角色（跌倒判定）")
    model = YOLO(str(PT_DIR / "yolov8n-pose.pt"))
    r = model.predict(SOURCE, verbose=False, save=True, exist_ok=True,
                      project=str(ROOT / "data" / "recordings"), name="pose")[0]
    k = r.keypoints
    if k is None:
        print("未检测到任何人体关键点。")
        return
    conf = k.conf
    print(f"检测到人数: {len(k)}")
    print(f"keypoints.xy   shape={tuple(k.xy.shape)}   (人数, 17, 2) 像素坐标")
    print(f"keypoints.xyn  shape={tuple(k.xyn.shape)}  归一化坐标")
    print(f"keypoints.conf shape={tuple(conf.shape) if conf is not None else None}  每点可见度")
    if len(k) and conf is not None:
        xy = k.xy[0].cpu().numpy()
        cf = conf[0].cpu().numpy()
        print("\n第 1 个人的 17 关键点 (COCO 顺序)：")
        for idx, name in enumerate(COCO_KPTS):
            x, y = xy[idx]
            print(f"  [{idx:2d}] {name:15s} x={x:6.1f} y={y:6.1f} conf={cf[idx]:.2f}")
    print(f"\n保存可视化 → data/recordings/pose/")


if __name__ == "__main__":
    print(f"数据源: {SOURCE}")
    inspect_detect()
    inspect_pose()
    line("完成 · 看清结构后即可进入 T0.3 定义 8 维事件向量")
