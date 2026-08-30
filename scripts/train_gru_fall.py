"""T0.6 前奏 · 简单 GRU 跌倒检测（时序 vs 单帧，看提升效果）。

支持两种特征来源，一次跑完对比：
  - det  ：**仅 yolov8n 检测**（边缘端真实可用，RK3588 NPU 上便宜）
           4 维 = [has_event, confidence, severity, 框高宽比h/w]（即事件向量的降维）
  - pose ：检测 + yolov8n-pose 姿态（6 维，多 躯干倾角/髋踝高差，需远端/较高算力）

对比逻辑（同一批测试视频）：
  单帧基线（det: severity≥0.5 触发；pose: 几何判定跌倒帧占比≥0.2）
  vs 简单 GRU（30 帧滑窗 → GRU → 视频级 max 概率，ROC 对齐误报率比较）。

动机（docs/08 实测）：单帧几何——无论纯框还是姿态——都分不开"跌倒 vs 日常躺卧"，
且低位视角下纯高宽比漏检 ~90%。这里看加一个 GRU 吃时序能提升多少，
以及"只靠检测特征"够不够。

用法：
    python scripts/train_gru_fall.py             # 两种特征源都跑（pose 特征已缓存则秒出）
    python scripts/train_gru_fall.py --recompute # 强制重新提取特征
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import io  # noqa: E402

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import detect_fall_video as dfv  # noqa: E402  复用 process_frame（同特征、同判定）
from common.event_vector import event_from_result  # noqa: E402

DATA_DIR = ROOT / "data" / "datasets" / "URFD"
CACHE_DIR = ROOT / "data" / "recordings" / "detect_fall"
CACHE = {"det": CACHE_DIR / "gru_features_det.npz", "pose": CACHE_DIR / "gru_features_pose.npz"}
SAVE_MODEL = {"det": ROOT / "models" / "pt" / "gru_fall_det.pt", "pose": ROOT / "models" / "pt" / "gru_fall_pose.pt"}
FEAT_DIM = {"det": 4, "pose": 6}

# ── GRU 超参（简单，够演示）──
WINDOW = 30          # 每窗口帧数（≈1s）
STRIDE = 10          # 滑窗步长
HIDDEN = 32
LAYERS = 1
EPOCHS = 40
BATCH = 64
LR = 1e-3
SEED = 0

SEVERITY_SUSPECT = 0.5  # det 单帧基线的触发阈值（与 event_vector 契约一致）
SF_RATIO_THRES = 0.2    # 单帧基线：跌倒帧占比 ≥ 该值判该视频跌倒（与 bench_pose_fall 一致）
TEST_FRAC = 0.25        # 测试视频占比（按视频划分，防同视频窗口泄漏）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────── 1. 特征提取（两种模式）──────────────────────────
def feats_from_event(ev, has_event) -> np.ndarray:
    """det 模式：一帧事件向量 → 4 维特征。"""
    f = np.zeros(4, dtype=np.float32)
    if has_event:
        x1, y1, x2, y2 = ev.bbox
        aspect = (y2 - y1) / max(x2 - x1, 1e-6)
        f[0] = 1.0
        f[1] = min(max(float(ev.confidence), 0.0), 1.0)
        f[2] = min(max(float(ev.severity), 0.0), 1.0)
        f[3] = min(max(aspect, 0.3), 4.0) / 4.0
    return np.nan_to_num(f, nan=0.0)


def feats_from_pose(row) -> np.ndarray:
    """pose 模式：process_frame 的一行日志 → 6 维特征（含姿态几何量）。"""
    f = np.zeros(6, dtype=np.float32)
    if row["has_event"]:
        f[0] = 1.0
        f[1] = min(max(float(row["confidence"]), 0.0), 1.0)
        f[2] = min(max(float(row["severity"]), 0.0), 1.0)
        f[3] = min(max(float(row["aspect_ratio"]), 0.3), 4.0) / 4.0
    if row["torso_angle"]:
        f[4] = min(max(float(row["torso_angle"]) / 90.0, 0.0), 1.0)
    if row["hip_ankle_ratio"]:
        f[5] = min(max(float(row["hip_ankle_ratio"]), 0.0), 1.0)
    else:
        f[5] = 1.0  # 关键点缺失→按直立处理
    return np.nan_to_num(f, nan=0.0)


def extract_features(mode: str) -> dict:
    """遍历 70 段视频，每段返回 (feats(T,F), sf(T,)单帧判定, label)。"""
    det = YOLO(str(dfv.PT_DIR / "yolov8n.pt"))
    pose = YOLO(str(dfv.PT_DIR / "yolov8n-pose.pt")) if mode == "pose" else None
    out = {}
    for split in ("fall", "adl"):
        for video in sorted((DATA_DIR / split).glob("*.mp4")):
            cap = cv2.VideoCapture(str(video))
            feats, sf = [], []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if mode == "pose":
                    _, row = dfv.process_frame(frame, det, pose, always_pose=True)
                    feats.append(feats_from_pose(row))
                    sf.append(float(row["fall"]))
                else:
                    ev = event_from_result(det(frame, verbose=False)[0])
                    feats.append(feats_from_event(ev, ev.has_event))
                    sf.append(float(ev.severity >= SEVERITY_SUSPECT))
            cap.release()
            out[f"{split}/{video.name}"] = {
                "feats": np.asarray(feats, dtype=np.float32),
                "sf": np.asarray(sf, dtype=np.float32),
                "label": 1.0 if split == "fall" else 0.0,
            }
    return out


def load_or_extract(mode: str, recompute: bool) -> dict:
    cache = CACHE[mode]
    if cache.exists() and not recompute:
        npz = np.load(cache, allow_pickle=True)
        names = [k for k in npz.files if not k.endswith("_sf") and not k.endswith("_label")]
        return {k: {"feats": npz[k], "sf": npz[f"{k}_sf"], "label": float(npz[f"{k}_label"][0])}
                for k in names}
    print(f"提取特征 [{mode}]（70 段 × 每帧{'检测+姿态' if mode == 'pose' else '仅检测'}）→ {cache}")
    data = extract_features(mode)
    np.savez(cache, **{k: v["feats"] for k, v in data.items()},
             **{f"{k}_sf": v["sf"] for k, v in data.items()},
             **{f"{k}_label": np.array([v["label"]]) for k, v in data.items()})
    return data


# ────────────────────────── 2. 滑窗 + 划分 + 模型 + 训练 ──────────────────────────
def make_windows(feats: np.ndarray, label: float):
    """单段视频 → 滑窗 (N,L,F)；窗口标签取视频级标签（弱监督）。"""
    T = len(feats)
    xs, ys = [], []
    for i in range(0, T - WINDOW + 1, STRIDE):
        xs.append(feats[i:i + WINDOW])
        ys.append(label)
    if not xs:
        return None, None
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def split_videos(data: dict, seed: int = SEED):
    rng = np.random.RandomState(seed)
    fall = sorted(k for k, v in data.items() if v["label"] == 1.0)
    adl = sorted(k for k, v in data.items() if v["label"] == 0.0)
    rng.shuffle(fall)
    rng.shuffle(adl)
    n_test_fall = max(1, round(len(fall) * TEST_FRAC))
    n_test_adl = max(1, round(len(adl) * TEST_FRAC))
    test = fall[:n_test_fall] + adl[:n_test_adl]
    train = [k for k in data if k not in test]
    return train, test


class FallGRU(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)  # 末帧隐状态 → logit


def train_model(train_x, train_y, test_x, test_y, in_dim, seed=SEED, verbose=True):
    torch.manual_seed(seed)
    model = FallGRU(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    tx = torch.tensor(train_x, device=DEVICE)
    ty = torch.tensor(train_y, device=DEVICE)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tx))
        total = 0.0
        for i in range(0, len(tx), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(tx[idx]), ty[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            acc = eval_acc(model, test_x, test_y)
            print(f"  epoch {epoch:3d}  train_loss={total / len(tx):.4f}  test_acc={acc * 100:.1f}%")
    return model


def eval_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(x, device=DEVICE))).cpu().numpy()
    return ((p >= 0.5).astype(int) == y.astype(int)).mean()


# ────────────────────────── 3. 评估 ──────────────────────────
def video_scores(model, videos, data):
    """每段测试视频 → GRU 窗口最大概率（任一窗口疑似跌倒即倾向跌倒）。"""
    model.eval()
    scores = {}
    with torch.no_grad():
        for name in videos:
            win, _ = make_windows(data[name]["feats"], data[name]["label"])
            if win is None:
                scores[name] = 0.0
                continue
            p = torch.sigmoid(model(torch.tensor(win, device=DEVICE))).cpu().numpy()
            scores[name] = float(p.max())
    return scores


def roc(labels, scores):
    """标准 ROC：返回 (fpr, tpr, AUC)，梯形积分。label 1 = 跌倒。"""
    order = np.argsort(scores)
    labs = labels[order]
    n_pos = float((labs == 1).sum())
    n_neg = float((labs == 0).sum())
    fpr, tpr = [0.0], [0.0]
    tp = fp = 0
    for lab in labs[::-1]:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    tpr.append(1.0)
    fpr.append(1.0)
    fpr, tpr = np.array(fpr), np.array(tpr)
    return fpr, tpr, float(np.trapz(tpr, fpr))


def tpr_at_fpr(fpr, tpr, target):
    """FPR ≤ target 时的最高跌倒检出率（TPR）。"""
    ok = fpr <= target + 1e-9
    return float(tpr[ok].max()) if ok.any() else 0.0


def run_mode(mode: str, data: dict, seed: int = SEED, verbose: bool = True) -> dict:
    if verbose:
        print(f"\n########## 特征源 [{mode}]（{FEAT_DIM[mode]} 维/帧）· seed={seed} ##########")
    train_videos, test_videos = split_videos(data, seed)

    all_x, all_y = [], []
    for name in train_videos:
        win, lab = make_windows(data[name]["feats"], data[name]["label"])
        if win is not None:
            all_x.append(win)
            all_y.append(lab)
    train_x = np.concatenate(all_x)
    train_y = np.concatenate(all_y)
    test_x = np.concatenate([w for w, _ in (make_windows(data[n]["feats"], data[n]["label"])
                                            for n in test_videos) if w is not None])
    test_y = np.concatenate([l for _, l in (make_windows(data[n]["feats"], data[n]["label"])
                                            for n in test_videos) if l is not None])
    if verbose:
        print(f"设备 {DEVICE} · 训练 {len(train_videos)} 视频 / 测试 {len(test_videos)} 视频 · "
              f"训练窗口 {len(train_x)} 条 / 测试窗口 {len(test_x)} 条")

    model = train_model(train_x, train_y, test_x, test_y, FEAT_DIM[mode], seed, verbose)
    if seed == 0:  # 只保留 seed=0 的模型文件
        torch.save(model.state_dict(), SAVE_MODEL[mode])

    # 视频级判定
    labels = np.array([data[v]["label"] for v in test_videos])
    sf_scores = np.array([float(data[v]["sf"].mean()) for v in test_videos])  # 单帧:跌倒帧占比
    gs = video_scores(model, test_videos, data)
    gs_scores = np.array([gs[v] for v in test_videos])

    fpr_sf, tpr_sf, auc_sf = roc(labels, sf_scores)
    fpr_gs, tpr_gs, auc_gs = roc(labels, gs_scores)

    sf_v = dict(zip(test_videos, sf_scores))
    base_tp = sum(1 for n in test_videos if data[n]["label"] == 1 and sf_v[n] >= SF_RATIO_THRES)
    base_fp = sum(1 for n in test_videos if data[n]["label"] == 0 and sf_v[n] >= SF_RATIO_THRES)
    print(f"\n  [视频级对比] 单帧基线(固定规则): 检出 {base_tp}/{(labels == 1).sum()} · "
          f"误报 {base_fp}/{(labels == 0).sum()}")
    print(f"  ROC-AUC（0.5=随机）: 单帧={auc_sf:.3f} → GRU={auc_gs:.3f}")
    for target in (0.0, 0.1, 0.2):
        print(f"  FPR≤{target:.1f} 时检出率: 单帧={tpr_at_fpr(fpr_sf, tpr_sf, target) * 100:.0f}%  "
              f"GRU={tpr_at_fpr(fpr_gs, tpr_gs, target) * 100:.0f}%")
    print(f"  模型已存: {SAVE_MODEL[mode]}")

    # 对齐基线误报率的工作点（GRU 单点），供最终汇总
    base_fpr = float((sf_scores[labels == 0] >= SF_RATIO_THRES).mean()) if (labels == 0).sum() else 0.0
    thr = min(sorted(set(gs_scores)), key=lambda t: abs(float((gs_scores[labels == 0] >= t).mean()) - base_fpr))
    gv = dict(zip(test_videos, gs_scores >= thr))
    g_tp = sum(1 for n in test_videos if data[n]["label"] == 1 and gv[n])
    g_fp = sum(1 for n in test_videos if data[n]["label"] == 0 and gv[n])
    print(f"  GRU(对齐基线误报,阈值={thr:.2f}): 检出 {g_tp}/{(labels == 1).sum()} · 误报 {g_fp}/{(labels == 0).sum()}")

    return {"auc_sf": auc_sf, "auc_gs": auc_gs, "tpr0_sf": tpr_at_fpr(fpr_sf, tpr_sf, 0.0),
            "tpr0_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.0), "tpr1_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.1),
            "tpr1_sf": tpr_at_fpr(fpr_sf, tpr_sf, 0.1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="强制重新提取特征")
    ap.add_argument("--seeds", type=int, default=5, help="随机 seed 数（每个 seed 重新划分+重训）")
    args = ap.parse_args()
    seeds = args.seeds

    metrics = {}  # mode -> [每 seed 的指标 dict]
    for mode in ("det", "pose"):
        data = load_or_extract(mode, args.recompute)
        print(f"\n########## 特征源 [{mode}] · {seeds} 个 seed ##########")
        mlist = []
        for s in range(seeds):
            r = run_mode(mode, data, seed=s, verbose=(seeds == 1))
            mlist.append(r)
            print(f"  seed{s}: 单帧@FPR0={r['tpr0_sf'] * 100:.0f}%  GRU@FPR0={r['tpr0_gs'] * 100:.0f}%  "
                  f"GRU@FPR≤0.1={r['tpr1_gs'] * 100:.0f}%  AUC 单帧→GRU={r['auc_sf']:.3f}→{r['auc_gs']:.3f}")
        metrics[mode] = mlist

    def mstd(key):
        a = np.array([r[key] for r in metrics[mode]])
        return f"{a.mean() * 100:.0f}±{a.std() * 100:.0f}"

    def mean(key):
        return float(np.array([r[key] for r in metrics[mode]]).mean())

    print(f"\n\n========== 稳健性汇总（mean±std over {seeds} seeds） ==========")
    print(f"{'特征源':6s} {'单帧检出@FPR0':>16s} {'GRU@FPR0':>12s} {'GRU@FPR≤0.1':>12s} {'AUC单帧':>9s} {'AUC-GRU':>9s}")
    for mode in ("det", "pose"):
        print(f"{mode:6s} {mstd('tpr0_sf'):>16s} {mstd('tpr0_gs'):>12s} {mstd('tpr1_gs'):>12s} "
              f"{mean('auc_sf'):8.3f} {mean('auc_gs'):8.3f}")
    print("\n注：det=仅检测(边缘端可用，4维)，pose=检测+姿态(对照组，6维)。")
    print("若 det 的 GRU 稳定接近/超过 pose，说明边缘端只靠检测特征+时序就够，可省去姿态卸载。")


if __name__ == "__main__":
    main()
