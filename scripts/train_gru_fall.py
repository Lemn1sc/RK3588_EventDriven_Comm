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
import json
import sys
from datetime import datetime, timezone
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
from common.metrics import (  # noqa: E402
    DEFAULT_AGGS, agg_scores, format_ties, roc_auc, roc_curve, ties_report, tpr_at_fpr,
)
from common.cache import (  # noqa: E402
    META_KEY, SPEC_KEY, check_dims, load_npz, save_npz, spec_sha1,
)

DATA_DIR = ROOT / "data" / "datasets" / "URFD"
CACHE_DIR = ROOT / "data" / "recordings" / "detect_fall"

# 特征集名带版本，缓存与权重都用它命名 —— 新旧版本并存、互不覆盖，
# 所以切回旧特征集不需要改代码，也不会把旧产物冲掉。
FEATSET = {"det": "det.v1", "pose": "pose.v1"}
CACHE = {m: CACHE_DIR / f"feats_{v}.npz" for m, v in FEATSET.items()}
SAVE_MODEL = {m: ROOT / "models" / "pt" / f"gru_fall_{m}.{FEATSET[m].split('.')[-1]}.pt"
              for m in FEATSET}
FEAT_DIM = {"det": 4, "pose": 6}

# 特征规格：指纹的输入。凡是会影响数值的常数都必须写进来 —— 否则把它们改掉
# 不会触发缓存失效，而"改常数"恰恰是最容易被悄悄改坏的一类改动。
FEAT_SPEC = {
    "det": {
        "featset": "det.v1", "dim": 4, "source": "yolov8n 检测 → event_vector",
        "names": ["has_event", "confidence", "severity", "aspect_norm"],
        "conf_clip": [0.0, 1.0], "sev_clip": [0.0, 1.0],
        "aspect_clip": [0.3, 4.0], "aspect_div": 4.0, "no_event_value": 0.0,
    },
    "pose": {
        "featset": "pose.v1", "dim": 6, "source": "yolov8n-pose 全帧 → judge 行",
        "names": ["has_event", "confidence", "severity", "aspect_norm",
                  "torso_tilt_norm", "hip_ankle_ratio"],
        "conf_clip": [0.0, 1.0], "sev_clip": [0.0, 1.0],
        "aspect_clip": [0.3, 4.0], "aspect_div": 4.0, "no_event_value": 0.0,
        "torso_div": 90.0, "hip_ankle_clip": [0.0, 1.0], "hip_ankle_missing": 1.0,
    },
}

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

# 指标落盘目录。experiments/results/ 在 .gitignore 里被放行了 json/csv ——
# 文档里的每个数字都该能在这里找到对应产物，否则半年后无人能核对。
RESULTS_DIR = ROOT / "experiments" / "results"


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


def rel(p: Path) -> Path | str:
    """尽量显示成仓库相对路径；不在仓库内（或本来就缺失）时原样返回。

    报错信息本身绝不能因为路径转换再抛一次异常 —— 那样真正的病因就被埋了。
    """
    try:
        return Path(p).resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return p


def require_inputs() -> None:
    """提取特征前的输入检查。

    没有这一步时，数据缺失会在后面 `np.concatenate([])` 处抛一个与病因无关的
    ValueError，新协作者（clone 之后必然没有那 100MB 数据集）会一脸茫然。
    """
    problems = []
    if not DATA_DIR.is_dir() or not any(DATA_DIR.glob("*/*.mp4")):
        problems.append(
            f"数据集缺失或为空：{rel(DATA_DIR)}\n"
            "      下载：https://gitcode.com/Universal-Tool/fa314 （URFD_video.zip，国内可直连）\n"
            "      解压出 URFD/{fall,adl}/*.mp4（30 段跌倒 + 40 段日常）放到该目录，详见 docs/08 §1"
        )
    for name in ("yolov8n.pt", "yolov8n-pose.pt"):
        if not (dfv.PT_DIR / name).exists():
            problems.append(
                f"模型缺失：{rel(dfv.PT_DIR / name)}"
                "（ultralytics 首次加载时会自动下载到该目录）"
            )
    if problems:
        raise SystemExit("输入检查未通过：\n  - " + "\n  - ".join(problems)
                         + "\n\n（若只是想用已有缓存复现旧结果，请勿加 --recompute。）")


def extract_features(mode: str) -> dict:
    """遍历 70 段视频，每段返回 (feats(T,F), sf(T,)单帧判定, label)。"""
    require_inputs()
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


def item_names(npz) -> list[str]:
    """缓存里的条目名（排除元数据键与 _sf / _label 后缀键）。"""
    return sorted(k for k in npz.files
                  if k not in (SPEC_KEY, META_KEY)
                  and not k.endswith("_sf") and not k.endswith("_label"))


def load_or_extract(mode: str, recompute: bool) -> dict:
    """取特征：优先读缓存（严格校验版本），否则重新提取。

    读缓存走 common.cache 的三层校验：文件名带特征集身份、内容指纹必须与当前
    代码一致、形状再兜一道。任何一条不符都报错停下，绝不"兼容读取"。
    """
    cache = CACHE[mode]
    what = f"[{mode}] 特征缓存"

    if not recompute:
        npz, _meta = load_npz(cache, FEAT_SPEC[mode], what=what)
        names = item_names(npz)
        if not names:
            raise SystemExit(f"{what}里没有任何视频条目：{rel(cache)}\n"
                             f"  → 加 --recompute 重新生成")
        check_dims(npz, names, FEAT_DIM[mode], what=what, path=cache)
        return {k: {"feats": npz[k], "sf": npz[f"{k}_sf"],
                    "label": float(npz[f"{k}_label"][0])} for k in names}

    print(f"提取特征 [{mode}]（70 段 × 每帧{'检测+姿态' if mode == 'pose' else '仅检测'}）"
          f"→ {rel(cache)}")
    data = extract_features(mode)
    if not data:
        raise SystemExit(f"[{mode}] 未提取到任何视频，检查 {rel(DATA_DIR)} 下的 *.mp4")
    save_npz(cache, FEAT_SPEC[mode], {
        **{k: v["feats"] for k, v in data.items()},
        **{f"{k}_sf": v["sf"] for k, v in data.items()},
        **{f"{k}_label": np.array([v["label"]]) for k, v in data.items()},
    }, n_videos=len(data))
    print(f"  已保存 {len(data)} 段（特征集 {FEATSET[mode]}，"
          f"指纹 {spec_sha1(FEAT_SPEC[mode])}）")
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
    """按视频划分训练/测试（防同视频的窗口泄漏到两边）。

    ⚠ 两个返回列表都**显式排序**，这不是洁癖：训练时所有窗口按视频顺序拼接成一个
    大数组，再由 torch.randperm 打乱成批次。所以视频的排列顺序会改变批次内容、
    进而改变训练结果 —— 而这以前是隐式的（取决于缓存文件里条目的写入顺序，
    例如"先 fall 后 adl"还是"先 adl 后 fall"）。同一个人换台机器重跑就可能对不上。
    显式排序后，结果只取决于数据内容和 seed，与字典/文件的构造顺序无关。
    """
    rng = np.random.RandomState(seed)
    fall = sorted(k for k, v in data.items() if v["label"] == 1.0)
    adl = sorted(k for k, v in data.items() if v["label"] == 0.0)
    rng.shuffle(fall)
    rng.shuffle(adl)
    n_test_fall = max(1, round(len(fall) * TEST_FRAC))
    n_test_adl = max(1, round(len(adl) * TEST_FRAC))
    test = sorted(fall[:n_test_fall] + adl[:n_test_adl])
    test_set = set(test)
    train = sorted(k for k in data if k not in test_set)
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


def save_checkpoint(path, model, mode: str) -> None:
    """保存权重 + 身份证（特征集名 / 指纹 / 输入维度 / 网络结构）。

    为什么不能只存裸 state_dict：**维度恰好一致时它会静默加载成功**（例如同一
    维度内调换了特征顺序），模型照跑不报错，只有指纹能拦住。维度不一致时 torch
    自己会抛 RuntimeError，但信息是 `size mismatch` 这种天书。
    """
    torch.save({"state_dict": model.state_dict(),
                "featset": FEATSET[mode],
                "feat_spec_sha1": spec_sha1(FEAT_SPEC[mode]),
                "in_dim": FEAT_DIM[mode],
                "hidden": HIDDEN, "layers": LAYERS,
                "window": WINDOW, "stride": STRIDE}, path)


def load_checkpoint(path, mode: str):
    """加载权重并校验身份（供 T0.4 的 edge_node 复用；本轮实验流程也用它做回读校验）。"""
    ckpt = torch.load(path, map_location="cpu")
    want = spec_sha1(FEAT_SPEC[mode])
    if ckpt.get("feat_spec_sha1") != want or ckpt.get("in_dim") != FEAT_DIM[mode]:
        raise SystemExit(
            f"权重与当前特征集不匹配：{rel(path)}\n"
            f"  权重：特征集 {ckpt.get('featset')} / 指纹 {ckpt.get('feat_spec_sha1')} / "
            f"{ckpt.get('in_dim')} 维\n"
            f"  当前：特征集 {FEATSET[mode]} / 指纹 {want} / {FEAT_DIM[mode]} 维"
        )
    model = FallGRU(ckpt["in_dim"], ckpt.get("hidden", HIDDEN), ckpt.get("layers", LAYERS))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    return model


# ────────────────────────── 3. 评估 ──────────────────────────

# 单帧基线在两种模式下用的规则本来就不同，报告里必须写明是哪一条
# （det 走契约的 severity 阈值，pose 走 judge 的框+姿态融合判定）。
SF_RULE = {"det": "severity>=0.5", "pose": "judge(框+姿态融合)"}


def build_windows(data: dict, videos) -> dict:
    """视频列表 → {视频名: (窗口张量 X, 窗口标签 y)}。

    短于 WINDOW 帧的视频没有窗口，会被跳过（调用方需处理"该视频无窗口"的情况）。
    """
    out = {}
    for name in videos:
        win, lab = make_windows(data[name]["feats"], data[name]["label"])
        if win is not None:
            out[name] = (win, lab)
    return out


def window_probs(model, windows: dict) -> dict:
    """{视频名: (X, y)} → {视频名: 每窗口跌倒概率}。每段只前向一次。"""
    model.eval()
    out = {}
    with torch.no_grad():
        for name, (win, _) in windows.items():
            out[name] = torch.sigmoid(model(torch.tensor(win, device=DEVICE))).cpu().numpy()
    return out


def run_mode(mode: str, data: dict, seed: int = SEED, verbose: bool = True) -> dict:
    if verbose:
        print(f"\n########## 特征源 [{mode}]（{FEAT_DIM[mode]} 维/帧）· seed={seed} ##########")
    train_videos, test_videos = split_videos(data, seed)

    # 窗口只切一次，训练/测试/评分共用。旧版把测试集切了 3 遍（两个列表推导各一遍
    # 取 X 和 y，video_scores 内部又一遍），纯浪费且是改造聚合方式时要动的地方。
    train_win = build_windows(data, train_videos)
    test_win = build_windows(data, test_videos)
    if not train_win or not test_win:
        raise SystemExit(
            f"[{mode}] 窗口为空：检查 data/datasets/URFD 是否就位、"
            f"以及视频长度是否都短于 WINDOW={WINDOW} 帧"
        )
    train_x = np.concatenate([w for w, _ in train_win.values()])
    train_y = np.concatenate([l for _, l in train_win.values()])
    test_x = np.concatenate([w for w, _ in test_win.values()])
    test_y = np.concatenate([l for _, l in test_win.values()])
    if verbose:
        print(f"设备 {DEVICE} · 训练 {len(train_videos)} 视频 / 测试 {len(test_videos)} 视频 · "
              f"训练窗口 {len(train_x)} 条 / 测试窗口 {len(test_x)} 条")

    model = train_model(train_x, train_y, test_x, test_y, FEAT_DIM[mode], seed, verbose)
    if seed == 0:  # 只保留 seed=0 的模型文件
        save_checkpoint(SAVE_MODEL[mode], model, mode)
        load_checkpoint(SAVE_MODEL[mode], mode)  # 回读校验：让保存路径也有测试守着

    # ── 视频级判定 ──
    labels = np.array([data[v]["label"] for v in test_videos])
    # 每段测试视频的窗口数（长度 ∝ 视频时长；这是"取 max 聚合"偏置的来源，必须报出来）
    n_win = np.array([len(test_win[v][0]) if v in test_win else 0 for v in test_videos])
    sf_scores = np.array([float(data[v]["sf"].mean()) for v in test_videos])  # 单帧:触发帧占比
    probs = window_probs(model, test_win)
    # 三种聚合同时算：max 会把"整段视频是否跌倒"交给最像跌倒的那个窗口，是口径最松、
    # 也最容易受视频长度影响的选法（窗口数 3~38，长视频机会多）；mean 会被大量正常帧
    # 稀释；top3mean 居中。结论对聚合方式的敏感性本身就是一条要报告的结果。
    gs_scores = {how: np.array([agg_scores(probs.get(v, []), how) for v in test_videos])
                 for how in DEFAULT_AGGS}

    # AUC 走 common.metrics：并列分数按秩公式处理（旧的 np.argsort+trapz 在并列时
    # 给出的是排序的副产品而非确定值，见 src/common/metrics.py 顶部说明）。
    auc_sf = roc_auc(labels, sf_scores)
    auc_gs = {how: roc_auc(labels, s) for how, s in gs_scores.items()}
    fpr_sf, tpr_sf = roc_curve(labels, sf_scores)
    gs_curves = {how: roc_curve(labels, s) for how, s in gs_scores.items()}
    fpr_gs, tpr_gs = gs_curves["max"]
    ties = ties_report(labels, sf_scores)

    sf_v = dict(zip(test_videos, sf_scores))
    base_tp = sum(1 for n in test_videos if data[n]["label"] == 1 and sf_v[n] >= SF_RATIO_THRES)
    base_fp = sum(1 for n in test_videos if data[n]["label"] == 0 and sf_v[n] >= SF_RATIO_THRES)
    if verbose:
        print(f"\n  [视频级对比] 单帧基线(规则 {SF_RULE[mode]}, 阈值 {SF_RATIO_THRES}): "
              f"检出 {base_tp}/{(labels == 1).sum()} · 误报 {base_fp}/{(labels == 0).sum()}")
        print(f"  单帧基线分数并列体检: {format_ties(ties)}")
        print(f"  ROC-AUC（0.5=随机）: 单帧={auc_sf:.3f} → GRU(max)={auc_gs['max']:.3f}")
        for how in DEFAULT_AGGS:
            f, t = gs_curves[how]
            print(f"    聚合 [{how:8s}] AUC={auc_gs[how]:.3f}  "
                  f"@FPR0={tpr_at_fpr(f, t, 0.0) * 100:.0f}%  "
                  f"@FPR≤0.1={tpr_at_fpr(f, t, 0.1) * 100:.0f}%")
        for target in (0.0, 0.1, 0.2):
            print(f"  FPR≤{target:.1f} 时检出率: 单帧={tpr_at_fpr(fpr_sf, tpr_sf, target) * 100:.0f}%  "
                  f"GRU(max)={tpr_at_fpr(fpr_gs, tpr_gs, target) * 100:.0f}%")
        print(f"  测试视频窗口数: {n_win.min()}~{n_win.max()}（相差 "
              f"{n_win.max() / max(n_win.min(), 1):.0f} 倍，max 聚合下长视频机会更多）")
        print(f"  模型已存: {SAVE_MODEL[mode]}")

    # 对齐基线误报率的工作点（GRU 单点，取 max 聚合），供最终汇总
    gs_max = gs_scores["max"]
    base_fpr = float((sf_scores[labels == 0] >= SF_RATIO_THRES).mean()) if (labels == 0).sum() else 0.0
    thr = min(sorted(set(gs_max)), key=lambda t: abs(float((gs_max[labels == 0] >= t).mean()) - base_fpr))
    gv = dict(zip(test_videos, gs_max >= thr))
    g_tp = sum(1 for n in test_videos if data[n]["label"] == 1 and gv[n])
    g_fp = sum(1 for n in test_videos if data[n]["label"] == 0 and gv[n])
    if verbose:
        print(f"  GRU(对齐基线误报,阈值={thr:.2f}): 检出 {g_tp}/{(labels == 1).sum()} · "
              f"误报 {g_fp}/{(labels == 0).sum()}")

    out = {"auc_sf": auc_sf, "auc_gs": auc_gs["max"],
           "tpr0_sf": tpr_at_fpr(fpr_sf, tpr_sf, 0.0),
           "tpr0_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.0),
           "tpr1_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.1),
           "tpr1_sf": tpr_at_fpr(fpr_sf, tpr_sf, 0.1),
           "ties_max_group": ties["max_group"],
           "n_win_min": int(n_win.min()), "n_win_max": int(n_win.max())}
    for how in DEFAULT_AGGS:  # 聚合敏感性：每个聚合各一套指标
        f, t = gs_curves[how]
        out[f"auc_gs_{how}"] = auc_gs[how]
        out[f"tpr0_gs_{how}"] = tpr_at_fpr(f, t, 0.0)
    return out


def config_snapshot() -> dict:
    """本次实验的全部可调量。写进 json，避免"改了配置忘了记录"。"""
    return {
        "window": WINDOW, "stride": STRIDE, "hidden": HIDDEN, "layers": LAYERS,
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED,
        "test_frac": TEST_FRAC, "severity_suspect": SEVERITY_SUSPECT,
        "sf_ratio_thres": SF_RATIO_THRES, "aggs": list(DEFAULT_AGGS),
    }


def env_snapshot() -> dict:
    """环境指纹。同一份代码在不同 torch/numpy 下可能给出不同数值，必须记录。"""
    import platform
    import subprocess

    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001  拿不到就不记，不让它挡住实验
            return None

    import ultralytics
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def write_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mstd(mlist, key, scale=100.0, fmt="{:.0f}±{:.0f}"):
    """mean±std。显式接收列表（旧版用闭包读循环变量 mode，多特征集扩展必踩坑）。"""
    a = np.array([r[key] for r in mlist], dtype=np.float64) * scale
    return fmt.format(a.mean(), a.std())


def mean_of(mlist, key):
    return float(np.array([r[key] for r in mlist], dtype=np.float64).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="强制重新提取特征")
    ap.add_argument("--seeds", type=int, default=5, help="随机 seed 数（每个 seed 重新划分+重训）")
    ap.add_argument("--out-json", type=Path, default=RESULTS_DIR / "metrics_latest.json",
                    help="指标落盘路径（默认 experiments/results/metrics_latest.json）")
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
                  f"GRU@FPR≤0.1={r['tpr1_gs'] * 100:.0f}%  "
                  f"AUC 单帧→GRU={r['auc_sf']:.3f}→{r['auc_gs']:.3f}  "
                  f"(基线并列组最大 {r['ties_max_group']})")
        metrics[mode] = mlist

    print(f"\n\n========== 稳健性汇总（mean±std over {seeds} seeds） ==========")
    print(f"{'特征源':6s} {'单帧检出@FPR0':>16s} {'GRU@FPR0':>12s} {'GRU@FPR≤0.1':>12s} "
          f"{'AUC单帧':>9s} {'AUC-GRU':>9s}")
    for mode in ("det", "pose"):
        mlist = metrics[mode]
        print(f"{mode:6s} {mstd(mlist, 'tpr0_sf'):>16s} {mstd(mlist, 'tpr0_gs'):>12s} "
              f"{mstd(mlist, 'tpr1_gs'):>12s} {mean_of(mlist, 'auc_sf'):8.3f} "
              f"{mean_of(mlist, 'auc_gs'):8.3f}")
    print(f"\n\n========== 聚合方式敏感性（同一批 GRU 窗口分数，只换汇总方式） ==========")
    print("窗口 → 视频级分数的汇总方式不同，结论会不同；单帧基线未参与，纯 GRU 侧对照。")
    print(f"{'特征源':6s} {'聚合':<10s} {'AUC':>16s} {'检出@FPR0':>16s}")
    for mode in ("det", "pose"):
        mlist = metrics[mode]
        for how in DEFAULT_AGGS:
            print(f"{mode:6s} {how:<10s} {mstd(mlist, f'auc_gs_{how}', scale=1.0, fmt='{:.3f}±{:.3f}'):>16s} "
                  f"{mstd(mlist, f'tpr0_gs_{how}'):>16s}")
    nw = [(r["n_win_min"], r["n_win_max"]) for r in metrics["det"]]
    print(f"\n每段测试视频的窗口数范围: {min(a for a, _ in nw)}~{max(b for _, b in nw)}"
          f"（max 聚合下窗口多的视频天然机会更多，这是与标签无关的长度偏置）")

    print(f"\n注：det=仅检测(边缘端可用，{FEAT_DIM['det']}维)，pose=检测+姿态(对照组，{FEAT_DIM['pose']}维)。")
    print("若 det 的 GRU 稳定接近/超过 pose，说明边缘端只靠检测特征+时序就够，可省去姿态卸载。")
    print("AUC 按并列取平均秩计算（common.metrics），单帧基线分数存在大量并列，"
          "旧的不稳定排序实现会给出偏高的任意值。")

    def _summary(mlist):
        return {k: {"mean": float(np.mean([r[k] for r in mlist])),
                    "std": float(np.std([r[k] for r in mlist]))}
                for k in mlist[0]}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/train_gru_fall.py",
        "env": env_snapshot(),
        "config": config_snapshot(),
        "seeds": seeds,
        "featset": {m: {"dim": FEAT_DIM[m], "single_frame_rule": SF_RULE[m],
                        "cache": str(CACHE[m].relative_to(ROOT))} for m in metrics},
        "per_seed": metrics,
        "summary": {m: _summary(metrics[m]) for m in metrics},
    }
    write_metrics_json(args.out_json, payload)
    print(f"\n指标已落盘: {rel(args.out_json)}")


if __name__ == "__main__":
    main()
