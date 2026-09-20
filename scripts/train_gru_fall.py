"""T0.6 前奏 · 简单 GRU 跌倒检测（时序 vs 单帧，看提升效果）。

本脚本只**消费**特征，不再自己跑模型。特征是 `scripts/gru_features.py` 从原始帧表
算出来的，所以「换一组特征看看」是秒级操作 —— 不需要重新推理、更不需要重训模型。

特征集（详见 scripts/gru_features.py 的注册表）：
  det.v1   4 维 = [has_event, confidence, severity, 框高宽比]   旧版基线，供对照
  det.v2a  6 维 = det.v1 + [框中心 y, 尺度 sqrt(面积)]
  det.v2   8 维 = det.v2a + [相对近 30 帧基线的中心下沉比, 尺度突变比 log2]
  pose.v1  6 维 = 旧版姿态基线（det.v1 的 4 维 + 躯干倾角 + 髋踝高差）
  pose.v2 10 维 = det.v2 的 8 维 + 同样的 2 维姿态量

det.v2 与 pose.v2 的前 8 维完全相同、只差 2 个姿态量，所以「姿态值不值得卸载」
这个对比是公平的消融。

对比逻辑（同一批测试视频）：
  单帧基线（severity≥0.5 或 judge 融合判定，两条都报）vs 简单 GRU
  （30 帧滑窗 → GRU → 视频级分数，ROC 对齐误报率比较）。

动机（docs/08 实测）：单帧几何——无论纯框还是姿态——都分不开「跌倒 vs 日常躺卧」，
且低位视角下纯高宽比漏检 ~90%。这里看加一个 GRU 吃时序能提升多少，
以及「只靠检测特征」够不够。

用法：
    python scripts/gru_features.py list                                   # 看有哪些特征集
    python scripts/gru_features.py frames --mode det                      # 产原始帧表（跑模型）
    python scripts/gru_features.py feats --featsets det.v1 det.v2         # 帧表 → 特征（秒级）
    python scripts/train_gru_fall.py                                      # 全部特征集，5 seed
    python scripts/train_gru_fall.py --featsets det.v1 det.v2 --seeds 5   # 消融对照
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import io  # noqa: E402

# Windows 控制台默认 GBK：中文与 ✓ 会乱码。stdout / stderr **都要设** ——
# 报错信息走的是 stderr，只设 stdout 的话出错时提示反而是乱码的，最难读的时候读不了。
for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.metrics import (  # noqa: E402
    DEFAULT_AGGS, agg_scores, format_ties, roc_auc, roc_curve, ties_report, tpr_at_fpr,
)
from common.cache import check_dims, load_npz, spec_sha1  # noqa: E402
import gru_features as gf  # noqa: E402  特征集注册表 / 原始帧表 / 单帧基线

rel = gf.rel  # 显示路径用，报错信息里不因为路径转换再抛异常


def model_path(featset: str) -> Path:
    """权重路径带**数据集 + 特征集**双重身份。

    只带特征集是不够的。URFD 与 URFD_clean 产出的特征**维度、列名、规格指纹
    完全相同**，只有数值分布不同 —— 同一个文件名下，换数据集重训会**静默冲掉**
    旧权重，而拿甲数据集训的权重去乙数据集上跑也**照样通过校验**。
    （两次都实测踩过：一次是冒烟测试冲掉了正式权重，一次是 checksum 拦不住。）
    """
    return ROOT / "models" / "pt" / f"gru_fall_{gf.DATA_DIR.name}_{featset}.pt"

# ── GRU 超参（简单，够演示）──
WINDOW = 30          # 每窗口帧数（≈1s）
STRIDE = 10          # 滑窗步长
HIDDEN = 32
LAYERS = 1
EPOCHS = 40
BATCH = 64
LR = 1e-3
SEED = 0

SF_RATIO_THRES = 0.2    # 单帧基线：触发帧占比 ≥ 该值判该视频跌倒（与 bench_pose_fall 一致）
TEST_FRAC = 0.25        # 测试视频占比（按视频划分，防同视频窗口泄漏）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 指标落盘目录。experiments/results/ 在 .gitignore 里被放行了 json/csv ——
# 文档里的每个数字都该能在这里找到对应产物，否则半年后无人能核对。
RESULTS_DIR = ROOT / "experiments" / "results"


# ──────────────── 1. 取特征与单帧基线（只读缓存，不再自己跑模型）────────────────

def load_feats(featset: str) -> dict:
    """严格加载特征缓存（文件名/指纹/形状三重校验，见 common.cache）。

    特征由 `scripts/gru_features.py` 从原始帧表产出，本脚本不再自己跑模型 ——
    所以「换个特征看看」是秒级操作，不需要重训也不需要重新推理。
    """
    fs = gf.FEATSETS[featset]
    cache = gf.FEAT_CACHE[featset]
    if not cache.exists():
        raise SystemExit(
            f"特征缓存不存在：{gf.rel(cache)}\n"
            f"  → 1) 先产原始帧表（跑模型，较慢；断点可续）：\n"
            f"        python scripts/gru_features.py frames --mode {fs.frames}"
            f"{' --resume' if fs.frames == 'pose' else ''}\n"
            f"  → 2) 再从帧表生成特征（秒级，可反复跑）：\n"
            f"        python scripts/gru_features.py feats --featsets {featset}"
        )
    what = f"[{featset}] 特征缓存"
    npz, _meta = load_npz(cache, fs.spec, what=what)
    names = sorted(k for k in npz.files
                   if k not in ("__meta__", "__spec_sha1__") and not k.endswith("_label"))
    if not names:
        raise SystemExit(f"{what}里没有任何视频条目：{gf.rel(cache)}")
    check_dims(npz, names, fs.dim, what=what, path=cache)
    return {k: {"feats": npz[k], "label": float(npz[f"{k}_label"][0])} for k in names}


def load_baselines(frameset: str, keys) -> dict:
    """从原始帧表重算单帧基线：{规则名: {视频名: 触发帧占比}}。

    为什么要回帧表取：单帧基线是"不接时序"的对照方法，它的判定规则（severity 阈值
    或 judge 融合判定）不属于任何特征集，而是帧级事实。两条规则都能从帧表重算，
    所以**任何特征集都能同时给出两条基线** —— 原来 det 与 pose 各用一条规则却被
    并排比较的口径问题，到这里就没了。
    """
    table = gf.load_frame_table(frameset)
    out: dict[str, dict[str, float]] = {r: {} for r in gf.SF_RULES}
    for k in keys:
        sfs = gf.single_frame_baselines(table[k])
        for rule in gf.SF_RULES:
            out[rule][k] = float(sfs[rule].mean())
    return out


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


def train_model(train_x, train_y, in_dim, seed=SEED, test_x=None, test_y=None, verbose=True):
    """训练。test_x/test_y 只用于打印中途准确率，可为 None（LOVO 的每折测试集只有
    一段视频、只有一类标签，窗口级准确率无意义，就不传）。"""
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
        if verbose and test_x is not None and (epoch % 10 == 0 or epoch == 1):
            acc = eval_acc(model, test_x, test_y)
            print(f"  epoch {epoch:3d}  train_loss={total / len(tx):.4f}  test_acc={acc * 100:.1f}%")
    return model


def eval_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(x, device=DEVICE))).cpu().numpy()
    return ((p >= 0.5).astype(int) == y.astype(int)).mean()


def save_checkpoint(path, model, featset: str) -> None:
    """保存权重 + 身份证（特征集名 / 指纹 / 输入维度 / 网络结构）。

    为什么不能只存裸 state_dict：**维度恰好一致时它会静默加载成功**（例如同一
    维度内调换了特征顺序），模型照跑不报错，只有指纹能拦住。维度不一致时 torch
    自己会抛 RuntimeError，但信息是 `size mismatch` 这种天书。
    """
    fs = gf.FEATSETS[featset]
    torch.save({"state_dict": model.state_dict(),
                "featset": featset,
                "feat_spec_sha1": spec_sha1(fs.spec),
                "in_dim": fs.dim,
                "dataset": gf.DATA_DIR.name,     # 数据集身份（见 model_path 的说明）
                "hidden": HIDDEN, "layers": LAYERS,
                "window": WINDOW, "stride": STRIDE}, path)


def load_checkpoint(path, featset: str):
    """加载权重并校验身份（供 T0.4 的 edge_node 复用；本轮实验流程也用它做回读校验）。"""
    fs = gf.FEATSETS[featset]
    ckpt = torch.load(path, map_location="cpu")
    want = spec_sha1(fs.spec)
    want_ds = gf.DATA_DIR.name
    if ckpt.get("feat_spec_sha1") != want or ckpt.get("in_dim") != fs.dim:
        raise SystemExit(
            f"权重与当前特征集不匹配：{rel(path)}\n"
            f"  权重：特征集 {ckpt.get('featset')} / 指纹 {ckpt.get('feat_spec_sha1')} / "
            f"{ckpt.get('in_dim')} 维\n"
            f"  当前：特征集 {featset} / 指纹 {want} / {fs.dim} 维"
        )
    # 数据集也要验：两个数据集的特征指纹**完全相同**，只验指纹拦不住跨集误用。
    # 说明：旧权重没有 dataset 字段（None），这里放行但提示一次，避免历史权重全废。
    got_ds = ckpt.get("dataset")
    if got_ds is None:
        print(f"  [提示] 该权重未记录数据集（旧格式），无法校验来源：{rel(path)}")
    elif got_ds != want_ds:
        raise SystemExit(
            f"权重来自别的数据集：{rel(path)}\n"
            f"  权重：{got_ds}\n"
            f"  当前：{want_ds}（{gf.DATA_DIR}）\n"
            f"  → 两个数据集的特征维度与指纹相同、只有数值分布不同，**不会报形状错**，\n"
            f"    跑出来的结果看着正常但没有意义。请用当前数据集重新训练。"
        )
    model = FallGRU(ckpt["in_dim"], ckpt.get("hidden", HIDDEN), ckpt.get("layers", LAYERS))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    return model


# ────────────────────────── 3. 评估 ──────────────────────────


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


#: 单帧基线规则的短名（用于指标键名）
SF_SLUG = {"severity>=0.5": "severity", "judge(框+姿态融合)": "judge"}
#: 各模式的主口径（保持与以往结果可比：det 看 severity，pose 看 judge）
PRIMARY_SF = {"det": "severity", "pose": "judge"}


def run_mode(featset: str, data: dict, bases: dict, seed: int = SEED,
             verbose: bool = True) -> dict:
    fs = gf.FEATSETS[featset]
    mode = fs.mode
    if verbose:
        print(f"\n########## 特征集 [{featset}]（{fs.dim} 维/帧，帧表 {fs.frames}）"
              f"· seed={seed} ##########")
    train_videos, test_videos = split_videos(data, seed)

    # 窗口只切一次，训练/测试/评分共用。旧版把测试集切了 3 遍（两个列表推导各一遍
    # 取 X 和 y，video_scores 内部又一遍），纯浪费且是改造聚合方式时要动的地方。
    train_win = build_windows(data, train_videos)
    test_win = build_windows(data, test_videos)
    if not train_win or not test_win:
        raise SystemExit(
            f"[{featset}] 窗口为空：检查视频是否都短于 WINDOW={WINDOW} 帧"
        )
    train_x = np.concatenate([w for w, _ in train_win.values()])
    train_y = np.concatenate([l for _, l in train_win.values()])
    test_x = np.concatenate([w for w, _ in test_win.values()])
    test_y = np.concatenate([l for _, l in test_win.values()])
    if verbose:
        print(f"设备 {DEVICE} · 训练 {len(train_videos)} 视频 / 测试 {len(test_videos)} 视频 · "
              f"训练窗口 {len(train_x)} 条 / 测试窗口 {len(test_x)} 条")

    model = train_model(train_x, train_y, fs.dim, seed, test_x, test_y, verbose)
    ckpt = model_path(featset)
    if seed == 0:  # 只保留 seed=0 的模型文件
        save_checkpoint(ckpt, model, featset)
        load_checkpoint(ckpt, featset)  # 回读校验：让保存路径也有测试守着

    # ── 视频级判定 ──
    labels = np.array([data[v]["label"] for v in test_videos])
    # 每段测试视频的窗口数（长度 ∝ 视频时长；这是"取 max 聚合"偏置的来源，必须报出来）
    n_win = np.array([len(test_win[v][0]) if v in test_win else 0 for v in test_videos])
    probs = window_probs(model, test_win)
    # 三种聚合同时算：max 会把"整段视频是否跌倒"交给最像跌倒的那个窗口，是口径最松、
    # 也最容易受视频长度影响的选法（窗口数 3~38，长视频机会多）；mean 会被大量正常帧
    # 稀释；top3mean 居中。结论对聚合方式的敏感性本身就是一条要报告的结果。
    gs_scores = {how: np.array([agg_scores(probs.get(v, []), how) for v in test_videos])
                 for how in DEFAULT_AGGS}

    # AUC 走 common.metrics：并列分数按秩公式处理（旧的 np.argsort+trapz 在并列时
    # 给出的是排序的副产品而非确定值，见 src/common/metrics.py 顶部说明）。
    auc_gs = {how: roc_auc(labels, s) for how, s in gs_scores.items()}
    gs_curves = {how: roc_curve(labels, s) for how, s in gs_scores.items()}
    fpr_gs, tpr_gs = gs_curves["max"]

    # ── 单帧基线：两条规则都报（任何特征集都能给出这两条，口径不再混）──
    sf = {slug: np.array([bases[rule][v] for v in test_videos])
          for rule, slug in SF_SLUG.items()}
    auc_sf = {slug: roc_auc(labels, s) for slug, s in sf.items()}
    sf_curves = {slug: roc_curve(labels, s) for slug, s in sf.items()}
    primary = PRIMARY_SF[mode]
    fpr_sf, tpr_sf = sf_curves[primary]
    ties = ties_report(labels, sf[primary])

    if verbose:
        for rule, slug in SF_SLUG.items():
            s = sf[slug]
            tp = sum(1 for n in test_videos
                     if data[n]["label"] == 1 and bases[rule][n] >= SF_RATIO_THRES)
            fp = sum(1 for n in test_videos
                     if data[n]["label"] == 0 and bases[rule][n] >= SF_RATIO_THRES)
            mark = " ←主口径" if slug == primary else ""
            print(f"  单帧基线[{rule:18s}] 检出 {tp}/{(labels == 1).sum()} · "
                  f"误报 {fp}/{(labels == 0).sum()} · AUC={auc_sf[slug]:.3f}{mark}")
        print(f"  基线分数并列体检（主口径）: {format_ties(ties)}")
        print(f"  ROC-AUC（0.5=随机）: 单帧({primary})={auc_sf[primary]:.3f} → "
              f"GRU(max)={auc_gs['max']:.3f}")
        for how in DEFAULT_AGGS:
            f, t = gs_curves[how]
            print(f"    聚合 [{how:8s}] AUC={auc_gs[how]:.3f}  "
                  f"@FPR0={tpr_at_fpr(f, t, 0.0) * 100:.0f}%  "
                  f"@FPR≤0.1={tpr_at_fpr(f, t, 0.1) * 100:.0f}%")
        print(f"  测试视频窗口数: {n_win.min()}~{n_win.max()}（相差 "
              f"{n_win.max() / max(n_win.min(), 1):.0f} 倍，max 聚合下长视频机会更多）")
        if seed == 0:
            print(f"  模型已存: {rel(ckpt)}")

    out = {"auc_sf": auc_sf[primary], "auc_gs": auc_gs["max"],
           "tpr0_sf": tpr_at_fpr(fpr_sf, tpr_sf, 0.0),
           "tpr0_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.0),
           "tpr1_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.1),
           "tpr1_sf": tpr_at_fpr(fpr_sf, tpr_sf, 0.1),
           "ties_max_group": ties["max_group"],
           "n_win_min": int(n_win.min()), "n_win_max": int(n_win.max())}
    for slug in SF_SLUG.values():  # 两条基线各自一套
        f, t = sf_curves[slug]
        out[f"auc_sf_{slug}"] = auc_sf[slug]
        out[f"tpr0_sf_{slug}"] = tpr_at_fpr(f, t, 0.0)
    for how in DEFAULT_AGGS:  # 聚合敏感性：每个聚合各一套指标
        f, t = gs_curves[how]
        out[f"auc_gs_{how}"] = auc_gs[how]
        out[f"tpr0_gs_{how}"] = tpr_at_fpr(f, t, 0.0)
    return out


# ────────────────────── 4. 留一视频交叉验证（LOVO）──────────────────────
#
# 为什么需要它：随机划分下每个 seed 的测试集只有 8 段跌倒视频，于是
#   AUC 的最小刻度 = 1/(8×10) = 0.0125       @FPR=0 的刻度 = 1/8 = 12.5%
# 在这个尺度上「谁比谁高 0.025」毫无意义 —— 实测 5 个 seed 里符号翻转 2 次。
#
# LOVO 让每段视频轮流当测试集，把 out-of-fold 分数拼起来算**一次** ROC：
# 分母变成 30 正 × 40 负 = 1200 对 → AUC 刻度 0.00083，比原来细 15 倍。
#
# 注意是「池化后只算一次 ROC」，不是「对 70 个单折 AUC 求平均」——
# 单折的测试集只有一段视频、只有一类标签，AUC 无定义。
def run_lovo(featset: str, data: dict, bases: dict, verbose: bool = True) -> dict:
    fs = gf.FEATSETS[featset]
    keys = sorted(data)
    wins = build_windows(data, keys)   # 每段只切一次，70 折复用
    if not wins:
        raise SystemExit(f"[{featset}] 没有任何视频能切出窗口（都短于 WINDOW={WINDOW} 帧）")

    print(f"\n########## LOVO · 特征集 [{featset}]（{fs.dim} 维）· {len(keys)} 折 ##########")
    oof = {how: {} for how in DEFAULT_AGGS}
    n_win: dict[str, int] = {}
    t0 = time.time()
    for i, held in enumerate(keys, 1):
        train_keys = [k for k in keys if k != held and k in wins]
        if len(train_keys) < 2:
            raise SystemExit("可训练的视频太少，LOVO 无意义")
        train_x = np.concatenate([wins[k][0] for k in train_keys])
        train_y = np.concatenate([wins[k][1] for k in train_keys])
        model = train_model(train_x, train_y, fs.dim, SEED, verbose=False)
        p = window_probs(model, {held: wins[held]})[held] if held in wins else np.array([])
        n_win[held] = len(p)
        for how in DEFAULT_AGGS:
            oof[how][held] = agg_scores(p, how)
        if verbose and (i % 10 == 0 or i == len(keys)):
            el = time.time() - t0
            print(f"  [{i}/{len(keys)}] 用时 {el:.0f}s  "
                  f"预计剩余 {el / i * (len(keys) - i) / 60:.1f}min", flush=True)

    labels = np.array([data[k]["label"] for k in keys])
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    gs = {how: np.array([oof[how][k] for k in keys]) for how in DEFAULT_AGGS}
    auc = {how: roc_auc(labels, s) for how, s in gs.items()}
    curves = {how: roc_curve(labels, s) for how, s in gs.items()}
    fpr_gs, tpr_gs = curves["max"]

    sf = {slug: np.array([bases[rule][k] for k in keys]) for rule, slug in SF_SLUG.items()}
    auc_sf = {slug: roc_auc(labels, s) for slug, s in sf.items()}
    sf_curves = {slug: roc_curve(labels, s) for slug, s in sf.items()}
    primary = PRIMARY_SF[fs.mode]

    if verbose:
        print(f"\n  池化后 {len(keys)} 段（{n_pos} 正 / {n_neg} 负）→ "
              f"AUC 最小刻度 1/{n_pos * n_neg} = {1 / (n_pos * n_neg):.5f}"
              f"（随机划分下是 1/80 = 0.0125，细了 {n_pos * n_neg / 80:.0f} 倍）")
        for rule, slug in SF_SLUG.items():
            f, t = sf_curves[slug]
            print(f"  单帧基线[{rule:18s}] AUC={auc_sf[slug]:.4f}  "
                  f"@FPR0={tpr_at_fpr(f, t, 0.0) * 100:.1f}%")
        for how in DEFAULT_AGGS:
            f, t = curves[how]
            print(f"  GRU [{how:8s}] AUC={auc[how]:.4f}  @FPR0={tpr_at_fpr(f, t, 0.0) * 100:.1f}%  "
                  f"@FPR≤0.1={tpr_at_fpr(f, t, 0.1) * 100:.1f}%")

    def at_zero_fpr(scores) -> list[str]:
        """零误报工作点能捞到的跌倒视频：分数严格高于**所有**日常视频者。

        记录具体是哪几段（而不是只记个数），这样两个特征集之间可以做配对比较：
        「姿态多捞回了哪几段」比「高了 0.025 个 AUC」可解释得多。
        """
        s = np.asarray(scores, dtype=np.float64)
        thr = s[labels == 0].max()
        return sorted(k for k, v in zip(keys, s) if data[k]["label"] == 1 and v > thr)

    out = {"auc_sf": auc_sf[primary], "auc_gs": auc["max"],
           "tpr0_sf": tpr_at_fpr(*sf_curves[primary], 0.0),
           "tpr0_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.0),
           "tpr1_gs": tpr_at_fpr(fpr_gs, tpr_gs, 0.1),
           "tpr1_sf": tpr_at_fpr(*sf_curves[primary], 0.1),
           "ties_max_group": ties_report(labels, sf[primary])["max_group"],
           "n_win_min": min(n_win.values()), "n_win_max": max(n_win.values()),
           "hits_at_fpr0": {how: at_zero_fpr(gs[how]) for how in DEFAULT_AGGS},
           "per_video_score": {how: {k: float(oof[how][k]) for k in keys}
                               for how in DEFAULT_AGGS}}
    for slug in SF_SLUG.values():
        out[f"auc_sf_{slug}"] = auc_sf[slug]
        out[f"tpr0_sf_{slug}"] = tpr_at_fpr(*sf_curves[slug], 0.0)
        out.setdefault("hits_at_fpr0", {})[f"sf_{slug}"] = at_zero_fpr(sf[slug])
    for how in DEFAULT_AGGS:
        f, t = curves[how]
        out[f"auc_gs_{how}"] = auc[how]
        out[f"tpr0_gs_{how}"] = tpr_at_fpr(f, t, 0.0)
    return out


def config_snapshot() -> dict:
    """本次实验的全部可调量。写进 json，避免"改了配置忘了记录"。"""
    return {
        "window": WINDOW, "stride": STRIDE, "hidden": HIDDEN, "layers": LAYERS,
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED,
        "test_frac": TEST_FRAC, "sf_ratio_thres": SF_RATIO_THRES,
        "aggs": list(DEFAULT_AGGS), "sf_rules": list(gf.SF_RULES),
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
    """mean±std。显式接收列表（旧版用闭包读循环变量 mode，多特征集扩展必踩坑）。

    只有一个样本时（LOVO 只有一趟）不打印意义为 0 的 ±0。
    """
    a = np.array([r[key] for r in mlist], dtype=np.float64) * scale
    if len(a) == 1:
        return fmt.split("±")[0].format(a[0])
    return fmt.format(a.mean(), a.std())


def mean_of(mlist, key):
    return float(np.array([r[key] for r in mlist], dtype=np.float64).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featsets", nargs="+", default=list(gf.FEATSETS),
                    choices=list(gf.FEATSETS),
                    help=f"要跑的特征集（默认全部）：{' '.join(gf.FEATSETS)}")
    ap.add_argument("--seeds", type=int, default=5, help="随机 seed 数（每个 seed 重新划分+重训）")
    ap.add_argument("--protocol", choices=["random", "lovo"], default="random",
                    help="random=按视频随机划分×N seed；lovo=留一视频交叉验证"
                         "（70 折，out-of-fold 池化后算一次 ROC，AUC 刻度细 15 倍）")
    ap.add_argument("--out-json", type=Path, default=RESULTS_DIR / "metrics_latest.json",
                    help="指标落盘路径（默认 experiments/results/metrics_latest.json）")
    ap.add_argument("--data", type=Path, default=None,
                    help="数据集根目录（默认 data/datasets/URFD）。切换会同时切特征缓存目录 —— "
                         "旧数据是 [深度|RGB] 拼接图，见 docs/08 §12")
    args = ap.parse_args()
    if args.data is not None:
        gf.set_dataset(args.data)
    seeds = args.seeds
    lovo = args.protocol == "lovo"

    metrics: dict[str, list] = {}   # 特征集 -> [每轮（seed 或 LOVO 一趟）的指标 dict]
    bases_cache: dict[str, dict] = {}   # 帧表 -> 单帧基线（同一张表被多个特征集复用）
    for featset in args.featsets:
        fs = gf.FEATSETS[featset]
        data = load_feats(featset)
        if fs.frames not in bases_cache:
            bases_cache[fs.frames] = load_baselines(fs.frames, data.keys())
        bases = bases_cache[fs.frames]
        if lovo:
            metrics[featset] = [run_lovo(featset, data, bases)]
            continue
        print(f"\n########## 特征集 [{featset}] · {seeds} 个 seed ##########")
        mlist = []
        for s in range(seeds):
            r = run_mode(featset, data, bases, seed=s, verbose=(seeds == 1))
            mlist.append(r)
            print(f"  seed{s}: 单帧@FPR0={r['tpr0_sf'] * 100:.0f}%  "
                  f"GRU@FPR0={r['tpr0_gs'] * 100:.0f}%  "
                  f"GRU@FPR≤0.1={r['tpr1_gs'] * 100:.0f}%  "
                  f"AUC 单帧→GRU={r['auc_sf']:.3f}→{r['auc_gs']:.3f}  "
                  f"(基线并列组最大 {r['ties_max_group']})")
        metrics[featset] = mlist

    names = list(metrics)
    prot_desc = ("LOVO（70 折，out-of-fold 池化后算一次 ROC）" if lovo
                 else f"mean±std over {seeds} seeds")

    print(f"\n\n========== 稳健性汇总（{prot_desc}） ==========")
    print(f"{'特征集':10s} {'维':>3s} {'单帧检出@FPR0':>15s} {'GRU@FPR0':>11s} "
          f"{'GRU@FPR≤0.1':>12s} {'AUC单帧':>9s} {'AUC-GRU':>9s}")
    for name in names:
        mlist = metrics[name]
        print(f"{name:10s} {gf.FEATSETS[name].dim:3d} {mstd(mlist, 'tpr0_sf'):>15s} "
              f"{mstd(mlist, 'tpr0_gs'):>11s} {mstd(mlist, 'tpr1_gs'):>12s} "
              f"{mean_of(mlist, 'auc_sf'):8.3f} {mean_of(mlist, 'auc_gs'):8.3f}")

    print(f"\n\n========== 聚合方式敏感性（同一批 GRU 窗口分数，只换汇总方式） ==========")
    print("窗口 → 视频级分数的汇总方式不同，结论会不同；单帧基线未参与，纯 GRU 侧对照。")
    print(f"{'特征集':10s} {'聚合':<10s} {'AUC':>16s} {'检出@FPR0':>16s}")
    for name in names:
        mlist = metrics[name]
        for how in DEFAULT_AGGS:
            print(f"{name:10s} {how:<10s} "
                  f"{mstd(mlist, f'auc_gs_{how}', scale=1.0, fmt='{:.3f}±{:.3f}'):>16s} "
                  f"{mstd(mlist, f'tpr0_gs_{how}'):>16s}")

    print(f"\n\n========== 单帧基线两条规则对比（同一批测试视频，不含任何时序） ==========")
    print("原来 det 用 severity、pose 用 judge，两条不同规则却被并排比较；现在两边都给。")
    print(f"{'特征集':10s} {'规则':<18s} {'AUC':>16s} {'检出@FPR0':>16s}")
    for name in names:
        mlist = metrics[name]
        for slug in SF_SLUG.values():
            print(f"{name:10s} {slug:<18s} "
                  f"{mstd(mlist, f'auc_sf_{slug}', scale=1.0, fmt='{:.3f}±{:.3f}'):>16s} "
                  f"{mstd(mlist, f'tpr0_sf_{slug}'):>16s}")

    if lovo:
        print(f"\n\n========== LOVO 配对对比（每段视频都当过一次测试集） ==========")
        print("LOVO 下每段视频都有 out-of-fold 分数，且各特征集用的是**同一批**折，")
        print("所以可以直接问：零误报工作点上各自捞回了哪几段跌倒视频。")
        all_fall = sorted(k for k in metrics[names[0]][0]["per_video_score"]["max"]
                          if k.startswith("fall/"))
        print(f"\n跌倒视频共 {len(all_fall)} 段。零误报时各特征集捞回的段数：")
        hits = {n: set(metrics[n][0]["hits_at_fpr0"]["max"]) for n in names}
        for n in names:
            print(f"  {n:10s} {len(hits[n]):2d}/{len(all_fall)}")
        base = set(metrics[names[0]][0]["hits_at_fpr0"][f"sf_{PRIMARY_SF['det']}"])
        print(f"  {'单帧基线':10s} {len(base):2d}/{len(all_fall)}（severity 规则）")

        print(f"\n{'对比 (后者 − 前者)':<26s} {'AUC 差':>9s} {'捞回数差':>9s}  "
              f"后者多捞回 / 少捞回")
        for a, b in (("det.v1", "det.v2a"), ("det.v2a", "det.v2"),
                     ("det.v1", "det.v2"), ("pose.v1", "pose.v2"),
                     ("det.v2", "pose.v2")):
            if a not in metrics or b not in metrics:
                print(f"{b} − {a:<16s} {'（本次未跑，跳过）':>29s}")
                continue
            ra, rb = metrics[a][0], metrics[b][0]
            extra, lost = sorted(hits[b] - hits[a]), sorted(hits[a] - hits[b])
            fmt = lambda xs: ", ".join(x.split("/")[1].replace("-cam0.mp4", "") for x in xs) or "—"
            print(f"{b} − {a:<16s} {rb['auc_gs'] - ra['auc_gs']:+9.4f} "
                  f"{len(hits[b]) - len(hits[a]):+9d}  多: {fmt(extra)} | 少: {fmt(lost)}")

        still_missed = sorted(set(all_fall) - hits[names[-1]])
        print(f"\n所有特征集都捞不回的跌倒视频（{len(still_missed)} 段）："
              f"{', '.join(x.split('/')[1].replace('-cam0.mp4', '') for x in still_missed) or '—'}")
        print("这些是数据本身的难例（低位视角下检测器都找不到人），不是模型能解决的。")

    nw = [(r["n_win_min"], r["n_win_max"]) for r in metrics[names[0]]]
    print(f"\n每段测试视频的窗口数范围: {min(a for a, _ in nw)}~{max(b for _, b in nw)}"
          f"（max 聚合下窗口多的视频天然机会更多，这是与标签无关的长度偏置）")

    print("\n注：det.* 只用检测特征（边缘端 RK3588 NPU 上便宜），pose.* 额外带姿态几何量"
          "（需远端，用来回答「姿态值不值得卸载」）。")
    print("det.v2 与 pose.v2 的前 8 维完全相同，只差 2 个姿态量 —— 这样对比才是公平消融。")
    print("AUC 按并列取平均秩计算（common.metrics），单帧基线分数存在大量并列，"
          "旧的不稳定排序实现会给出偏高的任意值。")

    def _summary(mlist):
        """只对数值型指标求 mean±std。

        per_seed 里还带着 hits_at_fpr0 / per_video_score 这类非数值结果
        （它们描述"捞回了哪几段"，本来就无法求均值），原样留在 per_seed 里。
        """
        return {k: {"mean": float(np.mean([r[k] for r in mlist])),
                    "std": float(np.std([r[k] for r in mlist]))}
                for k, v in mlist[0].items() if isinstance(v, (int, float))}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/train_gru_fall.py",
        "env": env_snapshot(),
        "config": config_snapshot(),
        "protocol": args.protocol,
        "seeds": (None if lovo else seeds),
        "note": ("LOVO：每段视频轮流当测试集，out-of-fold 分数池化后算一次 ROC；"
                 "per_seed 里只有一项" if lovo else
                 "random：每个 seed 按视频随机划分 52 训练 / 18 测试"),
        "featset": {n: {"dim": gf.FEATSETS[n].dim, "names": list(gf.FEATSETS[n].names),
                        "frames": gf.FEATSETS[n].frames,
                        "feat_spec_sha1": spec_sha1(gf.FEATSETS[n].spec),
                        "primary_sf_rule": PRIMARY_SF[gf.FEATSETS[n].mode],
                        "cache": str(gf.FEAT_CACHE[n].relative_to(ROOT))} for n in names},
        "per_seed": metrics,
        "summary": {n: _summary(metrics[n]) for n in names},
    }
    write_metrics_json(args.out_json, payload)
    print(f"\n指标已落盘: {rel(args.out_json)}")


if __name__ == "__main__":
    main()
