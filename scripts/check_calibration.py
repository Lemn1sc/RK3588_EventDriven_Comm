#!/usr/bin/env python
"""置信度校准自检（纯 numpy，不需要 torch / 数据）。

覆盖：
- Brier 的完美 / 最差 / 常数预测；
- 可靠性曲线的分箱、空 bin、p=1 边界、ECE；
- 保序回归的单调性、常数输入、线性插值；
- 输入校验与 JSON 可序列化（空 bin 用 None，不能是 NaN）。

用法：
    python scripts/check_calibration.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.calibration import (  # noqa: E402
    IsotonicCalibrator, brier_score, expected_calibration_error, isotonic_fit,
    reliability_curve,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"✓ {name}")
    else:
        FAILED.append(name)
        print(f"✗ {name}" + (f"\n    {detail}" if detail else ""))


def expect_raises(name: str, fn) -> None:
    try:
        fn()
    except (ValueError, RuntimeError):
        check(name, True)
    except Exception as exc:
        check(name, False, f"抛了非预期异常：{type(exc).__name__}: {exc}")
    else:
        check(name, False, "应当抛异常但没有")


def test_brier() -> None:
    check("完美预测 Brier=0",
          abs(brier_score([0, 1, 0, 1], [0, 1, 0, 1])) < 1e-12)
    check("完全反着预测 Brier=1",
          abs(brier_score([0, 1], [1, 0]) - 1.0) < 1e-12)
    check("常数 0.5 的 Brier=0.25",
          abs(brier_score([0, 1, 0, 1], [0.5] * 4) - 0.25) < 1e-12)


def test_reliability() -> None:
    y = [0, 1, 0, 1]
    p = [0.5] * 4
    rc = reliability_curve(y, p, n_bins=10)
    check("bin 边界长度 = n_bins+1", len(rc["bin_edges"]) == 11)
    check("counts 总和 = 样本数", sum(rc["counts"]) == 4)
    check("常数 0.5 的 ECE=0", abs(rc["ece"]) < 1e-12)
    check("常数 0.5 的 Brier=0.25", abs(rc["brier"] - 0.25) < 1e-12)
    check("p=0.5 落在中间 bin",
          rc["counts"][4] == 4 or rc["counts"][5] == 4, str(rc["counts"]))

    rc2 = reliability_curve([0, 1], [0.0, 1.0], n_bins=10)
    check("p=0 落第一个 bin", rc2["counts"][0] == 1, str(rc2["counts"]))
    check("p=1 落最后一个 bin", rc2["counts"][-1] == 1, str(rc2["counts"]))
    check("完美分箱 ECE=0", abs(rc2["ece"]) < 1e-12)


def test_isotonic() -> None:
    cal = isotonic_fit([0, 1, 2, 3], [0, 1, 0, 1])
    y = cal.y_
    check("保序回归输出单调不减", np.all(np.diff(y) >= -1e-12), str(y))
    pred = cal.transform([0, 0.5, 1, 2, 3])
    check("transform 输出单调不减", np.all(np.diff(pred) >= -1e-12), str(pred))
    check("transform 输出在 [0,1]", np.all((pred >= 0) & (pred <= 1)), str(pred))

    const1 = isotonic_fit([0, 1, 2], [1, 1, 1])
    check("全 1 输入 → 输出 1", np.allclose(const1.transform([0, 1, 2]), 1.0))
    const0 = isotonic_fit([0, 1, 2], [0, 0, 0])
    check("全 0 输入 → 输出 0", np.allclose(const0.transform([0, 1, 2]), 0.0))

    lin = isotonic_fit([0, 1], [0, 1])
    check("线性输入插值正确", abs(float(lin.transform([0.5])[0]) - 0.5) < 1e-9)

    clipped = lin.transform([-1, 2])
    check("越界输入保持端点常数", np.allclose(clipped, [0.0, 1.0]), str(clipped))


def test_validation() -> None:
    expect_raises("长度不一致应报错", lambda: brier_score([0, 1], [0]))
    expect_raises("非 0/1 标签应报错", lambda: brier_score([0, 2], [0.1, 0.9]))
    expect_raises("空输入应报错", lambda: brier_score([], []))
    expect_raises("n_bins<2 应报错", lambda: reliability_curve([0, 1], [0.1, 0.9], n_bins=1))
    expect_raises("未 fit 的 calibrator 应报错",
                  lambda: IsotonicCalibrator().transform([0.5]))
    expect_raises("isotonic 的非 0/1 标签应报错",
                  lambda: isotonic_fit([0, 1], [0, 2]))


def test_json() -> None:
    rc = reliability_curve([0, 1, 0], [0.1, 0.9, 0.1], n_bins=5)
    try:
        s = json.dumps(rc)
        ok = "bin_edges" in s and "null" in s   # 空 bin 必须是 None→null，不是 NaN
    except (TypeError, ValueError) as exc:
        ok = False
        s = str(exc)
    check("可靠性曲线可 JSON 序列化", ok, s[:120])


def test_ece() -> None:
    # 两个 bin，各 2 个样本：预测 0.2/0.8，真实频率 0/1 → ECE = 0.2
    y = [0, 0, 1, 1]
    p = [0.2, 0.2, 0.8, 0.8]
    ece = expected_calibration_error(y, p, n_bins=5)
    check("ECE 手算一致", abs(ece - 0.2) < 1e-9, f"实际 {ece}")


def main() -> int:
    for fn in (test_brier, test_reliability, test_isotonic, test_validation,
               test_json, test_ece):
        print(f"\n── {fn.__name__} ──")
        try:
            fn()
        except Exception as exc:
            FAILED.append(fn.__name__)
            print(f"✗ {fn.__name__} 抛异常：{type(exc).__name__}: {exc}")

    print()
    if FAILED:
        print(f"失败 {len(FAILED)} 项：{FAILED}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
