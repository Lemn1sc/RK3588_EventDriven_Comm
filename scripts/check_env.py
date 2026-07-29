"""环境自检脚本（T0.1 验收）。

跑通说明算法环境就绪，可以进入 T0.2。
    python scripts/check_env.py
"""
import importlib
import sys


def check_import(module: str, pip_name: str | None = None) -> bool:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK]   {module:<14} {ver}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {module:<14} 未安装 ({pip_name or module}) -> {e}")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    print("-" * 56)

    ok = True
    for mod in ("numpy", "cv2", "matplotlib", "torch", "torchvision", "ultralytics"):
        ok &= check_import(mod)

    print("-" * 56)
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  CUDA 可用: {cuda}")
        if cuda:
            print(f"  设备      : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本 : {torch.version.cuda}")
        else:
            print("  提示: 未检测到 CUDA。若有 NVIDIA 显卡，请确认装的是 CUDA 版 torch。")
            print("       (nano 模型用 CPU 也能跑，只是慢一些，不阻塞开发。)")
    except Exception as e:  # noqa: BLE001
        print(f"  torch 检测失败: {e}")
        ok = False

    print("-" * 56)
    print("环境就绪 ✓ 可以进入 T0.2" if ok else "存在缺失，请按上面提示补装依赖。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
