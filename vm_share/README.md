# VM 共享目录

Windows 与 Ubuntu 虚拟机之间传文件的地方。**内容不入库**（见 `.gitignore`），
只有这份说明在版本库里。

## 在 VMware 里怎么配

**虚拟机 → 设置 → 选项 → 共享文件夹 → 总是启用 → 添加**

| 项 | 值 |
|---|---|
| 主机路径 | `D:\Coding_Project\AI_Project\RK3588_EventDriven_Comm\vm_share` |
| 名称 | `vm_share` |

VM 里挂载：

```bash
sudo mkdir -p /mnt/hgfs
sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
ls /mnt/hgfs/vm_share
```

> 装了 `open-vm-tools` 后通常要**重启一次 VM** 才能挂上。
> 想开机自动挂载，把下面一行加进 `/etc/fstab`：
> ```
> .host:/  /mnt/hgfs  fuse.vmhgfs-fuse  allow_other,defaults  0  0
> ```

## 目录约定

```
vm_share/
├── to_vm/      Windows 放进去、VM 去读
└── from_vm/    VM 放进去、Windows 去读
```

**为什么分两个方向**：避免出现「这个文件是谁覆盖谁的」搞不清 —— 尤其是同名文件
（比如 `convert_to_rknn.py` 两边都改过）时，方向明确就不用猜。

## 现在 to_vm/ 里有什么

| 文件 | 用途 |
|---|---|
| `to_ubuntu.tar.gz` | 打包：ONNX + 200 张校准图 + 转换脚本（23 MB） |
| `rknn_toolkit2-1.5.2+...-cp38-...whl` | PC 端转换工具（38 MB，Python 3.8） |
| `requirements_cp38-1.5.2.txt` | 上面那个 wheel 的依赖清单 |
| `convert_to_rknn.py` | 转换 + 验证脚本的最新版 |

## 以后会走这里的文件

| 方向 | 文件 |
|---|---|
| **Windows → VM** | ONNX 模型（重新导出时）、校准集（重新生成时）、转换脚本更新、**板端推理代码** |
| **VM → Windows** | `.rknn` 转换产物、simulator 验证输出、板端实测数据 |

> **数据集的 16 GB 不从这里走** —— 它只在 Windows 侧，VM 里不需要。
> 板端推理只需要 `.rknn`（5.4 MB），不需要训练数据。
