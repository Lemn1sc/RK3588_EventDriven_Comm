# 模型 ↔ 传输 接口包（2026-11-18）

这是事件端（模型侧）交给控制端/传输端的接口定义：**只有定义和写端参考实现，
没有读端、传输实现**。同批发送的还有按 RKNN 2.3.2 重转的模型 `yolov8n_232.rknn`（4.1 MB）。

---

## 1. 包里有什么

| 文件 | 状态 | 说明 |
|---|---|---|
| `edge_shm.h` | **已定，只管适配** | 本机共享内存布局：Header / 记录 512 B / 事件 192 B / 帧环 |
| `edge_wire.h` | **建议稿，待拍板** | 板端 → 远端的 UDP 报文头：告警、图片分片、片段、结果 |
| `check_edge_shm_layout.c` | 工具 | 打印 `sizeof/offsetof`，用来核对 ABI |
| `shm_contract.md` | 背景 | 共享内存使用说明（字段含义、读写规则、常见疑问） |
| `reference/edge_shm.py` | 参考 | 写端实现，用来对照"写端到底保证了什么"；**读端不需要它** |

头文件依赖只有 `<stdint.h>` / `<stddef.h>`，C11 和 C++ 都能 include。
附 Python 是因为有些语义（补零、截断、`frame_id` 校验）写在代码里比写在文档里更清楚，
也方便你们写测试；C 读端不依赖它。
文件名和符号统一带 `edge_` 前缀，不会和 camera 侧的 `camera_shm` /
`shm_message` / `image_frame` 撞名。

---

## 2. 已定的部分：`/edge_infer_shm`

模型进程写、传输进程读，**独立于 `/camera_shm`**，两块可以共存。

```
HEADER      128 B    参数 + 发布序号 + 计数器
FRAME×64    32 KB    每推理帧一条（512 B），15 fps 约 4.3 s 历史
EVENT×16     3 KB    活跃事件快照（192 B），槽位 = event_id % 16
IMAGE×8     7.03 MB  BGR 原图环，槽位 = img_seq % 8
合计        7.07 MB
```

要点：

- 记录里的 `img_slot` 指向帧环槽位，`frame_id` 用来二次校验（防止读到被覆盖的旧图）；
- 图片是**原始 BGR**，不是 NV12、不是 JPEG；行跨度 = `w*3`；
- `n_det` 由写端填实际条数，读端校验 `n_det <= 6`；
- 坐标为归一化值，基准是**旋转后的摄像头原图**（当前 480×640），不是 letterbox 图；
- `best_conf` 是事件层分数（可能是规则分），`best_reliability` 才是校准概率，未校准时为 `-1`。

### 为什么图片不直接放在记录结构体里

一句话：**记录回答"发生了什么"，帧环回答"像素在哪"。** 一条记录 512 B，
一帧 BGR 921,600 B（480×640×3），差 **1800 倍**。所以记录里只放一个索引：

```c
uint16_t img_slot;   /* 偏移 46：本帧 BGR 在帧环中的槽位；0xFFFF = 无图 */
uint64_t frame_id;   /* 不用来定位，只用来校验——防止读到被覆盖的旧图 */
```

两块区域在同一块 `/edge_infer_shm` 里，一次 `mmap` 都能访问：

```
FRAME×64   32 KB     ← 证据：512 B/帧，记着「图在 3 号槽」
IMAGE×8    7.03 MB   ← 像素：BGR 原始字节，8 槽轮流覆盖
```

五条理由：

| 理由 | 如果把图内联会怎样 |
|---|---|
| **读路径分层**：只做决策（GRU、调度）时只读 512 B，决定要发图才碰 921 KB | 15 fps 下读端被迫搬 **13.8 MB/s**，而不是 **7.5 KB/s** |
| **环深度不同**：记录 64 槽（4.3 s 历史），图只有 8 槽（0.5 s） | 要么图环把记录历史砍到 0.5 s，要么记录区涨到 64×921 KB ≈ **56 MiB** |
| **写入粒度**：512 B 是微秒级，921 KB 约 0.5 ms | 坏帧/无框帧也得连着图一起写，"有记录、没图" 这种状态表达不了 |
| **两级告警**：信道差时只发几十字节 metadata 也能报警，图是可选增强 | metadata 必须等 921 KB 到齐才有意义，报警被拖慢 |
| **结构体定长**：读端能用 `seq` 直接算偏移 | 变长结构体，`seq → offset` 就不能直接算了 |

读端取图时，**`frame_id` 校验不能省**：

```c
/* 共享内存是原生对齐的，可以直接按结构体访问（这点和 edge_wire.h 的 packed 不同） */
const edge_frame_record *rec =
    (const edge_frame_record *)(base + edge_rec_offset(hdr->pub_seq));

if (rec->img_slot != EDGE_IMG_SLOT_NONE && rec->img_slot < EDGE_N_IMG) {
    uint64_t off = edge_img_offset(rec->img_slot);
    const edge_image_meta *meta = (const edge_image_meta *)(base + off);

    if (meta->frame_id == rec->frame_id) {           /* 图还没被覆盖 */
        const uint8_t *bgr = base + off + sizeof(edge_image_meta);  /* h*w*3 */
        /* 裁 ROI → JPEG → 发送 */
    } else {
        /* 已经被 8 帧之后的图覆盖：退回最新帧，或本次不发图 */
    }
}
```

失败语义：

- `img_slot == 0xFFFF`：这一帧没有图（坏帧，或当时没有可用帧）；
- `frame_id` 不匹配：图已被覆盖，**不要按槽位盲读**，否则会拿到别人的图；
- 两种情况下记录本身仍然有效——事件、框、置信度都还能用。这正是"没有图也能报警"的保证。

> 以上读取要在 `flock` 内完成。如果后续编码耗时较长，先把记录和图像**拷到本地缓冲**
> 再解锁——否则写端可能在你压缩到一半时覆盖这个槽。

### 三条和 `/camera_shm` 的差异（**别照抄现有接收端**）

| | `/camera_shm` | `/edge_infer_shm` |
|---|---|---|
| 槽位生命周期 | 租约：读完必须 `RELEASE` | **覆盖环：什么都不用归还** |
| 通知 | `SOCK_SEQPACKET` 的 READY/RELEASE | **轮询 `header.pub_seq`** |
| 互斥 | socket 握手 | **`flock("/tmp/edge_infer.lock")`** |

写端**从不等待消费者**。读端落后就丢，靠 `pub_seq` 的 gap 判断漏了几条，
靠 `img_overwrites` 判断图被覆盖了几次。没有背压，也就不会互相卡死。

---

## 3. 待拍板的问题

> 下面的问题都按**最终形态（事件驱动）**设计。
> v1 的"每个推理帧都传"只是把策略写死成 `transmit = true`，**接口不变**——
> 它同时也是对比实验里的"周期传输 100%"基线，所以必须和最终版本共用同一套布局。

### Q1. 通知机制：轮询还是门铃？

建议 **v1 轮询**：`if (hdr->pub_seq == last) usleep(1000);`，最多晚 1 ms，
而单帧推理本身 68 ms，这个延迟无意义。

如果希望事件驱动，可以复用 `ipc_transport` 加一个消息类型
`EDGE_RECORD_READY`（`frame_id` 字段放 `pub_seq`）。**共享内存布局不用改。**

### Q2. 线上报文用哪个字节序？

`edge_wire.h` 现在按**小端 + 1 字节对齐**写（板子和 PC 都是小端，头部只有几十字节，
直接 `memcpy` 最省事）。如果想让线上和本地 socket 一样统一用网络字节序，
改打包函数即可，**载荷和字段定义不受影响**。

### Q3. 片段（CLIP）现在就要吗？

建议 **v2 再做**，但报文结构先放着讨论。如果要做：

- 片段由**传输进程**维护滚动 JPEG 环（8~10 s，约 2 MB），不从模型侧帧环取；
- 用 **M-JPEG** 而不是 MP4：每帧独立，丢一包只坏一帧，板端也不用接硬件编码器；
- 片段是**报警之后的低优先级载荷**：信道不够就丢，不影响报警。

### Q4. 读端语言 + GRU 怎么跑？

读端 C/C++ 都可以，布局不受影响。GRU 推理我们建议：

| 方式 | 依赖 | 备注 |
|---|---|---|
| numpy 手写前向 | 板上已有 numpy | hidden=64 单步约 20~50 µs，v1 首选 |
| onnxruntime | 装 arm64 wheel | 你们已有 `my_gru.onnx`，可直接用 |
| RKNN | 需重转，GRU 算子支持不确定 | 不建议 |

### Q5. 分片与重传策略

`edge_wire.h` 默认按 1400 B 分片、不做重传（缺片丢整帧）。有不同意见直接改常量。

---

## 4. 怎么快速验证 ABI

```bash
# 1. 编译期检查（头文件里有 21 条 _Static_assert，失败会直接编译不过）
cc -std=c11 -Wall -Wextra -Werror -I . check_edge_shm_layout.c -o layout && ./layout

# 2. 输出应当包含
#    edge_frame_record.size=512
#    edge_shm_header.size=128
#    off_ev=32896   off_img=35968   total_size=7409024
```

该头文件已用 `-Wall -Wextra -Wshadow -Wconversion -Wsign-conversion -Wcast-qual -Werror`
（cameraApp 的告警集）编译通过。你们 aarch64 工具链上编译时，
`_Static_assert` 会再跑一遍。

---

