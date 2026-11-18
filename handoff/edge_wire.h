/*
 * 板上 → 远端 的**线上协议结构**（建议稿，不是定稿）
 *
 * 归属：控制端 / 传输端。本文件只提供"字段从哪来、怎么摆"的建议，
 * 不实现分包、重组、重传、调度。
 *
 * ── 与 edge_shm.h 的区别（很重要）──
 *
 *   edge_shm.h  本机共享内存：原生对齐、不打包、只在小端 ARM 上跑
 *   edge_wire.h 网络报文：      1 字节对齐、显式小端、跨机器传输
 *
 * 报文都是定长头 + 变长载荷。解析时建议 memcpy 到本地变量或逐字段读，
 * 不要对 packed 结构体取成员地址后强转（成员可能未对齐）。
 *
 * ── 待讨论（见 README）──
 *
 *   1. 字节序：这里用小端（两端都是 LE 硬件，头部只有几十字节）。
 *      若组里统一用网络字节序，只需改本文件的注释和打包函数，载荷不受影响。
 *   2. 是否需要 CLIP（片段）相关报文，还是 v2 再说。
 *   3. 分片大小是否定 1400 B、是否做重传（当前建议：不重传，丢整帧）。
 */
#ifndef EDGE_WIRE_H
#define EDGE_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define EDGE_WIRE_PACKED __attribute__((packed))
#else
#define EDGE_WIRE_PACKED
#endif

#if defined(__cplusplus)
#define EDGE_WIRE_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define EDGE_WIRE_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#define EDGE_WIRE_MAGIC   0x4544u   /* "ED"：用于在混杂 UDP 流量里快速识别 */
#define EDGE_WIRE_VERSION 1u

/* 单个 UDP 报文的载荷上限（含本文件里的头）。留出 IP/UDP 开销。 */
#define EDGE_WIRE_MTU 1400u

/* 单张图的字节上限（ROI 通常 2~10 KB，FULL 通常 30~80 KB）。 */
#define EDGE_MAX_IMAGE_BYTES (256u * 1024u)

/* ── 消息类型 ───────────────────────────────────────────────── */

enum {
    EDGE_MSG_ALERT      = 1,   /* 事件告警：几十字节，优先送 */
    EDGE_MSG_IMAGE      = 2,   /* ROI / FULL 帧，分片 */
    EDGE_MSG_CLIP_START = 3,   /* 片段清单（可选，v2） */
    EDGE_MSG_CLIP_FRAG  = 4,   /* 片段的一帧，分片（可选，v2） */
    EDGE_MSG_RESULT     = 5,   /* 远端 → 板端：确认结果 */
    EDGE_MSG_PING       = 6,   /* 板端 → 远端：RTT 探测 */
    EDGE_MSG_PONG       = 7    /* 远端 → 板端：原样回传 probe_ts */
};

/* ── 优先级（放在 flags 低 2 位）────────────────────────────── */

#define EDGE_PRIO_MASK      0x0003u
#define EDGE_PRIO_BACKGROUND 0u   /* 片段：拥塞时第一个丢 */
#define EDGE_PRIO_NORMAL     1u   /* 完整帧 / 图像分片 */
#define EDGE_PRIO_HIGH       2u   /* ROI 确认帧 */
#define EDGE_PRIO_CRITICAL   3u   /* 告警 / 结果 */

#define EDGE_FLAG_RETRANSMIT (1u << 8)   /* 保留：若以后做重传 */

/* ── 通用包头：每条报文都以它开头 ───────────────────────────── */

typedef struct EDGE_WIRE_PACKED {
    uint16_t magic;      /* EDGE_WIRE_MAGIC */
    uint8_t  version;    /* EDGE_WIRE_VERSION */
    uint8_t  msg_type;   /* EDGE_MSG_* */
    uint16_t flags;      /* 低 2 位优先级；其余保留 */
    uint32_t seq;        /* 发送序号（单调递增），用于丢包率统计 */
    uint64_t ts_ns;      /* 发送时刻（板端 CLOCK_MONOTONIC） */
} edge_wire_hdr;         /* 18 B */

/* ── EDGE_MSG_ALERT：事件告警 ────────────────────────────────
 *
 * 必须能在"没有任何图片"的情况下独立成立：远端只凭这条就知道
 * 发生了什么、有多急、框在哪。
 * 字段来源：edge_frame_record 的 best_* / event_vector / dets 主目标。
 */
typedef struct EDGE_WIRE_PACKED {
    uint32_t event_id;         /* 事件层 id，跨帧稳定 */
    uint64_t frame_id;         /* 摄像头帧号，供后续图片对账 */
    uint64_t capture_ts_ns;    /* 大端链路起点：端到端时延从这里算 */
    uint64_t kind_first_ts_ns; /* 告警起点（不是事件起点） */
    float    confidence;       /* 事件层分数（规则分，不是概率） */
    float    reliability;      /* 校准概率；未校准为 -1 */
    float    bbox[4];          /* 主目标归一化 xyxy */
    uint8_t  kind;             /* EventKind */
    uint8_t  state;            /* EventState */
    uint8_t  urgency;          /* Urgency */
    uint8_t  score_kind;       /* 0=detector 1=track_lost_rule 2=gru_prob 3=calibrated */
    uint16_t deadline_ms;      /* 控制端建议的送达期限；0=不限 */
    uint16_t pad;              /* 置 0 */
} edge_alert;                  /* 60 B */

/* ── EDGE_MSG_IMAGE：一帧图片的分片 ──────────────────────────
 *
 * 一张 JPEG 拆成 frag_cnt 个 UDP 报文，每片都带完整头（便于乱序到达）。
 * 重组键：(image_id, frag_idx)。整帧 image_len 字节收齐才解码；
 * 缺任意一片就丢弃这一帧 —— ROI/FULL 都是独立 JPEG，没有跨帧依赖。
 *
 * crop_rect 是**原图像素坐标**：远端把关键点映射回原图时必须用它。
 * FULL 模式下 crop_rect = (0,0,src_w,src_h)。
 */
typedef struct EDGE_WIRE_PACKED {
    uint32_t image_id;         /* 同一张图的分片共享此 id */
    uint32_t event_id;         /* 0 = 与事件无关 */
    uint64_t frame_id;
    uint64_t capture_ts_ns;
    uint32_t image_len;        /* 整张 JPEG 字节数 */
    uint16_t frag_idx;         /* 0..frag_cnt-1 */
    uint16_t frag_cnt;
    uint16_t crop_x1;
    uint16_t crop_y1;
    uint16_t crop_x2;
    uint16_t crop_y2;
    uint16_t src_w;            /* 原图宽（映射用） */
    uint16_t src_h;
    uint8_t  mode;             /* 0=ROI, 1=FULL */
    uint8_t  quality;          /* JPEG 质量，供远端参考 */
} edge_image_hdr;              /* 46 B */

/* ── EDGE_MSG_CLIP_START：片段清单（可选）────────────────────
 *
 * 先发一条，告诉远端"接下来会来多少帧"。远端据此判断缺帧比例，
 * 不必靠超时猜结束。
 */
typedef struct EDGE_WIRE_PACKED {
    uint32_t clip_id;
    uint32_t event_id;
    uint64_t start_ts_ns;      /* 片段起点 = kind_first_ts - 预录时长 */
    uint64_t end_ts_ns;
    uint32_t total_bytes_hint; /* 估算值，仅供远端预留缓冲 */
    uint16_t n_frames;         /* 计划帧数；降级时小于实际窗口帧数 */
    uint16_t fps;              /* 原始帧率（当前 15） */
    uint16_t w;
    uint16_t h;
    uint8_t  kind;
    uint8_t  urgency;
    uint8_t  quality;
    uint8_t  pad;
} edge_clip_manifest;          /* 40 B */

/* ── EDGE_MSG_CLIP_FRAG：片段中某一帧的一片 ────────────────
 *
 * 重组键：(clip_id, frame_idx, frag_idx)。M-JPEG：每帧独立，
 * 丢一片只坏一帧，不重传。
 */
typedef struct EDGE_WIRE_PACKED {
    uint32_t clip_id;
    uint64_t capture_ts_ns;    /* 该帧的采集时刻，用于还原时间轴 */
    uint16_t frame_idx;        /* 0..n_frames-1 */
    uint16_t frag_idx;
    uint16_t frag_cnt;
    uint16_t jpeg_len;         /* 该帧 JPEG 总字节数（≤ 65535） */
} edge_clip_frag_hdr;         /* 20 B */

/* ── EDGE_MSG_RESULT：远端确认结果 ──────────────────────────
 *
 * 远端回传 verdict + 少量姿态导出量（不必传 17×3 原始关键点）。
 * echo_ts_ns 原样回传 ALERT/IMAGE 里的 ts_ns，板端据此算 RTT，
 * **不需要两端时钟同步**。
 */
typedef struct EDGE_WIRE_PACKED {
    uint32_t event_id;
    uint64_t frame_id;
    uint64_t echo_ts_ns;       /* 回传请求方的 ts_ns */
    float    confidence;
    float    torso_angle_deg;  /* 躯干与竖直方向夹角 */
    float    head_hip_dy;      /* 头—髋高度差（归一化，图像 y 向下） */
    float    body_aspect;      /* 姿态外接框 高/宽 */
    uint32_t proc_ms;          /* 远端处理耗时 */
    uint8_t  verdict;          /* 0=NO_FALL, 1=FALL, 2=UNCERTAIN, 3=FAIL */
    uint8_t  pad[3];
} edge_result;                 /* 44 B */

/* ── EDGE_MSG_PING / PONG：RTT 探测 ─────────────────────────
 *
 * PING 带 probe_ts_ns；PONG 原样回传，板端用 (now - probe) 得到 RTT，
 * 不依赖两端时钟一致。
 */
typedef struct EDGE_WIRE_PACKED {
    uint64_t probe_ts_ns;
    uint32_t seq;
} edge_ping;                   /* 12 B */

/* ── 编译期尺寸检查（与 edge_shm.h 同样的纪律）────────────── */

EDGE_WIRE_ASSERT(sizeof(edge_wire_hdr) == 18u, "edge_wire_hdr must be 18 B");
EDGE_WIRE_ASSERT(sizeof(edge_alert) == 60u, "edge_alert must be 60 B");
EDGE_WIRE_ASSERT(sizeof(edge_image_hdr) == 46u, "edge_image_hdr must be 46 B");
EDGE_WIRE_ASSERT(sizeof(edge_clip_manifest) == 40u, "edge_clip_manifest must be 40 B");
EDGE_WIRE_ASSERT(sizeof(edge_clip_frag_hdr) == 20u, "edge_clip_frag_hdr must be 20 B");
EDGE_WIRE_ASSERT(sizeof(edge_result) == 44u, "edge_result must be 44 B");
EDGE_WIRE_ASSERT(sizeof(edge_ping) == 12u, "edge_ping must be 12 B");

#ifdef __cplusplus
}   /* extern "C" */
#endif

#endif /* EDGE_WIRE_H */
