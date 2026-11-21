/*
 * 模型 ↔ 传输 共享内存布局（C11）—— 交给控制端 / 传输端的唯一定义。
 *
 * 契约文档：docs/10_模型传输共享内存契约.md
 * Python 等价实现：src/edge_node/edge_shm.py（写端使用；本文件是给 C/C++ 读端 include 的）
 *
 * ── 怎么用 ──
 *
 * 只 include、不要改：结构体、常量、偏移都由这里定义。读端的职责是
 * shm_open + mmap + 校验 header + 轮询 pub_seq + 按 seq 取记录，不在本文件里。
 *
 * ── 三条必须知道的语义（和 camera_shm 完全不同）──
 *
 * 1. 这是**覆盖环**，不是租约：读端读完**不需要、也不能**发 RELEASE；
 *    写端从不等待消费者，落后就靠 header.pub_seq 的 gap 和 img_overwrites 体现。
 * 2. 没有 READY/RELEASE socket。读端轮询 header.pub_seq 即可（1 ms 间隔足够）。
 * 3. 跨进程互斥用 flock(/tmp/edge_infer.lock)，本文件不定义锁。
 *
 * ── ABI 约定 ──
 *
 * - 小端（RK3588 aarch64 原生）；
 * - 所有字段按 8 字节自然对齐，**不需要 `#pragma pack`**；加了也不会变，但别加；
 * - 所有宽度用 uint*_t / float，不用 long / bool；
 * - 下方 _Static_assert 会在编译期检查尺寸与关键偏移，布局漂移直接编译失败。
 */
#ifndef EDGE_SHM_H
#define EDGE_SHM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__cplusplus)
#define EDGE_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define EDGE_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

/* ── 名称与版本 ───────────────────────────────────────────────── */

#define EDGE_SHM_NAME   "/edge_infer_shm"
#define EDGE_LOCK_PATH  "/tmp/edge_infer.lock"

#define EDGE_SHM_MAGIC   0x45444731u   /* "EDG1" */
#define EDGE_SHM_VERSION 1u

/* ── 尺寸与数量 ───────────────────────────────────────────────── */

#define EDGE_SHM_HDR_SIZE   128u
#define EDGE_REC_SIZE       512u
#define EDGE_EV_SIZE        192u
#define EDGE_IMG_HDR_SIZE   32u

#define EDGE_N_REC  64u   /* 逐帧记录槽数（15 fps 约 4.3 s 历史） */
#define EDGE_N_EV   16u   /* 活跃事件槽数 */
#define EDGE_N_IMG  8u    /* 帧环槽数（15 fps 约 0.5 s） */

#define EDGE_MAX_DET        6u
#define EDGE_IMG_SLOT_NONE  0xFFFFu

#define EDGE_DEFAULT_IMG_W 480u
#define EDGE_DEFAULT_IMG_H 640u

/* ── 段偏移（与 docs/10 §1 一致：128 / 32896 / 35968 / 7409024）── */

#define EDGE_REC_OFF  (EDGE_SHM_HDR_SIZE)
#define EDGE_EV_OFF   (EDGE_REC_OFF + EDGE_N_REC * EDGE_REC_SIZE)
#define EDGE_IMG_OFF  ((EDGE_EV_OFF + EDGE_N_EV * EDGE_EV_SIZE + 63u) & ~63u)

#define EDGE_IMG_STRIDE    (EDGE_DEFAULT_IMG_W * 3u)
#define EDGE_IMG_SIZE      (EDGE_IMG_STRIDE * EDGE_DEFAULT_IMG_H)
#define EDGE_IMG_SLOT_SIZE (EDGE_IMG_HDR_SIZE + EDGE_IMG_SIZE)
#define EDGE_TOTAL_SIZE    (EDGE_IMG_OFF + EDGE_N_IMG * EDGE_IMG_SLOT_SIZE)

/* ── flags 位 ─────────────────────────────────────────────────── */

#define EDGE_MODEL_READY   (1u << 0)   /* header.flags */
#define EDGE_MODEL_EXITED  (1u << 1)   /* 正常释放/退出，读端可读完末条记录后退出 */

#define EDGE_FRAME_INFERRED      (1u << 0)   /* 本帧成功推理 */
#define EDGE_FRAME_BAD           (1u << 1)   /* 解码/推理失败，其余字段无效 */
#define EDGE_FRAME_NO_PERSON     (1u << 2)   /* 本帧没有 person 框 */
#define EDGE_FRAME_TS_FROM_CAM   (1u << 3)   /* capture_ts_ns 来自摄像头 */
#define EDGE_FRAME_TRACKER_ERROR (1u << 4)   /* 事件层异常，best_* 不可信 */

/*
 * kind / state / urgency 取 EventKind / EventState / Urgency 的数值：
 *   kind    0=空槽, 1=PERSON, 2=PERSON_ABNORMAL, 3=TRACK_LOST,
 *           4=SUSPECTED_FALL, 5=FALL_CONFIRMED, 6=BACKGROUND
 *   state   0=CANDIDATE, 1=ACTIVE, 2=CONFIRMED, 3=EXPIRED
 *   urgency 0=BACKGROUND, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL
 */

/* ── 结构体 ───────────────────────────────────────────────────── */

/* 单个检测框：24 B。cls 用 COCO 编号，person = 0。 */
typedef struct {
    uint16_t cls;
    uint16_t pad;   /* 必须为 0 */
    float    conf;
    float    x1;
    float    y1;
    float    x2;
    float    y2;
} edge_detection;

/*
 * 一帧推理结果：512 B。每个**推理帧**一条；被 stride 跳过的帧不发布。
 * n_det 由写端写成实际条数，读端应校验 n_det <= EDGE_MAX_DET。
 * 坐标为归一化值，基准是**旋转后的摄像头原图**（当前 480×640），不是 letterbox 图。
 */
typedef struct {
    uint64_t seq;              /* 发布号，从 1 开始；等于 header.pub_seq 当时值 */
    uint64_t frame_id;         /* 摄像头发布编号（camera_shm 的 frame_id），会跳号 */
    uint64_t capture_ts_ns;    /* 摄像头时间戳；无效时为 0 */
    uint64_t publish_ts_ns;    /* 写端发布时刻（CLOCK_MONOTONIC） */
    float    infer_ms;         /* 本帧从取帧到发布的耗时 */

    uint32_t flags;            /* EDGE_FRAME_* 位 */
    uint16_t n_det;            /* dets[] 中有效条数，0..6 */
    uint16_t n_person;         /* 全帧 person 总数（截断前） */
    uint16_t det_total;        /* 全帧检测框总数（截断前） */
    uint16_t img_slot;         /* 本帧 BGR 在帧环中的槽位；EDGE_IMG_SLOT_NONE = 无图 */

    float    event_vector[8];  /* 8 维事件向量，布局沿用 common.event_vector */

    float    aspect;           /* 主目标 高/宽 */
    float    cx;               /* 主目标归一化中心 */
    float    cy;
    float    w;                /* 主目标归一化宽高；框 = (cx-w/2, cy-h/2, cx+w/2, cy+h/2) */
    float    h;
    float    area_ratio;       /* 主目标框面积 / 画面面积 */

    uint32_t best_event_id;    /* 0 = 无事件 */
    uint8_t  best_kind;
    uint8_t  best_state;
    uint8_t  best_urgency;
    uint8_t  flags2;           /* 保留，置 0 */
    float    best_conf;        /* 事件层分数（规则分，**不是概率**） */
    float    best_reliability; /* 校准概率；未校准为 -1 */
    uint64_t kind_first_ts_ns; /* 当前 best_kind 首次出现（告警起点）；无则 0 */

    edge_detection dets[EDGE_MAX_DET];   /* person 优先，再按 conf 降序 */
    uint8_t  reserved[240];              /* 必须为 0；新字段只能放这里 */
} edge_frame_record;

/* 活跃事件快照：192 B。槽位 = event_id % EDGE_N_EV；kind=0 表示空槽。 */
typedef struct {
    uint64_t seq;
    uint32_t event_id;
    uint8_t  kind;
    uint8_t  state;
    uint8_t  urgency;
    uint8_t  pad0;
    uint64_t start_ts_ns;
    uint64_t last_seen_ts_ns;
    uint64_t kind_first_ts_ns;   /* 告警起点（不是 start_ts） */
    float    confidence;
    float    peak_confidence;
    float    reliability;        /* 未校准为 -1 */
    uint32_t hits;
    float    bbox[4];
    uint32_t pad1;
    uint32_t pad2;
    uint8_t  reserved[112];
} edge_event_snapshot;

/* 帧环槽位的 32 B 头；其后紧跟 img_size 字节的 BGR（行跨度 = w*3）。 */
typedef struct {
    uint64_t seq;
    uint64_t frame_id;
    uint64_t capture_ts_ns;
    uint32_t w;
    uint32_t h;
} edge_image_meta;

/* 全局头：128 B。 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t total_size;
    uint32_t n_rec;
    uint32_t rec_size;
    uint32_t n_ev;
    uint32_t ev_size;
    uint32_t n_img;
    uint32_t img_w;
    uint32_t img_h;
    uint32_t img_stride;
    uint32_t img_size;
    uint32_t flags;
    uint32_t pad0;
    uint32_t pad1;
    uint64_t pub_seq;            /* 已完成的记录发布数；读端轮询它 */
    uint64_t img_seq;            /* 帧环发布数 */
    uint64_t frames_ready;
    uint64_t frames_inferred;
    uint64_t frames_bad;
    uint64_t img_overwrites;
    uint32_t last_error;
    uint32_t pad2;
    uint64_t last_update_ts_ns;  /* 心跳：2 s 不变说明写端卡住/退出 */
} edge_shm_header;

/* ── 槽位定位（与 Python Layout 完全一致）────────────────────── */

/* 第 seq 条记录所在偏移；seq 从 1 开始。 */
static inline uint64_t edge_rec_offset(uint64_t seq)
{
    return (uint64_t)EDGE_REC_OFF + ((seq - 1u) % EDGE_N_REC) * EDGE_REC_SIZE;
}

/* 事件 id 对应槽偏移（event_id=0 也算，但语义上是空槽）。 */
static inline uint64_t edge_ev_offset(uint32_t event_id)
{
    return (uint64_t)EDGE_EV_OFF + (uint64_t)(event_id % EDGE_N_EV) * EDGE_EV_SIZE;
}

/* 帧环槽位偏移；slot 必须 < EDGE_N_IMG。 */
static inline uint64_t edge_img_offset(uint32_t slot)
{
    return (uint64_t)EDGE_IMG_OFF + (uint64_t)slot * EDGE_IMG_SLOT_SIZE;
}

/* ── 编译期 ABI 检查 ─────────────────────────────────────────── */

EDGE_STATIC_ASSERT(sizeof(edge_detection) == 24u, "edge_detection must be 24 B");
EDGE_STATIC_ASSERT(sizeof(edge_frame_record) == EDGE_REC_SIZE,
                   "edge_frame_record must be 512 B");
EDGE_STATIC_ASSERT(sizeof(edge_event_snapshot) == EDGE_EV_SIZE,
                   "edge_event_snapshot must be 192 B");
EDGE_STATIC_ASSERT(sizeof(edge_image_meta) == EDGE_IMG_HDR_SIZE,
                   "edge_image_meta must be 32 B");
EDGE_STATIC_ASSERT(sizeof(edge_shm_header) == EDGE_SHM_HDR_SIZE,
                   "edge_shm_header must be 128 B");

EDGE_STATIC_ASSERT(offsetof(edge_frame_record, img_slot) == 46u, "img_slot @46");
EDGE_STATIC_ASSERT(offsetof(edge_frame_record, event_vector) == 48u, "event_vector @48");
EDGE_STATIC_ASSERT(offsetof(edge_frame_record, best_kind) == 108u, "best_kind @108");
EDGE_STATIC_ASSERT(offsetof(edge_frame_record, best_conf) == 112u, "best_conf @112");
EDGE_STATIC_ASSERT(offsetof(edge_frame_record, kind_first_ts_ns) == 120u,
                   "kind_first_ts_ns @120");
EDGE_STATIC_ASSERT(offsetof(edge_frame_record, dets) == 128u, "dets @128");
EDGE_STATIC_ASSERT(offsetof(edge_frame_record, reserved) == 272u, "reserved @272");

EDGE_STATIC_ASSERT(offsetof(edge_event_snapshot, kind) == 12u, "kind @12");
EDGE_STATIC_ASSERT(offsetof(edge_event_snapshot, kind_first_ts_ns) == 32u,
                   "kind_first_ts_ns @32");
EDGE_STATIC_ASSERT(offsetof(edge_event_snapshot, bbox) == 56u, "bbox @56");

EDGE_STATIC_ASSERT(offsetof(edge_shm_header, pub_seq) == 64u, "pub_seq @64");
EDGE_STATIC_ASSERT(offsetof(edge_shm_header, last_update_ts_ns) == 120u,
                   "last_update_ts_ns @120");

EDGE_STATIC_ASSERT(EDGE_EV_OFF == 32896u, "events segment @32896");
EDGE_STATIC_ASSERT(EDGE_IMG_OFF == 35968u, "image segment @35968");
EDGE_STATIC_ASSERT(EDGE_IMG_SIZE == 921600u, "image payload must be 921600 B");
EDGE_STATIC_ASSERT(EDGE_TOTAL_SIZE == 7409024u, "total shm must be 7409024 B");

#ifdef __cplusplus
}   /* extern "C" */
#endif

#endif /* EDGE_SHM_H */
