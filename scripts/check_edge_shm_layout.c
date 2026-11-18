/*
 * 共享内存布局探针：把 edge_shm.h 的 sizeof / offsetof 打印成 key=value，
 * 供 scripts/check_edge_shm.py 与 Python 侧布局逐项对比。
 *
 * 编译：cc -std=c11 -I src/edge_node scripts/check_edge_shm_layout.c -o layout
 * 运行：./layout
 *
 * 头文件里的 _Static_assert 已经在编译期检查过尺寸与关键偏移；本程序额外打印出来，
 * 是为了和 Python 侧做**跨语言**逐项比对（等价于读端按文档偏移直接读）。
 */
#include <stdio.h>
#include <stddef.h>

#include "edge_shm.h"

#define PRINT(name, expr) \
    printf("%s=%llu\n", (name), (unsigned long long)(expr))

int main(void)
{
    PRINT("edge_shm_header.size", sizeof(edge_shm_header));
    PRINT("edge_frame_record.size", sizeof(edge_frame_record));
    PRINT("edge_detection.size", sizeof(edge_detection));
    PRINT("edge_event_snapshot.size", sizeof(edge_event_snapshot));
    PRINT("edge_image_meta.size", sizeof(edge_image_meta));

    PRINT("edge_frame_record.img_slot", offsetof(edge_frame_record, img_slot));
    PRINT("edge_frame_record.event_vector", offsetof(edge_frame_record, event_vector));
    PRINT("edge_frame_record.best_kind", offsetof(edge_frame_record, best_kind));
    PRINT("edge_frame_record.best_conf", offsetof(edge_frame_record, best_conf));
    PRINT("edge_frame_record.kind_first_ts_ns",
          offsetof(edge_frame_record, kind_first_ts_ns));
    PRINT("edge_frame_record.dets", offsetof(edge_frame_record, dets));
    PRINT("edge_frame_record.reserved", offsetof(edge_frame_record, reserved));

    PRINT("edge_event_snapshot.kind", offsetof(edge_event_snapshot, kind));
    PRINT("edge_event_snapshot.kind_first_ts_ns",
          offsetof(edge_event_snapshot, kind_first_ts_ns));
    PRINT("edge_event_snapshot.bbox", offsetof(edge_event_snapshot, bbox));

    PRINT("edge_shm_header.pub_seq", offsetof(edge_shm_header, pub_seq));
    PRINT("edge_shm_header.last_update_ts_ns",
          offsetof(edge_shm_header, last_update_ts_ns));

    PRINT("off_rec", EDGE_REC_OFF);
    PRINT("off_ev", EDGE_EV_OFF);
    PRINT("off_img", EDGE_IMG_OFF);
    PRINT("img_size", EDGE_IMG_SIZE);
    PRINT("img_slot_size", EDGE_IMG_SLOT_SIZE);
    PRINT("total_size", EDGE_TOTAL_SIZE);

    PRINT("rec_offset.1", edge_rec_offset(1u));
    PRINT("rec_offset.65", edge_rec_offset(65u));
    PRINT("ev_offset.17", edge_ev_offset(17u));
    PRINT("img_offset.7", edge_img_offset(7u));
    return 0;
}
