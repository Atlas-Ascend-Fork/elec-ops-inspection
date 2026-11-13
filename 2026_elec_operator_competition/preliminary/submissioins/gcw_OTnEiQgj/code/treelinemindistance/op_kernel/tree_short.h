/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tree_common.h"

namespace elec {

template <uint32_t width, uint32_t point_tile>
ELEC_FN void tree_short_eval_w8(
    const AscendC::LocalTensor<float>& work,
    const AscendC::LocalTensor<float>& result,
    uint32_t seg_n, uint32_t count) {
    constexpr uint32_t pairs = width * point_tile;
    constexpr uint32_t work_stride = (pairs & 64) ? pairs : (pairs + 64);
    auto s_ax = work[0], s_ux = work[24], s_vx = work[384], s_vy = work[392], s_vz = work[400];
    auto bpx = work[576], px = work[768];
    auto dx_p = work[1024], dy_p = work[1024 + 64], dz_p = work[1024 + 128];
    auto projection_p = work[1024 + 3 * work_stride];
    auto wx_p = work[1024 + 192], wy_p = work[1024 + 256], wz_p = work[1024 + 320];
    AscendC::Sub(s_ux, s_ux, s_ax, 24);
    AscendC::Brcb(bpx, px, 3, {1, 8});
    AscendC::Mul(s_vx, s_ux, s_ux, 24);
    const AscendC::BinaryRepeatParams seg_3axis{1, 1, 0, 8, 8, 1}, seg_blk0{1, 1, 0, 8, 8, 0};
    const AscendC::BinaryRepeatParams proj_3axis{1, 1, 0, 8, 0, 1}, contig{1, 1, 1, 8, 8, 8};
    const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8};
    AscendC::Sub<float, true>(dx_p, bpx, s_ax, uint64_t(64), 3, seg_3axis);
    AscendC::Add(s_vx, s_vx, s_vy, seg_n);
    AscendC::Add(s_vx, s_vx, s_vz, seg_n);
    AscendC::Maxs(s_vx, s_vx, bits_float(0x00800000u), seg_n);
    AscendC::Mul<float, true>(wx_p, dx_p, s_ux, uint64_t(64), 3, seg_3axis);
    AscendC::Add<float, false>(wx_p, wx_p, wy_p, uint64_t(64), 1, contig);
    AscendC::Add<float, false>(projection_p, wx_p, wz_p, uint64_t(64), 1, contig);
    AscendC::Div<float, false>(projection_p, projection_p, s_vx, uint64_t(64), 1, seg_blk0);
    AscendC::Maxs<float, false>(projection_p, projection_p, 0.0f, uint64_t(64), 1, ucontig);
    AscendC::Mins<float, false>(projection_p, projection_p, 1.0f, uint64_t(64), 1, ucontig);
    AscendC::Mul<float, false>(wx_p, projection_p, s_ux, uint64_t(64), 3, proj_3axis);
    AscendC::Sub<float, false>(dx_p, dx_p, wx_p, uint64_t(64), 3, contig);
    AscendC::Mul<float, false>(dx_p, dx_p, dx_p, uint64_t(64), 3, contig);
    AscendC::Add<float, false>(dx_p, dx_p, dy_p, uint64_t(64), 1, contig);
    AscendC::Add<float, false>(dx_p, dx_p, dz_p, uint64_t(64), 1, contig);
    AscendC::WholeReduceMin<float, true>(result, dx_p, static_cast<int32_t>(seg_n), count, 1, 1, 1);
}

template <uint32_t width, uint32_t point_tile>
ELEC_FN void tree_short_eval_w32(
    const AscendC::LocalTensor<float>& work,
    const AscendC::LocalTensor<float>& result,
    uint32_t seg_n, uint32_t count) {
    constexpr uint32_t pairs = width * point_tile;
    constexpr uint32_t work_stride = (pairs & 64) ? pairs : (pairs + 64);
    auto s_ax = work[0], s_ay = work[32], s_az = work[64];
    auto s_ux = work[96], s_uy = work[128], s_uz = work[160];
    auto s_vx = work[384], s_vy = work[416], s_vz = work[448];
    auto bpx = work[576], bpy = work[640], bpz = work[704], px = work[768];
    auto dx_p = work[1024], dy_p = work[1024 + work_stride], dz_p = work[1024 + 2 * work_stride];
    auto projection_p = work[1024 + 3 * work_stride];
    auto temp_p = work[1024 + 4 * work_stride], d2_p = work[1024 + 5 * work_stride], extra_p = work[1024 + 6 * work_stride];
    const AscendC::BinaryRepeatParams point_segment{1, 0, 1, 4, 1, 0}, segment{1, 1, 1, 4, 4, 0}, contig4{1, 1, 1, 4, 4, 4};
    const AscendC::UnaryRepeatParams ucontig4{1, 1, 4, 4};
    const uint8_t r_cnt = static_cast<uint8_t>(count);
    AscendC::Sub<float, true>(s_ux, s_ux, s_ax, uint64_t(32), 3, contig4);
    AscendC::Brcb(bpx, px, 3, {1, 8});
    AscendC::Mul<float, false>(s_vx, s_ux, s_ux, uint64_t(32), 3, contig4);
    AscendC::Add<float, false>(s_vx, s_vx, s_vy, uint64_t(32), 1, contig4);
    AscendC::Add<float, false>(s_vx, s_vx, s_vz, uint64_t(32), 1, contig4);
    AscendC::Maxs<float, false>(s_vx, s_vx, bits_float(0x00800000u), uint64_t(32), 1, ucontig4);
    AscendC::Sub<float, false>(dx_p, bpx, s_ax, uint64_t(32), r_cnt, point_segment);
    AscendC::Sub<float, false>(dy_p, bpy, s_ay, uint64_t(32), r_cnt, point_segment);
    AscendC::Sub<float, false>(dz_p, bpz, s_az, uint64_t(32), r_cnt, point_segment);
    AscendC::Mul<float, false>(projection_p, dx_p, s_ux, uint64_t(32), r_cnt, segment);
    AscendC::Mul<float, false>(temp_p, dy_p, s_uy, uint64_t(32), r_cnt, segment);
    AscendC::Mul<float, false>(d2_p, dz_p, s_uz, uint64_t(32), r_cnt, segment);
    AscendC::Add<float, false>(projection_p, projection_p, temp_p, uint64_t(32), r_cnt, contig4);
    AscendC::Add<float, false>(projection_p, projection_p, d2_p, uint64_t(32), r_cnt, contig4);
    AscendC::Div<float, false>(projection_p, projection_p, s_vx, uint64_t(32), r_cnt, segment);
    AscendC::Maxs<float, false>(projection_p, projection_p, 0.0f, uint64_t(32), r_cnt, ucontig4);
    AscendC::Mins<float, false>(projection_p, projection_p, 1.0f, uint64_t(32), r_cnt, ucontig4);
    AscendC::Mul<float, false>(temp_p, projection_p, s_ux, uint64_t(32), r_cnt, segment);
    AscendC::Mul<float, false>(d2_p, projection_p, s_uy, uint64_t(32), r_cnt, segment);
    AscendC::Mul<float, false>(extra_p, projection_p, s_uz, uint64_t(32), r_cnt, segment);
    AscendC::Sub<float, false>(dx_p, dx_p, temp_p, uint64_t(32), r_cnt, contig4);
    AscendC::Sub<float, false>(dy_p, dy_p, d2_p, uint64_t(32), r_cnt, contig4);
    AscendC::Sub<float, false>(dz_p, dz_p, extra_p, uint64_t(32), r_cnt, contig4);
    AscendC::Mul<float, false>(d2_p, dx_p, dx_p, uint64_t(32), r_cnt, contig4);
    AscendC::Mul<float, false>(temp_p, dy_p, dy_p, uint64_t(32), r_cnt, contig4);
    AscendC::Mul<float, false>(extra_p, dz_p, dz_p, uint64_t(32), r_cnt, contig4);
    AscendC::Add<float, false>(d2_p, d2_p, temp_p, uint64_t(32), r_cnt, contig4);
    AscendC::Add<float, false>(d2_p, d2_p, extra_p, uint64_t(32), r_cnt, contig4);
    AscendC::WholeReduceMin<float, true>(result, d2_p, static_cast<int32_t>(seg_n), count, 1, 1, 4);
}

template <typename T, uint32_t width, uint32_t point_tile>
class TreeShort {
public:
    static constexpr uint32_t pairs = width * point_tile, stage_size = 6 * width + 3 * point_tile;
    static constexpr uint32_t work_stride = (pairs & 64) ? pairs : (pairs + 64);
    static constexpr uint32_t raw_bytes = (stage_size * sizeof(T) + 31) / 32 * 32;
    static constexpr uint32_t stage_bytes = (stage_size * sizeof(float) + 31) / 32 * 32;
    static constexpr uint32_t work_floats = 1024 + 7 * work_stride;
    static constexpr uint32_t stage_at = raw_bytes, work_at = stage_at + stage_bytes;
    static constexpr uint32_t offset_at = work_at + work_floats * sizeof(float), result_at = offset_at + 96 * sizeof(int32_t);
    static constexpr uint32_t s_stride = (width == 8) ? 8 : 32;

    ELEC_FN void run(const Config& c, GM_ADDR points_address, GM_ADDR segments_address,
                     GM_ADDR distance_address, GM_ADDR index_address) {
        const uint32_t inv_d = static_cast<uint32_t>(c.k);
        const uint64_t per_batch = c.k >> 32;
        if (AscendC::GetBlockIdx() >= c.batch * per_batch) return;
        AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC, 0, stage_size);
        AscendC::LocalTensor<float> stage(AscendC::TPosition::VECCALC, stage_at, stage_size);
        auto values = (sizeof(T) == 2) ? stage : raw.template ReinterpretCast<float>();
        AscendC::LocalTensor<float> work(AscendC::TPosition::VECCALC, work_at, work_floats);
        AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC, offset_at, 96);
        AscendC::LocalTensor<float> result(AscendC::TPosition::VECCALC, result_at, 160);
        AscendC::GlobalTensor<T> points, segments, output;
        AscendC::GlobalTensor<int32_t> output_index;
        points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
        segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
        output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));
        bool offsets_ready = false;
        const uint32_t seg_n = static_cast<uint32_t>(c.n);
        for (uint64_t task = AscendC::GetBlockIdx(); task < c.batch * per_batch; task += AscendC::GetBlockNum()) {
            const uint64_t batch = (task * inv_d + task) >> 32;
            const uint64_t first = (task - batch * per_batch) * point_tile;
            const uint32_t count = static_cast<uint32_t>(c.m - first < point_tile ? c.m - first : point_tile);
            load_and_gather_short(c, batch, first, count, seg_n, offsets_ready, points, segments, raw, values, work, offsets);
            if constexpr (width == 8) {
                tree_short_eval_w8<width, point_tile>(work, result, seg_n, count);
            } else {
                tree_short_eval_w32<width, point_tile>(work, result, seg_n, count);
            }
            uint64_t rsvd_cnt = 0;
            const AscendC::GatherMaskParams gm_red{1, 1, 8, 0};
            auto distance = result[64];
            auto indices = result[96].template ReinterpretCast<int32_t>();
            auto packed = result[128].template ReinterpretCast<T>();
            AscendC::GatherMask(indices, result.template ReinterpretCast<int32_t>(), static_cast<uint8_t>(2), false, uint32_t(0), gm_red, rsvd_cnt);
            AscendC::GatherMask(distance, result, static_cast<uint8_t>(1), false, uint32_t(0), gm_red, rsvd_cnt);
            tree_write_outputs<T>(output, output_index, batch * c.m + first, distance, distance, packed, indices, count);
        }
    }

private:
    ELEC_FN void load_and_gather_short(
        const Config& c, uint64_t batch, uint64_t first, uint32_t count, uint32_t seg_n, bool& offsets_ready,
        const AscendC::GlobalTensor<T>& points,
        const AscendC::GlobalTensor<T>& segments,
        const AscendC::LocalTensor<T>& raw,
        const AscendC::LocalTensor<float>& values,
        const AscendC::LocalTensor<float>& work,
        const AscendC::LocalTensor<int32_t>& offsets) {
        const AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        if (offsets_ready) {
            tree_fence<AscendC::HardEvent::V_MTE2>();
            tree_fence<AscendC::HardEvent::MTE3_V>();
        }
        const AscendC::DataCopyExtParams load_segments{1, uint32_t(c.n * 6 * sizeof(T)), 0, 0, 0};
        const AscendC::DataCopyExtParams load_points{1, count * 3 * uint32_t(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(raw, segments[batch * c.n * 6], load_segments, padding);
        AscendC::DataCopyPad(raw[6 * width], points[(batch * c.m + first) * 3], load_points, padding);
        if (!offsets_ready) {
            auto seg_off = offsets, pt_off = offsets[32];
            for (uint32_t i = 0; i < 8; ++i) {
                seg_off.SetValue(i, static_cast<int32_t>(i) * 24);
                pt_off.SetValue(i, static_cast<int32_t>(i) * 12);
            }
            tree_fence<AscendC::HardEvent::S_V>();
            if constexpr (width == 32) {
                AscendC::Adds(seg_off[8], seg_off, int32_t(8 * 24), 8);
                AscendC::Adds(seg_off[16], seg_off, int32_t(16 * 24), 16);
            }
            offsets_ready = true;
        }
        tree_fence<AscendC::HardEvent::MTE2_V>();
        if constexpr (sizeof(T) == 2) {
            AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, 6 * width + count * 3);
        }
        auto seg_off_u32 = offsets.template ReinterpretCast<uint32_t>();
        auto pt_off_u32 = offsets[32].template ReinterpretCast<uint32_t>();
        for (uint32_t axis = 0; axis < 6; ++axis) {
            AscendC::Gather(work[axis * s_stride], values[axis], seg_off_u32, 0, seg_n);
        }
        AscendC::Gather(work[768], values[6 * width], pt_off_u32, 0, count);
        AscendC::Gather(work[776], values[6 * width + 1], pt_off_u32, 0, count);
        AscendC::Gather(work[784], values[6 * width + 2], pt_off_u32, 0, count);
    }
};

}  // namespace elec
