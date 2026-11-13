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

template <typename T, uint32_t points_per_tile = 16, bool HAS_TAIL1 = false, bool BATCH_PINNED = false>
class TreePairs {
public:
    static constexpr uint32_t segment_width = 64, point_stride = 512, seg_stride = 1024, seg_planes = 9;
    static constexpr uint32_t offset_count = 1024 + 2 * points_per_tile;
    static constexpr uint32_t seg_at = 256, raw_at = 37376, point_at = 43520;
    static constexpr uint32_t offset_at = 44288, red_all_at = 49152, result_at = 57344;
    static constexpr uint32_t pt_raw_at = 60416, pt_f32_at = 61440;
    static constexpr uint32_t dx_at = 65536, dy_at = 81920, dz_at = 98304, p0_at = 114688;
    static constexpr uint32_t r0_at = 131328, p1_at = 147712, q0_at = 164096;

    struct Buffers {
        AscendC::LocalTensor<T> raw{AscendC::TPosition::VECCALC, raw_at, 3 * point_stride * (sizeof(float) / sizeof(T))};
        AscendC::LocalTensor<T> point_raw{AscendC::TPosition::VECCALC, pt_raw_at, 192};
        AscendC::LocalTensor<float> point_f32{AscendC::TPosition::VECCALC, pt_f32_at, 192};
        AscendC::LocalTensor<float> soa{AscendC::TPosition::VECCALC, seg_at, seg_planes * seg_stride};
        AscendC::LocalTensor<float> r0{AscendC::TPosition::VECCALC, r0_at, 4096};
        AscendC::LocalTensor<float> dx{AscendC::TPosition::VECCALC, dx_at, 4096};
        AscendC::LocalTensor<float> dy{AscendC::TPosition::VECCALC, dy_at, 4096};
        AscendC::LocalTensor<float> dz{AscendC::TPosition::VECCALC, dz_at, 4096};
        AscendC::LocalTensor<float> p0{AscendC::TPosition::VECCALC, p0_at, 4096};
        AscendC::LocalTensor<float> p1{AscendC::TPosition::VECCALC, p1_at, 4096};
        AscendC::LocalTensor<float> q0{AscendC::TPosition::VECCALC, q0_at, 4096};
        AscendC::LocalTensor<float> point{AscendC::TPosition::VECCALC, point_at, 192};
        AscendC::LocalTensor<int32_t> offsets{AscendC::TPosition::VECCALC, offset_at, offset_count};
        AscendC::LocalTensor<float> result{AscendC::TPosition::VECCALC, result_at, 512};
        AscendC::LocalTensor<float> reduced_all{AscendC::TPosition::VECCALC, red_all_at, 2048};
        AscendC::LocalTensor<T> seg_raw{AscendC::TPosition::VECCALC, 131072, 8192 * (sizeof(float) / sizeof(T))};
        AscendC::LocalTensor<float> segments{AscendC::TPosition::VECCALC, dx_at, 8192};
    };

    ELEC_FN void init_offsets(const AscendC::LocalTensor<int32_t>& offsets, uint32_t n_pow2) {
        for (uint32_t i = 0; i < 8; ++i) {
            offsets.SetValue(i, static_cast<int32_t>(i) * 24);
            offsets.SetValue(1024 + i, static_cast<int32_t>(i) * 12);
        }
        tree_fence<AscendC::HardEvent::S_V>();
        if constexpr (HAS_TAIL1) {
            AscendC::Duplicate(offsets[n_pow2], static_cast<int32_t>(n_pow2) * 24, 64);
        }
        const uint32_t max_n = (n_pow2 > points_per_tile) ? n_pow2 : points_per_tile;
        for (uint32_t filled = 8; filled < max_n; filled *= 2) {
            if (filled < n_pow2) {
                AscendC::Adds(offsets[filled], offsets, static_cast<int32_t>(filled) * 24, filled);
            }
            if (filled < points_per_tile) {
                AscendC::Adds(offsets[1024 + filled], offsets[1024], static_cast<int32_t>(filled) * 12, filled);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    ELEC_FN void prepare_segments(const Config& c, const Buffers& b, uint32_t n_aligned, uint8_t n_rep) {
        auto ax = b.soa[0 * seg_stride], ay = b.soa[1 * seg_stride], az = b.soa[2 * seg_stride];
        auto ux = b.soa[3 * seg_stride], uy = b.soa[4 * seg_stride], uz = b.soa[5 * seg_stride];
        auto vx = b.soa[6 * seg_stride], vy = b.soa[7 * seg_stride], vz = b.soa[8 * seg_stride];
        const AscendC::BinaryRepeatParams contig{1, 1, 1, 8, 8, 8};
        const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8};
        if constexpr (sizeof(T) == 2) {
            AscendC::Cast(b.segments, b.seg_raw, AscendC::RoundMode::CAST_NONE, c.n * 6);
        }
        auto seg_src = (sizeof(T) == 2) ? b.segments : b.seg_raw.template ReinterpretCast<float>();
        for (uint32_t axis = 0; axis < 6; ++axis) {
            AscendC::Gather(b.soa[axis * seg_stride], seg_src[axis], b.offsets.template ReinterpretCast<uint32_t>(), 0, n_aligned);
        }
        AscendC::Sub<float, true>(ux, ux, ax, uint64_t(64), n_rep, contig);
        AscendC::Sub<float, false>(uy, uy, ay, uint64_t(64), n_rep, contig);
        AscendC::Sub<float, false>(uz, uz, az, uint64_t(64), n_rep, contig);
        AscendC::Mul<float, false>(vx, ux, ux, uint64_t(64), n_rep, contig);
        AscendC::Mul<float, false>(b.r0, uy, uy, uint64_t(64), n_rep, contig);
        AscendC::Mul<float, false>(b.q0, uz, uz, uint64_t(64), n_rep, contig);
        AscendC::Add<float, false>(vx, vx, b.r0, uint64_t(64), n_rep, contig);
        AscendC::Add<float, false>(vx, vx, b.q0, uint64_t(64), n_rep, contig);
        AscendC::Maxs<float, false>(vx, vx, bits_float(0x00800000u), uint64_t(64), n_rep, ucontig);
        AscendC::Div<float, false>(vy, uy, vx, uint64_t(64), n_rep, contig);
        AscendC::Div<float, false>(vz, uz, vx, uint64_t(64), n_rep, contig);
        AscendC::Div<float, false>(vx, ux, vx, uint64_t(64), n_rep, contig);
        if constexpr (HAS_TAIL1) {
            const uint32_t last_seg = static_cast<uint32_t>(c.n - 1);
            auto seg_brc = b.reduced_all[1280];
            for (uint32_t plane = 0; plane < 9; ++plane) {
                AscendC::Brcb(seg_brc[plane * 64], b.soa[plane * seg_stride + last_seg], 1, {1, 8});
            }
        }
    }

    ELEC_FN void eval_chunks(const Config& c, uint32_t n_pow2, uint32_t count, const Buffers& b) {
        auto broadcast = b.raw.template ReinterpretCast<float>();
        AscendC::Brcb(broadcast, b.point, 24, {1, 8});
        auto ax = b.soa[0 * seg_stride], ay = b.soa[1 * seg_stride], az = b.soa[2 * seg_stride];
        auto ux = b.soa[3 * seg_stride], uy = b.soa[4 * seg_stride], uz = b.soa[5 * seg_stride];
        auto vx = b.soa[6 * seg_stride], vy = b.soa[7 * seg_stride], vz = b.soa[8 * seg_stride];
        const AscendC::BinaryRepeatParams point_segment{1, 0, 1, 8, 1, 0}, segment{1, 1, 1, 8, 8, 0}, contig{1, 1, 1, 8, 8, 8};
        const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8};
        const uint8_t r_cnt = static_cast<uint8_t>(count);
        for (uint32_t start = 0; start < n_pow2; start += segment_width) {
            if (start == 0) {
                AscendC::Sub<float, true>(b.dx, broadcast, ax[start], uint64_t(64), r_cnt, point_segment);
            } else {
                AscendC::Sub<float, false>(b.dx, broadcast, ax[start], uint64_t(64), r_cnt, point_segment);
            }
            AscendC::Sub<float, false>(b.dy, broadcast[point_stride], ay[start], uint64_t(64), r_cnt, point_segment);
            AscendC::Sub<float, false>(b.dz, broadcast[2 * point_stride], az[start], uint64_t(64), r_cnt, point_segment);
            AscendC::Mul<float, false>(b.p0, b.dx, vx[start], uint64_t(64), r_cnt, segment);
            AscendC::Mul<float, false>(b.r0, b.dy, vy[start], uint64_t(64), r_cnt, segment);
            AscendC::Mul<float, false>(b.p1, b.dz, vz[start], uint64_t(64), r_cnt, segment);
            AscendC::Add<float, false>(b.p0, b.p0, b.r0, uint64_t(64), r_cnt, contig);
            AscendC::Add<float, false>(b.p0, b.p0, b.p1, uint64_t(64), r_cnt, contig);
            AscendC::Maxs<float, false>(b.q0, b.p0, 0.0f, uint64_t(64), r_cnt, ucontig);
            AscendC::Mins<float, false>(b.p0, b.q0, 1.0f, uint64_t(64), r_cnt, ucontig);
            AscendC::Mul<float, false>(b.r0, b.p0, ux[start], uint64_t(64), r_cnt, segment);
            AscendC::Mul<float, false>(b.p1, b.p0, uy[start], uint64_t(64), r_cnt, segment);
            AscendC::Mul<float, false>(b.q0, b.p0, uz[start], uint64_t(64), r_cnt, segment);
            AscendC::Sub<float, false>(b.dx, b.dx, b.r0, uint64_t(64), r_cnt, contig);
            AscendC::Sub<float, false>(b.dy, b.dy, b.p1, uint64_t(64), r_cnt, contig);
            AscendC::Sub<float, false>(b.dz, b.dz, b.q0, uint64_t(64), r_cnt, contig);
            AscendC::Mul<float, false>(b.dx, b.dx, b.dx, uint64_t(64), r_cnt, contig);
            AscendC::Mul<float, false>(b.r0, b.dy, b.dy, uint64_t(64), r_cnt, contig);
            AscendC::Mul<float, false>(b.p1, b.dz, b.dz, uint64_t(64), r_cnt, contig);
            AscendC::Add<float, false>(b.dx, b.dx, b.r0, uint64_t(64), r_cnt, contig);
            AscendC::Add<float, false>(b.p0, b.dx, b.p1, uint64_t(64), r_cnt, contig);
            auto red_dst = b.reduced_all[(start >> 6) * 128];
            const uint32_t valid = HAS_TAIL1 ? 64u : static_cast<uint32_t>(c.n - start < segment_width ? c.n - start : segment_width);
            if (valid == 64u) {
                AscendC::WholeReduceMin<float, false>(red_dst, b.p0, int32_t(64), static_cast<int32_t>(count), 1, 1, 8);
            } else {
                AscendC::WholeReduceMin<float, true>(red_dst, b.p0, static_cast<int32_t>(valid), static_cast<int32_t>(count), 1, 1, 8);
            }
        }
    }

    ELEC_FN void eval_tail1(uint32_t last_seg, uint32_t last_off, const Buffers& b) {
        auto seg_brc = b.reduced_all[1280];
        auto dy_i32 = b.dy.template ReinterpretCast<int32_t>();
        const AscendC::BinaryRepeatParams one_seg{1, 1, 0, 8, 8, 0}, contig{1, 1, 1, 8, 8, 8};
        const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8};
        AscendC::Sub<float, true>(b.dz, b.point, seg_brc, uint64_t(64), 1, one_seg);
        AscendC::Sub<float, false>(b.dz[64], b.point[64], seg_brc[64], uint64_t(64), 1, one_seg);
        AscendC::Sub<float, false>(b.dz[128], b.point[128], seg_brc[128], uint64_t(64), 1, one_seg);
        AscendC::Mul<float, false>(b.p0, b.dz, seg_brc[384], uint64_t(64), 1, one_seg);
        AscendC::Mul<float, false>(b.r0, b.dz[64], seg_brc[448], uint64_t(64), 1, one_seg);
        AscendC::Mul<float, false>(b.p1, b.dz[128], seg_brc[512], uint64_t(64), 1, one_seg);
        AscendC::Add<float, false>(b.p0, b.p0, b.r0, uint64_t(64), 1, contig);
        AscendC::Add<float, false>(b.p0, b.p0, b.p1, uint64_t(64), 1, contig);
        AscendC::Maxs<float, false>(b.q0, b.p0, 0.0f, uint64_t(64), 1, ucontig);
        AscendC::Mins<float, false>(b.p0, b.q0, 1.0f, uint64_t(64), 1, ucontig);
        AscendC::Mul<float, false>(b.r0, b.p0, seg_brc[192], uint64_t(64), 1, one_seg);
        AscendC::Mul<float, false>(b.p1, b.p0, seg_brc[256], uint64_t(64), 1, one_seg);
        AscendC::Mul<float, false>(b.q0, b.p0, seg_brc[320], uint64_t(64), 1, one_seg);
        AscendC::Sub<float, false>(b.dz, b.dz, b.r0, uint64_t(64), 1, contig);
        AscendC::Sub<float, false>(b.dz[64], b.dz[64], b.p1, uint64_t(64), 1, contig);
        AscendC::Sub<float, false>(b.dz[128], b.dz[128], b.q0, uint64_t(64), 1, contig);
        AscendC::Mul<float, false>(b.dz, b.dz, b.dz, uint64_t(64), 1, contig);
        AscendC::Mul<float, false>(b.r0, b.dz[64], b.dz[64], uint64_t(64), 1, contig);
        AscendC::Mul<float, false>(b.p1, b.dz[128], b.dz[128], uint64_t(64), 1, contig);
        AscendC::Add<float, false>(b.dz, b.dz, b.r0, uint64_t(64), 1, contig);
        AscendC::Add<float, false>(b.dx[last_off], b.dz, b.p1, uint64_t(64), 1, contig);
        AscendC::Duplicate<int32_t, false>(dy_i32[last_off], static_cast<int32_t>(last_seg), uint64_t(64), 1, 1, 8);
    }

    ELEC_FN void merge_chunk_winners(const Config& c, const Buffers& b, const AscendC::LocalTensor<int32_t>& best_index) {
        const uint32_t num_chunks = static_cast<uint32_t>((c.n + 63) >> 6);
        const uint32_t gm_chunks = HAS_TAIL1 ? (num_chunks - 1) : num_chunks;
        uint64_t rsvd_cnt = 0;
        const AscendC::GatherMaskParams gm_params{1, static_cast<uint16_t>(gm_chunks * 2), 8, 0};
        auto dy_i32 = b.dy.template ReinterpretCast<int32_t>();
        auto idx_dst = (num_chunks == 1) ? best_index : dy_i32;
        AscendC::GatherMask(idx_dst, b.reduced_all.template ReinterpretCast<int32_t>(), static_cast<uint8_t>(2), false, uint32_t(0), gm_params, rsvd_cnt);
        AscendC::GatherMask(b.dx, b.reduced_all, static_cast<uint8_t>(1), false, uint32_t(0), gm_params, rsvd_cnt);
        if (num_chunks <= 1) return;
        const AscendC::BinaryRepeatParams contig{1, 1, 1, 8, 8, 8};
        const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8};
        if (gm_chunks > 1) {
            AscendC::Adds<int32_t, true>(dy_i32[64], dy_i32[64], 64, uint64_t(64), 1, ucontig);
            for (uint32_t c_idx = 2; c_idx < gm_chunks; ++c_idx) {
                AscendC::Adds<int32_t, false>(dy_i32[c_idx * 64], dy_i32[c_idx * 64], static_cast<int32_t>(c_idx * 64), uint64_t(64), 1, ucontig);
            }
        }
        if constexpr (HAS_TAIL1) {
            eval_tail1(static_cast<uint32_t>(c.n - 1), gm_chunks * 64, b);
        }
        auto best_idx_f32 = best_index.template ReinterpretCast<float>();
        __ubuf__ float* dx_ptr = reinterpret_cast<__ubuf__ float*>(b.dx.GetPhyAddr());
        __ubuf__ float* dy_ptr = reinterpret_cast<__ubuf__ float*>(b.dy.GetPhyAddr());
        __ubuf__ float* best_idx_ptr = reinterpret_cast<__ubuf__ float*>(best_idx_f32.GetPhyAddr());
        for (uint32_t c_idx = 1; c_idx < num_chunks; ++c_idx) {
            AscendC::Compare<float, false>(b.dx[c_idx * 64], b.dx, AscendC::CMPMODE::LT, uint64_t(64), contig);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SelectCal<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(dx_ptr, dx_ptr + c_idx * 64, dx_ptr, 1, contig);
            __ubuf__ float* dst_ptr = (c_idx + 1 == num_chunks) ? best_idx_ptr : dy_ptr;
            AscendC::SelectCal<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(dst_ptr, dy_ptr + c_idx * 64, dy_ptr, 1, contig);
        }
    }

    ELEC_FN bool resolve_task(
        const Config& c, uint64_t task, uint32_t inv_d,
        uint64_t pinned_batch, uint64_t cid, uint64_t cores_per_batch,
        uint64_t per_batch, uint64_t full_waves, uint64_t tail_tile, uint64_t wave_cap,
        uint64_t& batch, uint64_t& first, uint32_t& count) {
        if constexpr (BATCH_PINNED) {
            batch = pinned_batch;
            if (task < full_waves) {
                first = (task * cores_per_batch + cid) * points_per_tile;
                count = points_per_tile;
                return true;
            }
            first = full_waves * wave_cap + cid * tail_tile;
            if (first >= c.m) return false;
            count = static_cast<uint32_t>(c.m - first < tail_tile ? c.m - first : tail_tile);
            return true;
        }
        batch = (task * inv_d + task) >> 32;
        first = (task - batch * per_batch) * points_per_tile;
        count = static_cast<uint32_t>(c.m - first < points_per_tile ? c.m - first : points_per_tile);
        return true;
    }

    ELEC_FN void load_task_inputs(
        const Config& c, const Buffers& b,
        const AscendC::GlobalTensor<T>& input_points,
        const AscendC::GlobalTensor<T>& input_segments,
        uint64_t task, uint64_t task_start, uint64_t batch, uint64_t& cached_batch,
        uint64_t first, uint32_t count, uint32_t n_pow2, uint32_t n_aligned, uint8_t n_rep,
        bool& offsets_ready) {
        const AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        const AscendC::DataCopyExtParams load_points{1, count * 3 * uint32_t(sizeof(T)), 0, 0, 0};
        if (batch != cached_batch) {
            if (offsets_ready) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                tree_fence<AscendC::HardEvent::V_MTE2>();
            }
            const AscendC::DataCopyExtParams load{1, static_cast<uint32_t>(c.n * 6 * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(b.seg_raw, input_segments[batch * c.n * 6], load, padding);
            AscendC::DataCopyPad(b.point_raw, input_points[(batch * c.m + first) * 3], load_points, padding);
            if (!offsets_ready) {
                init_offsets(b.offsets, n_pow2);
                offsets_ready = true;
            }
            tree_fence<AscendC::HardEvent::MTE2_V>();
            prepare_segments(c, b, n_aligned, n_rep);
            cached_batch = batch;
        } else {
            if (task != task_start) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::DataCopyPad(b.point_raw, input_points[(batch * c.m + first) * 3], load_points, padding);
            tree_fence<AscendC::HardEvent::MTE2_V>();
        }
        tree_gather_points_3d<T>(b.point, b.point_raw, b.point_f32, b.offsets[1024], count);
    }

    struct Schedule {
        uint32_t inv_d;
        uint64_t tail_tile, full_waves, cores_per_batch, per_batch, pinned_batch, cid, wave_cap, task_start, task_limit, task_step;
        ELEC_FN Schedule(const Config& c) {
            inv_d = static_cast<uint32_t>(c.k);
            const uint64_t d_val = BATCH_PINNED ? ((c.k >> 32) & 0xffu) : (c.k >> 32);
            tail_tile = BATCH_PINNED ? ((c.k >> 40) & 0xffu) : points_per_tile;
            full_waves = BATCH_PINNED ? (c.k >> 48) : 0;
            const uint64_t block_idx = AscendC::GetBlockIdx();
            cores_per_batch = BATCH_PINNED ? d_val : 1;
            per_batch = BATCH_PINNED ? 1 : d_val;
            pinned_batch = BATCH_PINNED ? ((block_idx * inv_d + block_idx) >> 32) : 0;
            cid = BATCH_PINNED ? (block_idx - pinned_batch * cores_per_batch) : 0;
            wave_cap = cores_per_batch * points_per_tile;
            task_start = BATCH_PINNED ? 0 : block_idx;
            task_limit = BATCH_PINNED ? (full_waves + (tail_tile > 0 ? 1 : 0)) : (c.batch * per_batch);
            task_step = BATCH_PINNED ? 1 : AscendC::GetBlockNum();
        }
    };

    ELEC_FN void run(const Config& c, GM_ADDR points_address, GM_ADDR segments_address,
                     GM_ADDR distance_address, GM_ADDR index_address) {
        const Schedule s(c);
        if (!BATCH_PINNED && s.task_start >= s.task_limit) return;
        Buffers b;
        auto distance = b.result[384];
        auto best_index = b.result[320].template ReinterpretCast<int32_t>();
        auto packed = b.result[448].template ReinterpretCast<T>();

        AscendC::GlobalTensor<T> input_points, input_segments, output;
        AscendC::GlobalTensor<int32_t> output_index;
        input_points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
        input_segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
        output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));

        uint64_t cached_batch = c.batch;
        bool offsets_ready = false;
        const uint32_t n_pow2 = HAS_TAIL1 ? static_cast<uint32_t>(c.n - 1) : static_cast<uint32_t>(c.n);
        const uint32_t n_aligned = (static_cast<uint32_t>(c.n) + 63u) & ~63u;
        const uint8_t n_rep = static_cast<uint8_t>(n_aligned >> 6);

        for (uint64_t task = s.task_start; task < s.task_limit; task += s.task_step) {
            uint64_t batch = 0, first = 0;
            uint32_t count = 0;
            if (!resolve_task(c, task, s.inv_d, s.pinned_batch, s.cid, s.cores_per_batch, s.per_batch, s.full_waves, s.tail_tile, s.wave_cap, batch, first, count)) {
                break;
            }
            load_task_inputs(c, b, input_points, input_segments, task, s.task_start, batch, cached_batch, first, count, n_pow2, n_aligned, n_rep, offsets_ready);
            const bool has_next = (task + s.task_step < s.task_limit) && (!BATCH_PINNED || (task + 1 < s.full_waves) || (s.full_waves * s.wave_cap + s.cid * s.tail_tile < c.m));
            if (has_next) AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            eval_chunks(c, n_pow2, count, b);
            if (task != s.task_start) tree_fence<AscendC::HardEvent::MTE3_V>();
            merge_chunk_winners(c, b, best_index);
            tree_write_outputs<T>(output, output_index, batch * c.m + first, b.dx, distance, packed, best_index, count);
        }
    }
};

}  // namespace elec
