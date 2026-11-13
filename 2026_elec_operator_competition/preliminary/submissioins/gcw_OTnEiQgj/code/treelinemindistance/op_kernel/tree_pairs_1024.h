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

template <typename T, uint32_t points_per_tile = 64>
class TreePairs1024 {
public:
    static constexpr uint32_t point_stride = 512, seg_stride = 1024, seg_planes = 9;
    static constexpr uint32_t offset_count = 1024 + 2 * points_per_tile;
    static constexpr uint32_t seg_at = 256, red_f16_at = seg_at + 7 * seg_stride * 4, soa_f16_at = 37120;
    static constexpr uint32_t raw_at = 55808, point_at = 61952, pt_f16_at = 62720, pt_raw_at = 63232, pt_f32_at = 64000, ramp_at = 65280;
    static constexpr uint32_t dx_at = 65536, dy_at = 81920, dz_at = 98304, p0_at = 114688;
    static constexpr uint32_t r0_at = 131328, p1_at = 147712, q0_at = 164096;
    static constexpr uint32_t offset_at = 180480, result_at = 185088, mask1k_at = 187136;
    static constexpr uint32_t brc_c1_at = 193024, c1_norm_at = 195072, inf_at = 195328;

    struct Buffers1024 {
        AscendC::LocalTensor<float> soa{AscendC::TPosition::VECCALC, seg_at, seg_planes * seg_stride};
        AscendC::LocalTensor<half> soa_f16{AscendC::TPosition::VECCALC, soa_f16_at, seg_planes * seg_stride};
        AscendC::LocalTensor<T> raw{AscendC::TPosition::VECCALC, raw_at, 3 * point_stride * (sizeof(float) / sizeof(T))};
        AscendC::LocalTensor<float> point{AscendC::TPosition::VECCALC, point_at, 192};
        AscendC::LocalTensor<half> point_f16{AscendC::TPosition::VECCALC, pt_f16_at, 192};
        AscendC::LocalTensor<T> point_raw{AscendC::TPosition::VECCALC, pt_raw_at, 192};
        AscendC::LocalTensor<float> point_f32{AscendC::TPosition::VECCALC, pt_f32_at, 192};
        AscendC::LocalTensor<uint8_t> mask{AscendC::TPosition::VECCALC, mask1k_at, 1024};
        AscendC::LocalTensor<half> ramp_f16{AscendC::TPosition::VECCALC, ramp_at, 128};
        AscendC::LocalTensor<int32_t> offsets{AscendC::TPosition::VECCALC, offset_at, offset_count};
        AscendC::LocalTensor<half> red_f16{AscendC::TPosition::VECCALC, red_f16_at, 3072};
        AscendC::LocalTensor<float> result{AscendC::TPosition::VECCALC, result_at, 512};
        AscendC::LocalTensor<half> c1_norm{AscendC::TPosition::VECCALC, c1_norm_at, 64};
        AscendC::LocalTensor<half> brc_c1{AscendC::TPosition::VECCALC, brc_c1_at, 1024};
        AscendC::LocalTensor<half> inf_f16{AscendC::TPosition::VECCALC, inf_at, 128};
        AscendC::LocalTensor<float> dx{AscendC::TPosition::VECCALC, dx_at, 4096};
        AscendC::LocalTensor<float> dy{AscendC::TPosition::VECCALC, dy_at, 4096};
        AscendC::LocalTensor<float> dz{AscendC::TPosition::VECCALC, dz_at, 4096};
        AscendC::LocalTensor<float> p0{AscendC::TPosition::VECCALC, p0_at, 4096};
        AscendC::LocalTensor<float> r0{AscendC::TPosition::VECCALC, r0_at, 4096};
        AscendC::LocalTensor<float> p1{AscendC::TPosition::VECCALC, p1_at, 4096};
        AscendC::LocalTensor<float> q0{AscendC::TPosition::VECCALC, q0_at, 4096};
        AscendC::LocalTensor<half> dx_f16{AscendC::TPosition::VECCALC, dx_at, 8192};
        AscendC::LocalTensor<half> dy_f16{AscendC::TPosition::VECCALC, dy_at, 8192};
        AscendC::LocalTensor<half> dz_f16{AscendC::TPosition::VECCALC, dz_at, 8192};
        AscendC::LocalTensor<half> p0_f16{AscendC::TPosition::VECCALC, p0_at, 8192};
        AscendC::LocalTensor<half> r0_f16{AscendC::TPosition::VECCALC, r0_at, 8192};
        AscendC::LocalTensor<half> p1_f16{AscendC::TPosition::VECCALC, p1_at, 8192};
        AscendC::LocalTensor<half> q0_f16{AscendC::TPosition::VECCALC, q0_at, 8192};
        AscendC::LocalTensor<T> seg_raw{AscendC::TPosition::VECCALC, 131072, 6144};
        AscendC::LocalTensor<float> segments{AscendC::TPosition::VECCALC, dx_at, 8192};
    };

    ELEC_FN void init_1024_tables(const Buffers1024& b) {
        auto ramp_i16 = b.ramp_f16.template ReinterpretCast<int16_t>();
        for (uint32_t i = 0; i < 8; ++i) {
            b.offsets.SetValue(i, static_cast<int32_t>(i) * 24);
            b.offsets.SetValue(1024 + i, static_cast<int32_t>(i) * 12);
        }
        for (uint32_t i = 0; i < 16; ++i) {
            ramp_i16.SetValue(i, static_cast<int16_t>(0x3800 + i));
        }
        auto mask_addr_ub = b.result.template ReinterpretCast<uint32_t>();
        const uint32_t mask_sel_addr = static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(b.mask.GetPhyAddr())));
        tree_fence<AscendC::HardEvent::S_V>();
        AscendC::Duplicate(mask_addr_ub, mask_sel_addr, 8);
        AscendC::Duplicate(b.inf_f16.template ReinterpretCast<uint16_t>(), uint16_t(0x7bffu), 128);
        for (uint32_t filled = 8; filled < 1024; filled *= 2) {
            AscendC::Adds(b.offsets[filled], b.offsets, static_cast<int32_t>(filled) * 24, filled);
            if (filled < points_per_tile) {
                AscendC::Adds(b.offsets[1024 + filled], b.offsets[1024], static_cast<int32_t>(filled) * 12, filled);
            }
            if (filled >= 16 && filled < 128) {
                AscendC::Adds(ramp_i16[filled], ramp_i16, static_cast<int16_t>(filled), filled);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    ELEC_FN float compute_coord_scale(const Buffers1024& b) {
        auto ax = b.soa[0 * seg_stride], ux = b.soa[3 * seg_stride];
        AscendC::Abs(b.dx, ax, 3072);
        AscendC::Abs(b.dy, ux, 3072);
        AscendC::Max(b.dx, b.dx, b.dy, 3072);
        AscendC::Max(b.dx, b.dx, b.dx[2048], 1024);
        AscendC::Max(b.dx, b.dx, b.dx[1024], 1024);
        AscendC::WholeReduceMax<float>(b.r0, b.dx, 64, 16, 1, 1, 8);
        AscendC::WholeReduceMax<float>(b.r0, b.r0, 32, 1, 1, 1, 8);
        AscendC::ResetMask();
        tree_fence<AscendC::HardEvent::V_S>();
        float max_coord = b.r0.GetValue(0);
        if (!(max_coord > 1e-6f)) max_coord = 1.0f;
        uint32_t exp_bits = float_bits(max_coord) & 0x7f800000u;
        exp_bits = (exp_bits < 0x1f800000u) ? 0x1f800000u : ((exp_bits > 0x5f800000u) ? 0x5f800000u : exp_bits);
        const float coord_scale = bits_float((254u - (exp_bits >> 23)) << 23);
        tree_fence<AscendC::HardEvent::S_V>();
        return coord_scale;
    }

    ELEC_FN float prepare_1024_segments(const Buffers1024& b) {
        auto ax = b.soa[0 * seg_stride], ux = b.soa[3 * seg_stride];
        auto vx = b.soa[6 * seg_stride], vy = b.soa[7 * seg_stride], vz = b.soa[8 * seg_stride];
        auto ax_f16 = b.soa_f16[0 * seg_stride], ux_f16 = b.soa_f16[3 * seg_stride], vx_f16 = b.soa_f16[6 * seg_stride];
        if constexpr (sizeof(T) == 2) {
            AscendC::Cast(b.segments, b.seg_raw, AscendC::RoundMode::CAST_NONE, 1024 * 6);
        }
        auto seg_src = (sizeof(T) == 2) ? b.segments : b.seg_raw.template ReinterpretCast<float>();
        for (uint32_t axis = 0; axis < 6; ++axis) {
            AscendC::Gather(b.soa[axis * seg_stride], seg_src[axis], b.offsets.template ReinterpretCast<uint32_t>(), 0, 1024);
        }
        AscendC::Sub(ux, ux, ax, 3072);
        AscendC::Mul(vx, ux, ux, 3072);
        AscendC::Add(vx, vx, vy, 1024);
        AscendC::Add(vx, vx, vz, 1024);
        AscendC::Maxs(vx, vx, bits_float(0x00800000u), 1024);
        AscendC::Duplicate(b.red_f16.template ReinterpretCast<int16_t>(), int16_t(0), 3072);
        AscendC::Duplicate(b.point, 0.0f, 192);
        AscendC::Duplicate(b.c1_norm.template ReinterpretCast<int16_t>(), int16_t(0x3800), 64);

        const float coord_scale = compute_coord_scale(b);
        AscendC::Muls(b.dx, ax, coord_scale, 3072);
        AscendC::Cast(ax_f16, b.dx, AscendC::RoundMode::CAST_RINT, 3072);
        AscendC::Muls(b.dx, ux, coord_scale, 3072);
        AscendC::Cast(ux_f16, b.dx, AscendC::RoundMode::CAST_RINT, 3072);
        AscendC::Mul(b.r0, b.dx, b.dx, 1024);
        AscendC::Mul(b.p1, b.dx[1024], b.dx[1024], 1024);
        AscendC::Add(b.r0, b.r0, b.p1, 1024);
        AscendC::Mul(b.p1, b.dx[2048], b.dx[2048], 1024);
        AscendC::Add(b.r0, b.r0, b.p1, 1024);
        AscendC::Maxs(b.r0, b.r0, bits_float(0x00800000u), 1024);
        AscendC::Div(b.dx, b.dx, b.r0, 1024);
        AscendC::Div(b.dx[1024], b.dx[1024], b.r0, 1024);
        AscendC::Div(b.dx[2048], b.dx[2048], b.r0, 1024);
        AscendC::Maxs(b.dx, b.dx, -65504.0f, 3072);
        AscendC::Mins(b.dx, b.dx, 65504.0f, 3072);
        AscendC::Cast(vx_f16, b.dx, AscendC::RoundMode::CAST_RINT, 3072);
        return coord_scale;
    }

    ELEC_FN void screen_1024_fp16(uint32_t count, const Buffers1024& b) {
        auto brc_f16 = b.raw.template ReinterpretCast<half>();
        auto mask_addr_ub = b.result.template ReinterpretCast<uint32_t>();
        auto ax_f16 = b.soa_f16[0 * seg_stride], ay_f16 = b.soa_f16[1 * seg_stride], az_f16 = b.soa_f16[2 * seg_stride];
        auto ux_f16 = b.soa_f16[3 * seg_stride], uy_f16 = b.soa_f16[4 * seg_stride], uz_f16 = b.soa_f16[5 * seg_stride];
        auto vx_f16 = b.soa_f16[6 * seg_stride], vy_f16 = b.soa_f16[7 * seg_stride], vz_f16 = b.soa_f16[8 * seg_stride];
        const AscendC::BinaryRepeatParams point_segment{1, 0, 1, 8, 1, 0}, segment{1, 1, 1, 8, 8, 0}, contig{1, 1, 1, 8, 8, 8};
        const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8};
        const AscendC::GatherMaskParams gm2{1, 2, 8, 0};
        const uint8_t r_cnt = static_cast<uint8_t>(count);
        uint64_t rsvd_cnt = 0;
        AscendC::AscendCUtils::SetMask<half>(128);
        for (uint32_t c_idx = 0; c_idx < 8; ++c_idx) {
            const uint32_t start = c_idx * 128;
            AscendC::Sub<half, false>(b.dx_f16, brc_f16, ax_f16[start], uint64_t(128), r_cnt, point_segment);
            AscendC::Sub<half, false>(b.dy_f16, brc_f16[1024], ay_f16[start], uint64_t(128), r_cnt, point_segment);
            AscendC::Sub<half, false>(b.dz_f16, brc_f16[2048], az_f16[start], uint64_t(128), r_cnt, point_segment);
            AscendC::Mul<half, false>(b.p0_f16, b.dx_f16, vx_f16[start], uint64_t(128), r_cnt, segment);
            AscendC::Mul<half, false>(b.r0_f16, b.dy_f16, vy_f16[start], uint64_t(128), r_cnt, segment);
            AscendC::Mul<half, false>(b.p1_f16, b.dz_f16, vz_f16[start], uint64_t(128), r_cnt, segment);
            AscendC::Add<half, false>(b.p0_f16, b.p0_f16, b.r0_f16, uint64_t(128), r_cnt, contig);
            AscendC::Add<half, false>(b.p0_f16, b.p0_f16, b.p1_f16, uint64_t(128), r_cnt, contig);
            AscendC::Maxs<half, false>(b.p0_f16, b.p0_f16, static_cast<half>(0.0f), uint64_t(128), r_cnt, ucontig);
            AscendC::Mins<half, false>(b.p0_f16, b.p0_f16, static_cast<half>(1.0f), uint64_t(128), r_cnt, ucontig);
            AscendC::Mul<half, false>(b.r0_f16, b.p0_f16, ux_f16[start], uint64_t(128), r_cnt, segment);
            AscendC::Mul<half, false>(b.p1_f16, b.p0_f16, uy_f16[start], uint64_t(128), r_cnt, segment);
            AscendC::Mul<half, false>(b.q0_f16, b.p0_f16, uz_f16[start], uint64_t(128), r_cnt, segment);
            AscendC::Sub<half, false>(b.dx_f16, b.dx_f16, b.r0_f16, uint64_t(128), r_cnt, contig);
            AscendC::Sub<half, false>(b.dy_f16, b.dy_f16, b.p1_f16, uint64_t(128), r_cnt, contig);
            AscendC::Sub<half, false>(b.dz_f16, b.dz_f16, b.q0_f16, uint64_t(128), r_cnt, contig);
            AscendC::Mul<half, false>(b.p0_f16, b.dx_f16, b.dx_f16, uint64_t(128), r_cnt, contig);
            AscendC::Mul<half, false>(b.r0_f16, b.dy_f16, b.dy_f16, uint64_t(128), r_cnt, contig);
            AscendC::Mul<half, false>(b.p1_f16, b.dz_f16, b.dz_f16, uint64_t(128), r_cnt, contig);
            AscendC::Add<half, false>(b.p0_f16, b.p0_f16, b.r0_f16, uint64_t(128), r_cnt, contig);
            AscendC::Add<half, false>(b.p0_f16, b.p0_f16, b.p1_f16, uint64_t(128), r_cnt, contig);
            AscendC::WholeReduceMin<half, false>(b.red_f16[(c_idx * 2) * 128], b.p0_f16, int32_t(128), static_cast<int32_t>(count), 1, 1, 8);
            AscendC::GatherMask(b.c1_norm.template ReinterpretCast<uint16_t>(), b.red_f16[(c_idx * 2) * 128].template ReinterpretCast<uint16_t>(), static_cast<uint8_t>(2), false, uint32_t(0), gm2, rsvd_cnt);
            AscendC::Adds<int16_t, true>(b.c1_norm.template ReinterpretCast<int16_t>(), b.c1_norm.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0x3800), uint64_t(128), 1, ucontig);
            AscendC::Brcb(b.brc_c1, b.c1_norm, 8, {1, 8});
            AscendC::Compare<half, uint8_t, false>(b.mask, b.brc_c1, b.ramp_f16, AscendC::CMPMODE::NE, uint64_t(128), r_cnt, point_segment);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::VselIntrinsicsImplPre(reinterpret_cast<__ubuf__ uint32_t*>(mask_addr_ub.GetPhyAddr()), AscendC::SELMODE::VSEL_CMPMASK_SPR);
            AscendC::SelectCal<half, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE>(reinterpret_cast<__ubuf__ half*>(b.p0_f16.GetPhyAddr()), reinterpret_cast<__ubuf__ half*>(b.p0_f16.GetPhyAddr()), reinterpret_cast<__ubuf__ half*>(b.inf_f16.GetPhyAddr()), static_cast<int32_t>(count), segment);
            AscendC::WholeReduceMin<half, false>(b.red_f16[(c_idx * 2 + 1) * 128], b.p0_f16, int32_t(128), static_cast<int32_t>(count), 1, 1, 8);
        }
    }

    ELEC_FN void sort_1024_candidates(const Buffers1024& b) {
        const AscendC::BinaryRepeatParams contig{1, 1, 1, 8, 8, 8};
        const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8}, ustep16{1, 1, 16, 16};
        const AscendC::GatherMaskParams gm16{1, 16, 8, 0};
        auto p0_i16 = b.p0_f16.template ReinterpretCast<int16_t>();
        auto r0_i16 = b.r0_f16.template ReinterpretCast<int16_t>();
        auto p1_i32 = b.p1.template ReinterpretCast<int32_t>();
        uint64_t rsvd_cnt = 0;
        AscendC::GatherMask(b.p0_f16.template ReinterpretCast<uint16_t>(), b.red_f16.template ReinterpretCast<uint16_t>(), static_cast<uint8_t>(2), false, uint32_t(0), gm16, rsvd_cnt);
        AscendC::AscendCUtils::SetMask<int16_t>(64);
        AscendC::Min<int16_t, false>(r0_i16, p0_i16, p0_i16[64], uint64_t(64), 8, contig);
        AscendC::Max<int16_t, false>(r0_i16[64], p0_i16, p0_i16[64], uint64_t(64), 8, contig);
        AscendC::AscendCUtils::SetMask<int16_t>(128);
        AscendC::Adds<int16_t, false>(r0_i16, r0_i16, static_cast<int16_t>(0x3800), uint64_t(128), 8, ucontig);
        AscendC::Adds<int16_t, false>(r0_i16[128], r0_i16[128], static_cast<int16_t>(128), uint64_t(128), 4, ustep16);
        AscendC::Adds<int16_t, false>(r0_i16[256], r0_i16[256], static_cast<int16_t>(256), uint64_t(128), 2, ucontig);
        AscendC::Adds<int16_t, false>(r0_i16[768], r0_i16[768], static_cast<int16_t>(256), uint64_t(128), 2, ucontig);
        AscendC::Adds<int16_t, false>(r0_i16[512], r0_i16[512], static_cast<int16_t>(512), uint64_t(128), 4, ucontig);
        AscendC::Cast(b.q0, b.r0_f16, AscendC::RoundMode::CAST_NONE, 1024);
        AscendC::Adds(b.q0, b.q0, -0.5f, 1024);
        AscendC::Muls(b.q0, b.q0, 8192.0f, 1024);
        AscendC::Cast(p1_i32, b.q0, AscendC::RoundMode::CAST_RINT, 1024);
    }

    ELEC_FN void eval_16_candidates_fp32(const Buffers1024& b) {
        auto ax = b.soa[0 * seg_stride], ay = b.soa[1 * seg_stride], az = b.soa[2 * seg_stride];
        auto ux = b.soa[3 * seg_stride], uy = b.soa[4 * seg_stride], uz = b.soa[5 * seg_stride], vx = b.soa[6 * seg_stride];
        auto cand_off = b.p1.template ReinterpretCast<uint32_t>();
        const AscendC::BinaryRepeatParams contig{1, 1, 1, 8, 8, 8}, pt_rows{1, 1, 1, 8, 0, 8};
        const AscendC::UnaryRepeatParams ucontig{1, 1, 8, 8};
        AscendC::Gather(b.dx, ax, cand_off, 0, 1024);
        AscendC::Gather(b.dy, ay, cand_off, 0, 1024);
        AscendC::Gather(b.dz, az, cand_off, 0, 1024);
        AscendC::Sub<float, true>(b.dx, b.point, b.dx, uint64_t(64), 16, pt_rows);
        AscendC::Sub<float, false>(b.dy, b.point[64], b.dy, uint64_t(64), 16, pt_rows);
        AscendC::Sub<float, false>(b.dz, b.point[128], b.dz, uint64_t(64), 16, pt_rows);
        AscendC::Gather(b.r0, ux, cand_off, 0, 1024);
        AscendC::Gather(b.q0, uy, cand_off, 0, 1024);
        AscendC::Gather(b.r0[1024], uz, cand_off, 0, 1024);
        AscendC::Gather(b.q0[1024], vx, cand_off, 0, 1024);
        AscendC::Mul<float, true>(b.p0, b.dx, b.r0, uint64_t(64), 16, contig);
        AscendC::Mul<float, false>(b.r0[2048], b.dy, b.q0, uint64_t(64), 16, contig);
        AscendC::Mul<float, false>(b.q0[2048], b.dz, b.r0[1024], uint64_t(64), 16, contig);
        AscendC::Add<float, false>(b.p0, b.p0, b.r0[2048], uint64_t(64), 16, contig);
        AscendC::Add<float, false>(b.p0, b.p0, b.q0[2048], uint64_t(64), 16, contig);
        AscendC::Div<float, false>(b.p0, b.p0, b.q0[1024], uint64_t(64), 16, contig);
        AscendC::Maxs<float, false>(b.p0, b.p0, 0.0f, uint64_t(64), 16, ucontig);
        AscendC::Mins<float, false>(b.p0, b.p0, 1.0f, uint64_t(64), 16, ucontig);
        AscendC::Mul<float, false>(b.r0, b.p0, b.r0, uint64_t(64), 16, contig);
        AscendC::Mul<float, false>(b.q0, b.p0, b.q0, uint64_t(64), 16, contig);
        AscendC::Mul<float, false>(b.r0[1024], b.p0, b.r0[1024], uint64_t(64), 16, contig);
        AscendC::Sub<float, false>(b.dx, b.dx, b.r0, uint64_t(64), 16, contig);
        AscendC::Sub<float, false>(b.dy, b.dy, b.q0, uint64_t(64), 16, contig);
        AscendC::Sub<float, false>(b.dz, b.dz, b.r0[1024], uint64_t(64), 16, contig);
        AscendC::Mul<float, false>(b.dx, b.dx, b.dx, uint64_t(64), 16, contig);
        AscendC::Mul<float, false>(b.r0, b.dy, b.dy, uint64_t(64), 16, contig);
        AscendC::Mul<float, false>(b.q0, b.dz, b.dz, uint64_t(64), 16, contig);
        AscendC::Add<float, false>(b.dx, b.dx, b.r0, uint64_t(64), 16, contig);
        AscendC::Add<float, false>(b.dx, b.dx, b.q0, uint64_t(64), 16, contig);
    }

    ELEC_FN void reduce_16_candidates(uint64_t task, uint32_t count, const Buffers1024& b, const AscendC::LocalTensor<int32_t>& best_index) {
        const AscendC::BinaryRepeatParams contig{1, 1, 1, 8, 8, 8};
        if (task != 0) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        auto best_idx_f32 = best_index.template ReinterpretCast<float>();
        for (uint32_t c_idx = 1; c_idx < 16; ++c_idx) {
            AscendC::Compare<float, false>(b.dx[c_idx * 64], b.dx, AscendC::CMPMODE::LT, uint64_t(64), contig);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SelectCal<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(reinterpret_cast<__ubuf__ float*>(b.dx.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(b.dx[c_idx * 64].GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(b.dx.GetPhyAddr()), 1, contig);
            auto dst_idx = (c_idx + 1 == 16) ? best_idx_f32 : b.p1;
            AscendC::SelectCal<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(reinterpret_cast<__ubuf__ float*>(dst_idx.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(b.p1[c_idx * 64].GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(b.p1.GetPhyAddr()), 1, contig);
        }
        AscendC::ShiftRight(best_index, best_index, 2, count);
    }

    ELEC_FN void prefetch_1024_points(
        const Config& c, uint64_t batch, uint64_t task,
        uint64_t full_waves, uint64_t cores_per_batch, uint64_t cid, uint64_t wave_cap, uint64_t tail_tile,
        const AscendC::GlobalTensor<T>& input_points,
        const AscendC::LocalTensor<T>& point_raw) {
        const AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        const uint64_t first = (task < full_waves) ? ((task * cores_per_batch + cid) * points_per_tile) : (full_waves * wave_cap + cid * tail_tile);
        if (first >= c.m) return;
        const uint32_t count = (task < full_waves) ? points_per_tile : static_cast<uint32_t>(c.m - first < tail_tile ? c.m - first : tail_tile);
        const AscendC::DataCopyExtParams load_pts{1, count * 3 * uint32_t(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(point_raw, input_points[(batch * c.m + first) * 3], load_pts, padding);
    }

    ELEC_FN void run(const Config& c, GM_ADDR points_address, GM_ADDR segments_address,
                     GM_ADDR distance_address, GM_ADDR index_address) {
        Buffers1024 b;
        AscendC::GlobalTensor<T> input_points, input_segments, output;
        AscendC::GlobalTensor<int32_t> output_index;
        input_points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
        input_segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
        output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));

        const uint32_t inv_d = static_cast<uint32_t>(c.k);
        const uint64_t cores_per_batch = (c.k >> 32) & 0xffu, tail_tile = (c.k >> 40) & 0xffu, full_waves = c.k >> 48;
        const uint64_t block_idx = AscendC::GetBlockIdx();
        const uint64_t batch = (block_idx * inv_d + block_idx) >> 32, cid = block_idx - batch * cores_per_batch;
        const uint64_t wave_cap = cores_per_batch * points_per_tile;
        const uint64_t task_limit = full_waves + ((c.m > full_waves * wave_cap) ? 1 : 0);

        const AscendC::DataCopyExtParams load{1, static_cast<uint32_t>(1024 * 6 * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(b.seg_raw, input_segments[batch * 1024 * 6], load, {false, 0, 0, 0});
        prefetch_1024_points(c, batch, 0, full_waves, cores_per_batch, cid, wave_cap, tail_tile, input_points, b.point_raw);
        init_1024_tables(b);
        tree_fence<AscendC::HardEvent::MTE2_V>();
        const float coord_scale = prepare_1024_segments(b);
        for (uint64_t task = 0; task < task_limit; ++task) {
            const uint64_t first = (task < full_waves) ? ((task * cores_per_batch + cid) * points_per_tile) : (full_waves * wave_cap + cid * tail_tile);
            if (first >= c.m) break;
            const uint32_t count = (task < full_waves) ? points_per_tile : static_cast<uint32_t>(c.m - first < tail_tile ? c.m - first : tail_tile);
            if (task != 0) AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tree_gather_points_3d<T>(b.point, b.point_raw, b.point_f32, b.offsets[1024], count);
            const bool has_next = (task + 1 < task_limit) && ((task + 1 < full_waves) || (full_waves * wave_cap + cid * tail_tile < c.m));
            if (has_next) {
                tree_fence<AscendC::HardEvent::V_MTE2>();
                prefetch_1024_points(c, batch, task + 1, full_waves, cores_per_batch, cid, wave_cap, tail_tile, input_points, b.point_raw);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            }
            AscendC::Muls(b.point_f32, b.point, coord_scale, 192);
            AscendC::Cast(b.point_f16, b.point_f32, AscendC::RoundMode::CAST_RINT, 192);
            AscendC::Brcb(b.raw.template ReinterpretCast<half>(), b.point_f16, 24, {1, 8});
            screen_1024_fp16(count, b);
            sort_1024_candidates(b);
            eval_16_candidates_fp32(b);
            reduce_16_candidates(task, count, b, b.result[320].template ReinterpretCast<int32_t>());
            tree_write_outputs<T>(output, output_index, batch * c.m + first, b.dx, b.result[384], b.result[448].template ReinterpretCast<T>(), b.result[320].template ReinterpretCast<int32_t>(), count);
            if (has_next) AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }
};

}  // namespace elec
