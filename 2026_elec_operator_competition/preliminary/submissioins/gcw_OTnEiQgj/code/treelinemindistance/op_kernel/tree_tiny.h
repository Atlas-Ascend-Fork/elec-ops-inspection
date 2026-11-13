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

union Half4 {
    uint64_t u64;
    half h[4];
};

union Half2 {
    uint32_t u32;
    half h[2];
};

union Float2 {
    uint64_t u64;
    float f[2];
};

struct TinySeg {
    float ax, ay, az;
    float ux, uy, uz;
    float len;
};

ELEC_FN float point_seg_dist2_sc(
    float px, float py, float pz,
    float ax, float ay, float az,
    float ux, float uy, float uz,
    float length) {
    const float dx = px - ax;
    const float dy = py - ay;
    const float dz = pz - az;
    const float dot = dx * ux + dy * uy + dz * uz;
    if (dot <= 0.0f || length <= 0.0f) {
        return dx * dx + dy * dy + dz * dz;
    }
    if (dot >= length) {
        const float ex = dx - ux;
        const float ey = dy - uy;
        const float ez = dz - uz;
        return ex * ex + ey * ey + ez * ez;
    }
    const float t = (length > 0.0f) ? (dot / length) : 0.0f;
    const float ex = dx - t * ux;
    const float ey = dy - t * uy;
    const float ez = dz - t * uz;
    return ex * ex + ey * ey + ez * ez;
}

template <typename T>
ELEC_FN TinySeg load_tiny_seg(GM_ADDR segments_addr, uint32_t s) {
    TinySeg seg{};
    if constexpr (sizeof(T) == 4) {
        const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
        Float2 s01{s64[s]}, s23{s64[s + 1u]}, s45{s64[s + 2u]};
        seg.ax = s01.f[0];
        seg.ay = s01.f[1];
        seg.az = s23.f[0];
        seg.ux = s23.f[1] - seg.ax;
        seg.uy = s45.f[0] - seg.ay;
        seg.uz = s45.f[1] - seg.az;
    } else {
        const __gm__ uint32_t* s32 = reinterpret_cast<const __gm__ uint32_t*>(segments_addr);
        Half2 s01{s32[s]}, s23{s32[s + 1u]}, s45{s32[s + 2u]};
        seg.ax = static_cast<float>(s01.h[0]);
        seg.ay = static_cast<float>(s01.h[1]);
        seg.az = static_cast<float>(s23.h[0]);
        seg.ux = static_cast<float>(s23.h[1]) - seg.ax;
        seg.uy = static_cast<float>(s45.h[0]) - seg.ay;
        seg.uz = static_cast<float>(s45.h[1]) - seg.az;
    }
    seg.len = seg.ux * seg.ux + seg.uy * seg.uy + seg.uz * seg.uz;
    return seg;
}

template <typename T>
ELEC_FN void load_tiny_pt0(GM_ADDR points_addr, const __gm__ T* pts, float& px, float& py, float& pz) {
    if constexpr (sizeof(T) == 4) {
        const __gm__ uint64_t* p64 = reinterpret_cast<const __gm__ uint64_t*>(points_addr);
        Float2 p01{p64[0]};
        px = p01.f[0];
        py = p01.f[1];
        pz = pts[2];
    } else {
        const __gm__ uint32_t* p32 = reinterpret_cast<const __gm__ uint32_t*>(points_addr);
        Half2 p01{p32[0]};
        px = static_cast<float>(p01.h[0]);
        py = static_cast<float>(p01.h[1]);
        pz = static_cast<float>(pts[2]);
    }
}

template <typename T>
ELEC_FN void tree_tiny_1seg(
    GM_ADDR points_addr, GM_ADDR segments_addr, const __gm__ T* pts,
    uint32_t count, const AscendC::LocalTensor<float>& distance, int32_t* best_idx) {
    const TinySeg seg0 = load_tiny_seg<T>(segments_addr, 0u);
    if (count == 1u) {
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        load_tiny_pt0<T>(points_addr, pts, px, py, pz);
        distance.SetValue(0, point_seg_dist2_sc(px, py, pz, seg0.ax, seg0.ay, seg0.az, seg0.ux, seg0.uy, seg0.uz, seg0.len));
        best_idx[0] = 0;
        return;
    }
    #pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t p = i * 3u;
        const float px = static_cast<float>(pts[p]);
        const float py = static_cast<float>(pts[p + 1u]);
        const float pz = static_cast<float>(pts[p + 2u]);
        distance.SetValue(i, point_seg_dist2_sc(px, py, pz, seg0.ax, seg0.ay, seg0.az, seg0.ux, seg0.uy, seg0.uz, seg0.len));
        best_idx[i] = 0;
    }
}

template <typename T>
ELEC_FN void tree_tiny_2seg(
    GM_ADDR points_addr, GM_ADDR segments_addr, const __gm__ T* pts,
    uint32_t count, const AscendC::LocalTensor<float>& distance, int32_t* best_idx) {
    TinySeg seg0{}, seg1{};
    if constexpr (sizeof(T) == 2) {
        const __gm__ uint64_t* s64 = reinterpret_cast<const __gm__ uint64_t*>(segments_addr);
        Half4 s0{s64[0]}, s1{s64[1]}, s2{s64[2]};
        seg0.ax = static_cast<float>(s0.h[0]); seg0.ay = static_cast<float>(s0.h[1]); seg0.az = static_cast<float>(s0.h[2]);
        seg0.ux = static_cast<float>(s0.h[3]) - seg0.ax; seg0.uy = static_cast<float>(s1.h[0]) - seg0.ay; seg0.uz = static_cast<float>(s1.h[1]) - seg0.az;
        seg1.ax = static_cast<float>(s1.h[2]); seg1.ay = static_cast<float>(s1.h[3]); seg1.az = static_cast<float>(s2.h[0]);
        seg1.ux = static_cast<float>(s2.h[1]) - seg1.ax; seg1.uy = static_cast<float>(s2.h[2]) - seg1.ay; seg1.uz = static_cast<float>(s2.h[3]) - seg1.az;
        seg0.len = seg0.ux * seg0.ux + seg0.uy * seg0.uy + seg0.uz * seg0.uz;
        seg1.len = seg1.ux * seg1.ux + seg1.uy * seg1.uy + seg1.uz * seg1.uz;
    } else {
        seg0 = load_tiny_seg<T>(segments_addr, 0u);
        seg1 = load_tiny_seg<T>(segments_addr, 3u);
    }
    if (count == 1u) {
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        load_tiny_pt0<T>(points_addr, pts, px, py, pz);
        const float d0 = point_seg_dist2_sc(px, py, pz, seg0.ax, seg0.ay, seg0.az, seg0.ux, seg0.uy, seg0.uz, seg0.len);
        const float d1 = point_seg_dist2_sc(px, py, pz, seg1.ax, seg1.ay, seg1.az, seg1.ux, seg1.uy, seg1.uz, seg1.len);
        distance.SetValue(0, d1 < d0 ? d1 : d0);
        best_idx[0] = d1 < d0 ? 1 : 0;
        return;
    }
    #pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t p = i * 3u;
        const float px = static_cast<float>(pts[p]);
        const float py = static_cast<float>(pts[p + 1u]);
        const float pz = static_cast<float>(pts[p + 2u]);
        const float d0 = point_seg_dist2_sc(px, py, pz, seg0.ax, seg0.ay, seg0.az, seg0.ux, seg0.uy, seg0.uz, seg0.len);
        const float d1 = point_seg_dist2_sc(px, py, pz, seg1.ax, seg1.ay, seg1.az, seg1.ux, seg1.uy, seg1.uz, seg1.len);
        distance.SetValue(i, d1 < d0 ? d1 : d0);
        best_idx[i] = d1 < d0 ? 1 : 0;
    }
}

template <typename T, bool FAST_PT0>
ELEC_FN void tree_tiny_general(
    GM_ADDR points_addr, GM_ADDR segments_addr, const __gm__ T* pts,
    uint32_t batch, uint32_t m_per_batch, uint32_t segments_per_batch,
    const AscendC::LocalTensor<float>& distance, int32_t* best_idx) {
    const uint32_t count = batch * m_per_batch;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t b = (m_per_batch > 0u) ? (i / m_per_batch) : 0u;
        const uint32_t seg_base = b * segments_per_batch * 3u;
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        if (FAST_PT0 && i == 0u) {
            load_tiny_pt0<T>(points_addr, pts, px, py, pz);
        } else {
            const uint32_t p = i * 3u;
            px = static_cast<float>(pts[p]);
            py = static_cast<float>(pts[p + 1u]);
            pz = static_cast<float>(pts[p + 2u]);
        }
        float best = bits_float(0x7f800000u);
        int32_t best_index = 0;
        #pragma clang loop unroll(disable)
        for (uint32_t j = 0; j < segments_per_batch; ++j) {
            const TinySeg seg = load_tiny_seg<T>(segments_addr, seg_base + j * 3u);
            const float value = point_seg_dist2_sc(px, py, pz, seg.ax, seg.ay, seg.az, seg.ux, seg.uy, seg.uz, seg.len);
            if (value < best) {
                best = value;
                best_index = static_cast<int32_t>(j);
            }
        }
        distance.SetValue(i, best);
        best_idx[i] = best_index;
    }
}

template <typename T>
ELEC_FN void tree_tiny_epilogue(
    GM_ADDR min_distance_addr, GM_ADDR nearest_segment_idx_addr,
    const AscendC::LocalTensor<float>& distance,
    const AscendC::LocalTensor<int32_t>& index,
    const AscendC::LocalTensor<T>& distance_half,
    const int32_t* best_idx, uint32_t count) {
    AscendC::GlobalTensor<T> output;
    AscendC::GlobalTensor<int32_t> output_index;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(min_distance_addr));
    output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nearest_segment_idx_addr));
    const AscendC::DataCopyExtParams store_distance{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    const AscendC::DataCopyExtParams store_index{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    const uint32_t vec_len = (count <= 8u) ? 8u : 32u;

    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    AscendC::Sqrt(distance, distance, vec_len);
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(distance_half, distance, AscendC::RoundMode::CAST_RINT, vec_len);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);

    for (uint32_t i = 0; i < count; ++i) {
        index.SetValue(i, best_idx[i]);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::DataCopyPad(output_index, index, store_index);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    if constexpr (sizeof(T) == 2) {
        AscendC::DataCopyPad(output, distance_half, store_distance);
    } else {
        AscendC::DataCopyPad(output, distance, store_distance);
    }
}

template <typename T, bool MULTI_SEG = false>
ELEC_FN void tree_tiny(
    GM_ADDR tiling_addr,
    GM_ADDR points_addr,
    GM_ADDR segments_addr,
    GM_ADDR min_distance_addr,
    GM_ADDR nearest_segment_idx_addr) {
    if (AscendC::GetBlockIdx() != 0) return;

    const uint64_t pre_p0 = reinterpret_cast<const volatile __gm__ uint64_t*>(points_addr)[0];
    const uint64_t pre_s0 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[0];
    const uint64_t pre_s1 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[4];
    if constexpr (MULTI_SEG) {
        const uint64_t pre_s2 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[8];
        const uint64_t pre_s3 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[12];
        const uint64_t pre_s4 = reinterpret_cast<const volatile __gm__ uint64_t*>(segments_addr)[16];
        (void)pre_s2; (void)pre_s3; (void)pre_s4;
    }
    const uint64_t b_raw = reinterpret_cast<const volatile __gm__ uint64_t*>(tiling_addr)[0];
    const uint64_t m_raw = reinterpret_cast<const volatile __gm__ uint64_t*>(tiling_addr)[1];
    const uint64_t n_raw = reinterpret_cast<const volatile __gm__ uint64_t*>(tiling_addr)[2];
    (void)pre_p0; (void)pre_s0; (void)pre_s1;

    const __gm__ T* pts = reinterpret_cast<const __gm__ T*>(points_addr);
    AscendC::LocalTensor<float> distance = local_tensor_at<float>(0, 128, 32);
    AscendC::LocalTensor<int32_t> index = local_tensor_at<int32_t>(128, 128, 32);
    AscendC::LocalTensor<T> distance_half = local_tensor_at<T>(256, 128, 64);

    const uint32_t batch = static_cast<uint32_t>(b_raw);
    const uint32_t m_per_batch = static_cast<uint32_t>(m_raw);
    const uint32_t segments_per_batch = static_cast<uint32_t>(n_raw);
    const uint32_t count = batch * m_per_batch;
    int32_t best_idx[32];

    if constexpr (!MULTI_SEG) {
        if (batch == 1u && segments_per_batch == 1u) {
            tree_tiny_1seg<T>(points_addr, segments_addr, pts, count, distance, best_idx);
        } else if (batch == 1u && segments_per_batch == 2u) {
            tree_tiny_2seg<T>(points_addr, segments_addr, pts, count, distance, best_idx);
        } else {
            tree_tiny_general<T, false>(points_addr, segments_addr, pts, batch, m_per_batch, segments_per_batch, distance, best_idx);
        }
    } else {
        tree_tiny_general<T, true>(points_addr, segments_addr, pts, batch, m_per_batch, segments_per_batch, distance, best_idx);
    }
    tree_tiny_epilogue<T>(min_distance_addr, nearest_segment_idx_addr, distance, index, distance_half, best_idx, count);
}

}  // namespace elec
