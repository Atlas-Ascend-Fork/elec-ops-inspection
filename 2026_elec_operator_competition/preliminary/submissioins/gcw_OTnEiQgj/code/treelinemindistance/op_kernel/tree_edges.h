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

template <int32_t stride>
ELEC_FN void tree_edge_offsets(const AscendC::LocalTensor<int32_t>& dst, uint32_t count) {
    const uint32_t init_n = count < 32 ? count : 32;
    for (uint32_t i = 0; i < init_n; ++i) {
        dst.SetValue(i, static_cast<int32_t>(i) * stride);
    }
    tree_fence<AscendC::HardEvent::S_V>();
    for (uint32_t filled = 32; filled < count; filled *= 2) {
        const uint32_t size = (count - filled < filled) ? (count - filled) : filled;
        AscendC::Adds(dst[filled], dst, static_cast<int32_t>(filled) * stride, size);
    }
}

ELEC_FN void tree_one_segment_eval(
    const AscendC::LocalTensor<float>& work,
    uint32_t stride, uint32_t count,
    float ax, float ay, float az,
    float ux, float uy, float uz,
    float vx, float vy, float vz, float length2) {
    auto dx = work, dy = work[stride], dz = work[2 * stride];
    auto projection = work[3 * stride], temp = work[4 * stride];
    auto t1 = work[5 * stride], t2 = work[6 * stride];
    AscendC::Adds(dx, dx, -ax, count);
    AscendC::Adds(dy, dy, -ay, count);
    AscendC::Adds(dz, dz, -az, count);
    if (length2 > 0.0f) {
        AscendC::Muls(projection, dx, vx, count);
        AscendC::Muls(temp, dy, vy, count);
        AscendC::Muls(t1, dz, vz, count);
        AscendC::Add(projection, projection, temp, count);
        AscendC::Add(projection, projection, t1, count);
        AscendC::Maxs(projection, projection, 0.0f, count);
        AscendC::Mins(projection, projection, 1.0f, count);
        AscendC::Muls(temp, projection, -ux, count);
        AscendC::Muls(t1, projection, -uy, count);
        AscendC::Muls(t2, projection, -uz, count);
        AscendC::Add(dx, dx, temp, count);
        AscendC::Add(dy, dy, t1, count);
        AscendC::Add(dz, dz, t2, count);
    }
    AscendC::Mul(dx, dx, dx, count);
    AscendC::Mul(dy, dy, dy, count);
    AscendC::Mul(dz, dz, dz, count);
    AscendC::Add(dx, dx, dy, count);
    AscendC::Add(dx, dx, dz, count);
    AscendC::Sqrt(dx, dx, count);
}

template <typename T>
ELEC_FN void tree_one_segment_load(
    const Config& c, uint64_t batch, uint64_t first, uint32_t count, bool& offsets_ready,
    const AscendC::GlobalTensor<T>& points,
    const AscendC::GlobalTensor<int32_t>& output_index,
    const AscendC::LocalTensor<T>& raw,
    const AscendC::LocalTensor<int32_t>& offsets,
    const AscendC::LocalTensor<int32_t>& indices) {
    constexpr uint32_t tile = 2048;
    const AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    if (offsets_ready) {
        tree_fence<AscendC::HardEvent::V_MTE2>();
        tree_fence<AscendC::HardEvent::MTE3_V>();
    }
    const AscendC::DataCopyExtParams load{1, count * 3 * uint32_t(sizeof(T)), 0, 0, 0};
    const AscendC::DataCopyExtParams store_index{1, count * 4, 0, 0, 0};
    AscendC::DataCopyPad(raw, points[(batch * c.m + first) * 3], load, padding);
    if (!offsets_ready) {
        tree_edge_offsets<12>(offsets, tile);
        AscendC::Duplicate(indices, int32_t(0), tile);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
        offsets_ready = true;
    }
    tree_fence<AscendC::HardEvent::MTE2_V>();
    AscendC::DataCopyPad(output_index[batch * c.m + first], indices, store_index);
}

template <typename T>
ELEC_FN void tree_one_segment_step(
    const Config& c, uint64_t batch, uint64_t first, uint32_t count, bool& offsets_ready,
    const AscendC::GlobalTensor<T>& points,
    const AscendC::GlobalTensor<T>& segments,
    const AscendC::GlobalTensor<T>& output,
    const AscendC::GlobalTensor<int32_t>& output_index,
    const AscendC::LocalTensor<T>& raw,
    const AscendC::LocalTensor<float>& work,
    const AscendC::LocalTensor<int32_t>& offsets,
    const AscendC::LocalTensor<int32_t>& indices) {
    constexpr uint32_t tile = 2048, stride = tile + 64;
    auto values = (sizeof(T) == 2) ? work[7 * stride] : raw.template ReinterpretCast<float>();
    auto half_output = work[5 * stride].template ReinterpretCast<T>();
    const AscendC::DataCopyExtParams store{1, count * uint32_t(sizeof(T)), 0, 0, 0};
    tree_one_segment_load<T>(c, batch, first, count, offsets_ready, points, output_index, raw, offsets, indices);
    const float ax = static_cast<float>(segments.GetValue(batch * 6));
    const float ay = static_cast<float>(segments.GetValue(batch * 6 + 1));
    const float az = static_cast<float>(segments.GetValue(batch * 6 + 2));
    const float ux = static_cast<float>(segments.GetValue(batch * 6 + 3)) - ax;
    const float uy = static_cast<float>(segments.GetValue(batch * 6 + 4)) - ay;
    const float uz = static_cast<float>(segments.GetValue(batch * 6 + 5)) - az;
    const float length2 = (ux * ux + uy * uy) + uz * uz;
    const float inv_length2 = (length2 > 0.0f) ? (1.0f / length2) : 0.0f;
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, count * 3);
    }
    auto off_u32 = offsets.template ReinterpretCast<uint32_t>();
    AscendC::Gather(work, values, off_u32, 0, count);
    AscendC::Gather(work[stride], values[1], off_u32, 0, count);
    AscendC::Gather(work[2 * stride], values[2], off_u32, 0, count);
    tree_one_segment_eval(work, stride, count, ax, ay, az, ux, uy, uz, ux * inv_length2, uy * inv_length2, uz * inv_length2, length2);
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(half_output, work, AscendC::RoundMode::CAST_RINT, count);
    }
    tree_fence<AscendC::HardEvent::V_MTE3>();
    if constexpr (sizeof(T) == 2) {
        AscendC::DataCopyPad(output[batch * c.m + first], half_output, store);
    } else {
        AscendC::DataCopyPad(output[batch * c.m + first], work, store);
    }
}

template <typename T>
ELEC_FN void tree_one_segment(
    const Config& c, GM_ADDR points_address, GM_ADDR segments_address,
    GM_ADDR distance_address, GM_ADDR index_address) {
    constexpr uint32_t tile = 2048, stride = tile + 64;
    constexpr uint32_t work_at = 3 * tile * sizeof(T);
    constexpr uint32_t offset_at = work_at + 9 * stride * sizeof(float);
    constexpr uint32_t index_at = offset_at + tile * sizeof(int32_t);
    AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC, 0, 3 * tile);
    AscendC::LocalTensor<float> work(AscendC::TPosition::VECCALC, work_at, 9 * stride);
    AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC, offset_at, tile);
    AscendC::LocalTensor<int32_t> indices(AscendC::TPosition::VECCALC, index_at, tile);
    AscendC::GlobalTensor<T> points, segments, output;
    AscendC::GlobalTensor<int32_t> output_index;
    points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
    segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
    output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));
    const uint64_t per_batch = (c.m + tile - 1) / tile;
    bool offsets_ready = false;
    for (uint64_t task = AscendC::GetBlockIdx(); task < c.batch * per_batch; task += AscendC::GetBlockNum()) {
        const uint64_t batch = (per_batch > 0) ? (task / per_batch) : 0;
        const uint64_t first = (per_batch > 0) ? ((task % per_batch) * tile) : 0;
        const uint32_t count = static_cast<uint32_t>(c.m - first < tile ? c.m - first : tile);
        tree_one_segment_step<T>(c, batch, first, count, offsets_ready, points, segments, output, output_index, raw, work, offsets, indices);
    }
}

ELEC_FN void tree_one_point_eval_1024(
    const AscendC::LocalTensor<float>& soa,
    const AscendC::LocalTensor<float>& work,
    float px, float py, float pz) {
    constexpr uint32_t tile = 1024;
    auto ax = soa[0], ay = soa[1024], az = soa[2048], ux = soa[3136], uy = soa[4160], uz = soa[5184];
    auto projection = soa[6272], length = soa[7360];
    auto dx = work[0], dy = work[1024], dz = work[2048], temp = work[3136], temp2 = work[4160], temp3 = work[5184];
    AscendC::Sub(ux, ax, ux, 3 * tile);
    AscendC::Mul(temp, ux, ux, 3 * tile);
    AscendC::Add(length, temp, temp2, tile);
    AscendC::Add(length, length, temp3, tile);
    AscendC::Maxs(length, length, bits_float(0x00800000u), tile);
    AscendC::Adds(dx, ax, -px, tile);
    AscendC::Adds(dy, ay, -py, tile);
    AscendC::Adds(dz, az, -pz, tile);
    AscendC::Mul(temp, dx, ux, 3 * tile);
    AscendC::Add(projection, temp, temp2, tile);
    AscendC::Add(projection, projection, temp3, tile);
    AscendC::Div(projection, projection, length, tile);
    AscendC::Maxs(projection, projection, 0.0f, tile);
    AscendC::Mins(projection, projection, 1.0f, tile);
    AscendC::Mul(temp, projection, ux, tile);
    AscendC::Mul(temp2, projection, uy, tile);
    AscendC::Mul(temp3, projection, uz, tile);
    AscendC::Sub(dx, dx, temp, 3 * tile);
    AscendC::Mul(dx, dx, dx, 3 * tile);
    AscendC::Add(dx, dx, dy, tile);
    AscendC::Add(dx, dx, dz, tile);
}

ELEC_FN void tree_one_point_eval_n(
    const AscendC::LocalTensor<float>& soa,
    const AscendC::LocalTensor<float>& work,
    uint32_t n, float px, float py, float pz) {
    auto ax = soa[0], ay = soa[1024], az = soa[2048], ux = soa[3136], uy = soa[4160], uz = soa[5184];
    auto projection = soa[6272], length = soa[7360];
    auto dx = work[0], dy = work[1024], dz = work[2048], temp = work[3136], temp2 = work[4160];
    AscendC::Sub(ux, ax, ux, n);
    AscendC::Sub(uy, ay, uy, n);
    AscendC::Sub(uz, az, uz, n);
    AscendC::Mul(length, ux, ux, n);
    AscendC::Mul(temp, uy, uy, n);
    AscendC::Mul(temp2, uz, uz, n);
    AscendC::Add(length, length, temp, n);
    AscendC::Add(length, length, temp2, n);
    AscendC::Maxs(length, length, bits_float(0x00800000u), n);
    AscendC::Adds(dx, ax, -px, n);
    AscendC::Adds(dy, ay, -py, n);
    AscendC::Adds(dz, az, -pz, n);
    AscendC::Mul(projection, dx, ux, n);
    AscendC::Mul(temp, dy, uy, n);
    AscendC::Mul(temp2, dz, uz, n);
    AscendC::Add(projection, projection, temp, n);
    AscendC::Add(projection, projection, temp2, n);
    AscendC::Div(projection, projection, length, n);
    AscendC::Maxs(projection, projection, 0.0f, n);
    AscendC::Mins(projection, projection, 1.0f, n);
    AscendC::Mul(ax, projection, ux, n);
    AscendC::Mul(ay, projection, uy, n);
    AscendC::Mul(az, projection, uz, n);
    AscendC::Sub(dx, dx, ax, n);
    AscendC::Sub(dy, dy, ay, n);
    AscendC::Sub(dz, dz, az, n);
    AscendC::Mul(dx, dx, dx, n);
    AscendC::Mul(dy, dy, dy, n);
    AscendC::Mul(dz, dz, dz, n);
    AscendC::Add(dx, dx, dy, n);
    AscendC::Add(dx, dx, dz, n);
}

template <typename T>
ELEC_FN void tree_one_point_reduce_and_store(
    const AscendC::GlobalTensor<T>& output,
    const AscendC::GlobalTensor<int32_t>& output_index,
    uint64_t batch, uint32_t n, uint32_t repeats,
    const AscendC::LocalTensor<float>& dx,
    const AscendC::LocalTensor<float>& projection,
    const AscendC::LocalTensor<float>& result) {
    auto dist_out = result[24];
    auto half_output = result[16].template ReinterpretCast<T>();
    auto index_out = result[8].template ReinterpretCast<int32_t>();
    const AscendC::DataCopyExtParams store{1, uint32_t(sizeof(T)), 0, 0, 0};
    const AscendC::DataCopyExtParams store_index{1, 4, 0, 0, 0};
    if (repeats == 1) {
        AscendC::WholeReduceMin(result, dx, static_cast<int32_t>(n), 1, 1, 1, 8);
    } else {
        AscendC::WholeReduceMin(projection, dx, 64, static_cast<int32_t>(repeats), 1, 1, 8);
        const uint64_t value_mask[2] = {((uint64_t(1) << (2 * repeats)) - 1) & uint64_t(0x55555555ULL), 0};
        AscendC::WholeReduceMin(result, projection, value_mask, 1, 1, 1, 8);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::Sqrt(dist_out, result, 8);
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(half_output, dist_out, AscendC::RoundMode::CAST_RINT, 8);
    }
    tree_fence<AscendC::HardEvent::V_MTE3>();
    if constexpr (sizeof(T) == 2) {
        AscendC::DataCopyPad(output[batch], half_output, store);
    } else {
        AscendC::DataCopyPad(output[batch], dist_out, store);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    const uint32_t slot = result.template ReinterpretCast<uint32_t>().GetValue(1);
    const int32_t winner = (repeats == 1)
        ? static_cast<int32_t>(slot)
        : static_cast<int32_t>((slot / 2) * 64 + projection.template ReinterpretCast<uint32_t>().GetValue(slot + 1));
    index_out.SetValue(0, winner);
    tree_fence<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(output_index[batch], index_out, store_index);
}

template <typename T>
ELEC_FN void tree_one_point_load_soa(
    uint64_t batch, uint32_t n, bool& offsets_ready,
    const AscendC::GlobalTensor<T>& segments,
    const AscendC::LocalTensor<T>& raw,
    const AscendC::LocalTensor<float>& input,
    const AscendC::LocalTensor<float>& soa,
    const AscendC::LocalTensor<float>& work,
    const AscendC::LocalTensor<int32_t>& offsets) {
    const AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    if (offsets_ready) {
        tree_fence<AscendC::HardEvent::V_MTE2>();
        tree_fence<AscendC::HardEvent::MTE3_V>();
        tree_fence<AscendC::HardEvent::MTE3_S>();
    }
    const AscendC::DataCopyExtParams load{1, static_cast<uint32_t>(n * 6 * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(raw, segments[batch * n * 6], load, padding);
    if (!offsets_ready) {
        tree_edge_offsets<24>(offsets, n);
        offsets_ready = true;
    }
    tree_fence<AscendC::HardEvent::MTE2_V>();
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(input, raw, AscendC::RoundMode::CAST_NONE, n * 6);
    }
    const uint32_t soa_off[6] = {0, 1024, 2048, 3136, 4160, 5184};
    for (uint32_t axis = 0; axis < 6; ++axis) {
        AscendC::Gather(soa[soa_off[axis]], input[axis], offsets.template ReinterpretCast<uint32_t>(), 0, n);
    }
    if (n % 64) {
        AscendC::Duplicate(work[(n / 64) * 64], bits_float(0x7f800000u), 64);
    }
}

template <typename T>
ELEC_FN void tree_one_point(
    const Config& c, GM_ADDR points_address, GM_ADDR segments_address,
    GM_ADDR distance_address, GM_ADDR index_address) {
    if (AscendC::GetBlockIdx() >= c.batch) return;
    constexpr uint32_t tile = 1024;
    constexpr uint32_t soa_at = 24576, work_at = soa_at + 8448 * sizeof(float);
    constexpr uint32_t offset_at = work_at + 6272 * sizeof(float), point_at = offset_at + tile * sizeof(int32_t);
    constexpr uint32_t result_at = point_at + 128;
    AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC, 0, 6 * tile);
    AscendC::LocalTensor<float> work(AscendC::TPosition::VECCALC, work_at, 6272);
    auto input = (sizeof(T) == 2) ? work : raw.template ReinterpretCast<float>();
    AscendC::LocalTensor<float> soa(AscendC::TPosition::VECCALC, soa_at, 8448);
    AscendC::LocalTensor<int32_t> offsets(AscendC::TPosition::VECCALC, offset_at, tile);
    AscendC::LocalTensor<float> result(AscendC::TPosition::VECCALC, result_at, 32);
    AscendC::GlobalTensor<T> points, segments, output;
    AscendC::GlobalTensor<int32_t> output_index;
    points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points_address));
    segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments_address));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distance_address));
    output_index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index_address));
    const uint32_t n = static_cast<uint32_t>(c.n);
    const uint32_t repeats = (n + 63) / 64;
    bool offsets_ready = false;
    for (uint64_t batch = AscendC::GetBlockIdx(); batch < c.batch; batch += AscendC::GetBlockNum()) {
        tree_one_point_load_soa<T>(batch, n, offsets_ready, segments, raw, input, soa, work, offsets);
        const float px = static_cast<float>(points.GetValue(batch * 3));
        const float py = static_cast<float>(points.GetValue(batch * 3 + 1));
        const float pz = static_cast<float>(points.GetValue(batch * 3 + 2));
        if (n == tile) {
            tree_one_point_eval_1024(soa, work, px, py, pz);
        } else {
            tree_one_point_eval_n(soa, work, n, px, py, pz);
        }
        tree_one_point_reduce_and_store<T>(output, output_index, batch, n, repeats, work, soa[6272], result);
    }
}

}  // namespace elec
