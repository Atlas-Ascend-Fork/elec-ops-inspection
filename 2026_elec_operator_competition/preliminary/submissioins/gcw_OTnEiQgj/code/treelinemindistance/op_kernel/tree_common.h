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
#include "kernel_operator.h"
#include "tree_line_min_distance_tiling.h"

#define ELEC_FN __aicore__ inline

namespace elec {

ELEC_FN uint32_t float_bits(float x) {
    union {
        float f;
        uint32_t u;
    } v = {x};
    return v.u;
}

ELEC_FN float bits_float(uint32_t x) {
    union {
        uint32_t u;
        float f;
    } v = {x};
    return v.f;
}

template <typename U>
ELEC_FN AscendC::LocalTensor<U> local_tensor_at(
    uint32_t byte_offset, uint32_t /*byte_size*/, uint32_t count) {
    return AscendC::LocalTensor<U>(AscendC::TPosition::VECCALC, byte_offset, count);
}

template <AscendC::HardEvent Event>
ELEC_FN void tree_fence() {
    AscendC::SetFlag<Event>(0);
    AscendC::WaitFlag<Event>(0);
}

template <typename T>
ELEC_FN void tree_gather_points_3d(
    const AscendC::LocalTensor<float>& point,
    const AscendC::LocalTensor<T>& point_raw,
    const AscendC::LocalTensor<float>& point_f32,
    const AscendC::LocalTensor<int32_t>& pt_offsets,
    uint32_t count) {
    auto off_u32 = pt_offsets.template ReinterpretCast<uint32_t>();
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(point_f32, point_raw, AscendC::RoundMode::CAST_NONE, count * 3);
        AscendC::Gather(point, point_f32, off_u32, 0, count);
        AscendC::Gather(point[64], point_f32[1], off_u32, 0, count);
        AscendC::Gather(point[128], point_f32[2], off_u32, 0, count);
    } else {
        auto point_values = point_raw.template ReinterpretCast<float>();
        AscendC::Gather(point, point_values, off_u32, 0, count);
        AscendC::Gather(point[64], point_values[1], off_u32, 0, count);
        AscendC::Gather(point[128], point_values[2], off_u32, 0, count);
    }
}

template <typename T>
ELEC_FN void tree_write_outputs(
    const AscendC::GlobalTensor<T>& output,
    const AscendC::GlobalTensor<int32_t>& output_index,
    uint64_t out_offset,
    const AscendC::LocalTensor<float>& d2_in,
    const AscendC::LocalTensor<float>& distance,
    const AscendC::LocalTensor<T>& packed,
    const AscendC::LocalTensor<int32_t>& best_index,
    uint32_t count) {
    const AscendC::DataCopyExtParams store{1, count * uint32_t(sizeof(T)), 0, 0, 0};
    const AscendC::DataCopyExtParams store_index{1, count * 4, 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::Sqrt(distance, d2_in, count);
    if constexpr (sizeof(T) == 2) {
        AscendC::Cast(packed, distance, AscendC::RoundMode::CAST_RINT, count);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopyPad(output_index[out_offset], best_index, store_index);
    tree_fence<AscendC::HardEvent::V_MTE3>();
    if constexpr (sizeof(T) == 2) {
        AscendC::DataCopyPad(output[out_offset], packed, store);
    } else {
        AscendC::DataCopyPad(output[out_offset], distance, store);
    }
}

}  // namespace elec
