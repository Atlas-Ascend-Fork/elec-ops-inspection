/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

namespace AscendC {

// The two wide sparse variants use the same local layout except for the dot scratch gap.
template <uint32_t Channels>
class SplineOwnerBuffers {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out, uint32_t n,
                                uint32_t e, uint32_t mean)
    {
        nodes = n;
        edgeCount = e;
        average = mean;
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        eg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(edges));
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pseudo));
        wg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight));
        yg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        el = LocalTensor<int64_t>(TPosition::VECCALC, 0, 128);
        pr = LocalTensor<float>(TPosition::VECCALC, 1024, 128);
        zeros = LocalTensor<float>(TPosition::VECCALC, 1536, 32 * 16);
        weights = LocalTensor<float>(TPosition::VECCALC, 3584, 4 * Channels * 16);
        features = LocalTensor<float>(TPosition::VECCALC, FeatureBase, 64 * Channels);
        agg = LocalTensor<float>(TPosition::VECCALC, AggregateBase, 4 * Channels);
        spread = LocalTensor<float>(TPosition::VECCALC, SpreadBase, 4 * Channels * 8);
        dot = LocalTensor<float>(TPosition::VECCALC, DotBase, 4 * Channels * 16);
    }

protected:
    uint32_t nodes;
    uint32_t edgeCount;
    uint32_t average;
    GlobalTensor<float> xg;
    GlobalTensor<float> pg;
    GlobalTensor<float> wg;
    GlobalTensor<float> yg;
    GlobalTensor<int64_t> eg;
    LocalTensor<int64_t> el;
    LocalTensor<float> pr;
    LocalTensor<float> zeros;
    LocalTensor<float> weights;
    LocalTensor<float> features;
    LocalTensor<float> agg;
    LocalTensor<float> spread;
    LocalTensor<float> dot;

private:
    static constexpr uint32_t FeatureBase = 3584 + 4 * Channels * 16 * 4;
    static constexpr uint32_t AggregateBase = FeatureBase + 64 * Channels * 4;
    static constexpr uint32_t SpreadBase = AggregateBase + 4 * Channels * 4;
    static constexpr uint32_t DotBase = SpreadBase + 4 * Channels * 8 * 4 + (Channels == 128 ? 256 : 0);
};

} // namespace AscendC
