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
#include "spline_conv_basis.h"
#include "spline_conv_sparse_buffers.h"
#include "spline_conv_single_stage.h"

namespace AscendC {

template <bool Closed0, bool Closed1, bool LinearReduce = false, bool ExplicitBarrier = !LinearReduce>
__aicore__ inline void RunSingleClearAhead(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                           uint32_t n)
{
    if (GetBlockIdx() != 0) {
        return;
    }
    SplineConvSingleStage<Closed0, Closed1, true> op;
    op.Init(x, edges, pseudo, weight, out);
    op.Preload(edges);
    op.Preload(pseudo);
    op.template ProcessBulk<true, LinearReduce, ExplicitBarrier>(n);
}

template <bool Closed0, bool Closed1, bool Tail>
__aicore__ inline void RunSingleBulkStaged(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out)
{
    const uint32_t n = 2 * GetBlockNum() - (Tail ? 1 : 0);
    if (n > 64) {
        RunSingleStaged<Closed0, Closed1, Tail>(x, edges, pseudo, weight, out);
    } else if (n <= 32) {
        RunSingleClearAhead<Closed0, Closed1, !Tail>(x, edges, pseudo, weight, out, n);
    } else {
        SplineConvSingleStage<Closed0, Closed1, true> op;
        op.Init(x, edges, pseudo, weight, out);
        op.template ProcessChunk<Tail>(n, edges, pseudo);
    }
}

template <bool Closed0, bool Closed1>
__aicore__ inline void RunSingleTailBulk(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out)
{
    const uint32_t blocks = GetBlockNum();
    if (blocks < 2 || blocks > 32) {
        RunSingleStaged<Closed0, Closed1, true>(x, edges, pseudo, weight, out);
        return;
    }
    const uint32_t n = 2 * blocks - 1;
    if (n <= 32) {
        RunSingleClearAhead<Closed0, Closed1>(x, edges, pseudo, weight, out, n);
        return;
    }
    const uint32_t block = GetBlockIdx();
    if (block >= 2) {
        return;
    }
    SplineConvSingleStage<Closed0, Closed1, true> op;
    op.Init(x, edges, pseudo, weight, out);
    op.Preload(edges);
    if (block == 0) {
        op.ClearTail(n);
    } else {
        op.Preload(pseudo);
        op.template ProcessBulk<false>(n);
    }
}

} // namespace AscendC
