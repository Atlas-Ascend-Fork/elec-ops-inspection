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

#include "kernel_operator.h"
#include "spline_conv_tiling.h"
#include "spline_conv_resident.h"
#include "spline_conv_streaming.h"
#include "spline_conv_small.h"
#include "spline_conv_single.h"
#include "spline_conv_single_stage.h"
#include "spline_conv_single_bulk_stage.h"
#include "spline_conv_local_weights.h"
#include "spline_conv_compact.h"
#include "spline_conv_edge_owner.h"
#include "spline_conv_four_owner.h"
#include "spline_conv_wide_owner.h"
#include "tiling_key_spline_conv.h"

using namespace AscendC;

namespace {
template <typename T>
__aicore__ inline bool RunBoundedSparse(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                        GM_ADDR workspace, const SplineConvTilingData &t)
{
    if (t.ci != 32 || t.streaming || !t.cacheGraph || !t.cacheWeights || !t.fusedDot || t.hasRoot || t.hasBias ||
        t.e >= t.n || t.co != 16 || t.ci != t.ciPad || t.dim != 2 || t.degree != 1 || t.kernels != 4) {
        return false;
    }
    SplineConvResident<T, false, 32> op;
    op.Init(x, edges, pseudo, weight, nullptr, nullptr, out, workspace, t);
    op.Process();
    return true;
}

template <typename T, uint32_t Single, bool Closed0, bool Closed1, uint32_t NodePad>
__aicore__ inline void RunSingleSpecialized(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                            GM_ADDR tiling)
{
    if constexpr (Single == 6) {
        RunSingleClearAhead<Closed0, Closed1, true, true>(x, edges, pseudo, weight, out, 31 - 2 * NodePad);
    } else if constexpr (Single == 5 && sizeof(T) == 4) {
        RunSingleTailBulk<Closed0, Closed1>(x, edges, pseudo, weight, out);
    } else if constexpr (Single == 4 && sizeof(T) == 4) {
        RunSingleStaged<Closed0, Closed1, true>(x, edges, pseudo, weight, out);
    } else if constexpr (Single == 3 && sizeof(T) == 4) {
        RunSingleBulkStaged<Closed0, Closed1, false>(x, edges, pseudo, weight, out);
    } else if constexpr (Single == 4) {
        RunSingleStatic<T, false, Closed0, Closed1, true, true>(x, edges, pseudo, weight, out, tiling);
    } else if constexpr (Single == 3) {
        RunSingleStatic<T, false, Closed0, Closed1, true>(x, edges, pseudo, weight, out, tiling);
    } else if constexpr (Single == 2) {
        RunSingleStatic<T, true, Closed0, Closed1>(x, edges, pseudo, weight, out, tiling);
    } else {
        RunSingleStatic<T>(x, edges, pseudo, weight, out, tiling);
    }
}

template <typename T, uint32_t Sparse, bool Closed0, bool Closed1, uint32_t NodePad>
__aicore__ inline void RunSparseSpecialized(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                            GM_ADDR tiling)
{
    if constexpr (Sparse == 5) {
        RunPartialEdgeOwner<NodePad, Closed0, Closed1>(x, edges, pseudo, weight, out, tiling);
    } else if constexpr (Sparse == 6) {
        RunFixedEdgeOwner<NodePad, Closed0, Closed1>(x, edges, pseudo, weight, out, tiling);
    } else if constexpr (Sparse == 8) {
        RunWideEarlyOwner<Closed0, Closed1>(x, edges, pseudo, weight, out, tiling);
    } else {
        const auto *td = reinterpret_cast<const __gm__ SplineConvTilingData *>(tiling);
        SplineConvTilingData minimal{};
        minimal.n = td->n;
        minimal.e = td->e;
        minimal.blocks = GetBlockNum();
        minimal.scale[0] = Closed0 ? 2 : 1;
        minimal.scale[1] = Closed1 ? 2 : 1;
        minimal.mean = td->mean;
        if constexpr (Sparse == 3) {
            SplineConvEdgeOwner<Closed0, Closed1> op;
            op.Init(x, edges, pseudo, weight, out, minimal.n, minimal.e, minimal.mean);
            op.Process();
        } else {
            SplineConvCompact<T, Sparse == 1 ? 32 : 128> op;
            op.Init(x, edges, pseudo, weight, out, minimal, nullptr);
            if (minimal.blocks == 8) {
                op.template Process<8>();
            } else {
                op.Process();
            }
        }
    }
}

template <typename T>
__aicore__ inline void RunGeneral(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR root, GM_ADDR bias,
                                  GM_ADDR out, GM_ADDR workspace, const SplineConvTilingData &t)
{
    if (RunSparseLocalWeights<T>(x, edges, pseudo, weight, out, t) ||
        RunCompactSparse<T>(x, edges, pseudo, weight, out, t) ||
        RunBoundedSparse<T>(x, edges, pseudo, weight, out, workspace, t)) {
        return;
    }
    if (t.smallScalar) {
        TPipe pipe;
        SplineConvSmall<T> op;
        op.Init(x, edges, pseudo, weight, root, bias, out, t, &pipe);
        op.Process();
    } else if (t.streaming) {
        SplineConvStreaming<T> op;
        op.Init(x, edges, pseudo, weight, root, bias, out, workspace, t);
        op.Process();
    } else if (t.kernels > 64) {
        SplineConvResident<T, true> op;
        op.Init(x, edges, pseudo, weight, root, bias, out, workspace, t);
        op.Process();
    } else {
        SplineConvResident<T> op;
        op.Init(x, edges, pseudo, weight, root, bias, out, workspace, t);
        op.Process();
    }
}
} // namespace

template <typename DT_X, uint32_t SINGLE, uint32_t SPARSE, bool CLOSED0, bool CLOSED1, uint32_t NODEPAD>
__global__ __aicore__ void spline_conv(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                       GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (SINGLE != 0) {
        RunSingleSpecialized<DT_X, SINGLE, CLOSED0, CLOSED1, NODEPAD>(x, edges, pseudo, weight, out, tiling);
    } else if constexpr (SPARSE != 0) {
        RunSparseSpecialized<DT_X, SPARSE, CLOSED0, CLOSED1, NODEPAD>(x, edges, pseudo, weight, out, tiling);
    } else {
        REGISTER_TILING_DEFAULT(SplineConvTilingData);
        GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData, t, tiling);
        RunGeneral<DT_X>(x, edges, pseudo, weight, nullptr, nullptr, out, workspace, t);
    }
}

template <typename DT_X, uint32_t SINGLE, uint32_t SPARSE, bool CLOSED0, bool CLOSED1, uint32_t NODEPAD>
__global__ __aicore__ void spline_conv_full(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR root,
                                            GM_ADDR bias, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (SINGLE != 0) {
        RunSingleSpecialized<DT_X, SINGLE, CLOSED0, CLOSED1, NODEPAD>(x, edges, pseudo, weight, out, tiling);
    } else if constexpr (SPARSE != 0) {
        RunSparseSpecialized<DT_X, SPARSE, CLOSED0, CLOSED1, NODEPAD>(x, edges, pseudo, weight, out, tiling);
    } else {
        REGISTER_TILING_DEFAULT(SplineConvTilingData);
        GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData, t, tiling);
        RunGeneral<DT_X>(x, edges, pseudo, weight, root, bias, out, workspace, t);
    }
}

namespace AscendC {
// The kernels use basic APIs and explicit scratch, with no system workspace consumer.
__aicore__ inline void SplineConvBasicWorkspace(GM_ADDR)
{
}
__aicore__ inline GM_ADDR SplineConvUserWorkspace(GM_ADDR workspace)
{
    return workspace + RESERVED_WORKSPACE;
}
} // namespace AscendC

#define SetSysWorkspaceForce SplineConvBasicWorkspace
#define GetUserWorkspace SplineConvUserWorkspace
