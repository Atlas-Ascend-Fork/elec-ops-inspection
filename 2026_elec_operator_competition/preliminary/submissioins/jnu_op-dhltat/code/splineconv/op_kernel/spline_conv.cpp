/*
 * Copyright (c) 2026 dhltat.
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

#include "spline_row_owner.h"
#include "spline_single_edge.h"
#include "spline_sparse_graph.h"
#include "spline_static_graph.h"
#include "spline_grid_single.h"

template <uint32_t MODE, uint32_t DIM, uint32_t DEG, uint32_t CI, uint32_t CO, uint32_t ES, uint32_t NN, typename INPUT_T>
__global__ __aicore__ void spline_conv(GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight,
                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    using DT_X = INPUT_T;
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    // Each launch establishes normal mode before any manual mask operation.
    SetMaskNorm();
    if constexpr (MODE == kBoundedUnitCubeSingleMode)
        UnitCubeSingleEdge<DT_X, CI, CO, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kUnitCubeSingleMode)
        UnitCubeSingleEdge<DT_X, CI, CO>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kGeneralSingleMode)
        GeneralSingleEdge<DT_X, CI, CO>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedCoordinateSingleMode)
        CachedCoordinateSingleEdge<CI, CO, NN>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedFloatSingleMode)
        CachedFloatSingleEdge<CI, CO, NN>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kPairedIncomingGraphMode)
        RuntimeSparseGraph<DT_X, CI, CO, true, false, kPairedOwnerGroupWidth, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kSmallBilinearSingleMode)
        RuntimeSingleEdge<DT_X, CI, CO, NN>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kSparseSeparateZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, false>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kSparseCoownedZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedSparseSeparateZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, false, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kCachedSparseCoownedZeroMode)
        RuntimeSparseGraph<DT_X, CI, CO, true, true>(x, edge_index, edge_attr, weight, out, tiling);
    else if constexpr (MODE == kStaticSparseMode) {
        if constexpr (ES == 8)
            RuntimeSparseGraph<DT_X, CI, CO, true, false, ES>(x, edge_index, edge_attr, weight, out, tiling);
        else if constexpr (ES != 0) {
            StaticEdgeGraph<DT_X, CI, CO, false, ES, true>(x, edge_index, edge_attr, weight, out, tiling);
        } else {
            GlobalTensor<uint64_t> metadata;
            metadata.SetGlobalBuffer((__gm__ uint64_t*)tiling);
            const uint32_t edgeCount = (metadata.GetValue(0) >> kSparseEdgeShift) & kSparseEdgeMask;
            StaticEdgeGraph<DT_X, CI, CO, false, 0, true>(x, edge_index, edge_attr, weight, out, tiling, edgeCount);
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData, t, tiling);
        KernelSplineConv<DT_X, MODE, DIM, DEG> op;
        op.Run(x, edge_index, edge_attr, weight, out, workspace, t);
    }
}

// Basic vector operations need no library system workspace. The generated
// entry passes the raw user workspace through unchanged; only the owned-row
// fallback consumes it, using disjoint private edge-list regions per core.
// No MODE0 entry is compiled, so this covers every executable template key.
#if (!defined(ASCENDC_CPU_DEBUG) || (ASCENDC_CPU_DEBUG == 0)) && \
    (!defined(ASCENDC_DUMP) || (ASCENDC_DUMP == 0)) && \
    (!defined(ASCENDC_TIME_STAMP_ON) || (ASCENDC_TIME_STAMP_ON == 0))
namespace AscendC {
__aicore__ inline void SplineNoSystemWorkspace(GM_ADDR) {}
__aicore__ inline GM_ADDR SplineRawUserWorkspace(GM_ADDR workspace) { return workspace; }
} // namespace AscendC
#define SetSysWorkspaceForce SplineNoSystemWorkspace
#define GetUserWorkspace SplineRawUserWorkspace
#endif


// CANN inlines the user template into a separate AIV entry. Annotate that
// same entry's declaration so its instruction address is consistently aligned.
// The seven arguments and the entire generated computation remain unchanged.
// Other frontends, including Host metadata extraction, have no such entry macro.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && defined(auto_gen_spline_conv_kernel)
extern "C" __aicore__ __attribute__((cce_kernel, aligned(512)))
void auto_gen_spline_conv_kernel(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
#endif
