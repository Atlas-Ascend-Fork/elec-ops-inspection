/*
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

#include "tree_line_common.h"
#include "tree_line_mode3.h"
#include "tree_line_static_batch_modes.h"
#include "tree_line_pair32.h"
#include "tree_line_specialized_static.h"
#include "tree_line_general_modes.h"
#include "tree_line_prune.h"
#include "tree_line_small_modes.h"

namespace TreeLineDispatch {

template <typename OpType>
__aicore__ inline void RunStaticOp(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
    const TreeLineMinDistanceTilingData& fixedTiling)
{
    InitSocState();
    OpType op;
    op.Init(points, segments, distances, indices, fixedTiling);
    op.Process();
}

template <typename OpType>
__aicore__ inline void RunPipeOp(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
    const TreeLineMinDistanceTilingData& tilingData)
{
    TPipe pipe;
    OpType op;
    op.Init(points, segments, distances, indices, tilingData, &pipe);
    op.Process();
}

template <typename DT_TREE_POINTS, uint32_t MODE>
__aicore__ inline void DispatchTinyCompileShape(
    GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx)
{
    constexpr uint32_t kTinyN = (MODE == 19U || MODE == 20U) ? 3U : 1U;
    constexpr uint32_t kTinyM = (MODE == 16U || MODE == 19U) ? 1U :
        ((MODE == 17U || MODE == 20U) ? 2U : 6U);
    if constexpr (sizeof(DT_TREE_POINTS) == 2U) {
        InitSocState();
    }
    KernelTreeLineSmallCompileShape<DT_TREE_POINTS> small;
    small.template ProcessFixed<kTinyN, kTinyM, false>(
        tree_points, line_segments, min_distance, nearest_segment_idx);
}

template <typename DT_TREE_POINTS, uint32_t MODE>
__aicore__ inline void DispatchStaticFixedMode(
    GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx)
{
    if constexpr (MODE == 21U) {
        if (GetBlockIdx() >= 3U) {
            return;
        }
        constexpr TreeLineMinDistanceTilingData kTiling{1U, 19U, 7U, 8U, 8U, 21U};
        RunStaticOp<KernelTreeLinePair32<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, kTiling);
    } else if constexpr (MODE == 5U) {
        constexpr TreeLineMinDistanceTilingData kTiling{1U, 33U, 31U, 4U, 9U, 5U};
        RunStaticOp<KernelTreeLinePair32<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, kTiling);
    } else if constexpr (MODE == 6U) {
        constexpr TreeLineMinDistanceTilingData kTiling{1U, 1U, 1024U, 64U, 1U, 6U};
        RunStaticOp<KernelTreeLineSinglePointStatic<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, kTiling);
    } else if constexpr (MODE == 7U) {
        constexpr TreeLineMinDistanceTilingData kTiling{1U, 16384U, 1U, 512U, 40U, 7U};
        RunStaticOp<KernelTreeLineSingleSegmentStatic<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, kTiling);
    }
}

template <typename DT_TREE_POINTS, uint32_t MODE>
__aicore__ inline void DispatchStaticBatchMode(
    GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx)
{
    if constexpr (MODE == 8U) {
        constexpr TreeLineMinDistanceTilingData kTiling{8U, 3U, 19U, 64U, 8U, 8U};
        RunStaticOp<KernelTreeLineMultiSmallStatic<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, kTiling);
    } else if constexpr (MODE == 9U) {
        constexpr TreeLineMinDistanceTilingData kTiling{2U, 256U, 64U, 64U, 32U, 9U};
        RunStaticOp<KernelTreeLineMultiSmallStatic<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, kTiling);
    } else if constexpr (MODE == 10U) {
        constexpr TreeLineMinDistanceTilingData kTiling{4U, 127U, 129U, 64U, 40U, 10U};
        RunStaticOp<KernelTreeLineSmallMStatic<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, kTiling);
    } else if constexpr (MODE == 12U) {
        InitSocState();
        KernelTreeLine513Static op;
        op.Process(tree_points, line_segments, min_distance, nearest_segment_idx);
    }
}

template <typename DT_TREE_POINTS, uint32_t MODE>
__aicore__ inline void DispatchDynamicMode(
    GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx,
    const TreeLineMinDistanceTilingData& tilingData)
{
    if constexpr (MODE == 2U) {
        RunStaticOp<KernelTreeLineSmall<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, tilingData);
    } else if constexpr (MODE == 3U) {
        RunPipeOp<KernelTreeLineMode3<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, tilingData);
    } else if constexpr (MODE == 4U) {
        RunPipeOp<KernelTreeLinePrune<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, tilingData);
    } else if constexpr (MODE == 11U) {
        RunPipeOp<KernelTreeLinePruneLookahead<DT_TREE_POINTS>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, tilingData);
    } else {
        RunPipeOp<KernelTreeLineMinDistance<DT_TREE_POINTS, MODE>>(
            tree_points, line_segments, min_distance, nearest_segment_idx, tilingData);
    }
}

template <typename DT_TREE_POINTS, uint32_t MODE>
__aicore__ inline void DispatchStaticOrDynamic(
    GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR tiling)
{
    constexpr bool kIsFixedSingle = (MODE == 5U || MODE == 6U || MODE == 7U || MODE == 21U);
    constexpr bool kIsFixedBatch = (MODE == 8U || MODE == 9U || MODE == 10U || MODE == 12U);
    if constexpr (kIsFixedSingle) {
        DispatchStaticFixedMode<DT_TREE_POINTS, MODE>(
            tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (kIsFixedBatch) {
        DispatchStaticBatchMode<DT_TREE_POINTS, MODE>(
            tree_points, line_segments, min_distance, nearest_segment_idx);
    } else {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, tiling_data, tiling);
        DispatchDynamicMode<DT_TREE_POINTS, MODE>(
            tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data);
    }
}

}  // namespace TreeLineDispatch

template <typename DT_TREE_POINTS, uint32_t MODE>
__global__ __aicore__ void tree_line_min_distance(
    GM_ADDR tree_points,
    GM_ADDR line_segments,
    GM_ADDR min_distance,
    GM_ADDR nearest_segment_idx,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    if constexpr (MODE >= 16U && MODE <= 20U) {
        TreeLineDispatch::DispatchTinyCompileShape<DT_TREE_POINTS, MODE>(
            tree_points, line_segments, min_distance, nearest_segment_idx);
    } else {
        TreeLineDispatch::DispatchStaticOrDynamic<DT_TREE_POINTS, MODE>(
            tree_points, line_segments, min_distance, nearest_segment_idx, tiling);
    }
}
