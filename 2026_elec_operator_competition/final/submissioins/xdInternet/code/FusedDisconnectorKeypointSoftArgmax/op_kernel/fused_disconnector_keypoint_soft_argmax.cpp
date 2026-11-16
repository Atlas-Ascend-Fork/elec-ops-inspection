/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"

#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"
#include "keypoint_stripe_reduce.h"
#include "keypoint_plane_fast.h"
#include "keypoint_mode2_general.h"
#include "keypoint_mode4_strip_exec.h"

namespace keypoint_dispatch {

struct StripeTplCfg {
    uint32_t m, s, pitch, ch;
    bool half1;
};

static __aicore__ inline constexpr StripeTplCfg GetStripeCfg(uint32_t mode)
{
    constexpr StripeTplCfg kTable[17] = {
        {1U, 5U, 1536U, 8U, false},
        {1U, 5U, 1536U, 8U, false},
        {1U, 5U, 1536U, 8U, false},
        {1U, 5U, 1536U, 8U, false},
        {1U, 5U, 1536U, 8U, false},
        {1U, 5U, 1536U, 8U, false},
        {1U, 5U, 1536U, 8U, true},
        {1U, 3U, 2432U, 2U, true},
        {2U, 5U, 1536U, 4U, true},
        {2U, 5U, 896U, 8U, false},
        {1U, 4U, 1536U, 8U, false},
        {1U, 3U, 1536U, 8U, false},
        {1U, 4U, 1792U, 4U, true},
        {1U, 5U, 1536U, 8U, false},
        {1U, 9U, 896U, 8U, true},
        {1U, 10U, 1792U, 4U, false},
        {1U, 5U, 3456U, 1U, false},
    };
    return (mode <= 16U) ? kTable[mode] : kTable[5U];
}

template <typename DirectOp>
static __attribute__((noinline, aligned(512))) __aicore__ void LaunchDirectOp(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData& tiling)
{
    AscendC::InitSocState();
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
    DirectOp op;
    op.Process(feat, weight, bias, keypoints, scores, tiling);
}

template <typename WorkspaceOp>
static __attribute__((noinline, aligned(512))) __aicore__ void LaunchWorkspaceOp(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    GM_ADDR workspace, const FusedDisconnectorKeypointSoftArgmaxTilingData& tiling)
{
    AscendC::InitSocState();
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
    WorkspaceOp wsOp;
    wsOp.Process(feat, weight, bias, keypoints, scores, workspace, tiling);
}

}

template <typename DT_FEAT, uint32_t KERNEL_MODE>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, tilingData, tiling);
    if constexpr (KERNEL_MODE == 13U) {
        keypoint_dispatch::LaunchDirectOp<keypoint_impl::PlaneFastAivKernel<3200U, 2U>>(
            feat, weight, bias, keypoints, scores, tilingData);
    } else if constexpr (KERNEL_MODE >= 5U && KERNEL_MODE <= 16U) {
        constexpr keypoint_dispatch::StripeTplCfg cfg = keypoint_dispatch::GetStripeCfg(KERNEL_MODE);
        keypoint_dispatch::LaunchDirectOp<
            keypoint_impl::StripeFastKernel<cfg.m, cfg.s, cfg.pitch, cfg.ch, cfg.half1>>(
            feat, weight, bias, keypoints, scores, tilingData);
    } else if constexpr (KERNEL_MODE == 4U) {
        keypoint_dispatch::LaunchWorkspaceOp<keypoint_impl::KeypointMode4BKFast>(
            feat, weight, bias, keypoints, scores, workspace, tilingData);
    } else {
        keypoint_dispatch::LaunchWorkspaceOp<keypoint_impl::KeypointMode2Kernel>(
            feat, weight, bias, keypoints, scores, workspace, tilingData);
    }
}
