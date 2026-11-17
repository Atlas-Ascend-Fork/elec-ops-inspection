#include "kernel_operator.h"
#include "fdk_single_entry.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"

template<uint32_t PATH>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, plan, tiling);
    auto selected = plan;
    if constexpr (PATH == 1) {
        wave1::RunVector(feat, weight, bias, keypoints, scores, selected, GetSysWorkSpacePtr());
    } else {
        selected.blocks = selected.batch * 4;
        wave1::RunVector(feat, weight, bias, keypoints, scores, selected, nullptr);
    }
}
