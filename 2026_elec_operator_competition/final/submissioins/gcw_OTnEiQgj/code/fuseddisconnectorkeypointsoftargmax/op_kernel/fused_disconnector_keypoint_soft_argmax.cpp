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

#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"
#include "kpt_p8.h"
#include "kpt_p9.h"

template <typename DT_FEAT, uint64_t SHAPE_MODE>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    (void)workspace;
    if constexpr (SHAPE_MODE < 12ULL) {
        elec::impl::EntryModeFixed<SHAPE_MODE>(feat, weight, bias, keypoints, scores, tiling);
    } else if constexpr (SHAPE_MODE < 15ULL) {
        elec::impl::EntryModeVar<static_cast<uint32_t>(SHAPE_MODE - 12ULL)>(
            feat, weight, bias, keypoints, scores, tiling);
    } else {
        elec::impl::EntryModeFull(feat, weight, bias, keypoints, scores, tiling);
    }
}

#if (!defined(ASCENDC_CPU_DEBUG) || (ASCENDC_CPU_DEBUG == 0)) && \
    !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
namespace AscendC {
__aicore__ inline void IgnoreSysWorkspace(GM_ADDR) {}
__aicore__ inline GM_ADDR RawUserWorkspace(GM_ADDR ws) { return ws; }
}
#undef SetSysWorkspaceForce
#define SetSysWorkspaceForce IgnoreSysWorkspace
#undef GetUserWorkspace
#define GetUserWorkspace RawUserWorkspace
#endif
