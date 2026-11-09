/*
 * Copyright (c) 2026 dihidao.
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
#include "detail/ld_device.h"

using namespace AscendC;

template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__global__ __aicore__ void linalg_det(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgDetTilingDataFull);
    if constexpr (LD_BATCH_SELECTED == 108U) {
#if defined(__DAV_C220_CUBE__)
#elif defined(__DAV_C220_VEC__)
        GM_ADDR ws = GetUserWorkspace(workspace);
        AscendC::LdetDetail::FullFallbackRun102(A, det, ws, false);
#endif
        return;
    }
#if defined(__DAV_C220_CUBE__)
    AscendC::LdetDetail::RunCubeRoute<LD_BATCH_SELECTED>(A, det, workspace, tiling);
#elif defined(__DAV_C220_VEC__)
    AscendC::LdetDetail::RunVectorRoute<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling);
#endif
}

#if defined(LDET_N16_EXPLICIT_SCALAR_ARGUMENTS)
namespace AscendC {
__aicore__ inline void LdetTinySkipUnusedSystemWorkspace() {}
__aicore__ inline GM_ADDR LdetTinyUnusedWorkspace() {
    return nullptr;
}
} // namespace AscendC
#define SetSysWorkspaceForce(...) LdetTinySkipUnusedSystemWorkspace()
#define GetUserWorkspace(...) LdetTinyUnusedWorkspace()
#define set_ffts_base_addr(...) ((void)0)
#endif
