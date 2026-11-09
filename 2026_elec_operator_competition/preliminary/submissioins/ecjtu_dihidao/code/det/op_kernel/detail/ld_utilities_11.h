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
#pragma once
// Private device implementation; included by ld_device.h.
#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A>
__aicore__ inline void RunUB216OrderRoute(GM_ADDR A, GM_ADDR det, const LinalgDetTilingData &tiling_data) {
    switch (tiling_data.n) {
    case 209U:
        RunUB216Single<DT_A, 209U>(A, det, tiling_data);
        break;
    case 210U:
        RunUB216Single<DT_A, 210U>(A, det, tiling_data);
        break;
    case 211U:
        RunUB216Single<DT_A, 211U>(A, det, tiling_data);
        break;
    case 212U:
        RunUB216Single<DT_A, 212U>(A, det, tiling_data);
        break;
    case 213U:
        RunUB216Single<DT_A, 213U>(A, det, tiling_data);
        break;
    case 214U:
        RunUB216Single<DT_A, 214U>(A, det, tiling_data);
        break;
    case 215U:
        RunUB216Single<DT_A, 215U>(A, det, tiling_data);
        break;
    case 216U:
        RunUB216Single<DT_A, 216U>(A, det, tiling_data);
        break;
    default:
        break;
    }
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledTailGroup4(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                          const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 22U) {

        if (GetBlockIdx() >= 8U) {
            return true;
        }
        RunWorkspaceVariant60<true, DT_A>(A, det, workspace, tiling_data);

        return true;
    }
    if constexpr (LD_BATCH == 25U) {

        KernelLinalgDet<DT_A, 25U> op;
        op.Init(A, det, workspace, tiling_data);
        op.Process();

        return true;
    }
    if constexpr (LD_BATCH == 23U) {

        RunUB216OrderRoute<DT_A>(A, det, tiling_data);

        return true;
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline void RunGenericVectorRoute(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if (TryN16PlaneRoute<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, tiling_data, tiling);
    if (TryTiledSpecialGroup1<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledSpecialGroup2<LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledSpecialGroup3<LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledSpecialGroup4<LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledSpecialGroup5<LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledTailGroup1<LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledTailGroup2<LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledTailGroup3<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;
    if (TryTiledTailGroup4<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling, tiling_data))
        return;

    linalg_detInitPhase19<DT_A, LD_BATCH>(A, det, workspace, tiling_data);
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline void RunVectorRoute(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    if (TryRagged106<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryRagged105Or107<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryN256DirectGroup1<LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryN256DirectGroup2<LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryN256DirectGroup3<LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryFixedDirectGroup1<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryFixedDirectGroup2<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryFixedDirectGroup3<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    if (TryFixedDirectGroup4<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling))
        return;
    RunGenericVectorRoute<DT_A, LD_BATCH_SELECTED>(A, det, workspace, tiling);
}
#endif
