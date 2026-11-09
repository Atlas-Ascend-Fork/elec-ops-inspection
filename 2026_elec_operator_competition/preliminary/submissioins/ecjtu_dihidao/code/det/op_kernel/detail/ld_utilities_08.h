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
#if (defined(__DAV_C220_VEC__))
using LdetFullFallbackPlan56 =
    LdetCyclicSettings46_7<6U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                           true, false, true, 160U, true, false, true, true, true, true>;
#endif

#if (defined(__DAV_C220_VEC__))
__aicore__ inline void FullFallbackRun102(GM_ADDR A, GM_ADDR det, GM_ADDR ws, bool tryHead = true) {
    TPipe pipe;
    const bool active = GetBlockIdx() % 8U < 6U;
    AscendC::ICachePreLoad(7);
    if (!tryHead) {
        if (active) {
            FullFallbackC22102<LdetFullFallbackPlan56> op;
            op.Run(A, det, ws);
        } else {
            for (uint32_t j = 0U; j < 6U; ++j)
                SyncAll<true>();
        }
        return;
    }
    if (active) {
        FullFallbackHead102<LdetFullFallbackPlan56> fast;
        fast.Run(A, det, ws);
    } else
        SyncAll<true>();
    SyncAll<true>();
    bool needsLuFallback = false;
    for (uint32_t i = 0U; i < 3U; ++i) {
        GM_ADDR flag = ws + 392640U + i * 320U + 256U;
        LuFlushFlag(flag);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        if (*reinterpret_cast<volatile __gm__ uint32_t *>(flag) != 0x6B1F0001U)
            needsLuFallback = true;
    }
    SyncAll<true>();
    if (needsLuFallback) {
        if (active) {
            FullFallbackC22102<LdetFullFallbackPlan56> op;
            op.Run(A, det, ws);
        } else {
            for (uint32_t j = 0U; j < 6U; ++j)
                SyncAll<true>();
        }
    }
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <typename DT_A, class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunUB216BatchPhase1(GM_ADDR &A, GM_ADDR &det, CapturedTilingTilingData &tiling_data) {

    switch (tiling_data.n) {
    case 209U:
        RunUB216Batch<DT_A, 209U>(A, det, tiling_data);
        break;
    case 210U:
        RunUB216Batch<DT_A, 210U>(A, det, tiling_data);
        break;
    case 211U:
        RunUB216Batch<DT_A, 211U>(A, det, tiling_data);
        break;
    case 212U:
        RunUB216Batch<DT_A, 212U>(A, det, tiling_data);
        break;
    case 213U:
        RunUB216Batch<DT_A, 213U>(A, det, tiling_data);
        break;
    case 214U:
        RunUB216Batch<DT_A, 214U>(A, det, tiling_data);
        break;
    case 215U:
        RunUB216Batch<DT_A, 215U>(A, det, tiling_data);
        break;
    case 216U:
        RunUB216Batch<DT_A, 216U>(A, det, tiling_data);
        break;
    default:
        break;
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detGetBlockNumPhase2(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                                   CapturedTilingTilingData &tiling_data) {

    if (tiling_data.groupSize == 2U) {
        const uint32_t groups = static_cast<uint32_t>(GetBlockNum()) / 2U;
        const uint32_t group = static_cast<uint32_t>(GetBlockIdx()) / 2U;
        for (uint32_t first = 0U; first < tiling_data.batchCount; first += groups) {
            const uint32_t index = first + group;
            CyclicRowLU<2U, true, false, 16U, true, true, 16U, false, false, false, false, true> op;
            op.Run<true>(A, det, GetUserWorkspace(workspace), index, index < tiling_data.batchCount);
        }
    } else {
        const uint32_t groups = static_cast<uint32_t>(GetBlockNum()) / 4U;
        const uint32_t group = static_cast<uint32_t>(GetBlockIdx()) / 4U;
        for (uint32_t first = 0U; first < tiling_data.batchCount; first += groups) {
            const uint32_t index = first + group;
            if (tiling_data.batchCount >= 3U && tiling_data.batchCount <= 11U && tiling_data.batchCount != 4U &&
                tiling_data.batchCount != 8U) {
                CyclicRowLU<4U, true, false, 40U, true, true, 32U, false, false, false, false, true, true, true> op;
                if (tiling_data.batchCount == 3U)
                    op.Run<true, true>(A, det, GetUserWorkspace(workspace), index, index < tiling_data.batchCount);
                else
                    op.Run<true>(A, det, GetUserWorkspace(workspace), index, index < tiling_data.batchCount);
            } else {
                CyclicRowLU<4U, true, false, 40U, true, true, 32U, false, false, false, false, true> op;
                op.Run<true>(A, det, GetUserWorkspace(workspace), index, index < tiling_data.batchCount);
            }
        }
    }
}
#endif

#if ((defined(__DAV_C220_CUBE__)))
__aicore__ inline void linalg_detDataCopyPhase3(GM_ADDR &A, GM_ADDR &tiling) {

    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    const uint32_t id = GetBlockIdx(), group = td.isLarge == 2U ? td.groupSize : 1U;
    if (group > 1U && id % group != 0U)
        return;
    const uint32_t start = group > 1U ? id / group : id * td.matPerCore;
    const uint32_t count = group > 1U ? 1U : td.matPerCore;
    const uint32_t words = td.n * td.n < 4096U ? td.n * td.n : 4096U, blocks = words / 8U;
    LocalTensor<float> local(TPosition::A1, 0, 4104U);
    for (uint32_t j = 0U; j < count && start + j < td.batchCount; ++j) {
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + static_cast<uint64_t>(start + j) * td.n * td.n,
                           td.n * td.n);
        if (blocks)
            DataCopy(local, gm, DataCopyParams{1U, static_cast<uint16_t>(blocks), 0U, 0U});
        if ((words & 7U) != 0U && words >= 8U)
            DataCopy(local[blocks * 8U], gm[words - 8U], DataCopyParams{1U, 1U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunPhase5(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                           CapturedTilingTilingData &tiling_data) {

    if (tiling_data.n == 192U) {
        CyclicRowLUMedium<4U, true, false, 40U, true, true, 32U, 192U, true, false, false, true, true> op;
        op.Run(A, det, GetUserWorkspace(workspace));
    } else if (tiling_data.n == 200U) {
        if (tiling_data.batchCount <= 8U) {
            AscendC::ICachePreLoad(5);
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                              true, false, true, 128U, true>
                op;
            op.Run(A, det, GetUserWorkspace(workspace), 0U, true, 200U, 4U);
        } else {
            if (tiling_data.batchCount >= 9U && tiling_data.batchCount <= 12U) {
                CyclicRowLUMedium<4U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false,
                                  false, true>
                    op;
                op.Run(A, det, GetUserWorkspace(workspace));
            } else {
                CyclicRowLUMedium<4U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true> op;
                op.Run(A, det, GetUserWorkspace(workspace));
            }
        }
    } else if (tiling_data.n == 208U) {
        CyclicRowLUMedium<4U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, true> op;
        op.Run(A, det, GetUserWorkspace(workspace));
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <typename DT_A>
__aicore__ inline void linalg_detGET_TILING_DATA_WITH_STRUCTPhase6(GM_ADDR &A, GM_ADDR &det, GM_ADDR &tiling,
                                                                   GM_ADDR &workspace) {

    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    const uint32_t n = td.n, b = td.batchCount;
    if (n == 19U && b == 8U) {
        const LinalgDetTilingData fixed{8U, 19U, 24U, 1U, 0U, 1U};
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(A, det, workspace, fixed, true, true, true);
        op.ProcessN19B8FixedEntry();
    } else {
        const LinalgDetTilingData local{b, n, td.stride, td.matPerCore, td.isLarge, td.groupSize};
        const bool family = n <= 19U || n == 24U || n == 32U || (n == 20U && b == 1U);
        const bool dma = n == 17U || n == 18U || n == 19U || ((n == 20U || n == 30U) && b == 1U);
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(A, det, workspace, local, family, dma, family);
        op.ProcessUnifiedSmallEntry();
    }
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunPhase7(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                           CapturedTilingTilingData &tiling_data) {

    if (tiling_data.n == 208U) {
        CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, false, false,
                          true, false, true, 128U, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace));
        return;
    }
    if (tiling_data.n == 192U) {
        CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 192U, true, false, false, true, true, false, false,
                          true, false, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace));
    } else if (tiling_data.n == 200U) {
        if (tiling_data.batchCount == 9U) {
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                              true, false, true, 128U, true>
                op;
            op.Run(A, det, GetUserWorkspace(workspace));
        } else {
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                              true, false, true, 128U, true>
                op;
            op.Run(A, det, GetUserWorkspace(workspace));
        }
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunPhase8(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                           CapturedTilingTilingData &tiling_data) {

    if (tiling_data.n == 192U) {
        CyclicRowLUMedium<6U, true, false, 40U, true, true, 32U, 192U, true, false, false, true, true> op;
        op.Run(A, det, GetUserWorkspace(workspace));
    } else if (tiling_data.n == 200U) {
        AscendC::ICachePreLoad(5);
        CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                          true, false, true, 128U, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace), 0U, true, 200U, (tiling_data.groupSize == 6U ? 6U : 7U));
    } else if (tiling_data.n == 208U) {
        if (tiling_data.groupSize == 6U) {
            CyclicRowLUMedium<6U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, true> op;
            op.Run(A, det, GetUserWorkspace(workspace));
        } else {
            CyclicRowLUMedium<7U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, true> op;
            op.Run(A, det, GetUserWorkspace(workspace));
        }
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase9(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(7);
    GM_ADDR privateWs = GetUserWorkspace(workspace);
    CyclicN200HeadPairs<
        LdetCyclicSettings46_7<6U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false,
                               false, true, false, true, 160U, true, false, true, true, true, true>>
        fast;
    fast.Run(A, det, privateWs);
    SyncAll<true>();
    bool needsLuFallback = false;
    for (uint32_t i = 0U; i < 3U; ++i) {
        GM_ADDR flag = privateWs + 392640U + i * 320U + 256U;
        LuFlushFlag(flag);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        if (*reinterpret_cast<volatile __gm__ uint32_t *>(flag) != 0x6B1F0001U)
            needsLuFallback = true;
    }
    SyncAll<true>();
    if (needsLuFallback) {
        CyclicRowLUN200C22<
            LdetCyclicSettings46_7<6U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false,
                                   false, true, false, true, 160U, true, false, true, true, true, true>>
            op;
        op.Run(A, det, privateWs);
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunPhase10(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                            CapturedTilingTilingData &tiling_data) {

    if (tiling_data.n == 192U) {
        CyclicRowLUMedium<2U, true, false, 40U, true, true, 32U, 192U, true, false, false, true, true, false, false,
                          true, false, true, 96U, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace));
    } else if (tiling_data.n == 200U) {
        if (tiling_data.groupSize == 3U) {
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                              true>
                op;
            op.Run(A, det, GetUserWorkspace(workspace));
        } else {
            CyclicRowLUMedium<2U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                              true>
                op;
            op.Run(A, det, GetUserWorkspace(workspace));
        }
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunPhase11(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                            CapturedTilingTilingData &tiling_data) {

    if (tiling_data.n == 192U) {
        if (tiling_data.batchCount == 12U) {
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 192U, false, true, true, true, true, true> op;
            op.Run(A, det, GetUserWorkspace(workspace));
        } else if (tiling_data.batchCount >= 13U && tiling_data.batchCount <= 16U) {
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 192U, true, false, false, true, true, false, false,
                              true>
                op;
            op.Run(A, det, GetUserWorkspace(workspace));
        } else {
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 192U, true, false, false, true, true> op;
            op.Run(A, det, GetUserWorkspace(workspace));
        }
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <bool UB_TAIL, typename DT_A, class CapturedTilingTilingData>
__aicore__ inline void RunWorkspaceVariant60(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                             CapturedTilingTilingData &tiling_data) {

    GM_ADDR ws = GetUserWorkspace(workspace);
    uint64_t span = (2ULL * tiling_data.n * tiling_data.stride + 512U * 24U + 32U + 256U) * 4U;
    GM_ADDR neg = ws + span;
    GM_ADDR u = neg + 32768U;
    GM_ADDR flags = u + 65536U;
    if (GetBlockIdx() == 0U)
        LuClear(flags);
    KernelLinalgDet<DT_A, 2, true, false, false, false, false, 0, true, true, true, UB_TAIL> op;
    op.SetCube(neg, u, flags, 1U);
    op.InitUserWorkspace(A, det, ws, tiling_data);
    op.Process();
    if (GetBlockIdx() == 0U) {
        for (uint32_t i = 0; i < 4U; ++i)
            LuPoll(flags + (13U + i) * 64U, 0x6ADBD431U);
        LuClear(flags);
    }
}
#endif

#if ((defined(__DAV_C220_CUBE__)))
__aicore__ inline void linalg_detDataCopyPhase14(GM_ADDR &A, GM_ADDR &tiling) {

    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    if (GetBlockIdx() != 0U)
        return;
    const uint32_t words = td.n * td.n, blocks = words / 8U;
    LocalTensor<float> local(TPosition::A1, 0, 4104U);
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A), words);
    if (blocks)
        DataCopy(local, gm, DataCopyParams{1U, static_cast<uint16_t>(blocks), 0U, 0U});
    if ((words & 7U) != 0U && words >= 8U)
        DataCopy(local[blocks * 8U], gm[words - 8U], DataCopyParams{1U, 1U, 0U, 0U});
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    return;
}
#endif
