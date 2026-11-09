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
#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunPhase15(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                            CapturedTilingTilingData &tiling_data) {

    if (tiling_data.groupSize == 7U) {
        CyclicRowLUMedium<7U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, true, false,
                          false, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace), 0U, true, tiling_data.n);
    } else if (tiling_data.groupSize == 4U) {
        CyclicRowLUMedium<4U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, true, false,
                          false, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace), 0U, true, tiling_data.n);
    } else {
        CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, true, false,
                          false, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace), 0U, true, tiling_data.n);
    }
    return;
}
#endif

#if ((defined(__DAV_C220_CUBE__)))
__aicore__ inline void linalg_detDataCopyPhase16(GM_ADDR &A, GM_ADDR &tiling) {

    const uint32_t id = GetBlockIdx();
    auto *plan = reinterpret_cast<__gm__ volatile uint64_t *>(tiling + 64U) + id * 2U;
    const uint64_t config = plan[1U];
    const uint16_t blocks = static_cast<uint16_t>(config >> 32U);
    if (blocks == 0U)
        return;
    const uint32_t words = static_cast<uint32_t>(config);
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A + plan[0U]), words);
    LocalTensor<float> local(TPosition::A1, 0, 8200U);
    DataCopy(local, gm, DataCopyParams{1U, blocks, 0U, 0U});
    if ((config >> 48U) != 0U)
        DataCopy(local[static_cast<uint32_t>(blocks) * 8U], gm[words - 8U], DataCopyParams{1U, 1U, 0U, 0U});
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    return;
}
#endif

#if ((defined(__DAV_C220_CUBE__)))
__aicore__ inline void linalg_detDataCopyPhase17(GM_ADDR &A, GM_ADDR &tiling) {

    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    const uint32_t start = GetBlockIdx() * 2U * td.matPerCore;
    if (start >= td.batchCount)
        return;
    const uint32_t count = td.batchCount - start < 2U * td.matPerCore ? td.batchCount - start : 2U * td.matPerCore;
    const uint32_t allWords = count * td.n * td.n, words = allWords < 8192U ? allWords : 8192U;
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A) + static_cast<uint64_t>(start) * td.n * td.n, allWords);
    LocalTensor<float> local(TPosition::A1, 0, 8192U);
    DataCopy(local, gm, DataCopyParams{1U, static_cast<uint16_t>(words / 8U), 0U, 0U});
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    return;
}
#endif

#if ((defined(__DAV_C220_CUBE__)))
template <uint32_t LD_BATCH_SELECTED> __aicore__ inline void linalg_detDataCopyPhase18() {

    constexpr uint32_t N = LD_BATCH_SELECTED == 64U ? 8U : 32U;
    const uint32_t id = GetBlockIdx();
    if (id >= 1U)
        return;
    auto *params = reinterpret_cast<__gm__ volatile uint64_t *>(get_para_base());
    GM_ADDR source = reinterpret_cast<GM_ADDR>(params[1U]);
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(source), N * N);
    LocalTensor<float> local(TPosition::A1, 0, N * N);
    DataCopy(local, gm, DataCopyParams{1U, static_cast<uint16_t>(N * N / 8U), 0U, 0U});
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <auto LD_BATCH>
__aicore__ inline constexpr auto linalg_detInitPhase19IndexSelect22Stage9()
    -> decltype(LD_BATCH == 19U ? 14U
                                : (LD_BATCH == 18U ? 5U : (LD_BATCH == 16U ? 2U : (LD_BATCH == 17U ? 1U : LD_BATCH)))) {
    if (LD_BATCH == 19U) {
        return 14U;
    }
    if (LD_BATCH == 18U) {
        return 5U;
    }
    if (LD_BATCH == 16U) {
        return 2U;
    }
    if (LD_BATCH == 17U) {
        return 1U;
    }
    return LD_BATCH;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <typename DT_A, auto LD_BATCH, class CapturedTilingTilingData>
__aicore__ inline void linalg_detInitPhase19(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                             CapturedTilingTilingData &tiling_data) {

    KernelLinalgDet<DT_A, (linalg_detInitPhase19IndexSelect22Stage9<LD_BATCH>()), LD_BATCH == 16U, LD_BATCH == 17U,
                    LD_BATCH >= 18U, false, LD_BATCH >= 19U, 0U>
        op;
    op.Init(A, det, workspace, tiling_data);
    op.Process();
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase20(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(7);
    if (GetBlockIdx() % 7U < 6U) {
        CyclicN256PacketSafe26624<LdetCyclicSettings46_4<7U, true, true, 40U, true, true, 32U, false, true, false,
                                                         false, true, true, false, true, true, true, true, true>>
            op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    } else {
        CyclicN256PacketSafe26624<LdetCyclicSettings46_4<7U, true, true, 40U, true, true, 32U, false, true, false,
                                                         false, true, true, false, true, true, true, true, false>>
            op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase21(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(7);
    if (GetBlockIdx() % 7U < 6U) {
        CyclicN256Seven<LdetCyclicSettings46_4<7U, true, true, 40U, true, true, 32U, false, true, false, false, true,
                                               true, false, true, true, true, true, true>>
            op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    } else {
        CyclicN256Seven<LdetCyclicSettings46_4<7U, true, true, 40U, true, true, 32U, false, true, false, false, true,
                                               true, false, true, true, true, true, false>>
            op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    }
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <typename DT_A>
__aicore__ inline void linalg_detDataCopyPadPhase22(GM_ADDR &input, const uint32_t &inputN, GM_ADDR &output) {

    if (GetBlockIdx() != 0U)
        return;
    LocalTensor<float> packed(TPosition::VECCALC, ::kSpMat, 64U * 72U);
    GlobalTensor<float> inputGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input), inputN * inputN);
    DataCopyPad(packed, inputGm, DataCopyExtParams{static_cast<uint16_t>(inputN), inputN * 4U, 0U, 1U, 0U},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    const LinalgDetTilingData capacity{1U, inputN, 72U, 1U, 0U, 1U};
    KernelLinalgDet<DT_A, 11U> op;
    op.Init(input, output, nullptr, capacity, false, false, false, true, false);
    op.ProcessCapacity64Prepared(inputN);
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
template <class CapturedTilingTilingData>
__aicore__ inline void linalg_detRunPhase23(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace,
                                            CapturedTilingTilingData &tiling_data) {

    if (tiling_data.n == 200U) {
        CyclicRowLUMedium<2U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false,
                          true, false, true, 128U, true, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace));
    } else {
        CyclicRowLUMedium<2U, true, false, 40U, true, true, 32U, 192U, true, false, false, true, true, false, false,
                          true, false, true, 96U, true, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace));
    }
    return;
}
#endif

#if ((defined(__DAV_C220_CUBE__)))
__aicore__ inline void linalg_detGET_TILING_DATA_WITH_STRUCTPhase24(GM_ADDR &tiling, GM_ADDR &workspace) {

    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    GM_ADDR ws = GetUserWorkspace(workspace);
    uint64_t span = (2ULL * td.n * td.stride + 512U * 24U + 32U + 256U) * 4U;
    GM_ADDR neg = ws + span;
    GM_ADDR u = neg + 32768U;
    GM_ADDR flags = u + 65536U;
    LuCube op;
    op.RunRange(ws, neg, u, flags, tiling + 64U, td.n, td.stride, 1U);
    LuPublish(flags + (13U + GetBlockIdx()) * 64U, 0x6ADBD431U);
}
#endif

#if ((defined(__DAV_C220_CUBE__)))
__aicore__ inline void linalg_detGET_TILING_DATA_WITH_STRUCTPhase25(GM_ADDR &tiling, GM_ADDR &workspace) {

    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
    GM_ADDR ws = GetUserWorkspace(workspace);
    uint64_t span = (2ULL * td.n * td.stride + 512U * 24U + 32U + 256U) * 4U;
    GM_ADDR neg = ws + span;
    GM_ADDR u = neg + 32768U;
    GM_ADDR flags = u + 65536U;
    LuCube op;
    op.RunTail208(ws, neg, u, flags, tiling + 64U, td.n, td.stride, 1U);
    LuPublish(flags + (13U + GetBlockIdx()) * 64U, 0x6ADBD431U);
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase27(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(7);
    CyclicG5PriorityRelay<LdetCyclicSettings46_1<5U, true, true, 40U, true, true, 32U, false, true, false, true, true,
                                                 true, false, true, true, true, true>>
        op;
    op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase28(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(7);
    CyclicRecycled45<LdetCyclicSettings46_1<5U, true, true, 40U, true, true, 32U, false, true, false, true, true, true,
                                            false, true, true, true, true>>
        op;
    op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase29(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(7);
    CyclicG4PriorityRelay<LdetCyclicSettings46_1<4U, true, false, 40U, true, true, 32U, false, true, true, true, true,
                                                 true, true, true, true, true, false, false, true, true>>
        op;
    op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase30(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(5);
    CyclicN256ReuseFirstTail<LdetCyclicSettings46_4<6U, true, true, 104U, true, true, 32U, false, true, false, false,
                                                    true, true, false, true, true, true, true, true>>
        op;
    op.Run<false, true>(A, det, GetUserWorkspace(workspace));
    return;
}
#endif

#if ((!((defined(__DAV_C220_CUBE__))) && (defined(__DAV_C220_VEC__))))
__aicore__ inline void linalg_detICachePreLoadPhase32(GM_ADDR &A, GM_ADDR &det, GM_ADDR &workspace) {

    AscendC::ICachePreLoad(5);
    CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 200U, true, false, false, true, true, false, false, true,
                      false, true, 128U, true>
        op;
    op.Run(A, det, GetUserWorkspace(workspace));
    return;
}
#endif

__aicore__ constexpr uint32_t LdetResolveRoute(uint32_t selected) {
    if (selected == 64U)
        return 13U;
    if (selected == 65U)
        return 33U;
    if (selected == 85U)
        return 58U;
    return selected;
}

#if (defined(__DAV_C220_CUBE__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline void RunCubeRoute(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH_SELECTED == 105U || LD_BATCH_SELECTED == 106U) {
        return linalg_detDataCopyPhase14(A, tiling);
    } else if constexpr (LD_BATCH_SELECTED == 64U || LD_BATCH_SELECTED == 65U) {
        return linalg_detDataCopyPhase18<LD_BATCH_SELECTED>();
    } else if constexpr (LD_BATCH_SELECTED == 77U) {
        return linalg_detDataCopyPhase16(A, tiling);
    } else if constexpr (LD_BATCH_SELECTED >= 98U) {
        return linalg_detDataCopyPhase17(A, tiling);
    } else if constexpr (LD_BATCH_SELECTED >= 64U) {
        return linalg_detDataCopyPhase3(A, tiling);
    } else if constexpr (LD_BATCH == 59U) {
        return;
    } else if constexpr (LD_BATCH == 20U) {
        LuN256BatchCube(workspace, tiling);
    } else if constexpr (LD_BATCH == 24U) {
        LuWideBatchCube(workspace, tiling);
    } else if constexpr (LD_BATCH == 21U) {
        if (GetBlockIdx() >= 4U)
            return;
        linalg_detGET_TILING_DATA_WITH_STRUCTPhase24(tiling, workspace);
    } else if constexpr (LD_BATCH == 22U) {
        if (GetBlockIdx() >= 4U)
            return;
        linalg_detGET_TILING_DATA_WITH_STRUCTPhase25(tiling, workspace);
    }
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryRagged106(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH_SELECTED == 106U) {
        AscendC::ICachePreLoad(1);
#if defined(LDET_RAGGED_EXPLICIT_SCALAR_ARGUMENTS)
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        auto *args = reinterpret_cast<__gm__ volatile uint64_t *>(get_para_base());
        GM_ADDR input = reinterpret_cast<GM_ADDR>(args[1U]);
        GM_ADDR output = reinterpret_cast<GM_ADDR>(args[2U]);
        auto *words = reinterpret_cast<__gm__ volatile uint32_t *>(args[4U]);
        const uint32_t inputN = words[1U];
#else
        GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
        GM_ADDR input = A;
        GM_ADDR output = det;
        const uint32_t inputN = td.n;
#endif
        {
            linalg_detDataCopyPadPhase22<DT_A>(input, inputN, output);
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryRagged105Or107(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH_SELECTED == 105U || LD_BATCH_SELECTED == 107U) {
        AscendC::ICachePreLoad(2);
#if defined(LDET_RAGGED_EXPLICIT_SCALAR_ARGUMENTS)
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        auto *args = reinterpret_cast<__gm__ volatile uint64_t *>(get_para_base());
        GM_ADDR input = reinterpret_cast<GM_ADDR>(args[1U]);
        GM_ADDR output = reinterpret_cast<GM_ADDR>(args[2U]);
        auto *words = reinterpret_cast<__gm__ volatile uint32_t *>(args[4U]);
        const LinalgDetTilingData local{words[0U], words[1U], words[2U], 1U, 0U, 1U};
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(input, output, nullptr, local, false, true, false);
        op.template ProcessRaggedMidRecurrenceEntry<2U>();
#else
        GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
        const LinalgDetTilingData local{td.batchCount, td.n, td.stride, td.matPerCore, td.isLarge, td.groupSize};
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(A, det, workspace, local, false, true, false);
        op.template ProcessRaggedMidRecurrenceEntry<2U>();
#endif
        {
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryN256DirectGroup1(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 103U) {
        {
            linalg_detICachePreLoadPhase27(A, det, workspace);
            return true;
        }
    }
    if constexpr (LD_BATCH == 51U) {
        {
            linalg_detICachePreLoadPhase28(A, det, workspace);
            return true;
        }
    }
    if constexpr (LD_BATCH == 52U) {
        CyclicN256Inplace120<3U> op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));
        {
            return true;
        }
    }
    return false;
}
#endif
