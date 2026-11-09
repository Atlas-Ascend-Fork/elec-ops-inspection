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
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryN256DirectGroup2(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 53U) {
        CyclicN256Inplace120<3U, true, false, true, true, true, 3U, false, true, true, true> op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));
        {
            return true;
        }
    }
    if constexpr (LD_BATCH == 54U) {
        CyclicN256Inplace120<3U, true, false, true, true, true, 3U, false, true, true> op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));
        {
            return true;
        }
    }
    if constexpr (LD_BATCH == 55U) {
        {
            linalg_detICachePreLoadPhase29(A, det, workspace);
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryN256DirectGroup3(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 104U) {
        {
            linalg_detICachePreLoadPhase20(A, det, workspace);
            return true;
        }
    }
    if constexpr (LD_BATCH == 37U) {
        {
            linalg_detICachePreLoadPhase21(A, det, workspace);
            return true;
        }
    }
    if constexpr (LD_BATCH == 27U) {
        {
            linalg_detICachePreLoadPhase30(A, det, workspace);
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryFixedDirectGroup1(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 50U) {

        const uint32_t task = static_cast<uint32_t>(GetBlockIdx());
        if ((task & 1U) != 0U) {
            return true;
        }
        constexpr LinalgDetTilingData td{4U, 64U, 72U, 1U, 0U, 1U};
        KernelLinalgDet<DT_A, 11U> op;
        op.Init(A, det, workspace, td, false, false, false, true, true, task >> 1U);
        op.ProcessN64OriginalWrapper();

        return true;
    }
    if constexpr (LD_BATCH == 49U) {

        GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
        KernelLinalgDet<DT_A, 11U, false, false, false, false, true> op;
        op.Init(A, det, workspace, td);
        op.Process();

        return true;
    }
    if constexpr (LD_BATCH == 48U) {

        AscendC::ICachePreLoad(2);
        KernelLinalgDet<DT_A, 9U> op;
        op.InitN15FastPairOnly(A, det);
        op.ProcessN15FastPairEntry(tiling);
        {
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryFixedDirectGroup2(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 47U) {

        GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
        const LinalgDetTilingData local{td.batchCount, td.n, td.stride, td.matPerCore, td.isLarge, td.groupSize};
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(A, det, workspace, local, false, true, false);
        op.ProcessRaggedMidRecurrenceEntry();

        return true;
    }

    if constexpr (LD_BATCH == 45U) {

        const LinalgDetTilingData fixed{8U, 18U, 24U, 1U, 0U, 1U};
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(A, det, workspace, fixed, true, true, true);
        op.ProcessN18B8FixedEntry();

        return true;
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryFixedDirectGroup3(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 43U || LD_BATCH == 44U) {

        constexpr uint32_t fixedBatch = LD_BATCH == 43U ? 6U : 8U;
        const LinalgDetTilingData fixed{fixedBatch, 19U, 24U, 1U, 0U, 1U};
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(A, det, workspace, fixed, true, true, true);
        op.ProcessN19B8FixedEntry();

        return true;
    }
    if constexpr (LD_BATCH == 62U) {

        GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, td, tiling);
        const LinalgDetTilingData local{td.batchCount, 16U, 24U, 1U, 0U, 1U};
        KernelLinalgDet<DT_A, 6U> op;
        op.Init(A, det, workspace, local, true, false, true);
        op.ProcessN16BatchTail4Entry();

        return true;
    }
    if constexpr (LD_BATCH == 42U) {

        linalg_detGET_TILING_DATA_WITH_STRUCTPhase6<DT_A>(A, det, tiling, workspace);

        return true;
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryFixedDirectGroup4(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);

    if constexpr (LD_BATCH == 30U) {

        CyclicRecycled45<LdetCyclicSettings46_1<4U, true, false, 40U, true, true, 32U, false, true, true, true, true,
                                                true, true, true, true, true>>
            op;
        op.Run<false, true>(A, det, GetUserWorkspace(workspace));

        return true;
    }
    if constexpr (LD_BATCH == 33U) {

        constexpr LinalgDetTilingData fixed_tiling{1U, 32U, 40U, 1U, 0U, 1U};
        KernelLinalgDet<DT_A, 10U> op;
        op.Init(A, det, workspace, fixed_tiling, true);
        op.template ProcessB1Combined<32U>();

        return true;
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryN16PlaneRoute(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 34U) {
        AscendC::ICachePreLoad(2);
#if defined(LDET_N16_EXPLICIT_SCALAR_ARGUMENTS)
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        auto *paramWords = reinterpret_cast<__gm__ volatile uint64_t *>(get_para_base());
        GM_ADDR tinyInput = reinterpret_cast<GM_ADDR>(paramWords[1U]);
        GM_ADDR tinyOutput = reinterpret_cast<GM_ADDR>(paramWords[2U]);
#else
        GM_ADDR tinyInput = A;
        GM_ADDR tinyOutput = det;
#endif
        KernelLinalgDet<DT_A, 9U> op;
        op.InitN16BlockOnly(tinyInput, tinyOutput);
        op.ProcessN16PlaneEntry(tiling);
        {
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledSpecialGroup1(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                             const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 32U) {
        {
            linalg_detRunUB216BatchPhase1<DT_A>(A, det, tiling_data);
            return true;
        }
    }
    if constexpr (LD_BATCH == 56U) {
        {
            linalg_detRunPhase15(A, det, workspace, tiling_data);
            return true;
        }
    }
    if constexpr (LD_BATCH == 41U) {
        if (tiling_data.n == 208U) {
            CyclicRowLUMedium<3U, true, false, 40U, true, true, 32U, 208U, true, false, false, true, true, true> op;
            op.Run(A, det, GetUserWorkspace(workspace));
        }
        {
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledSpecialGroup2(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                             const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 63U) {
        {
            linalg_detICachePreLoadPhase9(A, det, workspace);
            return true;
        }
    }

    if constexpr (LD_BATCH == 60U) {
        {
            linalg_detICachePreLoadPhase32(A, det, workspace);
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledSpecialGroup3(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                             const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 59U) {
        {
            linalg_detRunPhase23(A, det, workspace, tiling_data);
            return true;
        }
    }
    if constexpr (LD_BATCH == 58U) {
        {
            linalg_detRunPhase7(A, det, workspace, tiling_data);
            return true;
        }
    }
    if constexpr (LD_BATCH == 57U) {
        {
            linalg_detRunPhase10(A, det, workspace, tiling_data);
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledSpecialGroup4(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                             const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 40U) {
        {
            linalg_detRunPhase11(A, det, workspace, tiling_data);
            return true;
        }
    }

    if constexpr (LD_BATCH == 38U) {
        {
            linalg_detRunPhase8(A, det, workspace, tiling_data);
            return true;
        }
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledSpecialGroup5(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                             const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);

    if constexpr (LD_BATCH == 35U) {
        {
            linalg_detRunPhase5(A, det, workspace, tiling_data);
            return true;
        }
    }

    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledTailGroup1(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                          const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 31U) {

        linalg_detGetBlockNumPhase2(A, det, workspace, tiling_data);

        return true;
    }
    if constexpr (LD_BATCH == 27U) {

        CyclicRowLU<8U, false, false, 40U, true, true, 32U, false, false, true, true, true> op;
        op.Run(A, det, GetUserWorkspace(workspace));

        return true;
    }

    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledTailGroup2(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                          const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);

    if constexpr (LD_BATCH == 30U) {

        CyclicRowLU<4U, true, false, 40U, true, true, 32U, false, true, true, true, true, true, false, true, true, true>
            op;
        op.Run(A, det, GetUserWorkspace(workspace));

        return true;
    }

    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t LD_BATCH_SELECTED>
__aicore__ inline bool TryTiledTailGroup3(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling,
                                          const LinalgDetTilingData &tiling_data) {
    constexpr uint32_t LD_BATCH = LdetResolveRoute(LD_BATCH_SELECTED);
    if constexpr (LD_BATCH == 20U) {

        LuN256BatchVector(A, det, workspace, tiling);

        return true;
    }
    if constexpr (LD_BATCH == 24U) {

        LuWideBatchVector(A, det, workspace, tiling);

        return true;
    }
    if constexpr (LD_BATCH == 21U) {

        if (GetBlockIdx() >= 8U) {
            return true;
        }
        RunWorkspaceVariant60<false, DT_A>(A, det, workspace, tiling_data);

        return true;
    }
    return false;
}
#endif

#if (!defined(__DAV_C220_CUBE__) && defined(__DAV_C220_VEC__))
template <typename DT_A, uint32_t N>
__aicore__ inline void RunUB216Single(GM_ADDR A, GM_ADDR det, const LinalgDetTilingData &td) {
    KernelUB216<DT_A, N> op;
    op.Run(A, det, td);
}
#endif
