/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T> class ProjectionStreamCase3 {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
    __aicore__ inline __attribute__((always_inline)) void
    RunStaged97785d5(A0& ws, A1& off, A2& wg, A3& wu, A4& a2, A5& b2Local, A6& cg, A7& cu);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void RunStage52cd47f9(A0& row, A1& col, A2& gb1, A3& gb2,
                                                                           A4& gy, A5& ws, A6& off);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10>
    __aicore__ inline __attribute__((always_inline)) void RunStage8bb7089f(A0& workspace, A1& core, A2& row,
                                                                           A3& col, A4& gx, A5& gw1, A6& gw2,
                                                                           A7& gb1, A8& gb2, A9& gy, A10& ws);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR workspace);

private:
    static constexpr uint32_t M = 64U;
    static constexpr uint32_t N = 128U;
    static constexpr uint32_t D = 256U;
    static constexpr uint32_t Cap = M * N;
};
#endif
}

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
__aicore__ inline __attribute__((always_inline)) void
ProjectionStreamCase3<T>::RunStaged97785d5(A0& ws, A1& off, A2& wg, A3& wu, A4& a2, A5& b2Local, A6& cg,
                                           A7& cu)
{
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    LoadData2dParams bl;
    bl.repeatTimes = N * D / 256U;
    bl.srcStride = 1;
    LoadData(b2Local, wg, bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    MmadParams mad;
    mad.m = M;
    mad.n = N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    Mmad(cg, a2, b2Local, mad);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    FixpipeParamsV220 fix;
    fix.nSize = N;
    fix.mSize = M;
    fix.srcStride = M;
    fix.dstStride = N;
    fix.quantPre = QuantMode_t::F322F16;
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    Fixpipe<T, float, CFG_ROW_MAJOR>(ws[off], cg, fix);
    CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    LoadData(b2Local, wu, bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
    Mmad(cu, a2, b2Local, mad);
    SetFlag<HardEvent::M_FIX>(EVENT_ID1);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
    Fixpipe<T, float, CFG_ROW_MAJOR>(ws[off + Cap], cu, fix);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
ProjectionStreamCase3<T>::RunStage52cd47f9(A0& row, A1& col, A2& gb1, A3& gb2, A4& gy, A5& ws, A6& off)
{
    LocalTensor<T> inputG(TPosition::VECIN, 0U, Cap);
    LocalTensor<T> inputU(TPosition::VECIN, Cap * sizeof(T), Cap);
    LocalTensor<T> biasG(TPosition::VECIN, 33024U, N);
    LocalTensor<T> biasU(TPosition::VECIN, 33536U, N);
    LocalTensor<T> g(TPosition::VECCALC, 66048U, Cap);
    LocalTensor<T> u(TPosition::VECCALC, 82432U, Cap);
    LocalTensor<T> sig(TPosition::VECCALC, 131328U, Cap);
    LocalTensor<T> out(TPosition::VECOUT, 0U, Cap);
    DataCopy(biasG, gb1[col], N);
    DataCopy(biasU, gb2[col], N);
    CrossCoreWaitFlag(0x6);
    DataCopy(inputG, ws[off], Cap);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    constexpr uint64_t mask = 0;
    const BinaryRepeatParams biasRp{1, 1, 1, 8, 8, 0};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams ur{1, 1, 8, 8};
    Add<T, false>(g, inputG, biasG, mask, M, biasRp);
    Muls<T, false>(sig, g, static_cast<T>(-1), mask, Cap / 128U, ur);
    Exp<T, false>(sig, sig, mask, Cap / 128U, ur);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, Cap / 128U, ur);
    Div<T, false>(sig, g, sig, mask, Cap / 128U, br);
    CrossCoreWaitFlag(0x7);
    DataCopy(inputU, ws[off + Cap], Cap);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Add<T, false>(u, inputU, biasU, mask, M, biasRp);
    Mul<T, false>(out, sig, u, mask, Cap / 128U, br);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const DataCopyParams outputCopy{M, N / 16U, 0, (D - N) / 16U};
    DataCopy(gy[row * D + col], out, outputCopy);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10>
__aicore__ inline __attribute__((always_inline)) void
ProjectionStreamCase3<T>::RunStage8bb7089f(A0& workspace, A1& core, A2& row, A3& col, A4& gx, A5& gw1,
                                           A6& gw2, A7& gb1, A8& gb2, A9& gy, A10& ws)
{
    ws.SetGlobalBuffer((__gm__ T*)workspace, 16U * 2U * Cap);
    const uint32_t off = core * 2U * Cap;
    SetAtomicNone();
    if ASCEND_IS_AIC {
        LocalTensor<T> ax(TPosition::A1, 0U, M * D), wg(TPosition::B1, M * D * sizeof(T), N * D),
            wu(TPosition::B1, (M * D + N * D) * sizeof(T), N * D);
        LocalTensor<T> a2(TPosition::A2, 0U, M * D);
        LocalTensor<T> b2Local(TPosition::B2, 0U, N * D);
        LocalTensor<float> cg(TPosition::CO1, 0U, Cap);
        LocalTensor<float> cu(TPosition::CO1, Cap * sizeof(float), Cap);
        const Nd2NzParams ap{1, M, D, 0, D, M, 1, 0};
        const Nd2NzParams bp{1, N, D, 0, D, N, 1, 0};
        DataCopy(ax, gx[row * D], ap);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        DataCopy(wg, gw1[col * D], bp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        DataCopy(wu, gw2[col * D], bp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        LoadData2dParams al;
        al.repeatTimes = D / 16U;
        al.srcStride = M / 16U;
#pragma unroll
        for (uint32_t mb = 0; mb < M / 16U; ++mb) {
            LoadData(a2[mb * D * 16U], ax[mb * 256U], al);
        }
        RunStaged97785d5(ws, off, wg, wu, a2, b2Local, cg, cu);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x7);
    }
    if ASCEND_IS_AIV {
        SetMaskNorm();
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
        RunStage52cd47f9(row, col, gb1, gb2, gy, ws, off);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
__aicore__ inline void ProjectionStreamCase3<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                     GM_ADDR b2, GM_ADDR y, GM_ADDR workspace)
{
    const uint32_t core = GetBlockIdx();
    const uint32_t row = (core / 2U) * M;
    const uint32_t col = (core % 2U) * N;
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    GlobalTensor<T> ws;
    gx.SetGlobalBuffer((__gm__ T*)x, 512U * D);
    gy.SetGlobalBuffer((__gm__ T*)y, 512U * D);
    gw1.SetGlobalBuffer((__gm__ T*)w1, D * D);
    gw2.SetGlobalBuffer((__gm__ T*)w2, D * D);
    gb1.SetGlobalBuffer((__gm__ T*)b1, D);
    gb2.SetGlobalBuffer((__gm__ T*)b2, D);
    RunStage8bb7089f(workspace, core, row, col, gx, gw1, gw2, gb1, gb2, gy, ws);
    PipeBarrier<PIPE_ALL>();
}

#endif

}
