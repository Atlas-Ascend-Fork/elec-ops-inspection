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
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t D, uint32_t I, class B0, class B1, class C0, class C1, class C2, class C3, class C4>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitivec49e28b728(C0& piped_, C1& x_, C2& slots_, C3& grp_, C4& baseRow_, B0& off, B1& row)
{
    constexpr uint32_t CurTM = (I < 6U) ? 128U : 64U;
    if constexpr (I < 5U) {
        piped_.template Prefetch<I + 1U, 128U>(x_[(row + 128U) * D]);
    } else if constexpr (I == 5U) {
        const uint64_t pfRow = (grp_ < 8U) ? (row + 128U) : baseRow_;
        piped_.template Prefetch<6U, 64U>(x_[pfRow * D]);
    }
    piped_.template Compute<I, CurTM>(slots_[off]);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t D, uint32_t TM, uint32_t Iterations, auto N, auto InSlots, auto OutSlots, uint32_t I,
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    uint32_t CurTM = TM, class B0, class B1, class B2, class B3, class C0, class C1>
#else
    uint32_t CurTM = TM, bool Single = false, class B0, class B1, class B2, class B3, class C0,
    class C1>
#endif
__aicore__ inline __attribute__((always_inline)) void SharedPrimitiveac2daf19f8(C0& y_, C1& col_, B0& begin,
                                                                                B1& op, B2& out, B3& row)
{
    constexpr event_t vEv = VECTOR_SLOT_EVENT<I % InSlots>;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    constexpr uint32_t CurNR = CurTM / 2U;
#else
    constexpr uint32_t CurNR = Single ? CurTM : CurTM / 2U;
#endif
    constexpr uint32_t CurCap = CurNR * N;
    if constexpr (D == N) {
        DataCopy(y_[(row + begin) * D], out, CurCap);
    } else {
        DataCopy(y_[(row + begin) * D + col_], out, op);
    }
    if constexpr (I + OutSlots < Iterations) {
        SetFlag<HardEvent::MTE3_V>(vEv);
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t Iterations, auto N, auto InSlots, uint32_t I, uint32_t CurTM = TM,
          class B0, class B1, class B2, class B3, class B4, class B5>
#else
template <typename T, uint32_t TM, uint32_t Iterations, auto N, auto InSlots, uint32_t I, uint32_t CurTM = TM,
          bool Single = false, class B0, class B1, class B2, class B3, class B4, class B5>
#endif
__aicore__ inline __attribute__((always_inline)) void SharedPrimitivee3102f6fcd(B0& b1, B1& b2, B2& curInput,
                                                                                B3& g, B4& rp, B5& u)
{
    constexpr event_t vEv = VECTOR_SLOT_EVENT<I % InSlots>;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    constexpr uint32_t CurNR = CurTM / 2U;
#else
    constexpr uint32_t CurNR = Single ? CurTM : CurTM / 2U;
#endif
    constexpr uint32_t CurCap = CurNR * N;
    constexpr uint64_t mask = 0;
    if constexpr (N != 128U) {
        SetVectorMask<T, MaskMode::NORMAL>(0, ~0ULL);
    }
    Add<T, false>(g, curInput, b1, mask, static_cast<uint8_t>(CurNR), rp);
    Add<T, false>(u, curInput[N], b2, mask, static_cast<uint8_t>(CurNR), rp);
    if constexpr (I + InSlots < Iterations) {
        SetFlag<HardEvent::V_MTE2>(vEv);
    }
    static_assert(CurCap % 128U == 0U && CurCap / 128U <= 255U);
}
#endif
template <typename T, uint32_t FixedM, class B0, class B1, class B2, class B3, class B4, class B5, class B6,
          class B7, class B8, class B9, class B10, class B11, class B12>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitive3eb9fd04b0(B0& b1, B1& b2, B2& d, B3& gb1, B4& gb2, B5& gw1, B6& gw2, B7& gx, B8& gy, B9& w1,
                          B10& w2, B11& x, B12& y)
{
    gx.SetGlobalBuffer((__gm__ T*)x, FixedM * d);
    gy.SetGlobalBuffer((__gm__ T*)y, FixedM * d);
    gw1.SetGlobalBuffer((__gm__ T*)w1, d * d);
    gw2.SetGlobalBuffer((__gm__ T*)w2, d * d);
    gb1.SetGlobalBuffer((__gm__ T*)b1, d);
    gb2.SetGlobalBuffer((__gm__ T*)b2, d);
}
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t TM, uint32_t D, uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive02eeb40300(B0& a1, B1& a2_0, B2& a2_1)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr uint32_t HM = TM / 2U;
    if constexpr (I > 0U) {
        WaitFlag<HardEvent::MTE2_MTE1>(slot);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        LoadData2dParams al;
        al.repeatTimes = D / 16U;
        al.srcStride = CurTM / 16U;
#pragma unroll
        for (uint32_t mb = 0; mb < HM / 16U; ++mb) {
            LoadData(a2_0[mb * D * 16U], a1[mb * 256U], al);
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
#pragma unroll
        for (uint32_t mb = 0; mb < HM / 16U; ++mb) {
            LoadData(a2_1[mb * D * 16U], a1[(mb + HM / 16U) * 256U], al);
        }
        SetFlag<HardEvent::MTE1_MTE2>(slot);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
    }
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t FixedD, class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7,
          class B8, class B9>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitivee882d043f2(B0& col, B1& d, B2& gb1, B3& gb2, B4& gw1, B5& gw2, B6& hb1, B7& hb2, B8& hw1,
                          B9& hw2)
{
    DataCopy(hw1, gw1[static_cast<uint64_t>(col) * d], 16U * FixedD);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(hw2, gw2[static_cast<uint64_t>(col) * d], 16U * FixedD);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    DataCopy(hb1, gb1[col], 16U);
    DataCopy(hb2, gb2[col], 16U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
}
#endif
template <typename T, uint32_t D, uint32_t Cores, auto Plane, auto M, class B0, class B1, class B2, class B3,
          class B4, class B5, class B6, class C0, class C1, class C2, class C3, class C4, class C5, class C6>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitive88fe1d65b1(C0& x_, C1& w1_, C2& w2_, C3& b1_, C4& b2_, C5& y_, C6& slots_, B0& b1, B1& b2,
                          B2& w1, B3& w2, B4& workspace, B5& x, B6& y)
{
    x_.SetGlobalBuffer((__gm__ T*)x, M * D);
    y_.SetGlobalBuffer((__gm__ T*)y, M * D);
    w1_.SetGlobalBuffer((__gm__ T*)w1, D * D);
    w2_.SetGlobalBuffer((__gm__ T*)w2, D * D);
    b1_.SetGlobalBuffer((__gm__ T*)b1, D);
    b2_.SetGlobalBuffer((__gm__ T*)b2, D);
    slots_.SetGlobalBuffer((__gm__ T*)workspace, 2U * Plane * 4U * Cores);
    SetAtomicNone();
}

template <typename T, class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7,
          class B8, class B9, class B10, class B11, class B12>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitivefa859e08f7(B0& b1, B1& b2, B2& d, B3& gb1, B4& gb2, B5& gw1, B6& gw2, B7& gx, B8& gy, B9& w1,
                          B10& w2, B11& x, B12& y)
{
    gx.SetGlobalBuffer((__gm__ T*)x);
    gy.SetGlobalBuffer((__gm__ T*)y);
    gw1.SetGlobalBuffer((__gm__ T*)w1, d * d);
    gw2.SetGlobalBuffer((__gm__ T*)w2, d * d);
    gb1.SetGlobalBuffer((__gm__ T*)b1, d);
    gb2.SetGlobalBuffer((__gm__ T*)b2, d);
}

template <typename T, class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7,
          class B8, class B9, class B10, class B11>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitive9aae833572(B0& b1, B1& b2, B2& gb1, B3& gb2, B4& gw1, B5& gw2, B6& gx, B7& gy, B8& w1, B9& w2,
                          B10& x, B11& y)
{
    gx.SetGlobalBuffer((__gm__ T*)x);
    gw1.SetGlobalBuffer((__gm__ T*)w1);
    gw2.SetGlobalBuffer((__gm__ T*)w2);
    gb1.SetGlobalBuffer((__gm__ T*)b1);
    gb2.SetGlobalBuffer((__gm__ T*)b2);
    gy.SetGlobalBuffer((__gm__ T*)y);
}

template <bool First>
__aicore__ inline __attribute__((always_inline)) void
SharedKernel0c144b8d11(const LocalTensor<float>& ag, const LocalTensor<float>& au,
                       const LocalTensor<float>& sg, const LocalTensor<float>& su,
                       const LocalTensor<float>& pg, const LocalTensor<float>& pu,
                       const LocalTensor<float>& fx, const LocalTensor<float>& fw1,
                       const LocalTensor<float>& fw2, uint32_t ka, uint32_t k, uint32_t cols)
{
    const uint32_t width = ka - k < 64U ? ka - k : 64U;
    const uint8_t stride = static_cast<uint8_t>(ka / 8U);
    const BinaryRepeatParams rp{1, 1, 1, stride, 0, stride};
    Mul(pg[k], fx[k], fw1[k], static_cast<uint64_t>(width), static_cast<uint8_t>(cols), rp);
    Mul(pu[k], fx[k], fw2[k], static_cast<uint64_t>(width), static_cast<uint8_t>(cols), rp);
    PipeBarrier<PIPE_V>();
    if constexpr (First) {
        WholeReduceSum(ag, pg[k], width, cols, 1, 1, stride);
        WholeReduceSum(au, pu[k], width, cols, 1, 1, stride);
        PipeBarrier<PIPE_V>();
    } else {
        WholeReduceSum(sg, pg[k], width, cols, 1, 1, stride);
        WholeReduceSum(su, pu[k], width, cols, 1, 1, stride);
        PipeBarrier<PIPE_V>();
        Add(ag, ag, sg, cols);
        Add(au, au, su, cols);
        PipeBarrier<PIPE_V>();
    }
}
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, class A0, class A1, class A2, class A3, class A4,
          class A5, class A6, class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
SharedKerneld5184a4a53(A0& w1, A1& w2, A2& x0, A3& b1, A4& a1, A5& b2_0, A6& b2_1, A7& hap, A8& bp)
{
    constexpr uint32_t HM = TM / 2U;
    DataCopy(a1, x0, hap);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    DataCopy(b1, w1, bp);
    DataCopy(b1[N * D / 2U], w1[(N / 2U) * D], bp);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    DataCopy(a1[HM * D], x0[HM * D], hap);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    DataCopy(b1[N * D], w2, bp);
    DataCopy(b1[N * D + N * D / 2U], w2[(N / 2U) * D], bp);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    LocalTensor<T> a2_0(TPosition::A2, 0U, HM * D);
    LocalTensor<T> a2_1(TPosition::A2, HM * D * sizeof(T), HM * D);
    LoadData2dParams al;
    al.repeatTimes = D / 16U;
    al.srcStride = HM / 16U;
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
#pragma unroll
    for (uint32_t mb = 0; mb < HM / 16U; ++mb) {
        LoadData(a2_0[mb * D * 16U], a1[mb * 256U], al);
    }
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    LoadData2dParams bl;
    bl.repeatTimes = (N * D / 2U) / 256U;
    bl.srcStride = 1;
    LoadData(b2_0, b1, bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    LoadData(b2_1, b1[N * D / 2U], bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
#pragma unroll
    for (uint32_t mb = 0; mb < HM / 16U; ++mb) {
        LoadData(a2_1[mb * D * 16U], a1[HM * D + mb * 256U], al);
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t D, uint32_t I, uint32_t CurTM, auto A, auto SplitL0B>
__aicore__ inline __attribute__((always_inline)) void SharedKernel23bd47df61(const GlobalTensor<T>& x)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    if constexpr (I >= 2U || (SplitL0B && I == 1U)) {
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
    }
    LocalTensor<T> a1(TPosition::A1, (I % 2U) * A * sizeof(T), CurTM * D);
    const Nd2NzParams ap{1, CurTM, D, 0, D, CurTM, 1, 0};
    DataCopy(a1, x, ap);
    SetFlag<HardEvent::MTE2_MTE1>(slot);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t D, uint32_t N, uint32_t I, class B0, class B1, class B2, class B3>
__aicore__ inline __attribute__((always_inline)) void SharedKerneldc2a2bdd73(B0& b1, B1& b2_0, B2& b2_1,
                                                                             B3& bl)
{
    constexpr uint32_t nextB1Off = (I % 2U == 1U) ? 0U : (N * D);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
    LoadData(b2_0, b1[nextB1Off], bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    LoadData(b2_1, b1[nextB1Off + N * D / 2U], bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t TM, uint32_t D, uint32_t I, uint32_t CurTM = TM, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive338d40bc0d(B0& a1, B1& a2_0)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    WaitFlag<HardEvent::MTE2_MTE1>(slot);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    LoadData2dParams al;
    al.repeatTimes = D / 16U;
    al.srcStride = CurTM / 16U;
#pragma unroll
    for (uint32_t mb = 0; mb < CurTM / 16U; ++mb) {
        LoadData(a2_0[mb * D * 16U], a1[mb * 256U], al);
    }
    SetFlag<HardEvent::MTE1_MTE2>(slot);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t N, class B0, class B1, class B2, class B3, class B4>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive54529556e1(B0& a2_0, B1& b2_0, B2& b2_1,
                                                                                B3& c0, B4& dst)
{
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    MmadParams mad;
    mad.m = 32U;
    mad.n = N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    Mmad(c0, a2_0, b2_0, mad);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    Mmad(c0[32U * N], a2_0, b2_1, mad);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = 32U;
    fix.srcStride = 32U;
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c0, fix);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <uint32_t D, uint32_t N, class B0, class B1, class B2, class B3, class B4>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive754371e923(B0& a1, B1& a2_1, B2& b2_0,
                                                                                B3& b2_1, B4& c1)
{
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    LoadData2dParams al;
    al.repeatTimes = D / 16U;
    al.srcStride = 48U / 16U;
#pragma unroll
    for (uint32_t mb = 0; mb < 48U / 16U; ++mb) {
        LoadData(a2_1[mb * D * 16U], a1[mb * 256U], al);
    }
    SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
    MmadParams mad;
    mad.m = 48U;
    mad.n = N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    Mmad(c1, a2_1, b2_0, mad);
    Mmad(c1[48U * N], a2_1, b2_1, mad);
}
#endif
template <uint32_t TM, uint32_t D, uint32_t I, uint32_t CurTM = TM, bool Signal = false, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void SharedLoadTileA(B0& a1, B1& a2)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    if constexpr (I >= 2U) {
        WaitFlag<HardEvent::M_MTE1>(slot);
    }
    LoadData2dParams al;
    al.repeatTimes = D / 16U;
    al.srcStride = CurTM / 16U;
#pragma unroll
    for (uint32_t mb = 0; mb < CurTM / 16U; ++mb) {
        LoadData(a2[mb * D * 16U], a1[mb * 256U], al);
    }
    if constexpr (Signal) {
        SetFlag<HardEvent::MTE1_M>(slot);
    }
}
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t N, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive5ed408cbed(B0& c2, B1& dst)
{
    SetFlag<HardEvent::M_FIX>(EVENT_ID2);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID2);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = 48U;
    fix.srcStride = 48U;
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c2, fix);
    SetFlag<HardEvent::FIX_M>(EVENT_ID2);
}
#endif
