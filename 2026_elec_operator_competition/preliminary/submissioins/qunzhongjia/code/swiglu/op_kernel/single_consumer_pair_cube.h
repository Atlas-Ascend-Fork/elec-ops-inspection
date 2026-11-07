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
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
class SingleConsumerPairCube : public PairCubeCommon<T, TM, D, N, Iterations> {
public:
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM, uint32_t TM4 = TM0, class A0,
              class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
    __aicore__ inline __attribute__((always_inline)) void
    PrepareAllStageafe89876(A0& w1, A1& w2, A2& x0, A3& x1, A4& x2, A5& x3, A6& x4, A7& a1, A8& b1, A9& a2);
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM, uint32_t TM4 = TM0>
    __aicore__ inline void PrepareAll(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
                                      const GlobalTensor<T>& x0, const GlobalTensor<T>& x1,
                                      const GlobalTensor<T>& x2, const GlobalTensor<T>& x3,
                                      const GlobalTensor<T>& x4);
    template <uint32_t I, uint32_t CurTM = TM, class A0>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage093b73a7(A0& dst);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage38e86bde(A0& dst, A1& a1, A2& b2_0,
                                                                               A3& b2_1);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage0fee0c01(A0& dst, A1& a1, A2& b2_0,
                                                                               A3& b2_1);
    template <uint32_t I, uint32_t CurTM = TM, class A0>
    __aicore__ inline __attribute__((always_inline)) void ComputeStaged537ca52(A0& dst);
    template <uint32_t I, uint32_t CurTM = TM> __aicore__ inline void Compute(const GlobalTensor<T>& dst);
    __aicore__ inline void Drain();

private:
    using Common = PairCubeCommon<T, TM, D, N, Iterations>;
};
#endif
}

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3, uint32_t TM4, class A0, class A1, class A2,
          class A3, class A4, class A5, class A6, class A7, class A8, class A9>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerPairCube<T, TM, D, N, Iterations>::PrepareAllStageafe89876(A0& w1, A1& w2, A2& x0, A3& x1,
                                                                         A4& x2, A5& x3, A6& x4, A7& a1,
                                                                         A8& b1, A9& a2)
{
    const Nd2NzParams ap0{1, TM0, D, 0, D, TM0, 1, 0};
    const Nd2NzParams ap1{1, TM1, D, 0, D, TM1, 1, 0};
    const Nd2NzParams ap2{1, TM2, D, 0, D, TM2, 1, 0};
    const Nd2NzParams bp{1, N, D, 0, D, N, 1, 0};
    DataCopy(a1, x0, ap0);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    DataCopy(b1, w1, bp);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    DataCopy(b1[N * D], w2, bp);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    DataCopy(a1[Common::A], x1, ap1);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    if constexpr (Iterations >= 3U) {
        DataCopy(a1[2U * Common::A], x2, ap2);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
    }
    if constexpr (Iterations >= 4U) {
        const Nd2NzParams ap3{1, TM3, D, 0, D, TM3, 1, 0};
        DataCopy(a1[3U * Common::A], x3, ap3);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
    }
    if constexpr (Iterations >= 5U) {
        const Nd2NzParams ap4{1, TM4, D, 0, D, TM4, 1, 0};
        DataCopy(a1[4U * Common::A], x4, ap4);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID6);
    }
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData2dParams al;
    al.repeatTimes = D / 16U;
    al.srcStride = TM0 / 16U;
#pragma unroll

    for (uint32_t mb = 0; mb < TM0 / 16U; ++mb) {
        LoadData(a2[mb * D * 16U], a1[mb * 256U], al);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3, uint32_t TM4>
__aicore__ inline void SingleConsumerPairCube<T, TM, D, N, Iterations>::PrepareAll(
    const GlobalTensor<T>& w1, const GlobalTensor<T>& w2, const GlobalTensor<T>& x0,
    const GlobalTensor<T>& x1, const GlobalTensor<T>& x2, const GlobalTensor<T>& x3,
    const GlobalTensor<T>& x4)
{
    LocalTensor<T> a1(TPosition::A1, 0U, Iterations * Common::A);
    LocalTensor<T> b1(TPosition::B1, Iterations * Common::A * sizeof(T), Common::B);
    LocalTensor<T> a2(TPosition::A2, 0U, TM0 * D);
    LocalTensor<T> b2(TPosition::B2, 0U, Common::B);
    PrepareAllStageafe89876<TM0, TM1, TM2, TM3, TM4>(w1, w2, x0, x1, x2, x3, x4, a1, b1, a2);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    LoadData2dParams bl;
    bl.repeatTimes = N * D / 256U;
    bl.srcStride = 1;
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    LoadData(b2, b1, bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    LoadData(b2[N * D], b1[N * D], bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerPairCube<T, TM, D, N, Iterations>::ComputeStage093b73a7(A0& dst)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    LocalTensor<T> a1(TPosition::A1, I * Common::A * sizeof(T), CurTM * D);
    LocalTensor<T> a2(TPosition::A2, I * Common::A * sizeof(T), CurTM * D);
    LocalTensor<T> b2_0(TPosition::B2, 0U, N * D);
    LocalTensor<T> b2_1(TPosition::B2, N * D * sizeof(T), N * D);
    LocalTensor<float> c(TPosition::CO1, (I % 2U) * 2U * TM * N * sizeof(float), 2U * TM * N);
    if constexpr (I > 0U) {
        constexpr event_t mte2Ev = (I == 1U) ? EVENT_ID1 : ((I == 2U) ? EVENT_ID4 : EVENT_ID5);
        WaitFlag<HardEvent::MTE2_MTE1>(mte2Ev);
        ::SharedLoadTileA<TM, D, I, CurTM, true>(a1, a2);
    }
    Common::template ComputeStage06a53e30<I, CurTM>(dst, a2, b2_0, b2_1, c);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerPairCube<T, TM, D, N, Iterations>::ComputeStage38e86bde(A0& dst, A1& a1, A2& b2_0, A3& b2_1)
{
    LocalTensor<T> a2_2(TPosition::A2, 80U * D * sizeof(T), 48U * D);
    LocalTensor<float> c2(TPosition::CO1, 2U * 80U * N * sizeof(float), 2U * 48U * N);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
    ::SharedPrimitivedc71102ef9<D>(a1, a2_2);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID4);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID4);
    ::SharedPrimitived55642c3a1<D, N>(a2_2, b2_0, b2_1, c2);
    ::SharedPrimitive5ed408cbed<T, N>(c2, dst);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerPairCube<T, TM, D, N, Iterations>::ComputeStage0fee0c01(A0& dst, A1& a1, A2& b2_0, A3& b2_1)
{
    LocalTensor<T> a2_1(TPosition::A2, 32U * D * sizeof(T), 48U * D);
    LocalTensor<float> c1(TPosition::CO1, 2U * 32U * N * sizeof(float), 2U * 48U * N);
    ::SharedPrimitive754371e923<D, N>(a1, a2_1, b2_0, b2_1, c1);
    ::SharedPrimitive0689c680de<T, N>(c1, dst);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerPairCube<T, TM, D, N, Iterations>::ComputeStaged537ca52(A0& dst)
{
    LocalTensor<T> a1(TPosition::A1, I * Common::A * sizeof(T), CurTM * D);
    LocalTensor<T> b2_0(TPosition::B2, 0U, N * D);
    LocalTensor<T> b2_1(TPosition::B2, N * D * sizeof(T), N * D);
    if constexpr (I == 0U) {
        LocalTensor<T> a2_0(TPosition::A2, 0U, 32U * D);
        LocalTensor<float> c0(TPosition::CO1, 0U, 2U * 32U * N);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

        ::SharedPrimitive54529556e1<T, D, N>(a2_0, b2_0, b2_1, c0, dst);
    } else if constexpr (I == 1U) {
        ComputeStage0fee0c01<I, CurTM>(dst, a1, b2_0, b2_1);
    } else {
        ComputeStage38e86bde<I, CurTM>(dst, a1, b2_0, b2_1);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM>
__aicore__ inline void SingleConsumerPairCube<T, TM, D, N, Iterations>::Compute(const GlobalTensor<T>& dst)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = Common::DoubleC ? slot : EVENT_ID0;
    if constexpr (Common::SplitL0B) {
        Common::template ComputeSplit<I, CurTM>(dst);
    } else if constexpr (TM == 48U) {
        ComputeStaged537ca52<I, CurTM>(dst);
    } else {
        ComputeStage093b73a7<I, CurTM>(dst);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
__aicore__ inline void SingleConsumerPairCube<T, TM, D, N, Iterations>::Drain()
{
    if constexpr (TM != 48U) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }
    if constexpr (TM != 48U) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    }
    ::SharedPrimitivebd2d8d032a<TM, Common, Common::SplitL0B>();
}

#endif

}
