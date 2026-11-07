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
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
class DoubleBufferedPairCube {
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
class DoubleBufferedPairCube {
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
class DoubleBufferedPairCube : public PairCubeCommon<T, TM, D, N, Iterations> {
#endif
public:
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM, uint32_t TM4 = TM0, class A0,
              class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    PrepareAllStagee11ca2e9(A0& x2, A1& x3, A2& a1_0, A3& a1_2, A4& b1, A5& a2_0, A6& b2_0, A7& b2_1,
                            A8& ap2);
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM, uint32_t TM4 = TM0, class B0,
              class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
              class B10, class B11, class B12, class B13, class B14, class B15, class B16, class B17>
    __aicore__ inline __attribute__((always_inline)) void
    FinishPrepareAll(B0& a1_0, B1& a1_1, B2& a1_2, B3& a2_0, B4& ap0, B5& ap1, B6& ap2, B7& b1, B8& b2_0,
                     B9& b2_1, B10& bp, B11& w1, B12& w2, B13& x0, B14& x1, B15& x2, B16& x3, B17& x4);
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM, uint32_t TM4 = TM0>
    __aicore__ inline void PrepareAll(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
                                      const GlobalTensor<T>& x0, const GlobalTensor<T>& x1,
                                      const GlobalTensor<T>& x2, const GlobalTensor<T>& x3,
                                      const GlobalTensor<T>& x4);
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void SharedStaged3e4cc9684(B0& a1, B1& a2);
    template <uint32_t I, uint32_t CurTM = TM, class A0>
    __aicore__ inline __attribute__((always_inline)) void ComputeStageb52739cb(A0& dst);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage21e20e1c(A0& dst, A1& a1, A2& b2_0,
                                                                               A3& b2_1);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void ComputeStageeeb2fd57(A0& dst, A1& a1, A2& b2_0,
                                                                               A3& b2_1);
    template <uint32_t I, uint32_t CurTM = TM, class A0>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage584a2d8d(A0& dst);
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void SharedStagec7aaa4f442(B0& a1, B1& a2_0);
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <class B0, class B1, class B2>
    __aicore__ inline __attribute__((always_inline)) void SharedStageebbd2e0f1a(B0 & w1, B1 & w2, B2 & x0);
    __aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
                                          const GlobalTensor<T>& x0);
#else
    template <uint32_t FirstTM = TM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
              class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void PrepareWeightsStagec2e3b721(
        A0 & w1, A1 & w2, A2 & x0, A3 & b1, A4 & a1, A5 & b2_0, A6 & b2_1, A7 & hap, A8 & bp);
    template <uint32_t FirstTM = TM>
    __aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
                                          const GlobalTensor<T>& x0);
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM, class A0, class A1, class A2,
              class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
    __aicore__ inline __attribute__((always_inline)) void PrepareAllStagecfd973b0(
        A0 & x1, A1 & x2, A2 & x3, A3 & a1_0, A4 & a1_1, A5 & a1_2, A6 & b1, A7 & a2_0, A8 & b2_0, A9 & b2_1,
        A10 & ap1, A11 & ap2);
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM, class B0, class B1, class B2,
              class B3, class B4, class B5, class B6, class B7, class B8, class B9, class B10, class B11,
              class B12, class B13, class B14, class B15, class B16>
    __aicore__ inline __attribute__((always_inline)) void FinishPrepareAll(
        B0 & a1_0, B1 & a1_1, B2 & a1_2, B3 & a2_0, B4 & ap0, B5 & ap1, B6 & ap2, B7 & b1, B8 & b2_0,
        B9 & b2_1, B10 & bp, B11 & w1, B12 & w2, B13 & x0, B14 & x1, B15 & x2, B16 & x3);
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3 = TM>
    __aicore__ inline void PrepareAll(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
                                      const GlobalTensor<T>& x0, const GlobalTensor<T>& x1,
                                      const GlobalTensor<T>& x2, const GlobalTensor<T>& x3);
#endif
    template <uint32_t I, uint32_t CurTM = TM> __aicore__ inline void Prefetch(const GlobalTensor<T>& x);
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    template <uint32_t I, class B0, class B1, class B2, class B3>
    __aicore__ inline
        __attribute__((always_inline)) void SharedStageab5685b342(B0 & b1, B1 & b2_0, B2 & b2_1, B3 & bl);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage90cd7c22(
        A0 & b1, A1 & b2_0, A2 & b2_1, A3 & bl, A4 & a2_0, A5 & a2_1, A6 & c0, A7 & c1, A8 & mad);
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void SharedStaged3e4cc9684(B0 & a1, B1 & a2);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4>
    __aicore__ inline __attribute__((always_inline)) void ComputeStagef0f64ea7(A0 & a1, A1 & a2, A2 & b2_0,
                                                                               A3 & b2_1, A4 & c);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage56696d27(
        A0 & b1, A1 & b2_0, A2 & b2_1, A3 & bl, A4 & a2_0, A5 & c0, A6 & mad);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void ComputeStagecf6a2fb3(
        A0 & a1, A1 & b1, A2 & b2_0, A3 & b2_1, A4 & bl, A5 & a2_0, A6 & a2_1, A7 & c0, A8 & c1);
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void SharedStagec7aaa4f442(B0 & a1, B1 & a2_0);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6>
    __aicore__ inline __attribute__((always_inline)) void ComputeStageea253d2a(
        A0 & a1, A1 & b1, A2 & b2_0, A3 & b2_1, A4 & bl, A5 & a2_0, A6 & c0);
    template <uint32_t I, uint32_t CurTM = TM, class A0>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage5944224d(A0 & dst);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage69b25c34(A0 & dst, A1 & a1, A2 & b1,
                                                                               A3 & b2_0, A4 & b2_1, A5 & bl);
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2, class B3, class B4, class B5,
              class B6, class B7>
    __aicore__ inline __attribute__((always_inline)) void ComputeSplitSmall(
        B0 & a1, B1 & a2_0, B2 & b1, B3 & b2_0, B4 & b2_1, B5 & bl, B6 & c0, B7 & dst);
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2, class B3, class B4, class B5>
    __aicore__ inline __attribute__((always_inline)) void ComputeSplitLarge(B0 & a1, B1 & b1, B2 & b2_0,
                                                                            B3 & b2_1, B4 & bl, B5 & dst);
#else
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage6e2aa2f7(A0 & a1, A1 & b2, A2 & a2_0,
                                                                               A3 & a2_1, A4 & c0, A5 & c1);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2>
    __aicore__ inline __attribute__((always_inline)) void SharedStage69c5fc7c97(B0 & c0, B1 & c1, B2 & dst);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2>
    __aicore__ inline __attribute__((always_inline)) void ComputeStageede6278d(A0 & dst, A1 & a1, A2 & b2);
#else
#endif
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2, class B3>
    __aicore__ inline
        __attribute__((always_inline)) void SharedStage448bfd918a(B0 & a1, B1 & a2, B2 & b2, B3 & c);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline
        __attribute__((always_inline)) void ComputeStage5d29390b(A0 & dst, A1 & a1, A2 & b2, A3 & a2);
#else
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage71fca89b(A0 & a1, A1 & b2);
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2>
    __aicore__ inline __attribute__((always_inline)) void SharedStage69c5fc7c97(B0 & c0, B1 & c1, B2 & dst);
#endif
#endif
#endif
    template <uint32_t I, uint32_t CurTM = TM> __aicore__ inline void Compute(const GlobalTensor<T>& dst);
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    __aicore__ inline void Drain();
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
#else
    template <uint32_t I> __aicore__ inline void Store(const GlobalTensor<T>& dst);
#endif
    template <uint32_t Iterations> __aicore__ inline void Drain();
#endif
private:
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    static constexpr uint32_t A = TM * D;
    static constexpr uint32_t B = 2U * N * D;
    static constexpr bool DoubleA2 = (2U * A * sizeof(T) <= 64U * 1024U);
    static constexpr bool DoubleC = (4U * TM * N * sizeof(float) <= 128U * 1024U);
    static_assert((2U * A + B) * sizeof(T) <= 512U * 1024U, "L1 capacity");
    static_assert(A * sizeof(T) <= 64U * 1024U && B * sizeof(T) <= 64U * 1024U, "L0A/B capacity");
#elif (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    using Shape = PairCubeShape<T, TM, D, N, Iterations>;
    static constexpr uint32_t A = Shape::A;
    static constexpr uint32_t B = Shape::B;
    static constexpr bool SplitL0B = Shape::SplitL0B;
    static constexpr bool DoubleA2 = Shape::DoubleA2;
    static constexpr bool DoubleC = Shape::DoubleC;
    static constexpr uint32_t L1Slots = Shape::L1Slots;
#else
    using Common = PairCubeCommon<T, TM, D, N, Iterations>;
#endif
};
}
