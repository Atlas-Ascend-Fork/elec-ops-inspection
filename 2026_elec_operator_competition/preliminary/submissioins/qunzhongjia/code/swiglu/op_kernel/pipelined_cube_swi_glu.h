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
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN = 64U,
          uint32_t TailIterations = Iterations, uint32_t FullCores = Cores>
class PipelinedCubeSwiGlu {
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN = 64U>
class PipelinedCubeSwiGlu {
#endif
public:
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void RunStage46925810(A0& x, A1& w1, A2& b1, A3& w2,
                                                                           A4& b2, A5& y, A6& workspace);
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
#else
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void RunStagee3ff0211(A0 & x, A1 & w1, A2 & b1, A3 & w2,
                                                                           A4 & b2, A5 & y, A6 & workspace);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void RunStageec337e10(A0 & x, A1 & w1, A2 & b1, A3 & w2,
                                                                           A4 & b2, A5 & y, A6 & workspace);
#endif
    __aicore__ inline __attribute__((always_inline)) void SharedStageccbb7d440e();
#endif
#endif
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR workspace);

private:
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    template <uint32_t Slot> __aicore__ inline void WaitConsumed();
    template <uint32_t I, uint32_t CurIter> __aicore__ inline void Step();
    template <uint32_t I, uint32_t CurIter, class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void EpilogueSharedStage0(A0& row, A1& begin, A2& input,
                                                                               A3& out);
    template <uint32_t I, uint32_t CurIter, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6>
    __aicore__ inline __attribute__((always_inline)) void
    EpilogueStage23fd9fc7(A0& row, A1& begin, A2& input, A3& g, A4& u, A5& sig, A6& out);
    template <uint32_t I, uint32_t CurIter, class B0, class B1, class B2>
    __aicore__ inline __attribute__((always_inline)) void ApplyEpilogue(B0& begin, B1& off, B2& row);
    template <uint32_t I, uint32_t CurIter> __aicore__ inline void Epilogue(uint64_t off, uint64_t row);
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <uint32_t First, uint32_t Count> __aicore__ inline void PrefetchBatch();
#else
    template <uint32_t I, class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void SharedStage00bbcac747(B0 & off, B1 & row);
    template <uint32_t I, class B0>
    __aicore__ inline __attribute__((always_inline)) void StepGroupFour(B0 & off);
    template <uint32_t I, class B0>
    __aicore__ inline __attribute__((always_inline)) void StepDefault(B0 & off);
#endif
    template <uint32_t I> __aicore__ inline void Step();
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <uint32_t First, uint32_t Count> __aicore__ inline void EpilogueBatch();
#else
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <uint32_t I, uint32_t CurTM = TM, class A0>
    __aicore__ inline __attribute__((always_inline)) void EpilogueSharedStage0(A0 & input);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void EpilogueStage1f90073b(
        A0 & row, A1 & begin, A2 & input, A3 & g, A4 & u, A5 & sig, A6 & out, A7 & ur, A8 & br);
#else
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline
        __attribute__((always_inline)) void EpilogueStage224a0a0c(A0 & row, A1 & begin, A2 & input, A3 & out);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline
        __attribute__((always_inline)) void EpilogueSharedStage0(A0 & row, A1 & begin, A2 & input, A3 & out);
#endif
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6, class A7, class A8, class A9>
    __aicore__ inline
        __attribute__((always_inline)) void EpilogueSharedStage1(A0 & row, A1 & begin, A2 & input,
                                                                 A3 & curInput, A4 & b1, A5 & b2, A6 & g,
                                                                 A7 & u, A8 & sig, A9 & out);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2>
    __aicore__ inline __attribute__((always_inline)) void ApplyEpilogue(B0 & begin, B1 & off, B2 & row);
#else
    template <uint32_t I, uint32_t CurTM = TM, class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void ApplyEpilogue(B0 & begin, B1 & row);
#endif
    template <uint32_t I, uint32_t CurTM = TM> __aicore__ inline void Epilogue(uint64_t off, uint64_t row);
#endif
#endif
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t TN = D / N;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
#else
    static constexpr uint32_t Groups = Cores / TN;
#endif
    static constexpr uint32_t Plane = TM * N;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    static constexpr uint32_t M = TM * Cores / TN * Iterations;
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    static constexpr uint32_t M = TM * (FullCores * Iterations + (Cores - FullCores) * TailIterations) / TN;
#else
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    static constexpr uint32_t M =
        (Groups == 4U) ? 512U : ((Groups == 10U) ? 8192U : (TM * Groups * Iterations));
#else
    static constexpr uint32_t M = (Groups == 4U) ? 512U : ((Groups == 10U) ? 8192U : 4096U);
#endif
    static constexpr uint32_t FirstTM = (Groups == 4U) ? 32U : TM;
#endif
    static constexpr bool FastAiv = true;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
#else
    static constexpr bool SplitL0B = (2U * N * D * sizeof(T) > 64U * 1024U);
#endif
#endif
    static constexpr uint32_t NR = TM / 2U;
    static constexpr uint32_t Cap = NR * N;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    static constexpr uint32_t MaxCap = 2U * Cap;
    static_assert(D == 128U && N == 128U && TM == 64U && Cores == 16U && Iterations == 4U,
                  "Case5 four producer tiles, three vector batches");
    static_assert(180480U <= 192U * 1024U, "Distinct input/output batches fit UB");
    DoubleBufferedPairCube<T, TM, D, N> piped_;
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    static constexpr uint32_t InSlots = 4U;
    static constexpr uint32_t OutSlots = 4U;
    static_assert(D % N == 0U && TM % 32U == 0U, "Exact tile shape");
#else
    static constexpr uint32_t InSlots = SplitL0B ? 2U : 4U;
    static constexpr uint32_t OutSlots = SplitL0B ? 2U : 4U;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    static_assert(D % N == 0U && TM % 32U == 0U, "Exact tile shape");
#else
    static_assert(D % N == 0U && TM % 16U == 0U, "Exact tile shape");
#endif
#endif
    static_assert(TM * D * sizeof(T) <= 64U * 1024U, "L0A capacity");
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    static_assert(2U * N * D * sizeof(T) <= 64U * 1024U, "L0B capacity");
#else
    static_assert(N * D * sizeof(T) <= 64U * 1024U, "L0B half capacity");
#endif
    static_assert(2U * TM * N * sizeof(float) <= 128U * 1024U, "L0C capacity");
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    static_assert(((2U * InSlots + 3U + OutSlots) * Cap + 2U * N + 128U) * sizeof(T) <= 192U * 1024U,
                  "UB capacity");
    DoubleBufferedPairCube<T, TM, D, N> piped_;
#else
    static_assert(((2U * InSlots + 3U + OutSlots) * Cap + 2U * N) * sizeof(T) <= 192U * 1024U, "UB capacity");
    DoubleBufferedPairCube<T, TM, D, N, Iterations> piped_;
#endif
#endif
    GlobalTensor<T> x_;
    GlobalTensor<T> w1_;
    GlobalTensor<T> w2_;
    GlobalTensor<T> b1_;
    GlobalTensor<T> b2_;
    GlobalTensor<T> y_;
    GlobalTensor<T> slots_;
    uint32_t core_ = 0;
    uint32_t col_ = 0;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
#else
    uint32_t grp_ = 0;
    uint32_t baseRow_ = 0;
#endif
};
}
