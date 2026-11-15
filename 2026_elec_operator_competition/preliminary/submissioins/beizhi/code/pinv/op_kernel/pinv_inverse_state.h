/*
 * Copyright (c) 2026 Beizhi contributors.
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

template <class T, uint32_t N, bool StaticShape> class PinvInverseState {
protected:
    LinalgPinvTilingData t;
    GlobalTensor<T> input, output;
    GlobalTensor<float> exchange, iall, state;
    LocalTensor<float> mat, rm, piv, tmp, panel, red, broad, invs;
    LocalTensor<uint32_t> idx, perm, ofs;
    LocalTensor<T> raw;
    uint32_t part, groupid, first, ld, cols, panelCols;
    bool outputReady;
    __aicore__ inline void Fence();
    __aicore__ inline void Sync();
    __aicore__ inline void Store(GlobalTensor<float> dst, float a, float b, float c);
    __aicore__ inline void Load(GlobalTensor<float> src);
    __aicore__ inline void Sum(LocalTensor<float> dst, LocalTensor<float> src);
    __aicore__ inline float Dot(LocalTensor<float> x);
    __aicore__ inline uint32_t Pivot();
    __aicore__ inline float NormRoot(float z);
    template <typename ExtractType59_0, typename ExtractType59_1, typename ExtractType59_2, typename ExtractType59_3>
    __aicore__ inline void PowerRowBlock59(ExtractType59_0 &row, ExtractType59_1 &transpose, ExtractType59_2 &dst,
                                           ExtractType59_3 &base);
    __aicore__ inline void PowerRow(uint32_t batch, uint32_t row, bool transpose, bool inverse, LocalTensor<float> dst);
    __aicore__ inline void Update(uint32_t c0, uint32_t c1, uint32_t row);
    __aicore__ inline float PowerMultiply(LocalTensor<float> cache, LocalTensor<float> x, LocalTensor<float> y,
                                          LocalTensor<float> vals, uint32_t firstRow, uint32_t ac, uint32_t offset);
    __aicore__ __attribute__((noinline)) float PowerNorm(uint32_t batch, bool inverse);
    template <typename ExtractType80_0, typename ExtractType80_1, typename ExtractType80_2, typename ExtractType80_3,
              typename ExtractType80_4, typename ExtractType80_5, typename ExtractType80_6, typename ExtractType80_7,
              typename ExtractType80_8, typename ExtractType80_9>
    __aicore__ inline void SharedInverseBlock1(ExtractType80_0 &chunk, ExtractType80_1 &savedA, ExtractType80_2 &acc,
                                               ExtractType80_3 &acc1, ExtractType80_4 &acc2, ExtractType80_5 &acc3,
                                               ExtractType80_6 &coeff, ExtractType80_7 &cache, ExtractType80_8 &br,
                                               ExtractType80_9 &ac);

    template <typename ExtractType84_0, typename ExtractType84_1, typename ExtractType84_2, typename ExtractType84_3,
              typename ExtractType84_4, typename ExtractType84_5, typename ExtractType84_6, typename ExtractType84_7>
    __aicore__ inline void SharedInverseBlock2(ExtractType84_0 &acc, ExtractType84_1 &cache, ExtractType84_2 &br,
                                               ExtractType84_3 &ac, ExtractType84_4 &acc1, ExtractType84_5 &acc2,
                                               ExtractType84_6 &acc3, ExtractType84_7 &kc);

    template <typename ExtractType75_0, typename ExtractType75_1, typename ExtractType75_2, typename ExtractType75_3,
              typename ExtractType75_4, typename ExtractType75_5, typename ExtractType75_6, typename ExtractType75_7>
    __aicore__ inline void SharedInverseBlock3(ExtractType75_0 &acc, ExtractType75_1 &cache, ExtractType75_2 &br,
                                               ExtractType75_3 &ac, ExtractType75_4 &acc1, ExtractType75_5 &acc2,
                                               ExtractType75_6 &acc3, ExtractType75_7 &kc);

    template <typename ExtractType71_0, typename ExtractType71_1, typename ExtractType71_2, typename ExtractType71_3,
              typename ExtractType71_4, typename ExtractType71_5, typename ExtractType71_6, typename ExtractType71_7,
              typename ExtractType71_8, typename ExtractType71_9>
    __aicore__ inline void SharedInverseBlock4(ExtractType71_0 &inv, ExtractType71_1 &chunk, ExtractType71_2 &acc,
                                               ExtractType71_3 &cf, ExtractType71_4 &cache, ExtractType71_5 &br,
                                               ExtractType71_6 &ac, ExtractType71_7 &acc1, ExtractType71_8 &acc2,
                                               ExtractType71_9 &acc3);

    template <typename ExtractType68_0, typename ExtractType68_1, typename ExtractType68_2, typename ExtractType68_3,
              typename ExtractType68_4, typename ExtractType68_5, typename ExtractType68_6, typename ExtractType68_7,
              typename ExtractType68_8, typename ExtractType68_9>
    __aicore__ inline void SharedInverseBlock5(ExtractType68_0 &batch, ExtractType68_1 &residual,
                                               ExtractType68_2 &corrected, ExtractType68_3 &inv, ExtractType68_4 &chunk,
                                               ExtractType68_5 &acc, ExtractType68_6 &cf, ExtractType68_7 &cache,
                                               ExtractType68_8 &br, ExtractType68_9 &outputIndices);

    template <typename ExtractType92_0, typename ExtractType92_1>
    __aicore__ inline void SharedInverseBlock6(ExtractType92_0 &br, ExtractType92_1 &outputIndices);

    __aicore__ inline void SharedInverseBlock7();

    __aicore__ inline void SharedInverseBlock8();

    template <typename ExtractType77_0, typename ExtractType77_1, typename ExtractType77_2, typename ExtractType77_3>
    __aicore__ inline void SharedInverseBlock9(ExtractType77_0 &failed, ExtractType77_1 &base, ExtractType77_2 &ac,
                                               ExtractType77_3 &invall);

    template <typename ExtractType89_0, typename ExtractType89_1, typename ExtractType89_2, typename ExtractType89_3,
              typename ExtractType89_4>
    __aicore__ inline void SharedInverseBlock10(ExtractType89_0 &needs, ExtractType89_1 &unbounded,
                                                ExtractType89_2 &accept, ExtractType89_3 &myOk,
                                                ExtractType89_4 &sensitive);

    template <typename ExtractType91_0, typename ExtractType91_1>
    __aicore__ inline void SharedInverseBlock11(ExtractType91_0 &base, ExtractType91_1 &ac);

    __aicore__ inline void SharedInverseBlock12();

    __aicore__ inline void SharedInverseBlock13();

    __aicore__ inline void SharedInverseBlock14();

    template <typename ExtractType51_0> __aicore__ inline void SharedInverseBlock15(ExtractType51_0 &ac);
    __aicore__ inline void InitShape(LinalgPinvTilingData tt);

    __aicore__ inline void InitGlobalViews(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws);

    __aicore__ inline void InitLocalViews();

    __aicore__ inline bool FinalResult(bool failed);

    __aicore__ inline void BuildOutputIndices(LocalTensor<uint32_t> outputIndices);

    __aicore__ inline void BuildResidualBlock(GlobalTensor<float> inv, uint32_t c0, uint32_t chunk,
                                              GlobalTensor<float> savedA, LocalTensor<float> acc,
                                              LocalTensor<float> acc1, LocalTensor<float> acc2, LocalTensor<float> acc3,
                                              LocalTensor<float> coeff, LocalTensor<float> cache, LocalTensor<float> br,
                                              LocalTensor<uint32_t> outputIndices);

    __aicore__ inline void WriteBf16Inverse(uint32_t base, GlobalTensor<float> invall);

    __aicore__ inline void AssessGroup(float aa, float ii, float bb, uint32_t g, float &needs, float &unbounded,
                                       bool &accept, bool &myOk);

    __aicore__ inline void UpdateGeneric(uint32_t c0, uint32_t c1);

    __aicore__ inline void FinishRefinement(GlobalTensor<float> corrected, GlobalTensor<float> inv,
                                            LocalTensor<float> out);

    __aicore__ inline void CheckSpectralBound(uint32_t batch, float unbounded, bool myOk, bool &accept);

    __aicore__ inline void LoadRefinementColumns(uint32_t wp, GlobalTensor<float> inv, uint32_t cc, uint32_t kc);

    __aicore__ inline void LoadRefinementRow(uint32_t batch, uint32_t r, uint32_t wp, LocalTensor<float> &row,
                                             LocalTensor<float> aCache, bool cacheA);

    template <typename ExtractType88_0, typename ExtractType88_1, typename ExtractType88_2, typename ExtractType88_3,
              typename ExtractType88_4>
    __aicore__ inline void SharedPhase1(ExtractType88_0 &batch, ExtractType88_1 &wp, ExtractType88_2 &aCache,
                                        ExtractType88_3 &owned, ExtractType88_4 &cacheA);
    __aicore__ inline float PreparePivot(LocalTensor<float> col, uint32_t j, uint32_t pc, uint32_t row, bool &bad);

    __aicore__ inline void RestoreOwnerRows(uint32_t pc, uint32_t cnt);

    __aicore__ inline void LoadBatchColumns(uint32_t base, uint32_t ac);

    __aicore__ inline PinvBatchAssessment AssessBatch(float asq, float isq, bool bad);
};

#include "pinv_inverse_state_impl1.h"

#include "pinv_inverse_state_impl2.h"
