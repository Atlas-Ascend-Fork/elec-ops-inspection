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

template <class T, uint32_t N = 0> class ParallelInverse : private PinvInverseState<T, N, true> {
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buf;
    bool cubePrecision = true;
    FP32Cube *cube = nullptr;

    template <typename ExtractType78_0, typename ExtractType78_1, typename ExtractType78_2, typename ExtractType78_3,
              typename ExtractType78_4, typename ExtractType78_5, typename ExtractType78_6>
    __aicore__ inline void CubeRefineBlock78(ExtractType78_0 &batch, ExtractType78_1 &residual, ExtractType78_2 &savedA,
                                             ExtractType78_3 &corr, ExtractType78_4 &inv, ExtractType78_5 &row0,
                                             ExtractType78_6 &rows);
    template <typename ExtractType85_0, typename ExtractType85_1, typename ExtractType85_2>
    __aicore__ inline void CubeRefineBlock83Block85(ExtractType85_0 &aic, ExtractType85_1 &row0, ExtractType85_2 &rows);
    template <typename ExtractType83_0, typename ExtractType83_1, typename ExtractType83_2>
    __aicore__ inline void CubeRefineBlock83(ExtractType83_0 &aic, ExtractType83_1 &row0, ExtractType83_2 &rows);
    __aicore__ inline void CubeRefine(uint32_t batch);
    __aicore__ inline void CubeCorrection(uint32_t batch);
    template <typename ExtractType81_0, typename ExtractType81_1, typename ExtractType81_2, typename ExtractType81_3,
              typename ExtractType81_4, typename ExtractType81_5, typename ExtractType81_6, typename ExtractType81_7,
              typename ExtractType81_8, typename ExtractType81_9, typename ExtractType81_10, typename ExtractType81_11,
              typename ExtractType81_12>
    __aicore__ inline void
    RefineInverseBlock66Block67Block72Block81(ExtractType81_0 &batch, ExtractType81_1 &residual, ExtractType81_2 &wp,
                                              ExtractType81_3 &row, ExtractType81_4 &out, ExtractType81_5 &products,
                                              ExtractType81_6 &sums, ExtractType81_7 &segments,
                                              ExtractType81_8 &batchDots, ExtractType81_9 &aCache,
                                              ExtractType81_10 &cacheA, ExtractType81_11 &cc, ExtractType81_12 &kc);
    template <typename ExtractType72_0, typename ExtractType72_1, typename ExtractType72_2, typename ExtractType72_3,
              typename ExtractType72_4, typename ExtractType72_5, typename ExtractType72_6, typename ExtractType72_7,
              typename ExtractType72_8, typename ExtractType72_9, typename ExtractType72_10, typename ExtractType72_11,
              typename ExtractType72_12>
    __aicore__ inline void
    RefineInverseBlock66Block67Block72(ExtractType72_0 &batch, ExtractType72_1 &residual, ExtractType72_2 &inv,
                                       ExtractType72_3 &wp, ExtractType72_4 &row, ExtractType72_5 &out,
                                       ExtractType72_6 &chunk, ExtractType72_7 &products, ExtractType72_8 &sums,
                                       ExtractType72_9 &segments, ExtractType72_10 &batchDots, ExtractType72_11 &aCache,
                                       ExtractType72_12 &cacheA);
    template <typename ExtractType82_0, typename ExtractType82_1, typename ExtractType82_2, typename ExtractType82_3,
              typename ExtractType82_4, typename ExtractType82_5, typename ExtractType82_6, typename ExtractType82_7,
              typename ExtractType82_8, typename ExtractType82_9, typename ExtractType82_10, typename ExtractType82_11,
              typename ExtractType82_12, typename ExtractType82_13>
    __aicore__ inline void
    RefineInverseBlock66Block67Block73Block82(ExtractType82_0 &batch, ExtractType82_1 &residual, ExtractType82_2 &wp,
                                              ExtractType82_3 &row, ExtractType82_4 &out, ExtractType82_5 &products,
                                              ExtractType82_6 &sums, ExtractType82_7 &values, ExtractType82_8 &segments,
                                              ExtractType82_9 &batchDots, ExtractType82_10 &aCache,
                                              ExtractType82_11 &cacheA, ExtractType82_12 &cc, ExtractType82_13 &kc);
    template <typename ExtractType73_0, typename ExtractType73_1, typename ExtractType73_2, typename ExtractType73_3,
              typename ExtractType73_4, typename ExtractType73_5, typename ExtractType73_6, typename ExtractType73_7,
              typename ExtractType73_8, typename ExtractType73_9, typename ExtractType73_10, typename ExtractType73_11,
              typename ExtractType73_12, typename ExtractType73_13>
    __aicore__ inline void
    RefineInverseBlock66Block67Block73(ExtractType73_0 &batch, ExtractType73_1 &residual, ExtractType73_2 &inv,
                                       ExtractType73_3 &wp, ExtractType73_4 &row, ExtractType73_5 &out,
                                       ExtractType73_6 &chunk, ExtractType73_7 &products, ExtractType73_8 &sums,
                                       ExtractType73_9 &values, ExtractType73_10 &segments, ExtractType73_11 &batchDots,
                                       ExtractType73_12 &aCache, ExtractType73_13 &cacheA);

    template <typename ExtractType67_0, typename ExtractType67_1, typename ExtractType67_2, typename ExtractType67_3,
              typename ExtractType67_4, typename ExtractType67_5, typename ExtractType67_6>
    __aicore__ inline void RefineInverseBlock66Block67(ExtractType67_0 &batch, ExtractType67_1 &residual,
                                                       ExtractType67_2 &inv, ExtractType67_3 &wp, ExtractType67_4 &row,
                                                       ExtractType67_5 &out, ExtractType67_6 &chunk);

    template <typename ExtractType69_0, typename ExtractType69_1, typename ExtractType69_2, typename ExtractType69_3,
              typename ExtractType69_4, typename ExtractType69_5, typename ExtractType69_6, typename ExtractType69_7,
              typename ExtractType69_8, typename ExtractType69_9, typename ExtractType69_10, typename ExtractType69_11>
    __aicore__ inline void
    RefineInverseBlock66Block69(ExtractType69_0 &residual, ExtractType69_1 &inv, ExtractType69_2 &chunk,
                                ExtractType69_3 &savedA, ExtractType69_4 &acc, ExtractType69_5 &acc1,
                                ExtractType69_6 &acc2, ExtractType69_7 &acc3, ExtractType69_8 &coeff,
                                ExtractType69_9 &cache, ExtractType69_10 &br, ExtractType69_11 &outputIndices);
    template <typename ExtractType66_0, typename ExtractType66_1, typename ExtractType66_2, typename ExtractType66_3,
              typename ExtractType66_4, typename ExtractType66_5, typename ExtractType66_6, typename ExtractType66_7>
    __aicore__ inline bool RefineInverseBlock66(ExtractType66_0 &batch, ExtractType66_1 &residual,
                                                ExtractType66_2 &corrected, ExtractType66_3 &inv, ExtractType66_4 &wp,
                                                ExtractType66_5 &row, ExtractType66_6 &out, ExtractType66_7 &chunk);

    __aicore__ inline void RefineInverseBlock95();
    __aicore__ inline void RefineInverse(uint32_t batch);

public:
    __aicore__ inline TPipe *Pipe();
    __aicore__ inline void SetCube(FP32Cube *mm);
    __aicore__ inline void RefineOnly();

    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt);
    template <typename ExtractType79_0, typename ExtractType79_1, typename ExtractType79_2, typename ExtractType79_3>
    __aicore__ inline void RunBlock70Block79(ExtractType79_0 &bad, ExtractType79_1 &pc, ExtractType79_2 &slot,
                                             ExtractType79_3 &stop);
    template <typename ExtractType70_0, typename ExtractType70_1>
    __aicore__ inline void RunBlock70(ExtractType70_0 &bad, ExtractType70_1 &c1);
    template <typename ExtractType76_0> __aicore__ inline void RunBlock74Block76(ExtractType76_0 &invall);
    template <typename ExtractType74_0, typename ExtractType74_1>
    __aicore__ inline void RunBlock74(ExtractType74_0 &invall, ExtractType74_1 &isq);

    template <typename ExtractType93_0, typename ExtractType93_1, typename ExtractType93_2, typename ExtractType93_3,
              typename ExtractType93_4>
    __aicore__ inline void RunBlock93(ExtractType93_0 &batch, ExtractType93_1 &needs, ExtractType93_2 &unbounded,
                                      ExtractType93_3 &accept, ExtractType93_4 &myOk);
    template <typename ExtractType94_0, typename ExtractType94_1>
    __aicore__ inline void RunBlock94(ExtractType94_0 &batch, ExtractType94_1 &ac);

    __aicore__ inline float LoadBatchInput(uint32_t &batch, uint32_t &base, uint32_t &ac);
    template <typename ExtractType97_0> __aicore__ inline bool RunBlock97(ExtractType97_0 &failed);
    __aicore__ inline bool Run();
};

#include "pinv_parallel_inverse_impl1.h"

#include "pinv_parallel_inverse_impl2.h"
