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
// Included in the KernelLinalgDet class; preserves its member scope and access.
    template <auto earlyBroadcast>
    __aicore__ inline void ProcessSmallRawUBSetFlagPhase17(__ubuf__ float *&cachedColumnUB, uint32_t &cnt,
                                                           LocalTensor<float> &factors) {

        if constexpr (earlyBroadcast) {
            if (cnt > 1U) {
                ResetMask();
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB), 1, 8,
                      static_cast<uint8_t>((cnt + 7U) / 8U));
            }
        }
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }

    __aicore__ inline void ProcessSmallRawUBSetFlagPhase18(uint32_t &m, uint32_t &nn32, bool &tiny,
                                                           uint32_t &vMte2Pend) {

        uint64_t g = static_cast<uint64_t>(start_ + m);
        ProcessSmallRawUBDataCopyPhase1(g, m, nn32, tiny, vMte2Pend);
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        if (!tiny) {
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
            WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        }
    }
    template <bool DEFER_PROD, bool FULL_RAGGED_ROW, bool DIAGONAL_INDUCTION, bool DYNAMIC_SEARCH_MASK,
              bool GUARDED_TAIL2, bool PANEL_COMPENSATE, uint32_t SPR_MODE, auto batchedUpdate, auto earlyBroadcast,
              auto normalMask, auto paddedUpdate, class LoopArg0, class LoopArg1, class LoopArg2, class LoopArg3,
              class LoopArg4, class LoopArg5, class LoopArg6, class LoopArg7, class LoopArg8, class LoopArg9,
              class LoopArg10, class LoopArg11, class LoopArg12, class LoopArg13>
    __aicore__ inline bool SmallRawPivotStep(LoopArg0 &cachedAbsUB, LoopArg1 &cachedColumnUB, LoopArg2 &cachedMatrixUB,
                                             LoopArg3 &cachedOffsetUB, LoopArg4 &cachedReduceUB, LoopArg5 &cachedRowUB,
                                             LoopArg6 &diagonalOffset, LoopArg7 &ex, LoopArg8 &fastProd, LoopArg9 &k,
                                             LoopArg10 &pivotMask, LoopArg11 &prod, LoopArg12 &sign,
                                             LoopArg13 &singular) {

        if constexpr (GUARDED_TAIL2) {
            if (k + 2U == n_) {
                SetFlag<HardEvent::V_S>(::kEvVS);
                WaitFlag<HardEvent::V_S>(::kEvVS);
                float tail = SmallTailTwoGuarded(k * (s_ + 1U), s_);
                if (tail != 0.0f) {
                    ProdSmallCommon(fastProd, prod, ex, tail);
                    return false;
                }
            }
        }
        uint32_t kA = FULL_RAGGED_ROW ? 0U : k & ~(::kAlign - 1);
        uint32_t cnt = n_ - k;
        ProcessSmallRawUBvgatherPhase14<DIAGONAL_INDUCTION>(cachedAbsUB, cachedColumnUB, cachedMatrixUB, cachedOffsetUB,
                                                            cnt, diagonalOffset, k, pivotMask);
        uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
        ProcessSmallRawUBconstexprPhase15<normalMask>(cnt);
        LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
        ProcessSmallRawUBSetFlagPhase17<earlyBroadcast>(cachedColumnUB, cnt, factors);
        float maxv = 0.0f;
        bool sprPositive = false;
        uint32_t p = k;
        ProcessSmallRawUBconstexprPhase2<FULL_RAGGED_ROW, DIAGONAL_INDUCTION, DYNAMIC_SEARCH_MASK, SPR_MODE,
                                         normalMask>(cachedReduceUB, k, maxv, p, segN, sprPositive);
        if (normalMask ? (SPR_MODE == 2U ? !sprPositive : !(maxv > 0.0f)) : maxv == 0.0f) {
            singular = 1;
            return false;
        }
        ProcessSmallRawUBSmallMaskResetPhase12<DEFER_PROD, FULL_RAGGED_ROW, PANEL_COMPENSATE, batchedUpdate,
                                               earlyBroadcast, normalMask, paddedUpdate>(
            cachedAbsUB, cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, ex, factors, fastProd, k, kA, p, prod, sign);

        return true;
    }
    template <bool DEFER_PROD, bool FULL_RAGGED_ROW, bool DIAGONAL_INDUCTION, bool DYNAMIC_SEARCH_MASK,
              bool GUARDED_TAIL2, bool PANEL_COMPENSATE, uint32_t SPR_MODE, auto batchedUpdate, auto earlyBroadcast,
              auto normalMask, auto paddedUpdate, class LoopArg0, class LoopArg1, class LoopArg2, class LoopArg3,
              class LoopArg4, class LoopArg5, class LoopArg6, class LoopArg7, class LoopArg8, class LoopArg9>
    __aicore__ inline void SmallRawMatrixStep(LoopArg0 &cachedAbsUB, LoopArg1 &cachedColumnUB, LoopArg2 &cachedMatrixUB,
                                              LoopArg3 &cachedOffsetUB, LoopArg4 &cachedReduceUB, LoopArg5 &cachedRowUB,
                                              LoopArg6 &m, LoopArg7 &nn32, LoopArg8 &tiny, LoopArg9 &vMte2Pend) {

        ProcessSmallRawUBSetFlagPhase18(m, nn32, tiny, vMte2Pend);
        uint32_t ld_ = (tiny && n_ != s_) ? n_ : s_;
        float sign = 1.0f;
        float prod = 1.0f;
        int32_t ex = 0;
        bool fastProd = true;
        uint32_t singular = 0;
        if (tiny) {
            ProcessSmallRawUBStatePhase9<normalMask>(cachedMatrixUB, ex, fastProd, ld_, prod, sign, singular);
        } else {
            static_assert(!DIAGONAL_INDUCTION || (FULL_RAGGED_ROW && LD_BATCH == 6U),
                          "Diagonal induction is the guarded RawUB route only");
            uint32_t diagonalOffset = 0U;
            uint64_t pivotMask = 0x7FFFFFFFULL;
            if constexpr (DYNAMIC_SEARCH_MASK) {
                static_assert(DIAGONAL_INDUCTION && FULL_RAGGED_ROW && LD_BATCH == 6U, "Dynamic mask scope");
                pivotMask = (1ULL << n_) - 1ULL;
            }
            for (uint32_t k = 0; k < n_; ++k) {
                if (!SmallRawPivotStep<DEFER_PROD, FULL_RAGGED_ROW, DIAGONAL_INDUCTION, DYNAMIC_SEARCH_MASK,
                                       GUARDED_TAIL2, PANEL_COMPENSATE, SPR_MODE, batchedUpdate, earlyBroadcast,
                                       normalMask, paddedUpdate>(
                        cachedAbsUB, cachedColumnUB, cachedMatrixUB, cachedOffsetUB, cachedReduceUB, cachedRowUB,
                        diagonalOffset, ex, fastProd, k, pivotMask, prod, sign, singular))
                    break;
            }
        }
        ProcessSmallRawUBSetFlagPhase16(ex, m, prod, sign, singular, tiny, vMte2Pend);
    }
    template <bool DEFER_PROD = false, bool FULL_RAGGED_ROW = false, bool DIAGONAL_INDUCTION = false,
              bool DYNAMIC_SEARCH_MASK = false, bool GUARDED_TAIL2 = false, bool PANEL_COMPENSATE = false,
              uint32_t SPR_MODE = 0U>
    __aicore__ inline void ProcessSmallRawUB() {
        SmallRawViews51 views = GetSmallRawViews51();
        constexpr bool normalMask = LD_BATCH == 3U || LD_BATCH == 6U;
        constexpr bool batchedUpdate = LdetRouteKeyIs<1U, 3U, 5U, 6U, 15U>(LD_BATCH);
        constexpr bool paddedUpdate = LdetRouteKeyIs<5U, 6U, 15U>(LD_BATCH);
        constexpr bool earlyBroadcast = batchedUpdate && LD_BATCH != 15U;
        uint32_t nn32 = n_ * n_;
        uint32_t vMte2Pend = 0;
        bool tiny = SPR_MODE == 2U ? false : (n_ <= 8U);
        if (tiny && cnt_ * nn32 <= ::kSmallMaxN * ::kSmallMaxN) {
            return ProcessSmallRawUBDataCopyPadPhase13(nn32);
        }
        for (uint32_t m = 0; m < cnt_; ++m) {
            SmallRawMatrixStep<DEFER_PROD, FULL_RAGGED_ROW, DIAGONAL_INDUCTION, DYNAMIC_SEARCH_MASK, GUARDED_TAIL2,
                               PANEL_COMPENSATE, SPR_MODE, batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                views.absolute, views.column, views.matrix, views.offsets, views.reduction, views.row, m, nn32, tiny,
                vMte2Pend);
        }
    }

    __aicore__ inline void ProcessMediumRawFixedDataCopyPhase1DataCopyPadDepth22Stage1(uint64_t &g,
                                                                                       const uint32_t &rawN,
                                                                                       const uint64_t &rawNN,
                                                                                       const uint32_t &rawS) {

        if constexpr (LD_BATCH == 6U || PAD128) {
            for (uint32_t r0 = 0; r0 < rawN; r0 += 32U) {
                uint32_t rows = rawN - r0;
                if (rows > 32U)
                    rows = 32U;
                DataCopyExtParams cp{static_cast<uint16_t>(rows), static_cast<uint32_t>(rawN * sizeof(float)), 0,
                                     PAD128 ? (rawS - ((rawN + 7U) & ~7U)) / 8U : 0U, 0};
                DataCopyPad(mat_[r0 * rawS], aGm_[g * rawNN + static_cast<uint64_t>(r0) * rawN], cp,
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        } else {
            for (uint32_t i = 0; i < rawN; ++i) {
                DataCopyPad(mat_[i * rawS], aGm_[g * rawNN + static_cast<uint64_t>(i) * rawN],
                            {1, static_cast<uint32_t>(rawN * sizeof(float)), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        }
    }

    __aicore__ inline void ProcessMediumRawFixedDataCopyPhase1(uint64_t &g, uint32_t &m, uint32_t &nn32,
                                                               const uint32_t &rawN, const uint64_t &rawNN,
                                                               const uint32_t &rawS, bool &tiny, uint32_t &vMte2Pend) {

        WaitSmallInputReuse64(m, tiny, vMte2Pend);
        if (firstCopyPending_) {
            firstCopyPending_ = false;
        } else if (rawN == rawS) {
            DataCopy(mat_, aGm_[g * rawNN], nn32);
        } else if (tiny) {
            DataCopyPad(mat_[0], aGm_[g * rawNN], {1, static_cast<uint32_t>(nn32 * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            ProcessMediumRawFixedDataCopyPhase1DataCopyPadDepth22Stage1(g, rawN, rawNN, rawS);
        }
    }
    template <auto earlyBroadcast, auto normalMask>
    __aicore__ inline void SharedScalarUpdateStage30(__ubuf__ float *&cachedColumnUB, int32_t &ex, bool &fastProd,
                                                     uint32_t &k, uint32_t &p, float &piv, float &prod, float &sign) {
        if constexpr (earlyBroadcast) {
            piv = cachedColumnUB[p - k];
            if (p != k)
                sign = -sign;
        } else {
            if (p != k) {
                float tv = cachedColumnUB[0];
                colbuf_.SetValue(0, cachedColumnUB[p - k]);
                colbuf_.SetValue(p - k, tv);
                sign = -sign;
            }
            piv = cachedColumnUB[0];
        }
        if constexpr (!earlyBroadcast) {
            if constexpr (normalMask) {
                ProdSmallCommon(fastProd, prod, ex, piv);
            } else {
                ProdAccum(prod, ex, piv);
            }
        }
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask>
    __aicore__ inline void
    ProcessMediumRawFixedvmulsPhase2(__ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB,
                                     __ubuf__ float *&cachedRowUB, int32_t &ex, bool &fastProd, uint32_t &k,
                                     uint32_t &kA, uint32_t &p, float &piv, float &prod, const uint32_t &rawS,
                                     float &sign, uint32_t &wA) {

        SharedScalarUpdateStage30<earlyBroadcast, normalMask>(cachedColumnUB, ex, fastProd, k, p, piv, prod, sign);
        float inv = (batchedUpdate ? -1.0f : 1.0f) / piv;
        SmallMaskSet(wA);
        if (p != k) {
            __ubuf__ float *mp = reinterpret_cast<__ubuf__ float *>(cachedMatrixUB);
            vmuls(reinterpret_cast<__ubuf__ float *>(cachedRowUB), mp + (p * rawS + kA), inv, 1, 1, 1, 8, 8);
            vmuls(mp + (p * rawS + kA), mp + (k * rawS + kA), 1.0f, 1, 1, 1, 8, 8);
        } else {
            vmuls(reinterpret_cast<__ubuf__ float *>(cachedRowUB),
                  reinterpret_cast<__ubuf__ float *>(cachedMatrixUB) + (k * rawS + kA), inv, 1, 1, 1, 8, 8);
        }
        SmallMaskReset();
    }
    template <auto paddedUpdate>
    __aicore__ inline void SharedVmlaStage37(uint32_t &c0, __ubuf__ float *&cachedMatrixUB,
                                             __ubuf__ float *&cachedRowUB, uint32_t &cnt, LocalTensor<float> &factors,
                                             uint32_t &k, uint32_t &kA, const uint32_t &rawN, const uint32_t &rawS) {
        for (uint32_t col = c0; col < rawN; col += 64U) {
            uint32_t width = paddedUpdate ? rawS - col : rawN - col;
            if (width > 64U)
                width = 64U;
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            vmla(reinterpret_cast<__ubuf__ float *>((cachedMatrixUB + (k + 1U) * rawS + col)),
                 reinterpret_cast<__ubuf__ float *>((cachedRowUB + col - kA)),
                 reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()), static_cast<uint8_t>(cnt - 1U), 1, 1, 0,
                 static_cast<uint8_t>(rawS / 8U), 0, 1);
        }
    }
    template <auto earlyBroadcast, auto normalMask>
    __aicore__ inline void SharedVmulsStage41(__ubuf__ float *&cachedColumnUB, uint32_t &cnt,
                                              LocalTensor<float> &factors, uint32_t &k, uint32_t &p) {
        if constexpr (earlyBroadcast) {
            Shared24VmulsStage19(factors, k, p);
        } else {
            if constexpr (normalMask)
                ResetMask();
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB), 1, 8, static_cast<uint8_t>((cnt + 7U) / 8U));
            PipeBarrier<PIPE_V>();
        }
    }
    template <auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessMediumRawFixedvmulsPhase3(uint32_t &c0, __ubuf__ float *&cachedColumnUB,
                                                            __ubuf__ float *&cachedMatrixUB,
                                                            __ubuf__ float *&cachedRowUB, uint32_t &cnt,
                                                            LocalTensor<float> &factors, uint32_t &k, uint32_t &kA,
                                                            uint32_t &p, const uint32_t &rawN, const uint32_t &rawS) {

        if (cnt > 1U) {
            SharedVmulsStage41<earlyBroadcast, normalMask>(cachedColumnUB, cnt, factors, k, p);
            SharedVmlaStage37<paddedUpdate>(c0, cachedMatrixUB, cachedRowUB, cnt, factors, k, kA, rawN, rawS);
        }
    }
    template <auto normalMask>
    __aicore__ inline void ProcessMediumRawFixedSetValuePhase4(__ubuf__ float *&cachedMatrixUB, int32_t &ex,
                                                               bool &fastProd, uint32_t &k, uint32_t &ld_, uint32_t &p,
                                                               float &prod, const uint32_t &rawN, float &sign) {
        ScalarTinyStep56_1<normalMask, true>(rawN, ld_, k, p, cachedMatrixUB, ex, fastProd, prod, sign);
    }

    template <bool NEGATE>
    __aicore__ inline void ApplyRawFixedRows51(uint32_t &c0, __ubuf__ float *&cachedColumnUB,
                                               __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB,
                                               uint32_t &i, uint32_t &k, uint32_t &kA, const uint32_t &rawN,
                                               const uint32_t &rawS, uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>((cachedMatrixUB + (k + 1U) * rawS + c0));
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>((cachedRowUB + c0 - kA));
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>((cachedColumnUB + 1));
            UpdateRowsAxpy61<NEGATE>(rawN, rawS, i, dstP, rkP, lP);
        }
    }
    template <auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessMediumRawFixedPipeBarrierPhase6(uint32_t &c0, __ubuf__ float *&cachedColumnUB,
                                           __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB, uint32_t &cnt,
                                           LocalTensor<float> &factors, uint32_t &i, uint32_t &k, uint32_t &kA,
                                           const uint32_t &rawN, const uint32_t &rawS, uint32_t &w, uint32_t &wA) {

        if (wA > 64U) {
            ApplyRawFixedRows51<false>(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, rawN, rawS, w);
        } else {
            if (cnt > 1U) {
                if constexpr (normalMask)
                    ResetMask();
                LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB), 1, 8,
                      static_cast<uint8_t>((cnt + 7U) / 8U));
                PipeBarrier<PIPE_V>();
                SharedVmlaStage37<paddedUpdate>(c0, cachedMatrixUB, cachedRowUB, cnt, factors, k, kA, rawN, rawS);
            }
        }
    }

    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessMediumRawFixedconstexprPhase8(
        __ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB, uint32_t &cnt,
        int32_t &ex, LocalTensor<float> &factors, bool &fastProd, uint32_t &k, uint32_t &kA, uint32_t &p, float &prod,
        const uint32_t &rawN, const uint32_t &rawS, float &sign) {

        uint32_t wA = rawS - kA;
        float piv;
        ProcessMediumRawFixedvmulsPhase2<batchedUpdate, earlyBroadcast, normalMask>(
            cachedColumnUB, cachedMatrixUB, cachedRowUB, ex, fastProd, k, kA, p, piv, prod, rawS, sign, wA);
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = rawN - c0;
        uint32_t i = k + 1;
        if constexpr (LD_BATCH == 15U) {
            ProcessMediumRawFixedPipeBarrierPhase6<normalMask, paddedUpdate>(
                c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, factors, i, k, kA, rawN, rawS, w, wA);
        } else if constexpr (batchedUpdate) {
            ProcessMediumRawFixedvmulsPhase3<earlyBroadcast, normalMask, paddedUpdate>(
                c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, factors, k, kA, p, rawN, rawS);
        } else {
            ApplyRawFixedRows51<true>(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, rawN, rawS, w);
        }
        CompleteEarlyProduct64<earlyBroadcast, normalMask>(fastProd, prod, ex, piv);
    }
    template <auto normalMask>
    __aicore__ inline void ProcessMediumRawFixedStatePhase9(__ubuf__ float *&cachedMatrixUB, int32_t &ex,
                                                            bool &fastProd, uint32_t &ld_, float &prod,
                                                            const uint32_t &rawN, float &sign, uint32_t &singular) {

        for (uint32_t k = 0; k < rawN; ++k) {
            uint32_t p = k;
            float maxv = 0.0f;
            if (!FindScalarTinyPivot64<true>(cachedMatrixUB, rawN, ld_, k, maxv, p, singular))
                break;
            ProcessMediumRawFixedSetValuePhase4<normalMask>(cachedMatrixUB, ex, fastProd, k, ld_, p, prod, rawN, sign);
        }
    }

    __aicore__ inline void ProcessMediumRawFixedDataCopyPadPhase10(uint32_t &nn32, const uint64_t &rawNN) {
        CopyAndRunTiny56_1(rawNN, nn32);
    }
    template <auto normalMask>
    __aicore__ inline void ProcessMediumRawFixedconstexprPhase11(__ubuf__ float *&cachedReduceUB, uint32_t &k,
                                                                 float &maxv, uint32_t &p, uint32_t &segN) {

        if constexpr (normalMask) {
            maxv = cachedReduceUB[0];
            float idxF = cachedReduceUB[1];
            p = k + *reinterpret_cast<uint32_t *>(&idxF);
        } else {
            MergeMaxSegments64<true>(cachedReduceUB, segN, k, maxv, p);
        }
    }

    __aicore__ inline void ProcessMediumRawFixedSetFlagPhase13(int32_t &ex, uint32_t &m, float &prod, float &sign,
                                                               uint32_t &singular, bool &tiny, uint32_t &vMte2Pend) {
        return ProcessSmallRawUBSetFlagPhase16(ex, m, prod, sign, singular, tiny, vMte2Pend);
    }
    template <auto earlyBroadcast>
    __aicore__ inline void ProcessMediumRawFixedSetFlagPhase14(__ubuf__ float *&cachedColumnUB, uint32_t &cnt,
                                                               LocalTensor<float> &factors) {
        return ProcessSmallRawUBSetFlagPhase17<earlyBroadcast>(cachedColumnUB, cnt, factors);
    }
