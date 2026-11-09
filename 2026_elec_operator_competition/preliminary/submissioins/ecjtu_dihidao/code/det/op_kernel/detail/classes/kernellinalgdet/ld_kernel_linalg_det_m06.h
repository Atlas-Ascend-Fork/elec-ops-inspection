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
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallSetFlagPhase14(uint32_t &nn32, bool &tiny, uint32_t &vMte2Pend) {

        for (uint32_t m = 0; m < cnt_; ++m) {
            ProcessSmallDataCopyPhase1(m, nn32, tiny, vMte2Pend);
            WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
            if (!tiny) {
                SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
                WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
            }
            uint32_t ld_ = (tiny && n_ != s_) ? n_ : s_;
            float sign = 1.0f;
            float prod = 1.0f;
            int32_t ex = 0;
            bool fastProd = true;
            uint32_t singular = 0;
            if (tiny) {
                ProcessSmallGetValuePhase8<normalMask>(ex, fastProd, ld_, prod, sign, singular);
            } else {
                ProcessSmallvgatherPhase13<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(ex, fastProd, prod,
                                                                                                    sign, singular);
            }
            StoreSmallResult64(m, tiny, vMte2Pend, singular, sign, prod, ex);
        }
    }
    __aicore__ inline void ProcessSmall() {
        constexpr bool normalMask = LD_BATCH == 3U || LD_BATCH == 6U;
        constexpr bool batchedUpdate = LdetRouteKeyIs<1U, 3U, 5U, 6U, 15U>(LD_BATCH);
        constexpr bool paddedUpdate = LdetRouteKeyIs<5U, 6U, 15U>(LD_BATCH);
        constexpr bool earlyBroadcast = batchedUpdate && LD_BATCH != 15U;
        uint32_t nn32 = n_ * n_;
        uint32_t vMte2Pend = 0;
        bool tiny = (n_ <= 8U);
        if (tiny && cnt_ * nn32 <= ::kSmallMaxN * ::kSmallMaxN) {
            return ProcessSmallDataCopyPadPhase9(nn32);
        }
        ProcessSmallSetFlagPhase14<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(nn32, tiny, vMte2Pend);
    }

    __aicore__ inline void ProcessSmallRawUBDataCopyPhase1(uint64_t &g, uint32_t &m, uint32_t &nn32, bool &tiny,
                                                           uint32_t &vMte2Pend) {

        WaitSmallInputReuse64(m, tiny, vMte2Pend);
        if (firstCopyPending_) {
            firstCopyPending_ = false;
        } else if (n_ == s_) {
            DataCopy(mat_, aGm_[g * nn_], nn32);
        } else if (tiny) {
            DataCopyPad(mat_[0], aGm_[g * nn_], {1, static_cast<uint32_t>(nn32 * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            SharedDataCopyPadStage17(g);
        }
    }
    template <bool FULL_RAGGED_ROW, bool DIAGONAL_INDUCTION, bool DYNAMIC_SEARCH_MASK, uint32_t SPR_MODE,
              auto normalMask>
    __aicore__ inline void ProcessSmallRawUBconstexprPhase2(__ubuf__ float *&cachedReduceUB, uint32_t &k, float &maxv,
                                                            uint32_t &p, uint32_t &segN, bool &sprPositive) {

        if constexpr (normalMask) {
            if constexpr (SPR_MODE == 2U) {
                static_assert(LD_BATCH == 6U && FULL_RAGGED_ROW && DIAGONAL_INDUCTION && DYNAMIC_SEARCH_MASK,
                              "Raw SPR normal contiguous recurrence scope");
                uint32_t valueBits = 0U, indexBits = 0U;
                AscendC::GetReduceRepeatMaxMinSpr<float>(valueBits, indexBits);
                sprPositive = valueBits > 0U && valueBits <= 0x7f800000U;
                p = k + indexBits;
            } else if constexpr (SPR_MODE == 1U) {
                float indexFloat = 0.0f;
                AscendC::GetReduceRepeatMaxMinSpr<float>(maxv, indexFloat);
                p = k + *reinterpret_cast<uint32_t *>(&indexFloat);
            } else {
                maxv = cachedReduceUB[0];
                float idxF = cachedReduceUB[1];
                p = k + *reinterpret_cast<uint32_t *>(&idxF);
            }
        } else {
            MergeMaxSegments64<true>(cachedReduceUB, segN, k, maxv, p);
        }
    }
    template <bool FULL_RAGGED_ROW, bool PANEL_COMPENSATE>
    __aicore__ inline void ProcessSmallRawUBvmulsPhase3(__ubuf__ float *&cachedAbsUB, __ubuf__ float *&cachedMatrixUB,
                                                        __ubuf__ float *&cachedRowUB, LocalTensor<float> &factors,
                                                        float &inv, uint32_t &k, uint32_t &kA, uint32_t &p, float &piv,
                                                        float &sign) {

        if constexpr (FULL_RAGGED_ROW) {
            if constexpr (PANEL_COMPENSATE) {
                if ((k & 7U) == 0U) {
                    vmuls(cachedAbsUB, cachedMatrixUB + p * s_, 1.0f, 1, 1, 1, 8, 8);
                    vmuls(cachedRowUB, cachedMatrixUB + p * s_, inv, 1, 1, 1, 8, 8);
                    PipeBarrier<PIPE_V>();
                    vaxpy(cachedAbsUB, cachedRowUB, piv, 1, 1, 1, 8, 8);
                    PipeBarrier<PIPE_V>();
                    vaxpy(cachedRowUB, cachedAbsUB, inv, 1, 1, 1, 8, 8);
                } else {
                    vmuls(cachedRowUB, cachedMatrixUB + p * s_, inv, 1, 1, 1, 8, 8);
                }
            } else {
                vmuls(cachedRowUB, cachedMatrixUB + p * s_, inv, 1, 1, 1, 8, 8);
            }
            if (p != k) {
                sign = -sign;
                vmuls(cachedMatrixUB + p * s_, cachedMatrixUB + k * s_, 1.0f, 1, 1, 1, 8, 8);
                AscendCUtils::SetMask<float>(static_cast<int32_t>(8));
                vmuls(reinterpret_cast<__ubuf__ float *>(factors[(p - k) * 8U].GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
            }
        } else {
            if (p != k) {
                __ubuf__ float *mp = reinterpret_cast<__ubuf__ float *>(cachedMatrixUB);
                vmuls(reinterpret_cast<__ubuf__ float *>(cachedRowUB), mp + (p * s_ + kA), inv, 1, 1, 1, 8, 8);
                vmuls(mp + (p * s_ + kA), mp + (k * s_ + kA), 1.0f, 1, 1, 1, 8, 8);
            } else {
                vmuls(reinterpret_cast<__ubuf__ float *>(cachedRowUB),
                      reinterpret_cast<__ubuf__ float *>(cachedMatrixUB) + (k * s_ + kA), inv, 1, 1, 1, 8, 8);
            }
        }
    }
    template <bool DEFER_PROD, bool FULL_RAGGED_ROW, bool PANEL_COMPENSATE, auto batchedUpdate, auto earlyBroadcast,
              auto normalMask>
    __aicore__ inline void
    ProcessSmallRawUBconstexprPhase4(__ubuf__ float *&cachedAbsUB, __ubuf__ float *&cachedColumnUB,
                                     __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB, int32_t &ex,
                                     LocalTensor<float> &factors, bool &fastProd, uint32_t &k, uint32_t &kA,
                                     uint32_t &p, float &piv, float &prod, float &sign, uint32_t &wA) {

        if constexpr (earlyBroadcast) {
            piv = cachedColumnUB[p - k];
            if constexpr (!FULL_RAGGED_ROW) {
                if (p != k)
                    sign = -sign;
            }
        } else {
            if (p != k) {
                float tv = cachedColumnUB[0];
                colbuf_.SetValue(0, cachedColumnUB[p - k]);
                colbuf_.SetValue(p - k, tv);
                sign = -sign;
            }
            piv = cachedColumnUB[0];
        }
        if constexpr (!earlyBroadcast && !DEFER_PROD) {
            if constexpr (normalMask) {
                ProdSmallCommon(fastProd, prod, ex, piv);
            } else {
                ProdAccum(prod, ex, piv);
            }
        }
        float inv = (batchedUpdate ? -1.0f : 1.0f) / piv;
        SmallMaskSet(wA);
        ProcessSmallRawUBvmulsPhase3<FULL_RAGGED_ROW, PANEL_COMPENSATE>(cachedAbsUB, cachedMatrixUB, cachedRowUB,
                                                                        factors, inv, k, kA, p, piv, sign);
        SmallMaskReset();
    }
    template <auto normalMask>
    __aicore__ inline void ProcessSmallRawUBSetValuePhase5(__ubuf__ float *&cachedMatrixUB, int32_t &ex, bool &fastProd,
                                                           uint32_t &k, uint32_t &ld_, uint32_t &p, float &prod,
                                                           float &sign) {
        ScalarTinyStep56_1<normalMask, true>(n_, ld_, k, p, cachedMatrixUB, ex, fastProd, prod, sign);
    }

    __aicore__ inline void ProcessSmallRawUBvaxpyPhase6(uint32_t &c0, __ubuf__ float *&cachedColumnUB,
                                                        __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB,
                                                        uint32_t &i, uint32_t &k, uint32_t &kA, uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>((cachedMatrixUB + (k + 1U) * s_ + c0));
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>((cachedRowUB + c0 - kA));
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>((cachedColumnUB + 1));
            SharedVaxpyStage14(dstP, i, lP, rkP);
        }
    }
    template <bool FULL_RAGGED_ROW, auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallRawUBPipeBarrierPhase7(uint32_t &c0, __ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB,
                                       __ubuf__ float *&cachedRowUB, uint32_t &cnt, LocalTensor<float> &factors,
                                       uint32_t &i, uint32_t &k, uint32_t &kA, uint32_t &w, uint32_t &wA) {

        if (wA > 64U) {
            ProcessSmallRawUBvaxpyPhase6(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, w);
        } else {
            if (cnt > 1U) {
                if constexpr (normalMask)
                    ResetMask();
                LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB), 1, 8,
                      static_cast<uint8_t>((cnt + 7U) / 8U));
                PipeBarrier<PIPE_V>();
                for (uint32_t col = c0; col < n_; col += 64U) {
                    uint32_t width = paddedUpdate ? s_ - col : n_ - col;
                    if (width > 64U)
                        width = 64U;
                    AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
                    if constexpr (FULL_RAGGED_ROW)
                        PipeBarrier<PIPE_V>();
                    vmla(reinterpret_cast<__ubuf__ float *>((cachedMatrixUB + (k + 1U) * s_ + col)),
                         reinterpret_cast<__ubuf__ float *>((cachedRowUB + col - kA)),
                         reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()), static_cast<uint8_t>(cnt - 1U), 1,
                         1, 0, static_cast<uint8_t>(s_ / 8U), 0, 1);
                }
            }
        }
    }

    __aicore__ inline void ProcessSmallRawUBvaxpyPhase8(uint32_t &c0, __ubuf__ float *&cachedColumnUB,
                                                        __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB,
                                                        uint32_t &i, uint32_t &k, uint32_t &kA, uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>((cachedMatrixUB + (k + 1U) * s_ + c0));
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>((cachedRowUB + c0 - kA));
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>((cachedColumnUB + 1));
            SharedVaxpyStage12(dstP, i, lP, rkP);
        }
    }
    template <auto normalMask>
    __aicore__ inline void ProcessSmallRawUBStatePhase9(__ubuf__ float *&cachedMatrixUB, int32_t &ex, bool &fastProd,
                                                        uint32_t &ld_, float &prod, float &sign, uint32_t &singular) {

        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t p = k;
            float maxv = 0.0f;
            if (!FindScalarTinyPivot64<true>(cachedMatrixUB, n_, ld_, k, maxv, p, singular))
                break;
            ProcessSmallRawUBSetValuePhase5<normalMask>(cachedMatrixUB, ex, fastProd, k, ld_, p, prod, sign);
        }
    }

    __aicore__ inline void Shared24VmlaStage18(__ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB,
                                               uint32_t &cnt, uint32_t &col, LocalTensor<float> &factors, uint32_t &k,
                                               uint32_t &kA, uint32_t &width) {
        if (width > 64U)
            width = 64U;
        AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
        vmla(reinterpret_cast<__ubuf__ float *>((cachedMatrixUB + (k + 1U) * s_ + col)),
             reinterpret_cast<__ubuf__ float *>((cachedRowUB + col - kA)),
             reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()), static_cast<uint8_t>(cnt - 1U), 1, 1, 0,
             static_cast<uint8_t>(s_ / 8U), 0, 1);
    }
    template <auto paddedUpdate>
    __aicore__ inline void SharedVmlaStage36(uint32_t &c0, __ubuf__ float *&cachedMatrixUB,
                                             __ubuf__ float *&cachedRowUB, uint32_t &cnt, LocalTensor<float> &factors,
                                             uint32_t &k, uint32_t &kA) {
        for (uint32_t col = c0; col < n_; col += 64U) {
            uint32_t width = paddedUpdate ? s_ - col : n_ - col;
            Shared24VmlaStage18(cachedMatrixUB, cachedRowUB, cnt, col, factors, k, kA, width);
        }
    }
    template <bool FULL_RAGGED_ROW, auto paddedUpdate>
    __aicore__ inline void ProcessSmallRawUBPipeBarrierPhase10(uint32_t &c0, __ubuf__ float *&cachedMatrixUB,
                                                               __ubuf__ float *&cachedRowUB, uint32_t &cnt,
                                                               LocalTensor<float> &factors, uint32_t &k, uint32_t &kA) {

        if constexpr (FULL_RAGGED_ROW) {
            PipeBarrier<PIPE_V>();
            AscendCUtils::SetMask<float>(static_cast<int32_t>(s_));
            vmla(cachedMatrixUB + (k + 1U) * s_, cachedRowUB,
                 reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()), static_cast<uint8_t>(cnt - 1U), 1, 1, 0,
                 static_cast<uint8_t>(s_ / 8U), 0, 1);
        } else {
            SharedVmlaStage36<paddedUpdate>(c0, cachedMatrixUB, cachedRowUB, cnt, factors, k, kA);
        }
    }
    template <bool FULL_RAGGED_ROW, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallRawUBvmulsPhase11vmulsDepth22Stage1(
        uint32_t &c0, __ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB,
        uint32_t &cnt, LocalTensor<float> &factors, uint32_t &k, uint32_t &kA, uint32_t &p) {

        if (cnt > 1U) {
            if constexpr (earlyBroadcast) {
                if constexpr (!FULL_RAGGED_ROW) {
                    if (p != k) {
                        AscendCUtils::SetMask<float>(static_cast<int32_t>(8));
                        vmuls(reinterpret_cast<__ubuf__ float *>(factors[(p - k) * 8U].GetPhyAddr()),
                              reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
                    }
                }
            } else {
                if constexpr (normalMask)
                    ResetMask();
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB), 1, 8,
                      static_cast<uint8_t>((cnt + 7U) / 8U));
                PipeBarrier<PIPE_V>();
            }
            ProcessSmallRawUBPipeBarrierPhase10<FULL_RAGGED_ROW, paddedUpdate>(c0, cachedMatrixUB, cachedRowUB, cnt,
                                                                               factors, k, kA);
        }
    }
    template <bool FULL_RAGGED_ROW, auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallRawUBvmulsPhase11(uint32_t &c0, __ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB,
                                  __ubuf__ float *&cachedRowUB, uint32_t &cnt, LocalTensor<float> &factors, uint32_t &i,
                                  uint32_t &k, uint32_t &kA, uint32_t &p, uint32_t &w, uint32_t &wA) {

        if constexpr (LD_BATCH == 15U) {
            ProcessSmallRawUBPipeBarrierPhase7<FULL_RAGGED_ROW, normalMask, paddedUpdate>(
                c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, factors, i, k, kA, w, wA);
        } else if constexpr (batchedUpdate) {
            ProcessSmallRawUBvmulsPhase11vmulsDepth22Stage1<FULL_RAGGED_ROW, earlyBroadcast, normalMask, paddedUpdate>(
                c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, factors, k, kA, p);
        } else {
            ProcessSmallRawUBvaxpyPhase8(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, w);
        }
    }
    template <bool DEFER_PROD, bool FULL_RAGGED_ROW, bool PANEL_COMPENSATE, auto batchedUpdate, auto earlyBroadcast,
              auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallRawUBSmallMaskResetPhase12(__ubuf__ float *&cachedAbsUB, __ubuf__ float *&cachedColumnUB,
                                           __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB, uint32_t &cnt,
                                           int32_t &ex, LocalTensor<float> &factors, bool &fastProd, uint32_t &k,
                                           uint32_t &kA, uint32_t &p, float &prod, float &sign) {

        uint32_t wA = s_ - kA;
        float piv;
        ProcessSmallRawUBconstexprPhase4<DEFER_PROD, FULL_RAGGED_ROW, PANEL_COMPENSATE, batchedUpdate, earlyBroadcast,
                                         normalMask>(cachedAbsUB, cachedColumnUB, cachedMatrixUB, cachedRowUB, ex,
                                                     factors, fastProd, k, kA, p, piv, prod, sign, wA);
        uint32_t c0 = FULL_RAGGED_ROW ? 0U : (k + 1) & ~(::kAlign - 1);
        uint32_t w = n_ - c0;
        uint32_t i = k + 1;
        ProcessSmallRawUBvmulsPhase11<FULL_RAGGED_ROW, batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
            c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, factors, i, k, kA, p, w, wA);
        SmallMaskReset();
        if constexpr (earlyBroadcast || DEFER_PROD) {
            if constexpr (normalMask) {
                ProdSmallCommon(fastProd, prod, ex, piv);
            } else {
                ProdAccum(prod, ex, piv);
            }
        }
    }
    __aicore__ inline void ProcessSmallRawUBDataCopyPadPhase13(uint32_t &nn32) {
        return ProcessSmallDataCopyPadPhase9(nn32);
    }
    template <bool DIAGONAL_INDUCTION>
    __aicore__ inline void
    ProcessSmallRawUBvgatherPhase14(__ubuf__ float *&cachedAbsUB, __ubuf__ float *&cachedColumnUB,
                                    __ubuf__ float *&cachedMatrixUB, __ubuf__ uint32_t *&cachedOffsetUB, uint32_t &cnt,
                                    uint32_t &diagonalOffset, uint32_t &k, uint64_t &pivotMask) {

        uint32_t be = DIAGONAL_INDUCTION ? diagonalOffset : k * (s_ + 1U);
        if constexpr (DIAGONAL_INDUCTION)
            diagonalOffset += s_ + 1U;
        if constexpr (DIAGONAL_INDUCTION) {
            AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(0ULL, pivotMask);
            pivotMask >>= 1U;
        } else {
            SmallMaskSet(cnt);
        }
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB),
                reinterpret_cast<__ubuf__ uint32_t *>((cachedOffsetUB + (be & 7U) * ::kSmallMaxN)),
                reinterpret_cast<uint64_t>(cachedMatrixUB) + (be & ~7U) * sizeof(float), 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(reinterpret_cast<__ubuf__ float *>(cachedAbsUB), reinterpret_cast<__ubuf__ float *>(cachedColumnUB), 1, 1,
             1, 8, 8);
    }
    template <auto normalMask> __aicore__ inline void ProcessSmallRawUBconstexprPhase15(uint32_t &cnt) {
        return ProcessSmallconstexprPhase11<normalMask>(cnt);
    }

    __aicore__ inline void ProcessSmallRawUBSetFlagPhase16(int32_t &ex, uint32_t &m, float &prod, float &sign,
                                                           uint32_t &singular, bool &tiny, uint32_t &vMte2Pend) {

        StoreSmallResult64(m, tiny, vMte2Pend, singular, sign, prod, ex);
    }
