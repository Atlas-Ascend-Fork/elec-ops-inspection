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
    template <uint32_t TN, bool EARLY_FACTOR, bool ACTIVE_N, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallFixedN64CachedvmulsPhase2(uint32_t &c0, uint32_t &cnt, LocalTensor<float> &factors,
                                          __ubuf__ float *&fixedMatrixUB, __ubuf__ float *&fixedRowUB, uint32_t &k,
                                          uint32_t &kA, uint32_t &p) {

        if (cnt > 1U) {
            if (!EARLY_FACTOR && p != k) {
                AscendCUtils::SetMask<float>(static_cast<int32_t>(8));
                vmuls(reinterpret_cast<__ubuf__ float *>(factors[(p - k) * 8U].GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
            }
            for (uint32_t col = c0; col < TN; col += 64U) {
                uint32_t width = paddedUpdate ? (TN + 8U) - col : TN - col;
                if (width > 64U)
                    width = 64U;
                if constexpr (ACTIVE_N) {
                    AscendCUtils::SetMask<float>(0ULL, ~0ULL >> col);
                } else {
                    AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
                }
                if constexpr (TN == 64U) {
                    vmla(reinterpret_cast<__ubuf__ float *>((fixedMatrixUB + (k + 1U) * (TN + 8U) + col)),
                         reinterpret_cast<__ubuf__ float *>((fixedRowUB + col - kA)),
                         reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()), static_cast<uint8_t>(cnt - 1U), 1,
                         1, 0, static_cast<uint8_t>((TN + 8U) / 8U), 0, 1);
                } else {
                    vmla(reinterpret_cast<__ubuf__ float *>((fixedMatrixUB + (k + 1U) * (TN + 8U) + col)),
                         reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()),
                         reinterpret_cast<__ubuf__ float *>((fixedRowUB + col - kA)), static_cast<uint8_t>(cnt - 1U), 1,
                         0, 1, static_cast<uint8_t>((TN + 8U) / 8U), 1, 0);
                }
            }
        }
    }
    template <uint32_t TN, bool EARLY_FACTOR, bool DIRECT_GATHER, bool NORMAL_SPR, bool RAW_SPR, bool ACTIVE_N>
    __aicore__ inline void ProcessSmallFixedN64CachedSetFlagPhase3(uint64_t &g, uint32_t &m, uint32_t &nn32,
                                                                   bool &tiny) {

        bool firstCopyAlreadyIssued = false;
        if constexpr (TN == 64U && EARLY_FACTOR && DIRECT_GATHER && NORMAL_SPR) {
            firstCopyAlreadyIssued = (m == 0U && firstCopyPending_);
        }
        ProcessSmallFixedN64CachedDataCopyPhase1<TN>(firstCopyAlreadyIssued, g, nn32, tiny);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        if (!tiny) {
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
            WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        }
        if constexpr (TN == 64U && EARLY_FACTOR && DIRECT_GATHER && NORMAL_SPR && RAW_SPR && !ACTIVE_N) {
            if (firstCopyAlreadyIssued) {
                // Copy raw bits, including subnormals and signed zero, into original stride72.
                // The ACTIVE_N prepared-capacity route retains its original padded input.
                LocalTensor<uint16_t> stageSrc(TPosition::VECCALC, ::kSpMat + 8192U * sizeof(float), 8192U);
                LocalTensor<uint16_t> stageDst(TPosition::VECCALC, ::kSpMat, 64U * 72U * 2U);
                Or<uint16_t>(stageDst, stageSrc, stageSrc, static_cast<uint64_t>(128U), static_cast<uint8_t>(64U),
                             {1, 1, 1, 9, 8, 8});
                PipeBarrier<PIPE_V>();
            }
        }
    }
    template <uint32_t TN, bool NORMAL_SPR, bool RAW_SPR, auto normalMask>
    __aicore__ inline void ProcessSmallFixedN64CachedconstexprPhase4(__ubuf__ float *&fixedReduceUB, uint32_t &k,
                                                                     float &maxv, uint32_t &p, uint32_t &rawAbsBits,
                                                                     uint32_t &segN) {

        if constexpr (RAW_SPR) {
            static_assert(TN == 64U && NORMAL_SPR, "capacity raw SPR scope");
            uint32_t rawIndexBits = 0U;
            AscendC::GetReduceRepeatMaxMinSpr<float>(rawAbsBits, rawIndexBits);
            p = k + rawIndexBits;
        } else {
            if constexpr (normalMask) {
                float idxF;
                if constexpr (NORMAL_SPR) {
                    static_assert(TN == 64U, "Normal SPR n64 scope");
                    AscendC::GetReduceRepeatMaxMinSpr<float>(maxv, idxF);
                } else {
                    maxv = fixedReduceUB[0];
                    idxF = fixedReduceUB[1];
                }
                p = k + *reinterpret_cast<uint32_t *>(&idxF);
            } else {
                MergeMaxSegments64<true>(fixedReduceUB, segN, k, maxv, p);
            }
        }
    }
    template <uint32_t TN, auto normalMask>
    __aicore__ inline void
    ProcessSmallFixedN64CachedSetValuePhase5(int32_t &ex, bool &fastProd, __ubuf__ float *&fixedMatrixUB, uint32_t &k,
                                             uint32_t &ld_, uint32_t &p, float &prod, float &sign) {
        ScalarTinyStep56_1<normalMask, true>(TN, ld_, k, p, fixedMatrixUB, ex, fastProd, prod, sign);
    }
    template <uint32_t TN>
    __aicore__ inline void SharedVaxpyStage5(__ubuf__ float *&dstP, uint32_t &i, __ubuf__ float *&lP,
                                             __ubuf__ float *&rkP) {
        UpdateRowsAxpy61<true>(TN, (TN + 8U), i, dstP, rkP, lP);
    }
    template <uint32_t TN>
    __aicore__ inline void ProcessSmallFixedN64CachedvaxpyPhase6(uint32_t &c0, __ubuf__ float *&fixedColumnUB,
                                                                 __ubuf__ float *&fixedMatrixUB,
                                                                 __ubuf__ float *&fixedRowUB, uint32_t &i, uint32_t &k,
                                                                 uint32_t &kA, uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>((fixedMatrixUB + (k + 1U) * (TN + 8U) + c0));
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>((fixedRowUB + c0 - kA));
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>((fixedColumnUB + 1));
            SharedVaxpyStage5<TN>(dstP, i, lP, rkP);
        }
    }
    template <uint32_t TN, bool EARLY_FACTOR, bool ACTIVE_N, auto batchedUpdate, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallFixedN64CachedvmulsPhase7(uint32_t &cnt, LocalTensor<float> &factors, __ubuf__ float *&fixedColumnUB,
                                          __ubuf__ float *&fixedMatrixUB, __ubuf__ float *&fixedRowUB, uint32_t &k,
                                          uint32_t &kA, uint32_t &p, float &piv, float &sign, uint32_t &wA) {

        if (p != k)
            sign = -sign;
        float inv = (batchedUpdate ? -1.0f : 1.0f) / piv;
        if constexpr (ACTIVE_N) {
            AscendCUtils::SetMask<float>(0ULL, ~0ULL >> kA);
        } else {
            SmallMaskSet(wA);
        }
        if (p != k) {
            __ubuf__ float *mp = reinterpret_cast<__ubuf__ float *>(fixedMatrixUB);
            vmuls(reinterpret_cast<__ubuf__ float *>(fixedRowUB), mp + (p * (TN + 8U) + kA), inv, 1, 1, 1, 8, 8);
            vmuls(mp + (p * (TN + 8U) + kA), mp + (k * (TN + 8U) + kA), 1.0f, 1, 1, 1, 8, 8);
        } else {
            vmuls(reinterpret_cast<__ubuf__ float *>(fixedRowUB),
                  reinterpret_cast<__ubuf__ float *>(fixedMatrixUB) + (k * (TN + 8U) + kA), inv, 1, 1, 1, 8, 8);
        }
        if constexpr (ACTIVE_N) {
            PipeBarrier<PIPE_V>();
        }
        SmallMaskReset();
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = TN - c0;
        uint32_t i = k + 1;
        if constexpr (batchedUpdate) {
            ProcessSmallFixedN64CachedvmulsPhase2<TN, EARLY_FACTOR, ACTIVE_N, paddedUpdate>(
                c0, cnt, factors, fixedMatrixUB, fixedRowUB, k, kA, p);
        } else {
            ProcessSmallFixedN64CachedvaxpyPhase6<TN>(c0, fixedColumnUB, fixedMatrixUB, fixedRowUB, i, k, kA, w);
        }
    }
    template <uint32_t TN, bool EARLY_FACTOR, bool DIRECT_GATHER, bool NORMAL_SPR, bool ACTIVE_N, auto batchedUpdate,
              auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallFixedN64CachedvmulsPhase8(uint32_t &cnt, int32_t &ex, LocalTensor<float> &factors, bool &fastProd,
                                          __ubuf__ float *&fixedColumnUB, __ubuf__ float *&fixedMatrixUB,
                                          __ubuf__ float *&fixedRowUB, uint32_t &k, uint32_t &kA, uint32_t &p,
                                          float &prod, float &sign) {

        uint32_t wA = ((EARLY_FACTOR && DIRECT_GATHER && NORMAL_SPR) ? TN : (TN + 8U)) - kA;
        if constexpr (EARLY_FACTOR) {
            if (p != k && cnt > 1U) {
                AscendCUtils::SetMask<float>(static_cast<int32_t>(8));
                vmuls(reinterpret_cast<__ubuf__ float *>(factors[(p - k) * 8U].GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
            }
        }
        float piv = fixedColumnUB[p - k];
        ProcessSmallFixedN64CachedvmulsPhase7<TN, EARLY_FACTOR, ACTIVE_N, batchedUpdate, paddedUpdate>(
            cnt, factors, fixedColumnUB, fixedMatrixUB, fixedRowUB, k, kA, p, piv, sign, wA);
        SmallMaskReset();
        if constexpr (normalMask) {
            ProdSmallCommon(fastProd, prod, ex, piv);
        } else {
            ProdAccum(prod, ex, piv);
        }
    }
    template <uint32_t TN, auto normalMask>
    __aicore__ inline void ProcessSmallFixedN64CachedStatePhase9(const uint32_t &activeN, int32_t &ex, bool &fastProd,
                                                                 __ubuf__ float *&fixedMatrixUB, uint32_t &ld_,
                                                                 float &prod, float &sign, uint32_t &singular) {

        for (uint32_t k = 0; k < activeN; ++k) {
            uint32_t p = k;
            float maxv = 0.0f;
            if (!FindScalarTinyPivot64<true>(fixedMatrixUB, TN, ld_, k, maxv, p, singular))
                break;
            ProcessSmallFixedN64CachedSetValuePhase5<TN, normalMask>(ex, fastProd, fixedMatrixUB, k, ld_, p, prod,
                                                                     sign);
        }
    }
    template <uint32_t TN, bool EARLY_FACTOR, bool DIRECT_GATHER, bool NORMAL_SPR, bool ACTIVE_N>
    __aicore__ inline void ProcessSmallFixedN64CachedvgatherPhase10(uint32_t &cnt, __ubuf__ float *&fixedAbsUB,
                                                                    __ubuf__ float *&fixedColumnUB,
                                                                    __ubuf__ float *&fixedMatrixUB,
                                                                    __ubuf__ uint32_t *&fixedOffsetUB, uint32_t &k) {

        uint32_t be = k * ((TN + 8U) + 1U);
        if constexpr (ACTIVE_N) {
            static_assert(TN == 64U && EARLY_FACTOR && DIRECT_GATHER && NORMAL_SPR);
            AscendCUtils::SetMask<float>(0ULL, ~0ULL >> (64U - cnt));
        } else {
            SmallMaskSet(cnt);
        }
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(fixedColumnUB),
                reinterpret_cast<__ubuf__ uint32_t *>((fixedOffsetUB + (DIRECT_GATHER ? 0U : (be & 7U) * ::kSmallMaxN))),
                reinterpret_cast<uint64_t>(fixedMatrixUB) + (DIRECT_GATHER ? be : (be & ~7U)) * sizeof(float), 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(reinterpret_cast<__ubuf__ float *>(fixedAbsUB), reinterpret_cast<__ubuf__ float *>(fixedColumnUB), 1, 1, 1,
             8, 8);
    }
    template <uint32_t TN> __aicore__ inline void ProcessSmallFixedN64CachedDataCopyPadPhase11(uint32_t &nn32) {
        CopyAndRunTiny56_1((static_cast<uint64_t>(TN) * TN), nn32);
    }

    template <uint32_t TN, bool EARLY_FACTOR, bool DIRECT_GATHER, bool NORMAL_SPR, bool RAW_SPR, bool ACTIVE_N,
              auto batchedUpdate, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallFixedN64CachedSetFlagPhase13(
        const uint32_t &activeN, int32_t &ex, bool &fastProd, __ubuf__ float *&fixedAbsUB,
        __ubuf__ float *&fixedColumnUB, __ubuf__ float *&fixedMatrixUB, __ubuf__ uint32_t *&fixedOffsetUB,
        __ubuf__ float *&fixedReduceUB, __ubuf__ float *&fixedRowUB, float &prod, float &sign, uint32_t &singular) {

#pragma nounroll
        for (uint32_t k = 0; k < activeN; ++k) {
            uint32_t kA = k & ~(::kAlign - 1);
            uint32_t cnt = activeN - k;
            ProcessSmallFixedN64CachedvgatherPhase10<TN, EARLY_FACTOR, DIRECT_GATHER, NORMAL_SPR, ACTIVE_N>(
                cnt, fixedAbsUB, fixedColumnUB, fixedMatrixUB, fixedOffsetUB, k);
            uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
            ProcessSmallRawUBconstexprPhase15<normalMask>(cnt);
            LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
            if (cnt > 1U) {
                ResetMask();
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(fixedColumnUB), 1, 8,
                      static_cast<uint8_t>((cnt + 7U) / 8U));
            }
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
            float maxv = 0.0f;
            uint32_t p = k;
            uint32_t rawAbsBits = 0U;
            ProcessSmallFixedN64CachedconstexprPhase4<TN, NORMAL_SPR, RAW_SPR, normalMask>(fixedReduceUB, k, maxv, p,
                                                                                           rawAbsBits, segN);
            if (RAW_SPR ? !(rawAbsBits > 0U && rawAbsBits <= 0x7f800000U)
                        : (normalMask ? !(maxv > 0.0f) : maxv == 0.0f)) {
                singular = 1;
                break;
            }
            ProcessSmallFixedN64CachedvmulsPhase8<TN, EARLY_FACTOR, DIRECT_GATHER, NORMAL_SPR, ACTIVE_N, batchedUpdate,
                                                  normalMask, paddedUpdate>(
                cnt, ex, factors, fastProd, fixedColumnUB, fixedMatrixUB, fixedRowUB, k, kA, p, prod, sign);
        }
    }
    template <uint32_t TN, bool EARLY_FACTOR, bool DIRECT_GATHER, bool NORMAL_SPR, bool RAW_SPR, bool ACTIVE_N>
    __aicore__ inline void ProcessSmallFixedN64CachedWaitFlagPhase14(uint32_t &m, uint32_t &nn32, bool &tiny,
                                                                     uint32_t &vMte2Pend) {

        uint64_t g = static_cast<uint64_t>(start_ + m);
        WaitSmallInputReuse64(m, tiny, vMte2Pend);
        ProcessSmallFixedN64CachedSetFlagPhase3<TN, EARLY_FACTOR, DIRECT_GATHER, NORMAL_SPR, RAW_SPR, ACTIVE_N>(
            g, m, nn32, tiny);
    }

    template <uint32_t TN, bool EARLY_FACTOR = false, bool DIRECT_GATHER = false, bool NORMAL_SPR = false,
              bool RAW_SPR = false, bool ACTIVE_N = false>
    __aicore__ inline void ProcessSmallFixedN64Cached() {
        const uint32_t activeN = ACTIVE_N ? n_ : TN;
        __ubuf__ float *fixedColumnUB = reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr());
        __ubuf__ float *fixedAbsUB = reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr());
        __ubuf__ float *fixedRowUB = reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr());
        __ubuf__ float *fixedReduceUB = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
        __ubuf__ float *fixedMatrixUB = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
        __ubuf__ uint32_t *fixedOffsetUB =
            reinterpret_cast<__ubuf__ uint32_t *>(DIRECT_GATHER ? tabU_.GetPhyAddr() : tabRU_.GetPhyAddr());
        constexpr bool normalMask = true;
        constexpr bool batchedUpdate = true;
        constexpr bool paddedUpdate = false;
        uint32_t nn32 = TN * TN;
        uint32_t vMte2Pend = 0;
        bool tiny = (TN <= 8U);
        if (tiny && cnt_ * nn32 <= ::kSmallMaxN * ::kSmallMaxN) {
            return ProcessSmallFixedN64CachedDataCopyPadPhase11<TN>(nn32);
        }
        for (uint32_t m = 0; m < cnt_; ++m) {
            ProcessSmallFixedN64CachedWaitFlagPhase14<TN, EARLY_FACTOR, DIRECT_GATHER, NORMAL_SPR, RAW_SPR, ACTIVE_N>(
                m, nn32, tiny, vMte2Pend);
            uint32_t ld_ = (tiny && TN != (TN + 8U)) ? TN : (TN + 8U);
            float sign = 1.0f;
            float prod = 1.0f;
            int32_t ex = 0;
            bool fastProd = true;
            uint32_t singular = 0;
            if (tiny) {
                ProcessSmallFixedN64CachedStatePhase9<TN, normalMask>(activeN, ex, fastProd, fixedMatrixUB, ld_, prod,
                                                                      sign, singular);
            } else {
                ProcessSmallFixedN64CachedSetFlagPhase13<TN, EARLY_FACTOR, DIRECT_GATHER, NORMAL_SPR, RAW_SPR, ACTIVE_N,
                                                         batchedUpdate, normalMask, paddedUpdate>(
                    activeN, ex, fastProd, fixedAbsUB, fixedColumnUB, fixedMatrixUB, fixedOffsetUB, fixedReduceUB,
                    fixedRowUB, prod, sign, singular);
            }
            ProcessMediumRawFixedSetFlagPhase13(ex, m, prod, sign, singular, tiny, vMte2Pend);
        }
    }
    __aicore__ inline void SmallIdentityCopy(__ubuf__ float *dst, __ubuf__ float *src, uint32_t count) {
        copy_ubuf_to_ubuf(reinterpret_cast<__ubuf__ void *>(dst), reinterpret_cast<__ubuf__ void *>(src), 0U, 1U,
                          static_cast<uint16_t>(count / 8U), 0U, 0U);
    }

    __aicore__ inline void N16TailFourAcceptedProdSmallCommonPhase1(int32_t &ex, bool &fastProd, float &prod,
                                                                    uint32_t &selected, float &sign, float &t11,
                                                                    float &t12, float &t13, float &t21, float &t22,
                                                                    float &t23, float &t31, float &t32, float &t33) {

        if (selected == 2U) {
            sign = -sign;
            float swap1 = t11;
            t11 = t21;
            t21 = swap1;
            float swap2 = t12;
            t12 = t22;
            t22 = swap2;
            float swap3 = t13;
            t13 = t23;
            t23 = swap3;
        }
        if (selected == 3U) {
            sign = -sign;
            float swap1 = t11;
            t11 = t31;
            t31 = swap1;
            float swap2 = t12;
            t12 = t32;
            t32 = swap2;
            float swap3 = t13;
            t13 = t33;
            t33 = swap3;
        }
        ProdSmallCommon(fastProd, prod, ex, t11);
        float inverse = LdetSharedOperations::ReciprocalWithZero58(t11, -1.0f);
        float normalized2 = t12 * inverse;
        float normalized3 = t13 * inverse;
        t22 = t22 + t21 * normalized2;
        t23 = t23 + t21 * normalized3;
        t32 = t32 + t31 * normalized2;
        t33 = t33 + t31 * normalized3;
    }
