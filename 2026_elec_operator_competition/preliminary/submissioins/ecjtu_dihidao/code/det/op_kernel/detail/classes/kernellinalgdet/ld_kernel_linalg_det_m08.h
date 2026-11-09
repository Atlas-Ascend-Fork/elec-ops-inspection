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
    __aicore__ inline void
    ProcessMediumRawFixedvgatherPhase15(__ubuf__ float *&cachedAbsUB, __ubuf__ float *&cachedColumnUB,
                                        __ubuf__ float *&cachedMatrixUB, __ubuf__ uint32_t *&cachedOffsetUB,
                                        __ubuf__ float *&cachedReduceUB, __ubuf__ float *&cachedRowUB, int32_t &ex,
                                        bool &fastProd, uint32_t &ld_, uint32_t &m, float &prod, const uint32_t &rawN,
                                        const uint32_t &rawS, float &sign, bool &tiny, uint32_t &vMte2Pend) {

        uint32_t singular = 0;
        if (tiny) {
            ProcessMediumRawFixedStatePhase9<normalMask>(cachedMatrixUB, ex, fastProd, ld_, prod, rawN, sign, singular);
        } else {
            for (uint32_t k = 0; k < rawN; ++k) {
                uint32_t kA = k & ~(::kAlign - 1);
                uint32_t cnt = rawN - k;
                uint32_t be = k * (rawS + 1U);
                GatherAbsColumn56(cachedColumnUB, cachedOffsetUB, cachedMatrixUB, cachedAbsUB, be, cnt);
                uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
                ProcessSmallRawUBconstexprPhase15<normalMask>(cnt);
                LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
                ProcessMediumRawFixedSetFlagPhase14<earlyBroadcast>(cachedColumnUB, cnt, factors);
                float maxv = 0.0f;
                uint32_t p = k;
                ProcessMediumRawFixedconstexprPhase11<normalMask>(cachedReduceUB, k, maxv, p, segN);
                if (normalMask ? !(maxv > 0.0f) : maxv == 0.0f) {
                    singular = 1;
                    break;
                }
                ProcessMediumRawFixedconstexprPhase8<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                    cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, ex, factors, fastProd, k, kA, p, prod, rawN, rawS,
                    sign);
            }
        }
        ProcessMediumRawFixedSetFlagPhase13(ex, m, prod, sign, singular, tiny, vMte2Pend);
    }
    template <uint32_t FIXED_N> __aicore__ inline void ProcessMediumRawFixed() {
        const uint32_t rawN = FIXED_N ? FIXED_N : n_;
        const uint32_t rawS = FIXED_N ? FIXED_N : s_;
        const uint64_t rawNN = FIXED_N ? static_cast<uint64_t>(FIXED_N) * FIXED_N : nn_;
        SmallRawViews51 views = GetSmallRawViews51();
        constexpr bool normalMask = LD_BATCH == 3U || LD_BATCH == 6U;
        constexpr bool batchedUpdate = LdetRouteKeyIs<1U, 3U, 5U, 6U, 15U>(LD_BATCH);
        constexpr bool paddedUpdate = LdetRouteKeyIs<5U, 6U, 15U>(LD_BATCH);
        constexpr bool earlyBroadcast = batchedUpdate && LD_BATCH != 15U;
        uint32_t nn32 = rawN * rawN;
        uint32_t vMte2Pend = 0;
        bool tiny = (rawN <= 8U);
        if (tiny && cnt_ * nn32 <= ::kSmallMaxN * ::kSmallMaxN) {
            return ProcessMediumRawFixedDataCopyPadPhase10(nn32, rawNN);
        }
        for (uint32_t m = 0; m < cnt_; ++m) {
            uint64_t g = static_cast<uint64_t>(start_ + m);
            ProcessMediumRawFixedDataCopyPhase1(g, m, nn32, rawN, rawNN, rawS, tiny, vMte2Pend);
            SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
            WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
            if (!tiny) {
                SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
                WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
            }
            uint32_t ld_ = (tiny && rawN != rawS) ? rawN : rawS;
            float sign = 1.0f;
            float prod = 1.0f;
            int32_t ex = 0;
            bool fastProd = true;
            ProcessMediumRawFixedvgatherPhase15<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                views.absolute, views.column, views.matrix, views.offsets, views.reduction, views.row, ex, fastProd,
                ld_, m, prod, rawN, rawS, sign, tiny, vMte2Pend);
        }
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask>
    __aicore__ inline void ProcessSmallPrefetchRawUBvmulsPhase1(__ubuf__ float *&cachedColumnUB,
                                                                __ubuf__ float *&cachedMatrixUB,
                                                                __ubuf__ float *&cachedRowUB, int32_t &ex,
                                                                bool &fastProd, uint32_t &k, uint32_t &kA, uint32_t &p,
                                                                float &piv, float &prod, float &sign, uint32_t &wA) {

        SharedScalarUpdateStage30<earlyBroadcast, normalMask>(cachedColumnUB, ex, fastProd, k, p, piv, prod, sign);
        float inv = (batchedUpdate ? -1.0f : 1.0f) / piv;
        SmallMaskSet(wA);
        if (p != k) {
            __ubuf__ float *mp = reinterpret_cast<__ubuf__ float *>(cachedMatrixUB);
            vmuls(reinterpret_cast<__ubuf__ float *>(cachedRowUB), mp + (p * s_ + kA), inv, 1, 1, 1, 8, 8);
            vmuls(mp + (p * s_ + kA), mp + (k * s_ + kA), 1.0f, 1, 1, 1, 8, 8);
        } else {
            vmuls(reinterpret_cast<__ubuf__ float *>(cachedRowUB),
                  reinterpret_cast<__ubuf__ float *>(cachedMatrixUB) + (k * s_ + kA), inv, 1, 1, 1, 8, 8);
        }
        SmallMaskReset();
    }
    template <auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallPrefetchRawUBvmulsPhase2(uint32_t &c0, __ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB,
                                         __ubuf__ float *&cachedRowUB, uint32_t &cnt, LocalTensor<float> &factors,
                                         uint32_t &k, uint32_t &kA, uint32_t &p) {

        if (cnt > 1U) {
            SharedVmulsStage41<earlyBroadcast, normalMask>(cachedColumnUB, cnt, factors, k, p);
            SharedVmlaStage36<paddedUpdate>(c0, cachedMatrixUB, cachedRowUB, cnt, factors, k, kA);
        }
    }

    __aicore__ inline void ProcessSmallPrefetchRawUBvaxpyPhase3(uint32_t &c0, __ubuf__ float *&cachedColumnUB,
                                                                __ubuf__ float *&cachedMatrixUB,
                                                                __ubuf__ float *&cachedRowUB, uint32_t &i, uint32_t &k,
                                                                uint32_t &kA, uint32_t &w) {
        return ProcessSmallRawUBvaxpyPhase8(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, w);
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallPrefetchRawUBconstexprPhase4(__ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB,
                                             __ubuf__ float *&cachedRowUB, uint32_t &cnt, int32_t &ex,
                                             LocalTensor<float> &factors, bool &fastProd, uint32_t &k, uint32_t &kA,
                                             uint32_t &p, float &prod, float &sign) {

        uint32_t wA = s_ - kA;
        float piv;
        ProcessSmallPrefetchRawUBvmulsPhase1<batchedUpdate, earlyBroadcast, normalMask>(
            cachedColumnUB, cachedMatrixUB, cachedRowUB, ex, fastProd, k, kA, p, piv, prod, sign, wA);
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = n_ - c0;
        uint32_t i = k + 1;
        if constexpr (batchedUpdate) {
            ProcessSmallPrefetchRawUBvmulsPhase2<earlyBroadcast, normalMask, paddedUpdate>(
                c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, factors, k, kA, p);
        } else {
            ProcessSmallPrefetchRawUBvaxpyPhase3(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, w);
        }
        CompleteEarlyProduct64<earlyBroadcast, normalMask>(fastProd, prod, ex, piv);
    }
    template <auto normalMask>
    __aicore__ inline void ProcessSmallPrefetchRawUBconstexprPhase5(__ubuf__ float *&cachedReduceUB, uint32_t &k,
                                                                    float &maxv, uint32_t &p, uint32_t &segN) {
        return ProcessMediumRawFixedconstexprPhase11<normalMask>(cachedReduceUB, k, maxv, p, segN);
    }

    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallPrefetchRawUBvgatherPhase7(__ubuf__ float *&cachedAbsUB, __ubuf__ float *&cachedColumnUB,
                                           __ubuf__ float *&cachedMatrixUB, __ubuf__ uint32_t *&cachedOffsetUB,
                                           __ubuf__ float *&cachedReduceUB, __ubuf__ float *&cachedRowUB, int32_t &ex,
                                           bool &fastProd, float &prod, float &sign, uint32_t &singular) {

        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t kA = k & ~(::kAlign - 1);
            uint32_t cnt = n_ - k;
            uint32_t be = k * (s_ + 1U);
            GatherAbsColumn56(cachedColumnUB, cachedOffsetUB, cachedMatrixUB, cachedAbsUB, be, cnt);
            uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
            ProcessSmallRawUBconstexprPhase15<normalMask>(cnt);
            LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
            if constexpr (earlyBroadcast) {
                if (cnt > 1U) {
                    ResetMask();
                    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB), 1, 8,
                          static_cast<uint8_t>((cnt + 7U) / 8U));
                }
            }
            PrefetchPivotReady();
            float maxv = 0.0f;
            uint32_t p = k;
            ProcessSmallPrefetchRawUBconstexprPhase5<normalMask>(cachedReduceUB, k, maxv, p, segN);
            if (normalMask ? !(maxv > 0.0f) : maxv == 0.0f) {
                singular = 1;
                break;
            }
            ProcessSmallPrefetchRawUBconstexprPhase4<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, ex, factors, fastProd, k, kA, p, prod, sign);
        }
    }
    __aicore__ inline void ProcessSmallPrefetchRawUB() {
        __ubuf__ float *cachedColumnUB = reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr());
        __ubuf__ float *cachedAbsUB = reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr());
        __ubuf__ float *cachedRowUB = reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr());
        __ubuf__ float *cachedReduceUB = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
        __ubuf__ uint32_t *cachedOffsetUB = reinterpret_cast<__ubuf__ uint32_t *>(tabRU_.GetPhyAddr());
        constexpr bool normalMask = true;
        constexpr bool batchedUpdate = true;
        constexpr bool paddedUpdate = true;
        constexpr bool earlyBroadcast = batchedUpdate;
        const uint32_t slotElems = n_ * s_;
        LocalTensor<float> slots = mat_;
        StartPrefetchSlots54(slots);
        for (uint32_t m = 0; m < cnt_; ++m) {
            BindPrefetchSlot54(slots, m, slotElems);
            __ubuf__ float *cachedMatrixUB = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
            PrefetchFollowingSlot64(slots, m, slotElems);
            float sign = 1.0f;
            float prod = 1.0f;
            int32_t ex = 0;
            bool fastProd = true;
            uint32_t singular = 0;
            ProcessSmallPrefetchRawUBvgatherPhase7<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                cachedAbsUB, cachedColumnUB, cachedMatrixUB, cachedOffsetUB, cachedReduceUB, cachedRowUB, ex, fastProd,
                prod, sign, singular);
            if (m + 2U < cnt_)
                PrefetchReuseReady();
            float detVal = (singular != 0U) ? 0.0f : sign * ProdFinal(prod, ex);
            res_.SetValue(m, detVal);
        }
        mat_ = slots;
    }
