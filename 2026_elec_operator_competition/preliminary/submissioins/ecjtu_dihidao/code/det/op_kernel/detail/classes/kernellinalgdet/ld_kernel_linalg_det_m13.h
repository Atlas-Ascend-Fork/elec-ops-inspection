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
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR>
    __aicore__ inline void
    ProcessSmallFixedconstexprPhase18(__ubuf__ float *&cacheAbs, __ubuf__ float *&cacheColumn,
                                      __ubuf__ float *&cacheFactors, __ubuf__ float *&cacheMatrix,
                                      __ubuf__ uint32_t *&cacheOffsets, __ubuf__ float *&cacheRow,
                                      LocalTensor<float> &onceFactors) {

        if constexpr (CACHE_VEC_PTR) {
            static_assert(TN == 32U && FULL_ROW, "Vector cache scope");
            cacheColumn = reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr());
            cacheAbs = reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr());
            cacheRow = reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr());
            cacheMatrix = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
            cacheOffsets = reinterpret_cast<__ubuf__ uint32_t *>(tabRU_.GetPhyAddr());
            onceFactors = LocalTensor<float>(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
            cacheFactors = reinterpret_cast<__ubuf__ float *>(onceFactors.GetPhyAddr());
        }
    }
    template <uint32_t TN>
    __aicore__ inline void ProcessSmallFixedSetFlagPhase19(int32_t &ex, bool &fastProd, float &prod,
                                                           uint32_t &singular) {

        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
        float finalPivot = mat_.GetValue((TN - 1U) * ((TN + 8U) + 1U));
        float finalAbs = finalPivot < 0.0f ? -finalPivot : finalPivot;
        if (!(finalAbs > 0.0f))
            singular = 1U;
        else
            ProdSmallCommon(fastProd, prod, ex, finalPivot);
    }
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool LAST_FAST, bool PACKED_PIVOT, bool EARLY_PIVOT,
              bool DIRECT_GATHER, auto batchedUpdate, auto nativeSmallCopies, auto normalMask, auto paddedUpdate,
              class LoopArg0, class LoopArg1, class LoopArg2, class LoopArg3, class LoopArg4, class LoopArg5,
              class LoopArg6, class LoopArg7, class LoopArg8, class LoopArg9, class LoopArg10, class LoopArg11>
    __aicore__ inline bool SmallFixedPivotStep(LoopArg0 &cacheAbs, LoopArg1 &cacheColumn, LoopArg2 &cacheFactors,
                                               LoopArg3 &cacheMatrix, LoopArg4 &cacheOffsets, LoopArg5 &cacheRow,
                                               LoopArg6 &ex, LoopArg7 &fastProd, LoopArg8 &k, LoopArg9 &prod,
                                               LoopArg10 &sign, LoopArg11 &singular) {

        if constexpr (TN == 32U && FULL_ROW) {
            if (k == 30U) {
                SetFlag<HardEvent::V_S>(::kEvVS);
                WaitFlag<HardEvent::V_S>(::kEvVS);
                float tailDet = N32TailTwoGuarded();
                if (tailDet != 0.0f) {
                    ProdSmallCommon(fastProd, prod, ex, tailDet);
                    return false;
                }
            }
        }
        if constexpr (LAST_FAST) {
            static_assert(TN == 16U || TN == 32U, "Last finish scope");
            if (k + 1U == TN) {
                ProcessSmallFixedSetFlagPhase19<TN>(ex, fastProd, prod, singular);
                return false;
            }
        }
        uint32_t kA = (TN == 32U && FULL_ROW) ? (k & ~(::kAlign - 1U)) : (FULL_ROW ? 0U : k & ~(::kAlign - 1U));
        uint32_t cnt = TN - k;
        ProcessSmallFixedvgatherPhase6<TN, FULL_ROW, CACHE_VEC_PTR, DIRECT_GATHER>(cacheAbs, cacheColumn, cacheMatrix,
                                                                                   cacheOffsets, cnt, k);
        uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
        ProcessSmallFixedSetFlagPhase14<FULL_ROW, EARLY_PIVOT, normalMask>(cnt);
        LocalTensor<float> factors;
        ProcessSmallFixedSetFlagPhase10<TN, FULL_ROW, CACHE_VEC_PTR, EARLY_PIVOT>(cacheColumn, cacheFactors, cnt,
                                                                                  factors, kA);
        float maxv = 0.0f;
        bool packedPositive = false;
        uint32_t p = k;
        ProcessSmallFixedconstexprPhase16<TN, FULL_ROW, LAST_FAST, PACKED_PIVOT, DIRECT_GATHER, normalMask>(
            k, maxv, p, packedPositive, segN);
        if (normalMask ? ((PACKED_PIVOT && (TN == 16U || TN == 32U)) ? !packedPositive : !(maxv > 0.0f))
                       : maxv == 0.0f) {
            singular = 1;
            return false;
        }
        ProcessSmallFixedconstexprPhase13<TN, FULL_ROW, CACHE_VEC_PTR, LAST_FAST, EARLY_PIVOT, batchedUpdate,
                                          nativeSmallCopies, normalMask, paddedUpdate>(
            cacheColumn, cacheFactors, cacheMatrix, cacheRow, cnt, ex, factors, fastProd, k, kA, p, prod, sign);

        return true;
    }
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool LAST_FAST, bool PACKED_PIVOT, bool EARLY_PIVOT,
              bool DIRECT_GATHER, auto batchedUpdate, auto nativeSmallCopies, auto normalMask, auto paddedUpdate,
              class LoopArg0, class LoopArg1, class LoopArg2, class LoopArg3, class LoopArg4, class LoopArg5,
              class LoopArg6, class LoopArg7, class LoopArg8, class LoopArg9>
    __aicore__ inline void SmallFixedMatrixStep(LoopArg0 &cacheAbs, LoopArg1 &cacheColumn, LoopArg2 &cacheFactors,
                                                LoopArg3 &cacheMatrix, LoopArg4 &cacheOffsets, LoopArg5 &cacheRow,
                                                LoopArg6 &m, LoopArg7 &nn32, LoopArg8 &tiny, LoopArg9 &vMte2Pend) {

        ProcessSmallFixedSetFlagPhase11<TN, FULL_ROW>(m, nn32, tiny, vMte2Pend);
        uint32_t ld_ = (tiny && TN != (TN + 8U)) ? TN : (TN + 8U);
        float sign = 1.0f;
        float prod = 1.0f;
        int32_t ex = 0;
        bool fastProd = true;
        uint32_t singular = 0;
        if (tiny) {
            ProcessSmallFixedGetValuePhase12<TN, normalMask>(ex, fastProd, ld_, prod, sign, singular);
        } else {
#pragma nounroll
            for (uint32_t k = 0; k < TN; ++k) {
                if (!SmallFixedPivotStep<TN, FULL_ROW, CACHE_VEC_PTR, LAST_FAST, PACKED_PIVOT, EARLY_PIVOT,
                                         DIRECT_GATHER, batchedUpdate, nativeSmallCopies, normalMask, paddedUpdate>(
                        cacheAbs, cacheColumn, cacheFactors, cacheMatrix, cacheOffsets, cacheRow, ex, fastProd, k, prod,
                        sign, singular))
                    break;
            }
        }
        ProcessMediumRawFixedSetFlagPhase13(ex, m, prod, sign, singular, tiny, vMte2Pend);
    }
    template <uint32_t TN, bool FULL_ROW = false, bool CACHE_VEC_PTR = false, bool LAST_FAST = false,
              bool PACKED_PIVOT = false, bool EARLY_PIVOT = false, bool DIRECT_GATHER = false>
    __aicore__ inline void ProcessSmallFixed() {
        constexpr bool nativeSmallCopies = TN == 32U;
        __ubuf__ float *cacheColumn = nullptr, *cacheAbs = nullptr, *cacheRow = nullptr;
        __ubuf__ float *cacheMatrix = nullptr, *cacheFactors = nullptr;
        __ubuf__ uint32_t *cacheOffsets = nullptr;
        LocalTensor<float> onceFactors;
        ProcessSmallFixedconstexprPhase18<TN, FULL_ROW, CACHE_VEC_PTR>(cacheAbs, cacheColumn, cacheFactors, cacheMatrix,
                                                                       cacheOffsets, cacheRow, onceFactors);
        constexpr bool normalMask = true;
        constexpr bool batchedUpdate = true;
        constexpr bool paddedUpdate = false;
        uint32_t nn32 = TN * TN;
        uint32_t vMte2Pend = 0;
        bool tiny = (TN <= 8U);
        if (tiny && cnt_ * nn32 <= ::kSmallMaxN * ::kSmallMaxN) {
            return ProcessSmallFixedDataCopyPadPhase15<TN>(nn32);
        }
        for (uint32_t m = 0; m < cnt_; ++m) {
            SmallFixedMatrixStep<TN, FULL_ROW, CACHE_VEC_PTR, LAST_FAST, PACKED_PIVOT, EARLY_PIVOT, DIRECT_GATHER,
                                 batchedUpdate, nativeSmallCopies, normalMask, paddedUpdate>(
                cacheAbs, cacheColumn, cacheFactors, cacheMatrix, cacheOffsets, cacheRow, m, nn32, tiny, vMte2Pend);
        }
    }
    __aicore__ inline __inout_pipe__(S) uint64_t N32ScalarPairRead(__ubuf__ volatile uint64_t *pointer) {
        return pointer[0];
    }
    __aicore__ inline __inout_pipe__(S) float N32ScalarColumnRead(__ubuf__ volatile float *pointer, uint32_t index) {
        return pointer[index];
    }
    template <bool CACHED_PRODUCT>
    __aicore__ inline void
    ProcessFamilyColumnsPipeBarrierPhase1(const uint32_t &aligned, uint32_t &alignedK, __ubuf__ float *&column,
                                          uint32_t &count, int32_t &exponent, __ubuf__ float *&factors, bool &fast,
                                          uint32_t &k, __ubuf__ float *&matrix, uint64_t &pair, float &prod,
                                          __ubuf__ float *&row, float &sign, const uint32_t &stride) {

        uint32_t pivotRow = k + static_cast<uint32_t>(pair >> 32U);
        float pivot = N32ScalarColumnRead(reinterpret_cast<__ubuf__ volatile float *>(column), pivotRow - k);
        if (pivotRow != k) {
            sign = -sign;
            PipeBarrier<PIPE_V>();
            SmallIdentityCopy(factors + (pivotRow - k) * 8U, factors, 8U);
        }
        float inverse = -1.0f / pivot;
        vmuls(row, matrix + pivotRow * stride + alignedK, inverse, 1, 1, 1, 8, 8);
        if (pivotRow != k)
            SmallIdentityCopy(matrix + pivotRow * stride + alignedK, matrix + k * stride + alignedK,
                              aligned - alignedK);
        PipeBarrier<PIPE_V>();
        vmla(matrix + (k + 1U) * stride + alignedK, row, factors + 8U, static_cast<uint8_t>(count - 1U), 1, 1, 0,
             static_cast<uint8_t>(stride / 8U), 0, 1);
        if constexpr (CACHED_PRODUCT) {
            uint32_t raw =
                N15ScalarColumnBitsRead(reinterpret_cast<__ubuf__ volatile uint32_t *>(column), pivotRow - k);
            N15ProdCachedBits(fast, prod, exponent, pivot, raw);
        } else
            ProdSmallCommon(fast, prod, exponent, pivot);
    }
    template <bool DIRECT_GATHER>
    __aicore__ inline void ProcessFamilyColumnsPipeBarrierPhase2(__ubuf__ float *&absolute, uint32_t &alignedK,
                                                                 uint32_t &base, __ubuf__ float *&column,
                                                                 uint32_t &count, __ubuf__ float *&factors, uint32_t &k,
                                                                 const uint64_t &liveMask, __ubuf__ float *&matrix,
                                                                 __ubuf__ uint32_t *&offsets) {

        PipeBarrier<PIPE_V>();
        AscendCUtils::SetMask<float>(0ULL, liveMask >> k);
        if constexpr (DIRECT_GATHER) {
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(column), offsets,
                    reinterpret_cast<uint64_t>(matrix) + base * 4U, 8U, 1U);
        } else {
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(column), offsets + (base & 7U) * ::kSmallMaxN,
                    reinterpret_cast<uint64_t>(matrix) + (base & ~7U) * 4U, 8U, 1U);
        }
        PipeBarrier<PIPE_V>();
        vabs(absolute, column, 1, 1, 1, 8, 8);
        LdetVcmax128(tmp_, absbuf_, count);
        SetFlag<HardEvent::V_S>(::kEvVS);
        AscendCUtils::SetMask<float>(0ULL, liveMask >> alignedK);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors), reinterpret_cast<__ubuf__ uint32_t *>(column), 1U, 8U,
              static_cast<uint8_t>((count + 7U) / 8U));
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }

    __aicore__ inline void ProcessFamilyColumnsSetFlagPhase3(int32_t &exponent, bool &fast, const uint32_t &n,
                                                             float &prod, bool &singular, const uint32_t &stride) {

        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
        float pivot = mat_.GetValue((n - 1U) * (stride + 1U));
        float magnitude = pivot < 0.0f ? -pivot : pivot;
        if (!(magnitude > 0.0f))
            singular = true;
        else
            ProdSmallCommon(fast, prod, exponent, pivot);
    }

    __aicore__ inline void ProcessFamilyColumnsDataCopyPadPhase4(const uint32_t &aligned, const uint32_t &n,
                                                                 const uint32_t &stride) {

        if (firstCopyPending_)
            firstCopyPending_ = false;
        else
            DataCopyPad(mat_, aGm_[static_cast<uint64_t>(start_) * n * n],
                        DataCopyExtParams{static_cast<uint16_t>(n), n * 4U, 0U, (stride - aligned) / 8U, 0U},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
    }
    template <uint32_t FIXED_N, uint32_t FIXED_STRIDE, bool DIRECT_GATHER, bool CACHED_PRODUCT, bool N16_TAIL4,
              class LoopArg0, class LoopArg1, class LoopArg2, class LoopArg3, class LoopArg4, class LoopArg5,
              class LoopArg6, class LoopArg7, class LoopArg8, class LoopArg9, class LoopArg10, class LoopArg11,
              class LoopArg12, class LoopArg13, class LoopArg14, class LoopArg15, class LoopArg16>
    __aicore__ inline bool FamilyColumnPivotStep(LoopArg0 &absolute, LoopArg1 &aligned, LoopArg2 &column,
                                                 LoopArg3 &exponent, LoopArg4 &factors, LoopArg5 &fast, LoopArg6 &k,
                                                 LoopArg7 &liveMask, LoopArg8 &matrix, LoopArg9 &n, LoopArg10 &offsets,
                                                 LoopArg11 &prod, LoopArg12 &reduce, LoopArg13 &row, LoopArg14 &sign,
                                                 LoopArg15 &singular, LoopArg16 &stride) {

        if constexpr (N16_TAIL4) {
            static_assert(FIXED_N == 16U && FIXED_STRIDE == 24U, "n16 tail4 scope");
            if (k == 12U) {
                SetFlag<HardEvent::V_S>(::kEvVS);
                WaitFlag<HardEvent::V_S>(::kEvVS);
                N16TailFourAccepted(sign, prod, exponent, fast, singular);
                return false;
            }
        } else {
            if (k + 2U == n) {
                SetFlag<HardEvent::V_S>(::kEvVS);
                WaitFlag<HardEvent::V_S>(::kEvVS);
                float tail = SmallTailTwoGuarded((n - 2U) * (stride + 1U), stride);
                if (tail != 0.0f) {
                    ProdSmallCommon(fast, prod, exponent, tail);
                    return false;
                }
            }
            if (k + 1U == n) {
                ProcessFamilyColumnsSetFlagPhase3(exponent, fast, n, prod, singular, stride);
                return false;
            }
        }
        uint32_t alignedK = k & ~7U, count = n - k, base = k * (stride + 1U);
        ProcessFamilyColumnsPipeBarrierPhase2<DIRECT_GATHER>(absolute, alignedK, base, column, count, factors, k,
                                                             liveMask, matrix, offsets);
        uint64_t pair = N32ScalarPairRead(reduce);
        uint32_t bits = static_cast<uint32_t>(pair);
        if (!(bits > 0U && bits <= 0x7f800000U)) {
            singular = true;
            return false;
        }
        ProcessFamilyColumnsPipeBarrierPhase1<CACHED_PRODUCT>(aligned, alignedK, column, count, exponent, factors, fast,
                                                              k, matrix, pair, prod, row, sign, stride);

        return true;
    }
    template <uint32_t FIXED_N = 0U, uint32_t FIXED_STRIDE = 0U, bool DIRECT_GATHER = false,
              bool CACHED_PRODUCT = false, bool N16_TAIL4 = false>
    __aicore__ inline void ProcessFamilyColumns() {
        const uint32_t n = FIXED_N ? FIXED_N : n_;
        const uint32_t stride = FIXED_STRIDE ? FIXED_STRIDE : s_;
        const uint32_t aligned = (n + 7U) & ~7U;
        const uint64_t liveMask = (1ULL << n) - 1ULL;
        auto matrix = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
        auto column = reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr());
        auto absolute = reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr());
        auto row = reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr());
        auto offsets = reinterpret_cast<__ubuf__ uint32_t *>(DIRECT_GATHER ? tabU_.GetPhyAddr() : tabRU_.GetPhyAddr());
        auto reduce = reinterpret_cast<__ubuf__ volatile uint64_t *>(tmp_.GetPhyAddr());
        LocalTensor<float> factorTensor(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8U);
        auto factors = reinterpret_cast<__ubuf__ float *>(factorTensor.GetPhyAddr());
        ProcessFamilyColumnsDataCopyPadPhase4(aligned, n, stride);
        float sign = 1.0f, prod = 1.0f;
        int32_t exponent = 0;
        bool fast = true, singular = false;
#pragma nounroll
        for (uint32_t k = 0; k < n; ++k) {
            if (!FamilyColumnPivotStep<FIXED_N, FIXED_STRIDE, DIRECT_GATHER, CACHED_PRODUCT, N16_TAIL4>(
                    absolute, aligned, column, exponent, factors, fast, k, liveMask, matrix, n, offsets, prod, reduce,
                    row, sign, singular, stride))
                break;
        }
        res_.SetValue(0U, singular ? 0.0f : sign * ProdFinal(prod, exponent));
    }
    __aicore__ inline __inout_pipe__(S) uint32_t
        N15ScalarColumnBitsRead(__ubuf__ volatile uint32_t *pointer, uint32_t index) {
        return pointer[index];
    }
    __aicore__ inline void N15ProdCachedBits(bool &fast, float &prod, int32_t &ex, float piv, uint32_t bits) {
        uint32_t ef = (bits >> 23U) & 255U;
        if (fast && ef >= 2U && ef <= 252U) {
            ex += static_cast<int32_t>(ef) - 126;
            union {
                uint32_t u;
                float f;
            } cv;
            cv.u = (bits & 0x807fffffU) | 0x3f000000U;
            prod *= cv.f;
        } else {
            fast = false;
            ProdAccum(prod, ex, piv);
        }
    }
    __aicore__ inline __inout_pipe__(S) float SmallTailTwoGuarded(uint32_t base, uint32_t stride) {
        float a = mat_.GetValue(base), b = mat_.GetValue(base + 1U);
        float c = mat_.GetValue(base + stride), d = mat_.GetValue(base + stride + 1U);
        float ad = a * d, bc = b * c, result = ad - bc;
        float scale = (ad < 0.0f ? -ad : ad) + (bc < 0.0f ? -bc : bc);
        float magnitude = result < 0.0f ? -result : result;
        if (scale >= 0x1p-100f && scale <= 0x1p100f && magnitude > scale * 0x1p-6f)
            return result;
        return 0.0f;
    }
    __aicore__ inline __inout_pipe__(S) float N32TailTwoGuarded() {
        float a = mat_.GetValue(30U * 40U + 30U);
        float b = mat_.GetValue(30U * 40U + 31U);
        float c = mat_.GetValue(31U * 40U + 30U);
        float d = mat_.GetValue(31U * 40U + 31U);
        float ad = a * d;
        float bc = b * c;
        float result = ad - bc;
        float scale = (ad < 0.0f ? -ad : ad) + (bc < 0.0f ? -bc : bc);
        float magnitude = result < 0.0f ? -result : result;
        if (scale >= 0x1p-100f && scale <= 0x1p100f && magnitude > scale * 0x1p-6f)
            return result;
        return 0.0f;
    }
    __aicore__ inline void ProdSmallCommon(bool &fast, float &prod, int32_t &ex, float piv) {
        union {
            float f;
            uint32_t u;
        } cv;
        cv.f = piv;
        uint32_t ef = (cv.u >> 23) & 0xffU;
        if (fast && ef >= 2U && ef <= 252U) {
            ex += static_cast<int32_t>(ef) - 126;
            cv.u = (cv.u & 0x807fffffU) | 0x3f000000U;
            prod *= cv.f;
        } else {
            fast = false;
            ProdAccum(prod, ex, piv);
        }
    }
