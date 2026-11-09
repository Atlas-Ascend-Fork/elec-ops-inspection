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
    template <class PhaseArg1>
    __aicore__ inline bool N16DualPlanesSetFlagPhase3(__ubuf__ float *&column, PhaseArg1 &currentRow,
                                                      uint64_t &diagonalAddress, __ubuf__ float *&factors,
                                                      __ubuf__ uint32_t *&offsets, float &product, __ubuf__ float *&row,
                                                      uint32_t &signMask) {

#pragma nounroll
        for (uint32_t count = 16U; count > 0U; --count) {
            if (count == 1U) {
                SetFlag<HardEvent::V_S>(::kEvVS);
                WaitFlag<HardEvent::V_S>(::kEvVS);
                float pivot = mat_.GetValue(615U);
                float magnitude = pivot < 0.0f ? -pivot : pivot;
                if (!(magnitude >= 0x1p-7f && magnitude <= 0x1p7f)) {
                    return false;
                }
                product *= pivot;
                break;
            }
            N16DualPlanesvgatherPhase2(column, count, diagonalAddress, factors, offsets);
            uint32_t magnitudeBits = 0U, index = 0U;
            AscendC::GetReduceRepeatMaxMinSpr<float>(magnitudeBits, index);

            uint32_t p = index & 15U;
            if (static_cast<uint32_t>(magnitudeBits - 0x3c000000U) > 0x07000000U || p >= count) {
                return false;
            }

            N16DualPlanesvmulsPhase1(column, count, currentRow, diagonalAddress, factors, index, p, product, row,
                                     signMask);
        }
        union {
            float f;
            uint32_t u;
        } result;
        result.f = product;
        result.u ^= signMask;
        ResetMask();
        res_.SetValue(0U, result.f);
        return true;
    }
    __aicore__ inline bool N16DualPlanes() {
        static_assert(LD_BATCH == 9U, "n16 signed-plane scope");
        auto matrix = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
        auto offsets = reinterpret_cast<__ubuf__ uint32_t *>(tabU_.GetPhyAddr());
        auto column = reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr());
        auto row = reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr());
        auto factors = reinterpret_cast<__ubuf__ float *>(
            LocalTensor<float>(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8U).GetPhyAddr());
        SmallMaskSet(16U);
        vadds(reinterpret_cast<__ubuf__ int32_t *>(offsets + 16U), reinterpret_cast<__ubuf__ int32_t *>(offsets), 64, 1,
              1, 1, 8, 8);
        if (firstCopyPending_)
            firstCopyPending_ = false;
        else
            DataCopyPad(mat_, aGm_[0], {16U, 64U, 0U, 3U, 0U}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        vmuls(matrix + 16U, matrix, -1.0f, 16U, 1, 1, 5, 5);
        PipeBarrier<PIPE_V>();
        float product = 1.0f;
        uint32_t signMask = 0U;
        auto currentRow = matrix;
        uint64_t diagonalAddress = reinterpret_cast<uint64_t>(matrix);
        return N16DualPlanesSetFlagPhase3(column, currentRow, diagonalAddress, factors, offsets, product, row,
                                          signMask);
    }

    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool EARLY_PIVOT, auto batchedUpdate,
              auto nativeSmallCopies>
    __aicore__ inline void ProcessSmallFixedPipeBarrierPhase1(__ubuf__ float *&cacheFactors,
                                                              __ubuf__ float *&cacheMatrix, __ubuf__ float *&cacheRow,
                                                              uint32_t &k, uint32_t &kA, uint32_t &p, float &piv,
                                                              float &sign, uint32_t &wA) {

        if (p != k) {
            sign = -sign;
            if constexpr (TN == 32U && FULL_ROW) {
                PipeBarrier<PIPE_V>();
                SmallIdentityCopy(cacheFactors + (p - k) * 8U, cacheFactors, 8U);
            }
        }
        float inv = (batchedUpdate ? -1.0f : 1.0f) / piv;
        if constexpr (EARLY_PIVOT && !FULL_ROW)
            PipeBarrier<PIPE_V>();
        if constexpr (!(TN == 32U && FULL_ROW))
            SmallMaskSet(wA);
        if (p != k) {
            __ubuf__ float *mp = (CACHE_VEC_PTR ? cacheMatrix : reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr()));
            vmuls((CACHE_VEC_PTR ? cacheRow : reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr())),
                  mp + (p * (TN + 8U) + kA), inv, 1, 1, 1, 8, 8);
            if (nativeSmallCopies) {
                SmallIdentityCopy(mp + (p * (TN + 8U) + kA), mp + (k * (TN + 8U) + kA), wA);
            } else {
                vmuls(mp + (p * (TN + 8U) + kA), mp + (k * (TN + 8U) + kA), 1.0f, 1, 1, 1, 8, 8);
            }
        } else {
            vmuls((CACHE_VEC_PTR ? cacheRow : reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr())),
                  (CACHE_VEC_PTR ? cacheMatrix : reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr())) +
                      (k * (TN + 8U) + kA),
                  inv, 1, 1, 1, 8, 8);
        }
        SmallMaskReset();
    }
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool LAST_FAST, auto nativeSmallCopies>
    __aicore__ inline void ProcessSmallFixedPipeBarrierPhase2(__ubuf__ float *&cacheFactors,
                                                              LocalTensor<float> &factors, uint32_t &k, uint32_t &p) {

        if constexpr (!(TN == 32U && FULL_ROW)) {
            if (p != k) {
                if constexpr (FULL_ROW)
                    PipeBarrier<PIPE_V>();
                if (!nativeSmallCopies)
                    AscendCUtils::SetMask<float>(static_cast<int32_t>(8));
                if (nativeSmallCopies) {
                    SmallIdentityCopy(
                        (CACHE_VEC_PTR ? cacheFactors + (p - k) * 8U
                                       : reinterpret_cast<__ubuf__ float *>(factors[(p - k) * 8U].GetPhyAddr())),
                        (CACHE_VEC_PTR ? cacheFactors : reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr())), 8U);
                } else {
                    vmuls((CACHE_VEC_PTR ? cacheFactors + (p - k) * 8U
                                         : reinterpret_cast<__ubuf__ float *>(factors[(p - k) * 8U].GetPhyAddr())),
                          (CACHE_VEC_PTR ? cacheFactors : reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr())),
                          1.0f, 1, 1, 1, 8, 8);
                }
            }
        }
        if constexpr (TN == 16U && LAST_FAST)
            PipeBarrier<PIPE_V>();
    }
    template <uint32_t TN>
    __aicore__ inline void ProcessSmallFixedDataCopyPhase3(uint64_t &g, uint32_t &nn32, bool &tiny) {

        if (firstCopyPending_) {
            firstCopyPending_ = false;
        } else
            CopyFixedMatrix56<TN>(g, nn32, tiny);
    }
    template <uint32_t TN, bool FULL_ROW, bool LAST_FAST, bool PACKED_PIVOT, bool DIRECT_GATHER>
    __aicore__ inline void ProcessSmallFixedconstexprPhase4(uint32_t &k, float &maxv, uint32_t &p,
                                                            bool &packedPositive) {

        if constexpr (PACKED_PIVOT) {
            static_assert(::kSpTmp % 8U == 0U, "aligned reduction pair");
            uint64_t pair;
            if constexpr (TN == 16U && DIRECT_GATHER && LAST_FAST) {
                uint32_t valueBits = 0U, indexBits = 0U;
                AscendC::GetReduceRepeatMaxMinSpr<float>(valueBits, indexBits);
                pair = static_cast<uint64_t>(valueBits) | (static_cast<uint64_t>(indexBits) << 32U);
            } else if constexpr (TN == 32U && FULL_ROW)
                pair = N32ScalarPairRead(reinterpret_cast<__ubuf__ volatile uint64_t *>(tmp_.GetPhyAddr()));
            else
                pair = tmp_.template ReinterpretCast<uint64_t>().GetValue(0);
            if constexpr (TN == 16U || TN == 32U) {
                const uint32_t valueBits = static_cast<uint32_t>(pair);
                packedPositive = valueBits > 0U && valueBits <= 0x7f800000U;
            } else {
                union {
                    uint32_t u;
                    float f;
                } value;
                value.u = static_cast<uint32_t>(pair);
                maxv = value.f;
            }
            p = k + static_cast<uint32_t>(pair >> 32U);
        } else {
            maxv = tmp_.GetValue(0);
            float idxF = tmp_.GetValue(1);
            p = k + *reinterpret_cast<uint32_t *>(&idxF);
        }
    }
    template <uint32_t TN, auto normalMask>
    __aicore__ inline void ProcessSmallFixedGetValuePhase5(int32_t &ex, bool &fastProd, uint32_t &k, uint32_t &ld_,
                                                           uint32_t &p, float &prod, float &sign) {
        ScalarTinyStep56_1<normalMask, false>(TN, ld_, k, p, nullptr, ex, fastProd, prod, sign);
    }
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool DIRECT_GATHER>
    __aicore__ inline void ProcessSmallFixedvgatherPhase6(__ubuf__ float *&cacheAbs, __ubuf__ float *&cacheColumn,
                                                          __ubuf__ float *&cacheMatrix,
                                                          __ubuf__ uint32_t *&cacheOffsets, uint32_t &cnt,
                                                          uint32_t &k) {

        uint32_t be = k * ((TN + 8U) + 1U);
        if constexpr (TN == 32U && FULL_ROW) {
            const uint64_t lowMask = 0xffffffffULL >> k;
            AscendCUtils::SetMask<float>(0ULL, lowMask);
        } else
            SmallMaskSet(cnt);
        vgather((reinterpret_cast<__ubuf__ uint32_t *>(
                    CACHE_VEC_PTR ? cacheColumn : reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr()))),
                (DIRECT_GATHER ? reinterpret_cast<__ubuf__ uint32_t *>(tabU_.GetPhyAddr())
                               : (CACHE_VEC_PTR ? cacheOffsets + (be & 7U) * ::kSmallMaxN
                                                : reinterpret_cast<__ubuf__ uint32_t *>(
                                                      tabRU_[(be & 7U) * ::kSmallMaxN].GetPhyAddr()))),
                (reinterpret_cast<uint64_t>(CACHE_VEC_PTR ? cacheMatrix
                                                          : reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr()))) +
                    (DIRECT_GATHER ? be : (be & ~7U)) * sizeof(float),
                8, 1);
        PipeBarrier<PIPE_V>();
        vabs((CACHE_VEC_PTR ? cacheAbs : reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr())),
             (CACHE_VEC_PTR ? cacheColumn : reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr())), 1, 1, 1, 8, 8);
    }
    template <uint32_t TN, bool CACHE_VEC_PTR, auto paddedUpdate>
    __aicore__ inline void ProcessSmallFixedvmlaPhase7(uint32_t &c0, __ubuf__ float *&cacheFactors, uint32_t &cnt,
                                                       LocalTensor<float> &factors, uint32_t &k, uint32_t &kA) {

        for (uint32_t col = c0; col < TN; col += 64U) {
            uint32_t width = paddedUpdate ? (TN + 8U) - col : TN - col;
            if (width > 64U)
                width = 64U;
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            if constexpr (TN == 64U) {
                vmla(reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * (TN + 8U) + col].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(rowkbuf_[col - kA].GetPhyAddr()),
                     (CACHE_VEC_PTR ? cacheFactors + 8U : reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr())),
                     static_cast<uint8_t>(cnt - 1U), 1, 1, 0, static_cast<uint8_t>((TN + 8U) / 8U), 0, 1);
            } else {
                vmla(reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * (TN + 8U) + col].GetPhyAddr()),
                     (CACHE_VEC_PTR ? cacheFactors + 8U : reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr())),
                     reinterpret_cast<__ubuf__ float *>(rowkbuf_[col - kA].GetPhyAddr()),
                     static_cast<uint8_t>(cnt - 1U), 1, 0, 1, static_cast<uint8_t>((TN + 8U) / 8U), 1, 0);
            }
        }
    }
    template <uint32_t TN>
    __aicore__ inline void ProcessSmallFixedvaxpyPhase8(uint32_t &c0, uint32_t &i, uint32_t &k, uint32_t &kA,
                                                        uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * (TN + 8U) + c0].GetPhyAddr());
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>(rowkbuf_[c0 - kA].GetPhyAddr());
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>(colbuf_[1].GetPhyAddr());
            SharedVaxpyStage5<TN>(dstP, i, lP, rkP);
        }
    }
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool LAST_FAST, auto batchedUpdate,
              auto nativeSmallCopies, auto paddedUpdate>
    __aicore__ inline void ProcessSmallFixedPipeBarrierPhase9(__ubuf__ float *&cacheFactors,
                                                              __ubuf__ float *&cacheMatrix, __ubuf__ float *&cacheRow,
                                                              uint32_t &cnt, LocalTensor<float> &factors, uint32_t &k,
                                                              uint32_t &kA, uint32_t &p) {

        uint32_t c0 = FULL_ROW ? kA : (k + 1) & ~(::kAlign - 1);
        uint32_t w = TN - c0;
        uint32_t i = k + 1;
        if constexpr (batchedUpdate) {
            if (cnt > 1U) {
                ProcessSmallFixedPipeBarrierPhase2<TN, FULL_ROW, CACHE_VEC_PTR, LAST_FAST, nativeSmallCopies>(
                    cacheFactors, factors, k, p);
                if constexpr (FULL_ROW) {
                    PipeBarrier<PIPE_V>();
                    static_assert(TN == 32U, "Full-row route is n32 only");
                    if constexpr (!nativeSmallCopies)
                        AscendCUtils::SetMask<float>(static_cast<int32_t>(TN));
                    vmla((CACHE_VEC_PTR
                              ? cacheMatrix + (k + 1U) * (TN + 8U) + c0
                              : reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * (TN + 8U) + c0].GetPhyAddr())),
                         (CACHE_VEC_PTR ? cacheRow : reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr())),
                         (CACHE_VEC_PTR ? cacheFactors + 8U
                                        : reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr())),
                         static_cast<uint8_t>(cnt - 1U), 1, 1, 0, static_cast<uint8_t>((TN + 8U) / 8U), 0, 1);
                } else {
                    ProcessSmallFixedvmlaPhase7<TN, CACHE_VEC_PTR, paddedUpdate>(c0, cacheFactors, cnt, factors, k, kA);
                }
            }
        } else {
            ProcessSmallFixedvaxpyPhase8<TN>(c0, i, k, kA, w);
        }
        SmallMaskReset();
    }
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool EARLY_PIVOT>
    __aicore__ inline void ProcessSmallFixedSetFlagPhase10(__ubuf__ float *&cacheColumn, __ubuf__ float *&cacheFactors,
                                                           uint32_t &cnt, LocalTensor<float> &factors, uint32_t &kA) {

        if constexpr (!CACHE_VEC_PTR)
            factors = LocalTensor<float>(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
        if (cnt > 1U) {
            if constexpr (TN == 32U && FULL_ROW) {
                const uint64_t rowMask = 0xffffffffULL >> kA;
                AscendCUtils::SetMask<float>(0ULL, rowMask);
            } else
                ResetMask();
            vbrcb((reinterpret_cast<__ubuf__ uint32_t *>(
                      CACHE_VEC_PTR ? cacheFactors : reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr()))),
                  (reinterpret_cast<__ubuf__ uint32_t *>(
                      CACHE_VEC_PTR ? cacheColumn : reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr()))),
                  1, 8, static_cast<uint8_t>((cnt + 7U) / 8U));
        }
        if constexpr (!FULL_ROW && !EARLY_PIVOT)
            SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }
    template <uint32_t TN, bool FULL_ROW>
    __aicore__ inline void ProcessSmallFixedSetFlagPhase11(uint32_t &m, uint32_t &nn32, bool &tiny,
                                                           uint32_t &vMte2Pend) {

        uint64_t g = static_cast<uint64_t>(start_ + m);
        WaitSmallInputReuse64(m, tiny, vMte2Pend);
        ProcessSmallFixedDataCopyPhase3<TN>(g, nn32, tiny);
        if constexpr (!(TN == 32U && FULL_ROW)) {
            SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
            WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        }
        if (!tiny) {
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
            WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        }
    }
    template <uint32_t TN, auto normalMask>
    __aicore__ inline void ProcessSmallFixedGetValuePhase12(int32_t &ex, bool &fastProd, uint32_t &ld_, float &prod,
                                                            float &sign, uint32_t &singular) {

        for (uint32_t k = 0; k < TN; ++k) {
            uint32_t p = k;
            float maxv = 0.0f;
            if (!FindScalarTinyPivot64<false>(nullptr, TN, ld_, k, maxv, p, singular))
                break;
            ProcessSmallFixedGetValuePhase5<TN, normalMask>(ex, fastProd, k, ld_, p, prod, sign);
        }
    }
    template <uint32_t TN, bool FULL_ROW, bool CACHE_VEC_PTR, bool LAST_FAST, bool EARLY_PIVOT, auto batchedUpdate,
              auto nativeSmallCopies, auto normalMask, auto paddedUpdate>
    __aicore__ inline void
    ProcessSmallFixedconstexprPhase13(__ubuf__ float *&cacheColumn, __ubuf__ float *&cacheFactors,
                                      __ubuf__ float *&cacheMatrix, __ubuf__ float *&cacheRow, uint32_t &cnt,
                                      int32_t &ex, LocalTensor<float> &factors, bool &fastProd, uint32_t &k,
                                      uint32_t &kA, uint32_t &p, float &prod, float &sign) {

        uint32_t wA = FULL_ROW ? TN - kA : (TN + 8U) - kA;
        float piv;
        if constexpr (TN == 32U && FULL_ROW)
            piv = N32ScalarColumnRead(reinterpret_cast<__ubuf__ volatile float *>(cacheColumn), p - k);
        else
            piv = colbuf_.GetValue(p - k);
        ProcessSmallFixedPipeBarrierPhase1<TN, FULL_ROW, CACHE_VEC_PTR, EARLY_PIVOT, batchedUpdate, nativeSmallCopies>(
            cacheFactors, cacheMatrix, cacheRow, k, kA, p, piv, sign, wA);
        ProcessSmallFixedPipeBarrierPhase9<TN, FULL_ROW, CACHE_VEC_PTR, LAST_FAST, batchedUpdate, nativeSmallCopies,
                                           paddedUpdate>(cacheFactors, cacheMatrix, cacheRow, cnt, factors, k, kA, p);
        if constexpr (normalMask) {
            ProdSmallCommon(fastProd, prod, ex, piv);
        } else {
            ProdAccum(prod, ex, piv);
        }
    }
    template <bool FULL_ROW, bool EARLY_PIVOT, auto normalMask>
    __aicore__ inline void ProcessSmallFixedSetFlagPhase14(uint32_t &cnt) {

        Shared10ScalarUpdateStage7<normalMask>(cnt);
        if constexpr (FULL_ROW || EARLY_PIVOT)
            SetFlag<HardEvent::V_S>(::kEvVS);
    }
    template <uint32_t TN> __aicore__ inline void ProcessSmallFixedDataCopyPadPhase15(uint32_t &nn32) {
        return ProcessSmallFixedN64CachedDataCopyPadPhase11<TN>(nn32);
    }
    template <uint32_t TN, bool FULL_ROW, bool LAST_FAST, bool PACKED_PIVOT, bool DIRECT_GATHER, auto normalMask>
    __aicore__ inline void ProcessSmallFixedconstexprPhase16(uint32_t &k, float &maxv, uint32_t &p,
                                                             bool &packedPositive, uint32_t &segN) {

        if constexpr (normalMask) {
            ProcessSmallFixedconstexprPhase4<TN, FULL_ROW, LAST_FAST, PACKED_PIVOT, DIRECT_GATHER>(k, maxv, p,
                                                                                                   packedPositive);
        } else {
            MergeMaxSegments64<false>(nullptr, segN, k, maxv, p);
        }
    }
