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
    __aicore__ inline void ProdAccum(float &prod, int32_t &ex, float piv) {
        return LdetSharedOperations::ProdAccumPureShared33Stage1(prod, ex, piv);
    }
    __aicore__ inline float ProdFinal(float prod, int32_t ex) {
        return LdetSharedOperations::ProdFinalPureShared33Stage2(prod, ex);
    }
    template <uint32_t TN, class PhaseArg0>
    __aicore__ inline void TinyLURegStatePhase1(PhaseArg0 &a, uint32_t &k, float &maxv, uint32_t &p) {

#pragma unroll
        for (uint32_t i = k; i < TN; ++i) {
            float v = a[i * TN + k];
            float av = v < 0.0f ? -v : v;
            if (av > maxv) {
                maxv = av;
                p = i;
            }
        }
    }
    template <uint32_t TN, class PhaseArg0>
    __aicore__ inline void TinyLURegStatePhase2(PhaseArg0 &a, uint32_t &k, uint32_t &p, float &sign) {

#pragma unroll
        for (uint32_t j = k; j < TN; ++j) {
            float tv = a[k * TN + j];
            a[k * TN + j] = a[p * TN + j];
            a[p * TN + j] = tv;
        }
        sign = -sign;
    }
    template <uint32_t TN, class Capture0>
    __aicore__ inline void Shared24ScalarUpdateStage15(Capture0 &a, uint32_t &k, float &piv) {
        float inv = 1.0f / piv;
#pragma unroll
        for (uint32_t i = k + 1; i < TN; ++i) {
            float l = a[i * TN + k] * inv;
#pragma unroll
            for (uint32_t j = k + 1; j < TN; ++j) {
                a[i * TN + j] -= l * a[k * TN + j];
            }
        }
    }
    template <uint32_t TN, bool FAST_PROD = false> __aicore__ inline float TinyLUReg(uint32_t base) {
        float a[TN * TN];
#pragma unroll
        for (uint32_t i = 0; i < TN; ++i) {
#pragma unroll
            for (uint32_t j = 0; j < TN; ++j) {
                a[i * TN + j] = mat_.GetValue(base + i * TN + j);
            }
        }
        float sign = 1.0f;
        float prod = 1.0f;
        int32_t ex = 0;
        bool fastProd = true;
        bool singular = false;
#pragma unroll
        for (uint32_t k = 0; k < TN; ++k) {
            uint32_t p = k;
            float maxv = 0.0f;
            TinyLURegStatePhase1<TN>(a, k, maxv, p);
            if (maxv == 0.0f) {
                singular = true;
                break;
            }
            if (p != k) {
                TinyLURegStatePhase2<TN>(a, k, p, sign);
            }
            float piv = a[k * TN + k];
            if constexpr (FAST_PROD) {
                ProdSmallCommon(fastProd, prod, ex, piv);
            } else {
                ProdAccum(prod, ex, piv);
            }
            Shared24ScalarUpdateStage15<TN>(a, k, piv);
        }
        return singular ? 0.0f : sign * ProdFinal(prod, ex);
    }
    template <bool FAST_PROD = false> __aicore__ inline float TinyLUDispatch(uint32_t base) {
        switch (n_) {
        case 1U:
            return TinyLUReg<1, FAST_PROD>(base);
        case 2U:
            return TinyLUReg<2, FAST_PROD>(base);
        case 3U:
            return TinyLUReg<3, FAST_PROD>(base);
        case 4U:
            return TinyLUReg<4, FAST_PROD>(base);
        case 5U:
            return TinyLUReg<5, FAST_PROD>(base);
        case 6U:
            return TinyLUReg<6, FAST_PROD>(base);
        case 7U:
            return TinyLUReg<7, FAST_PROD>(base);
        default:
            return TinyLUReg<8, FAST_PROD>(base);
        }
    }
    template <uint32_t TN, class PhaseArg0>
    __aicore__ inline void TinyLUTailStatePhase1(PhaseArg0 &a, uint32_t &k, float &maxv, uint32_t &p) {
        return TinyLURegStatePhase1<TN, PhaseArg0>(a, k, maxv, p);
    }
    template <uint32_t TN>
    __aicore__ inline bool TinyLUTail(uint32_t base, float &sign, float &prod, int32_t &ex, bool &fastProd) {
        constexpr bool FAST_PROD = true;
        float a[TN * TN];
#pragma unroll
        for (uint32_t i = 0; i < TN; ++i) {
#pragma unroll
            for (uint32_t j = 0; j < TN; ++j) {
                a[i * TN + j] = mat_.GetValue(base + i * 8U + j);
            }
        }
        bool singular = false;
#pragma unroll
        for (uint32_t k = 0; k < TN; ++k) {
            uint32_t p = k;
            float maxv = 0.0f;
            TinyLUTailStatePhase1<TN>(a, k, maxv, p);
            if (maxv == 0.0f) {
                singular = true;
                break;
            }
            if (p != k) {
#pragma unroll
                for (uint32_t j = k; j < TN; ++j) {
                    float tv = a[k * TN + j];
                    a[k * TN + j] = a[p * TN + j];
                    a[p * TN + j] = tv;
                }
                sign = -sign;
            }
            float piv = a[k * TN + k];
            if constexpr (FAST_PROD) {
                ProdSmallCommon(fastProd, prod, ex, piv);
            } else {
                ProdAccum(prod, ex, piv);
            }
            Shared24ScalarUpdateStage15<TN>(a, k, piv);
        }
        return singular;
    }
    __aicore__ inline bool TinyDetTail4Guarded(uint32_t base, float &d4) {
        const float a00 = mat_.GetValue(base + 0U);
        const float a01 = mat_.GetValue(base + 1U);
        const float a02 = mat_.GetValue(base + 2U);
        const float a03 = mat_.GetValue(base + 3U);
        const float a10 = mat_.GetValue(base + 8U);
        const float a11 = mat_.GetValue(base + 9U);
        const float a12 = mat_.GetValue(base + 10U);
        const float a13 = mat_.GetValue(base + 11U);
        const float a20 = mat_.GetValue(base + 16U);
        const float a21 = mat_.GetValue(base + 17U);
        const float a22 = mat_.GetValue(base + 18U);
        const float a23 = mat_.GetValue(base + 19U);
        const float a30 = mat_.GetValue(base + 24U);
        const float a31 = mat_.GetValue(base + 25U);
        const float a32 = mat_.GetValue(base + 26U);
        const float a33 = mat_.GetValue(base + 27U);
        const float s0 = (a00 * a00 + a01 * a01) + (a02 * a02 + a03 * a03);
        const float s1 = (a10 * a10 + a11 * a11) + (a12 * a12 + a13 * a13);
        const float s2 = (a20 * a20 + a21 * a21) + (a22 * a22 + a23 * a23);
        const float s3 = (a30 * a30 + a31 * a31) + (a32 * a32 + a33 * a33);
        if (!(s0 >= 5.9604644775390625e-8f && s0 <= 16777216.0f && s1 >= 5.9604644775390625e-8f && s1 <= 16777216.0f &&
              s2 >= 5.9604644775390625e-8f && s2 <= 16777216.0f && s3 >= 5.9604644775390625e-8f && s3 <= 16777216.0f))
            return false;
        const float q01 = a00 * a11 - a01 * a10;
        const float q02 = a00 * a12 - a02 * a10;
        const float q03 = a00 * a13 - a03 * a10;
        const float q12 = a01 * a12 - a02 * a11;
        const float q13 = a01 * a13 - a03 * a11;
        const float q23 = a02 * a13 - a03 * a12;
        const float r01 = a20 * a31 - a21 * a30;
        const float r02 = a20 * a32 - a22 * a30;
        const float r03 = a20 * a33 - a23 * a30;
        const float r12 = a21 * a32 - a22 * a31;
        const float r13 = a21 * a33 - a23 * a31;
        const float r23 = a22 * a33 - a23 * a32;
        d4 = (((((q01 * r23 - q02 * r13) + q03 * r12) + q12 * r03) - q13 * r02) + q23 * r01);
        const float energyProduct = ((s0 * s1) * s2) * s3;
        const float detSquared = d4 * d4;
        return detSquared > 0.0035f * energyProduct;
    }
    template <uint32_t TN>
    __aicore__ inline void ProcessTinyGuardedLUvaxpyPhase1(uint32_t &c0, uint32_t &i, uint32_t &k, uint32_t &kA,
                                                           uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP =
                reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * ((TN + 7U) & ~7U) + c0].GetPhyAddr());
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>(rowkbuf_[c0 - kA].GetPhyAddr());
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>(colbuf_[1].GetPhyAddr());
            UpdateRowsAxpy61<true>(TN, ((TN + 7U) & ~7U), i, dstP, rkP, lP);
        }
    }
    template <uint32_t TN, auto normalMask, auto paddedUpdate>
    __aicore__ inline void SharedPipeBarrierStage7(uint32_t &c0, uint32_t &cnt, float &inv, uint32_t &k, uint32_t &kA) {
        if (cnt > 1U) {
            if constexpr (normalMask)
                ResetMask();
            Muls(colbuf_, colbuf_, -inv, cnt);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((cnt + 7U) / 8U));
            PipeBarrier<PIPE_V>();
            for (uint32_t col = c0; col < TN; col += 64U) {
                uint32_t width = paddedUpdate ? ((TN + 7U) & ~7U) - col : TN - col;
                if (width > 64U)
                    width = 64U;
                AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
                LdetTinyMulRows(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float *>(rowkbuf_[col - kA].GetPhyAddr()),
                                static_cast<uint8_t>(cnt - 1U));
                LdetTinyAddRows(reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * 8U + col].GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                                static_cast<uint8_t>(cnt - 1U));
            }
        }
    }
    template <uint32_t TN, auto batchedUpdate, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessTinyGuardedLUMulsPhase2(uint32_t &cnt, float &inv, uint32_t &k, uint32_t &kA) {

        SmallMaskReset();
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = TN - c0;
        uint32_t i = k + 1;
        if constexpr (batchedUpdate) {
            SharedPipeBarrierStage7<TN, normalMask, paddedUpdate>(c0, cnt, inv, k, kA);
        } else {
            ProcessTinyGuardedLUvaxpyPhase1<TN>(c0, i, k, kA, w);
        }
        SmallMaskReset();
    }
    template <uint32_t TN> __aicore__ inline void SharedVmulsStage9(uint32_t &k, uint32_t &kA, uint32_t &p) {
        if (p != k) {
            __ubuf__ float *mp = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
            vmuls(reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr()), mp + (p * ((TN + 7U) & ~7U) + kA), 1.0f, 1,
                  1, 1, 8, 8);
            vmuls(mp + (p * ((TN + 7U) & ~7U) + kA), mp + (k * ((TN + 7U) & ~7U) + kA), 1.0f, 1, 1, 1, 8, 8);
        } else {
            vmuls(reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr()) + (k * ((TN + 7U) & ~7U) + kA), 1.0f, 1, 1, 1,
                  8, 8);
        }
    }
    template <bool GUARDED_PATH, uint32_t TN, auto batchedUpdate, auto normalMask, auto paddedUpdate>
    __aicore__ inline void NormalizeTinyPivot60(uint32_t &cnt, int32_t &ex, bool &fastProd, uint32_t &k, uint32_t &kA,
                                                uint32_t &p, float &prod, float &sign) {

        uint32_t wA = ((TN + 7U) & ~7U) - kA;
        if (p != k) {
            float tv = colbuf_.GetValue(0);
            colbuf_.SetValue(0, colbuf_.GetValue(p - k));
            colbuf_.SetValue(p - k, tv);
            sign = -sign;
        }
        float piv = colbuf_.GetValue(0);
        if constexpr (normalMask) {
            ProdSmallCommon(fastProd, prod, ex, piv);
        } else {
            ProdAccum(prod, ex, piv);
        }
        float inv = LdetSharedOperations::ReciprocalWithZero58(piv, 1.0f);
        SmallMaskSet(wA);
        SharedVmulsStage9<TN>(k, kA, p);
        if constexpr (GUARDED_PATH)
            ProcessTinyGuardedLUMulsPhase2<TN, batchedUpdate, normalMask, paddedUpdate>(cnt, inv, k, kA);
        else
            ProcessTinyVectorLUMulsPhase2<TN, batchedUpdate, normalMask, paddedUpdate>(cnt, inv, k, kA);
    }
    template <uint32_t TN, bool GUARDED_DET4>
    __aicore__ inline void ProcessTinyGuardedLUSetFlagPhase4(int32_t &ex, bool &fastProd, uint32_t &m, float &prod,
                                                             float &sign, uint32_t &singular) {

        if (singular == 0U) {
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
            if constexpr (GUARDED_DET4) {
                float d4 = 0.0f;
                if (TinyDetTail4Guarded(4U * 8U + 4U, d4)) {
                    ProdSmallCommon(fastProd, prod, ex, d4);
                } else {
                    if (TinyLUTail<4>(4U * 8U + 4U, sign, prod, ex, fastProd))
                        singular = 1U;
                }
            } else {
                if (TinyLUTail<3>((TN - 3U) * 8U + TN - 3U, sign, prod, ex, fastProd))
                    singular = 1U;
            }
        }
        res_.SetValue(m, singular != 0U ? 0.0f : sign * ProdFinal(prod, ex));
    }
    template <auto normalMask> __aicore__ inline void Shared10ScalarUpdateStage7(uint32_t &cnt) {
        if constexpr (normalMask) {
            LdetVcmax128(tmp_, absbuf_, cnt);
        } else {
            SmallMaskReset();
            SmallMaskSet(cnt);
            for (uint32_t c = 0, coff = 0; c < (cnt + 127U) >> 7; ++c, coff += 128U) {
                uint32_t cc = cnt - coff;
                if (cc > 128U)
                    cc = 128U;
                AscendCUtils::SetMask<float>(0, cc);
                LdetVcmax128(tmp_[c * 8], absbuf_[coff], cc);
            }
        }
        SmallMaskReset();
    }
    template <auto normalMask> __aicore__ inline void SharedScalarUpdateStage44(uint32_t &cnt) {
        Shared10ScalarUpdateStage7<normalMask>(cnt);
    }
    template <auto normalMask> __aicore__ inline void ProcessTinyGuardedLUSetFlagPhase5(uint32_t &cnt) {

        SharedScalarUpdateStage44<normalMask>(cnt);
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }
    template <auto normalMask>
    __aicore__ inline void ProcessTinyGuardedLUconstexprPhase6(uint32_t &k, float &maxv, uint32_t &p, uint32_t &segN) {

        if constexpr (normalMask) {
            maxv = tmp_.GetValue(0);
            float idxF = tmp_.GetValue(1);
            p = k + *reinterpret_cast<uint32_t *>(&idxF);
        } else {
            MergeMaxSegments64<false>(nullptr, segN, k, maxv, p);
        }
    }
    template <uint32_t TN> __aicore__ inline void Shared24VgatherStage8(uint32_t &cnt, uint32_t &k) {
        uint32_t be = k * (((TN + 7U) & ~7U) + 1U);
        SmallMaskSet(cnt);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(tabRU_[(be & 7U) * ::kSmallMaxN].GetPhyAddr()),
                reinterpret_cast<uint64_t>(mat_.GetPhyAddr()) + (be & ~7U) * sizeof(float), 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
             reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr()), 1, 1, 1, 8, 8);
    }
