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
    template <uint32_t TN, bool GUARDED_DET4 = false> __aicore__ inline void ProcessTinyGuardedLU() {
        constexpr uint32_t tailN = GUARDED_DET4 ? 4U : 3U;
        constexpr bool normalMask = true;
        constexpr bool batchedUpdate = true;
        constexpr bool paddedUpdate = true;
        LocalTensor<float> original = mat_;
        LoadTinyBatch60<TN>();
        for (uint32_t m = 0; m < cnt_; ++m) {
            mat_ = original[m * TN * 8U];
            TinyProductState60 productState{1.0f, 1.0f, 0, true, 0};
#pragma nounroll
            for (uint32_t k = 0; k < TN - tailN; ++k) {
                uint32_t kA = k & ~(::kAlign - 1);
                uint32_t cnt = TN - k;
                Shared24VgatherStage8<TN>(cnt, k);
                uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
                ProcessTinyGuardedLUSetFlagPhase5<normalMask>(cnt);
                float maxv = 0.0f;
                uint32_t p = k;
                ProcessTinyGuardedLUconstexprPhase6<normalMask>(k, maxv, p, segN);
                if (normalMask ? !(maxv > 0.0f) : maxv == 0.0f) {
                    productState.singular = 1;
                    break;
                }
                NormalizeTinyPivot60<true, TN, batchedUpdate, normalMask, paddedUpdate>(
                    cnt, productState.ex, productState.fastProd, k, kA, p, productState.prod, productState.sign);
            }
            ProcessTinyGuardedLUSetFlagPhase4<TN, GUARDED_DET4>(
                productState.ex, productState.fastProd, m, productState.prod, productState.sign, productState.singular);
        }
        mat_ = original;
    }
    template <uint32_t TN>
    __aicore__ inline void ProcessTinyVectorLUvaxpyPhase1(uint32_t &c0, uint32_t &i, uint32_t &k, uint32_t &kA,
                                                          uint32_t &w) {
        return ProcessTinyGuardedLUvaxpyPhase1<TN>(c0, i, k, kA, w);
    }
    template <uint32_t TN, auto batchedUpdate, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessTinyVectorLUMulsPhase2(uint32_t &cnt, float &inv, uint32_t &k, uint32_t &kA) {

        SmallMaskReset();
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = TN - c0;
        uint32_t i = k + 1;
        if constexpr (batchedUpdate) {
            SharedPipeBarrierStage7<TN, normalMask, paddedUpdate>(c0, cnt, inv, k, kA);
        } else {
            ProcessTinyVectorLUvaxpyPhase1<TN>(c0, i, k, kA, w);
        }
        SmallMaskReset();
    }

    template <auto normalMask> __aicore__ inline void ProcessTinyVectorLUSetFlagPhase4(uint32_t &cnt) {
        return ProcessTinyGuardedLUSetFlagPhase5<normalMask>(cnt);
    }
    template <auto normalMask>
    __aicore__ inline void ProcessTinyVectorLUconstexprPhase5(uint32_t &k, float &maxv, uint32_t &p, uint32_t &segN) {
        return ProcessTinyGuardedLUconstexprPhase6<normalMask>(k, maxv, p, segN);
    }
    template <uint32_t TN, auto batchedUpdate, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessTinyVectorLUvgatherPhase6(int32_t &ex, bool &fastProd, uint32_t &m, float &prod,
                                                            float &sign, uint32_t &singular) {

#pragma nounroll
        for (uint32_t k = 0; k < TN - 3U; ++k) {
            uint32_t kA = k & ~(::kAlign - 1);
            uint32_t cnt = TN - k;
            Shared24VgatherStage8<TN>(cnt, k);
            uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
            ProcessTinyVectorLUSetFlagPhase4<normalMask>(cnt);
            float maxv = 0.0f;
            uint32_t p = k;
            ProcessTinyVectorLUconstexprPhase5<normalMask>(k, maxv, p, segN);
            if (normalMask ? !(maxv > 0.0f) : maxv == 0.0f) {
                singular = 1;
                break;
            }
            NormalizeTinyPivot60<false, TN, batchedUpdate, normalMask, paddedUpdate>(cnt, ex, fastProd, k, kA, p, prod,
                                                                                     sign);
        }
        if (singular == 0U) {
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
            if (TinyLUTail<3>((TN - 3U) * 8U + TN - 3U, sign, prod, ex, fastProd))
                singular = 1U;
        }
        res_.SetValue(m, singular != 0U ? 0.0f : sign * ProdFinal(prod, ex));
    }
    template <uint32_t TN> __aicore__ inline void ProcessTinyVectorLU() {
        constexpr bool normalMask = true;
        constexpr bool batchedUpdate = true;
        constexpr bool paddedUpdate = true;
        LocalTensor<float> original = mat_;
        LoadTinyBatch60<TN>();
        for (uint32_t m = 0; m < cnt_; ++m) {
            mat_ = original[m * TN * 8U];
            TinyProductState60 productState{1.0f, 1.0f, 0, true, 0};
            ProcessTinyVectorLUvgatherPhase6<TN, batchedUpdate, normalMask, paddedUpdate>(
                productState.ex, productState.fastProd, m, productState.prod, productState.sign, productState.singular);
        }
        mat_ = original;
    }
    __aicore__ inline void SharedDataCopyPadStage17(uint64_t &g) {
        if constexpr (LD_BATCH == 6U || PAD128) {
            for (uint32_t r0 = 0; r0 < n_; r0 += 32U) {
                uint32_t rows = n_ - r0;
                if (rows > 32U)
                    rows = 32U;
                DataCopyExtParams cp{static_cast<uint16_t>(rows), static_cast<uint32_t>(n_ * sizeof(float)), 0,
                                     PAD128 ? (s_ - ((n_ + 7U) & ~7U)) / 8U : 0U, 0};
                DataCopyPad(mat_[r0 * s_], aGm_[g * nn_ + static_cast<uint64_t>(r0) * n_], cp,
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        } else {
            for (uint32_t i = 0; i < n_; ++i) {
                DataCopyPad(mat_[i * s_], aGm_[g * nn_ + static_cast<uint64_t>(i) * n_],
                            {1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        }
    }

    __aicore__ inline void ProcessSmallDataCopyPhase1(uint32_t &m, uint32_t &nn32, bool &tiny, uint32_t &vMte2Pend) {

        uint64_t g = static_cast<uint64_t>(start_ + m);
        WaitSmallInputReuse64(m, tiny, vMte2Pend);
        if (n_ == s_) {
            DataCopy(mat_, aGm_[g * nn_], nn32);
        } else if (tiny) {
            DataCopyPad(mat_[0], aGm_[g * nn_], {1, static_cast<uint32_t>(nn32 * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            SharedDataCopyPadStage17(g);
        }
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask>
    __aicore__ inline void ProcessSmallvmulsPhase2(int32_t &ex, bool &fastProd, uint32_t &k, uint32_t &kA, uint32_t &p,
                                                   float &piv, float &prod, float &sign, uint32_t &wA) {

        if constexpr (earlyBroadcast) {
            piv = colbuf_.GetValue(p - k);
            if (p != k)
                sign = -sign;
        } else {
            if (p != k) {
                float tv = colbuf_.GetValue(0);
                colbuf_.SetValue(0, colbuf_.GetValue(p - k));
                colbuf_.SetValue(p - k, tv);
                sign = -sign;
            }
            piv = colbuf_.GetValue(0);
        }
        if constexpr (!earlyBroadcast) {
            if constexpr (normalMask) {
                ProdSmallCommon(fastProd, prod, ex, piv);
            } else {
                ProdAccum(prod, ex, piv);
            }
        }
        float inv = (batchedUpdate ? -1.0f : 1.0f) / piv;
        SmallMaskSet(wA);
        if (p != k) {
            __ubuf__ float *mp = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
            vmuls(reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr()), mp + (p * s_ + kA), inv, 1, 1, 1, 8, 8);
            vmuls(mp + (p * s_ + kA), mp + (k * s_ + kA), 1.0f, 1, 1, 1, 8, 8);
        } else {
            vmuls(reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr()) + (k * s_ + kA), inv, 1, 1, 1, 8, 8);
        }
        SmallMaskReset();
    }
    template <auto paddedUpdate>
    __aicore__ inline void SharedVmlaStage34(uint32_t &c0, uint32_t &cnt, LocalTensor<float> &factors, uint32_t &k,
                                             uint32_t &kA) {
        for (uint32_t col = c0; col < n_; col += 64U) {
            uint32_t width = paddedUpdate ? s_ - col : n_ - col;
            if (width > 64U)
                width = 64U;
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            vmla(reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * s_ + col].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(rowkbuf_[col - kA].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(factors[8].GetPhyAddr()), static_cast<uint8_t>(cnt - 1U), 1, 1, 0,
                 static_cast<uint8_t>(s_ / 8U), 0, 1);
        }
    }

    __aicore__ inline void Shared24VmulsStage19(LocalTensor<float> &factors, uint32_t &k, uint32_t &p) {
        if (p != k) {
            AscendCUtils::SetMask<float>(static_cast<int32_t>(8));
            vmuls(reinterpret_cast<__ubuf__ float *>(factors[(p - k) * 8U].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
        }
    }
    template <auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallvmulsPhase3(uint32_t &c0, uint32_t &cnt, LocalTensor<float> &factors,
                                                   uint32_t &k, uint32_t &kA, uint32_t &p) {

        if (cnt > 1U) {
            if constexpr (earlyBroadcast) {
                Shared24VmulsStage19(factors, k, p);
            } else {
                if constexpr (normalMask)
                    ResetMask();
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()), 1, 8,
                      static_cast<uint8_t>((cnt + 7U) / 8U));
                PipeBarrier<PIPE_V>();
            }
            SharedVmlaStage34<paddedUpdate>(c0, cnt, factors, k, kA);
        }
    }
    template <auto normalMask>
    __aicore__ inline void ProcessSmallGetValuePhase4(int32_t &ex, bool &fastProd, uint32_t &k, uint32_t &ld_,
                                                      uint32_t &p, float &prod, float &sign) {
        ScalarTinyStep56_1<normalMask, false>(n_, ld_, k, p, nullptr, ex, fastProd, prod, sign);
    }

    __aicore__ inline void SharedVaxpyStage14(__ubuf__ float *&dstP, uint32_t &i, __ubuf__ float *&lP,
                                              __ubuf__ float *&rkP) {
        UpdateRowsAxpy61<false>(n_, s_, i, dstP, rkP, lP);
    }

    __aicore__ inline void ProcessSmallvaxpyPhase5(uint32_t &c0, uint32_t &i, uint32_t &k, uint32_t &kA, uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * s_ + c0].GetPhyAddr());
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>(rowkbuf_[c0 - kA].GetPhyAddr());
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>(colbuf_[1].GetPhyAddr());
            SharedVaxpyStage14(dstP, i, lP, rkP);
        }
    }

    __aicore__ inline void SharedVaxpyStage12(__ubuf__ float *&dstP, uint32_t &i, __ubuf__ float *&lP,
                                              __ubuf__ float *&rkP) {
        UpdateRowsAxpy61<true>(n_, s_, i, dstP, rkP, lP);
    }

    __aicore__ inline void ProcessSmallvaxpyPhase6(uint32_t &c0, uint32_t &i, uint32_t &k, uint32_t &kA, uint32_t &w) {

        SmallMaskSet(w);
        {
            __ubuf__ float *dstP = reinterpret_cast<__ubuf__ float *>(mat_[(k + 1U) * s_ + c0].GetPhyAddr());
            __ubuf__ float *rkP = reinterpret_cast<__ubuf__ float *>(rowkbuf_[c0 - kA].GetPhyAddr());
            __ubuf__ float *lP = reinterpret_cast<__ubuf__ float *>(colbuf_[1].GetPhyAddr());
            SharedVaxpyStage12(dstP, i, lP, rkP);
        }
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallPipeBarrierPhase7(uint32_t &cnt, int32_t &ex, LocalTensor<float> &factors,
                                                         bool &fastProd, uint32_t &k, uint32_t &kA, uint32_t &p,
                                                         float &piv, float &prod, float &sign, uint32_t &wA) {

        ProcessSmallvmulsPhase2<batchedUpdate, earlyBroadcast, normalMask>(ex, fastProd, k, kA, p, piv, prod, sign, wA);
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = n_ - c0;
        uint32_t i = k + 1;
        if constexpr (LD_BATCH == 15U) {
            if (wA > 64U) {
                ProcessSmallvaxpyPhase5(c0, i, k, kA, w);
            } else {
                if (cnt > 1U) {
                    if constexpr (normalMask)
                        ResetMask();
                    LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
                    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()), 1, 8,
                          static_cast<uint8_t>((cnt + 7U) / 8U));
                    PipeBarrier<PIPE_V>();
                    SharedVmlaStage34<paddedUpdate>(c0, cnt, factors, k, kA);
                }
            }
        } else if constexpr (batchedUpdate) {
            ProcessSmallvmulsPhase3<earlyBroadcast, normalMask, paddedUpdate>(c0, cnt, factors, k, kA, p);
        } else {
            ProcessSmallvaxpyPhase6(c0, i, k, kA, w);
        }
    }
    template <auto normalMask>
    __aicore__ inline void ProcessSmallGetValuePhase8(int32_t &ex, bool &fastProd, uint32_t &ld_, float &prod,
                                                      float &sign, uint32_t &singular) {

        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t p = k;
            float maxv = 0.0f;
            if (!FindScalarTinyPivot64<false>(nullptr, n_, ld_, k, maxv, p, singular))
                break;
            ProcessSmallGetValuePhase4<normalMask>(ex, fastProd, k, ld_, p, prod, sign);
        }
    }
    __aicore__ inline void ProcessSmallDataCopyPadPhase9(uint32_t &nn32) {
        CopyAndRunTiny56_1(nn_, nn32);
    }

    template <auto normalMask> __aicore__ inline void ProcessSmallconstexprPhase11(uint32_t &cnt) {

        SharedScalarUpdateStage44<normalMask>(cnt);
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallSmallMaskResetPhase12(uint32_t &cnt, int32_t &ex, LocalTensor<float> &factors,
                                                             bool &fastProd, uint32_t &k, uint32_t &kA, uint32_t &p,
                                                             float &prod, float &sign) {

        uint32_t wA = s_ - kA;
        float piv;
        ProcessSmallPipeBarrierPhase7<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
            cnt, ex, factors, fastProd, k, kA, p, piv, prod, sign, wA);
        CompleteEarlyProduct64<earlyBroadcast, normalMask>(fastProd, prod, ex, piv);
    }
    __aicore__ inline void Shared24VgatherStage7(uint32_t &cnt, uint32_t &k) {
        uint32_t be = k * (s_ + 1U);
        SmallMaskSet(cnt);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(tabRU_[(be & 7U) * ::kSmallMaxN].GetPhyAddr()),
                reinterpret_cast<uint64_t>(mat_.GetPhyAddr()) + (be & ~7U) * sizeof(float), 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
             reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr()), 1, 1, 1, 8, 8);
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallvgatherPhase13(int32_t &ex, bool &fastProd, float &prod, float &sign,
                                                      uint32_t &singular) {

        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t kA = k & ~(::kAlign - 1);
            uint32_t cnt = n_ - k;
            Shared24VgatherStage7(cnt, k);
            uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
            ProcessSmallconstexprPhase11<normalMask>(cnt);
            LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
            if constexpr (earlyBroadcast) {
                if (cnt > 1U) {
                    ResetMask();
                    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()), 1, 8,
                          static_cast<uint8_t>((cnt + 7U) / 8U));
                }
            }
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
            float maxv = 0.0f;
            uint32_t p = k;
            ProcessTinyVectorLUconstexprPhase5<normalMask>(k, maxv, p, segN);
            if (normalMask ? !(maxv > 0.0f) : maxv == 0.0f) {
                singular = 1;
                break;
            }
            ProcessSmallSmallMaskResetPhase12<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                cnt, ex, factors, fastProd, k, kA, p, prod, sign);
        }
    }
