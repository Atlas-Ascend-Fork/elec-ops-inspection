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
    template <bool LAZY, auto kCmStride, class PhaseArg0>
    __aicore__ inline void ProcessLargeParvmulsPhase16(PhaseArg0 &alive, uint32_t &cnt, uint32_t &j0, uint32_t &nCh,
                                                       uint32_t &t, uint32_t &tA) {

        if (!LAZY || t == 0U) {
            LDET_MASK_SET(cnt);
            if constexpr (LAZY) {
                LdetPanelAbs(absbuf_, panelCM_[t * kCmStride]);
                LdetPanelMul(absbuf_, absbuf_, alive);
            } else {
                vabs(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(panelCM_[t * kCmStride + tA].GetPhyAddr()), 1, 1, 1, 8, 8);
            }
            if (j0 != 0U) {
                AscendCUtils::SetMask<float>(0, j0);
                vmuls(reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr()), 0.0f, 1, 1, 1, 8, 8);
            }
            ReduceColumnChunks64(nCh, cnt);
        }
    }
    template <auto kCmStride>
    __aicore__ inline void ProcessLargeParGetValuePhase17(uint32_t &bw, uint32_t &js, uint32_t &p, uint32_t &t) {

        for (; js + 7U < bw; js += 8U) {
            float va0 = panelCM_.GetValue(js * kCmStride + t);
            float vb0 = panelCM_.GetValue(js * kCmStride + p);
            float va1 = panelCM_.GetValue((js + 1U) * kCmStride + t);
            float vb1 = panelCM_.GetValue((js + 1U) * kCmStride + p);
            float va2 = panelCM_.GetValue((js + 2U) * kCmStride + t);
            float vb2 = panelCM_.GetValue((js + 2U) * kCmStride + p);
            float va3 = panelCM_.GetValue((js + 3U) * kCmStride + t);
            float vb3 = panelCM_.GetValue((js + 3U) * kCmStride + p);
            float va4 = panelCM_.GetValue((js + 4U) * kCmStride + t);
            float vb4 = panelCM_.GetValue((js + 4U) * kCmStride + p);
            float va5 = panelCM_.GetValue((js + 5U) * kCmStride + t);
            float vb5 = panelCM_.GetValue((js + 5U) * kCmStride + p);
            float va6 = panelCM_.GetValue((js + 6U) * kCmStride + t);
            float vb6 = panelCM_.GetValue((js + 6U) * kCmStride + p);
            float va7 = panelCM_.GetValue((js + 7U) * kCmStride + t);
            float vb7 = panelCM_.GetValue((js + 7U) * kCmStride + p);
            panelCM_.SetValue(js * kCmStride + t, vb0);
            panelCM_.SetValue(js * kCmStride + p, va0);
            panelCM_.SetValue((js + 1U) * kCmStride + t, vb1);
            panelCM_.SetValue((js + 1U) * kCmStride + p, va1);
            panelCM_.SetValue((js + 2U) * kCmStride + t, vb2);
            panelCM_.SetValue((js + 2U) * kCmStride + p, va2);
            panelCM_.SetValue((js + 3U) * kCmStride + t, vb3);
            panelCM_.SetValue((js + 3U) * kCmStride + p, va3);
            panelCM_.SetValue((js + 4U) * kCmStride + t, vb4);
            panelCM_.SetValue((js + 4U) * kCmStride + p, va4);
            panelCM_.SetValue((js + 5U) * kCmStride + t, vb5);
            panelCM_.SetValue((js + 5U) * kCmStride + p, va5);
            panelCM_.SetValue((js + 6U) * kCmStride + t, vb6);
            panelCM_.SetValue((js + 6U) * kCmStride + p, va6);
            panelCM_.SetValue((js + 7U) * kCmStride + t, vb7);
            panelCM_.SetValue((js + 7U) * kCmStride + p, va7);
        }
    }
    template <auto kCmStride>
    __aicore__ inline void ProcessLargeParSetFlagPhase18(uint32_t &bw, uint32_t &cnt, int32_t &ex, uint32_t &p,
                                                         float &prod, uint32_t &t, uint32_t &tA) {

        float piv = panelCM_.GetValue(t * kCmStride + t);
        ProdAccum(prod, ex, piv);
        float inv = 1.0f / piv;
        if (p != t) {
            SetFlag<HardEvent::S_V>(::kEvSV);
            WaitFlag<HardEvent::S_V>(::kEvSV);
        }
        LDET_MASK_SET(cnt);
        vmuls(reinterpret_cast<__ubuf__ float *>(panelCM_[t * kCmStride + tA].GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(panelCM_[t * kCmStride + tA].GetPhyAddr()), inv, 1, 1, 1, 8, 8);
        uint32_t je = t + 1;
        for (; je + 3U < bw; je += 4U) {
            float pj0 = panelCM_.GetValue(je * kCmStride + t);
            float pj1 = panelCM_.GetValue((je + 1U) * kCmStride + t);
            float pj2 = panelCM_.GetValue((je + 2U) * kCmStride + t);
            float pj3 = panelCM_.GetValue((je + 3U) * kCmStride + t);
            LDET_AXPY_TRK(panelCM_[je * kCmStride + tA], panelCM_[t * kCmStride + tA], -pj0, cnt);
            LDET_AXPY_TRK(panelCM_[(je + 1U) * kCmStride + tA], panelCM_[t * kCmStride + tA], -pj1, cnt);
            LDET_AXPY_TRK(panelCM_[(je + 2U) * kCmStride + tA], panelCM_[t * kCmStride + tA], -pj2, cnt);
            LDET_AXPY_TRK(panelCM_[(je + 3U) * kCmStride + tA], panelCM_[t * kCmStride + tA], -pj3, cnt);
        }
        for (; je < bw; ++je) {
            float pj = panelCM_.GetValue(je * kCmStride + t);
            LDET_AXPY_TRK(panelCM_[je * kCmStride + tA], panelCM_[t * kCmStride + tA], -pj, cnt);
        }
        LDET_MASK_RST();
    }
    template <bool LAZY, auto kCmStride>
    __aicore__ inline void SharedDataCopyPadStage39(uint64_t &base, uint32_t &bw, uint32_t &h) {
        if constexpr (FAST_MC && LAZY) {
            uint32_t hp = (h + 7U) & ~7U;
            CopyIn2D(lbuf_, base + 2U * nnS_, bw, hp, kCmStride, kCmStride);
        } else {
            for (uint32_t j = 0; j < bw; ++j) {
                DataCopyPad(lbuf_[j * kCmStride], wsGm_[base + 2U * nnS_ + j * kCmStride], {1, h * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        }
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
    }
    template <bool LAZY, auto evS2M3, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParDataCopyPadPhase19(uint64_t &barBase, uint64_t &base, uint32_t &bw,
                                                             uint32_t &curReg, uint32_t &h, uint32_t &kb,
                                                             uint32_t &panelIdx, uint32_t &rowsPer, uint32_t &singular,
                                                             uint32_t &tw) {

        if (rank_ != 0U) {
            DataCopyPad(metaU_, wsGmU_[base + 2U * nnS_ + kCmStride * kPanel], {1, 128, 0, 0, 0},
                        DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
            WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
            if (metaU_.GetValue(25) != 0U) {
                SharedDataCopyPadStage39<LAZY, kCmStride>(base, bw, h);
            }
        }
        UpdateSchurRegion54<false, LAZY, kCmStride, kPanel>(base, bw, curReg, h, kb, panelIdx, rowsPer, singular, tw);
        ProcessLargeParSetFlagPhase9<evS2M3>(barBase, panelIdx);
    }
    template <auto kCmStride>
    __aicore__ inline void Shared24ScalarUpdateStage22(uint32_t &bw, uint32_t &js, uint32_t &p, float &sign,
                                                       uint32_t &t) {
        for (; js < bw; ++js) {
            float va = panelCM_.GetValue(js * kCmStride + t);
            float vb = panelCM_.GetValue(js * kCmStride + p);
            panelCM_.SetValue(js * kCmStride + t, vb);
            panelCM_.SetValue(js * kCmStride + p, va);
        }
        sign = -sign;
    }
    template <bool LAZY, auto kCmStride>
    __aicore__ inline void ProcessLargeParconstexprPhase20(LocalTensor<float> &alive, uint32_t &bw, uint32_t &cnt,
                                                           int32_t &ex, uint32_t &h, LocalTensor<uint32_t> &invPos,
                                                           uint32_t &nCh, uint32_t &p, bool &panelUpdatePending,
                                                           float &prod, LocalTensor<uint32_t> &rowOffsets, float &sign,
                                                           uint32_t &t, uint32_t &tA) {

        if constexpr (LAZY) {
            uint32_t logicalP = invPos.GetValue(p) / 4U;
            uint32_t physicalT = posU_.GetValue(t) / 4U;
            metaU_.SetValue(1U + t, logicalP);
            UpdatePanelPivot54<false, kCmStride>(alive, bw, ex, h, invPos, logicalP, nCh, p, panelUpdatePending,
                                                 physicalT, prod, rowOffsets, sign, t);
        } else {
            metaU_.SetValue(1 + t, p);
            if (p != t) {
                uint32_t js = 0;
                ProcessLargeParGetValuePhase17<kCmStride>(bw, js, p, t);
                Shared24ScalarUpdateStage22<kCmStride>(bw, js, p, sign, t);
            }
            ProcessLargeParSetFlagPhase18<kCmStride>(bw, cnt, ex, p, prod, t, tA);
        }
    }
    template <bool LAZY>
    __aicore__ inline void ProcessLargeParSetFlagPhase21(uint32_t &bw, uint32_t &h, uint32_t &sExact, uint32_t &t,
                                                         uint32_t &tA) {

        if (bw < h) {
            uint32_t sCross = (bw - tA) >> 6;
            sExact = ((bw - tA) & 63U) != 0U ? sCross + 1U : sCross;
        }
        if constexpr (LAZY) {
            if (t == 0U) {
                SetFlag<HardEvent::V_S>(::kEvVS);
                WaitFlag<HardEvent::V_S>(::kEvVS);
            } else {
                WaitFlag<HardEvent::V_S>(::kEvVMte3);
            }
        } else {
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
        }
    }
    template <bool LAZY, auto evS2M3, auto kCmStride>
    __aicore__ inline void
    ProcessLargeParSetFlagPhase22(LocalTensor<float> &alive, uint64_t &base, uint32_t &bw, int32_t &ex, uint32_t &h,
                                  LocalTensor<uint32_t> &invPos, uint32_t &l21nz, float &prod,
                                  LocalTensor<uint32_t> &rowOffsets, float &sign, uint32_t &singular) {

        bool panelUpdatePending = false;
        for (uint32_t t = 0; t < bw; ++t) {
            PanelStepGeometry61 geometry = GetPanelStepGeometry61<LAZY>(t, h);
            ProcessLargeParvmulsPhase16<LAZY, kCmStride>(alive, geometry.cnt, geometry.j0, geometry.nCh, t,
                                                         geometry.tA);
            uint32_t sExact = geometry.segN;
            ProcessLargeParSetFlagPhase21<LAZY>(bw, h, sExact, t, geometry.tA);
            float maxv = 0.0f;
            uint32_t p = geometry.tA;
            ProcessLargeParLDET_RM_PAIR_OFFPhase10<kCmStride>(bw, h, l21nz, maxv, p, sExact, geometry.segN, t,
                                                              geometry.tA);
            if (RejectZeroPanelPivot64<LAZY>(maxv, panelUpdatePending, singular))
                break;
            ProcessLargeParconstexprPhase20<LAZY, kCmStride>(alive, bw, geometry.cnt, ex, h, invPos, geometry.nCh, p,
                                                             panelUpdatePending, prod, rowOffsets, sign, t,
                                                             geometry.tA);
        }
        if constexpr (LAZY) {
            LDET_MASK_SET(bw * kCmStride);
            LdetPanelMuls(panelCM_, -1.0f);
            LDET_MASK_RST();
        }
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
        ProcessLargeParSetFlagPhase3<LAZY, evS2M3, kCmStride>(base, bw, h, l21nz, singular);
    }
    template <auto kCmStride, auto kPanel>
    __aicore__ inline void SharedSetFlagStage24(uint64_t &base, uint32_t &bw, uint32_t &bwp, uint32_t &curReg,
                                                uint32_t &h, uint32_t &kb) {
        CopyIn2D(panel_, base + static_cast<uint64_t>(curReg) * nnS_ + static_cast<uint64_t>(kb) * s_ + kb, h, bwp, s_,
                 kPanel);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        for (uint32_t j = 0; j < bw; ++j) {
            Adds(coloffI_, tabI_[0], static_cast<int32_t>(j * sizeof(float)), h);
            Gather(panelCM_[j * kCmStride], panel_, coloffU_, 0U, h);
        }
    }
    template <bool LAZY, auto evS2M3, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParSetFlagPhase23(uint64_t &barBase, uint64_t &base, uint32_t &bw, uint32_t &bwp,
                                                         uint32_t &curReg, int32_t &ex, uint32_t &h, uint32_t &kb,
                                                         uint32_t &panelIdx, float &prod, float &sign,
                                                         uint32_t &singular) {

        SharedSetFlagStage24<kCmStride, kPanel>(base, bw, bwp, curReg, h, kb);
        LocalTensor<uint32_t> invPos(TPosition::VECCALC, kLpPanel + 2048, kCmStride);
        LocalTensor<float> alive(TPosition::VECCALC, kLpPanel + 4096, kCmStride);
        LocalTensor<uint32_t> rowOffsets(TPosition::VECCALC, kLpPanel + 34816, 128);
        if constexpr (LAZY) {
            LocalTensor<uint32_t> ids(TPosition::VECCALC, kLpPanel + 32768, kCmStride);
            uint32_t hp = (h + 7U) & ~7U;
            DataCopy(posU_, ids, hp);
            DataCopy(invPos, ids, hp);
            Duplicate(alive, 1.0f, h);
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
        }
        uint32_t l21nz = LAZY ? 1U : 0U;
        ProcessLargeParSetFlagPhase22<LAZY, evS2M3, kCmStride>(alive, base, bw, ex, h, invPos, l21nz, prod, rowOffsets,
                                                               sign, singular);
        PublishPanelMetadata60<evS2M3, kCmStride, kPanel, 128U>(base, barBase, panelIdx);
    }

    __aicore__ inline void SharedDuplicateStage33(uint64_t &barBase, int32_t &ex, float &prod, float &sign,
                                                  uint32_t &singular) {
        if (rank_ == 0U) {
            Duplicate(barU_, 0u, ::kBarElems);
            SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
            WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
            DataCopyPad(wsGmU_[barBase], barU_, {1, ::kBarElems * sizeof(uint32_t), 0, 0, 0});
            float detVal = (singular != 0U) ? 0.0f : sign * ProdFinal(prod, ex);
            res_.SetValue(0, detVal);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(detGm_[matIdx_], res_, {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
        }
    }
    template <bool LAZY, auto evS2M3, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParDuplicatePhase24(uint64_t &barBase, uint64_t &base, uint32_t &rowsPer) {

        ProcessLargeParDataCopyPadPhase7<evS2M3>(barBase, base, rowsPer);
        float sign = 1.0f;
        float prod = 1.0f;
        int32_t ex = 0;
        uint32_t singular = 0;
        uint32_t curReg = 0;
        for (uint32_t kb = 0; kb < n_; kb += kPanel) {
            uint32_t h = n_ - kb;
            uint32_t bw = h < kPanel ? h : kPanel;
            uint32_t bwp = (bw + ::kAlign - 1) / ::kAlign * ::kAlign;
            uint32_t tw = n_ - kb - bw;
            uint32_t panelIdx = kb / kPanel;
            if (rank_ == 0U) {
                ProcessLargeParSetFlagPhase23<LAZY, evS2M3, kCmStride, kPanel>(barBase, base, bw, bwp, curReg, ex, h,
                                                                               kb, panelIdx, prod, sign, singular);
            } else {
                ProcessLargeParDataCopyPadPhase8<LAZY>(barBase, panelIdx);
            }
            ProcessLargeParDataCopyPadPhase19<LAZY, evS2M3, kCmStride, kPanel>(barBase, base, bw, curReg, h, kb,
                                                                               panelIdx, rowsPer, singular, tw);
            if (singular != 0U) {
                break;
            }
        }
        SharedDuplicateStage33(barBase, ex, prod, sign, singular);
    }
    template <bool LAZY = false> __aicore__ inline void ProcessLargePar() {
        constexpr uint32_t kPanel = LAZY ? 16U : 24U;
        constexpr uint32_t kCmStride = FAST_MC && LAZY ? 256U : ::kMaxN;
        if constexpr (LD_BATCH == 2U && !LAZY) {
            if (rank_ == 0U) {
                for (uint32_t i = 0; i < n_; ++i) {
                    tabI_.SetValue(i, static_cast<int32_t>(i * kPanel * sizeof(float)));
                }
            }
        }
        uint64_t base = static_cast<uint64_t>(matIdx_) * wsSpan_;
        uint64_t barBase = base + 2U * nnS_ + kCmStride * kPanel + 32;
        constexpr int32_t evS2M3 = ::kEvSV;
        uint32_t rowsPer = (n_ + groupSize_ - 1) / groupSize_;
        ProcessLargeParDuplicatePhase24<LAZY, evS2M3, kCmStride, kPanel>(barBase, base, rowsPer);
    }

    __aicore__ inline void FinishTail208vaxpyPhase1(uint32_t &c0, __ubuf__ float *&cachedColumnUB,
                                                    __ubuf__ float *&cachedMatrixUB, __ubuf__ float *&cachedRowUB,
                                                    uint32_t &i, uint32_t &k, uint32_t &kA, uint32_t &w) {
        return ProcessSmallRawUBvaxpyPhase6(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, w);
    }

    __aicore__ inline void FinishTail208vmulsPhase2(__ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB,
                                                    __ubuf__ float *&cachedRowUB, uint32_t &cnt, int32_t &ex,
                                                    LocalTensor<float> &factors, uint32_t &k, uint32_t &kA, uint32_t &p,
                                                    float &piv, float &prod, uint32_t &wA) {

        ProdAccum(prod, ex, piv);
        float inv = -1.0f / piv;
        SmallMaskSet(wA);
        if (p != k) {
            __ubuf__ float *mp = cachedMatrixUB;
            vmuls(cachedRowUB, mp + (p * s_ + kA), inv, 1, 1, 1, 8, 8);
            vmuls(mp + (p * s_ + kA), mp + (k * s_ + kA), 1.0f, 1, 1, 1, 8, 8);
        } else {
            vmuls(cachedRowUB, cachedMatrixUB + (k * s_ + kA), inv, 1, 1, 1, 8, 8);
        }
        SmallMaskReset();
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = n_ - c0;
        uint32_t i = k + 1;
        if (wA > 64U) {
            FinishTail208vaxpyPhase1(c0, cachedColumnUB, cachedMatrixUB, cachedRowUB, i, k, kA, w);
        } else {
            if (cnt > 1U) {
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB), 1, 8,
                      static_cast<uint8_t>((cnt + 7U) / 8U));
                PipeBarrier<PIPE_V>();
                for (uint32_t col = c0; col < n_; col += 64U) {
                    uint32_t width = s_ - col;
                    Shared24VmlaStage18(cachedMatrixUB, cachedRowUB, cnt, col, factors, k, kA, width);
                }
            }
        }
    }
