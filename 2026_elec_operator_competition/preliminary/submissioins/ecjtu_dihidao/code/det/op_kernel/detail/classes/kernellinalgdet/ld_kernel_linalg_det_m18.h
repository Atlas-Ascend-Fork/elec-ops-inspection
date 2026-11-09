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
    template <bool LAZY, auto evS2M3, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessSuper32SetFlagPhase21(uint64_t &barBase, uint64_t &base, uint32_t &bw, uint32_t &bwp,
                                                        uint32_t &curReg, int32_t &ex, uint32_t &h, uint32_t &kb,
                                                        uint32_t &panelIdx, float &prod, float &sign,
                                                        uint32_t &singular) {

        SharedSetFlagStage24<kCmStride, kPanel>(base, bw, bwp, curReg, h, kb);
        LocalTensor<uint32_t> invPos(TPosition::VECCALC, kInvPos, kCmStride);
        LocalTensor<float> alive(TPosition::VECCALC, kAlive, kCmStride);
        LocalTensor<uint32_t> rowOffsets(TPosition::VECCALC, kRows, 256);
        if constexpr (LAZY) {
            LocalTensor<uint32_t> ids(TPosition::VECCALC, kIds, kCmStride);
            uint32_t hp = (h + 7U) & ~7U;
            DataCopy(posU_, ids, hp);
            DataCopy(invPos, ids, hp);
            Duplicate(alive, 1.0f, h);
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
        }
        uint32_t l21nz = LAZY ? 1U : 0U;
        ProcessSuper32SetFlagPhase20<LAZY, evS2M3, kCmStride>(alive, base, bw, ex, h, invPos, l21nz, prod, rowOffsets,
                                                              sign, singular);
        PublishPanelMetadata60<evS2M3, kCmStride, kPanel, 192U>(base, barBase, panelIdx);
    }
    template <bool LAZY, auto evS2M3, auto kCmStride, auto kPanel, auto kUStride>
    __aicore__ inline void ProcessSuper32constexprPhase22(uint64_t &barBase, uint64_t &base, uint32_t &curReg,
                                                          int32_t &ex, uint32_t &h, uint32_t &kb, float &prod,
                                                          uint32_t &rowsPer, float &sign, uint32_t &singular) {

        uint32_t bw = h < kPanel ? h : kPanel;
        if constexpr (WIDE512) {
            if (h <= 224U)
                bw = 16U;
        } else if constexpr (FORCE_FAST) {
            if (n_ == 256U && kb == 32U)
                bw = 16U;
        }
        uint32_t bwp = (bw + ::kAlign - 1) / ::kAlign * ::kAlign;
        uint32_t tw = n_ - kb - bw;
        uint32_t panelIdx = kb / kPanel;
        if (rank_ == 0U) {
            ProcessSuper32SetFlagPhase21<LAZY, evS2M3, kCmStride, kPanel>(barBase, base, bw, bwp, curReg, ex, h, kb,
                                                                          panelIdx, prod, sign, singular);
        } else {
            ProcessSuper32DataCopyPadPhase4<LAZY>(barBase, panelIdx);
        }
        ProcessSuper32DataCopyPadPhase17<LAZY, evS2M3, kCmStride, kPanel, kUStride>(barBase, base, bw, curReg, h, kb,
                                                                                    panelIdx, rowsPer, singular, tw);
    }
    template <bool LAZY = false> __aicore__ inline void ProcessSuper32() {
        constexpr uint32_t kPanel = 32U;
        constexpr uint32_t kUStride = kFactorPitch;
        constexpr uint32_t kCmStride = kFactorPitch;
        if constexpr (LD_BATCH == 2U && !LAZY) {
            if (rank_ == 0U) {
                for (uint32_t i = 0; i < n_; ++i) {
                    tabI_.SetValue(i, static_cast<int32_t>(i * kPanel * sizeof(float)));
                }
            }
        }
        uint64_t base = static_cast<uint64_t>(matIdx_) * wsSpan_;
        uint64_t barBase = base + 2U * nnS_ + kCmStride * kPanel + 48;
        constexpr int32_t evS2M3 = ::kEvSV;
        uint32_t rowsPer = (n_ + groupSize_ - 1) / groupSize_;
        ProcessSuper32DataCopyPadPhase3<evS2M3>(barBase, base, rowsPer);
        float sign = 1.0f;
        float prod = 1.0f;
        int32_t ex = 0;
        uint32_t singular = 0;
        uint32_t curReg = 0;
        for (uint32_t kb = 0; kb < n_; kb += (WIDE512 ? (n_ - kb <= 224U ? 16U : kPanel)
                                                      : ((FORCE_FAST && n_ == 256U && kb == 32U) ? 16U : kPanel))) {
            uint32_t h = n_ - kb;
            if constexpr (UB_TAIL) {
                if (h <= 208U) {
                    if (rank_ == 0U)
                        FinishTail208(base + static_cast<uint64_t>(curReg) * nnS_, kb, h, sign, prod, ex, singular);
                    break;
                }
            }
            ProcessSuper32constexprPhase22<LAZY, evS2M3, kCmStride, kPanel, kUStride>(barBase, base, curReg, ex, h, kb,
                                                                                      prod, rowsPer, sign, singular);
            if (singular != 0U) {
                break;
            }
        }
        SharedDuplicateStage33(barBase, ex, prod, sign, singular);
    }

    __aicore__ inline void ProcessLargeLDET_RM_PAIR_OFFPhase1(uint32_t &bw, uint32_t &h, uint32_t &l21nz, float &maxv,
                                                              uint32_t &p, uint32_t &sExact, uint32_t &segN,
                                                              uint32_t &t, uint32_t &tA) {

        SharedScalarUpdateStage31(l21nz, maxv, p, sExact, segN, tA);
        if (l21nz == 0U && bw < h && sExact != ((bw - tA) >> 6)) {
            uint32_t rEnd = tA + (((bw - tA) >> 6) + 1U) * 64U;
            if (rEnd > h) {
                rEnd = h;
            }
            for (uint32_t r = bw; r < rEnd; ++r) {
                if (panelCM_.GetValue(t * ::kMaxN + r) != 0.0f) {
                    l21nz = 1U;
                    break;
                }
            }
        }
    }

    __aicore__ inline void ProcessLargeGetValuePhase2(uint32_t &bw, uint32_t &js, uint32_t &p, uint32_t &t) {

        for (; js + 7U < bw; js += 8U) {
            float va0 = panelCM_.GetValue(js * ::kMaxN + t);
            float vb0 = panelCM_.GetValue(js * ::kMaxN + p);
            float va1 = panelCM_.GetValue((js + 1U) * ::kMaxN + t);
            float vb1 = panelCM_.GetValue((js + 1U) * ::kMaxN + p);
            float va2 = panelCM_.GetValue((js + 2U) * ::kMaxN + t);
            float vb2 = panelCM_.GetValue((js + 2U) * ::kMaxN + p);
            float va3 = panelCM_.GetValue((js + 3U) * ::kMaxN + t);
            float vb3 = panelCM_.GetValue((js + 3U) * ::kMaxN + p);
            float va4 = panelCM_.GetValue((js + 4U) * ::kMaxN + t);
            float vb4 = panelCM_.GetValue((js + 4U) * ::kMaxN + p);
            float va5 = panelCM_.GetValue((js + 5U) * ::kMaxN + t);
            float vb5 = panelCM_.GetValue((js + 5U) * ::kMaxN + p);
            float va6 = panelCM_.GetValue((js + 6U) * ::kMaxN + t);
            float vb6 = panelCM_.GetValue((js + 6U) * ::kMaxN + p);
            float va7 = panelCM_.GetValue((js + 7U) * ::kMaxN + t);
            float vb7 = panelCM_.GetValue((js + 7U) * ::kMaxN + p);
            panelCM_.SetValue(js * ::kMaxN + t, vb0);
            panelCM_.SetValue(js * ::kMaxN + p, va0);
            panelCM_.SetValue((js + 1U) * ::kMaxN + t, vb1);
            panelCM_.SetValue((js + 1U) * ::kMaxN + p, va1);
            panelCM_.SetValue((js + 2U) * ::kMaxN + t, vb2);
            panelCM_.SetValue((js + 2U) * ::kMaxN + p, va2);
            panelCM_.SetValue((js + 3U) * ::kMaxN + t, vb3);
            panelCM_.SetValue((js + 3U) * ::kMaxN + p, va3);
            panelCM_.SetValue((js + 4U) * ::kMaxN + t, vb4);
            panelCM_.SetValue((js + 4U) * ::kMaxN + p, va4);
            panelCM_.SetValue((js + 5U) * ::kMaxN + t, vb5);
            panelCM_.SetValue((js + 5U) * ::kMaxN + p, va5);
            panelCM_.SetValue((js + 6U) * ::kMaxN + t, vb6);
            panelCM_.SetValue((js + 6U) * ::kMaxN + p, va6);
            panelCM_.SetValue((js + 7U) * ::kMaxN + t, vb7);
            panelCM_.SetValue((js + 7U) * ::kMaxN + p, va7);
        }
    }

    __aicore__ inline void ProcessLargeDataCopyPadPhase3(uint32_t &W, uint64_t &base, uint32_t &bw, uint32_t &kb,
                                                         uint32_t &p, float &sign, uint32_t &swapPend, uint32_t &t,
                                                         uint32_t &tb, uint32_t &wpad) {

        if (p != t) {
            uint32_t js = 0;
            ProcessLargeGetValuePhase2(bw, js, p, t);
            for (; js < bw; ++js) {
                float va = panelCM_.GetValue(js * ::kMaxN + t);
                float vb = panelCM_.GetValue(js * ::kMaxN + p);
                panelCM_.SetValue(js * ::kMaxN + t, vb);
                panelCM_.SetValue(js * ::kMaxN + p, va);
            }
            if (W > 0) {
                if (swapPend != 0U) {
                    WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
                    swapPend = 0;
                }
                uint64_t r1 = base + static_cast<uint64_t>(kb + t) * s_ + (kb + tb);
                uint64_t r2 = base + static_cast<uint64_t>(kb + p) * s_ + (kb + tb);
                DataCopyPad(tile_, wsGm_[r1], {1, wpad * 4, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
                DataCopyPad(tile_[::kMaxN], wsGm_[r2], {1, wpad * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_MTE3>(::kEvMte2Mte3);
                WaitFlag<HardEvent::MTE2_MTE3>(::kEvMte2Mte3);
                DataCopyPad(wsGm_[r2], tile_, {1, wpad * 4, 0, 0, 0});
                DataCopyPad(wsGm_[r1], tile_[::kMaxN], {1, wpad * 4, 0, 0, 0});
                SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
                swapPend = 1;
            }
            sign = -sign;
        }
    }

    __aicore__ inline void ProcessLargeGetValuePhase4(uint32_t &W, uint32_t &cur, uint32_t &i, float &l0, float &l1,
                                                      float &l10, float &l11, float &l12, float &l13, float &l2,
                                                      float &l3, float &l4, float &l5, float &l6, float &l7, float &l8,
                                                      float &l9, uint32_t &lb) {

        float l14 = panelCM_.GetValue(14 * ::kMaxN + lb);
        float l15 = panelCM_.GetValue(15 * ::kMaxN + lb);
        float l16 = panelCM_.GetValue(16 * ::kMaxN + lb);
        float l17 = panelCM_.GetValue(17 * ::kMaxN + lb);
        float l18 = panelCM_.GetValue(18 * ::kMaxN + lb);
        float l19 = panelCM_.GetValue(19 * ::kMaxN + lb);
        float l20 = panelCM_.GetValue(20 * ::kMaxN + lb);
        float l21 = panelCM_.GetValue(21 * ::kMaxN + lb);
        float l22 = panelCM_.GetValue(22 * ::kMaxN + lb);
        float l23 = panelCM_.GetValue(23 * ::kMaxN + lb);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[0 * ::kUbW], -l0, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[1 * ::kUbW], -l1, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[2 * ::kUbW], -l2, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[3 * ::kUbW], -l3, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[4 * ::kUbW], -l4, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[5 * ::kUbW], -l5, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[6 * ::kUbW], -l6, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[7 * ::kUbW], -l7, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[8 * ::kUbW], -l8, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[9 * ::kUbW], -l9, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[10 * ::kUbW], -l10, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[11 * ::kUbW], -l11, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[12 * ::kUbW], -l12, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[13 * ::kUbW], -l13, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[14 * ::kUbW], -l14, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[15 * ::kUbW], -l15, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[16 * ::kUbW], -l16, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[17 * ::kUbW], -l17, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[18 * ::kUbW], -l18, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[19 * ::kUbW], -l19, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[20 * ::kUbW], -l20, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[21 * ::kUbW], -l21, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[22 * ::kUbW], -l22, W);
        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[23 * ::kUbW], -l23, W);
    }

    __aicore__ inline void ProcessLargeWaitFlagPhase5(uint32_t &W, uint32_t &bw, uint32_t &cur, uint32_t &kb,
                                                      uint32_t &r0, uint32_t &rows) {

        if (rows > ::kTileRows) {
            rows = ::kTileRows;
        }
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        LDET_MASK_SET(W);
        if (bw == kPanel) {
            for (uint32_t i = 0; i < rows; ++i) {
                uint32_t lb = r0 - kb + i;
                float l0 = panelCM_.GetValue(0 * ::kMaxN + lb);
                float l1 = panelCM_.GetValue(1 * ::kMaxN + lb);
                float l2 = panelCM_.GetValue(2 * ::kMaxN + lb);
                float l3 = panelCM_.GetValue(3 * ::kMaxN + lb);
                float l4 = panelCM_.GetValue(4 * ::kMaxN + lb);
                float l5 = panelCM_.GetValue(5 * ::kMaxN + lb);
                float l6 = panelCM_.GetValue(6 * ::kMaxN + lb);
                float l7 = panelCM_.GetValue(7 * ::kMaxN + lb);
                float l8 = panelCM_.GetValue(8 * ::kMaxN + lb);
                float l9 = panelCM_.GetValue(9 * ::kMaxN + lb);
                float l10 = panelCM_.GetValue(10 * ::kMaxN + lb);
                float l11 = panelCM_.GetValue(11 * ::kMaxN + lb);
                float l12 = panelCM_.GetValue(12 * ::kMaxN + lb);
                float l13 = panelCM_.GetValue(13 * ::kMaxN + lb);
                ProcessLargeGetValuePhase4(W, cur, i, l0, l1, l10, l11, l12, l13, l2, l3, l4, l5, l6, l7, l8, l9, lb);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                for (uint32_t t = 0; t < bw; ++t) {
                    float l = panelCM_.GetValue(t * ::kMaxN + (r0 - kb + i));
                    LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[t * ::kUbW], -l, W);
                }
            }
        }
        LDET_MASK_RST();
    }

    __aicore__ inline void ProcessLargeSetFlagPhase6(uint32_t &W, uint64_t &base, uint32_t &bw, uint32_t &kb,
                                                     uint32_t &nTiles, uint32_t &rBase, uint32_t &rows0, uint32_t &tb,
                                                     uint32_t &wpad) {

        if (rows0 > ::kTileRows) {
            rows0 = ::kTileRows;
        }
        CopyIn2D(tile_, base + static_cast<uint64_t>(rBase) * s_ + (kb + tb), rows0, wpad, s_, ::kUbW);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        for (uint32_t ti = 0; ti < nTiles; ++ti) {
            uint32_t cur = (ti & 1U) * ::kTileRows * ::kUbW;
            uint32_t r0 = rBase + ti * ::kTileRows;
            uint32_t rows = n_ - r0;
            ProcessLargeWaitFlagPhase5(W, bw, cur, kb, r0, rows);
            LDET_VFENCE();
            SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
            WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
            if (ti + 1 < nTiles) {
                uint32_t r1 = r0 + ::kTileRows;
                uint32_t rows1 = n_ - r1;
                if (rows1 > ::kTileRows) {
                    rows1 = ::kTileRows;
                }
                if (ti >= 1U) {
                    WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
                }
                CopyOut2D(base + static_cast<uint64_t>(r0) * s_ + (kb + tb), tile_[cur], rows, wpad, ::kUbW, s_);
                SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
                CopyIn2D(tile_[((ti + 1) & 1U) * ::kTileRows * ::kUbW], base + static_cast<uint64_t>(r1) * s_ + (kb + tb),
                         rows1, wpad, s_, ::kUbW);
                SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
            } else {
                CopyOut2D(base + static_cast<uint64_t>(r0) * s_ + (kb + tb), tile_[cur], rows, wpad, ::kUbW, s_);
                SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
            }
        }
    }

    __aicore__ inline void ProcessLargeSetFlagPhase7(uint32_t &W, uint64_t &base, uint32_t &bw, uint32_t &kb,
                                                     uint32_t &l21nz, uint32_t &m, uint32_t &nTiles, uint32_t &rBase,
                                                     uint32_t &singular, uint32_t &tb, uint32_t &vMte2Pend,
                                                     uint32_t &wpad) {

        if (n_ > rBase) {
            nTiles = (n_ - rBase + ::kTileRows - 1) / ::kTileRows;
        }
        if (singular == 0U && W > 0 && l21nz != 0U && nTiles > 0) {
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
            LDET_VFENCE();
            SetFlag<HardEvent::V_MTE2>(::kEvVMte2);
            WaitFlag<HardEvent::V_MTE2>(::kEvVMte2);
            CopyIn2D(ubuf_, base + static_cast<uint64_t>(kb) * s_ + (kb + tb), bw, wpad, s_, ::kUbW);
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
            WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
            LDET_MASK_SET(W);
            for (uint32_t t = 1; t < bw; ++t) {
                for (uint32_t s2 = 0; s2 < t; ++s2) {
                    float l = panelCM_.GetValue(s2 * ::kMaxN + t);
                    LDET_AXPY_TRK(ubuf_[t * ::kUbW], ubuf_[s2 * ::kUbW], -l, W);
                }
            }
            LDET_MASK_RST();
            {
                uint32_t rows0 = n_ - rBase;
                ProcessLargeSetFlagPhase6(W, base, bw, kb, nTiles, rBase, rows0, tb, wpad);
                WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
                if (nTiles >= 2U) {
                    WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
                }
            }
        }
        if (kb + kPanel < n_ || m + 1 < cnt_) {
            LDET_VFENCE();
            SetFlag<HardEvent::V_MTE2>(::kEvVMte2);
            vMte2Pend = 1;
        }
    }

    __aicore__ inline void ProcessLargeSetFlagPhase8(uint32_t &W, uint64_t &base, uint32_t &bw, uint32_t &cnt,
                                                     int32_t &ex, uint32_t &kb, uint32_t &p, float &prod, float &sign,
                                                     uint32_t &swapPend, uint32_t &t, uint32_t &tA, uint32_t &tb,
                                                     uint32_t &wpad) {

        ProcessLargeDataCopyPadPhase3(W, base, bw, kb, p, sign, swapPend, t, tb, wpad);
        float piv = panelCM_.GetValue(t * ::kMaxN + t);
        ProdAccum(prod, ex, piv);
        float inv = LdetSharedOperations::ReciprocalWithZero58(piv, 1.0f);
        if (p != t) {
            SetFlag<HardEvent::S_V>(::kEvSV);
            WaitFlag<HardEvent::S_V>(::kEvSV);
        }
        LDET_MASK_SET(cnt);
        vmuls(reinterpret_cast<__ubuf__ float *>(panelCM_[t * ::kMaxN + tA].GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(panelCM_[t * ::kMaxN + tA].GetPhyAddr()), inv, 1, 1, 1, 8, 8);
        uint32_t je = t + 1;
        for (; je + 3U < bw; je += 4U) {
            float pj0 = panelCM_.GetValue(je * ::kMaxN + t);
            float pj1 = panelCM_.GetValue((je + 1U) * ::kMaxN + t);
            float pj2 = panelCM_.GetValue((je + 2U) * ::kMaxN + t);
            float pj3 = panelCM_.GetValue((je + 3U) * ::kMaxN + t);
            LDET_AXPY_TRK(panelCM_[je * ::kMaxN + tA], panelCM_[t * ::kMaxN + tA], -pj0, cnt);
            LDET_AXPY_TRK(panelCM_[(je + 1U) * ::kMaxN + tA], panelCM_[t * ::kMaxN + tA], -pj1, cnt);
            LDET_AXPY_TRK(panelCM_[(je + 2U) * ::kMaxN + tA], panelCM_[t * ::kMaxN + tA], -pj2, cnt);
            LDET_AXPY_TRK(panelCM_[(je + 3U) * ::kMaxN + tA], panelCM_[t * ::kMaxN + tA], -pj3, cnt);
        }
        for (; je < bw; ++je) {
            float pj = panelCM_.GetValue(je * ::kMaxN + t);
            LDET_AXPY_TRK(panelCM_[je * ::kMaxN + tA], panelCM_[t * ::kMaxN + tA], -pj, cnt);
        }
        LDET_MASK_RST();
    }
