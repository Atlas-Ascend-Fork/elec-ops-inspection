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
    template <bool LAZY, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParSetFlagPhase6(uint32_t &bw, uint32_t &cur, uint32_t &kb, uint32_t &l21,
                                                        uint32_t &r0, uint32_t &rows, uint32_t &tw, uint32_t &w) {

        if (rows > ::kTileRowsPar) {
            rows = ::kTileRowsPar;
        }
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        if (l21 != 0U && !(CUBE_LU && LAZY && tw > 128U)) {
            LDET_MASK_SET(w);
            if (bw == kPanel) {
                for (uint32_t i = 0; i < rows; ++i) {
                    uint32_t lb = r0 - kb + i;
                    float l0 = lbuf_.GetValue(0 * kCmStride + lb);
                    float l1 = lbuf_.GetValue(1 * kCmStride + lb);
                    float l2 = lbuf_.GetValue(2 * kCmStride + lb);
                    float l3 = lbuf_.GetValue(3 * kCmStride + lb);
                    float l4 = lbuf_.GetValue(4 * kCmStride + lb);
                    float l5 = lbuf_.GetValue(5 * kCmStride + lb);
                    float l6 = lbuf_.GetValue(6 * kCmStride + lb);
                    ProcessLargeParGetValuePhase5<kCmStride, kPanel>(cur, i, l0, l1, l2, l3, l4, l5, l6, lb, w);
                }
            } else {
                for (uint32_t i = 0; i < rows; ++i) {
                    for (uint32_t t = 0; t < bw; ++t) {
                        float l = lbuf_.GetValue(t * kCmStride + (r0 - kb + i));
                        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[t * ::kUbW], -l, w);
                    }
                }
            }
        }
        LDET_MASK_RST();
        LDET_VFENCE();
        SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
        WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
    }
    template <auto evS2M3>
    __aicore__ inline void ProcessLargeParDataCopyPadPhase7(uint64_t &barBase, uint64_t &base, uint32_t &rowsPer) {

        rowsPer = (rowsPer + 3U) / 4U * 4U;
        uint32_t rBegin = rank_ * rowsPer;
        uint32_t rEnd = rBegin + rowsPer;
        if (rBegin > n_) {
            rBegin = n_;
        }
        if (rEnd > n_) {
            rEnd = n_;
        }
        WsInitRange(matIdx_, rBegin, rEnd, base);
        if (rank_ == 0U) {
            uint32_t preLo = ::PreLo();
            uint32_t preHi = ::PreHi();
            for (uint32_t r = 1; r < groupSize_; ++r) {
                while (true) {
                    DataCopyPad(metaU_, wsGmU_[barBase + r * ::kBarSlots], {1, 32, 0, 0, 0},
                                DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
                    SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
                    WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
                    if (metaU_.GetValue(0) == preLo && metaU_.GetValue(1) == preHi) {
                        break;
                    }
                }
            }
        } else {
            SetFlag<HardEvent::MTE3_S>(::kEvVMte2);
            WaitFlag<HardEvent::MTE3_S>(::kEvVMte2);
            barU_.SetValue(0, ::PreLo());
            barU_.SetValue(1, ::PreHi());
            SetFlag<HardEvent::S_MTE3>(evS2M3);
            WaitFlag<HardEvent::S_MTE3>(evS2M3);
            DataCopyPad(wsGmU_[barBase + rank_ * ::kBarSlots], barU_, {1, 32, 0, 0, 0});
        }
    }
    template <bool LAZY>
    __aicore__ inline void ProcessLargeParDataCopyPadPhase8constexprDepth22Stage1(uint32_t &spins) {

        ++spins;
        if constexpr (FAST_MC && LAZY) {
            bool shorterPoll = n_ >= 224U;
            if (spins >= (shorterPoll ? 2U : 8U)) {
                for (volatile uint32_t d = 0; d < (shorterPoll ? 64U : 256U); ++d) {
                }
            }
        } else {
            if (spins >= 8U) {
                for (volatile uint32_t d = 0; d < 256U; ++d) {
                }
            }
        }
    }
    template <bool LAZY>
    __aicore__ inline void ProcessLargeParDataCopyPadPhase8(uint64_t &barBase, uint32_t &panelIdx) {

        uint32_t tagLo = ::TagLo(panelIdx);
        uint32_t tagHi = ::TagHi(panelIdx);
        uint32_t spins = 0;
        while (true) {
            DataCopyPad(metaU_, wsGmU_[barBase], {1, 32, 0, 0, 0}, DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
            WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
            if (metaU_.GetValue(0) == tagLo && metaU_.GetValue(1) == tagHi) {
                break;
            }
            ProcessLargeParDataCopyPadPhase8constexprDepth22Stage1<LAZY>(spins);
        }
    }

    __aicore__ inline void ProcessLargeParSetFlagPhase9SetFlagDepth22Stage1(uint64_t &barBase, uint32_t &panelIdx) {

        SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
        uint32_t t2Lo = ::T2Lo(panelIdx);
        uint32_t t2Hi = ::T2Hi(panelIdx);
        if (groupSize_ > 1U) {
            uint32_t pollBytes = (groupSize_ - 1U) * ::kBarSlots * sizeof(uint32_t);
            bool allReady = false;
            while (!allReady) {
                DataCopyPad(pollU_, wsGmU_[barBase + ::kBarSlots], {1, pollBytes, 0, 0, 0},
                            DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
                WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
                uint32_t got = 1U;
                for (uint32_t r = 0; r + 1U < groupSize_; ++r) {
                    if (pollU_.GetValue(r * ::kBarSlots) == t2Lo && pollU_.GetValue(r * ::kBarSlots + 1U) == t2Hi) {
                        ++got;
                    }
                }
                allReady = (got == groupSize_);
            }
        }
    }
    template <auto evS2M3> __aicore__ inline void ProcessLargeParSetFlagPhase9(uint64_t &barBase, uint32_t &panelIdx) {

        if (rank_ == 0U) {
            ProcessLargeParSetFlagPhase9SetFlagDepth22Stage1(barBase, panelIdx);
        } else {
            SetFlag<HardEvent::MTE3_S>(::kEvVMte2);
            WaitFlag<HardEvent::MTE3_S>(::kEvVMte2);
            barU_.SetValue(0, ::T2Lo(panelIdx));
            barU_.SetValue(1, ::T2Hi(panelIdx));
            SetFlag<HardEvent::S_MTE3>(evS2M3);
            WaitFlag<HardEvent::S_MTE3>(evS2M3);
            DataCopyPad(wsGmU_[barBase + rank_ * ::kBarSlots], barU_, {1, 32, 0, 0, 0});
        }
    }

    __aicore__ inline void SharedScalarUpdateStage31(uint32_t &l21nz, float &maxv, uint32_t &p, uint32_t &sExact,
                                                     uint32_t &segN, uint32_t &tA) {
        for (uint32_t s = 0; s < segN; ++s) {
            uint32_t so = LDET_RM_PAIR_OFF(s);
            float v = tmp_.GetValue(so);
            if (v > maxv) {
                float idxF = tmp_.GetValue(so + 1);
                maxv = v;
                p = tA + s * 64U + *reinterpret_cast<uint32_t *>(&idxF);
            }
            if (l21nz == 0U && v != 0.0f && s >= sExact) {
                l21nz = 1U;
            }
        }
    }
    template <auto kCmStride>
    __aicore__ inline void ProcessLargeParLDET_RM_PAIR_OFFPhase10(uint32_t &bw, uint32_t &h, uint32_t &l21nz,
                                                                  float &maxv, uint32_t &p, uint32_t &sExact,
                                                                  uint32_t &segN, uint32_t &t, uint32_t &tA) {

        SharedScalarUpdateStage31(l21nz, maxv, p, sExact, segN, tA);
        if (l21nz == 0U && bw < h && sExact != ((bw - tA) >> 6)) {
            uint32_t rEnd2 = tA + (((bw - tA) >> 6) + 1U) * 64U;
            if (rEnd2 > h) {
                rEnd2 = h;
            }
            for (uint32_t r = bw; r < rEnd2; ++r) {
                if (panelCM_.GetValue(t * kCmStride + r) != 0.0f) {
                    l21nz = 1U;
                    break;
                }
            }
        }
    }

    __aicore__ inline void Shared10WaitFlagStage3(uint32_t &bw, uint32_t &cur, uint64_t &dstReg, uint32_t &kb,
                                                  uint32_t &nTiles, uint32_t &r0, uint32_t &rows, uint64_t &srcReg,
                                                  uint32_t &ti, uint32_t &w, uint32_t &wR1, uint32_t &wpad) {
        if (ti + 1 < nTiles) {
            uint32_t r1 = r0 + ::kTileRowsPar;
            uint32_t rows1 = wR1 - r1;
            if (rows1 > ::kTileRowsPar) {
                rows1 = ::kTileRowsPar;
            }
            if (ti >= 1U) {
                WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
            }
            CopyOut2D(dstReg + static_cast<uint64_t>(r0) * s_ + kb + bw, tile_[cur], rows, wpad, ::kUbW, s_);
            SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
            for (uint32_t i = 0; i < rows1; ++i) {
                uint64_t rs = srcReg + static_cast<uint64_t>(kb + posU_.GetValue(r1 - kb + i)) * s_ + kb + bw;
                DataCopyPad(tile_[((ti + 1) & 1U) * ::kTileRowsPar * ::kUbW + i * ::kUbW], wsGm_[rs], {1, w * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        } else {
            CopyOut2D(dstReg + static_cast<uint64_t>(r0) * s_ + kb + bw, tile_[cur], rows, wpad, ::kUbW, s_);
            SetFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
        }
    }
    template <bool LAZY, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParSetFlagPhase11(uint32_t &bw, uint64_t &dstReg, uint32_t &kb, uint32_t &l21,
                                                         uint32_t &nTiles, uint64_t &srcReg, uint32_t &tw, uint32_t &w,
                                                         uint32_t &wR0, uint32_t &wR1, uint32_t &wpad) {

        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        for (uint32_t ti = 0; ti < nTiles; ++ti) {
            uint32_t cur = (ti & 1U) * ::kTileRowsPar * ::kUbW;
            uint32_t r0 = wR0 + ti * ::kTileRowsPar;
            uint32_t rows = wR1 - r0;
            ProcessLargeParSetFlagPhase6<LAZY, kCmStride, kPanel>(bw, cur, kb, l21, r0, rows, tw, w);
            Shared10WaitFlagStage3(bw, cur, dstReg, kb, nTiles, r0, rows, srcReg, ti, w, wR1, wpad);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
    }
    template <bool SUPER, auto kCmStride, class Alive, class Positions>
    __aicore__ inline void UpdatePanelPivot54(Alive &alive, uint32_t &bw, int32_t &ex, uint32_t &h, Positions &invPos,
                                              uint32_t &logicalP, uint32_t &nCh, uint32_t &p, bool &panelUpdatePending,
                                              uint32_t &physicalT, float &prod, LocalTensor<uint32_t> &rowOffsets,
                                              float &sign, uint32_t &t) {

        if (logicalP != t) {
            posU_.SetValue(t, p * 4U);
            posU_.SetValue(logicalP, physicalT * 4U);
            invPos.SetValue(p, t * 4U);
            invPos.SetValue(physicalT, logicalP * 4U);
            sign = -sign;
        }
        float piv = panelCM_.GetValue(t * kCmStride + p);
        ProdAccum(prod, ex, piv);
        float inv = -1.0f / piv;
        alive.SetValue(p, 0.0f);
        if (panelUpdatePending) {
            WaitFlag<HardEvent::V_S>(::kEvVS);
            panelUpdatePending = false;
        }
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        if (t + 1U < bw) {
            Gather(tmp_[32], panelCM_[p & ~7U], rowOffsets[(p & 7U) * (SUPER ? 32U : 16U)], 0U, bw);
            PipeBarrier<PIPE_V>();
        }
        LDET_MASK_SET(h);
        LdetPanelMul(panelCM_[t * kCmStride], panelCM_[t * kCmStride], alive);
        LdetPanelMuls(panelCM_[t * kCmStride], inv);
        LDET_MASK_RST();
        if (t + 1U < bw) {
            LocalTensor<float> factors(TPosition::VECCALC, SUPER ? kLpCp + 4096U : kLpTmp + 128U * sizeof(float),
                                       SUPER ? 256U : 128U);
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(tmp_[32].GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((bw + 7U) / 8U));
            if constexpr (SUPER)
                ProcessSuper32PipeBarrierPhase2<kCmStride>(alive, bw, factors, h, nCh, t);
            else
                ProcessLargeParPipeBarrierPhase4<kCmStride>(alive, bw, factors, h, nCh, t);
            SetFlag<HardEvent::V_S>(::kEvVS);
            panelUpdatePending = true;
        }
    }
    template <bool LAZY, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParDataCopyPadPhase13(uint32_t &bw, uint32_t &curReg, uint64_t &dstReg,
                                                             uint32_t &kb, uint32_t &l21, uint64_t &srcReg,
                                                             uint32_t &tw, uint32_t &w, uint32_t &wR0, uint32_t &wR1,
                                                             uint32_t &wpad) {

        if (wR0 < wR1) {
            ProcessLargeParSetFlagPhase2<LAZY, kCmStride>(bw, kb, l21, srcReg, tw, w);
            {
                uint32_t nTiles = (wR1 - wR0 + ::kTileRowsPar - 1) / ::kTileRowsPar;
                uint32_t rows0 = wR1 - wR0;
                if (rows0 > ::kTileRowsPar) {
                    rows0 = ::kTileRowsPar;
                }
                for (uint32_t i = 0; i < rows0; ++i) {
                    uint64_t rs = srcReg + static_cast<uint64_t>(kb + posU_.GetValue(wR0 - kb + i)) * s_ + kb + bw;
                    DataCopyPad(tile_[i * ::kUbW], wsGm_[rs], {1, w * 4, 0, 0, 0},
                                DataCopyPadExtParams<float>{false, 0, 0, 0});
                }
                ProcessLargeParSetFlagPhase11<LAZY, kCmStride, kPanel>(bw, dstReg, kb, l21, nTiles, srcReg, tw, w, wR0,
                                                                       wR1, wpad);
                if (nTiles >= 2U) {
                    WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
                }
            }
            if constexpr (CUBE_LU && LAZY) {
                if (tw > 128U) {
                    CubePrepareL(kb, wR0, wR1);
                }
            }
        }
        curReg = 1U - curReg;
    }

    __aicore__ inline void SharedScalarUpdateStage28(uint32_t &bw, uint32_t &h) {
        for (uint32_t i = 0; i < h; ++i) {
            posU_.SetValue(i, i);
        }
        for (uint32_t t = 0; t < bw; ++t) {
            uint32_t p = metaU_.GetValue(1 + t);
            if (p != t) {
                uint32_t va = posU_.GetValue(t);
                uint32_t vb = posU_.GetValue(p);
                posU_.SetValue(t, vb);
                posU_.SetValue(p, va);
            }
        }
    }
    template <bool LAZY, auto kCmStride, auto kPanel>
    __aicore__ inline void ProcessLargeParGetValuePhase14(uint64_t &base, uint32_t &bw, uint32_t &curReg, uint32_t &h,
                                                          uint32_t &kb, uint32_t &tw, uint32_t &wR0, uint32_t &wR1) {

        uint32_t w = tw;
        uint32_t wpad = (w + ::kAlign - 1) / ::kAlign * ::kAlign;
        uint32_t l21 = metaU_.GetValue(25);
        uint32_t needPos = PanelPositionChanged60(bw);
        uint64_t srcReg = base + static_cast<uint64_t>(curReg) * nnS_;
        if (l21 == 0U && needPos == 0U) {
        } else {
            uint64_t dstReg = base + static_cast<uint64_t>(1U - curReg) * nnS_;
            SharedScalarUpdateStage28(bw, h);
            ProcessLargeParDataCopyPadPhase13<LAZY, kCmStride, kPanel>(bw, curReg, dstReg, kb, l21, srcReg, tw, w, wR0,
                                                                       wR1, wpad);
        }
    }
    template <bool LAZY>
    __aicore__ inline void SharedScalarUpdateStage19(uint32_t &bw, uint32_t &kb, uint32_t &rowsAll, uint32_t &tw,
                                                     uint32_t &wR0, uint32_t &wR1) {
        if (wR1 > n_) {
            wR1 = n_;
        }
        if constexpr (CUBE_LU && LAZY) {
            if (tw > 128U) {
                uint32_t band;
                if constexpr (FORCE_FAST)
                    band = LuN256Span(rowsAll, groupSize_ / 2U);
                else
                    band = LuSpan(rowsAll);
                uint32_t half = ((band / 2U + 15U) / 16U) * 16U;
                uint32_t owner = rank_ / 2U;
                wR0 = kb + bw + owner * band + (rank_ % 2U) * half;
                wR1 = wR0 + half;
                uint32_t limit = kb + bw + (owner + 1U) * band;
                if (wR1 > limit)
                    wR1 = limit;
                if (wR1 > n_)
                    wR1 = n_;
            }
        }
    }
    template <bool SUPER, bool LAZY, auto kCmStride, auto kPanel, auto kUStride = 0U>
    __aicore__ inline void UpdateSchurRegion54(uint64_t &base, uint32_t &bw, uint32_t &curReg, uint32_t &h,
                                               uint32_t &kb, uint32_t &panelIdx, uint32_t &rowsPer, uint32_t &singular,
                                               uint32_t &tw) {

        singular = metaU_.GetValue(0);
        if (tw > 0 && singular == 0U && kb + kPanel < n_) {
            uint32_t rowsAll = n_ - kb - bw;
            uint32_t rowsPer = (rowsAll + groupSize_ - 1) / groupSize_;
            uint32_t wR0 = kb + bw + rank_ * rowsPer;
            uint32_t wR1 = wR0 + rowsPer;
            SharedScalarUpdateStage19<LAZY>(bw, kb, rowsAll, tw, wR0, wR1);
            if constexpr (SUPER)
                ProcessSuper32GetValuePhase12<LAZY, kCmStride, kUStride>(base, bw, curReg, h, kb, tw, wR0, wR1);
            else
                ProcessLargeParGetValuePhase14<LAZY, kCmStride, kPanel>(base, bw, curReg, h, kb, tw, wR0, wR1);
        }
        if constexpr (CUBE_LU && LAZY) {
            if (tw > 128U) {
                CubeStageEnd(panelIdx, singular, curReg, tw);
            }
        }
    }
