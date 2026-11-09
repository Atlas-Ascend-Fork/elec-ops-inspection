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
    __aicore__ inline void FinishTail208SetFlagPhase3(uint32_t &globalK, uint32_t &originalS, uint64_t &sourceBase,
                                                      uint32_t &tailN) {

        uint32_t tailStride = (tailN + 7U) & ~7U;
        mat_ = LocalTensor<float>(TPosition::VECCALC, ::kSpMat, ::kSmallMaxN * ::kSmallMaxN);
        colbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpCol, ::kSmallMaxN);
        absbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpAbs, ::kSmallMaxN);
        rowkbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRowk, ::kSmallMaxN);
        tabI_ = LocalTensor<int32_t>(TPosition::VECCALC, ::kSpTab, ::kSmallMaxN);
        tabRI_ = LocalTensor<int32_t>(TPosition::VECCALC, ::kSpTabR, 8U * ::kSmallMaxN);
        tabRU_ = LocalTensor<uint32_t>(TPosition::VECCALC, ::kSpTabR, 8U * ::kSmallMaxN);
        tmp_ = LocalTensor<float>(TPosition::VECCALC, ::kSpTmp, 256U);
        res_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRes, ::kMaxPerCore);
        for (uint32_t i = 0; i < tailN; ++i)
            tabI_.SetValue(i, static_cast<int32_t>(i * tailStride * 4U));
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        LDET_MASK_SET(tailN);
        for (uint32_t r = 0; r < 8U; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(tabRI_[r * ::kSmallMaxN].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tabI_.GetPhyAddr()), static_cast<int32_t>(r * 4U), 1, 1, 1, 8,
                  8);
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
        CopyIn2D(mat_, sourceBase + static_cast<uint64_t>(globalK) * originalS + globalK, tailN, tailStride, originalS,
                 tailStride);
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        n_ = tailN;
        s_ = tailStride;
    }
    __aicore__ inline void FinishTail208SetFlagPhase4(uint32_t &cnt) {

        SmallMaskReset();
        SmallMaskSet(cnt);
        for (uint32_t c = 0, coff = 0; c < (cnt + 127U) >> 7; ++c, coff += 128U) {
            uint32_t cc = cnt - coff;
            if (cc > 128U)
                cc = 128U;
            AscendCUtils::SetMask<float>(0, cc);
            LdetVcmax128(tmp_[c * 8], absbuf_[coff], cc);
        }
        SmallMaskReset();
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }

    __aicore__ inline void FinishTail208SetValuePhase5(__ubuf__ float *&cachedColumnUB, __ubuf__ float *&cachedMatrixUB,
                                                       __ubuf__ float *&cachedRowUB, uint32_t &cnt, int32_t &ex,
                                                       LocalTensor<float> &factors, uint32_t &k, uint32_t &kA,
                                                       uint32_t &p, float &prod, float &sign) {

        uint32_t wA = s_ - kA;
        float piv;
        if (p != k) {
            float tv = cachedColumnUB[0];
            colbuf_.SetValue(0, cachedColumnUB[p - k]);
            colbuf_.SetValue(p - k, tv);
            sign = -sign;
        }
        piv = cachedColumnUB[0];
        FinishTail208vmulsPhase2(cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, ex, factors, k, kA, p, piv, prod,
                                 wA);
        SmallMaskReset();
    }
    __aicore__ inline void FinishTail208(uint64_t sourceBase, uint32_t globalK, uint32_t tailN, float &sign,
                                         float &prod, int32_t &ex, uint32_t &singular) {
        PipeBarrier<PIPE_ALL>();
        uint32_t originalN = n_, originalS = s_;
        FinishTail208SetFlagPhase3(globalK, originalS, sourceBase, tailN);
        __ubuf__ float *cachedColumnUB = reinterpret_cast<__ubuf__ float *>(colbuf_.GetPhyAddr());
        __ubuf__ float *cachedAbsUB = reinterpret_cast<__ubuf__ float *>(absbuf_.GetPhyAddr());
        __ubuf__ float *cachedRowUB = reinterpret_cast<__ubuf__ float *>(rowkbuf_.GetPhyAddr());
        __ubuf__ float *cachedReduceUB = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
        __ubuf__ float *cachedMatrixUB = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
        __ubuf__ uint32_t *cachedOffsetUB = reinterpret_cast<__ubuf__ uint32_t *>(tabRU_.GetPhyAddr());
        LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t kA = k & ~(::kAlign - 1);
            uint32_t cnt = n_ - k;
            uint32_t be = k * (s_ + 1U);
            SmallMaskSet(cnt);
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(cachedColumnUB),
                    reinterpret_cast<__ubuf__ uint32_t *>((cachedOffsetUB + (be & 7U) * ::kSmallMaxN)),
                    reinterpret_cast<uint64_t>(cachedMatrixUB) + (be & ~7U) * sizeof(float), 8, 1);
            PipeBarrier<PIPE_V>();
            vabs(cachedAbsUB, cachedColumnUB, 1, 1, 1, 8, 8);
            uint32_t segN = (cnt + 63U) / 64U;
            FinishTail208SetFlagPhase4(cnt);
            float maxv = 0.0f;
            uint32_t p = k;
            MergeMaxSegments64<true>(cachedReduceUB, segN, k, maxv, p);
            if (maxv == 0.0f) {
                singular = 1;
                break;
            }
            FinishTail208SetValuePhase5(cachedColumnUB, cachedMatrixUB, cachedRowUB, cnt, ex, factors, k, kA, p, prod,
                                        sign);
        }
        PipeBarrier<PIPE_ALL>();
        n_ = originalN;
        s_ = originalS;
    }
    template <bool LAZY, auto evS2M3, auto kCmStride>
    __aicore__ inline void ProcessSuper32SetFlagPhase1(uint64_t &base, uint32_t &bw, uint32_t &h, uint32_t &l21nz,
                                                       uint32_t &singular) {

        SharedSetFlagStage25<LAZY, kCmStride>(bw, h);
        metaU_.SetValue(0, singular);
        metaU_.SetValue(33, l21nz);
        SharedSetFlagStage26<LAZY, evS2M3, kCmStride>(base, bw, h, l21nz);
    }
    template <auto kCmStride, class PhaseArg0>
    __aicore__ inline void ProcessSuper32PipeBarrierPhase2(PhaseArg0 &alive, uint32_t &bw, LocalTensor<float> &factors,
                                                           uint32_t &h, uint32_t &nCh, uint32_t &t) {
        return ProcessLargeParPipeBarrierPhase4<kCmStride, PhaseArg0>(alive, bw, factors, h, nCh, t);
    }
    template <auto evS2M3>
    __aicore__ inline void ProcessSuper32DataCopyPadPhase3(uint64_t &barBase, uint64_t &base, uint32_t &rowsPer) {
        return ProcessLargeParDataCopyPadPhase7<evS2M3>(barBase, base, rowsPer);
    }
    template <bool LAZY> __aicore__ inline void ProcessSuper32DataCopyPadPhase4(uint64_t &barBase, uint32_t &panelIdx) {
        return ProcessLargeParDataCopyPadPhase8<LAZY>(barBase, panelIdx);
    }
    template <auto evS2M3> __aicore__ inline void ProcessSuper32SetFlagPhase5(uint64_t &barBase, uint32_t &panelIdx) {
        return ProcessLargeParSetFlagPhase9<evS2M3>(barBase, panelIdx);
    }
    template <auto kCmStride>
    __aicore__ inline void ProcessSuper32LDET_RM_PAIR_OFFPhase6(uint32_t &bw, uint32_t &h, uint32_t &l21nz, float &maxv,
                                                                uint32_t &p, uint32_t &sExact, uint32_t &segN,
                                                                uint32_t &t, uint32_t &tA) {
        return ProcessLargeParLDET_RM_PAIR_OFFPhase10<kCmStride>(bw, h, l21nz, maxv, p, sExact, segN, t, tA);
    }
    template <auto kCmStride, auto kUStride>
    __aicore__ inline void ProcessSuper32SetFlagPhase7(uint32_t &bw, uint32_t &kb, uint32_t &l21, uint64_t &srcReg,
                                                       uint32_t &w) {

        if (l21 != 0U) {
            LoadPivotedUpper60<kUStride>(bw, kb, srcReg, w);
            SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
            WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
            if constexpr (BATCH_U01) {
                if (bw == 16U)
                    SuperTriangularU01(w, 0U);
                else
                    SuperBatchedU01(w);
            } else {
                LDET_MASK_SET(w);
                for (uint32_t t = 1; t < bw; ++t) {
                    for (uint32_t s2 = 0; s2 < t; ++s2) {
                        float l = lbuf_.GetValue(s2 * kCmStride + t);
                        LDET_AXPY_TRK(ubuf_[t * kUStride], ubuf_[s2 * kUStride], -l, w);
                    }
                }
                LDET_MASK_RST();
            }
        }
    }

    __aicore__ inline void ProcessSuper32SetFlagPhase9(uint32_t &bw, uint32_t &cur, uint64_t &dstReg, uint32_t &kb,
                                                       uint32_t &nTiles, uint32_t &r0, uint32_t &rows, uint64_t &srcReg,
                                                       uint32_t &ti, uint32_t &w, uint32_t &wR1, uint32_t &wpad) {

        LDET_MASK_RST();
        LDET_VFENCE();
        SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
        WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
        Shared10WaitFlagStage3(bw, cur, dstReg, kb, nTiles, r0, rows, srcReg, ti, w, wR1, wpad);
    }
    template <bool LAZY, auto kCmStride, auto kUStride>
    __aicore__ inline void ProcessSuper32DataCopyPadPhase10(uint32_t &bw, uint64_t &dstReg, uint32_t &kb, uint32_t &l21,
                                                            uint32_t &nTiles, uint64_t &srcReg, uint32_t &tw,
                                                            uint32_t &w, uint32_t &wR0, uint32_t &wR1, uint32_t &wpad) {

        uint32_t rows0 = wR1 - wR0;
        if (rows0 > ::kTileRowsPar) {
            rows0 = ::kTileRowsPar;
        }
        for (uint32_t i = 0; i < rows0; ++i) {
            uint64_t rs = srcReg + static_cast<uint64_t>(kb + posU_.GetValue(wR0 - kb + i)) * s_ + kb + bw;
            DataCopyPad(tile_[i * ::kUbW], wsGm_[rs], {1, w * 4, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        for (uint32_t ti = 0; ti < nTiles; ++ti) {
            uint32_t cur = (ti & 1U) * ::kTileRowsPar * ::kUbW;
            uint32_t r0 = wR0 + ti * ::kTileRowsPar;
            uint32_t rows = wR1 - r0;
            if (rows > ::kTileRowsPar) {
                rows = ::kTileRowsPar;
            }
            WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
            if (l21 != 0U && !(CUBE_LU && LAZY && tw > 128U)) {
                LDET_MASK_SET(w);
                for (uint32_t i = 0; i < rows; ++i) {
                    uint32_t lb = r0 - kb + i;
                    for (uint32_t j = 0; j < bw; ++j) {
                        float l = lbuf_.GetValue(j * kCmStride + lb);
                        LDET_AXPY_TRK(tile_[cur + i * ::kUbW], ubuf_[j * kUStride], -l, w);
                    }
                }
            }
            ProcessSuper32SetFlagPhase9(bw, cur, dstReg, kb, nTiles, r0, rows, srcReg, ti, w, wR1, wpad);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
    }
    template <bool LAZY, auto kCmStride, auto kUStride>
    __aicore__ inline void
    ProcessSuper32SetFlagPhase11SetFlagDepth35Stage1(uint32_t &bw, uint64_t &dstReg, uint32_t &kb, uint32_t &l21,
                                                     uint64_t &srcReg, uint32_t &tw, uint32_t &w, uint32_t &wR0,
                                                     uint32_t &wR1, uint32_t &wpad) {

        ProcessSuper32SetFlagPhase7<kCmStride, kUStride>(bw, kb, l21, srcReg, w);
        if constexpr (CUBE_LU && LAZY) {
            if (tw > 128U) {
                if (rank_ == 0U) {
                    LDET_VFENCE();
                    SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
                    WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
                    DataCopyPad(
                        this->cubeU_, ubuf_,
                        DataCopyExtParams{static_cast<uint16_t>(bw), w * 4U, (kUStride - w) / 8U, (512U - w) * 4U, 0});
                }
            }
        }
        {
            uint32_t nTiles = (wR1 - wR0 + ::kTileRowsPar - 1) / ::kTileRowsPar;
            ProcessSuper32DataCopyPadPhase10<LAZY, kCmStride, kUStride>(bw, dstReg, kb, l21, nTiles, srcReg, tw, w, wR0,
                                                                        wR1, wpad);
            if (nTiles >= 2U) {
                WaitFlag<HardEvent::MTE3_MTE2>(::kEvMte3Mte2);
            }
        }
        if constexpr (CUBE_LU && LAZY) {
            if (tw > 128U) {
                SuperPrepareL(kb, wR0, wR1, bw);
            }
        }
    }
    template <bool LAZY, auto kCmStride, auto kUStride>
    __aicore__ inline void ProcessSuper32SetFlagPhase11(uint32_t &bw, uint32_t &curReg, uint64_t &dstReg, uint32_t &kb,
                                                        uint32_t &l21, uint64_t &srcReg, uint32_t &tw, uint32_t &w,
                                                        uint32_t &wR0, uint32_t &wR1, uint32_t &wpad) {

        if (wR0 < wR1) {
            ProcessSuper32SetFlagPhase11SetFlagDepth35Stage1<LAZY, kCmStride, kUStride>(bw, dstReg, kb, l21, srcReg, tw,
                                                                                        w, wR0, wR1, wpad);
        }
        curReg = 1U - curReg;
    }
    template <bool LAZY, auto kCmStride, auto kUStride>
    __aicore__ inline void ProcessSuper32GetValuePhase12(uint64_t &base, uint32_t &bw, uint32_t &curReg, uint32_t &h,
                                                         uint32_t &kb, uint32_t &tw, uint32_t &wR0, uint32_t &wR1) {

        uint32_t w = tw;
        uint32_t wpad = (w + ::kAlign - 1) / ::kAlign * ::kAlign;
        uint32_t l21 = metaU_.GetValue(33);
        uint32_t needPos = PanelPositionChanged60(bw);
        uint64_t srcReg = base + static_cast<uint64_t>(curReg) * nnS_;
        if (l21 == 0U && needPos == 0U) {
        } else {
            uint64_t dstReg = base + static_cast<uint64_t>(1U - curReg) * nnS_;
            SharedScalarUpdateStage28(bw, h);
            ProcessSuper32SetFlagPhase11<LAZY, kCmStride, kUStride>(bw, curReg, dstReg, kb, l21, srcReg, tw, w, wR0,
                                                                    wR1, wpad);
        }
    }

    template <bool LAZY, auto kCmStride, class PhaseArg0>
    __aicore__ inline void ProcessSuper32vmulsPhase14(PhaseArg0 &alive, uint32_t &cnt, uint32_t &j0, uint32_t &nCh,
                                                      uint32_t &t, uint32_t &tA) {
        return ProcessLargeParvmulsPhase16<LAZY, kCmStride, PhaseArg0>(alive, cnt, j0, nCh, t, tA);
    }
    template <auto kCmStride>
    __aicore__ inline void ProcessSuper32GetValuePhase15(uint32_t &bw, uint32_t &js, uint32_t &p, uint32_t &t) {
        return ProcessLargeParGetValuePhase17<kCmStride>(bw, js, p, t);
    }
    template <auto kCmStride>
    __aicore__ inline void ProcessSuper32SetFlagPhase16(uint32_t &bw, uint32_t &cnt, int32_t &ex, uint32_t &p,
                                                        float &prod, uint32_t &t, uint32_t &tA) {
        return ProcessLargeParSetFlagPhase18<kCmStride>(bw, cnt, ex, p, prod, t, tA);
    }
    template <bool LAZY, auto evS2M3, auto kCmStride, auto kPanel, auto kUStride>
    __aicore__ inline void ProcessSuper32DataCopyPadPhase17(uint64_t &barBase, uint64_t &base, uint32_t &bw,
                                                            uint32_t &curReg, uint32_t &h, uint32_t &kb,
                                                            uint32_t &panelIdx, uint32_t &rowsPer, uint32_t &singular,
                                                            uint32_t &tw) {

        if (rank_ != 0U) {
            DataCopyPad(metaU_, wsGmU_[base + 2U * nnS_ + kCmStride * kPanel], {1, 192, 0, 0, 0},
                        DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
            WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
            if (metaU_.GetValue(33) != 0U) {
                SharedDataCopyPadStage39<LAZY, kCmStride>(base, bw, h);
            }
        }
        UpdateSchurRegion54<true, LAZY, kCmStride, kPanel, kUStride>(base, bw, curReg, h, kb, panelIdx, rowsPer,
                                                                     singular, tw);
        ProcessSuper32SetFlagPhase5<evS2M3>(barBase, panelIdx);
    }
    template <bool LAZY, auto kCmStride, class PhaseArg0, class PhaseArg5>
    __aicore__ inline void
    ProcessSuper32constexprPhase18(PhaseArg0 &alive, uint32_t &bw, uint32_t &cnt, int32_t &ex, uint32_t &h,
                                   PhaseArg5 &invPos, uint32_t &nCh, uint32_t &p, bool &panelUpdatePending, float &prod,
                                   LocalTensor<uint32_t> &rowOffsets, float &sign, uint32_t &t, uint32_t &tA) {

        if constexpr (LAZY) {
            uint32_t logicalP = invPos.GetValue(p) / 4U;
            uint32_t physicalT = posU_.GetValue(t) / 4U;
            metaU_.SetValue(1U + t, logicalP);
            UpdatePanelPivot54<true, kCmStride>(alive, bw, ex, h, invPos, logicalP, nCh, p, panelUpdatePending,
                                                physicalT, prod, rowOffsets, sign, t);
        } else {
            metaU_.SetValue(1 + t, p);
            if (p != t) {
                uint32_t js = 0;
                ProcessSuper32GetValuePhase15<kCmStride>(bw, js, p, t);
                Shared24ScalarUpdateStage22<kCmStride>(bw, js, p, sign, t);
            }
            ProcessSuper32SetFlagPhase16<kCmStride>(bw, cnt, ex, p, prod, t, tA);
        }
    }
    template <bool LAZY>
    __aicore__ inline void ProcessSuper32SetFlagPhase19(uint32_t &bw, uint32_t &h, uint32_t &sExact, uint32_t &t,
                                                        uint32_t &tA) {
        return ProcessLargeParSetFlagPhase21<LAZY>(bw, h, sExact, t, tA);
    }
    template <bool LAZY, auto evS2M3, auto kCmStride, class PhaseArg0, class PhaseArg5>
    __aicore__ inline void ProcessSuper32SetFlagPhase20(PhaseArg0 &alive, uint64_t &base, uint32_t &bw, int32_t &ex,
                                                        uint32_t &h, PhaseArg5 &invPos, uint32_t &l21nz, float &prod,
                                                        LocalTensor<uint32_t> &rowOffsets, float &sign,
                                                        uint32_t &singular) {

        bool panelUpdatePending = false;
        for (uint32_t t = 0; t < bw; ++t) {
            PanelStepGeometry61 geometry = GetPanelStepGeometry61<LAZY>(t, h);
            ProcessSuper32vmulsPhase14<LAZY, kCmStride>(alive, geometry.cnt, geometry.j0, geometry.nCh, t, geometry.tA);
            uint32_t sExact = geometry.segN;
            ProcessSuper32SetFlagPhase19<LAZY>(bw, h, sExact, t, geometry.tA);
            float maxv = 0.0f;
            uint32_t p = geometry.tA;
            ProcessSuper32LDET_RM_PAIR_OFFPhase6<kCmStride>(bw, h, l21nz, maxv, p, sExact, geometry.segN, t,
                                                            geometry.tA);
            if (RejectZeroPanelPivot64<LAZY>(maxv, panelUpdatePending, singular))
                break;
            ProcessSuper32constexprPhase18<LAZY, kCmStride>(alive, bw, geometry.cnt, ex, h, invPos, geometry.nCh, p,
                                                            panelUpdatePending, prod, rowOffsets, sign, t, geometry.tA);
        }
        if constexpr (LAZY) {
            LDET_MASK_SET(bw * kCmStride);
            LdetPanelMuls(panelCM_, -1.0f);
            LDET_MASK_RST();
        }
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
        ProcessSuper32SetFlagPhase1<LAZY, evS2M3, kCmStride>(base, bw, h, l21nz, singular);
    }
