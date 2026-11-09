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
    __aicore__ inline KernelLinalgDet() {}
    __aicore__ inline void SetCube(GM_ADDR neg, GM_ADDR u, GM_ADDR flags, uint32_t epoch) {
        this->cubeNeg_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(neg));
        this->cubeU_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(u));
        this->cubeFlags_ = flags;
        this->cubeEpoch_ = epoch;
    }
    __aicore__ inline void CubeControl(uint32_t mode) {
        if (rank_ == 0U) {
            auto p = reinterpret_cast<volatile __gm__ uint32_t *>(this->cubeFlags_ + 12U * 64U);
            p[1] = mode;
            p[2] = 0xC1493B27U;
            LuPublish(this->cubeFlags_ + 12U * 64U, this->cubeEpoch_ * 32U + 31U);
        }
    }
    __aicore__ inline void CubePrepareL(uint32_t kb, uint32_t first, uint32_t last) {
        uint32_t rows = last - first;
        for (uint32_t j = 0; j < 16U; ++j)
            Muls(cpbuf_[j * 32U], lbuf_[j * 256U + first - kb], -1.0f, rows);
        SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
        WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
        DataCopyPad(this->cubeNeg_[first - kb - 16U], cpbuf_,
                    DataCopyExtParams{16, rows * 4U, (32U - rows) / 8U, (256U - rows) * 4U, 0});
    }
    __aicore__ inline __inout_pipe__(V) void LdetU01Brcb(const LocalTensor<float> &dst, const LocalTensor<float> &src) {
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(dst.GetPhyAddr()),
              reinterpret_cast<__ubuf__ uint32_t *>(src.GetPhyAddr()), 1, 8, 2);
    }
    __aicore__ inline void SuperTriangularU01(uint32_t w, uint32_t base) {
        constexpr uint32_t pitch = kFactorPitch;
        constexpr uint32_t scratch = kScratch;
        static_assert(scratch + 576U <= 196608U, "Triangular U01 scratch exceeds UB");
        LocalTensor<float> coefficients(TPosition::VECCALC, scratch, 16U);
        LocalTensor<float> factors(TPosition::VECCALC, scratch + 64U, 128U);
        for (uint32_t j = 0; j < 15U; ++j) {
            uint32_t s = base + j;
            Muls(coefficients, lbuf_[s * pitch + base], -1.0f, 16U);
            LdetU01Brcb(factors, coefficients);
            for (uint32_t col = 0; col < w; col += 64U) {
                uint32_t count = w - col;
                if (count > 64U)
                    count = 64U;
                AscendCUtils::SetMask<float>(static_cast<int32_t>(count));
                LdetPanelMla(ubuf_[(s + 1U) * pitch + col], factors[(j + 1U) * 8U], ubuf_[s * pitch + col],
                             static_cast<uint8_t>(15U - j), static_cast<uint8_t>(pitch / 8U));
            }
        }
        PipeBarrier<PIPE_V>();
        LDET_MASK_RST();
    }
    __aicore__ inline void SuperBatchedU01(uint32_t w) {
        constexpr uint32_t pitch = kFactorPitch;
        constexpr uint32_t scratch = kScratch;
        static_assert(scratch + 576U <= 196608U, "Batched U01 scratch exceeds UB");
        LocalTensor<float> coefficients(TPosition::VECCALC, scratch, 16U);
        LocalTensor<float> factors(TPosition::VECCALC, scratch + 64U, 128U);
        SuperTriangularU01(w, 0U);
        for (uint32_t s = 0; s < 16U; ++s) {
            Muls(coefficients, lbuf_[s * kFactorPitch + 16U], -1.0f, 16U);
            LdetU01Brcb(factors, coefficients);
            for (uint32_t col = 0; col < w; col += 64U) {
                uint32_t count = w - col;
                if (count > 64U)
                    count = 64U;
                AscendCUtils::SetMask<float>(static_cast<int32_t>(count));
                LdetPanelMla(ubuf_[16U * pitch + col], factors, ubuf_[s * pitch + col], 16U,
                             static_cast<uint8_t>(pitch / 8U));
            }
        }
        PipeBarrier<PIPE_V>();
        LDET_MASK_RST();
        SuperTriangularU01(w, 16U);
    }
    __aicore__ inline void SuperPrepareL(uint32_t kb, uint32_t first, uint32_t last, uint32_t bw = 32U) {
        if constexpr (FORCE_FAST) {
            for (uint32_t r = first; r < last; r += 32U) {
                if (r != first) {
                    SetFlag<HardEvent::MTE3_V>(::kEvVMte3);
                    WaitFlag<HardEvent::MTE3_V>(::kEvVMte3);
                }
                uint32_t rows = last - r;
                if (rows > 32U)
                    rows = 32U;
                for (uint32_t j = 0; j < bw; ++j)
                    Muls(cpbuf_[j * 32U], lbuf_[j * kFactorPitch + r - kb], -1.0f, rows);
                SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
                WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
                DataCopyPad(this->cubeNeg_[r - kb - bw], cpbuf_,
                            DataCopyExtParams{static_cast<uint16_t>(bw), rows * 4U, (32U - rows) / 8U,
                                              (kFactorPitch - rows) * 4U, 0});
            }
        } else {
            uint32_t rows = last - first;
            for (uint32_t j = 0; j < 32U; ++j)
                Muls(cpbuf_[j * 32U], lbuf_[j * kFactorPitch + first - kb], -1.0f, rows);
            SetFlag<HardEvent::V_MTE3>(::kEvVMte3);
            WaitFlag<HardEvent::V_MTE3>(::kEvVMte3);
            DataCopyPad(this->cubeNeg_[first - kb - 32U], cpbuf_,
                        DataCopyExtParams{32, rows * 4U, (32U - rows) / 8U, (kFactorPitch - rows) * 4U, 0});
        }
    }
    __aicore__ inline void CubeStageEnd(uint32_t panelIdx, uint32_t singular, uint32_t curReg, uint32_t tw) {
        SetFlag<HardEvent::MTE3_S>(::kEvVMte2);
        WaitFlag<HardEvent::MTE3_S>(::kEvVMte2);
        if (rank_ == 0U) {
            auto p = reinterpret_cast<volatile __gm__ uint32_t *>(this->cubeFlags_);
            p[1] = singular;
            p[2] = curReg;
            p[3] = tw;
        }
        uint32_t token = this->cubeEpoch_ * 32U + panelIdx + 1U;
        LuPublish(this->cubeFlags_ + rank_ * 64U, token);
        LuPoll(this->cubeFlags_ + (8U + rank_ / 2U) * 64U, token);
    }
    __aicore__ inline void Shared24ScalarUpdateStage10() {
        tabI_ = LocalTensor<int32_t>(TPosition::VECCALC, ::kSpTab, ::kSmallMaxN);
        coloffI_ = LocalTensor<int32_t>(TPosition::VECCALC, ::kSpColoff, ::kSmallMaxN);
        coloffU_ = LocalTensor<uint32_t>(TPosition::VECCALC, ::kSpColoff, ::kSmallMaxN);
        tabU_ = LocalTensor<uint32_t>(TPosition::VECCALC, ::kSpTab, ::kSmallMaxN);
        tabRI_ = LocalTensor<int32_t>(TPosition::VECCALC, ::kSpTabR, 8 * ::kSmallMaxN);
        tabRU_ = LocalTensor<uint32_t>(TPosition::VECCALC, ::kSpTabR, 8 * ::kSmallMaxN);
        tmp_ = LocalTensor<float>(TPosition::VECCALC, ::kSpTmp, 256);
        res_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRes, ::kMaxPerCore);
    }
    template <class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ inline void InitDataCopyPhase1(PhaseArg0 &familyScratch, PhaseArg1 &prefetchFirst,
                                              PhaseArg2 &prefetchN64) {

        if constexpr (LD_BATCH == 9U || LD_BATCH == 10U) {
            constexpr uint32_t rowBias = 256U;
            static_assert(::kSmallMaxN - rowBias / 4U >= 40U, "Biased fixed row scratch too short");
            rowkbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRowk + rowBias, ::kSmallMaxN - rowBias / 4U);
        } else {
            if (familyScratch)
                rowkbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRowk + 256U, ::kSmallMaxN - 64U);
            else
                rowkbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRowk, ::kSmallMaxN);
        }
        Shared24ScalarUpdateStage10();
        if constexpr (LD_BATCH == 11U) {
            if (prefetchN64 && cnt_ == 1U) {
                // Exact n64/B4: contiguous payload staging in unused matrix reservation.
                DataCopy(mat_[8192U], aGm_[static_cast<uint64_t>(start_) * 4096U], 4096U);
                firstCopyPending_ = true;
            }
        }
        if constexpr (LdetRouteKeyIs<6U, 9U, 10U>(LD_BATCH)) {
            if (cnt_ == 1U && (prefetchFirst || n_ == 16U || n_ == 31U || n_ == 32U)) {
                DataCopyPad(
                    mat_, aGm_[static_cast<uint64_t>(start_) * nn_],
                    DataCopyExtParams{static_cast<uint16_t>(n_), n_ * 4U, 0U, (s_ - ((n_ + 7U) & ~7U)) / 8U, 0U},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
                firstCopyPending_ = true;
            }
        }
    }
    inline void InitSetFlagPhase2SetFlagDepth22Stage1() {

        for (uint32_t i = 0; i < ::kMaxN; ++i) {
            tabI_.SetValue(i, static_cast<int32_t>(i * (LD_BATCH == 2U ? 16U : kPanel) * sizeof(float)));
        }
        if constexpr (LD_BATCH == 2U) {
            LocalTensor<uint32_t> ids(TPosition::VECCALC, kLpPanel + 32768, ::kMaxN);
            for (uint32_t i = 0; i < n_; ++i)
                ids.SetValue(i, i * 4U);
            LocalTensor<uint32_t> rows(TPosition::VECCALC, kLpPanel + 34816, 128);
            for (uint32_t rem = 0; rem < 8U; ++rem) {
                for (uint32_t j = 0; j < 16U; ++j) {
                    rows.SetValue(rem * 16U + j, (j * (FAST_MC ? 256U : ::kMaxN) + rem) * 4U);
                }
            }
        }
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
    }
    __aicore__ inline void InitSetFlagPhase2(GM_ADDR &userWs) {

        tile_ = LocalTensor<float>(TPosition::VECCALC, kLpTile, 2 * ::kTileRows * ::kUbW);
        tmp_ = LocalTensor<float>(TPosition::VECCALC, kLpTmp, 256);
        cpbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpCp, 4 * ::kMaxN);
        res_ = LocalTensor<float>(TPosition::VECCALC, kLpRes, ::kMaxPerCore);
        metaU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpMeta, 32);
        barU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpCp, ::kBarElems);
        posU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpPanel, ::kMaxN);
        dstU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpPanel + ::kMaxN * sizeof(float), 64);
        srcU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpPanel + ::kMaxN * sizeof(float) + 256, 64);
        pollU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpPanel + 768 * sizeof(float), 256);
        wsGmU_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(userWs));
        colbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpUbuf, ::kMaxN);
        absbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpUbuf + ::kMaxN * sizeof(float), ::kMaxN);
        tabI_ = LocalTensor<int32_t>(TPosition::VECCALC, kLpTab, ::kMaxN);
        coloffI_ = LocalTensor<int32_t>(TPosition::VECCALC, kLpColoff, ::kMaxN);
        coloffU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpColoff, ::kMaxN);
        if (isLarge_ != 0U) {
            InitSetFlagPhase2SetFlagDepth22Stage1();
        }
    }
    template <class PhaseArg0> __aicore__ inline void InitPipeBarrierPhase3(PhaseArg0 &n32VectorInit) {

        LDET_MASK_SET(n_);
        for (uint32_t r = 0; r < 8U; ++r) {
            vadds(reinterpret_cast<__ubuf__ int32_t *>(tabRI_[r * ::kSmallMaxN].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tabI_.GetPhyAddr()), static_cast<int32_t>(r * sizeof(float)), 1,
                  1, 1, 8, 8);
        }
        LDET_MASK_RST();
        if (n32VectorInit) {
            PipeBarrier<PIPE_V>();
        } else {
            SetFlag<HardEvent::V_S>(::kEvVS);
            WaitFlag<HardEvent::V_S>(::kEvVS);
        }
    }
    template <class PhaseArg0> __aicore__ inline void InitGetBlockIdxPhase4(PhaseArg0 &startOverride) {

        start_ = (startOverride == 0xffffffffU) ? GetBlockIdx() * matPerCore_ : startOverride;
        uint32_t end = start_ + matPerCore_;
        if (end > batch_) {
            end = batch_;
        }
        cnt_ = (start_ < end) ? (end - start_) : 0;
    }
    __aicore__ inline void InitSetFlagPhase5() {

        for (uint32_t i = 0; i < n_; ++i) {
            tabI_.SetValue(i, static_cast<int32_t>(i * s_ * sizeof(float)));
        }
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        tabRI_ = LocalTensor<int32_t>(TPosition::VECCALC, ::kSpTabR, 8 * ::kSmallMaxN);
        tabRU_ = LocalTensor<uint32_t>(TPosition::VECCALC, ::kSpTabR, 8 * ::kSmallMaxN);
    }
    template <class PhaseArg0> __aicore__ inline void InitStatePhase6(PhaseArg0 &td) {

        n_ = td.n;
        s_ = td.stride;
        batch_ = td.batchCount;
        matPerCore_ = td.matPerCore;
        isLarge_ = td.isLarge;
        nn_ = static_cast<uint64_t>(n_) * n_;
        nnS_ = static_cast<uint64_t>(n_) * s_;
    }
    __aicore__ inline void InitGetUserWorkspacePhase7(GM_ADDR &workspace) {

        GM_ADDR userWs = GetUserWorkspace(workspace);
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs));
        panel_ = LocalTensor<float>(TPosition::VECCALC, kLpPanel, ::kMaxN * kPanel);
        lbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpLbuf, ::kMaxN * kPanel);
        panelCM_ = LocalTensor<float>(TPosition::VECCALC, kLpLbuf, ::kMaxN * kPanel);
        ubuf_ = LocalTensor<float>(TPosition::VECCALC, kLpUbuf, kPanel * ::kUbW);
        InitSetFlagPhase2(userWs);
    }
    __aicore__ inline void SharedSetFlagStage38() {
        constexpr uint32_t tinyN = LD_BATCH == 12U ? 7U : 8U;
#pragma unroll
        for (uint32_t r = 0; r < tinyN - 3U; ++r) {
#pragma unroll
            for (uint32_t i = 0; i < tinyN; ++i) {
                tabRI_.SetValue(r * ::kSmallMaxN + i, static_cast<int32_t>(i * 32U + r * 4U));
            }
        }
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR det, GM_ADDR workspace, const LinalgDetTilingData &td,
                                bool n32VectorInit = false, bool prefetchFirst = false, bool familyScratch = false,
                                bool singleColumnOffsets = false, bool prefetchN64 = false,
                                uint32_t startOverride = 0xffffffffU) {
        InitStatePhase6(td);
        if (isLarge_ == 2U) {
            groupSize_ = td.groupSize;
            rank_ = GetBlockIdx() % groupSize_;
            matIdx_ = GetBlockIdx() / groupSize_;
            wsSpan_ = 2U * nnS_ + ::kMaxN * kPanel + 32 + ::kBarElems;
            cnt_ = 1;
        } else {
            if (LD_BATCH == 9U && singleColumnOffsets && n_ == 16U && batch_ == 1U) {
                start_ = 0U;
                cnt_ = 1U;
            } else {
                InitGetBlockIdxPhase4(startOverride);
                if (cnt_ == 0U) {
                    return;
                }
            }
        }
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(a));
        detGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(det));
        if (isLarge_ != 0U) {
            InitGetUserWorkspacePhase7(workspace);
        } else {
            mat_ = LocalTensor<float>(TPosition::VECCALC, ::kSpMat, ::kSmallMaxN * ::kSmallMaxN);
            colbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpCol, ::kSmallMaxN);
            absbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpAbs, ::kSmallMaxN);
            InitDataCopyPhase1(familyScratch, prefetchFirst, prefetchN64);
            if constexpr (LdetRouteKeyIs<12U, 13U, 25U>(LD_BATCH)) {
                SharedSetFlagStage38();
            } else if (n_ > 8U && LD_BATCH != 8U) {
                InitSetFlagPhase5();
                if (singleColumnOffsets)
                    return;
                InitPipeBarrierPhase3(n32VectorInit);
            }
        }
    }
    inline void InitUserWorkspaceSetFlagPhase1SetFlagDepth22Stage1() {

        for (uint32_t i = 0; i < ::kMaxN; ++i) {
            tabI_.SetValue(i,
                           static_cast<int32_t>(i * (SUPER32 ? 32U : (LD_BATCH == 2U ? 16U : kPanel)) * sizeof(float)));
        }
        if constexpr (LD_BATCH == 2U) {
            LocalTensor<uint32_t> ids(TPosition::VECCALC, kIds, ::kMaxN);
            for (uint32_t i = 0; i < n_; ++i)
                ids.SetValue(i, i * 4U);
            LocalTensor<uint32_t> rows(TPosition::VECCALC, kRows, SUPER32 ? 256U : 128U);
            for (uint32_t rem = 0; rem < 8U; ++rem) {
                for (uint32_t j = 0; j < (SUPER32 ? 32U : 16U); ++j) {
                    rows.SetValue(rem * (SUPER32 ? 32U : 16U) + j, (j * (FAST_MC ? kFactorPitch : ::kMaxN) + rem) * 4U);
                }
            }
        }
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
    }
    __aicore__ inline void InitUserWorkspaceSetFlagPhase1(GM_ADDR &userWs) {

        cpbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpCp, 4 * ::kMaxN);
        res_ = LocalTensor<float>(TPosition::VECCALC, kLpRes, ::kMaxPerCore);
        metaU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpMeta, SUPER32 ? 48U : 32U);
        barU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kBar, ::kBarElems);
        posU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kPos, ::kMaxN);
        dstU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpPanel + ::kMaxN * sizeof(float), 64);
        srcU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpPanel + ::kMaxN * sizeof(float) + 256, 64);
        pollU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kPoll, 256);
        wsGmU_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(userWs));
        colbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpUbuf, ::kMaxN);
        absbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpUbuf + ::kMaxN * sizeof(float), ::kMaxN);
        tabI_ = LocalTensor<int32_t>(TPosition::VECCALC, kLpTab, ::kMaxN);
        coloffI_ = LocalTensor<int32_t>(TPosition::VECCALC, kLpColoff, ::kMaxN);
        coloffU_ = LocalTensor<uint32_t>(TPosition::VECCALC, kLpColoff, ::kMaxN);
        if (isLarge_ != 0U) {
            InitUserWorkspaceSetFlagPhase1SetFlagDepth22Stage1();
        }
    }
    __aicore__ inline void InitUserWorkspaceLocalTensorPhase2() {

        mat_ = LocalTensor<float>(TPosition::VECCALC, ::kSpMat, ::kSmallMaxN * ::kSmallMaxN);
        colbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpCol, ::kSmallMaxN);
        absbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpAbs, ::kSmallMaxN);
        if constexpr (LD_BATCH == 9U || LD_BATCH == 10U) {
            constexpr uint32_t rowBias = 256U;
            static_assert(::kSmallMaxN - rowBias / 4U >= 40U, "Biased fixed row scratch too short");
            rowkbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRowk + rowBias, ::kSmallMaxN - rowBias / 4U);
        } else {
            rowkbuf_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRowk, ::kSmallMaxN);
        }
        Shared24ScalarUpdateStage10();
    }
