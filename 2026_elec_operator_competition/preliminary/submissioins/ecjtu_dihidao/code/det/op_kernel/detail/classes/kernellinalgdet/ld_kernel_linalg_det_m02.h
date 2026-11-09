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
    __aicore__ inline void InitUserWorkspaceSetFlagPhase3() {

        for (uint32_t i = 0; i < n_; ++i) {
            tabI_.SetValue(i, static_cast<int32_t>(i * s_ * sizeof(float)));
        }
        SetFlag<HardEvent::S_V>(::kEvSV);
        WaitFlag<HardEvent::S_V>(::kEvSV);
        tabRI_ = LocalTensor<int32_t>(TPosition::VECCALC, ::kSpTabR, 8 * ::kSmallMaxN);
        tabRU_ = LocalTensor<uint32_t>(TPosition::VECCALC, ::kSpTabR, 8 * ::kSmallMaxN);
        LDET_MASK_SET(n_);
        for (uint32_t r = 0; r < 8U; ++r) {
            vadds(reinterpret_cast<__ubuf__ int32_t *>(tabRI_[r * ::kSmallMaxN].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tabI_.GetPhyAddr()), static_cast<int32_t>(r * sizeof(float)), 1,
                  1, 1, 8, 8);
        }
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }
    __aicore__ inline void InitUserWorkspaceconstexprPhase4(GM_ADDR &workspace) {

        GM_ADDR userWs;
        if constexpr (CUBE_LU)
            userWs = workspace;
        else
            userWs = GetUserWorkspace(workspace);
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs));
        panel_ = LocalTensor<float>(TPosition::VECCALC, kLpPanel, ::kMaxN * kPanel);
        lbuf_ = LocalTensor<float>(TPosition::VECCALC, kLpLbuf, ::kMaxN * kPanel);
        panelCM_ = LocalTensor<float>(TPosition::VECCALC, kLpLbuf, ::kMaxN * kPanel);
        ubuf_ = LocalTensor<float>(TPosition::VECCALC, kLpUbuf, kPanel * ::kUbW);
        tile_ = LocalTensor<float>(TPosition::VECCALC, kLpTile, 2 * ::kTileRows * ::kUbW);
        tmp_ = LocalTensor<float>(TPosition::VECCALC, kLpTmp, 256);
        InitUserWorkspaceSetFlagPhase1(userWs);
    }
    __aicore__ inline void InitUserWorkspace(GM_ADDR a, GM_ADDR det, GM_ADDR workspace, const LinalgDetTilingData &td,
                                             uint32_t logicalBlock = GetBlockIdx()) {
        n_ = td.n;
        s_ = td.stride;
        batch_ = td.batchCount;
        matPerCore_ = td.matPerCore;
        isLarge_ = td.isLarge;
        nn_ = static_cast<uint64_t>(n_) * n_;
        nnS_ = static_cast<uint64_t>(n_) * s_;
        if (isLarge_ == 2U) {
            groupSize_ = td.groupSize;
            rank_ = logicalBlock % groupSize_;
            matIdx_ = logicalBlock / groupSize_;
            wsSpan_ = 2U * nnS_ + ::kMaxN * kPanel + (WIDE512 ? 48U : 32U) + ::kBarElems;
            cnt_ = 1;
        } else {
            start_ = logicalBlock * matPerCore_;
            uint32_t end = start_ + matPerCore_;
            if (end > batch_) {
                end = batch_;
            }
            cnt_ = (start_ < end) ? (end - start_) : 0;
            if (cnt_ == 0U) {
                return;
            }
        }
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(a));
        detGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(det));
        if (isLarge_ != 0U) {
            InitUserWorkspaceconstexprPhase4(workspace);
        } else {
            InitUserWorkspaceLocalTensorPhase2();
            if constexpr (LdetRouteKeyIs<12U, 13U, 25U>(LD_BATCH)) {
                SharedSetFlagStage38();
            } else if (n_ > 8U && LD_BATCH != 8U) {
                InitUserWorkspaceSetFlagPhase3();
            }
        }
    }

    __aicore__ inline void ProcessGetValuePhase1(uint32_t &nonzero, float &offMax) {

        for (uint32_t i = 0; i < 8U; ++i) {
            float v = tmp_.GetValue(i);
            if (v != 0.0f)
                ++nonzero;
            v = v < 0.0f ? -v : v;
            if (v > offMax)
                offMax = v;
        }
        for (uint32_t i = 1; i < 8U; ++i) {
            float v = tmp_.GetValue(8U + i);
            v = v < 0.0f ? -v : v;
            if (v > offMax)
                offMax = v;
        }
    }
    __aicore__ inline void ProcessconstexprPhase2() {

        if constexpr (LD_BATCH == 15U) {
            if (n_ == 192U)
                ProcessMediumRawFixed<192U>();
            else if ((n_ == 200U && batch_ >= 5U) || n_ == 208U)
                ProcessSmallRawUB<true>();
            else
                ProcessSmallRawUB();
        } else if constexpr (LD_BATCH == 5U || LD_BATCH == 6U)
            ProcessSmallRawUB();
        else
            ProcessSmall();
    }
    __aicore__ inline void ProcessconstexprPhase3() {

        if constexpr (LD_BATCH == 11U)
            ProcessSmallPairs();
        else {
            if (cnt_ == 1U)
                ProcessSmallPrefetch();
            else
                ProcessSmallPairs();
        }
    }
    __aicore__ inline void ProcessconstexprPhase4() {

        if constexpr (CUBE_LU)
            CubeControl(1U);
        if constexpr (SUPER32)
            ProcessSuper32<true>();
        else
            ProcessLargePar<true>();
    }
    __aicore__ inline void ProcessProcessLargePhase5() {

        if (isLarge_ != 0U) {
            ProcessLarge();
        } else {
            ProcessSmall();
        }
    }
    __aicore__ inline void ProcessDataCopyPadPhase6() {

        uint64_t probe = static_cast<uint64_t>(matIdx_) * nn_ + static_cast<uint64_t>(n_ - 1U) * n_;
        DataCopyPad(tmp_, aGm_[probe], {1, 32, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(tmp_[8], aGm_[static_cast<uint64_t>(matIdx_) * nn_], {1, 32, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
    }
    __aicore__ inline bool ProcessconstexprPhase7constexprDepth22Stage1() {

        if constexpr (CUBE_LU && SUPER32 && FORCE_FAST) {
            CubeControl(1U);
            ProcessSuper32<true>();
        } else {
            ProcessDataCopyPadPhase6();
            uint32_t nonzero = 0;
            float offMax = 0.0f;
            ProcessGetValuePhase1(nonzero, offMax);
            float diag = tmp_.GetValue(8);
            diag = diag < 0.0f ? -diag : diag;
            bool dense = nonzero > 2U && diag <= 8.0f * offMax;
            if (dense) {
                ProcessconstexprPhase4();
            } else {
                if constexpr (CUBE_LU)
                    CubeControl(0U);
                uint32_t fallbackGroup = (n_ + 31U) / 32U;
                if (groupSize_ > fallbackGroup) {
                    if (rank_ >= fallbackGroup)
                        return true;
                    groupSize_ = fallbackGroup;
                }
                ProcessLargePar<false>();
            }
        }
        return false;
    }
    __aicore__ inline void ProcessconstexprPhase7() {

        if constexpr (LD_BATCH == 2U) {
            if (ProcessconstexprPhase7constexprDepth22Stage1())
                return;
        } else {
            ProcessLargePar<false>();
        }
        return;
    }
    __aicore__ inline void ProcessProcessAlignedPairs2Depth22Stage1() {

        if (matPerCore_ == 2U)
            ProcessAlignedPairs2<16>();
        else
            ProcessSmallFixed<16>();
    }
    __aicore__ inline void ProcessProcessAlignedPairs2Depth22Stage2() {

        if (matPerCore_ == 2U)
            ProcessAlignedPairs2<32>();
        else
            ProcessSmallFixed<32>();
    }
    __aicore__ inline bool ProcessStateDepth22Stage3() {

        if (cnt_ == 0U) {
            return true;
        }
        return false;
    }
    __aicore__ inline void Process() {
        if (ProcessStateDepth22Stage3())
            return;
        if constexpr (DUAL_LU) {
            ProcessconstexprPhase3();
        } else if constexpr (LD_BATCH == 14U) {
            ProcessSmallPrefetchRawUB();
        } else if constexpr (LD_BATCH == 12U) {
            ProcessTinyVectorLU<7>();
        } else if constexpr (LD_BATCH == 13U) {
            ProcessTinyVectorLU<8>();
        } else if constexpr (LD_BATCH == 25U) {
            ProcessTinyGuardedLU<8, true>();
        } else if constexpr (LD_BATCH == 9U) {
            ProcessProcessAlignedPairs2Depth22Stage1();
        } else if constexpr (LD_BATCH == 10U) {
            ProcessProcessAlignedPairs2Depth22Stage2();
        } else if constexpr (LD_BATCH == 11U) {
            ProcessSmallFixedN64Cached<64>();
        } else if constexpr (LD_BATCH == 8U) {
            ProcessTinyFixed<8>();
        } else if constexpr (LD_BATCH == 7U) {
            ProcessTinyDirect();
        } else if constexpr (LdetRouteKeyIs<1U, 3U, 4U, 5U, 6U, 15U>(LD_BATCH)) {
            ProcessconstexprPhase2();
        } else {
            if (isLarge_ == 2U) {
                return ProcessconstexprPhase7();
            }
            ProcessProcessLargePhase5();
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, cnt_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
    }
    __aicore__ inline void ProcessCapacity64Prepared(uint32_t inputN) {
        if (cnt_ == 0U)
            return;
        firstCopyPending_ = true;
        ProcessSmallFixedN64Cached<64U, true, true, true, true, true>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, 4U, 0, 0, 0});
    }
    __aicore__ inline void ProcessN64OriginalWrapper() {
        if (cnt_ == 0U)
            return;
        ProcessSmallFixedN64Cached<64U, true, true, true, true>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, cnt_ * 4U, 0, 0, 0});
    }
    __aicore__ inline void ProcessB1FullRow() {
        if (cnt_ == 0U)
            return;
        ProcessSmallFixed<32U, true, true, true, true, true>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, cnt_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
    }
    template <uint32_t TN, bool DIRECT_GATHER = false> __aicore__ inline void ProcessB1Combined() {
        if (cnt_ == 0U)
            return;
        ProcessSmallFixed<TN, TN == 32U, TN == 32U, true, true, true, DIRECT_GATHER>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, cnt_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
    }
    __aicore__ inline void InitN15FastPairOnly(GM_ADDR input, GM_ADDR output) {
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(input));
        detGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        mat_ = LocalTensor<float>(TPosition::VECCALC, 0U, 600U);
        res_ = LocalTensor<float>(TPosition::VECCALC, 12288U, 8U);
        DataCopyPad(mat_, aGm_[0], {15U, 60U, 0U, 3U, 0U}, DataCopyPadExtParams<float>{true, 0U, 1U, 0.0f});
    }

    __aicore__ inline void ProcessN15FastPairEntrySetFlagPhase1(__ubuf__ float *&broad, __ubuf__ float *&mat,
                                                                __ubuf__ uint32_t *&mp, float &product,
                                                                __ubuf__ float *&rp, __ubuf__ float *&up) {

#pragma unroll
        for (uint32_t phase = 0U; phase < 6U; ++phase) {
            constexpr uint32_t stride = 40U;
            const uint32_t k = phase * 2U, p = k * (stride + 1U);
            const uint64_t diagonal = reinterpret_cast<uint64_t>(mat) + k * 164U;
            SetFlag<HardEvent::V_S>(2);
            set_vector_mask(0ULL, 0xffffffffULL);
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(rp), mp + phase * 32U, diagonal, 8U, 1U);
            WaitFlag<HardEvent::V_S>(2);
            float a = mat_.GetValue(p), b = mat_.GetValue(p + 1U);
            float c = mat_.GetValue(p + stride), d = mat_.GetValue(p + stride + 1U);
            float delta = a * d - b * c, inverse = 1.0f / delta;
            product *= delta;
            PipeBarrier<PIPE_V>();
            set_vector_mask(0ULL, 0xffffffffULL);
            vmuls(rp, rp, inverse, 1U, 1U, 1U, 8U, 8U);
            auto qp = mat + k * stride;
            set_vector_mask(0ULL, 0xffffULL);
            vmuls(up, qp, -d, 1U, 1U, 1U, 8U, 8U);
            vmuls(up + 16U, qp, c, 1U, 1U, 1U, 8U, 8U);
            PipeBarrier<PIPE_V>();
            ResetMask();
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(broad), reinterpret_cast<__ubuf__ uint32_t *>(rp), 1U, 8U, 4U);
            set_vector_mask(0ULL, 0xffffULL);
            vaxpy(up, qp + stride, b, 1U, 1U, 1U, 8U, 8U);
            vaxpy(up + 16U, qp + stride, -a, 1U, 1U, 1U, 8U, 8U);
            PipeBarrier<PIPE_V>();
            auto dst = mat + (k + 2U) * stride;
            vmla(dst, broad, up, static_cast<uint8_t>(13U - k), 1U, 0U, 1U, 5U, 1U, 0U);
            PipeBarrier<PIPE_V>();
            vmla(dst, broad + 128U, up + 16U, static_cast<uint8_t>(13U - k), 1U, 0U, 1U, 5U, 1U, 0U);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ProcessN15FastPairEntry(GM_ADDR tiling) {
        LocalTensor<uint32_t> maps(TPosition::VECCALC, 4096U, 232U);
        GlobalTensor<uint32_t> tables;
        tables.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling + 576U), 232U);
        DataCopy(maps, tables, 232U);
        auto mp = reinterpret_cast<__ubuf__ uint32_t *>(maps.GetPhyAddr());
        auto mat = reinterpret_cast<__ubuf__ float *>(mat_.GetPhyAddr());
        auto up = reinterpret_cast<__ubuf__ float *>(9152U);
        auto rp = reinterpret_cast<__ubuf__ float *>(9408U);
        auto broad = reinterpret_cast<__ubuf__ float *>(9664U);
        SetMaskNorm();
        float product = 1.0f;
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        ProcessN15FastPairEntrySetFlagPhase1(broad, mat, mp, product, rp, up);
        auto lhs = reinterpret_cast<__ubuf__ float *>(7808U);
        auto terms = reinterpret_cast<__ubuf__ float *>(8832U);
        auto ds = reinterpret_cast<__ubuf__ float *>(12224U);
        auto result = reinterpret_cast<__ubuf__ float *>(res_.GetPhyAddr());
        set_vector_mask(0ULL, 0xffffffffffULL);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(lhs), mp + 192U, reinterpret_cast<uint64_t>(mat) + 1968U, 8U, 1U);
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 0x707ULL);
        vmul(terms, lhs, lhs + 16U, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 0x7ULL);
        vsub(terms, terms, terms + 8U, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        vmul(terms, terms, lhs + 32U, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        vcadd(ds, terms, 1U, 1U, 1U, 8U, 0);
        PipeBarrier<PIPE_V>();
        set_vector_mask(0ULL, 1ULL);
        vmuls(result, ds, product, 1U, 1U, 1U, 8U, 8U);
        ResetMask();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[0], res_, {1U, 4U, 0U, 0U, 0U});
    }
    __aicore__ inline void InitN16BlockOnly(GM_ADDR input, GM_ADDR output) {
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_A *>(input));
        detGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        mat_ = LocalTensor<float>(TPosition::VECCALC, ::kSpMat, 640U);
        res_ = LocalTensor<float>(TPosition::VECCALC, ::kSpRes, 8U);
        DataCopyPad(mat_, aGm_[0], {16U, 64U, 0U, 3U, 0U}, DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
    __aicore__ inline void ProcessN16PlaneEntry(GM_ADDR tiling) {
        N16BlockFour(tiling);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[0], res_, {1, 4U, 0, 0, 0});
    }
