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
    __aicore__ inline void InitSmallLUState(SmallLUState &state, uint32_t slot, uint32_t scratch) {
        constexpr uint32_t plane = ::kSmallMaxN;
        constexpr uint32_t words = 3U * plane + 256U + 8U * plane;
        state.matrix = reinterpret_cast<__ubuf__ float *>(mat_[slot * n_ * s_].GetPhyAddr());
        uint32_t base = ::kSpMat + (scratch + slot * words) * sizeof(float);
        state.column = reinterpret_cast<__ubuf__ float *>(base);
        state.absolute = reinterpret_cast<__ubuf__ float *>(base + plane * 4U);
        state.row = reinterpret_cast<__ubuf__ float *>(base + 2U * plane * 4U);
        state.reduction = reinterpret_cast<__ubuf__ float *>(base + 3U * plane * 4U);
        state.factors = reinterpret_cast<__ubuf__ float *>(base + (3U * plane + 256U) * 4U);
    }
    __aicore__ inline void CopySmallLUState(SmallLUState &state, uint32_t matrix) {
        LocalTensor<float> matrixTensor(TPosition::VECCALC,
                                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(state.matrix)), n_ * s_);
        for (uint32_t first = 0; first < n_; first += 32U) {
            uint32_t rows = n_ - first;
            if (rows > 32U)
                rows = 32U;
            DataCopyExtParams copy{static_cast<uint16_t>(rows), n_ * 4U, 0, (s_ - ((n_ + 7U) & ~7U)) / 8U, 0};
            DataCopyPad(matrixTensor[first * s_],
                        aGm_[static_cast<uint64_t>(matrix) * nn_ + static_cast<uint64_t>(first) * n_], copy,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    }
    template <uint32_t SLOT> __aicore__ inline void IssueSmallLUPivot(SmallLUState &state) {
        if (state.done)
            return;
        uint32_t count = n_ - state.k;
        uint32_t diagonal = state.k * (s_ + 1U);
        SmallMaskSet(count);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(state.column),
                reinterpret_cast<__ubuf__ uint32_t *>(tabRU_[(diagonal & 7U) * ::kSmallMaxN].GetPhyAddr()),
                reinterpret_cast<uint64_t>(state.matrix) + (diagonal & ~7U) * 4U, 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(reinterpret_cast<__ubuf__ float *>(state.absolute), reinterpret_cast<__ubuf__ float *>(state.column), 1, 1,
             1, 8, 8);
        LocalTensor<float> reduction(TPosition::VECCALC,
                                     static_cast<uint32_t>(reinterpret_cast<uint64_t>(state.reduction)), 256U);
        LocalTensor<float> absolute(TPosition::VECCALC,
                                    static_cast<uint32_t>(reinterpret_cast<uint64_t>(state.absolute)), ::kSmallMaxN);
        LdetVcmax128(reduction, absolute, count);
        SmallMaskReset();
        if (count > 1U) {
            ResetMask();
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.factors),
                  reinterpret_cast<__ubuf__ uint32_t *>(state.column), 1, 8, static_cast<uint8_t>((count + 7U) / 8U));
        }
        SetFlag<HardEvent::V_S>(2U + SLOT);
        state.pending = true;
    }
    template <class PhaseArg3>
    __aicore__ inline void FinishSmallLUPivotvmulsPhase1(uint32_t &alignedK, float &inverse, uint32_t &p,
                                                         PhaseArg3 &state) {

        SmallMaskSet(s_ - alignedK);
        __ubuf__ float *matrix = reinterpret_cast<__ubuf__ float *>(state.matrix);
        vmuls(reinterpret_cast<__ubuf__ float *>(state.row), matrix + p * s_ + alignedK, inverse, 1, 1, 1, 8, 8);
        if (p != state.k) {
            vmuls(matrix + p * s_ + alignedK, matrix + state.k * s_ + alignedK, 1.0f, 1, 1, 1, 8, 8);
        }
        SmallMaskReset();
        uint32_t count = n_ - state.k;
        if (count > 1U) {
            if (p != state.k) {
                AscendCUtils::SetMask<float>(static_cast<int32_t>(8));
                vmuls(reinterpret_cast<__ubuf__ float *>((state.factors + (p - state.k) * 8U)),
                      reinterpret_cast<__ubuf__ float *>(state.factors), 1.0f, 1, 1, 1, 8, 8);
            }
            PipeBarrier<PIPE_V>();
            const bool padded = (n_ % 8U != 0U);
            uint32_t first = (state.k + 1U) & ~7U;
            for (uint32_t col = first; col < n_; col += 64U) {
                uint32_t width = (padded ? s_ : n_) - col;
                if (width > 64U)
                    width = 64U;
                AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
                if constexpr (LD_BATCH == 9U || LD_BATCH == 10U) {
                    vmla(matrix + (state.k + 1U) * s_ + col, reinterpret_cast<__ubuf__ float *>((state.factors + 8)),
                         reinterpret_cast<__ubuf__ float *>((state.row + col - alignedK)),
                         static_cast<uint8_t>(count - 1U), 1, 0, 1, static_cast<uint8_t>(s_ / 8U), 1, 0);
                } else {
                    vmla(matrix + (state.k + 1U) * s_ + col,
                         reinterpret_cast<__ubuf__ float *>((state.row + col - alignedK)),
                         reinterpret_cast<__ubuf__ float *>((state.factors + 8)), static_cast<uint8_t>(count - 1U), 1,
                         1, 0, static_cast<uint8_t>(s_ / 8U), 0, 1);
                }
            }
        }
    }
    template <uint32_t SLOT> __aicore__ inline void FinishSmallLUPivot(SmallLUState &state) {
        if (!state.pending)
            return;
        WaitFlag<HardEvent::V_S>(2U + SLOT);
        state.pending = false;
        float maximum = state.reduction[0];
        if (!(maximum > 0.0f)) {
            state.singular = true;
            state.done = true;
            return;
        }
        float indexFloat = state.reduction[1];
        uint32_t p = state.k + *reinterpret_cast<uint32_t *>(&indexFloat);
        float pivot = state.column[p - state.k];
        if (p != state.k)
            state.sign = -state.sign;
        float inverse = -1.0f / pivot;
        uint32_t alignedK = state.k & ~7U;
        FinishSmallLUPivotvmulsPhase1(alignedK, inverse, p, state);
        SmallMaskReset();
        ProdSmallCommon(state.fast, state.product, state.exponent, pivot);
        ++state.k;
        if (state.k == n_)
            state.done = true;
    }
    template <uint32_t TN> __aicore__ inline void ProcessAlignedPairs2() {
        static_assert(TN == 16U || TN == 32U, "Aligned two-state scope");
        constexpr uint32_t stateWords = 3U * ::kSmallMaxN + 256U + 8U * ::kSmallMaxN;
        static_assert((2U * TN * (TN + 8U) + 2U * stateWords) * 4U <= ::kSpCol,
                      "Two-state workspace overlaps shared tables");
        const uint32_t scratch = (2U * n_ * s_ + 7U) & ~7U;
        SmallLUState a, b;
        InitSmallLUState(a, 0U, scratch);
        InitSmallLUState(b, 1U, scratch);
        if (firstCopyPending_)
            firstCopyPending_ = false;
        else
            CopySmallLUState(a, start_);
        if (cnt_ > 1U)
            CopySmallLUState(b, start_ + 1U);
        else
            b.done = true;
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        IssueSmallLUPivot<0U>(a);
        IssueSmallLUPivot<1U>(b);
#pragma nounroll
        for (uint32_t k = 0; k < TN; ++k) {
            FinishSmallLUPivot<0U>(a);
            IssueSmallLUPivot<0U>(a);
            FinishSmallLUPivot<1U>(b);
            IssueSmallLUPivot<1U>(b);
        }
        res_.SetValue(0, a.singular ? 0.0f : a.sign * ProdFinal(a.product, a.exponent));
        if (cnt_ > 1U)
            res_.SetValue(1, b.singular ? 0.0f : b.sign * ProdFinal(b.product, b.exponent));
    }
    template <class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ inline void ProcessSmallPairsSetFlagPhase1(PhaseArg0 &a, PhaseArg1 &b, PhaseArg2 &c, uint32_t &matrix) {

        CopySmallLUState(a, start_ + matrix);
        if (matrix + 1U < cnt_)
            CopySmallLUState(b, start_ + matrix + 1U);
        else
            b.done = true;
        if (matrix + 2U < cnt_)
            CopySmallLUState(c, start_ + matrix + 2U);
        else
            c.done = true;
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
        WaitFlag<HardEvent::MTE2_S>(::kEvMte2S);
        SetFlag<HardEvent::MTE2_V>(::kEvMte2V);
        WaitFlag<HardEvent::MTE2_V>(::kEvMte2V);
        IssueSmallLUPivot<0U>(a);
        IssueSmallLUPivot<1U>(b);
        IssueSmallLUPivot<2U>(c);
        while (!a.done || !b.done || !c.done) {
            FinishSmallLUPivot<0U>(a);
            if constexpr (PAIR_SCHEDULE == 0U)
                IssueSmallLUPivot<0U>(a);
            FinishSmallLUPivot<1U>(b);
            if constexpr (PAIR_SCHEDULE != 0U)
                IssueSmallLUPivot<0U>(a);
            IssueSmallLUPivot<1U>(b);
            FinishSmallLUPivot<2U>(c);
            IssueSmallLUPivot<2U>(c);
        }
        res_.SetValue(matrix, a.singular ? 0.0f : a.sign * ProdFinal(a.product, a.exponent));
        if (matrix + 1U < cnt_) {
            res_.SetValue(matrix + 1U, b.singular ? 0.0f : b.sign * ProdFinal(b.product, b.exponent));
        }
        if (matrix + 2U < cnt_) {
            res_.SetValue(matrix + 2U, c.singular ? 0.0f : c.sign * ProdFinal(c.product, c.exponent));
        }
    }
    __aicore__ inline void ProcessSmallPairs() {
        uint32_t scratch = (3U * n_ * s_ + 7U) & ~7U;
        for (uint32_t matrix = 0; matrix < cnt_; matrix += 3U) {
            if (matrix != 0U) {
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::S_MTE2>(::kEvSMte2);
                WaitFlag<HardEvent::S_MTE2>(::kEvSMte2);
                SetFlag<HardEvent::V_MTE2>(::kEvVMte2);
                WaitFlag<HardEvent::V_MTE2>(::kEvVMte2);
            }
            SmallLUState a;
            SmallLUState b;
            SmallLUState c;
            InitSmallLUState(a, 0U, scratch);
            InitSmallLUState(b, 1U, scratch);
            InitSmallLUState(c, 2U, scratch);
            ProcessSmallPairsSetFlagPhase1(a, b, c, matrix);
        }
    }
    __aicore__ inline void CopyPrefetchSlot(LocalTensor<float> dst, uint32_t matrix) {
        for (uint32_t r0 = 0; r0 < n_; r0 += 32U) {
            uint32_t rows = n_ - r0;
            if (rows > 32U)
                rows = 32U;
            DataCopyPad(dst[r0 * s_], aGm_[static_cast<uint64_t>(matrix) * nn_ + static_cast<uint64_t>(r0) * n_],
                        {static_cast<uint16_t>(rows), n_ * 4U, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
    }
    __aicore__ inline void PrefetchPivotReady() {
        SetFlag<HardEvent::V_S>(::kEvVS);
        WaitFlag<HardEvent::V_S>(::kEvVS);
    }
    __aicore__ inline void PrefetchReuseReady() {
        SetFlag<HardEvent::V_MTE2>(::kEvVMte2);
        WaitFlag<HardEvent::V_MTE2>(::kEvVMte2);
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask>
    __aicore__ inline void ProcessSmallPrefetchvmulsPhase1(int32_t &ex, bool &fastProd, uint32_t &k, uint32_t &kA,
                                                           uint32_t &p, float &piv, float &prod, float &sign,
                                                           uint32_t &wA) {
        return ProcessSmallvmulsPhase2<batchedUpdate, earlyBroadcast, normalMask>(ex, fastProd, k, kA, p, piv, prod,
                                                                                  sign, wA);
    }
    template <auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallPrefetchvmulsPhase2(uint32_t &c0, uint32_t &cnt, LocalTensor<float> &factors,
                                                           uint32_t &k, uint32_t &kA, uint32_t &p) {
        return ProcessSmallvmulsPhase3<earlyBroadcast, normalMask, paddedUpdate>(c0, cnt, factors, k, kA, p);
    }

    __aicore__ inline void ProcessSmallPrefetchvaxpyPhase3(uint32_t &c0, uint32_t &i, uint32_t &k, uint32_t &kA,
                                                           uint32_t &w) {
        return ProcessSmallvaxpyPhase6(c0, i, k, kA, w);
    }
    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallPrefetchconstexprPhase4(uint32_t &cnt, int32_t &ex, LocalTensor<float> &factors,
                                                               bool &fastProd, uint32_t &k, uint32_t &kA, uint32_t &p,
                                                               float &prod, float &sign) {

        uint32_t wA = s_ - kA;
        float piv;
        ProcessSmallPrefetchvmulsPhase1<batchedUpdate, earlyBroadcast, normalMask>(ex, fastProd, k, kA, p, piv, prod,
                                                                                   sign, wA);
        uint32_t c0 = (k + 1) & ~(::kAlign - 1);
        uint32_t w = n_ - c0;
        uint32_t i = k + 1;
        if constexpr (batchedUpdate) {
            ProcessSmallPrefetchvmulsPhase2<earlyBroadcast, normalMask, paddedUpdate>(c0, cnt, factors, k, kA, p);
        } else {
            ProcessSmallPrefetchvaxpyPhase3(c0, i, k, kA, w);
        }
        CompleteEarlyProduct64<earlyBroadcast, normalMask>(fastProd, prod, ex, piv);
    }

    template <auto batchedUpdate, auto earlyBroadcast, auto normalMask, auto paddedUpdate>
    __aicore__ inline void ProcessSmallPrefetchvgatherPhase7(int32_t &ex, bool &fastProd, float &prod, float &sign,
                                                             uint32_t &singular) {

        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t kA = k & ~(::kAlign - 1);
            uint32_t cnt = n_ - k;
            Shared24VgatherStage7(cnt, k);
            uint32_t segN = normalMask ? 1U : (cnt + 63U) / 64U;
            ProcessSmallRawUBconstexprPhase15<normalMask>(cnt);
            LocalTensor<float> factors(TPosition::VECCALC, ::kSpBrc, ::kSmallMaxN * 8);
            if constexpr (earlyBroadcast) {
                if (cnt > 1U) {
                    ResetMask();
                    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(factors.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t *>(colbuf_.GetPhyAddr()), 1, 8,
                          static_cast<uint8_t>((cnt + 7U) / 8U));
                }
            }
            PrefetchPivotReady();
            float maxv = 0.0f;
            uint32_t p = k;
            ProcessTinyVectorLUconstexprPhase5<normalMask>(k, maxv, p, segN);
            if (normalMask ? !(maxv > 0.0f) : maxv == 0.0f) {
                singular = 1;
                break;
            }
            ProcessSmallPrefetchconstexprPhase4<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                cnt, ex, factors, fastProd, k, kA, p, prod, sign);
        }
    }
    __aicore__ inline void ProcessSmallPrefetch() {
        constexpr bool normalMask = true;
        constexpr bool batchedUpdate = true;
        constexpr bool paddedUpdate = true;
        constexpr bool earlyBroadcast = batchedUpdate;
        const uint32_t slotElems = n_ * s_;
        LocalTensor<float> slots = mat_;
        StartPrefetchSlots54(slots);
        for (uint32_t m = 0; m < cnt_; ++m) {
            BindPrefetchSlot54(slots, m, slotElems);
            PrefetchFollowingSlot64(slots, m, slotElems);
            float sign = 1.0f;
            float prod = 1.0f;
            int32_t ex = 0;
            bool fastProd = true;
            uint32_t singular = 0;
            ProcessSmallPrefetchvgatherPhase7<batchedUpdate, earlyBroadcast, normalMask, paddedUpdate>(
                ex, fastProd, prod, sign, singular);
            if (m + 2U < cnt_)
                PrefetchReuseReady();
            float detVal = (singular != 0U) ? 0.0f : sign * ProdFinal(prod, ex);
            res_.SetValue(m, detVal);
        }
        mat_ = slots;
    }
    template <uint32_t TN> __aicore__ inline void SharedDataCopyPadStage8(uint64_t &g) {
        if constexpr (LD_BATCH == 6U) {
            for (uint32_t r0 = 0; r0 < TN; r0 += 32U) {
                uint32_t rows = TN - r0;
                if (rows > 32U)
                    rows = 32U;
                DataCopyExtParams cp{static_cast<uint16_t>(rows), static_cast<uint32_t>(TN * sizeof(float)), 0, 0, 0};
                DataCopyPad(mat_[r0 * (TN + 8U)],
                            aGm_[g * (static_cast<uint64_t>(TN) * TN) + static_cast<uint64_t>(r0) * TN], cp,
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        } else {
            for (uint32_t r0 = 0; r0 < TN; r0 += 32U) {
                uint32_t rows = TN - r0;
                if (rows > 32U)
                    rows = 32U;
                DataCopyPad(mat_[r0 * (TN + 8U)], aGm_[g * (static_cast<uint64_t>(TN) * TN) + r0 * TN],
                            {static_cast<uint16_t>(rows), TN * 4U, 0, 1, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        }
    }
    template <uint32_t TN>
    __aicore__ inline void ProcessSmallFixedN64CachedDataCopyPhase1(bool &firstCopyAlreadyIssued, uint64_t &g,
                                                                    uint32_t &nn32, bool &tiny) {

        if (firstCopyAlreadyIssued) {
            firstCopyPending_ = false;
        } else {
            CopyFixedMatrix56<TN>(g, nn32, tiny);
        }
        SetFlag<HardEvent::MTE2_S>(::kEvMte2S);
    }
