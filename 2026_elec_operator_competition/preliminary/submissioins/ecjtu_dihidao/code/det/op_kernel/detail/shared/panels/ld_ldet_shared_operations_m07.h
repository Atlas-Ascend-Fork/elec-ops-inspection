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
// Included in the LdetSharedOperations class; preserves its member scope and access.
    template <auto T, auto TS, class PhaseArg2, class PhaseArg6, class PhaseArg7, class PhaseArg8, class PhaseArg9,
              class PhaseArg10>
    __aicore__ static inline void FinishLocalTailBlock2Shared57PipeBarrierPhase2(
        __ubuf__ float *&ap, __ubuf__ float *&b0, PhaseArg2 &b1, __ubuf__ float *&cp, __ubuf__ uint32_t *&ip,
        __ubuf__ float *&mp, PhaseArg6 &r0, PhaseArg7 &r1, PhaseArg8 &state, PhaseArg9 &t0, PhaseArg10 &t1) {

        state.VS();
        for (uint32_t f = 0U; f < T; f += 2U) {
            if (f != 0U)
                WaitFlag<HardEvent::V_S>(4);
            const uint32_t h2 = T - f - 2U;
            float a = mp[f * TS + f], b = mp[f * TS + f + 1U], c = mp[(f + 1U) * TS + f],
                  d = mp[(f + 1U) * TS + f + 1U];
            float ad = a * d, bc = b * c, delta = ad - bc;
            float da = delta < 0.0f ? -delta : delta, pa = ad < 0.0f ? -ad : ad, pb = bc < 0.0f ? -bc : bc;
            const uint32_t exponent = (state.word(delta) >> 23U) & 255U;
            if (exponent == 0U || exponent == 255U || !(da >= 0.25f * (pa + pb))) {
                state.singular = 3U;
                break;
            }
            float inverse = 1.0f / delta;
            float m00 = -d * inverse, m01 = b * inverse, m10 = c * inverse, m11 = -a * inverse;
            if (((state.word(m00) >> 23U) & 255U) == 255U || ((state.word(m01) >> 23U) & 255U) == 255U ||
                ((state.word(m10) >> 23U) & 255U) == 255U || ((state.word(m11) >> 23U) & 255U) == 255U) {
                state.singular = 3U;
                break;
            }
            if (!h2) {
                state.acc(delta);
                break;
            }
            const uint32_t address0 = (f + 2U) * TS + f, address1 = address0 + 1U;
            LDET_MASK_SET(h2);
            FinishLocalTailBlock2Shared57vgatherPhase1<T, TS>(address0, address1, ap, b0, b1, cp, f, h2, ip, m00, m01,
                                                              m10, m11, mp, r0, r1, t0, t1);
            ResetMask();
            SetMaskNorm();
            state.acc(delta);
            PipeBarrier<PIPE_V>();
        }
    }
    template <class Self> __aicore__ static inline void FinishLocalTailBlock2Shared57(Self &state) {
        constexpr uint32_t T = Self::NMAX - Self::LdetPolicy::LOCAL_CUT, TS = T + 8U;
        static_assert(Self::PANEL + (T + 8U) * TS * sizeof(float) <= Self::BRC,
                      "retired PANEL plus FAC tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, Self::PANEL, (T + 8U) * TS);
        DataCopy(tm, state.mat[Self::LdetPolicy::LOCAL_CUT * Self::S],
                 {static_cast<uint16_t>(T), T / 8U, (Self::S - T) / 8U, (TS - T) / 8U});
        PipeBarrier<PIPE_V>();
        PrepareTailRowIndices46<TS, T - 32U>(state);
        LDET_MASK_SET(T);
        for (uint32_t j = 0; j < 8U; ++j)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[j * T].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1,
                  8, 8);
        LDET_MASK_RST();
        PipeBarrier<PIPE_V>();
        auto views = GetTailRawViews51(state, tm);
        auto r0 = views.rp, r1 = views.tp, t0 = views.tp + 64U, t1 = views.tp + 128U;
        auto b0 = reinterpret_cast<__ubuf__ float *>(state.brc.GetPhyAddr()), b1 = b0 + T * 8U;
        FinishLocalTailBlock2Shared57PipeBarrierPhase2<T, TS>(views.ap, b0, b1, views.cp, views.ip, views.mp, r0, r1,
                                                              state, t0, t1);
        state.VS();
        state.meta.SetValue(0, state.singular ? 0.0f : state.sign * state.final());
        state.SM();
        DataCopyPad(state.out, state.meta, DataCopyExtParams{1, 4, 0, 0, 0});
        state.MS();
    }
    template <class Self> __aicore__ static inline void FinishLocalTailShared58(Self &state) {
        if constexpr (Self::LdetPolicy::DIM == 200U && Self::LdetPolicy::GROUP == 6U &&
                      Self::LdetPolicy::LOCAL_CUT == 160U && Self::LdetPolicy::SPLIT_OWNERS) {
            if (state.singular) {
                state.FinishLocalTailGepp();
                return;
            }
            const float savedSign = state.sign, savedProd = state.prod;
            const int32_t savedEx = state.ex;
            state.FinishLocalTailBlock2();
            if (state.singular == 3U) {
                state.sign = savedSign;
                state.prod = savedProd;
                state.ex = savedEx;
                state.singular = 0U;
                state.FinishLocalTailGepp();
            }
        } else
            state.FinishLocalTailGepp();
    }
    template <class Self> __aicore__ static inline void waitReadyShared59(Self &state, uint32_t q, uint32_t value) {
        GM_ADDR flag = state.prefixFlag + q * 64U;
        uint32_t spins = 0U;
        do {
            LuFlushFlag(flag);
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            if (*reinterpret_cast<volatile __gm__ uint32_t *>(flag) >= value)
                break;
            if (++spins > 8U)
                for (volatile uint32_t delay = 0U; delay < 64U; ++delay) {
                }
        } while (true);
    }
    template <class Self> __aicore__ static inline void publishShared60(Self &state, uint32_t kb) {
        state.state();
        state.SM();
        DataCopy(state.wu[Self::B * Self::NMAX + Self::META_WORDS], state.mu, Self::META_WORDS);
        state.VM();
        DataCopy(state.ws[16U * state.facPitch(kb)], state.fac[16U * state.facPitch(kb)], 16U * state.facPitch(kb));
        state.MS();
        LuPublish(state.prefixFlag + (kb / Self::B) * 64U, 0x6A1F0020U);
    }
    template <class Self, uint32_t FIRST, uint32_t LAST>
    __aicore__ static inline void receiveRangeShared61(Self &state, uint32_t kb) {
        static_assert(FIRST < LAST && LAST <= Self::B, "FAC range");
        DataCopy(state.mu, state.wu[Self::B * Self::NMAX + (FIRST == 0U ? 0U : Self::META_WORDS)], Self::META_WORDS);
        Shared24DataCopyStage16<FIRST, LAST>(kb, state);
    }

    template <class Self> __aicore__ static inline float pairAbsShared64(Self &state, float x) {
        return x < 0.0f ? -x : x;
    }
    template <class Self> __aicore__ static inline bool headPivotShared65(Self &state) {
        state.panel = state.mat;
        state.issuePreparedPivot(0U, 0U);
        WaitFlag<HardEvent::V_S>(2);
        auto tp = reinterpret_cast<__ubuf__ float *>(state.tmp.GetPhyAddr());
        float maximum = 0.0f;
        uint32_t pivot = 0U;
        for (uint32_t j = 0U; j < 4U; ++j) {
            float v = tp[2U * j];
            if (v > maximum) {
                maximum = v;
                pivot = j * 64U + state.word(tp[2U * j + 1U]);
            }
        }
        state.mu.SetValue(4U, pivot);
        if (maximum == 0.0f) {
            state.singular = 1U;
            return false;
        }
        if (pivot != 0U) {
            auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr());
            LocalTensor<float> scratch(TPosition::VECCALC, Self::PANEL + 16384U, 64U);
            auto rp = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());
            LDET_MASK_SET(Self::B);
            vmuls(rp, mp + pivot * Self::S, 1.0f, 1, 1, 1, 8, 8);
            vmuls(mp + pivot * Self::S, mp, 1.0f, 1, 1, 1, 8, 8);
            PipeBarrier<PIPE_V>();
            vmuls(mp, rp, 1.0f, 1, 1, 1, 8, 8);
            LDET_MASK_RST();
            PipeBarrier<PIPE_V>();
            state.VS();
            state.sign = -state.sign;
        }
        return true;
    }
    template <class Self, class PhaseArg0, class PhaseArg1, class PhaseArg14, class PhaseArg15>
    __aicore__ static inline void
    factorPairsShared66PipeBarrierPhase1(PhaseArg0 &b0, PhaseArg1 &b1, float &delta, uint32_t &f, const uint32_t &h,
                                         const uint32_t &k, uint32_t &kb, float &m00, float &m01, float &m10,
                                         float &m11, __ubuf__ float *&mp, bool &prefixPosted, __ubuf__ float *&r0,
                                         PhaseArg14 &r1, PhaseArg15 &state) {

        PipeBarrier<PIPE_V>();
        const uint32_t start = (f + 2U) & ~7U, parts = (Self::B - start) / 8U;
        const uint8_t repeats = static_cast<uint8_t>((h + 7U) / 8U);
        if (parts) {
            SetMaskNorm();
            AscendCUtils::SetMask<float>(64);
            vmla(mp + (k + 2U) * Self::S + start, r0 + start, b0, repeats, Self::S / 8U, 0, Self::S / 8U, Self::S, 0,
                 Self::S);
            PipeBarrier<PIPE_V>();
            vmla(mp + (k + 2U) * Self::S + start, r1 + start, b1, repeats, Self::S / 8U, 0, Self::S / 8U, Self::S, 0,
                 Self::S);
            PipeBarrier<PIPE_V>();
            if (f + 2U < Self::B)
                SetFlag<HardEvent::V_S>(4);
            for (uint32_t j = 1U; j < parts; ++j)
                vmla(mp + (k + 2U) * Self::S + start + j * 8U, r0 + start + j * 8U, b0, repeats, Self::S / 8U, 0,
                     Self::S / 8U, Self::S, 0, Self::S);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 1U; j < parts; ++j)
                vmla(mp + (k + 2U) * Self::S + start + j * 8U, r1 + start + j * 8U, b1, repeats, Self::S / 8U, 0,
                     Self::S / 8U, Self::S, 0, Self::S);
            ResetMask();
            SetMaskNorm();
        }
        state.mu.SetValue(8U + 2U * f, state.word(m00));
        state.mu.SetValue(9U + 2U * f, state.word(m01));
        state.mu.SetValue(10U + 2U * f, state.word(m10));
        state.mu.SetValue(11U + 2U * f, state.word(m11));
        state.acc(delta);
        PipeBarrier<PIPE_V>();
        if (f == 14U) {
            state.publishPrefix16(kb);
            prefixPosted = true;
        }
    }
    template <class Self, class PhaseArg0, class PhaseArg1, class PhaseArg16, class PhaseArg17, class PhaseArg18,
              class PhaseArg19>
    __aicore__ static inline void
    factorPairsShared66vmulsPhase2(PhaseArg0 &b0, PhaseArg1 &b1, __ubuf__ float *&c0, __ubuf__ float *&c1, float &delta,
                                   uint32_t &f, const uint32_t &h, const uint32_t &k, uint32_t &kb, float &m00,
                                   float &m01, float &m10, float &m11, __ubuf__ float *&mp, bool &prefixPosted,
                                   __ubuf__ float *&r0, PhaseArg16 &r1, PhaseArg17 &state, PhaseArg18 &t0,
                                   PhaseArg19 &t1) {

        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(b0), reinterpret_cast<__ubuf__ uint32_t *>(c0), Self::S / 8U,
              Self::S, static_cast<uint8_t>((h + 7U) / 8U));
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(b1), reinterpret_cast<__ubuf__ uint32_t *>(c1), Self::S / 8U,
              Self::S, static_cast<uint8_t>((h + 7U) / 8U));
        LDET_MASK_SET(Self::B);
        vmuls(r0, mp + k * Self::S, m00, 1, 1, 1, 8, 8);
        vmuls(r1, mp + (k + 1U) * Self::S, m11, 1, 1, 1, 8, 8);
        vmuls(t0, mp + (k + 1U) * Self::S, m01, 1, 1, 1, 8, 8);
        vmuls(t1, mp + k * Self::S, m10, 1, 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
        vadd(r0, r0, t0, 1, 1, 1, 1, 8, 8, 8);
        vadd(r1, r1, t1, 1, 1, 1, 1, 8, 8, 8);
        LDET_MASK_RST();
        factorPairsShared66PipeBarrierPhase1<Self>(b0, b1, delta, f, h, k, kb, m00, m01, m10, m11, mp, prefixPosted, r0,
                                                   r1, state);
    }
    template <class Self, class PhaseArg5, class PhaseArg6, class PhaseArg7, class PhaseArg8, class PhaseArg9,
              class PhaseArg10>
    __aicore__ static inline void
    factorPairsShared66vgatherPhase3(__ubuf__ uint32_t *&ip, uint32_t &kb, __ubuf__ float *&mp, bool &prefixPosted,
                                     __ubuf__ float *&r0, PhaseArg5 &r1, PhaseArg6 &records0, PhaseArg7 &records1,
                                     PhaseArg8 &state, PhaseArg9 &t0, PhaseArg10 &t1) {

        for (uint32_t f = 0U; f < Self::B; f += 2U) {
            if (f)
                WaitFlag<HardEvent::V_S>(4);
            const uint32_t k = kb + f, h = state.n - k - 2U;
            auto c0 = reinterpret_cast<__ubuf__ float *>(state.fac[f * state.facPitch(kb)].GetPhyAddr());
            auto c1 = reinterpret_cast<__ubuf__ float *>(state.fac[(f + 1U) * state.facPitch(kb)].GetPhyAddr());
            const uint32_t a0 = (k + 2U) * Self::S + f, a1 = a0 + 1U;
            auto b0 = records0 + (k + 2U) * Self::S;
            auto b1 = records1 + (k + 2U) * Self::S;
            LDET_MASK_SET(h);
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(c0), ip + (f & 7U) * Self::NMAX,
                    reinterpret_cast<uint64_t>(mp) + (a0 & ~7U) * 4U, 8, 1);
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(c1), ip + ((f + 1U) & 7U) * Self::NMAX,
                    reinterpret_cast<uint64_t>(mp) + (a1 & ~7U) * 4U, 8, 1);
            PipeBarrier<PIPE_V>();
            LDET_MASK_RST();
            float a = mp[k * Self::S + f], b = mp[k * Self::S + f + 1U], c = mp[(k + 1U) * Self::S + f],
                  d = mp[(k + 1U) * Self::S + f + 1U];
            float ad = a * d, bc = b * c, delta = ad - bc;
            uint32_t exponent = (state.word(delta) >> 23U) & 255U;
            bool safe = exponent != 0U && exponent != 255U &&
                        state.pairAbs(delta) >= .25f * (state.pairAbs(ad) + state.pairAbs(bc));
            float inverse = safe ? 1.0f / delta : 0.0f;
            float m00 = -d * inverse, m01 = b * inverse, m10 = c * inverse, m11 = -a * inverse;
            safe = safe && ((state.word(m00) >> 23U) & 255U) != 255U && ((state.word(m01) >> 23U) & 255U) != 255U &&
                   ((state.word(m10) >> 23U) & 255U) != 255U && ((state.word(m11) >> 23U) & 255U) != 255U;
            if (!safe) {
                state.singular = 3U;
                break;
            }
            factorPairsShared66vmulsPhase2<Self>(b0, b1, c0, c1, delta, f, h, k, kb, m00, m01, m10, m11, mp,
                                                 prefixPosted, r0, r1, state, t0, t1);
        }
        state.VS();
    }
    template <class Self> __aicore__ static inline void factorPairsShared66(Self &state, uint32_t kb, uint32_t q) {
        state.panel = state.mat;
        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr());
        auto ip = reinterpret_cast<__ubuf__ uint32_t *>(state.lanes.GetPhyAddr());
        LocalTensor<float> rows(TPosition::VECCALC, Self::PANEL + 16384U, 256U);
        auto records0 = mp + 64U;
        auto records1 = mp + 72U;

        auto r0 = reinterpret_cast<__ubuf__ float *>(rows.GetPhyAddr());
        auto r1 = r0 + 64U, t0 = r0 + 128U, t1 = r0 + 192U;
        bool prefixPosted = false;
        state.VS();
        if (kb == 0U && !state.headPivot()) {
            state.publishPrefix16(kb);
            return;
        }
        factorPairsShared66vgatherPhase3<Self>(ip, kb, mp, prefixPosted, r0, r1, records0, records1, state, t0, t1);
        if (!prefixPosted)
            state.publishPrefix16(kb);
    }
    template <class Self, uint32_t FIRST, uint32_t LAST, class PhaseArg3>
    __aicore__ static inline void ApplyPairsShared67PipeBarrierPhase1(uint32_t &kb, __ubuf__ float *&mp,
                                                                      __ubuf__ float *&r0, PhaseArg3 &state,
                                                                      __ubuf__ float *&wp) {

        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(wp),
              reinterpret_cast<__ubuf__ uint32_t *>(state.meta[8U + 2U * FIRST].GetPhyAddr()), 1, 8,
              static_cast<uint8_t>((LAST - FIRST) / 4U));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        AscendCUtils::SetMask<float>(state.rank == Self::G - 1U ? 64 : 32);
        if constexpr (FIRST == 0U) {
            if (kb == 0U) {
                const uint32_t pivot = state.mu.GetValue(4U);
                if (pivot != 0U) {
                    vmuls(r0, mp + pivot * Self::S, 1.0f, 1, 1, 1, 8, 8);
                    vmuls(mp + pivot * Self::S, mp, 1.0f, 1, 1, 1, 8, 8);
                    PipeBarrier<PIPE_V>();
                    vmuls(mp, r0, 1.0f, 1, 1, 1, 8, 8);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }
    template <class Self, class PhaseArg0, class PhaseArg1, class PhaseArg6, class PhaseArg9>
    __aicore__ static inline void ApplyPairsShared67PipeBarrierPhase2(PhaseArg0 &b0, PhaseArg1 &b1, uint32_t &f,
                                                                      const uint32_t &h, const uint32_t &k,
                                                                      uint32_t &kb, PhaseArg6 &m, __ubuf__ float *&mp,
                                                                      __ubuf__ float *&r0, PhaseArg9 &state) {

        vmul(r0, mp + k * Self::S, m, 2, 1, 1, 0, 8, 0, 2);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(b0),
              reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), Self::S / 8U,
              Self::S, static_cast<uint8_t>((h + 7U) / 8U));
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(b1),
              reinterpret_cast<__ubuf__ uint32_t *>(state.fac[(f + 1U) * state.facPitch(kb)].GetPhyAddr()),
              Self::S / 8U, Self::S, static_cast<uint8_t>((h + 7U) / 8U));
        PipeBarrier<PIPE_V>();
        vmla(r0, mp + (k + 1U) * Self::S, m + 8U, 2, 1, 1, 0, 8, 0, 2);
        PipeBarrier<PIPE_V>();
    }
    template <class Self, uint32_t FIRST = 0U, uint32_t LAST = Self::B, bool TILE40 = false>
    __aicore__ static inline void ApplyPairsShared67(Self &state, uint32_t kb, uint32_t q) {
        if (state.rank <= q)
            return;
        static_assert(Self::S == 136U && Self::B == 32U && Self::G == 6U, "private n200 six-role scope");
        constexpr uint32_t CB0 = 64U, CB1 = 96U;
        static_assert(CB0 >= 64U && CB1 + 8U <= Self::S, "coefficient records outside all live columns");
        LocalTensor<float> rows(TPosition::VECCALC, Self::PANEL + 16384U, 128U);
        LocalTensor<float> weights(TPosition::VECCALC, Self::PANEL + 18432U, 512U);
        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr());
        auto r0 = reinterpret_cast<__ubuf__ float *>(rows.GetPhyAddr()), r1 = r0 + 64U;
        auto wp = reinterpret_cast<__ubuf__ float *>(weights.GetPhyAddr());
        ApplyPairsShared67PipeBarrierPhase1<Self, FIRST, LAST>(kb, mp, r0, state, wp);
        for (uint32_t f = FIRST; f < LAST; f += 2U) {
            const uint32_t k = kb + f, h = state.n - k - 2U;
            auto m = wp + 16U * (f - FIRST);
            auto b0 = mp + (k + 2U) * Self::S + CB0, b1 = mp + (k + 2U) * Self::S + CB1;
            ApplyPairsShared67PipeBarrierPhase2<Self>(b0, b1, f, h, k, kb, m, mp, r0, state);
            if constexpr (TILE40) {
                const uint8_t repeats = static_cast<uint8_t>((h + 7U) / 8U);
#pragma unroll
                for (uint32_t j = 0U; j < 5U; ++j)
                    vmla(mp + (k + 2U) * Self::S + j * 8U, r0 + j * 8U, b0, repeats, Self::S / 8U, 0, Self::S / 8U,
                         Self::S, 0, Self::S);
                PipeBarrier<PIPE_V>();
#pragma unroll
                for (uint32_t j = 0U; j < 5U; ++j)
                    vmla(mp + (k + 2U) * Self::S + j * 8U, r1 + j * 8U, b1, repeats, Self::S / 8U, 0, Self::S / 8U,
                         Self::S, 0, Self::S);
                PipeBarrier<PIPE_V>();
            } else {
                vmla(mp + (k + 2U) * Self::S, r0, b0, static_cast<uint8_t>(h), 1, 1, 0, Self::S / 8U, 0, Self::S / 8U);
                PipeBarrier<PIPE_V>();
                vmla(mp + (k + 2U) * Self::S, r1, b1, static_cast<uint8_t>(h), 1, 1, 0, Self::S / 8U, 0, Self::S / 8U);
                PipeBarrier<PIPE_V>();
            }
        }
        ResetMask();
        SetMaskNorm();
        state.VS();
    }
    template <class Self> __aicore__ static inline void factorShared68(Self &state, uint32_t kb, uint32_t q) {
        state.factorPairs(kb, q);
    }
