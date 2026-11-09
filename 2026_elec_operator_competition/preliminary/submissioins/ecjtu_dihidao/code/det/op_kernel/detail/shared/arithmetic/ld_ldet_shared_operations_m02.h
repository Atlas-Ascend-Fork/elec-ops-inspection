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
    template <class Self, bool EXPORT = true>
    __aicore__ static inline bool updatePanelAndPrepareShared20(Self &state, uint32_t kb, uint32_t f, uint32_t h,
                                                                uint32_t c0, uint32_t width) {
        if (h < 64U || width < 16U || f + 1U >= Self::B) {
            state.updatePanelRows8x8(state.panel[(kb + f + 1U) * Self::P + c0], state.prow[c0], h, width);
            return false;
        }
        Shared24ScalarUpdateStage11<EXPORT, Self>(c0, f, h, kb, state, width);
        return true;
    }
    template <auto T, auto TS, class Capture6>
    __aicore__ static inline void SharedVgatherStage42(__ubuf__ float *&ap, __ubuf__ float *&cp, __ubuf__ uint32_t *&ip,
                                                       __ubuf__ float *&mp, const uint32_t &nf, const uint32_t &nh,
                                                       Capture6 &state) {
        PrepareNextTailColumn60<T, TS>(state, nf, nh, ap, cp, ip, mp);
    }
    template <auto TS, class Capture5>
    __aicore__ static inline void Shared24VmlaStage12(uint32_t &f, const uint32_t &first, uint32_t &h,
                                                      __ubuf__ float *&mp, __ubuf__ float *&rp, Capture5 &state) {
        SetMaskNorm();
        AscendCUtils::SetMask<float>(64);
        vmla(mp + (f + 1U) * TS + first, rp + first, reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()),
             static_cast<uint8_t>((h - 1U + 7U) / 8U), TS / 8U, 0, 1, TS, 0, 8);
        PipeBarrier<PIPE_V>();
    }
    template <auto T, auto TS, class Capture5>
    __aicore__ static inline void Shared24VmlaStage21(uint32_t &f, const uint32_t &first, uint32_t &h,
                                                      __ubuf__ float *&mp, __ubuf__ float *&rp, Capture5 &state) {
        SetMaskNorm();
        AscendCUtils::SetMask<float>(static_cast<int32_t>(T - first - 8U));
        vmla(mp + (f + 1U) * TS + first + 8U, rp + first + 8U,
             reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()), static_cast<uint8_t>(h - 1U), 1, 1, 0,
             TS / 8U, 0, 1);
        ResetMask();
        SetMaskNorm();
    }
    template <auto T, auto TS, class PhaseArg10>
    __aicore__ static inline void
    Finish64Shared21PipeBarrierPhase1(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f, uint32_t &h,
                                      __ubuf__ uint32_t *&ip, __ubuf__ float *&mp, const uint32_t &p, bool &pendingFar,
                                      bool &prepared, __ubuf__ float *&rp, PhaseArg10 &state) {

        state.repair(p - f);
        PipeBarrier<PIPE_V>();
        const uint32_t first = (f + 1U) & ~7U;
        prepared = false;
        if (h > 1U && first < T) {
            Shared24VmlaStage12<TS>(f, first, h, mp, rp, state);
            const uint32_t nf = f + 1U, nh = h - 1U;
            SharedVgatherStage42<T, TS>(ap, cp, ip, mp, nf, nh, state);
            prepared = true;
            if (first + 8U < T) {
                Shared24VmlaStage21<T, TS>(f, first, h, mp, rp, state);
                pendingFar = true;
            }
        }
    }
    template <auto T, auto TS, class Block36Capture6_7>
    __aicore__ static inline void Shared36PipeBarrierStage7(uint32_t &f, uint32_t &h, float &inv, __ubuf__ float *&mp,
                                                            const uint32_t &p, bool &pendingFar, __ubuf__ float *&rp,
                                                            Block36Capture6_7 &state) {
        if (pendingFar)
            PipeBarrier<PIPE_V>();
        pendingFar = false;
        if (h > 1U)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(state.col.GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((h + 7U) / 8U));
        LDET_MASK_SET(T);
        vmuls(rp, mp + p * TS, inv, 1, 1, 1, 8, 8);
        if (p != f)
            vmuls(mp + p * TS, mp + f * TS, 1.0f, 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        if (p != f)
            PipeBarrier<PIPE_V>();
    }
    template <auto T, auto TS, class Capture7>
    __aicore__ static inline void SharedPipeBarrierStage35(uint32_t &f, uint32_t &h, float &inv, __ubuf__ float *&mp,
                                                           const uint32_t &p, bool &pendingFar, __ubuf__ float *&rp,
                                                           Capture7 &state) {
        Shared36PipeBarrierStage7<T, TS>(f, h, inv, mp, p, pendingFar, rp, state);
    }
    template <class Self, auto T, auto TS, class PhaseArg12>
    __aicore__ static inline void
    Finish64Shared21PipeBarrierPhase2(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f, uint32_t &h, float &inv,
                                      __ubuf__ uint32_t *&ip, __ubuf__ float *&mp, const uint32_t &p, bool &pendingFar,
                                      float &piv, bool &prepared, __ubuf__ float *&rp, PhaseArg12 &state) {

        if (p != f)
            state.sign = -state.sign;
        SharedPipeBarrierStage35<T, TS>(f, h, inv, mp, p, pendingFar, rp, state);
        Finish64Shared21PipeBarrierPhase1<T, TS>(ap, cp, f, h, ip, mp, p, pendingFar, prepared, rp, state);
        state.acc(piv);
        if constexpr (Self::LdetPolicy::PANEL_ACC) {
            if ((f & 31U) == 31U)
                state.normalizeProd();
        }
    }
    template <auto T, auto TS, class Capture0> __aicore__ static inline void SharedPipeBarrierStage21(Capture0 &state) {
        PrepareTailRowIndices46<TS, 32>(state);
        LDET_MASK_SET(T);
        for (uint32_t j = 0; j < 8U; ++j)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[j * T].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1,
                  8, 8);
        LDET_MASK_RST();
    }
    template <class Self, auto T, auto TS, class PhaseArg0, class PhaseArg1>
    __aicore__ static inline void Finish64Shared21DataCopyPhase3(PhaseArg0 &state, PhaseArg1 &tm) {

        DataCopy(tm, state.mat[192U * Self::S], {static_cast<uint16_t>(T), T / 8U, (Self::S - T) / 8U, (TS - T) / 8U});
        PipeBarrier<PIPE_V>();
        SharedPipeBarrierStage21<T, TS>(state);
        PipeBarrier<PIPE_V>();
    }
    template <auto T, auto TS, class PhaseArg7>
    __aicore__ static inline void Finish64Shared21vgatherPhase4(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                                uint32_t &h, __ubuf__ uint32_t *&ip,
                                                                __ubuf__ float *&mp, bool &prepared, PhaseArg7 &state) {

        if (!prepared) {
            LDET_MASK_SET(h);
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(cp), ip + (f & 7U) * T,
                    reinterpret_cast<uint64_t>(mp) + (f * TS + (f & ~7U)) * 4U, 8, 1);
            PipeBarrier<PIPE_V>();
            vabs(ap, cp, 1, 1, 1, 8, 8);
            LDET_MASK_RST();
            LDET_MASK_SET(h);
            LdetVcmax128(state.tmp, state.ab, h);
            LDET_MASK_RST();
            SetFlag<HardEvent::V_S>(2);
        }
        WaitFlag<HardEvent::V_S>(2);
    }
    template <class Self, auto T, auto TS, class Capture0, class Capture1>
    __aicore__ static inline void SharedScalarUpdateStage11(Capture0 &state, Capture1 &tm) {
        auto views = GetTailRawViews51(state, tm);
        bool prepared = false, pendingFar = false;
        for (uint32_t f = 0; f < T; ++f) {
            uint32_t h = T - f;
            Finish64Shared21vgatherPhase4<T, TS>(views.ap, views.cp, f, h, views.ip, views.mp, prepared, state);
            const uint32_t ri = state.word(views.tp[1]);
            if (ri >= h) {
                state.singular = 1;
                break;
            }
            const uint32_t p = f + ri;
            float piv = views.cp[ri];
            if (piv == 0.0f || ((state.word(piv) >> 23U) & 255U) == 255U) {
                state.acc(piv);
                break;
            }
            float inv = -1.0f / piv;
            if (((state.word(inv) >> 23U) & 255U) == 255U) {
                state.acc(piv);
                state.singular = 2;
                break;
            }
            Finish64Shared21PipeBarrierPhase2<Self, T, TS>(views.ap, views.cp, f, h, inv, views.ip, views.mp, p,
                                                           pendingFar, piv, prepared, views.rp, state);
        }
    }
    template <class Self> __aicore__ static inline void Finish64Shared21(Self &state) {
        constexpr uint32_t T = 64U, TS = 72U;
        static_assert((T + 8U) * TS <= Self::P * (Self::NMAX + Self::GUARD_ROWS), "retired PANEL tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, Self::PANEL, (T + 8U) * TS);
        Finish64Shared21DataCopyPhase3<Self, T, TS>(state, tm);
        SharedScalarUpdateStage11<Self, T, TS>(state, tm);
        if constexpr (Self::LdetPolicy::PANEL_ACC)
            state.normalizeProd();
        state.VS();
        state.meta.SetValue(0, state.singular ? 0.0f : state.sign * state.final());
        state.SM();
        DataCopyPad(state.out, state.meta, DataCopyExtParams{1, 4, 0, 0, 0});
        state.MS();
    }

    __aicore__ static inline auto bandFirstShared22StateDepth22Stage1IndexSelect22Stage1(uint32_t &r)
        -> decltype(r == 0U ? 0U : (r == 1U ? 2U : (r == 2U ? 3U : (r == 3U ? 4U : 6U)))) {
        if (r == 0U) {
            return 0U;
        }
        if (r == 1U) {
            return 2U;
        }
        if (r == 2U) {
            return 3U;
        }
        if (r == 3U) {
            return 4U;
        }
        return 6U;
    }
    template <class Depth22Arg0_1>
    __aicore__ static inline uint32_t bandFirstShared22StateDepth22Stage1(uint32_t &r, Depth22Arg0_1 &state) {

        if (state.recycleState && r == 0U)
            return 5U;
        return bandFirstShared22StateDepth22Stage1IndexSelect22Stage1(r);
    }
    template <class Self> __aicore__ static inline uint32_t bandFirstShared22(Self &state, uint32_t r) {
        if constexpr (Self::G == 5U) {
            return bandFirstShared22StateDepth22Stage1(r, state);
        } else {
            if (state.recycleState && r == 0U)
                return 3U;
            if (state.recycleState >= 2U && r == 1U)
                return 5U;
            return r * 2U;
        }
    }
    template <class Self> __aicore__ static inline GM_ADDR prefixTicketShared23(Self &state, uint32_t q) {
        if constexpr (Self::LdetPolicy::HETERO5)
            return reinterpret_cast<GM_ADDR>(state.relay.GetPhyAddr(192U * 32U)) + q * 64U;
        else
            return state.prefixBase + q * 64U;
    }
    template <class Self> __aicore__ static inline void publishPrefix16Shared24(Self &state, uint32_t kb) {
        const uint32_t slot = ((kb / Self::B) & 1U) * Self::GM_WORDS;
        state.bindPayloadMeta(kb);
        state.state();
        state.SM();
        state.VM();
        DataCopy(state.ws[slot + Self::B * state.facPitch(kb)], state.meta, Self::META_WORDS);
        DataCopy(state.ws[slot], state.fac, 16U * state.facPitch(kb));
        state.MS();
        LuPublish(state.prefixTicket(kb / Self::B), 0x6A1F0010U);
    }
    template <uint32_t FIRST, uint32_t LAST, class Self, class Capture1>
    __aicore__ static inline void SharedDataCopyStage18(uint32_t &kb, Capture1 &state) {
        const uint32_t slot = ((kb / Self::B) & 1U) * Self::GM_WORDS;
        state.bindPayloadMeta(kb);
        if constexpr (LAST == Self::B) {
            DataCopy(state.fac[FIRST * state.facPitch(kb)], state.ws[slot + FIRST * state.facPitch(kb)],
                     (LAST - FIRST) * state.facPitch(kb) + Self::META_WORDS);
        } else {
            DataCopy(state.meta, state.ws[slot + Self::B * state.facPitch(kb)], Self::META_WORDS);
            DataCopy(state.fac[FIRST * state.facPitch(kb)], state.ws[slot + FIRST * state.facPitch(kb)],
                     (LAST - FIRST) * state.facPitch(kb));
        }
        state.MV();
        state.VS();
        state.sign = state.value(state.mu.GetValue(0));
        state.prod = state.value(state.mu.GetValue(1));
        state.ex = int32_t(state.mu.GetValue(2));
        state.singular = state.mu.GetValue(3);
    }
    template <class Self, uint32_t FIRST, uint32_t LAST>
    __aicore__ static inline void receiveRangeShared25(Self &state, uint32_t kb) {
        static_assert(FIRST < LAST && LAST <= Self::B, "FAC prefix range");
        if constexpr (((Self::LdetPolicy::GROUP == 4U && Self::LdetPolicy::PHASE_TRIM) ||
                       (Self::LdetPolicy::GROUP == 5U && Self::LdetPolicy::HETERO5)) &&
                      FIRST == 0U) {
            SetFlag<HardEvent::V_MTE2>(4);
            WaitFlag<HardEvent::V_MTE2>(4);
        }
        SharedDataCopyStage18<FIRST, LAST, Self>(kb, state);
    }
    template <class Self> __aicore__ static inline void publishShared26(Self &state, uint32_t kb) {
        const uint32_t slot = ((kb / Self::B) & 1U) * Self::GM_WORDS;
        state.bindPayloadMeta(kb);
        state.state();
        state.SM();
        state.VM();
        if constexpr (Self::LdetPolicy::PREFIX_STREAM) {
            if (kb) {
                DataCopy(state.ws[slot + 16U * state.facPitch(kb)], state.fac[16U * state.facPitch(kb)],
                         16U * state.facPitch(kb) + Self::META_WORDS);
            } else
                DataCopy(state.ws[slot], state.fac, Self::B * state.facPitch(kb) + Self::META_WORDS);
        } else
            DataCopy(state.ws[slot], state.fac, Self::B * state.facPitch(kb) + Self::META_WORDS);
        state.MS();
    }
    template <class Self, bool EXPORT = true>
    __aicore__ static inline bool updatePanelAndPrepareShared27(Self &state, uint32_t kb, uint32_t f, uint32_t h,
                                                                uint32_t c0, uint32_t width) {
        if (h < 64U || width < (Self::FAST_OWNER ? 8U : 16U) || f + 1U >= Self::B) {
            state.updatePanelRows8x8(state.panel[(kb + f + 1U) * Self::P + c0], state.prow[c0], h, width);
            return false;
        }
        Shared24ScalarUpdateStage11<EXPORT, Self>(c0, f, h, kb, state, width);
        return true;
    }
    template <bool EXPORT, class Self, class Capture4>
    __aicore__ static inline void Shared24PipeBarrierStage17(uint32_t &f, float &inv, uint32_t &p, float &piv,
                                                             Capture4 &state) {
        if constexpr (Self::LdetPolicy::DEFER_BOOKKEEP) {
            state.acc(piv);
            if constexpr (EXPORT) {
                state.mu.SetValue(8 + f, p);
                state.mu.SetValue(8U + Self::B + f, state.word(inv));
            }
        }
        PipeBarrier<PIPE_V>();
    }
    template <bool EXPORT, class Self, class Capture8>
    __aicore__ static inline void SharedPipeBarrierStage16(uint32_t &f, uint32_t &h, float &inv, uint32_t &k,
                                                           uint32_t &kb, uint32_t &p, float &piv, bool &pivotPrepared,
                                                           Capture8 &state) {
        if constexpr (Self::LdetPolicy::NORMAL_COPY)
            ResetMask();
        else
            LDET_MASK_RST();
        if constexpr (Self::GUARDED || Self::LdetPolicy::PIVOT8_AHEAD) {
            if (p != k)
                PipeBarrier<PIPE_V>();
        }
        state.repair(p - k);
        PipeBarrier<PIPE_V>();
        uint32_t c0 = (f + 1) & ~7U, w = Self::B - c0;
        if constexpr (Self::LdetPolicy::FAC_REPAIRED && EXPORT)
            state.patchFac(f, kb, p - k);
        if constexpr (Self::LdetPolicy::PIVOT8_AHEAD) {
            pivotPrepared = w ? state.template updatePanelAndPrepare<EXPORT>(kb, f, h, c0, w) : false;
        } else {
            if (w)
                state.updatePanelRows8x8(state.panel[(k + 1) * Self::P + c0], state.prow[c0], h, w);
        }
        Shared24PipeBarrierStage17<EXPORT, Self>(f, inv, p, piv, state);
    }
    template <class Self, bool EXPORT, class PhaseArg9>
    __aicore__ static inline void
    factorShared28PipeBarrierPhase1(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb, uint32_t &p,
                                    float &piv, bool &pivotPrepared, bool &prefixPosted, PhaseArg9 &state) {

        SharedPipeBarrierStage16<EXPORT, Self>(f, h, inv, k, kb, p, piv, pivotPrepared, state);
        if constexpr (Self::LdetPolicy::PREFIX_STREAM && EXPORT) {
            if (kb && f == 15U) {
                state.publishPrefix16(kb);
                prefixPosted = true;
            }
        }
    }
    template <class Self, class Capture7>
    __aicore__ static inline void SharedVgatherStage10(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                                       uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip,
                                                       __ubuf__ float *&pp, Capture7 &state) {
        Shared10VgatherStage4<Self>(ap, be, cp, f, h, ip, pp, state);
    }
    template <bool EXPORT, class Self, class Capture10>
    __aicore__ static inline void Shared10SetFlagStage5(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                                        uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k,
                                                        uint32_t &kb, bool &pivotPrepared, __ubuf__ float *&pp,
                                                        Capture10 &state) {
        if constexpr (Self::LdetPolicy::DIRECT_FAC && EXPORT)
            cp = reinterpret_cast<__ubuf__ float *>(state.fac[f * state.facPitch(kb)].GetPhyAddr());
        if constexpr (Self::LdetPolicy::PIVOT8_AHEAD) {
            if (!pivotPrepared)
                state.template issuePreparedPivot<EXPORT>(k, f);
        } else {
            SharedVgatherStage10<Self>(ap, be, cp, f, h, ip, pp, state);
        }
        if constexpr (Self::GUARDED && !Self::LdetPolicy::PIVOT8_AHEAD)
            SetFlag<HardEvent::V_S>(2);
    }
