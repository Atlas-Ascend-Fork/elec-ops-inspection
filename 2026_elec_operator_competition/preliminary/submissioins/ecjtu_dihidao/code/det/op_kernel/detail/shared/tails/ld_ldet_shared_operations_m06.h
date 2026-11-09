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
    template <class Self, uint32_t FIRST, uint32_t LAST, class PhaseArg4>
    __aicore__ static inline void ApplyPanelUpdatesShared54vmulsPhase1(uint32_t &begin, uint32_t &kb,
                                                                       __ubuf__ float *&mp, __ubuf__ float *&rp,
                                                                       PhaseArg4 &state) {

        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(64);
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t kk = kb + f, h = state.n - kk, pivot = state.mu.GetValue(8U + f);
            float inverse = state.value(state.mu.GetValue(8U + Self::B + f));
            vmuls(rp, mp + pivot * Self::S + begin, inverse, 2, 1, 1, 8, 8);
            if (pivot != kk)
                vmuls(mp + pivot * Self::S + begin, mp + kk * Self::S + begin, 1.0f, 2, 1, 1, 8, 8);
            if (h > 1U)
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                      static_cast<uint8_t>((h + 7U) / 8U));
            PipeBarrier<PIPE_V>();
            if (h > 1U) {
                vmla(mp + (kk + 1U) * Self::S + begin, rp,
                     reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()), static_cast<uint8_t>(h - 1U), 1, 1,
                     0, Self::S / 8U, 0, 1);
                if constexpr (Self::NATIVE208) {
                    for (uint32_t off = 64U; off < Self::LdetPolicy::DIM - Self::LdetPolicy::LOCAL_CUT; off += 8U)
                        vmla(mp + (kk + 1U) * Self::S + begin + off, rp + off,
                             reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()),
                             static_cast<uint8_t>((h - 1U + 7U) / 8U), Self::S / 8U, 0, 1, Self::S, 0, 8);
                } else {
                    vmla(mp + (kk + 1U) * Self::S + begin + 64U, rp + 64U,
                         reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()),
                         static_cast<uint8_t>((h - 1U + 7U) / 8U), Self::S / 8U, 0, 1, Self::S, 0, 8);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        ResetMask();
        SetMaskNorm();
        state.VS();
        return;
    }
    template <class Self, uint32_t FIRST, uint32_t LAST, class PhaseArg4>
    __aicore__ static inline void ApplyPanelUpdatesShared54PipeBarrierPhase2(uint32_t &begin, uint32_t &kb,
                                                                             __ubuf__ float *&mp, __ubuf__ float *&rp,
                                                                             PhaseArg4 &state, uint32_t &width) {

        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(static_cast<int32_t>(width));
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
            float inv = state.value(state.mu.GetValue(8U + Self::B + f));
            if constexpr (Self::META_AHEAD) {
                if (f)
                    PipeBarrier<PIPE_V>();
            }
            vmuls(rp, mp + p * Self::S + begin, inv, 1, 1, 1, 8, 8);
            if (p != k)
                vmuls(mp + p * Self::S + begin, mp + k * Self::S + begin, 1.0f, 1, 1, 1, 8, 8);
            if (h > 1)
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                      static_cast<uint8_t>((h + 7) / 8));
            PipeBarrier<PIPE_V>();
            if (h > 1)
                vmla(reinterpret_cast<__ubuf__ float *>(state.mat[(k + 1) * Self::S + begin].GetPhyAddr()), rp,
                     reinterpret_cast<__ubuf__ float *>(state.brc[8].GetPhyAddr()), static_cast<uint8_t>(h - 1U), 1, 1,
                     0, Self::S / 8U, 0, 1);
            if constexpr (!Self::META_AHEAD)
                PipeBarrier<PIPE_V>();
        }
        if constexpr (Self::META_AHEAD)
            PipeBarrier<PIPE_V>();
        ResetMask();
        SetMaskNorm();
        state.VS();
        return;
    }
    template <class Self, uint32_t FIRST, uint32_t LAST, class PhaseArg4>
    __aicore__ static inline void ApplyPanelUpdatesShared54PipeBarrierPhase3(uint32_t &begin, uint32_t &kb,
                                                                             __ubuf__ float *&mp, __ubuf__ float *&rp,
                                                                             PhaseArg4 &state, uint32_t &width) {

        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
            float inv = state.value(state.mu.GetValue(8U + Self::B + f));
            if constexpr (Self::META_AHEAD) {
                if (f)
                    PipeBarrier<PIPE_V>();
            }
            LDET_MASK_SET(width);
            vmuls(rp, mp + p * Self::S + begin, inv, 1, 1, 1, 8, 8);
            if (p != k)
                vmuls(mp + p * Self::S + begin, mp + k * Self::S + begin, 1.0f, 1, 1, 1, 8, 8);
            LDET_MASK_RST();
            SharedVbrcbStage40<Self>(begin, f, h, k, kb, p, state, width);
            if constexpr (!Self::META_AHEAD)
                PipeBarrier<PIPE_V>();
        }
        if constexpr (Self::META_AHEAD)
            PipeBarrier<PIPE_V>();
        state.VS();
    }
    template <class Self, class PhaseArg2>
    __aicore__ static inline void ApplyPanelUpdatesShared54constexprPhase4(uint32_t &first, uint32_t &q,
                                                                           PhaseArg2 &state) {

        if constexpr (Self::LdetPolicy::CONTIGUOUS_BANDS) {
            const uint32_t base = state.bandFirst(state.rank);
            first = q < base ? 0U : q - base + 1U;
        } else {
            first = q < state.rank ? 0U : (q - state.rank) / Self::G + 1U;
        }
    }
    template <class Self, uint32_t FIRST = 0U, uint32_t LAST = Self::B>
    __aicore__ static inline void ApplyPanelUpdatesShared54(Self &state, uint32_t kb, uint32_t q) {
        uint32_t first;
        ApplyPanelUpdatesShared54constexprPhase4<Self>(first, q, state);
        uint32_t begin = first * Self::B, width = state.owned * Self::B > begin ? state.owned * Self::B - begin : 0U;
        if (!width)
            return;
        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr());
        auto rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
        if constexpr (Self::TUNED_PANEL && Self::LdetPolicy::FAC_REPAIRED &&
                      (Self::S == 64U || Self::BALANCED200 || Self::PAIR200)) {
            if (!Self::LdetPolicy::LOCAL_TAIL || (Self::LdetPolicy::DIM != 200U && !Self::NATIVE208) || width <= 64U) {
                return ApplyPanelUpdatesShared54PipeBarrierPhase2<Self, FIRST, LAST>(begin, kb, mp, rp, state, width);
            }
        }
        if constexpr ((Self::BALANCED200 || Self::PAIR200) && Self::LdetPolicy::LOCAL_TAIL) {
            static_assert(Self::S >= 128U && !Self::LdetPolicy::EXTRA,
                          "128 physical row storage is initialized by original Init");
            if (state.rank == Self::G - 1U && width > 64U) {
                return ApplyPanelUpdatesShared54vmulsPhase1<Self, FIRST, LAST>(begin, kb, mp, rp, state);
            }
        }
        ApplyPanelUpdatesShared54PipeBarrierPhase3<Self, FIRST, LAST>(begin, kb, mp, rp, state, width);
    }
    template <class Self, class Depth22Arg2_1>
    __aicore__ static inline void copyInputShared55DataCopyPadDepth22Stage1(uint32_t &j, Depth22Arg2_1 &state) {

        const uint32_t sourceColumn =
            (Self::LdetPolicy::CONTIGUOUS_BANDS ? state.bandFirst(state.rank) + j : state.rank + j * Self::G) * Self::B;
        const uint32_t width = state.n - sourceColumn < Self::B ? state.n - sourceColumn : Self::B;
        if constexpr (Self::LdetPolicy::PAD_INPUT) {
            if (sourceColumn < state.inputDim) {
                const uint32_t validWidth =
                    state.inputDim - sourceColumn < Self::B ? state.inputDim - sourceColumn : Self::B;
                const uint32_t alignedWidth = (validWidth + 7U) & ~7U;
                LocalTensor<float> dst(TPosition::VECIN, Self::MAT + j * Self::B * sizeof(float),
                                       Self::NMAX * Self::S - j * Self::B);
                DataCopyPad(
                    dst, state.in[sourceColumn],
                    DataCopyExtParams{static_cast<uint16_t>(state.inputDim), validWidth * 4U,
                                      (state.inputDim - validWidth) * 4U, (Self::S - alignedWidth) / 8U, 0U},
                    DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>(alignedWidth - validWidth), 0.0f});
            }
        } else {
            DataCopy(state.mat[j * Self::B], state.in[sourceColumn],
                     {static_cast<uint16_t>(state.n), static_cast<uint16_t>(width / 8U),
                      static_cast<uint16_t>((state.n - width) / 8U), static_cast<uint16_t>((Self::S - width) / 8U)});
        }
    }
    template <class Self> __aicore__ static inline void copyInputShared55(Self &state, bool valid) {
        if constexpr (Self::LdetPolicy::PAD_INPUT) {
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        if (valid)
            for (uint32_t j = 0; j < state.owned; ++j) {
                copyInputShared55DataCopyPadDepth22Stage1<Self>(j, state);
            }
    }
    template <auto T, auto TS, class PhaseArg12>
    __aicore__ static inline void
    FinishLocalTailGeppShared56vgatherPhase1(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                             const uint32_t &first, uint32_t &h, __ubuf__ uint32_t *&ip,
                                             __ubuf__ float *&mp, const uint32_t &nf, const uint32_t &nh,
                                             bool &pendingFar, bool &prepared, __ubuf__ float *&rp, PhaseArg12 &state) {

        SharedVgatherStage42<T, TS>(ap, cp, ip, mp, nf, nh, state);
        prepared = true;
        if (first + 8U < T) {
            if constexpr (T > 72U) {
                for (uint32_t off = first + 8U; off < T; off += 64U) {
                    SetMaskNorm();
                    AscendCUtils::SetMask<float>(static_cast<int32_t>(T - off < 64U ? T - off : 64U));
                    vmla(mp + (f + 1U) * TS + off, rp + off,
                         reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()), static_cast<uint8_t>(h - 1U),
                         1, 1, 0, TS / 8U, 0, 1);
                }
                ResetMask();
                SetMaskNorm();
                pendingFar = true;
            } else {
                Shared24VmlaStage21<T, TS>(f, first, h, mp, rp, state);
                pendingFar = true;
            }
        }
    }
    template <class Self, auto T, auto TS, class PhaseArg12>
    __aicore__ static inline void
    FinishLocalTailGeppShared56PipeBarrierPhase2(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f, uint32_t &h,
                                                 float &inv, __ubuf__ uint32_t *&ip, __ubuf__ float *&mp,
                                                 const uint32_t &p, bool &pendingFar, float &piv, bool &prepared,
                                                 __ubuf__ float *&rp, PhaseArg12 &state) {

        SharedPipeBarrierStage35<T, TS>(f, h, inv, mp, p, pendingFar, rp, state);
        state.repair(p - f);
        PipeBarrier<PIPE_V>();
        const uint32_t first = (f + 1U) & ~7U;
        prepared = false;
        if (h > 1U && first < T) {
            Shared24VmlaStage12<TS>(f, first, h, mp, rp, state);
            const uint32_t nf = f + 1U, nh = h - 1U;
            FinishLocalTailGeppShared56vgatherPhase1<T, TS>(ap, cp, f, first, h, ip, mp, nf, nh, pendingFar, prepared,
                                                            rp, state);
        }
        state.acc(piv);
        if constexpr (Self::LdetPolicy::PANEL_ACC) {
            if ((f & 31U) == 31U)
                state.normalizeProd();
        }
    }
    template <class Self, auto T, auto TS, class PhaseArg0, class PhaseArg1>
    __aicore__ static inline void FinishLocalTailGeppShared56DataCopyPhase3(PhaseArg0 &state, PhaseArg1 &tm) {

        DataCopy(tm, state.mat[Self::LdetPolicy::LOCAL_CUT * Self::S],
                 {static_cast<uint16_t>(T), T / 8U, (Self::S - T) / 8U, (TS - T) / 8U});
        PipeBarrier<PIPE_V>();
        PrepareTailRowIndices46<TS, 32>(state);
        if constexpr (T > 64U) {
            Adds(state.tab[64U], state.tab, static_cast<int32_t>(64U * TS * 4U), T - 64U);
            PipeBarrier<PIPE_V>();
        }
        LDET_MASK_SET(T);
        for (uint32_t j = 0; j < 8U; ++j)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[j * T].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1,
                  8, 8);
        LDET_MASK_RST();
        PipeBarrier<PIPE_V>();
    }
    template <auto T, class PhaseArg3>
    __aicore__ static inline void FinishLocalTailGeppShared56wordPhase4(uint32_t &h, float &maxv, uint32_t &ri,
                                                                        PhaseArg3 &state, __ubuf__ float *&tp) {

        if (tp[0] > maxv) {
            maxv = tp[0];
            ri = state.word(tp[1]);
        }
        if constexpr (T > 64U) {
            if (h > 64U && tp[2] > maxv) {
                maxv = tp[2];
                ri = 64U + state.word(tp[3]);
            }
        }
    }

    template <class Self, auto T, auto TS, class PhaseArg7>
    __aicore__ static inline void
    FinishLocalTailGeppShared56constexprPhase6(__ubuf__ float *&ap, __ubuf__ float *&cp, __ubuf__ uint32_t *&ip,
                                               __ubuf__ float *&mp, bool &pendingFar, bool &prepared,
                                               __ubuf__ float *&rp, PhaseArg7 &state, __ubuf__ float *&tp) {

        for (uint32_t f = 0; f < T && !state.singular; ++f) {
            uint32_t h = T - f;
            Finish64Shared21vgatherPhase4<T, TS>(ap, cp, f, h, ip, mp, prepared, state);
            float maxv = 0.0f;
            uint32_t ri = 0U;
            FinishLocalTailGeppShared56wordPhase4<T>(h, maxv, ri, state, tp);
            if (maxv == 0.0f || ri >= h) {
                state.singular = 1;
                break;
            }
            float invAbs = 0.0f;
            if constexpr (Self::MAXV_INVERSE)
                invAbs = -1.0f / maxv;
            const uint32_t p = f + ri;
            float piv = cp[ri];
            if (piv == 0.0f || ((state.word(piv) >> 23U) & 255U) == 255U) {
                state.acc(piv);
                break;
            }
            float inv;
            if constexpr (Self::MAXV_INVERSE)
                inv = state.value(state.word(invAbs) ^ (state.word(piv) & 0x80000000U));
            else
                inv = -1.0f / piv;
            if (((state.word(inv) >> 23U) & 255U) == 255U) {
                state.acc(piv);
                state.singular = 2;
                break;
            }
            if (p != f)
                state.sign = -state.sign;
            FinishLocalTailGeppShared56PipeBarrierPhase2<Self, T, TS>(ap, cp, f, h, inv, ip, mp, p, pendingFar, piv,
                                                                      prepared, rp, state);
        }
    }
    template <class Self> __aicore__ static inline void FinishLocalTailGeppShared56(Self &state) {
        constexpr uint32_t T = Self::NMAX - Self::LdetPolicy::LOCAL_CUT, TS = T + 8U;
        static_assert(Self::PANEL + (T + 8U) * TS * sizeof(float) <= Self::BRC,
                      "retired PANEL plus FAC tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, Self::PANEL, (T + 8U) * TS);
        FinishLocalTailGeppShared56DataCopyPhase3<Self, T, TS>(state, tm);
        auto views = GetTailRawViews51(state, tm);
        bool prepared = false, pendingFar = false;
        FinishLocalTailGeppShared56constexprPhase6<Self, T, TS>(views.ap, views.cp, views.ip, views.mp, pendingFar,
                                                                prepared, views.rp, state, views.tp);
        if constexpr (Self::LdetPolicy::PANEL_ACC)
            state.normalizeProd();
        state.VS();
        state.meta.SetValue(0, state.singular ? 0.0f : state.sign * state.final());
        state.SM();
        DataCopyPad(state.out, state.meta, DataCopyExtParams{1, 4, 0, 0, 0});
        state.MS();
    }
    template <auto T, auto TS, class PhaseArg4, class PhaseArg14, class PhaseArg15, class PhaseArg16, class PhaseArg17>
    __aicore__ static inline void FinishLocalTailBlock2Shared57vgatherPhase1(
        const uint32_t &address0, const uint32_t &address1, __ubuf__ float *&ap, __ubuf__ float *&b0, PhaseArg4 &b1,
        __ubuf__ float *&cp, uint32_t &f, const uint32_t &h2, __ubuf__ uint32_t *&ip, float &m00, float &m01,
        float &m10, float &m11, __ubuf__ float *&mp, PhaseArg14 &r0, PhaseArg15 &r1, PhaseArg16 &t0, PhaseArg17 &t1) {

        vgather(reinterpret_cast<__ubuf__ uint32_t *>(cp), ip + (f & 7U) * T,
                reinterpret_cast<uint64_t>(mp) + (address0 & ~7U) * 4U, 8, 1);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(ap), ip + ((f + 1U) & 7U) * T,
                reinterpret_cast<uint64_t>(mp) + (address1 & ~7U) * 4U, 8, 1);
        LDET_MASK_RST();
        PipeBarrier<PIPE_V>();
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(b0), reinterpret_cast<__ubuf__ uint32_t *>(cp), 1, 8,
              static_cast<uint8_t>((h2 + 7U) / 8U));
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(b1), reinterpret_cast<__ubuf__ uint32_t *>(ap), 1, 8,
              static_cast<uint8_t>((h2 + 7U) / 8U));
        LDET_MASK_SET(T);
        vmuls(r0, mp + f * TS, m00, 1, 1, 1, 8, 8);
        vmuls(r1, mp + (f + 1U) * TS, m11, 1, 1, 1, 8, 8);
        vmuls(t0, mp + (f + 1U) * TS, m01, 1, 1, 1, 8, 8);
        vmuls(t1, mp + f * TS, m10, 1, 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
        vadd(r0, r0, t0, 1, 1, 1, 1, 8, 8, 8);
        vadd(r1, r1, t1, 1, 1, 1, 1, 8, 8, 8);
        LDET_MASK_RST();
        PipeBarrier<PIPE_V>();
        const uint32_t start = (f + 2U) & ~7U, parts = (T - start) / 8U;
        const uint8_t repeats = static_cast<uint8_t>((h2 + 7U) / 8U);
        SetMaskNorm();
        AscendCUtils::SetMask<float>(64);
        vmla(mp + (f + 2U) * TS + start, r0 + start, b0, repeats, TS / 8U, 0, 1, TS, 0, 8);
        PipeBarrier<PIPE_V>();
        vmla(mp + (f + 2U) * TS + start, r1 + start, b1, repeats, TS / 8U, 0, 1, TS, 0, 8);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(4);
        for (uint32_t j = 1U; j < parts; ++j)
            vmla(mp + (f + 2U) * TS + start + j * 8U, r0 + start + j * 8U, b0, repeats, TS / 8U, 0, 1, TS, 0, 8);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 1U; j < parts; ++j)
            vmla(mp + (f + 2U) * TS + start + j * 8U, r1 + start + j * 8U, b1, repeats, TS / 8U, 0, 1, TS, 0, 8);
    }
