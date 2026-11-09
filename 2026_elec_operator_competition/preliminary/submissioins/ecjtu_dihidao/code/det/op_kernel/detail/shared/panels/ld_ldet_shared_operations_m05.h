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
    template <class Self>
    __aicore__ static inline bool updatePanelAndPrepareShared52(Self &state, uint32_t kb, uint32_t f, uint32_t h,
                                                                uint32_t c0, uint32_t width) {
        if ((Self::TUNED_PANEL ? (h < 2U || width < 8U) : (h < 64U || width < 16U)) || f + 1U >= Self::B ||
            kb + f + 1U >= state.n) {
            state.updatePanelRows8x8(state.panel[(kb + f + 1U) * Self::P + c0], state.prow[c0], h, width);
            return false;
        }
        const uint32_t rows = h - 1U, parts = width / 8U;
        auto dst = reinterpret_cast<__ubuf__ float *>(state.panel[(kb + f + 1U) * Self::P + c0].GetPhyAddr());
        auto pivot = reinterpret_cast<__ubuf__ float *>(state.prow[c0].GetPhyAddr());
        auto coeff = reinterpret_cast<__ubuf__ float *>(state.brc[8].GetPhyAddr());
        if constexpr (!Self::OWNER_NORMAL) {
            SetMaskNorm();
            AscendCUtils::SetMask<float>(64);
        }
        vmla(dst, pivot, coeff, static_cast<uint8_t>((rows + 7U) / 8U), Self::P / 8U, 0, 1, Self::P, 0, 8);
        PipeBarrier<PIPE_V>();
        state.issuePreparedPivot(kb + f + 1U, f + 1U);
        if constexpr (Self::FAR_STATIC) {
            if (width > 8U) {
                if constexpr (!Self::OWNER_NORMAL)
                    AscendCUtils::SetMask<float>(64);
                const uint8_t repeats = static_cast<uint8_t>((rows + 7U) / 8U);
                vmla(dst + 8U, pivot + 8U, coeff, repeats, Self::P / 8U, 0, 1, Self::P, 0, 8);
                if (width > 16U)
                    vmla(dst + 16U, pivot + 16U, coeff, repeats, Self::P / 8U, 0, 1, Self::P, 0, 8);
                if (width > 24U)
                    vmla(dst + 24U, pivot + 24U, coeff, repeats, Self::P / 8U, 0, 1, Self::P, 0, 8);
                if constexpr (!Self::OWNER_NORMAL) {
                    ResetMask();
                    SetMaskNorm();
                }
            }
        } else {
            AscendCUtils::SetMask<float>(64);
            for (uint32_t j = 1; j < parts; ++j)
                vmla(dst + j * 8U, pivot + j * 8U, coeff, static_cast<uint8_t>((rows + 7U) / 8U), Self::P / 8U, 0, 1,
                     Self::P, 0, 8);
            ResetMask();
            SetMaskNorm();
        }
        return true;
    }
    template <class Self, bool EXPORT, class PhaseArg10>
    __aicore__ static inline void
    factorShared53vmulsPhase1(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb, uint32_t &p, float &piv,
                              bool &pivotPrepared, __ubuf__ float *&pp, __ubuf__ float *&rp, PhaseArg10 &state) {

        vmuls(rp, pp + p * Self::P, inv, 1, 1, 1, 8, 8);
        if (p != k)
            vmuls(pp + p * Self::P, pp + k * Self::P, 1.0f, 1, 1, 1, 8, 8);
        if constexpr (Self::OWNER_NORMAL)
            ResetMask();
        else
            LDET_MASK_RST();
        if constexpr (!Self::EARLY_REPAIR) {
            if constexpr ((Self::LdetPolicy::DIM == 192U || Self::LdetPolicy::DIM == 200U ||
                           Self::LdetPolicy::DIM == 208U)) {
                if (p != k)
                    PipeBarrier<PIPE_V>();
            }
            state.repair(p - k);
        }
        PipeBarrier<PIPE_V>();
        uint32_t c0 = (f + 1) & ~7U, w = Self::B - c0;
        if constexpr (Self::LdetPolicy::FAC_REPAIRED && EXPORT)
            state.patchFac(f, kb, p - k);
        if constexpr (Self::PIVOT_AHEAD) {
            pivotPrepared = w ? state.updatePanelAndPrepare(kb, f, h, c0, w) : false;
        } else {
            if (w)
                state.updatePanelRows8x8(state.panel[(k + 1) * Self::P + c0], state.prow[c0], h, w);
        }
        Shared24PipeBarrierStage17<EXPORT, Self>(f, inv, p, piv, state);
    }
    template <class Self, class PhaseArg9>
    __aicore__ static inline void factorShared53vgatherPhase2(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                                              uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip,
                                                              uint32_t &k, bool &pivotPrepared, __ubuf__ float *&pp,
                                                              PhaseArg9 &state) {

        if constexpr (Self::PIVOT_AHEAD) {
            if (!pivotPrepared)
                state.issuePreparedPivot(k, f);
        } else {
            LDET_MASK_SET(h);
            vgather(reinterpret_cast<__ubuf__ uint32_t *>(cp), ip + (f & 7U) * Self::NMAX,
                    reinterpret_cast<uint64_t>(pp) + (be & ~7U) * 4U, 8, 1);
            PipeBarrier<PIPE_V>();
            vabs(ap, cp, 1, 1, 1, 8, 8);
            SharedScalarUpdateStage27<Self>(h, state);
            LDET_MASK_RST();
        }
        if constexpr (!Self::PIVOT_AHEAD &&
                      (Self::LdetPolicy::DIM == 192U || Self::LdetPolicy::DIM == 200U || Self::LdetPolicy::DIM == 208U))
            SetFlag<HardEvent::V_S>(2);
        if (h > 1)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(cp), 1, 8, static_cast<uint8_t>((h + 7) / 8));
    }
    template <class Capture4>
    __aicore__ static inline void Shared24ScalarUpdateStage20(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                              Capture4 &state, __ubuf__ float *&tp) {
        {
            float v = ReadPivotSegment64(state, tp, 0U, k, maxv, p);
        }
        if (h > 64U) {
            float v = ReadPivotSegment64(state, tp, 2U, k + 64U, maxv, p);
        }
    }
    template <class Self, class PhaseArg4>
    __aicore__ static inline void factorShared53wordPhase3(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                           PhaseArg4 &state, __ubuf__ float *&tp) {

        Shared24ScalarUpdateStage20(h, k, maxv, p, state, tp);
        if (h > 128U) {
            float v = tp[4];
            if (v > maxv) {
                maxv = v;
                p = k + 128U + state.word(tp[5]);
            }
        }
        if constexpr (Self::NMAX > 192U) {
            if (h > 192U) {
                float v = tp[6];
                if (v > maxv) {
                    maxv = v;
                    p = k + 192U + state.word(tp[7]);
                }
            }
        }
    }
    template <class Self, class PhaseArg4>
    __aicore__ static inline void factorShared53wordPhase4(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                           PhaseArg4 &state, __ubuf__ float *&tp) {

        Shared24ScalarUpdateStage20(h, k, maxv, p, state, tp);
        if (h > 128U) {
            float v = tp[8];
            if (v > maxv) {
                maxv = v;
                p = k + 128U + state.word(tp[9]);
            }
        }
        if constexpr (Self::NMAX > 192U) {
            if (h > 192U) {
                float v = tp[10];
                if (v > maxv) {
                    maxv = v;
                    p = k + 192U + state.word(tp[11]);
                }
            }
        }
    }
    template <class Self, bool EXPORT, class PhaseArg12>
    __aicore__ static inline void
    factorShared53DataCopyPhase5(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb, uint32_t &p,
                                 float &piv, bool &pivotPrepared, __ubuf__ float *&pp, bool &prefixPosted, uint32_t &q,
                                 __ubuf__ float *&rp, PhaseArg12 &state) {

        if (p != k)
            state.sign = -state.sign;
        if constexpr (EXPORT && !Self::LdetPolicy::DEFER_BOOKKEEP) {
            state.mu.SetValue(8 + f, p);
            state.mu.SetValue(8U + Self::B + f, state.word(inv));
        }
        if constexpr (EXPORT && !Self::LdetPolicy::EARLY_FAC)
            DataCopy(state.fac[f * state.facPitch(kb)], state.col, ((h + 7) / 8) * 8);
        if constexpr (Self::OWNER_NORMAL)
            SetVectorMask<float, MaskMode::NORMAL>(static_cast<int32_t>(Self::B));
        else
            LDET_MASK_SET(Self::B);
        factorShared53vmulsPhase1<Self, EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, pp, rp, state);
        if constexpr (Self::LdetPolicy::PREFIX_STREAM) {
            if ((Self::LdetPolicy::SPLIT_OWNERS || q == 1U || q == 3U) && f == 15U) {
                state.publishPrefix16(kb);
                prefixPosted = true;
            }
        }
    }
    template <class Self, class PhaseArg4>
    __aicore__ static inline void factorShared53wordPhase6(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                           PhaseArg4 &state, __ubuf__ float *&tp) {

        {
            float v = tp[0];
            const uint32_t mask = 0U - static_cast<uint32_t>(v > maxv);
            const uint32_t candidate = k + state.word(tp[1]);
            p = (p & ~mask) | (candidate & mask);
            maxv = state.value((state.word(maxv) & ~mask) | (state.word(v) & mask));
        }
        SharedScalarUpdateStage20(h, k, maxv, p, state, tp);
        if constexpr (Self::NMAX > 192U) {
            if (h > 192U) {
                float v = tp[6];
                const uint32_t mask = 0U - static_cast<uint32_t>(v > maxv);
                const uint32_t candidate = k + 192U + state.word(tp[7]);
                p = (p & ~mask) | (candidate & mask);
                maxv = state.value((state.word(maxv) & ~mask) | (state.word(v) & mask));
            }
        }
    }
    template <class Depth22Arg1_4>
    __aicore__ static inline void
    factorShared53constexprPhase7LDET_RM_PAIR_OFFDepth22Stage1(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                               Depth22Arg1_4 &state, __ubuf__ float *&tp) {

        for (uint32_t s = 0; s < (h + 63) / 64; ++s) {
            uint32_t so = LDET_RM_PAIR_OFF(s);
            float v = tp[so];
            if (v > maxv) {
                maxv = v;
                p = k + s * 64 + state.word(tp[so + 1]);
            }
        }
    }
    template <class Self, class PhaseArg4>
    __aicore__ static inline void factorShared53constexprPhase7(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                                PhaseArg4 &state, __ubuf__ float *&tp) {

        if constexpr (Self::TUNED_PANEL && Self::LdetPolicy::WORD_SELECT) {
            factorShared53wordPhase6<Self>(h, k, maxv, p, state, tp);
        } else if constexpr (Self::TUNED_PANEL) {
            factorShared53wordPhase3<Self>(h, k, maxv, p, state, tp);
        } else if constexpr (Self::LdetPolicy::FIXED_MERGE) {
            factorShared53wordPhase4<Self>(h, k, maxv, p, state, tp);
        } else {
            factorShared53constexprPhase7LDET_RM_PAIR_OFFDepth22Stage1(h, k, maxv, p, state, tp);
        }
    }
    template <class Self, class PhaseArg1>
    __aicore__ static inline void factorShared53DataCopyPhase8(uint32_t &q, PhaseArg1 &state) {

        if constexpr (Self::LdetPolicy::INPLACE_PANEL) {
            state.panel = state.mat[(Self::LdetPolicy::CONTIGUOUS_BANDS
                                         ? (Self::BALANCED200 ? q - state.bandFirst(state.rank) : q % Self::OWNER_BANDS)
                                         : q / Self::G) *
                                    Self::B];
        } else {
            DataCopy(
                state.panel,
                state.mat[(Self::LdetPolicy::CONTIGUOUS_BANDS
                               ? (Self::BALANCED200 ? q - state.bandFirst(state.rank) : q % Self::OWNER_BANDS)
                               : q / Self::G) *
                          Self::B],
                {static_cast<uint16_t>(state.n), Self::B / 8U, (Self::S - Self::B) / 8U, (Self::P - Self::B) / 8U});
            PipeBarrier<PIPE_V>();
        }
    }
    template <class Self, class PhaseArg3>
    __aicore__ static inline void factorShared53constexprPhase9(uint32_t &kb, bool &prefixPosted, uint32_t &q,
                                                                PhaseArg3 &state) {

        if constexpr (Self::LdetPolicy::PANEL_ACC)
            state.normalizeProd();
        state.VS();
        if constexpr (Self::LdetPolicy::PREFIX_STREAM) {
            if ((Self::LdetPolicy::SPLIT_OWNERS || q == 1U || q == 3U) && !prefixPosted)
                state.publishPrefix16(kb);
        }
    }
    template <class Self, bool EXPORT, class PhaseArg10>
    __aicore__ static inline void factorShared53DataCopyPhase10(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                                                uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip,
                                                                uint32_t &k, uint32_t &kb, bool &pivotPrepared,
                                                                __ubuf__ float *&pp, PhaseArg10 &state) {

        factorShared53vgatherPhase2<Self>(ap, be, cp, f, h, ip, k, pivotPrepared, pp, state);
        if constexpr (Self::LdetPolicy::EARLY_FAC && EXPORT)
            DataCopy(state.fac[f * state.facPitch(kb)], state.col, ((h + 7) / 8) * 8);
        if constexpr ((Self::LdetPolicy::DIM == 192U || Self::LdetPolicy::DIM == 200U || Self::LdetPolicy::DIM == 208U))
            WaitFlag<HardEvent::V_S>(2);
        else
            state.VS();
    }
    template <class Self, class PhaseArg2>
    __aicore__ static inline void factorShared53PipeBarrierPhase11(uint32_t &k, uint32_t &p, PhaseArg2 &state) {

        if constexpr (Self::EARLY_REPAIR) {
            if (p != k) {
                PipeBarrier<PIPE_V>();
                state.repair(p - k);
            }
        }
    }
    template <class Self, bool EXPORT, class LoopArg10>
    __aicore__ static inline bool factorShared53PivotStep(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                          __ubuf__ uint32_t *&ip, uint32_t &kb, bool &pivotPrepared,
                                                          __ubuf__ float *&pp, bool &prefixPosted, uint32_t &q,
                                                          __ubuf__ float *&rp, LoopArg10 &state, __ubuf__ float *&tp) {

        uint32_t k = kb + f, h = state.n - k, be = k * Self::P + f;
        factorShared53DataCopyPhase10<Self, EXPORT>(ap, be, cp, f, h, ip, k, kb, pivotPrepared, pp, state);
        float maxv = 0;
        uint32_t p = k;
        factorShared53constexprPhase7<Self>(h, k, maxv, p, state, tp);
        if (maxv == 0) {
            state.singular = 1;
            return false;
        }
        float invAbs = 0.0f;
        if constexpr (Self::MAXV_INVERSE)
            invAbs = -1.0f / maxv;
        float piv = cp[p - k];
        factorShared53PipeBarrierPhase11<Self>(k, p, state);
        if constexpr (Self::LdetPolicy::DEFER_BOOKKEEP) {
            if (piv == 0.0f || ((state.word(piv) >> 23) & 255U) == 255U) {
                state.acc(piv);
                return false;
            }
        } else {
            state.acc(piv);
            if (state.singular)
                return false;
        }
        float inv;
        if constexpr (Self::MAXV_INVERSE)
            inv = state.value(state.word(invAbs) ^ (state.word(piv) & 0x80000000U));
        else
            inv = -1.0f / piv;
        if (((state.word(inv) >> 23) & 255U) == 255U) {
            if constexpr (Self::LdetPolicy::DEFER_BOOKKEEP)
                state.acc(piv);
            state.singular = 2;
            return false;
        }
        factorShared53DataCopyPhase5<Self, EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, pp, prefixPosted, q, rp,
                                                   state);

        return true;
    }
    template <class Self, bool EXPORT = true>
    __aicore__ static inline void factorShared53(Self &state, uint32_t kb, uint32_t q) {
        factorShared53DataCopyPhase8<Self>(q, state);
        auto factorViews = GetFactorRawViews60(state);
        bool pivotPrepared = false;
        bool prefixPosted = false;
        for (uint32_t f = 0; f < Self::B && kb + f < state.n; ++f) {
            if (!factorShared53PivotStep<Self, EXPORT>(factorViews.ap, factorViews.cp, f, factorViews.ip, kb,
                                                       pivotPrepared, factorViews.pp, prefixPosted, q, factorViews.rp,
                                                       state, factorViews.tp))
                break;
        }
        factorShared53constexprPhase9<Self>(kb, prefixPosted, q, state);
    }
