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
    template <class Self, bool EXPORT, class PhaseArg10>
    __aicore__ static inline void factorShared28vgatherPhase2(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                                              uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip,
                                                              uint32_t &k, uint32_t &kb, bool &pivotPrepared,
                                                              __ubuf__ float *&pp, PhaseArg10 &state) {

        Shared10SetFlagStage5<EXPORT, Self>(ap, be, cp, f, h, ip, k, kb, pivotPrepared, pp, state);
        if (h > 1)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(cp), 1, 8, static_cast<uint8_t>((h + 7) / 8));
        if constexpr (EXPORT && !Self::LdetPolicy::DIRECT_FAC)
            DataCopy(state.fac[f * state.facPitch(kb)], state.col, ((h + 7U) / 8U) * 8U);
    }
    template <class PhaseArg4>
    __aicore__ static inline void factorShared28GetPhyAddrPhase3(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                                 PhaseArg4 &state) {

        auto records = reinterpret_cast<__ubuf__ uint64_t *>(state.tmp.GetPhyAddr());
        const uint64_t r0 = records[0], r1 = records[1];
        const uint64_t r2 = h > 128U ? records[2] : 0ULL;
        const uint64_t r3 = h > 192U ? records[3] : 0ULL;
        uint64_t selected = 0ULL;
        uint32_t segment = 0U;
        float v = state.value(static_cast<uint32_t>(r0));
        if (v > maxv) {
            maxv = v;
            selected = r0;
        }
        v = state.value(static_cast<uint32_t>(r1));
        if (v > maxv) {
            maxv = v;
            selected = r1;
            segment = 64U;
        }
        v = state.value(static_cast<uint32_t>(r2));
        if (v > maxv) {
            maxv = v;
            selected = r2;
            segment = 128U;
        }
        v = state.value(static_cast<uint32_t>(r3));
        if (v > maxv) {
            maxv = v;
            selected = r3;
            segment = 192U;
        }
        p = k + segment + static_cast<uint32_t>(selected >> 32U);
    }
    template <class Block36Capture7_4>
    __aicore__ static inline void Shared36ScalarUpdateStage8(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                             Block36Capture7_4 &state, __ubuf__ float *&tp, float &v) {
        if (h > 128U) {
            v = tp[4];
            if (v > maxv) {
                maxv = v;
                p = k + 128U + state.word(tp[5]);
            }
        }
        if (h > 192U) {
            v = tp[6];
            if (v > maxv) {
                maxv = v;
                p = k + 192U + state.word(tp[7]);
            }
        }
    }
    template <class PhaseArg4>
    __aicore__ static inline void factorShared28wordPhase4(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                           PhaseArg4 &state, __ubuf__ float *&tp) {

        float v = ReadFirstPivotSegments60(state, h, k, maxv, p, tp);
        Shared36ScalarUpdateStage8(h, k, maxv, p, state, tp, v);
    }
    template <class Block36Capture8_4>
    __aicore__ static inline void Shared36ScalarUpdateStage9(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                             Block36Capture8_4 &state, __ubuf__ float *&tp, float &v) {
        if (h > 128U) {
            v = tp[8];
            if (v > maxv) {
                maxv = v;
                p = k + 128U + state.word(tp[9]);
            }
        }
        if (h > 192U) {
            v = tp[10];
            if (v > maxv) {
                maxv = v;
                p = k + 192U + state.word(tp[11]);
            }
        }
    }
    template <class PhaseArg4>
    __aicore__ static inline void factorShared28wordPhase5(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                           PhaseArg4 &state, __ubuf__ float *&tp) {

        float v = ReadFirstPivotSegments60(state, h, k, maxv, p, tp);
        Shared36ScalarUpdateStage9(h, k, maxv, p, state, tp, v);
    }
    template <class Block36Capture2_4>
    __aicore__ static inline void Shared36ScalarUpdateStage3(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                             Block36Capture2_4 &state, __ubuf__ float *&tp) {
        if (h > 64U) {
            float v = tp[2];
            const uint32_t mask = 0U - static_cast<uint32_t>(v > maxv);
            const uint32_t candidate = k + 64U + state.word(tp[3]);
            p = (p & ~mask) | (candidate & mask);
            maxv = state.value((state.word(maxv) & ~mask) | (state.word(v) & mask));
        }
        if (h > 128U) {
            float v = tp[4];
            const uint32_t mask = 0U - static_cast<uint32_t>(v > maxv);
            const uint32_t candidate = k + 128U + state.word(tp[5]);
            p = (p & ~mask) | (candidate & mask);
            maxv = state.value((state.word(maxv) & ~mask) | (state.word(v) & mask));
        }
    }
    template <class Capture4>
    __aicore__ static inline void SharedScalarUpdateStage20(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                            Capture4 &state, __ubuf__ float *&tp) {
        Shared36ScalarUpdateStage3(h, k, maxv, p, state, tp);
    }
    template <class PhaseArg4>
    __aicore__ static inline void factorShared28wordPhase6(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                           PhaseArg4 &state, __ubuf__ float *&tp) {

        {
            float v = tp[0];
            const uint32_t mask = 0U - static_cast<uint32_t>(v > maxv);
            const uint32_t candidate = k + 0U + state.word(tp[1]);
            p = (p & ~mask) | (candidate & mask);
            maxv = state.value((state.word(maxv) & ~mask) | (state.word(v) & mask));
        }
        SharedScalarUpdateStage20(h, k, maxv, p, state, tp);
        if (h > 192U) {
            float v = tp[6];
            const uint32_t mask = 0U - static_cast<uint32_t>(v > maxv);
            const uint32_t candidate = k + 192U + state.word(tp[7]);
            p = (p & ~mask) | (candidate & mask);
            maxv = state.value((state.word(maxv) & ~mask) | (state.word(v) & mask));
        }
    }
    template <class Self, class PhaseArg2>
    __aicore__ static inline void factorShared28DataCopyPhase7(uint32_t &kb, uint32_t &q, PhaseArg2 &state) {

        state.bindPayloadMeta(kb);
        if constexpr (Self::LdetPolicy::GROUP == 8U && Self::LdetPolicy::PRUNE_STAGE_WAITS) {
            static_assert(Self::B == 32U && Self::S == Self::P, "B4 G8 owns exactly one in-place panel");
        } else if constexpr (Self::LdetPolicy::ACTIVE_PANEL_ROWS) {
            DataCopy(state.panel[kb * Self::P],
                     state.mat[kb * Self::S +
                               (Self::LdetPolicy::CONTIGUOUS_BANDS ? (q - state.bandFirst(state.rank)) : q / Self::G) *
                                   Self::B],
                     {static_cast<uint16_t>(state.n - kb), Self::B / 8U, (Self::S - Self::B) / 8U,
                      (Self::P - Self::B) / 8U});
        } else {
            DataCopy(
                state.panel,
                state.mat[(Self::LdetPolicy::CONTIGUOUS_BANDS ? (q - state.bandFirst(state.rank)) : q / Self::G) *
                          Self::B],
                {static_cast<uint16_t>(state.n), Self::B / 8U, (Self::S - Self::B) / 8U, (Self::P - Self::B) / 8U});
        }
        PipeBarrier<PIPE_V>();
    }
    template <bool EXPORT, class Self, class Capture6>
    __aicore__ static inline void SharedVmulsStage23(uint32_t &f, float &inv, uint32_t &k, uint32_t &p,
                                                     __ubuf__ float *&pp, __ubuf__ float *&rp, Capture6 &state) {
        if (p != k)
            state.sign = -state.sign;
        if constexpr (EXPORT && !Self::LdetPolicy::DEFER_BOOKKEEP) {
            state.mu.SetValue(8 + f, p);
            state.mu.SetValue(8U + Self::B + f, state.word(inv));
        }
        if constexpr (Self::LdetPolicy::NORMAL_COPY)
            AscendCUtils::SetMask<float>(static_cast<int32_t>(Self::B));
        else
            LDET_MASK_SET(Self::B);
        vmuls(rp, pp + p * Self::P, inv, 1, 1, 1, 8, 8);
        if (p != k)
            vmuls(pp + p * Self::P, pp + k * Self::P, 1.0f, 1, 1, 1, 8, 8);
    }
    template <class Self, bool EXPORT, class PhaseArg11>
    __aicore__ static inline void factorShared28vmulsPhase8(uint32_t &f, uint32_t &h, float &inv, uint32_t &k,
                                                            uint32_t &kb, uint32_t &p, float &piv, bool &pivotPrepared,
                                                            __ubuf__ float *&pp, bool &prefixPosted,
                                                            __ubuf__ float *&rp, PhaseArg11 &state) {

        SharedVmulsStage23<EXPORT, Self>(f, inv, k, p, pp, rp, state);
        factorShared28PipeBarrierPhase1<Self, EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, prefixPosted, state);
    }
    template <class Self, bool EXPORT, class PhaseArg2>
    __aicore__ static inline void factorShared28constexprPhase9(uint32_t &kb, bool &prefixPosted, PhaseArg2 &state) {

        if constexpr (Self::LdetPolicy::PANEL_ACC)
            state.normalizeProd();
        if constexpr ((Self::LdetPolicy::GROUP != 5U && !Self::LdetPolicy::PHASE_TRIM) || !Self::LdetPolicy::PANEL_MASK)
            state.VS();
        if constexpr (Self::LdetPolicy::PREFIX_STREAM && EXPORT) {
            if (kb && !prefixPosted)
                state.publishPrefix16(kb);
        }
    }
    template <class Self, class PhaseArg4>
    __aicore__ static inline void factorShared28constexprPhase10(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                                 PhaseArg4 &state, __ubuf__ float *&tp) {

        if constexpr (Self::GUARDED && Self::FAST_OWNER) {
            factorShared28GetPhyAddrPhase3(h, k, maxv, p, state);
        } else if constexpr (Self::GUARDED && Self::BRANCHLESS_SEGMENTS) {
            factorShared28wordPhase6(h, k, maxv, p, state, tp);
        } else if constexpr (Self::GUARDED) {
            factorShared28wordPhase4(h, k, maxv, p, state, tp);
        } else {
            factorShared28wordPhase5(h, k, maxv, p, state, tp);
        }
    }
    template <class Self, bool EXPORT, class LoopArg9>
    __aicore__ static inline bool factorShared28PivotStep(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                          __ubuf__ uint32_t *&ip, uint32_t &kb, bool &pivotPrepared,
                                                          __ubuf__ float *&pp, bool &prefixPosted, __ubuf__ float *&rp,
                                                          LoopArg9 &state, __ubuf__ float *&tp) {

        uint32_t k = kb + f, h = state.n - k, be = k * Self::P + f;
        factorShared28vgatherPhase2<Self, EXPORT>(ap, be, cp, f, h, ip, k, kb, pivotPrepared, pp, state);
        if constexpr (Self::GUARDED || Self::LdetPolicy::PIVOT8_AHEAD)
            WaitFlag<HardEvent::V_S>(2);
        else
            state.VS();
        float maxv = 0;
        uint32_t p = k;
        factorShared28constexprPhase10<Self>(h, k, maxv, p, state, tp);
        if (maxv == 0) {
            state.singular = 1;
            return false;
        }
        const bool normalPivot = Self::FAST_OWNER && maxv >= 1.1754943508222875e-38f && maxv <= 3.4028234663852886e38f;
        float piv = cp[p - k];
        if constexpr (Self::LdetPolicy::DEFER_BOOKKEEP) {
            if (!normalPivot && (piv == 0.0f || ((state.word(piv) >> 23) & 255U) == 255U)) {
                state.acc(piv);
                return false;
            }
        } else {
            state.acc(piv);
            if (state.singular)
                return false;
        }
        float inv = -1.0f / piv;
        if (!normalPivot && ((state.word(inv) >> 23) & 255U) == 255U) {
            if constexpr (Self::LdetPolicy::DEFER_BOOKKEEP)
                state.acc(piv);
            state.singular = 2;
            return false;
        }
        factorShared28vmulsPhase8<Self, EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, pp, prefixPosted, rp, state);

        return true;
    }
    template <class Self, bool EXPORT = true>
    __aicore__ static inline void factorShared28(Self &state, uint32_t kb, uint32_t q) {
        factorShared28DataCopyPhase7<Self>(kb, q, state);
        auto factorViews = GetFactorRawViews60(state);
        bool pivotPrepared = false;
        bool prefixPosted = false;
        for (uint32_t f = 0; f < Self::B; ++f) {
            if (!factorShared28PivotStep<Self, EXPORT>(factorViews.ap, factorViews.cp, f, factorViews.ip, kb,
                                                       pivotPrepared, factorViews.pp, prefixPosted, factorViews.rp,
                                                       state, factorViews.tp))
                break;
        }
        factorShared28constexprPhase9<Self, EXPORT>(kb, prefixPosted, state);
    }
    template <class Self, class Block36Capture0_9>
    __aicore__ static inline void Shared36VmulsStage1(const uint32_t &begin, uint32_t &f, uint32_t &h, float &inv,
                                                      uint32_t &k, uint32_t &kb, __ubuf__ float *&mp, uint32_t &p,
                                                      __ubuf__ float *&rp, Block36Capture0_9 &state,
                                                      const uint32_t &width) {
        for (uint32_t off = 0; off < width; off += 64U) {
            const uint32_t count = width - off < 64U ? width - off : 64U;
            AscendCUtils::SetMask<float>(static_cast<int32_t>(count));
            vmuls(rp + off, mp + p * Self::S + begin + off, inv, 1, 1, 1, 8, 8);
            if (p != k)
                vmuls(mp + p * Self::S + begin + off, mp + k * Self::S + begin + off, 1.0f, 1, 1, 1, 8, 8);
        }
        if (h > 1U)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((h + 7U) / 8U));
        PipeBarrier<PIPE_V>();
        for (uint32_t off = 0; off < width; off += 64U) {
            const uint32_t count = width - off < 64U ? width - off : 64U;
            AscendCUtils::SetMask<float>(static_cast<int32_t>(count));
            if (h > 1U)
                vmla(mp + (k + 1U) * Self::S + begin + off, rp + off,
                     reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()), static_cast<uint8_t>(h - 1U), 1, 1,
                     0, Self::S / 8U, 0, 1);
        }
    }
    template <class Self, uint32_t FIRST, uint32_t LAST, class PhaseArg4>
    __aicore__ static inline void ApplyPanelUpdatesShared29PipeBarrierPhase1(const uint32_t &begin, uint32_t &kb,
                                                                             __ubuf__ float *&mp, __ubuf__ float *&rp,
                                                                             PhaseArg4 &state, const uint32_t &width) {

        SetMaskNorm();
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
            float inv = state.value(state.mu.GetValue(8U + Self::B + f));
            if constexpr (Self::LdetPolicy::GROUP != 5U) {
                if (f)
                    PipeBarrier<PIPE_V>();
            }
            Shared36VmulsStage1<Self>(begin, f, h, inv, k, kb, mp, p, rp, state, width);
            if constexpr (Self::LdetPolicy::GROUP == 5U)
                PipeBarrier<PIPE_V>();
        }
        if constexpr (Self::LdetPolicy::GROUP != 5U)
            PipeBarrier<PIPE_V>();
        ResetMask();
    }
    template <class Self, class Capture8>
    __aicore__ static inline void SharedVmulsStage22(const uint32_t &begin, uint32_t &f, uint32_t &h, uint32_t &k,
                                                     uint32_t &kb, __ubuf__ float *&mp, uint32_t &p,
                                                     __ubuf__ float *&rp, Capture8 &state) {
        if (p != k)
            vmuls(mp + p * Self::S + begin, mp + k * Self::S + begin, 1.0f, 1, 1, 1, 8, 8);
        if (h > 1U)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((h + 7U) / 8U));
        PipeBarrier<PIPE_V>();
        if (h > 1U)
            vmla(mp + (k + 1U) * Self::S + begin, rp, reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()),
                 static_cast<uint8_t>(h - 1U), 1, 1, 0, Self::S / 8U, 0, 1);
    }
    template <class Self, uint32_t FIRST, uint32_t LAST, class PhaseArg2>
    __aicore__ static inline void ApplyPanelUpdatesShared29PipeBarrierPhase2(const uint32_t &begin, uint32_t &kb,
                                                                             PhaseArg2 &state, const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr()),
             rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
        SetMaskNorm();
        AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
            float inv = state.value(state.mu.GetValue(8U + Self::B + f));
            if constexpr (Self::LdetPolicy::GROUP != 5U) {
                if (f)
                    PipeBarrier<PIPE_V>();
            }
            vmuls(rp, mp + p * Self::S + begin, inv, 1, 1, 1, 8, 8);
            SharedVmulsStage22<Self>(begin, f, h, k, kb, mp, p, rp, state);
            if constexpr (Self::LdetPolicy::GROUP == 5U)
                PipeBarrier<PIPE_V>();
        }
        if constexpr (Self::LdetPolicy::GROUP != 5U)
            PipeBarrier<PIPE_V>();
        ResetMask();
        SetMaskNorm();
        if constexpr (Self::LdetPolicy::GROUP != 5U && !Self::LdetPolicy::PHASE_TRIM)
            state.VS();
        return;
    }
