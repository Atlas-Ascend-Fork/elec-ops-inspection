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
    template <class Self, class Capture0, class Capture6, class Capture7>
    __aicore__ static inline void SharedVbrcbStage40(Capture0 &begin, uint32_t &f, uint32_t &h, uint32_t &k,
                                                     uint32_t &kb, uint32_t &p, Capture6 &state, Capture7 &width) {
        if (h > 1)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((h + 7) / 8));
        PipeBarrier<PIPE_V>();
        if constexpr (!Self::LdetPolicy::FAC_REPAIRED) {
            state.repair(p - k);
            PipeBarrier<PIPE_V>();
        }
        state.updateRows(state.mat[(k + 1) * Self::S + begin], state.row, h, Self::S, width);
    }
    template <class Self, uint32_t FIRST, uint32_t LAST, class PhaseArg2>
    __aicore__ static inline void ApplyPanelUpdatesShared29vmulsPhase3(const uint32_t &begin, uint32_t &kb,
                                                                       PhaseArg2 &state, const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr());
        auto rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
            float inv = state.value(state.mu.GetValue(8U + Self::B + f));
            if constexpr (Self::LdetPolicy::NORMAL_COPY)
                AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            else
                LDET_MASK_SET(width);
            vmuls(rp, mp + p * Self::S + begin, inv, 1, 1, 1, 8, 8);
            if (p != k)
                vmuls(mp + p * Self::S + begin, mp + k * Self::S + begin, 1.0f, 1, 1, 1, 8, 8);
            if constexpr (Self::LdetPolicy::NORMAL_COPY)
                ResetMask();
            else
                LDET_MASK_RST();
            SharedVbrcbStage40<Self>(begin, f, h, k, kb, p, state, width);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (!Self::LdetPolicy::PRUNE_STAGE_WAITS)
            state.VS();
    }
    template <class Self, uint32_t FIRST = 0U, uint32_t LAST = Self::B>
    __aicore__ static inline void ApplyPanelUpdatesShared29(Self &state, uint32_t kb, uint32_t q) {
        uint32_t first = FirstOwnedColumn54<Self, Self::LdetPolicy::CONTIGUOUS_BANDS>(state, q);
        const uint32_t begin = first * Self::B,
                       width = state.owned * Self::B > begin ? state.owned * Self::B - begin : 0;
        if (!width)
            return;
        if constexpr (Self::LdetPolicy::HETERO3) {
            auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr()),
                 rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
            ApplyPanelUpdatesShared29PipeBarrierPhase1<Self, FIRST, LAST>(begin, kb, mp, rp, state, width);
            SetMaskNorm();
            state.VS();
            return;
        }
        if constexpr (Self::LdetPolicy::PANEL_MASK) {
            return ApplyPanelUpdatesShared29PipeBarrierPhase2<Self, FIRST, LAST>(begin, kb, state, width);
        }
        ApplyPanelUpdatesShared29vmulsPhase3<Self, FIRST, LAST>(begin, kb, state, width);
    }
    template <class Self> __aicore__ static inline void PrepareTail64IndicesShared30(Self &state) {
        constexpr uint32_t T = 64U, TS = 72U;
        PrepareTailRowIndices46<TS, 32>(state);
        LDET_MASK_SET(T);
        for (uint32_t j = 0; j < 8U; ++j)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[j * T].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1,
                  8, 8);
        LDET_MASK_RST();
    }

    template <class Self, auto T, auto TS, class PhaseArg0, class PhaseArg1>
    __aicore__ static inline void Finish64Shared31DataCopyPhase3(PhaseArg0 &state, PhaseArg1 &tm) {

        DataCopy(tm, state.mat[192U * Self::S], {static_cast<uint16_t>(T), T / 8U, (Self::S - T) / 8U, (TS - T) / 8U});
        PipeBarrier<PIPE_V>();
        if constexpr (!Self::LdetPolicy::EARLY_TAIL_INDEX) {
            SharedPipeBarrierStage21<T, TS>(state);
        }
        PipeBarrier<PIPE_V>();
    }
    template <class Self, class PhaseArg0>
    __aicore__ static inline void Finish64Shared31DataCopyPadPhase4(PhaseArg0 &state) {

        if constexpr (Self::LdetPolicy::PANEL_ACC)
            state.normalizeProd();
        if constexpr ((Self::LdetPolicy::GROUP != 5U && !Self::LdetPolicy::PHASE_TRIM) || !Self::LdetPolicy::PANEL_MASK)
            state.VS();
        state.meta.SetValue(0, state.singular ? 0.0f : state.sign * state.final());
        state.SM();
        DataCopyPad(state.out, state.meta, DataCopyExtParams{1, 4, 0, 0, 0});
        state.MS();
    }

    template <class Self> __aicore__ static inline void Finish64Shared31(Self &state) {
        constexpr uint32_t T = 64U, TS = 72U;
        static_assert((T + 8U) * TS <= Self::P * (Self::NMAX + Self::GUARD_ROWS), "retired PANEL tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, Self::PANEL, (T + 8U) * TS);
        Finish64Shared31DataCopyPhase3<Self, T, TS>(state, tm);
        SharedScalarUpdateStage11<Self, T, TS>(state, tm);
        Finish64Shared31DataCopyPadPhase4<Self>(state);
    }
    template <class Self> __aicore__ static inline GM_ADDR prefixTicketShared32(Self &state, uint32_t q) {
        return state.prefixBase + q * 64U;
    }
    template <class Self, uint32_t FIRST, uint32_t LAST>
    __aicore__ static inline void receiveRangeShared33(Self &state, uint32_t kb) {
        static_assert(FIRST < LAST && LAST <= Self::B, "FAC prefix range");
        SharedDataCopyStage18<FIRST, LAST, Self>(kb, state);
    }
    template <class Self> __aicore__ static inline uint32_t bandFirstShared34(Self &state, uint32_t r) {
        if constexpr (Self::LdetPolicy::HETERO6)
            return r < 6U ? r : 6U;
        else
            return r * 2U;
    }
    template <class Self> __aicore__ static inline void publishShared35(Self &state, uint32_t kb) {
        const uint32_t slot = ((kb / Self::B) & 1U) * Self::GM_WORDS;
        state.bindPayloadMeta(kb);
        state.state();
        state.SM();
        state.VM();
        if constexpr (Self::LdetPolicy::PREFIX_STREAM) {
            DataCopy(state.ws[slot + 16U * state.facPitch(kb)], state.fac[16U * state.facPitch(kb)],
                     16U * state.facPitch(kb) + Self::META_WORDS);
        } else
            DataCopy(state.ws[slot], state.fac, Self::B * state.facPitch(kb) + Self::META_WORDS);
        state.MS();
    }
    template <class Self>
    __aicore__ static inline void updatePanelRows8x8Shared36(Self &state, const LocalTensor<float> &dst,
                                                             const LocalTensor<float> &pivotRow, uint32_t h,
                                                             uint32_t width) {
        SharedVmlaStage2<Self>(dst, h, pivotRow, state, width);
    }
    template <class Self, bool EXPORT = true>
    __aicore__ static inline bool updatePanelAndPrepareShared37(Self &state, uint32_t kb, uint32_t f, uint32_t h,
                                                                uint32_t c0, uint32_t width) {
        if (width < 8U) {
            state.updatePanelRows8x8(state.panel[(kb + f + 1U) * Self::P + c0], state.prow[c0], h, width);
            return false;
        }
        const uint32_t rows = h - 1U, parts = width / 8U, full = rows / 8U, tail = rows % 8U;
        auto dst = reinterpret_cast<__ubuf__ float *>(state.panel[(kb + f + 1U) * Self::P + c0].GetPhyAddr());
        auto pivot = reinterpret_cast<__ubuf__ float *>(state.prow[c0].GetPhyAddr());
        Shared10VmlaStage1<EXPORT, Self>(dst, f, full, kb, parts, pivot, rows, state, tail);
        ResetMask();
        SetMaskNorm();
        return true;
    }

    template <class Self, uint32_t FIRST = 0U, uint32_t LAST = Self::B>
    __aicore__ static inline void ApplyPanelUpdatesShared38(Self &state, uint32_t kb, uint32_t q) {
        uint32_t first = FirstOwnedColumn54<Self, Self::LdetPolicy::CONTIGUOUS_BANDS>(state, q);
        const uint32_t begin = first * Self::B,
                       width = state.owned * Self::B > begin ? state.owned * Self::B - begin : 0;
        if (!width)
            return;
        if constexpr (Self::LdetPolicy::PANEL_MASK) {
            auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr()),
                 rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
            SetMaskNorm();
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            for (uint32_t f = FIRST; f < LAST; ++f) {
                uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
                float inv = state.value(state.mu.GetValue(8U + Self::B + f));
                vmuls(rp, mp + p * Self::S + begin, inv, 1, 1, 1, 8, 8);
                SharedVmulsStage22<Self>(begin, f, h, k, kb, mp, p, rp, state);
                PipeBarrier<PIPE_V>();
            }
            ResetMask();
            SetMaskNorm();
            if constexpr (!Self::LdetPolicy::PRUNE_STAGE_WAITS) {
                if (state.rank != q + 1U)
                    state.VS();
            }
            return;
        }
        ApplyPanelUpdatesShared29vmulsPhase3<Self, FIRST, LAST>(begin, kb, state, width);
    }

    template <class Self> __aicore__ static inline uint32_t bandFirstShared41(const Self &state, uint32_t r) {
        if constexpr (Self::LdetPolicy::SPLIT_OWNERS)
            return r;
        else if constexpr (Self::BALANCED200) {
            if constexpr (Self::LdetPolicy::LOCAL_TAIL)
                return r * 2U;
            else
                return r == 0U ? 0U : 3U + (r - 1U) * 2U;
        } else
            return r * Self::OWNER_BANDS;
    }
    template <class Self> __aicore__ static inline uint32_t bandCountShared42(const Self &state, uint32_t r) {
        if constexpr (Self::LdetPolicy::SPLIT_OWNERS)
            return r == Self::G - 1U ? ((Self::NMAX + Self::B - 1U) / Self::B - (Self::G - 1U)) : 1U;
        else if constexpr (Self::BALANCED200) {
            if constexpr (Self::LdetPolicy::LOCAL_TAIL)
                return r == 2U ? 3U : 2U;
            else
                return r == 0U ? 3U : 2U;
        } else
            return Self::OWNER_BANDS;
    }
    template <class Self> __aicore__ static inline uint32_t ownerOfShared43(const Self &state, uint32_t q) {
        if constexpr (Self::LdetPolicy::SPLIT_OWNERS)
            return q < Self::G - 1U ? q : Self::G - 1U;
        else if constexpr (Self::BALANCED200) {
            if constexpr (Self::LdetPolicy::LOCAL_TAIL)
                return q < 2U ? 0U : (q < 4U ? 1U : 2U);
            else
                return q < 3U ? 0U : (q < 5U ? 1U : 2U);
        } else
            return q / Self::OWNER_BANDS;
    }
    template <class Self> __aicore__ static inline void receiveShared44(Self &state, uint32_t kb) {
        DataCopy(state.mu, state.wu[Self::B * Self::NMAX], Self::META_WORDS);
        DataCopy(state.fac, state.ws, Self::B * state.facPitch(kb));
        state.MV();
        state.VS();
        state.sign = state.value(state.mu.GetValue(0));
        state.prod = state.value(state.mu.GetValue(1));
        state.ex = int32_t(state.mu.GetValue(2));
        state.singular = state.mu.GetValue(3);
    }
    template <class Self> __aicore__ static inline void publishShared45(Self &state, uint32_t kb) {
        state.state();
        state.SM();
        DataCopy(state.wu[Self::B * Self::NMAX], state.mu, Self::META_WORDS);
        if constexpr (!Self::TRIM_PUBLISH)
            state.VM();
        DataCopy(state.ws, state.fac, Self::B * state.facPitch(kb));
        if constexpr (!Self::TRIM_PUBLISH)
            state.MS();
    }
    template <class Self> __aicore__ static inline void publishPrefix16Shared46(Self &state, uint32_t kb) {
        state.state();
        state.SM();
        DataCopy(state.wu[Self::B * Self::NMAX], state.mu, Self::META_WORDS);
        state.VM();
        DataCopy(state.ws, state.fac, 16U * state.facPitch(kb));
        state.MS();
        LuPublish(state.prefixFlag + (Self::LdetPolicy::SPLIT_OWNERS ? kb / Self::B * 64U : (kb == 96U ? 64U : 0U)),
                  0x6A1F0010U);
    }
    template <uint32_t FIRST, uint32_t LAST, class Capture1>
    __aicore__ static inline void Shared24DataCopyStage16(uint32_t &kb, Capture1 &state) {
        DataCopy(state.fac[FIRST * state.facPitch(kb)], state.ws[FIRST * state.facPitch(kb)],
                 (LAST - FIRST) * state.facPitch(kb));
        state.MV();
        state.VS();
        state.sign = state.value(state.mu.GetValue(0));
        state.prod = state.value(state.mu.GetValue(1));
        state.ex = int32_t(state.mu.GetValue(2));
        state.singular = state.mu.GetValue(3);
    }
    template <class Self, uint32_t FIRST, uint32_t LAST>
    __aicore__ static inline void receiveRangeShared47(Self &state, uint32_t kb) {
        static_assert(FIRST < LAST && LAST <= Self::B, "FAC range");
        DataCopy(state.mu, state.wu[Self::B * Self::NMAX], Self::META_WORDS);
        Shared24DataCopyStage16<FIRST, LAST>(kb, state);
    }
    template <class Self> __aicore__ static inline void repairShared48(Self &state, uint32_t pRel) {
        if (pRel) {
            if constexpr (Self::OWNER_NORMAL)
                SetVectorMask<float, MaskMode::NORMAL>(8);
            else
                LDET_MASK_SET(8);
            vmuls(reinterpret_cast<__ubuf__ float *>(state.brc[pRel * 8].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(state.brc.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
            if constexpr (Self::OWNER_NORMAL)
                ResetMask();
            else
                LDET_MASK_RST();
        }
    }
    template <class Self>
    __aicore__ static inline void updateRowsShared49(Self &state, const LocalTensor<float> &dst,
                                                     const LocalTensor<float> &pivotRow, uint32_t h, uint32_t stride,
                                                     uint32_t width) {
        if (h <= 1 || !width)
            return;
        SetMaskNorm();
        if constexpr (Self::LdetPolicy::GROUP == 2U ||
                      (Self::LdetPolicy::LOCAL_TAIL && Self::LdetPolicy::DIM == 200U)) {
            Shared24VmlaStage1(dst, h, pivotRow, state, stride, width);
        } else {
            Shared24VmlaStage2(dst, h, pivotRow, state, stride, width);
        }
        ResetMask();
        SetMaskNorm();
    }
    template <class Self>
    __aicore__ static inline void updatePanelRows8x8Shared50(Self &state, const LocalTensor<float> &dst,
                                                             const LocalTensor<float> &pivotRow, uint32_t h,
                                                             uint32_t width) {
        if (h < 64U) {
            state.updateRows(dst, pivotRow, h, Self::P, width);
            return;
        }
        const uint32_t rows = h - 1U, full = rows / 8U, tail = rows % 8U, parts = width / 8U;
        SetMaskNorm();
        AscendCUtils::SetMask<float>(64);
        if constexpr ((Self::LdetPolicy::DIM == 192U || Self::LdetPolicy::DIM == 200U ||
                       Self::LdetPolicy::DIM == 208U)) {
            Shared24VmlaStage14<Self>(dst, parts, pivotRow, rows, state);
        } else {
            Shared10PipeBarrierStage2<Self>(dst, full, parts, pivotRow, state, tail);
        }
        ResetMask();
        SetMaskNorm();
    }
    template <class Self, class Capture0, class Capture1>
    __aicore__ static inline void SharedScalarUpdateStage27(Capture0 &h, Capture1 &state) {
        LDET_MASK_RST();
        LDET_MASK_SET(h);
        if constexpr (Self::LdetPolicy::FIXED_REDUCE) {
            const uint32_t head = h < 128U ? h : 128U;
            AscendCUtils::SetMask<float>(0, head);
            LdetVcmax128(state.tmp[0U], state.ab[0U], head);
            if (h > 128U) {
                const uint32_t tail = h - 128U;
                AscendCUtils::SetMask<float>(0, tail);
                LdetVcmax128(state.tmp[8U], state.ab[128U], tail);
            }
        } else {
            for (uint32_t c = 0, off = 0; c < (h + 127) / 128; ++c, off += 128) {
                uint32_t count = h - off < 128 ? h - off : 128;
                AscendCUtils::SetMask<float>(0, count);
                LdetVcmax128(state.tmp[c * 8], state.ab[off], count);
            }
        }
    }
    template <class Self>
    __aicore__ static inline void issuePreparedPivotShared51(Self &state, uint32_t k, uint32_t f) {
        const uint32_t h = state.n - k, be = k * Self::P + f;
        auto cp = reinterpret_cast<__ubuf__ float *>(state.col.GetPhyAddr());
        auto ap = reinterpret_cast<__ubuf__ float *>(state.ab.GetPhyAddr());
        auto pp = reinterpret_cast<__ubuf__ float *>(state.panel.GetPhyAddr());
        auto ip = reinterpret_cast<__ubuf__ uint32_t *>(state.lanes.GetPhyAddr());
        LDET_MASK_SET(h);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(cp), ip + (f & 7U) * Self::NMAX,
                reinterpret_cast<uint64_t>(pp) + (be & ~7U) * 4U, 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(ap, cp, 1, 1, 1, 8, 8);
        if constexpr (Self::TUNED_PANEL) {
            LdetVcmax128(state.tmp, state.ab, h);
        } else {
            SharedScalarUpdateStage27<Self>(h, state);
        }
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(2);
    }
