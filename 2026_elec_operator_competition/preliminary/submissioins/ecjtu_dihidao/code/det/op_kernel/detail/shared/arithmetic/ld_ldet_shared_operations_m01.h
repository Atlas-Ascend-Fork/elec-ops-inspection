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
    template <class Self> __aicore__ static inline void VSShared0(Self &state) {
        SetFlag<HardEvent::V_S>(2);
        WaitFlag<HardEvent::V_S>(2);
    }
    template <class Self> __aicore__ static inline void SVShared1(Self &state) {
        SetFlag<HardEvent::S_V>(3);
        WaitFlag<HardEvent::S_V>(3);
    }
    template <class Self> __aicore__ static inline void MVShared2(Self &state) {
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
    }
    template <class Self> __aicore__ static inline void VMShared3(Self &state) {
        SetFlag<HardEvent::V_MTE3>(6);
        WaitFlag<HardEvent::V_MTE3>(6);
    }
    template <class Self> __aicore__ static inline void SMShared4(Self &state) {
        SetFlag<HardEvent::S_MTE3>(3);
        WaitFlag<HardEvent::S_MTE3>(3);
    }
    template <class Self> __aicore__ static inline void MSShared5(Self &state) {
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }
    template <class Self> __aicore__ static inline uint32_t wordShared6(Self &state, float f) {
        union {
            float f;
            uint32_t u;
        } x;
        x.f = f;
        return x.u;
    }
    template <class Self> __aicore__ static inline float valueShared7(Self &state, uint32_t u) {
        union {
            float f;
            uint32_t u;
        } x;
        x.u = u;
        return x.f;
    }
    template <class Self> __aicore__ static inline void normalizeProdShared8(Self &state) {
        float a = state.prod < 0 ? -state.prod : state.prod;
        if (a >= ::kTwoPow60) {
            state.prod *= ::kTwoPowNeg64;
            state.ex += 64;
        } else if (a != 0 && a < ::kTwoPowNeg63) {
            state.prod *= ::kTwoPow64;
            state.ex -= 64;
        }
    }
    template <class Self> __aicore__ static inline void accShared9(Self &state, float piv) {
        uint32_t u = state.word(piv), ef = (u >> 23) & 255U;
        if (ef == 255U) {
            state.singular = 2;
            state.prod = 0;
            state.ex = 0;
            return;
        }
        if (piv == 0) {
            state.singular = 1;
            state.prod = 0;
            state.ex = 0;
            return;
        }
        float m;
        int32_t e;
        if (ef >= 2 && ef <= 253) {
            e = int32_t(ef) - 126;
            m = state.value((u & 0x807fffffU) | 0x3f000000U);
            if constexpr (Self::LdetPolicy::PANEL_ACC) {
                state.prod *= m;
                state.ex += e;
                return;
            }
        } else {
            if constexpr (Self::LdetPolicy::PANEL_ACC)
                state.normalizeProd();
            m = piv;
            e = 0;
            float a = m < 0 ? -m : m;
            while (a < 1) {
                m *= ::kTwoPow64;
                e -= 64;
                a *= ::kTwoPow64;
            }
            while (a >= 2) {
                m *= ::kTwoPowNeg64;
                e += 64;
                a *= ::kTwoPowNeg64;
            }
        }
        state.prod *= m;
        state.ex += e;
        state.normalizeProd();
    }
    template <class Self> __aicore__ static inline float finalShared10(Self &state) {
        float p = state.prod;
        int32_t e = state.ex;
        while (e >= 64) {
            p *= ::kTwoPow64;
            e -= 64;
        }
        while (e <= -64) {
            p *= ::kTwoPowNeg64;
            e += 64;
        }
        if (e)
            p *= state.value(uint32_t(127 + e) << 23);
        return p;
    }
    template <class Self> __aicore__ static inline void stateShared11(Self &state) {
        state.mu.SetValue(0, state.word(state.sign));
        state.mu.SetValue(1, state.word(state.prod));
        state.mu.SetValue(2, uint32_t(state.ex));
        state.mu.SetValue(3, state.singular);
    }
    template <class Self> __aicore__ static inline uint32_t facPitchShared12(Self &state, uint32_t kb) {
        if constexpr (Self::LdetPolicy::COMPACT)
            return Self::NMAX - kb;
        else
            return Self::NMAX;
    }
    template <class Self> __aicore__ static inline void bindPayloadMetaShared13(Self &state, uint32_t kb) {
        state.meta =
            LocalTensor<float>(TPosition::VECCALC, Self::FAC + Self::B * state.facPitch(kb) * 4U, Self::META_WORDS);
        state.mu = state.meta.template ReinterpretCast<uint32_t>();
    }
    template <class Self> __aicore__ static inline void receiveShared14(Self &state, uint32_t kb) {
        const uint32_t slot = ((kb / Self::B) & 1U) * Self::GM_WORDS;
        state.bindPayloadMeta(kb);
        DataCopy(state.fac, state.ws[slot], Self::B * state.facPitch(kb) + Self::META_WORDS);
        state.MV();
        state.VS();
        state.sign = state.value(state.mu.GetValue(0));
        state.prod = state.value(state.mu.GetValue(1));
        state.ex = int32_t(state.mu.GetValue(2));
        state.singular = state.mu.GetValue(3);
    }
    template <class Self> __aicore__ static inline void repairShared15(Self &state, uint32_t pRel) {
        if (pRel) {
            if constexpr (Self::LdetPolicy::NORMAL_COPY)
                AscendCUtils::SetMask<float>(8);
            else
                LDET_MASK_SET(8);
            vmuls(reinterpret_cast<__ubuf__ float *>(state.brc[pRel * 8].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(state.brc.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
            if constexpr (Self::LdetPolicy::NORMAL_COPY)
                ResetMask();
            else
                LDET_MASK_RST();
        }
    }
    template <class Self>
    __aicore__ static inline void patchFacShared16(Self &state, uint32_t f, uint32_t kb, uint32_t pRel) {
        if (pRel) {
            SetVectorMask<float, MaskMode::NORMAL>(0ULL, 1ULL << (pRel & 7U));
            vmuls(reinterpret_cast<__ubuf__ float *>(state.fac[f * state.facPitch(kb) + (pRel & ~7U)].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(state.brc.GetPhyAddr()), 1.0f, 1, 1, 1, 8, 8);
            ResetMask();
        }
    }
    template <class Capture3>
    __aicore__ static inline void Shared24VmlaStage1(const LocalTensor<float> &dst, uint32_t &h,
                                                     const LocalTensor<float> &pivotRow, Capture3 &state,
                                                     uint32_t &stride, uint32_t &width) {
        for (uint32_t off = 0U; off < width; off += 64U) {
            const uint32_t count = width - off < 64U ? width - off : 64U;
            AscendCUtils::SetMask<float>(static_cast<int32_t>(count));
            vmla(reinterpret_cast<__ubuf__ float *>(dst[off].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(pivotRow[off].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(state.brc[8].GetPhyAddr()), static_cast<uint8_t>(h - 1), 1, 1, 0,
                 static_cast<uint8_t>(stride / 8), 0, 1);
        }
    }
    template <class Capture3>
    __aicore__ static inline void Shared24VmlaStage2(const LocalTensor<float> &dst, uint32_t &h,
                                                     const LocalTensor<float> &pivotRow, Capture3 &state,
                                                     uint32_t &stride, uint32_t &width) {
        AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
        vmla(reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr()),
             reinterpret_cast<__ubuf__ float *>(pivotRow.GetPhyAddr()),
             reinterpret_cast<__ubuf__ float *>(state.brc[8].GetPhyAddr()), static_cast<uint8_t>(h - 1), 1, 1, 0,
             static_cast<uint8_t>(stride / 8), 0, 1);
    }
    template <class Self>
    __aicore__ static inline void updateRowsShared17(Self &state, const LocalTensor<float> &dst,
                                                     const LocalTensor<float> &pivotRow, uint32_t h, uint32_t stride,
                                                     uint32_t width) {
        if (h <= 1 || !width)
            return;
        SetMaskNorm();
        if constexpr (Self::LdetPolicy::GROUP == 2U) {
            Shared24VmlaStage1(dst, h, pivotRow, state, stride, width);
        } else {
            Shared24VmlaStage2(dst, h, pivotRow, state, stride, width);
        }
        ResetMask();
        SetMaskNorm();
    }
    template <class Self, class Capture4>
    __aicore__ static inline void Shared10PipeBarrierStage2(const LocalTensor<float> &dst, const uint32_t &full,
                                                            const uint32_t &parts, const LocalTensor<float> &pivotRow,
                                                            Capture4 &state, const uint32_t &tail) {
        vadds(reinterpret_cast<__ubuf__ int32_t *>(state.tmp.GetPhyAddr()),
              reinterpret_cast<__ubuf__ int32_t *>(pivotRow.GetPhyAddr()), int32_t(0), static_cast<uint8_t>(parts), 1,
              0, 8, 1);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < parts; ++j) {
            AscendCUtils::SetMask<float>(64);
            vmla(reinterpret_cast<__ubuf__ float *>(dst[j * 8U].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(state.tmp[j * 64U].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(state.brc[8].GetPhyAddr()), static_cast<uint8_t>(full),
                 Self::P / 8U, 1, 1, Self::P, 0, 8);
            if (tail) {
                AscendCUtils::SetMask<float>(static_cast<int32_t>(tail * 8U));
                vmla(reinterpret_cast<__ubuf__ float *>(dst[full * 8U * Self::P + j * 8U].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(state.tmp[j * 64U].GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(state.brc[8U + full * 64U].GetPhyAddr()), 1, Self::P / 8U, 1, 1,
                     Self::P, 0, 8);
            }
        }
    }
    template <class Self, class Capture4>
    __aicore__ static inline void Shared24VmlaStage14(const LocalTensor<float> &dst, const uint32_t &parts,
                                                      const LocalTensor<float> &pivotRow, const uint32_t &rows,
                                                      Capture4 &state) {
        for (uint32_t j = 0; j < parts; ++j)
            vmla(reinterpret_cast<__ubuf__ float *>(dst[j * 8U].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(pivotRow[j * 8U].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(state.brc[8].GetPhyAddr()), static_cast<uint8_t>((rows + 7U) / 8U),
                 Self::P / 8U, 0, 1, Self::P, 0, 8);
    }
    template <class Self, class Capture3>
    __aicore__ static inline void SharedVmlaStage2(const LocalTensor<float> &dst, uint32_t &h,
                                                   const LocalTensor<float> &pivotRow, Capture3 &state,
                                                   uint32_t &width) {
        const uint32_t rows = h - 1U, full = rows / 8U, tail = rows % 8U, parts = width / 8U;
        SetMaskNorm();
        AscendCUtils::SetMask<float>(64);
        if constexpr (Self::GUARDED) {
            Shared24VmlaStage14<Self>(dst, parts, pivotRow, rows, state);
        } else {
            Shared10PipeBarrierStage2<Self>(dst, full, parts, pivotRow, state, tail);
        }
        ResetMask();
        SetMaskNorm();
    }
    template <class Self>
    __aicore__ static inline void updatePanelRows8x8Shared18(Self &state, const LocalTensor<float> &dst,
                                                             const LocalTensor<float> &pivotRow, uint32_t h,
                                                             uint32_t width) {
        if (h < 64U) {
            state.updateRows(dst, pivotRow, h, Self::P, width);
            return;
        }
        SharedVmlaStage2<Self>(dst, h, pivotRow, state, width);
    }
    template <class Self, class Block36Capture1_1, class Block36Capture1_4, class Block36Capture1_7>
    __aicore__ static inline void Shared36VgatherStage2(__ubuf__ float *&ap, Block36Capture1_1 &be, __ubuf__ float *&cp,
                                                        uint32_t &f, Block36Capture1_4 &h, __ubuf__ uint32_t *&ip,
                                                        __ubuf__ float *&pp, Block36Capture1_7 &state) {
        LDET_MASK_SET(h);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(cp), ip + (f & 7U) * Self::NMAX,
                reinterpret_cast<uint64_t>(pp) + (be & ~7U) * 4U, 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(ap, cp, 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        LDET_MASK_SET(h);
        if constexpr (Self::GUARDED) {
            LdetVcmax128(state.tmp, state.ab, h);
        } else {
            const uint32_t head = h < 128U ? h : 128U;
            AscendCUtils::SetMask<float>(0, head);
            LdetVcmax128(state.tmp, state.ab, head);
            if (h > 128U) {
                const uint32_t tail = h - 128U;
                AscendCUtils::SetMask<float>(0, tail);
                LdetVcmax128(state.tmp[8], state.ab[128], tail);
            }
        }
        LDET_MASK_RST();
    }
    template <class Self, class Capture1, class Capture4, class Capture7>
    __aicore__ static inline void Shared10VgatherStage4(__ubuf__ float *&ap, Capture1 &be, __ubuf__ float *&cp,
                                                        uint32_t &f, Capture4 &h, __ubuf__ uint32_t *&ip,
                                                        __ubuf__ float *&pp, Capture7 &state) {
        Shared36VgatherStage2<Self>(ap, be, cp, f, h, ip, pp, state);
    }
    template <class Self, bool EXPORT = true>
    __aicore__ static inline void issuePreparedPivotShared19(Self &state, uint32_t k, uint32_t f) {
        const uint32_t h = state.n - k, be = k * Self::P + f;
        auto cp = reinterpret_cast<__ubuf__ float *>(state.col.GetPhyAddr());
        if constexpr (Self::LdetPolicy::DIRECT_FAC && EXPORT)
            cp = reinterpret_cast<__ubuf__ float *>(state.fac[f * state.facPitch(k - f)].GetPhyAddr());
        auto ap = reinterpret_cast<__ubuf__ float *>(state.ab.GetPhyAddr());
        auto pp = reinterpret_cast<__ubuf__ float *>(state.panel.GetPhyAddr());
        auto ip = reinterpret_cast<__ubuf__ uint32_t *>(state.lanes.GetPhyAddr());
        Shared10VgatherStage4<Self>(ap, be, cp, f, h, ip, pp, state);
        SetFlag<HardEvent::V_S>(2);
    }
    template <bool EXPORT, class Self, class Capture7>
    __aicore__ static inline void Shared10VmlaStage1(__ubuf__ float *&dst, uint32_t &f, const uint32_t &full,
                                                     uint32_t &kb, const uint32_t &parts, __ubuf__ float *&pivot,
                                                     const uint32_t &rows, Capture7 &state, const uint32_t &tail) {
        auto coeff = reinterpret_cast<__ubuf__ float *>(state.brc[8].GetPhyAddr());
        auto oldPivotTiles = reinterpret_cast<__ubuf__ float *>(state.tmp.GetPhyAddr());
        SetMaskNorm();
        AscendCUtils::SetMask<float>(64);
        if constexpr (Self::GUARDED) {
            vmla(dst, pivot, coeff, static_cast<uint8_t>((rows + 7U) / 8U), Self::P / 8U, 0, 1, Self::P, 0, 8);
        } else {
            vadds(reinterpret_cast<__ubuf__ int32_t *>(oldPivotTiles), reinterpret_cast<__ubuf__ int32_t *>(pivot),
                  int32_t(0), static_cast<uint8_t>(parts), 1, 0, 8, 1);
            PipeBarrier<PIPE_V>();
            vmla(dst, oldPivotTiles, coeff, static_cast<uint8_t>(full), Self::P / 8U, 1, 1, Self::P, 0, 8);
            if (tail) {
                AscendCUtils::SetMask<float>(static_cast<int32_t>(tail * 8U));
                vmla(dst + full * 8U * Self::P, oldPivotTiles, coeff + full * 64U, 1, Self::P / 8U, 1, 1, Self::P, 0,
                     8);
            }
        }
        PipeBarrier<PIPE_V>();
        state.template issuePreparedPivot<EXPORT>(kb + f + 1U, f + 1U);
        AscendCUtils::SetMask<float>(64);
        if constexpr (Self::GUARDED) {
            for (uint32_t j = 1; j < parts; ++j)
                vmla(dst + j * 8U, pivot + j * 8U, coeff, static_cast<uint8_t>((rows + 7U) / 8U), Self::P / 8U, 0, 1,
                     Self::P, 0, 8);
        } else {
            for (uint32_t j = 1; j < parts; ++j) {
                AscendCUtils::SetMask<float>(64);
                vmla(dst + j * 8U, oldPivotTiles + j * 64U, coeff, static_cast<uint8_t>(full), Self::P / 8U, 1, 1,
                     Self::P, 0, 8);
                if (tail) {
                    AscendCUtils::SetMask<float>(static_cast<int32_t>(tail * 8U));
                    vmla(dst + full * 8U * Self::P + j * 8U, oldPivotTiles + j * 64U, coeff + full * 64U, 1,
                         Self::P / 8U, 1, 1, Self::P, 0, 8);
                }
            }
        }
    }
    template <bool EXPORT, class Self, class Capture7>
    __aicore__ static inline void SharedVmlaStage1(__ubuf__ float *&dst, uint32_t &f, const uint32_t &full,
                                                   uint32_t &kb, const uint32_t &parts, __ubuf__ float *&pivot,
                                                   const uint32_t &rows, Capture7 &state, const uint32_t &tail) {
        Shared10VmlaStage1<EXPORT, Self>(dst, f, full, kb, parts, pivot, rows, state, tail);
    }
    template <bool EXPORT, class Self, class Capture4>
    __aicore__ static inline void Shared24ScalarUpdateStage11(uint32_t &c0, uint32_t &f, uint32_t &h, uint32_t &kb,
                                                              Capture4 &state, uint32_t &width) {
        const uint32_t rows = h - 1U, parts = width / 8U, full = rows / 8U, tail = rows % 8U;
        auto dst = reinterpret_cast<__ubuf__ float *>(state.panel[(kb + f + 1U) * Self::P + c0].GetPhyAddr());
        auto pivot = reinterpret_cast<__ubuf__ float *>(state.prow[c0].GetPhyAddr());
        SharedVmlaStage1<EXPORT, Self>(dst, f, full, kb, parts, pivot, rows, state, tail);
        ResetMask();
        SetMaskNorm();
    }
