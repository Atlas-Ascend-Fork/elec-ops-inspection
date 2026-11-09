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
// Included in the CyclicRowLUMedium class; preserves its member scope and access.
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicRowLUMedium>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicRowLUMedium>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicRowLUMedium>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicRowLUMedium>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicRowLUMedium>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        return LdetSharedOperations::receiveShared44<CyclicRowLUMedium>(*this, kb);
    }
    __aicore__ inline void publish(uint32_t kb) {
        return LdetSharedOperations::publishShared45<CyclicRowLUMedium>(*this, kb);
    }
    __aicore__ inline void publishPrefix16(uint32_t kb) {
        return LdetSharedOperations::publishPrefix16Shared46<CyclicRowLUMedium>(*this, kb);
    }
    template <uint32_t FIRST, uint32_t LAST> __aicore__ inline void receiveRange(uint32_t kb) {
        return LdetSharedOperations::receiveRangeShared47<CyclicRowLUMedium, FIRST, LAST>(*this, kb);
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared48<CyclicRowLUMedium>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<CyclicRowLUMedium>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared49<CyclicRowLUMedium>(*this, dst, pivotRow, h, stride, width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared50<CyclicRowLUMedium>(*this, dst, pivotRow, h, width);
    }

    __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared51<CyclicRowLUMedium>(*this, k, f);
    }

    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared52<CyclicRowLUMedium>(*this, kb, f, h, c0, width);
    }

    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::factorShared53<CyclicRowLUMedium, EXPORT>(*this, kb, q);
    }

    template <uint32_t FIRST = 0U, uint32_t LAST = B>
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::ApplyPanelUpdatesShared54<CyclicRowLUMedium, FIRST, LAST>(*this, kb, q);
    }

    __aicore__ inline void copyInput(bool valid) {
        return LdetSharedOperations::copyInputShared55<CyclicRowLUMedium>(*this, valid);
    }

    template <auto T, auto TS>
    __aicore__ inline void FinishLocalTailvgatherPhase1(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                        const uint32_t &first, uint32_t &h, __ubuf__ uint32_t *&ip,
                                                        __ubuf__ float *&mp, const uint32_t &nf, const uint32_t &nh,
                                                        bool &pendingFar, bool &prepared, __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwnerVgatherStage33<CyclicRowLUMedium, T, TS>(*this, ap, cp, ip, mp, nf, nh);
        prepared = true;
        if (first + 8U < T) {
            if constexpr (T > 72U) {
                for (uint32_t off = first + 8U; off < T; off += 64U) {
                    SetMaskNorm();
                    AscendCUtils::SetMask<float>(static_cast<int32_t>(T - off < 64U ? T - off : 64U));
                    vmla(mp + (f + 1U) * TS + off, rp + off, reinterpret_cast<__ubuf__ float *>(brc[8U].GetPhyAddr()),
                         static_cast<uint8_t>(h - 1U), 1, 1, 0, TS / 8U, 0, 1);
                }
                ResetMask();
                SetMaskNorm();
                pendingFar = true;
            } else {
                LdetSharedOperations::SharedOwner26VmlaStage6<CyclicRowLUMedium, T, TS>(*this, f, first, h, mp, rp);
                pendingFar = true;
            }
        }
    }
    template <auto T, auto TS>
    __aicore__ inline void FinishLocalTailPipeBarrierPhase2(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                            uint32_t &h, float &inv, __ubuf__ uint32_t *&ip,
                                                            __ubuf__ float *&mp, const uint32_t &p, bool &pendingFar,
                                                            float &piv, bool &prepared, __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwnerPipeBarrierStage15<CyclicRowLUMedium, T, TS>(*this, f, h, inv, mp, p,
                                                                                      pendingFar, rp);
        const uint32_t first = (f + 1U) & ~7U;
        prepared = false;
        if (h > 1U && first < T) {
            SetMaskNorm();
            AscendCUtils::SetMask<float>(64);
            vmla(mp + (f + 1U) * TS + first, rp + first, reinterpret_cast<__ubuf__ float *>(brc[8U].GetPhyAddr()),
                 static_cast<uint8_t>((h - 1U + 7U) / 8U), TS / 8U, 0, 1, TS, 0, 8);
            PipeBarrier<PIPE_V>();
            const uint32_t nf = f + 1U, nh = h - 1U;
            FinishLocalTailvgatherPhase1<T, TS>(ap, cp, f, first, h, ip, mp, nf, nh, pendingFar, prepared, rp);
        }
        acc(piv);
        if constexpr (PANEL_ACC) {
            if ((f & 31U) == 31U)
                normalizeProd();
        }
    }
    template <auto T, auto TS, class PhaseArg0> __aicore__ inline void FinishLocalTailDataCopyPhase3(PhaseArg0 &tm) {

        DataCopy(tm, mat[LOCAL_CUT * S], {static_cast<uint16_t>(T), T / 8U, (S - T) / 8U, (TS - T) / 8U});
        LdetSharedOperations::SharedOwnerPipeBarrierStage25<CyclicRowLUMedium, TS>(*this);
        if constexpr (T > 64U) {
            Adds(tab[64U], tab, static_cast<int32_t>(64U * TS * 4U), T - 64U);
            PipeBarrier<PIPE_V>();
        }
        LDET_MASK_SET(T);
        for (uint32_t j = 0; j < 8U; ++j)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[j * T].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        PipeBarrier<PIPE_V>();
    }
    template <auto T>
    __aicore__ inline void FinishLocalTailwordPhase4(uint32_t &h, float &maxv, uint32_t &ri, __ubuf__ float *&tp) {

        if (tp[0] > maxv) {
            maxv = tp[0];
            ri = word(tp[1]);
        }
        if constexpr (T > 64U) {
            if (h > 64U && tp[2] > maxv) {
                maxv = tp[2];
                ri = 64U + word(tp[3]);
            }
        }
    }
    template <auto T, auto TS>
    __aicore__ inline void FinishLocalTailvgatherPhase5(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                                        uint32_t &h, __ubuf__ uint32_t *&ip, __ubuf__ float *&mp,
                                                        bool &prepared) {
        return LdetSharedOperations::Finish64vgatherPhase4SharedPhase10<CyclicRowLUMedium, T, TS>(*this, ap, cp, f, h,
                                                                                                  ip, mp, prepared);
    }
    template <auto T, auto TS>
    __aicore__ inline void FinishLocalTailconstexprPhase6(__ubuf__ float *&ap, __ubuf__ float *&cp,
                                                          __ubuf__ uint32_t *&ip, __ubuf__ float *&mp,
                                                          __ubuf__ float *&rp, __ubuf__ float *&tp) {

        bool prepared = false, pendingFar = false;
        for (uint32_t f = 0; f < T && !singular; ++f) {
            uint32_t h = T - f;
            FinishLocalTailvgatherPhase5<T, TS>(ap, cp, f, h, ip, mp, prepared);
            float maxv = 0.0f;
            uint32_t ri = 0U;
            FinishLocalTailwordPhase4<T>(h, maxv, ri, tp);
            if (maxv == 0.0f || ri >= h) {
                singular = 1;
                break;
            }
            float invAbs = 0.0f;
            if constexpr (MAXV_INVERSE)
                invAbs = -1.0f / maxv;
            const uint32_t p = f + ri;
            float piv = cp[ri];
            if (piv == 0.0f || ((word(piv) >> 23U) & 255U) == 255U) {
                acc(piv);
                break;
            }
            float inv;
            if constexpr (MAXV_INVERSE)
                inv = value(word(invAbs) ^ (word(piv) & 0x80000000U));
            else
                inv = -1.0f / piv;
            if (((word(inv) >> 23U) & 255U) == 255U) {
                acc(piv);
                singular = 2;
                break;
            }
            FinishLocalTailPipeBarrierPhase2<T, TS>(ap, cp, f, h, inv, ip, mp, p, pendingFar, piv, prepared, rp);
        }
        if constexpr (PANEL_ACC)
            normalizeProd();
    }
    __aicore__ inline void FinishLocalTail() {
        constexpr uint32_t T = NMAX - LOCAL_CUT, TS = T + 8U;
        static_assert(PANEL + (T + 8U) * TS * sizeof(float) <= BRC, "retired PANEL plus FAC tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, PANEL, (T + 8U) * TS);
        FinishLocalTailDataCopyPhase3<T, TS>(tm);
        auto views = LdetSharedOperations::GetTailRawViews51(*this, tm);
        FinishLocalTailconstexprPhase6<T, TS>(views.ap, views.cp, views.ip, views.mp, views.rp, views.tp);
        VS();
        meta.SetValue(0, singular ? 0.0f : sign * final());
        SM();
        DataCopyPad(out, meta, DataCopyExtParams{1, 4, 0, 0, 0});
        MS();
    }
