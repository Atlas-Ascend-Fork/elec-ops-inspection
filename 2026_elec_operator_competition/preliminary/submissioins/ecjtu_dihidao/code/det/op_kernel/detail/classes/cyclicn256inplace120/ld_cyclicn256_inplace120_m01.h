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
// Included in the CyclicN256Inplace120 class; preserves its member scope and access.
    __aicore__ inline void notifyTag(uint32_t slot, uint32_t value) {
        auto ut = DISJOINT_SIGNAL ? signalBuf : tmp.template ReinterpretCast<uint32_t>();
        ut.SetValue(0, value);
        SM();
        DataCopy(tags[slot * 16U], ut, 8U);
        MS();
    }
    __aicore__ inline void waitTags() {
        auto ut = DISJOINT_SIGNAL ? signalBuf : tmp.template ReinterpretCast<uint32_t>();
        do {
            if constexpr (FAC_SLOTS == 2U)
                DataCopy(ut, tags, {2U, 1U, 1U, 0U});
            else
                DataCopy(ut, tags, 8U);
            SetFlag<HardEvent::MTE2_S>(5);
            WaitFlag<HardEvent::MTE2_S>(5);
        } while (ut.GetValue(0U) != 1U || (FAC_SLOTS == 2U && ut.GetValue(8U) != 1U));
    }
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline uint32_t bandFirst(uint32_t r) {
        if (rebalanced == 0U)
            return r == 0U ? 0U : (r == 1U ? 2U : 5U);
        if (rebalanced == 1U)
            return r == 0U ? 2U : (r == 1U ? 3U : 5U);
        return r == 0U ? (rebalanced == 3U ? 5U : 2U) : (r == 1U ? 4U : 6U);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicN256Inplace120>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicN256Inplace120>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicN256Inplace120>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicN256Inplace120>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicN256Inplace120>(*this, kb);
    }
    __aicore__ inline void bindPayloadMeta(uint32_t kb) {
        return LdetSharedOperations::bindPayloadMetaShared13<CyclicN256Inplace120>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        const uint32_t slot = (FAC_SLOTS == 2U ? ((kb / B) & 1U) : ((kb / B) % FAC_SLOTS)) * GM_WORDS;
        bindPayloadMeta(kb);
        DataCopy(fac, ws[slot], B * facPitch(kb) + META_WORDS);
        MV();
        if constexpr (SCALAR_DMA_RECEIVE) {
            SetFlag<HardEvent::MTE2_S>(5);
            WaitFlag<HardEvent::MTE2_S>(5);
        } else
            VS();
        sign = value(mu.GetValue(0));
        prod = value(mu.GetValue(1));
        ex = int32_t(mu.GetValue(2));
        singular = mu.GetValue(3);
    }
    __aicore__ inline void publish(uint32_t kb) {
        const uint32_t slot = (FAC_SLOTS == 2U ? ((kb / B) & 1U) : ((kb / B) % FAC_SLOTS)) * GM_WORDS;
        bindPayloadMeta(kb);
        state();
        SM();
        if constexpr (!PUBLISH_VM_TRIM)
            VM();
        DataCopy(ws[slot], fac, B * facPitch(kb) + META_WORDS);
        MS();
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared15<CyclicN256Inplace120>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<CyclicN256Inplace120>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared17<CyclicN256Inplace120>(*this, dst, pivotRow, h, stride, width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared18<CyclicN256Inplace120>(*this, dst, pivotRow, h, width);
    }

    template <bool EXPORT = true> __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared19<CyclicN256Inplace120, EXPORT>(*this, k, f);
    }

    template <bool EXPORT = true>
    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        if (PROVEN_PANEL_BOUNDS ? width < 16U : (h < 64U || width < 16U || f + 1U >= B)) {
            updatePanelRows8x8(panel[(kb + f + 1U) * P + c0], prow[c0], h, width);
            return false;
        }
        const uint32_t rows = h - 1U, parts = width / 8U, full = rows / 8U, tail = rows % 8U;
        auto dst = reinterpret_cast<__ubuf__ float *>(panel[(kb + f + 1U) * P + c0].GetPhyAddr());
        auto pivot = reinterpret_cast<__ubuf__ float *>(prow[c0].GetPhyAddr());
        auto coeff = reinterpret_cast<__ubuf__ float *>(brc[8].GetPhyAddr());
        auto oldPivotTiles = reinterpret_cast<__ubuf__ float *>(tmp.GetPhyAddr());
        SetMaskNorm();
        AscendCUtils::SetMask<float>(64);
        if constexpr (GUARDED) {
            vmla(dst, pivot, coeff, static_cast<uint8_t>((rows + 7U) / 8U), P / 8U, 0, 1, P, 0, 8);
        } else {
            vadds(reinterpret_cast<__ubuf__ int32_t *>(oldPivotTiles), reinterpret_cast<__ubuf__ int32_t *>(pivot),
                  int32_t(0), static_cast<uint8_t>(parts), 1, 0, 8, 1);
            PipeBarrier<PIPE_V>();
            vmla(dst, oldPivotTiles, coeff, static_cast<uint8_t>(full), P / 8U, 1, 1, P, 0, 8);
            if (tail) {
                AscendCUtils::SetMask<float>(static_cast<int32_t>(tail * 8U));
                vmla(dst + full * 8U * P, oldPivotTiles, coeff + full * 64U, 1, P / 8U, 1, 1, P, 0, 8);
            }
        }
        PipeBarrier<PIPE_V>();
        issuePreparedPivot<EXPORT>(kb + f + 1U, f + 1U);
        AscendCUtils::SetMask<float>(64);
        if constexpr (GUARDED) {
            for (uint32_t j = 1; j < parts; ++j)
                vmla(dst + j * 8U, pivot + j * 8U, coeff, static_cast<uint8_t>((rows + 7U) / 8U), P / 8U, 0, 1, P, 0,
                     8);
        } else {
            for (uint32_t j = 1; j < parts; ++j) {
                AscendCUtils::SetMask<float>(64);
                vmla(dst + j * 8U, oldPivotTiles + j * 64U, coeff, static_cast<uint8_t>(full), P / 8U, 1, 1, P, 0, 8);
                if (tail) {
                    AscendCUtils::SetMask<float>(static_cast<int32_t>(tail * 8U));
                    vmla(dst + full * 8U * P + j * 8U, oldPivotTiles + j * 64U, coeff + full * 64U, 1, P / 8U, 1, 1, P,
                         0, 8);
                }
            }
        }
        ResetMask();
        SetMaskNorm();
        return true;
    }
    template <bool EXPORT>
    __aicore__ inline void factorvmulsPhase7(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                             uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                             __ubuf__ float *&rp) {

        if constexpr (NORMAL_COPY)
            AscendCUtils::SetMask<float>(static_cast<int32_t>(B));
        else
            LDET_MASK_SET(B);
        vmuls(rp, pp + p * P, inv, 1, 1, 1, 8, 8);
        if (p != k)
            vmuls(pp + p * P, pp + k * P, 1.0f, 1, 1, 1, 8, 8);
        if constexpr (NORMAL_COPY)
            ResetMask();
        else
            LDET_MASK_RST();
        if constexpr (GUARDED || PIVOT8_AHEAD) {
            if (p != k)
                PipeBarrier<PIPE_V>();
        }
        repair(p - k);
        PipeBarrier<PIPE_V>();
        uint32_t c0 = (f + 1) & ~7U, w = B - c0;
        if constexpr (FAC_REPAIRED && EXPORT)
            patchFac(f, kb, p - k);
        if constexpr (PIVOT8_AHEAD) {
            pivotPrepared = w ? updatePanelAndPrepare<EXPORT>(kb, f, h, c0, w) : false;
        } else {
            if (w)
                updatePanelRows8x8(panel[(k + 1) * P + c0], prow[c0], h, w);
        }
        LdetSharedOperations::FinalizeDeferredPivot64<CyclicN256Inplace120, DEFER_BOOKKEEP, EXPORT>(*this, f, p, piv,
                                                                                                    inv);
    }
    template <bool EXPORT>
    __aicore__ inline void factorvgatherPhase8(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp, uint32_t &f,
                                               uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k, uint32_t &kb,
                                               bool &pivotPrepared, __ubuf__ float *&pp) {

        if constexpr (DIRECT_FAC && EXPORT)
            cp = reinterpret_cast<__ubuf__ float *>(fac[f * facPitch(kb)].GetPhyAddr());
        if constexpr (PIVOT8_AHEAD) {
            if (!pivotPrepared)
                issuePreparedPivot<EXPORT>(k, f);
        } else {
            LdetSharedOperations::SharedOwnerVgatherStage16<CyclicN256Inplace120>(*this, ap, be, cp, f, h, ip, pp);
        }
        if constexpr (GUARDED && !PIVOT8_AHEAD)
            SetFlag<HardEvent::V_S>(2);
        if (h > 1)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(brc.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t *>(cp), 1,
                  8, static_cast<uint8_t>((h + 7) / 8));
        if constexpr (EXPORT && !DIRECT_FAC)
            DataCopy(fac[f * facPitch(kb)], col, ((h + 7U) / 8U) * 8U);
    }

    __aicore__ inline void factorwordPhase9(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p, __ubuf__ float *&tp) {
        return LdetSharedOperations::factorwordPhase3SharedPhase2<CyclicN256Inplace120>(*this, h, k, maxv, p, tp);
    }

    __aicore__ inline void factorwordPhase10(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p, __ubuf__ float *&tp) {
        return LdetSharedOperations::factorwordPhase10SharedPhase9<CyclicN256Inplace120>(*this, h, k, maxv, p, tp);
    }

    __aicore__ inline void factorconstexprPhase11(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                  __ubuf__ float *&tp) {

        if constexpr (GUARDED && BRANCHLESS_SEGMENTS) {
            LdetSharedOperations::SharedOwnerScalarUpdateStage4<CyclicN256Inplace120>(*this, h, k, maxv, p, tp);
        } else if constexpr (GUARDED) {
            factorwordPhase9(h, k, maxv, p, tp);
        } else {
            factorwordPhase10(h, k, maxv, p, tp);
        }
    }
    template <bool EXPORT>
    __aicore__ inline void factorconstexprPhase12(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                                  uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                                  __ubuf__ float *&rp) {

        if (p != k)
            sign = -sign;
        if constexpr (EXPORT && !DEFER_BOOKKEEP) {
            mu.SetValue(8 + f, p);
            mu.SetValue(8U + B + f, word(inv));
        }
        factorvmulsPhase7<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, pp, rp);
    }
    template <bool EXPORT, class LoopArg0, class LoopArg1, class LoopArg2, class LoopArg3, class LoopArg4,
              class LoopArg5, class LoopArg6, class LoopArg7, class LoopArg8>
    __aicore__ inline bool InplaceFactorPivotStep(LoopArg0 &ap, LoopArg1 &cp, LoopArg2 &f, LoopArg3 &ip, LoopArg4 &kb,
                                                  LoopArg5 &pivotPrepared, LoopArg6 &pp, LoopArg7 &rp, LoopArg8 &tp) {

        uint32_t k = kb + f, h = n - k, be = k * P + f;
        factorvgatherPhase8<EXPORT>(ap, be, cp, f, h, ip, k, kb, pivotPrepared, pp);
        if constexpr (GUARDED || PIVOT8_AHEAD)
            WaitFlag<HardEvent::V_S>(2);
        else
            VS();
        float maxv = 0;
        uint32_t p = k;
        factorconstexprPhase11(h, k, maxv, p, tp);
        if (maxv == 0) {
            singular = 1;
            return false;
        }
        float piv = cp[p - k];
        const uint32_t pivotExponent = NORMAL_INVERSE_GUARD ? ((word(piv) >> 23U) & 255U) : 0U;
        if constexpr (DEFER_BOOKKEEP) {
            if (piv == 0.0f || (NORMAL_INVERSE_GUARD ? pivotExponent : ((word(piv) >> 23) & 255U)) == 255U) {
                acc(piv);
                return false;
            }
        } else {
            acc(piv);
            if (singular)
                return false;
        }
        float inv = -1.0f / piv;
        if ((!NORMAL_INVERSE_GUARD || pivotExponent == 0U) && ((word(inv) >> 23) & 255U) == 255U) {
            if constexpr (DEFER_BOOKKEEP)
                acc(piv);
            singular = 2;
            return false;
        }
        factorconstexprPhase12<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, pp, rp);

        return true;
    }
    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        bindPayloadMeta(kb);
        const uint32_t column = (q - bandFirst(rank)) * B;
        panel = LocalTensor<float>(TPosition::VECCALC, MAT + (matShift + column) * 4U, P * (NMAX + GUARD_ROWS));
        PipeBarrier<PIPE_V>();
        auto factorViews = LdetSharedOperations::GetFactorRawViews60(*this);
        bool pivotPrepared = false;
        for (uint32_t f = 0; f < B; ++f) {
            if (!InplaceFactorPivotStep<EXPORT>(factorViews.ap, factorViews.cp, f, factorViews.ip, kb, pivotPrepared,
                                                factorViews.pp, factorViews.rp, factorViews.tp))
                break;
        }
        if constexpr (PANEL_ACC)
            normalizeProd();
        VS();
    }

    __aicore__ inline void ApplyPanelUpdatesPipeBarrierPhase4(const uint32_t &begin, uint32_t &kb,
                                                              const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr()),
             rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
        SetMaskNorm();
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, p = mu.GetValue(8 + f);
            float inv = value(mu.GetValue(8U + B + f));
            if (f)
                PipeBarrier<PIPE_V>();
            LdetSharedOperations::SharedOwnerVmulsStage8<CyclicN256Inplace120>(*this, begin, f, h, inv, k, kb, mp, p,
                                                                               rp, width);
        }
        PipeBarrier<PIPE_V>();
        ResetMask();
        SetMaskNorm();
        VS();
        return;
    }

    __aicore__ inline void ApplyPanelUpdatesvmulsPhase5(const uint32_t &begin, uint32_t &kb, const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr());
        auto rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, p = mu.GetValue(8 + f);
            float inv = value(mu.GetValue(8U + B + f));
            if constexpr (NORMAL_COPY)
                AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            else
                LDET_MASK_SET(width);
            vmuls(rp, mp + p * S + begin, inv, 1, 1, 1, 8, 8);
            if (p != k)
                vmuls(mp + p * S + begin, mp + k * S + begin, 1.0f, 1, 1, 1, 8, 8);
            if constexpr (NORMAL_COPY)
                ResetMask();
            else
                LDET_MASK_RST();
            LdetSharedOperations::SharedOwner26VbrcbStage1<CyclicN256Inplace120>(*this, f, h, kb);
            LdetSharedOperations::RepairAndUpdateRows64<CyclicN256Inplace120, FAC_REPAIRED>(*this, p, k, begin, h,
                                                                                            width);
        }
        if constexpr (!PRUNE_STAGE_WAITS)
            VS();
    }
