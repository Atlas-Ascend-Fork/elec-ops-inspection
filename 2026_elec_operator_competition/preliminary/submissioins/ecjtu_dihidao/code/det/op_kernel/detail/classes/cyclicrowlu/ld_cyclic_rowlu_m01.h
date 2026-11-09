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
// Included in the CyclicRowLU class; preserves its member scope and access.
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicRowLU>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicRowLU>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicRowLU>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicRowLU>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicRowLU>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicRowLU>(*this);
    }

    __aicore__ inline uint32_t bandFirst(uint32_t r) {
        if constexpr (HETERO3)
            return r * 3U;
        else if constexpr (HETERO5)
            return LdetSharedOperations::bandFirstShared22StateDepth22Stage1IndexSelect22Stage1(r);
        else
            return r * 2U;
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicRowLU>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicRowLU>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicRowLU>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicRowLU>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicRowLU>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicRowLU>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicRowLU>(*this, kb);
    }
    __aicore__ inline void bindPayloadMeta(uint32_t kb) {
        return LdetSharedOperations::bindPayloadMetaShared13<CyclicRowLU>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        return LdetSharedOperations::receiveShared14<CyclicRowLU>(*this, kb);
    }
    __aicore__ inline void publish(uint32_t kb) {
        const uint32_t slot = ((kb / B) & 1U) * GM_WORDS;
        bindPayloadMeta(kb);
        state();
        SM();
        VM();
        DataCopy(ws[slot], fac, B * facPitch(kb) + META_WORDS);
        MS();
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared15<CyclicRowLU>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<CyclicRowLU>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared17<CyclicRowLU>(*this, dst, pivotRow, h, stride, width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared18<CyclicRowLU>(*this, dst, pivotRow, h, width);
    }

    template <bool EXPORT = true> __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared19<CyclicRowLU, EXPORT>(*this, k, f);
    }

    template <bool EXPORT = true>
    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared20<CyclicRowLU, EXPORT>(*this, kb, f, h, c0, width);
    }

    template <bool EXPORT>
    __aicore__ inline void factorvmulsPhase1(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                             uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                             __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwnerVmulsStage11<CyclicRowLU, FAC_REPAIRED, NORMAL_COPY, PIVOT8_AHEAD, EXPORT>(
            *this, f, h, inv, k, kb, p, pivotPrepared, pp, rp);
        LdetSharedOperations::FinalizeDeferredPivot64<CyclicRowLU, DEFER_BOOKKEEP, EXPORT>(*this, f, p, piv, inv);
    }
    template <bool EXPORT>
    __aicore__ inline void factorvgatherPhase2(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp, uint32_t &f,
                                               uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k, uint32_t &kb,
                                               bool &pivotPrepared, __ubuf__ float *&pp) {
        return LdetSharedOperations::factorvgatherPhase2SharedPhase1<CyclicRowLU, DIRECT_FAC, PIVOT8_AHEAD, EXPORT>(
            *this, ap, be, cp, f, h, ip, k, kb, pivotPrepared, pp);
    }

    __aicore__ inline void factorwordPhase3(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p, __ubuf__ float *&tp) {
        return LdetSharedOperations::factorwordPhase3SharedPhase2<CyclicRowLU>(*this, h, k, maxv, p, tp);
    }

    __aicore__ inline void factorconstexprPhase4(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                 __ubuf__ float *&tp) {

        if constexpr (GUARDED) {
            factorwordPhase3(h, k, maxv, p, tp);
        } else {
            float v = tp[0];
            if (v > maxv) {
                maxv = v;
                p = k + word(tp[1]);
            }
            if (h > 64U) {
                v = tp[2];
                if (v > maxv) {
                    maxv = v;
                    p = k + 64U + word(tp[3]);
                }
            }
            LdetSharedOperations::SharedOwner26ScalarUpdateStage4<CyclicRowLU>(*this, h, k, maxv, p, tp, v);
        }
    }

    __aicore__ inline void factorDataCopyPhase5(uint32_t &kb, uint32_t &q) {

        bindPayloadMeta(kb);
        if constexpr (GROUP == 8U && PRUNE_STAGE_WAITS) {
            static_assert(B == 32U && S == P, "B4 G8 owns exactly one in-place panel");
        } else if constexpr (ACTIVE_PANEL_ROWS) {
            DataCopy(
                panel[kb * P],
                mat[kb * S + (CONTIGUOUS_BANDS ? ((HETERO5 || HETERO3) ? q - bandFirst(rank) : q % 2U) : q / G) * B],
                {static_cast<uint16_t>(n - kb), B / 8U, (S - B) / 8U, (P - B) / 8U});
        } else {
            DataCopy(panel, mat[(CONTIGUOUS_BANDS ? ((HETERO5 || HETERO3) ? q - bandFirst(rank) : q % 2U) : q / G) * B],
                     {static_cast<uint16_t>(n), B / 8U, (S - B) / 8U, (P - B) / 8U});
        }
        PipeBarrier<PIPE_V>();
    }
    template <bool EXPORT>
    __aicore__ inline void factorconstexprPhase6(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                                 uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                                 __ubuf__ float *&rp) {

        if (p != k)
            sign = -sign;
        if constexpr (EXPORT && !DEFER_BOOKKEEP) {
            mu.SetValue(8 + f, p);
            mu.SetValue(8U + B + f, word(inv));
        }
        factorvmulsPhase1<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, pp, rp);
    }
    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        factorDataCopyPhase5(kb, q);
        auto factorViews = LdetSharedOperations::GetFactorRawViews60(*this);
        bool pivotPrepared = false;
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, be = k * P + f;
            factorvgatherPhase2<EXPORT>(factorViews.ap, be, factorViews.cp, f, h, factorViews.ip, k, kb, pivotPrepared,
                                        factorViews.pp);
            if constexpr (GUARDED || PIVOT8_AHEAD)
                WaitFlag<HardEvent::V_S>(2);
            else
                VS();
            float maxv = 0;
            uint32_t p = k;
            factorconstexprPhase4(h, k, maxv, p, factorViews.tp);
            if (maxv == 0) {
                singular = 1;
                break;
            }
            float piv = factorViews.cp[p - k];
            if constexpr (DEFER_BOOKKEEP) {
                if (piv == 0.0f || ((word(piv) >> 23) & 255U) == 255U) {
                    acc(piv);
                    break;
                }
            } else {
                acc(piv);
                if (singular)
                    break;
            }
            float inv = -1.0f / piv;
            if (((word(inv) >> 23) & 255U) == 255U) {
                if constexpr (DEFER_BOOKKEEP)
                    acc(piv);
                singular = 2;
                break;
            }
            factorconstexprPhase6<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, factorViews.pp, factorViews.rp);
        }
        if constexpr (PANEL_ACC)
            normalizeProd();
        VS();
    }

    __aicore__ inline void ApplyPanelUpdatesPipeBarrierPhase1(const uint32_t &begin, uint32_t &kb, __ubuf__ float *&mp,
                                                              __ubuf__ float *&rp, const uint32_t &width) {

        SetMaskNorm();
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, p = mu.GetValue(8 + f);
            float inv = value(mu.GetValue(8U + B + f));
            if constexpr (GROUP != 5U) {
                if (f)
                    PipeBarrier<PIPE_V>();
            }
            LdetSharedOperations::SharedOwnerVmulsStage8<CyclicRowLU>(*this, begin, f, h, inv, k, kb, mp, p, rp, width);
            if constexpr (GROUP == 5U)
                PipeBarrier<PIPE_V>();
        }
        if constexpr (GROUP != 5U)
            PipeBarrier<PIPE_V>();
        ResetMask();
    }

    __aicore__ inline void ApplyPanelUpdatesPipeBarrierPhase2(const uint32_t &begin, uint32_t &kb,
                                                              const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr()),
             rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
        SetMaskNorm();
        AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, p = mu.GetValue(8 + f);
            float inv = value(mu.GetValue(8U + B + f));
            if constexpr (GROUP != 5U) {
                if (f)
                    PipeBarrier<PIPE_V>();
            }
            LdetSharedOperations::SharedOwnerVmulsStage20<CyclicRowLU>(*this, begin, f, h, inv, k, kb, mp, p, rp);
            if constexpr (GROUP == 5U)
                PipeBarrier<PIPE_V>();
        }
        if constexpr (GROUP != 5U)
            PipeBarrier<PIPE_V>();
        ResetMask();
        SetMaskNorm();
        if constexpr (!PRUNE_STAGE_WAITS)
            VS();
        return;
    }

    __aicore__ inline void ApplyPanelUpdatesvmulsPhase3(const uint32_t &begin, uint32_t &kb, const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr());
        auto rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, p = mu.GetValue(8 + f);
            float inv = value(mu.GetValue(8U + B + f));
            LdetSharedOperations::SharedOwnerVmulsStage29<CyclicRowLU, NORMAL_COPY>(*this, begin, inv, k, mp, p, rp,
                                                                                    width);
            LdetSharedOperations::SharedOwner26VbrcbStage1<CyclicRowLU>(*this, f, h, kb);
            LdetSharedOperations::RepairAndUpdateRows64<CyclicRowLU, FAC_REPAIRED>(*this, p, k, begin, h, width);
        }
        if constexpr (!PRUNE_STAGE_WAITS)
            VS();
    }
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        uint32_t first = LdetSharedOperations::FirstOwnedColumn54<CyclicRowLU, CONTIGUOUS_BANDS>(*this, q);
        const uint32_t begin = first * B, width = owned * B > begin ? owned * B - begin : 0;
        if (!width)
            return;
        if constexpr (HETERO3) {
            auto mp = reinterpret_cast<__ubuf__ float *>(mat.GetPhyAddr()),
                 rp = reinterpret_cast<__ubuf__ float *>(row.GetPhyAddr());
            ApplyPanelUpdatesPipeBarrierPhase1(begin, kb, mp, rp, width);
            SetMaskNorm();
            VS();
            return;
        }
        if constexpr (PANEL_MASK) {
            return ApplyPanelUpdatesPipeBarrierPhase2(begin, kb, width);
        }
        ApplyPanelUpdatesvmulsPhase3(begin, kb, width);
    }
    __aicore__ inline void Finish64() {
        return LdetSharedOperations::Finish64Shared21<CyclicRowLU>(*this);
    }
