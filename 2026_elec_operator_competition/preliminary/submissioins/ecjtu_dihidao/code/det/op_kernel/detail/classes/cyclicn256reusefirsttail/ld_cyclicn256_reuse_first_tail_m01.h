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
// Included in the CyclicN256ReuseFirstTail class; preserves its member scope and access.
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline uint32_t bandFirst(uint32_t r) {
        return LdetSharedOperations::bandFirstShared34<CyclicN256ReuseFirstTail>(*this, r);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicN256ReuseFirstTail>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicN256ReuseFirstTail>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicN256ReuseFirstTail>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicN256ReuseFirstTail>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicN256ReuseFirstTail>(*this, kb);
    }
    __aicore__ inline void bindPayloadMeta(uint32_t kb) {
        return LdetSharedOperations::bindPayloadMetaShared13<CyclicN256ReuseFirstTail>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        return LdetSharedOperations::receiveShared14<CyclicN256ReuseFirstTail>(*this, kb);
    }
    __aicore__ inline GM_ADDR prefixTicket(uint32_t q) {
        return LdetSharedOperations::prefixTicketShared32<CyclicN256ReuseFirstTail>(*this, q);
    }
    __aicore__ inline void publishPrefix16(uint32_t kb) {
        return LdetSharedOperations::publishPrefix16Shared24<CyclicN256ReuseFirstTail>(*this, kb);
    }
    template <uint32_t FIRST, uint32_t LAST> __aicore__ inline void receiveRange(uint32_t kb) {
        return LdetSharedOperations::receiveRangeShared33<CyclicN256ReuseFirstTail, FIRST, LAST>(*this, kb);
    }

    __aicore__ inline void publish(uint32_t kb) {
        return LdetSharedOperations::publishShared35<CyclicN256ReuseFirstTail>(*this, kb);
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared15<CyclicN256ReuseFirstTail>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<CyclicN256ReuseFirstTail>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared17<CyclicN256ReuseFirstTail>(*this, dst, pivotRow, h, stride,
                                                                                  width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared36<CyclicN256ReuseFirstTail>(*this, dst, pivotRow, h,
                                                                                          width);
    }

    template <bool EXPORT = true> __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared19<CyclicN256ReuseFirstTail, EXPORT>(*this, k, f);
    }

    template <bool EXPORT = true>
    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared37<CyclicN256ReuseFirstTail, EXPORT>(*this, kb, f, h,
                                                                                                     c0, width);
    }

    template <bool EXPORT>
    __aicore__ inline void factorPipeBarrierPhase37(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                                    uint32_t &p, float &piv, bool &pivotPrepared, bool &prefixPosted) {
        return LdetSharedOperations::factorPipeBarrierPhase21SharedPhase13<
            CyclicN256ReuseFirstTail, ClassPlan46::DEFER_BOOKKEEP, ClassPlan46::FAC_REPAIRED, ClassPlan46::NORMAL_COPY,
            ClassPlan46::PIVOT8_AHEAD, EXPORT>(*this, f, h, inv, k, kb, p, piv, pivotPrepared, prefixPosted);
    }
    template <bool EXPORT>
    __aicore__ inline void factorvgatherPhase38(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp, uint32_t &f,
                                                uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k, uint32_t &kb,
                                                bool &pivotPrepared, __ubuf__ float *&pp) {
        return LdetSharedOperations::factorvgatherPhase22SharedPhase14<
            CyclicN256ReuseFirstTail, ClassPlan46::DIRECT_FAC, ClassPlan46::PIVOT8_AHEAD, EXPORT>(
            *this, ap, be, cp, f, h, ip, k, kb, pivotPrepared, pp);
    }

    __aicore__ inline void factorGetPhyAddrPhase39(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p) {
        return LdetSharedOperations::factorGetPhyAddrPhase23SharedPhase15<CyclicN256ReuseFirstTail>(*this, h, k, maxv,
                                                                                                    p);
    }

    __aicore__ inline void factorwordPhase40(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p, __ubuf__ float *&tp) {
        return LdetSharedOperations::factorwordPhase24SharedPhase16<CyclicN256ReuseFirstTail>(*this, h, k, maxv, p, tp);
    }

    __aicore__ inline void factorconstexprPhase41(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                  __ubuf__ float *&tp) {

        if constexpr (GUARDED && ClassPlan46::HETERO6) {
            factorGetPhyAddrPhase39(h, k, maxv, p);
        } else if constexpr (GUARDED) {
            factorwordPhase40(h, k, maxv, p, tp);
        } else {
            float v = tp[0];
            if (v > maxv) {
                maxv = v;
                p = k + word(tp[1]);
            }
            {
                v = tp[2];
                if (v > maxv) {
                    maxv = v;
                    p = k + 64U + word(tp[3]);
                }
            }
            LdetSharedOperations::SharedOwner26ScalarUpdateStage4<CyclicN256ReuseFirstTail>(*this, h, k, maxv, p, tp,
                                                                                            v);
        }
    }

    __aicore__ inline void factorDataCopyPhase42DataCopyDepth22Stage1(uint32_t &kb, uint32_t &q) {

        DataCopy(
            panel[kb * P],
            mat[kb * S +
                (ClassPlan46::CONTIGUOUS_BANDS ? (ClassPlan46::HETERO6 ? q - bandFirst(rank) : q % 2U) : q / G) * B],
            {static_cast<uint16_t>(n - kb), B / 8U, (S - B) / 8U, (P - B) / 8U});
    }

    __aicore__ inline void factorDataCopyPhase42(uint32_t &kb, uint32_t &q) {

        bindPayloadMeta(kb);
        if constexpr (ClassPlan46::THIN_H6) {
            static_assert(S == P && S == 104U && B == 32U, "singleownedpanelalreadycurrentinMAT");
        } else if constexpr (ClassPlan46::GROUP == 8U && ClassPlan46::PRUNE_STAGE_WAITS) {
            static_assert(B == 32U && S == P, "B4 G8 owns exactly one in-place panel");
        } else if constexpr (ClassPlan46::ACTIVE_PANEL_ROWS) {
            factorDataCopyPhase42DataCopyDepth22Stage1(kb, q);
        } else {
            DataCopy(
                panel,
                mat[(ClassPlan46::CONTIGUOUS_BANDS ? (ClassPlan46::HETERO6 ? q - bandFirst(rank) : q % 2U) : q / G) *
                    B],
                {static_cast<uint16_t>(n), B / 8U, (S - B) / 8U, (P - B) / 8U});
        }
        PipeBarrier<PIPE_V>();
    }
    template <bool EXPORT>
    __aicore__ inline void factorvmulsPhase43(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                              uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                              bool &prefixPosted, __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwnerVmulsStage26<CyclicN256ReuseFirstTail, ClassPlan46::DEFER_BOOKKEEP,
                                                      ClassPlan46::NORMAL_COPY, EXPORT>(*this, f, inv, k, p, pp, rp);
        factorPipeBarrierPhase37<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, prefixPosted);
    }
    template <bool EXPORT> __aicore__ inline void factorconstexprPhase44(uint32_t &kb, bool &prefixPosted) {

        if constexpr (ClassPlan46::PANEL_ACC)
            normalizeProd();
        VS();
        if constexpr (PREFIX_STREAM && EXPORT) {
            if (!prefixPosted)
                publishPrefix16(kb);
        }
    }
    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        factorDataCopyPhase42(kb, q);
        auto factorViews = LdetSharedOperations::GetFactorRawViews60(*this);
        bool pivotPrepared = false;
        bool prefixPosted = false;
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, be = k * P + f;
            factorvgatherPhase38<EXPORT>(factorViews.ap, be, factorViews.cp, f, h, factorViews.ip, k, kb, pivotPrepared,
                                         factorViews.pp);
            if constexpr (GUARDED || ClassPlan46::PIVOT8_AHEAD)
                WaitFlag<HardEvent::V_S>(2);
            else
                VS();
            float maxv = 0;
            uint32_t p = k;
            factorconstexprPhase41(h, k, maxv, p, factorViews.tp);
            float piv, inv;
            if (!LdetSharedOperations::PrepareSignedPivot51<ClassPlan46::DEFER_BOOKKEEP>(*this, maxv, factorViews.cp, k,
                                                                                         p, piv, inv))
                break;
            factorvmulsPhase43<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, factorViews.pp, prefixPosted,
                                       factorViews.rp);
        }
        factorconstexprPhase44<EXPORT>(kb, prefixPosted);
    }
    template <uint32_t FIRST, uint32_t LAST>
    __aicore__ inline void ApplyPanelUpdatesvmulsPhase9(const uint32_t &begin, uint32_t &kb, const uint32_t &width) {
        return LdetSharedOperations::ApplyPanelUpdatesvmulsPhase6SharedPhase12<
            CyclicN256ReuseFirstTail, ClassPlan46::FAC_REPAIRED, ClassPlan46::NORMAL_COPY,
            ClassPlan46::PRUNE_STAGE_WAITS, FIRST, LAST>(*this, begin, kb, width);
    }
    template <uint32_t FIRST = 0U, uint32_t LAST = B>
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        const uint32_t begin = rank == 0U ? B : 0U;
        const uint32_t width = rank == 0U ? 2U * B : (q < rank ? B : 0U);
        if (!width)
            return;
        if constexpr (ClassPlan46::PANEL_MASK) {
            LdetSharedOperations::SharedOwnerVmulsStage5<CyclicN256ReuseFirstTail, ClassPlan46::PRUNE_STAGE_WAITS,
                                                         FIRST, LAST>(*this, begin, kb, width);
            return;
        }
        ApplyPanelUpdatesvmulsPhase9<FIRST, LAST>(begin, kb, width);
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64vmlaPhase5(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                              const uint32_t &first, uint32_t &h, __ubuf__ uint32_t *&ip,
                                              __ubuf__ float *&mp, bool &pendingFar, float &piv, bool &prepared,
                                              __ubuf__ float *&rp) {
        return LdetSharedOperations::Finish64GeppvmlaPhase1SharedPhase22<CyclicN256ReuseFirstTail,
                                                                         ClassPlan46::PANEL_ACC, T, TS>(
            *this, ap, cp, f, first, h, ip, mp, pendingFar, piv, prepared, rp);
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64PipeBarrierPhase6(__ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f, uint32_t &h,
                                                     float &inv, __ubuf__ uint32_t *&ip, __ubuf__ float *&mp,
                                                     const uint32_t &p, bool &pendingFar, float &piv, bool &prepared,
                                                     __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwner15PipeBarrierStage2<CyclicN256ReuseFirstTail, T, TS>(*this, f, h, inv, mp, p,
                                                                                              pendingFar, rp);
        const uint32_t first = (f + 1U) & ~7U;
        prepared = false;
        Finish64vmlaPhase5<T, TS>(ap, cp, f, first, h, ip, mp, pendingFar, piv, prepared, rp);
    }
    template <auto T, auto TS>
    __aicore__ inline void Finish64vgatherPhase7(__ubuf__ float *&ap, __ubuf__ float *&cp, __ubuf__ uint32_t *&ip,
                                                 __ubuf__ float *&mp, bool &pendingFar, bool &prepared,
                                                 __ubuf__ float *&rp, __ubuf__ float *&tp) {

        for (uint32_t f = 0; f < T; ++f) {
            uint32_t h = T - f;
            LdetSharedOperations::SharedOwnerVgatherStage10<CyclicN256ReuseFirstTail, T, TS>(*this, ap, cp, f, h, ip,
                                                                                             mp, prepared);
            uint32_t p = 0U;
            float piv = 0.0f, inv = 0.0f;
            if (!LdetSharedOperations::PrepareTailPivot60(*this, tp, cp, h, f, p, piv, inv))
                break;
            Finish64PipeBarrierPhase6<T, TS>(ap, cp, f, h, inv, ip, mp, p, pendingFar, piv, prepared, rp);
        }
    }
    __aicore__ inline void Finish64() {
        constexpr uint32_t T = 64U, TS = 72U;
        static_assert((T + 8U) * TS <= P * (NMAX + GUARD_ROWS), "retired PANEL tail+guard fits");
        auto tm = LocalTensor<float>(TPosition::VECCALC, PANEL, (T + 8U) * TS);
        DataCopy(tm, mat[192U * S + B], {static_cast<uint16_t>(T), T / 8U, (S - T) / 8U, (TS - T) / 8U});
        PipeBarrier<PIPE_V>();
        LdetSharedOperations::PrepareTailRowIndices46<TS, 32>(*this);
        LDET_MASK_SET(T);
        for (uint32_t j = 0; j < 8U; ++j)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[j * T].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), static_cast<int32_t>(j * 4U), 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        PipeBarrier<PIPE_V>();
        auto views = LdetSharedOperations::GetTailRawViews51(*this, tm);
        bool prepared = false, pendingFar = false;
        Finish64vgatherPhase7<T, TS>(views.ap, views.cp, views.ip, views.mp, pendingFar, prepared, views.rp, views.tp);
        if constexpr (ClassPlan46::PANEL_ACC)
            normalizeProd();
        VS();
        meta.SetValue(0, singular ? 0.0f : sign * final());
        SM();
        DataCopyPad(out, meta, DataCopyExtParams{1, 4, 0, 0, 0});
        MS();
    }
