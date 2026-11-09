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
// Included in the CyclicN256Seven class; preserves its member scope and access.
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicN256Seven>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicN256Seven>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicN256Seven>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicN256Seven>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicN256Seven>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicN256Seven>(*this);
    }
    __aicore__ inline uint32_t bandFirst(uint32_t r) {
        return LdetSharedOperations::bandFirstShared34<CyclicN256Seven>(*this, r);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicN256Seven>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicN256Seven>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicN256Seven>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicN256Seven>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicN256Seven>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicN256Seven>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicN256Seven>(*this, kb);
    }
    __aicore__ inline void bindPayloadMeta(uint32_t kb) {
        return LdetSharedOperations::bindPayloadMetaShared13<CyclicN256Seven>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        return LdetSharedOperations::receiveShared14<CyclicN256Seven>(*this, kb);
    }
    __aicore__ inline GM_ADDR prefixTicket(uint32_t q) {
        return LdetSharedOperations::prefixTicketShared32<CyclicN256Seven>(*this, q);
    }
    __aicore__ inline void publishPrefix16(uint32_t kb) {
        return LdetSharedOperations::publishPrefix16Shared24<CyclicN256Seven>(*this, kb);
    }
    template <uint32_t FIRST, uint32_t LAST> __aicore__ inline void receiveRange(uint32_t kb) {
        return LdetSharedOperations::receiveRangeShared33<CyclicN256Seven, FIRST, LAST>(*this, kb);
    }

    __aicore__ inline void publish(uint32_t kb) {
        return LdetSharedOperations::publishShared35<CyclicN256Seven>(*this, kb);
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared15<CyclicN256Seven>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<CyclicN256Seven>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared17<CyclicN256Seven>(*this, dst, pivotRow, h, stride, width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared36<CyclicN256Seven>(*this, dst, pivotRow, h, width);
    }

    template <bool EXPORT = true> __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared19<CyclicN256Seven, EXPORT>(*this, k, f);
    }

    template <bool EXPORT = true>
    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared37<CyclicN256Seven, EXPORT>(*this, kb, f, h, c0, width);
    }

    template <bool EXPORT>
    __aicore__ inline void factorPipeBarrierPhase21(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                                    uint32_t &p, float &piv, bool &pivotPrepared, bool &prefixPosted) {
        return LdetSharedOperations::factorPipeBarrierPhase21SharedPhase13<
            CyclicN256Seven, ClassPlan46::DEFER_BOOKKEEP, ClassPlan46::FAC_REPAIRED, ClassPlan46::NORMAL_COPY,
            ClassPlan46::PIVOT8_AHEAD, EXPORT>(*this, f, h, inv, k, kb, p, piv, pivotPrepared, prefixPosted);
    }
    template <bool EXPORT>
    __aicore__ inline void factorvgatherPhase22(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp, uint32_t &f,
                                                uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k, uint32_t &kb,
                                                bool &pivotPrepared, __ubuf__ float *&pp) {
        return LdetSharedOperations::factorvgatherPhase22SharedPhase14<CyclicN256Seven, ClassPlan46::DIRECT_FAC,
                                                                       ClassPlan46::PIVOT8_AHEAD, EXPORT>(
            *this, ap, be, cp, f, h, ip, k, kb, pivotPrepared, pp);
    }

    __aicore__ inline void factorGetPhyAddrPhase23(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p) {
        return LdetSharedOperations::factorGetPhyAddrPhase23SharedPhase15<CyclicN256Seven>(*this, h, k, maxv, p);
    }

    __aicore__ inline void factorwordPhase24(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p, __ubuf__ float *&tp) {
        return LdetSharedOperations::factorwordPhase24SharedPhase16<CyclicN256Seven>(*this, h, k, maxv, p, tp);
    }

    __aicore__ inline void factorconstexprPhase25(uint32_t &h, uint32_t &k, float &maxv, uint32_t &p,
                                                  __ubuf__ float *&tp) {

        if constexpr (GUARDED && ClassPlan46::HETERO6) {
            factorGetPhyAddrPhase23(h, k, maxv, p);
        } else if constexpr (GUARDED) {
            factorwordPhase24(h, k, maxv, p, tp);
        } else {
            LdetSharedOperations::SharedOwnerScalarUpdateStage24<CyclicN256Seven>(*this, h, k, maxv, p, tp);
        }
    }

    __aicore__ inline void factorDataCopyPhase26(uint32_t &kb, uint32_t &q) {
        return LdetSharedOperations::factorDataCopyPhase18SharedPhase11<
            CyclicN256Seven, ClassPlan46::ACTIVE_PANEL_ROWS, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::GROUP,
            ClassPlan46::HETERO6, ClassPlan46::PRUNE_STAGE_WAITS, ClassPlan46::THIN_H6>(*this, kb, q);
    }
    template <bool EXPORT>
    __aicore__ inline void factorvmulsPhase27(uint32_t &f, uint32_t &h, float &inv, uint32_t &k, uint32_t &kb,
                                              uint32_t &p, float &piv, bool &pivotPrepared, __ubuf__ float *&pp,
                                              bool &prefixPosted, __ubuf__ float *&rp) {

        LdetSharedOperations::SharedOwnerVmulsStage26<CyclicN256Seven, ClassPlan46::DEFER_BOOKKEEP,
                                                      ClassPlan46::NORMAL_COPY, EXPORT>(*this, f, inv, k, p, pp, rp);
        factorPipeBarrierPhase21<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, prefixPosted);
    }
    template <bool EXPORT> __aicore__ inline void factorconstexprPhase28(uint32_t &kb, bool &prefixPosted) {

        if constexpr (ClassPlan46::PANEL_ACC)
            normalizeProd();
        VS();
        if constexpr (PREFIX_STREAM && EXPORT) {
            if (!prefixPosted)
                publishPrefix16(kb);
        }
    }
    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        factorDataCopyPhase26(kb, q);
        auto factorViews = LdetSharedOperations::GetFactorRawViews60(*this);
        bool pivotPrepared = false;
        bool prefixPosted = false;
        for (uint32_t f = 0; f < B; ++f) {
            uint32_t k = kb + f, h = n - k, be = k * P + f;
            factorvgatherPhase22<EXPORT>(factorViews.ap, be, factorViews.cp, f, h, factorViews.ip, k, kb, pivotPrepared,
                                         factorViews.pp);
            if constexpr (GUARDED || ClassPlan46::PIVOT8_AHEAD)
                WaitFlag<HardEvent::V_S>(2);
            else
                VS();
            float maxv = 0;
            uint32_t p = k;
            factorconstexprPhase25(h, k, maxv, p, factorViews.tp);
            float piv, inv;
            if (!LdetSharedOperations::PrepareSignedPivot51<ClassPlan46::DEFER_BOOKKEEP>(*this, maxv, factorViews.cp, k,
                                                                                         p, piv, inv))
                break;
            factorvmulsPhase27<EXPORT>(f, h, inv, k, kb, p, piv, pivotPrepared, factorViews.pp, prefixPosted,
                                       factorViews.rp);
        }
        factorconstexprPhase28<EXPORT>(kb, prefixPosted);
    }
    template <uint32_t FIRST = 0U, uint32_t LAST = B>
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::ApplyPanelUpdatesShared38<CyclicN256Seven, FIRST, LAST>(*this, kb, q);
    }

    __aicore__ inline void Finish64() {
        return LdetSharedOperations::Finish64Shared21<CyclicN256Seven>(*this);
    }
