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
// Included in the CyclicG4PriorityRelay class; preserves its member scope and access.
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline uint32_t bandFirst(uint32_t r) {
        return LdetSharedOperations::bandFirstShared22<CyclicG4PriorityRelay>(*this, r);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicG4PriorityRelay>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicG4PriorityRelay>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicG4PriorityRelay>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicG4PriorityRelay>(*this, kb);
    }
    __aicore__ inline void bindPayloadMeta(uint32_t kb) {
        return LdetSharedOperations::bindPayloadMetaShared13<CyclicG4PriorityRelay>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        return LdetSharedOperations::receiveShared14<CyclicG4PriorityRelay>(*this, kb);
    }
    __aicore__ inline GM_ADDR prefixTicket(uint32_t q) {
        return LdetSharedOperations::prefixTicketShared23<CyclicG4PriorityRelay>(*this, q);
    }
    __aicore__ inline void publishPrefix16(uint32_t kb) {
        return LdetSharedOperations::publishPrefix16Shared24<CyclicG4PriorityRelay>(*this, kb);
    }
    template <uint32_t FIRST, uint32_t LAST> __aicore__ inline void receiveRange(uint32_t kb) {
        return LdetSharedOperations::receiveRangeShared25<CyclicG4PriorityRelay, FIRST, LAST>(*this, kb);
    }

    __aicore__ inline void publish(uint32_t kb) {
        return LdetSharedOperations::publishShared26<CyclicG4PriorityRelay>(*this, kb);
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared15<CyclicG4PriorityRelay>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<CyclicG4PriorityRelay>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared17<CyclicG4PriorityRelay>(*this, dst, pivotRow, h, stride, width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared18<CyclicG4PriorityRelay>(*this, dst, pivotRow, h, width);
    }

    template <bool EXPORT = true> __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared19<CyclicG4PriorityRelay, EXPORT>(*this, k, f);
    }

    template <bool EXPORT = true>
    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared27<CyclicG4PriorityRelay, EXPORT>(*this, kb, f, h, c0,
                                                                                                  width);
    }

    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::factorShared28<CyclicG4PriorityRelay, EXPORT>(*this, kb, q);
    }

    template <uint32_t FIRST = 0U, uint32_t LAST = B>
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::ApplyPanelUpdatesShared29<CyclicG4PriorityRelay, FIRST, LAST>(*this, kb, q);
    }

    __aicore__ inline void ApplyPriorityRelay32(uint32_t kb, uint32_t q) {
        static_assert(G == 4U && S == 64U && B == 32U && ClassPlan46::FAC_REPAIRED && ClassPlan46::PHASE_TRIM,
                      "priorityrelayonlyG4");
        LdetSharedOperations::SharedOwnerPipeBarrierStage1<CyclicG4PriorityRelay>(*this, kb);
    }
    __aicore__ inline void PrepareTail64Indices() {
        return LdetSharedOperations::PrepareTail64IndicesShared30<CyclicG4PriorityRelay>(*this);
    }

    __aicore__ inline void Finish64() {
        return LdetSharedOperations::Finish64Shared31<CyclicG4PriorityRelay>(*this);
    }
