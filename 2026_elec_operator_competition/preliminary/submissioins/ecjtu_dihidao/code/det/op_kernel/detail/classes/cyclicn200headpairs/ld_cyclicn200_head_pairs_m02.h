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
// Included in the CyclicN200HeadPairs class; preserves its member scope and access.
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<CyclicN200HeadPairs>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<CyclicN200HeadPairs>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<CyclicN200HeadPairs>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<CyclicN200HeadPairs>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        return LdetSharedOperations::receiveShared44<CyclicN200HeadPairs>(*this, kb);
    }
    __aicore__ inline void waitReady(uint32_t q, uint32_t value) {
        return LdetSharedOperations::waitReadyShared59<CyclicN200HeadPairs>(*this, q, value);
    }
    __aicore__ inline void publish(uint32_t kb) {
        return LdetSharedOperations::publishShared60<CyclicN200HeadPairs>(*this, kb);
    }
    __aicore__ inline void publishPrefix16(uint32_t kb) {
        return LdetSharedOperations::publishPrefix16Shared46<CyclicN200HeadPairs>(*this, kb);
    }
    template <uint32_t FIRST, uint32_t LAST> __aicore__ inline void receiveRange(uint32_t kb) {
        return LdetSharedOperations::receiveRangeShared61<CyclicN200HeadPairs, FIRST, LAST>(*this, kb);
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared48<CyclicN200HeadPairs>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<CyclicN200HeadPairs>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared49<CyclicN200HeadPairs>(*this, dst, pivotRow, h, stride, width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared50<CyclicN200HeadPairs>(*this, dst, pivotRow, h, width);
    }

    __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared51<CyclicN200HeadPairs>(*this, k, f);
    }

    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared52<CyclicN200HeadPairs>(*this, kb, f, h, c0, width);
    }

    template <bool EXPORT = true> __aicore__ inline void factorOriginal(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::factorShared53<CyclicN200HeadPairs, EXPORT>(*this, kb, q);
    }

    template <uint32_t FIRST = 0U, uint32_t LAST = B> __aicore__ inline void ApplyOriginal(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::ApplyPanelUpdatesShared54<CyclicN200HeadPairs, FIRST, LAST>(*this, kb, q);
    }

    __aicore__ inline void copyInput(bool valid) {
        return LdetSharedOperations::copyInputShared55<CyclicN200HeadPairs>(*this, valid);
    }

    __aicore__ inline void FinishLocalTailGepp() {
        return LdetSharedOperations::FinishLocalTailGeppShared56<CyclicN200HeadPairs>(*this);
    }

    __aicore__ inline void FinishLocalTailBlock2() {
        return LdetSharedOperations::FinishLocalTailBlock2Shared57<CyclicN200HeadPairs>(*this);
    }

    __aicore__ inline void FinishLocalTail() {
        return LdetSharedOperations::FinishLocalTailShared58<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline float pairAbs(float x) {
        return LdetSharedOperations::pairAbsShared64<CyclicN200HeadPairs>(*this, x);
    }
    __aicore__ inline bool headPivot() {
        return LdetSharedOperations::headPivotShared65<CyclicN200HeadPairs>(*this);
    }
    __aicore__ inline void factorPairs(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::factorPairsShared66<CyclicN200HeadPairs>(*this, kb, q);
    }
    template <uint32_t FIRST = 0U, uint32_t LAST = B, bool TILE40 = false>
    __aicore__ inline void ApplyPairs(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::ApplyPairsShared67<CyclicN200HeadPairs, FIRST, LAST, TILE40>(*this, kb, q);
    }
    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::factorShared68<CyclicN200HeadPairs>(*this, kb, q);
    }
    template <uint32_t FIRST = 0U, uint32_t LAST = B>
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::ApplyPanelUpdatesShared69<CyclicN200HeadPairs, FIRST, LAST>(*this, kb, q);
    }
