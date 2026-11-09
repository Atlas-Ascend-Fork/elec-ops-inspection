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
// Included in the FullFallbackC22102 class; preserves its member scope and access.
    __aicore__ inline void VS() {
        return LdetSharedOperations::VSShared0<FullFallbackC22102>(*this);
    }
    __aicore__ inline void SV() {
        return LdetSharedOperations::SVShared1<FullFallbackC22102>(*this);
    }
    __aicore__ inline void MV() {
        return LdetSharedOperations::MVShared2<FullFallbackC22102>(*this);
    }
    __aicore__ inline void VM() {
        return LdetSharedOperations::VMShared3<FullFallbackC22102>(*this);
    }
    __aicore__ inline void SM() {
        return LdetSharedOperations::SMShared4<FullFallbackC22102>(*this);
    }
    __aicore__ inline void MS() {
        return LdetSharedOperations::MSShared5<FullFallbackC22102>(*this);
    }
    __aicore__ inline uint32_t word(float f) {
        return LdetSharedOperations::wordShared6<FullFallbackC22102>(*this, f);
    }
    __aicore__ inline float value(uint32_t u) {
        return LdetSharedOperations::valueShared7<FullFallbackC22102>(*this, u);
    }
    __aicore__ inline void normalizeProd() {
        return LdetSharedOperations::normalizeProdShared8<FullFallbackC22102>(*this);
    }
    __aicore__ inline void acc(float piv) {
        return LdetSharedOperations::accShared9<FullFallbackC22102>(*this, piv);
    }

    __aicore__ inline float final() {
        return LdetSharedOperations::finalShared10<FullFallbackC22102>(*this);
    }
    __aicore__ inline void state() {
        return LdetSharedOperations::stateShared11<FullFallbackC22102>(*this);
    }
    __aicore__ inline uint32_t facPitch(uint32_t kb) {
        return LdetSharedOperations::facPitchShared12<FullFallbackC22102>(*this, kb);
    }
    __aicore__ inline void receive(uint32_t kb) {
        return LdetSharedOperations::receiveShared44<FullFallbackC22102>(*this, kb);
    }
    __aicore__ inline void publish(uint32_t kb) {
        return LdetSharedOperations::publishShared45<FullFallbackC22102>(*this, kb);
    }
    __aicore__ inline void publishPrefix16(uint32_t kb) {
        return LdetSharedOperations::publishPrefix16Shared46<FullFallbackC22102>(*this, kb);
    }
    template <uint32_t FIRST, uint32_t LAST> __aicore__ inline void receiveRange(uint32_t kb) {
        return LdetSharedOperations::receiveRangeShared47<FullFallbackC22102, FIRST, LAST>(*this, kb);
    }
    __aicore__ inline void repair(uint32_t pRel) {
        return LdetSharedOperations::repairShared48<FullFallbackC22102>(*this, pRel);
    }
    __aicore__ inline void patchFac(uint32_t f, uint32_t kb, uint32_t pRel) {
        return LdetSharedOperations::patchFacShared16<FullFallbackC22102>(*this, f, kb, pRel);
    }
    __aicore__ inline void updateRows(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow, uint32_t h,
                                      uint32_t stride, uint32_t width) {
        return LdetSharedOperations::updateRowsShared49<FullFallbackC22102>(*this, dst, pivotRow, h, stride, width);
    }

    __aicore__ inline void updatePanelRows8x8(const LocalTensor<float> &dst, const LocalTensor<float> &pivotRow,
                                              uint32_t h, uint32_t width) {
        return LdetSharedOperations::updatePanelRows8x8Shared50<FullFallbackC22102>(*this, dst, pivotRow, h, width);
    }

    __aicore__ inline void issuePreparedPivot(uint32_t k, uint32_t f) {
        return LdetSharedOperations::issuePreparedPivotShared51<FullFallbackC22102>(*this, k, f);
    }

    __aicore__ inline bool updatePanelAndPrepare(uint32_t kb, uint32_t f, uint32_t h, uint32_t c0, uint32_t width) {
        return LdetSharedOperations::updatePanelAndPrepareShared52<FullFallbackC22102>(*this, kb, f, h, c0, width);
    }

    template <bool EXPORT = true> __aicore__ inline void factor(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::factorShared53<FullFallbackC22102, EXPORT>(*this, kb, q);
    }

    template <uint32_t FIRST = 0U, uint32_t LAST = B>
    __aicore__ inline void ApplyPanelUpdates(uint32_t kb, uint32_t q) {
        return LdetSharedOperations::ApplyPanelUpdatesShared54<FullFallbackC22102, FIRST, LAST>(*this, kb, q);
    }

    __aicore__ inline void copyInput(bool valid) {
        return LdetSharedOperations::copyInputShared55<FullFallbackC22102>(*this, valid);
    }

    __aicore__ inline void FinishLocalTailGepp() {
        return LdetSharedOperations::FinishLocalTailGeppShared56<FullFallbackC22102>(*this);
    }

    __aicore__ inline void FinishLocalTailBlock2() {
        return LdetSharedOperations::FinishLocalTailBlock2Shared57<FullFallbackC22102>(*this);
    }

    __aicore__ inline void FinishLocalTail() {
        return LdetSharedOperations::FinishLocalTailShared58<FullFallbackC22102>(*this);
    }
