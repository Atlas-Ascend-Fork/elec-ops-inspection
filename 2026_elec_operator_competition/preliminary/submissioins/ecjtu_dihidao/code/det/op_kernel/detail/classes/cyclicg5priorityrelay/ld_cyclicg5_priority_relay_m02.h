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
// Included in the CyclicG5PriorityRelay class; preserves its member scope and access.
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunGetBlockIdxPhase23(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                                 const bool &valid, GM_ADDR &workspace) {
        return LdetSharedOperations::RunGetBlockIdxPhase11SharedPhase3<
            CyclicG5PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::GROUP, ClassPlan46::HETERO3,
            ClassPlan46::HETERO5, ClassPlan46::PRUNE_STAGE_WAITS, WAVED, PhaseArg1>(*this, input, matrixIndex, output,
                                                                                    valid, workspace);
    }

    __aicore__ inline void RunfactorPhase24(const bool &actualPrefixConsumer, uint32_t &kb, uint32_t &owner,
                                            uint32_t &q, const bool &valid) {
        return LdetSharedOperations::RunfactorPhase12SharedPhase4<CyclicG5PriorityRelay, ClassPlan46::HETERO5>(
            *this, actualPrefixConsumer, kb, owner, q, valid);
    }

    __aicore__ inline void RunbandFirstPhase25(const bool &actualPrefixConsumer, uint32_t &kb, uint32_t &owner,
                                               uint32_t &q, const bool &valid) {

        LdetSharedOperations::SharedOwner26ScalarUpdateStage8<CyclicG5PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS>(
            *this, actualPrefixConsumer, kb, owner, q, valid);
        if (valid && !singular) {
            if (q == 1U && rank == 3U)
                ApplyPriorityRelay32(kb, q);
            else {
                LdetSharedOperations::PrefixAwarePanelUpdate64<CyclicG5PriorityRelay, PREFIX_STREAM>(
                    *this, kb, q, actualPrefixConsumer);
            }
        }
    }
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunDuplicatePhase26(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                               const bool &valid, GM_ADDR &workspace) {

        static_assert(!PREFIX_STREAM || !WAVED, "prefix requires original single wave q0 freshness barrier");
        recycleState = 0U;
        n = NMAX;
        RunGetBlockIdxPhase23<WAVED>(input, matrixIndex, output, valid, workspace);
        LdetSharedOperations::SharedOwnerDuplicateStage17<CyclicG5PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS,
                                                          ClassPlan46::EXTRA, ClassPlan46::HETERO3,
                                                          ClassPlan46::HETERO5, ClassPlan46::PANEL_MASK>(*this);
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunDataCopyPhase27(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                              const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase26<WAVED>(input, matrixIndex, output, valid, workspace);
        LdetSharedOperations::SharedOwner15DataCopyStage3<CyclicG5PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS,
                                                          ClassPlan46::HETERO3, ClassPlan46::HETERO5, ONE_INPUT_DMA>(
            *this, valid);
    }
    __aicore__ inline void RunDataCopyPhase28(uint32_t &q) {
        return LdetSharedOperations::RunDataCopyPhase16SharedPhase5<CyclicG5PriorityRelay>(*this, q);
    }

    __aicore__ inline void RunDataCopyPhase29DataCopyDepth22Stage1(uint32_t &q) {

        if constexpr (G == 5U) {
            LdetSharedOperations::RecycleFiveWay56<CyclicG5PriorityRelay>(*this, q);
        } else {
            RunDataCopyPhase28(q);
        }
    }

    __aicore__ inline void RunDataCopyPhase29RunfactorPhase24Depth22Stage2(const bool &actualPrefixConsumer,
                                                                           uint32_t &kb, uint32_t &owner, uint32_t &q,
                                                                           const bool &valid) {

        RunfactorPhase24(actualPrefixConsumer, kb, owner, q, valid);
        if (valid) {
            RunDataCopyPhase29DataCopyDepth22Stage1(q);
        }
        RunbandFirstPhase25(actualPrefixConsumer, kb, owner, q, valid);
    }

    __aicore__ inline auto RunDataCopyPhase29IndexSelect22Stage5(uint32_t &q)
        -> decltype(ClassPlan46::HETERO5 ? (q == 4U ? 0U : (q == 5U ? 4U : q))
                                         : (q == 1U ? 1U : (q == 2U ? 0U : (q == 3U ? 2U : (q == 4U ? 1U : 3U))))) {
        return LdetSharedOperations::SelectPrefixConsumer32<ClassPlan46::HETERO5>(q);
    }

    __aicore__ inline void RunDataCopyPhase29(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        const bool actualPrefixConsumer = PREFIX_STREAM && q > 0U && rank == (RunDataCopyPhase29IndexSelect22Stage5(q));
        RunDataCopyPhase29RunfactorPhase24Depth22Stage2(actualPrefixConsumer, kb, owner, q, valid);
    }

    __aicore__ inline void RunDataCopyPhase30constexprDepth22Stage1(uint32_t &kb, uint32_t &owner, uint32_t &q,
                                                                    const bool &valid) {

        if (valid && !singular) {
            if constexpr (G == 5U) {
            } else {
                LdetSharedOperations::SharedOwnerDataCopyStage32<CyclicG5PriorityRelay>(*this, q);
            }
        }
        RunDataCopyPhase29(kb, owner, q, valid);
    }

    __aicore__ inline auto RunDataCopyPhase30IndexSelect22Stage6(uint32_t &q)
        -> decltype(G == 5U ? (q < 2U ? 0U : (q < 5U ? q - 1U : 0U))
                            : (q < 2U ? 0U : (q == 2U ? 1U : (q == 3U ? 0U : (q == 4U ? 2U : 1U))))) {
        return LdetSharedOperations::SelectPanelOwner32<G>(q);
    }

    __aicore__ inline void RunDataCopyPhase30(uint32_t &rounds, const bool &valid) {

        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            uint32_t kb = q * B, owner = RunDataCopyPhase30IndexSelect22Stage6(q);
            RunDataCopyPhase30constexprDepth22Stage1(kb, owner, q, valid);
        }
    }
    template <bool WAVED> __aicore__ inline void RunDataCopyPadPhase31(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase19SharedPhase6<
            CyclicG5PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA, ClassPlan46::FINAL_NO_EXPORT,
            WAVED>(*this, rounds, valid);
    }
    inline void RunPipeBarrierPhase32() {
        return LdetSharedOperations::RunPipeBarrierPhase20SharedPhase7<CyclicG5PriorityRelay>(*this);
    }
    __aicore__ inline void RunPipeBarrierPhase33(const bool &valid) {

        if constexpr (PREFIX_STREAM) {
            if constexpr (ClassPlan46::HETERO5) {
                static_assert(192U * 32U * 4U + 6U * 64U <= 224U * 32U * 4U, "G5 tickets inside relay slack");
                if (valid)
                    LuPublish(prefixTicket(rank + 1U), 0U);
            } else if (valid)
                for (uint32_t t = rank; t < 6U; t += G)
                    LuPublish(prefixTicket(t), 0U);
        }
        if constexpr (ClassPlan46::EARLY_TAIL_INDEX) {
            RunPipeBarrierPhase32();
        } else {
            LdetSharedOperations::SharedOwner15PipeBarrierStage1<CyclicG5PriorityRelay>(*this);
            LDET_MASK_RST();
        }
        MV();
        if constexpr ((ClassPlan46::GROUP != 5U && !ClassPlan46::PHASE_TRIM) || !ClassPlan46::PANEL_MASK)
            VS();
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = !WAVED || active;
        RunDataCopyPhase27<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        RunPipeBarrierPhase33(valid);
        if constexpr (!ClassPlan46::COMPACT && !ClassPlan46::PRUNE_STAGE_WAITS)
            SyncAll<true>();
        uint32_t rounds = ClassPlan46::UB_FINISH64 ? 7U : blocks;
        RunDataCopyPhase30(rounds, valid);
        if constexpr (ClassPlan46::UB_FINISH64) {
            LdetSharedOperations::SharedOwner26DataCopyPadStage10<CyclicG5PriorityRelay, ClassPlan46::HETERO3,
                                                                  ClassPlan46::HETERO5, WAVED>(*this, valid);
            return;
        }
        RunDataCopyPadPhase31<WAVED>(rounds, valid);
    }
