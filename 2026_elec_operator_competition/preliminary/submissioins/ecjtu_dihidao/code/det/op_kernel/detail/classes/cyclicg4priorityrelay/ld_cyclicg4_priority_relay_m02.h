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
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunGetBlockIdxPhase34(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                                 const bool &valid, GM_ADDR &workspace) {
        return LdetSharedOperations::RunGetBlockIdxPhase11SharedPhase3<
            CyclicG4PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::GROUP, ClassPlan46::HETERO3,
            ClassPlan46::HETERO5, ClassPlan46::PRUNE_STAGE_WAITS, WAVED, PhaseArg1>(*this, input, matrixIndex, output,
                                                                                    valid, workspace);
    }

    __aicore__ inline void RunbandFirstPhase35ApplyPriorityRelay32Depth22Stage1(const bool &actualPrefixConsumer,
                                                                                uint32_t &kb, const bool &priorityRelay,
                                                                                uint32_t &q, const bool &valid) {

        if (valid && !singular) {
            if (priorityRelay && ((q == 0U && rank == 1U) || (q == 2U && rank == 2U)))
                ApplyPriorityRelay32(kb, q);
            else {
                LdetSharedOperations::PrefixAwarePanelUpdate64<CyclicG4PriorityRelay, PREFIX_STREAM>(
                    *this, kb, q, actualPrefixConsumer);
            }
        }
    }

    __aicore__ inline void RunbandFirstPhase35bandFirstDepth22Stage2(const bool &actualPrefixConsumer, uint32_t &kb,
                                                                     uint32_t &owner, uint32_t &q, const bool &valid) {

        if (valid && rank != owner &&
            (ClassPlan46::CONTIGUOUS_BANDS ? bandFirst(rank) + owned - 1U : rank + (owned - 1U) * G) > q) {
            if constexpr (PREFIX_STREAM) {
                if (actualPrefixConsumer)
                    receiveRange<16U, 32U>(kb);
                else
                    receive(kb);
            } else
                receive(kb);
        }
    }

    __aicore__ inline void RunbandFirstPhase35(const bool &actualPrefixConsumer, uint32_t &kb, uint32_t &owner,
                                               const bool &priorityRelay, uint32_t &q, const bool &valid) {

        RunbandFirstPhase35bandFirstDepth22Stage2(actualPrefixConsumer, kb, owner, q, valid);
        RunbandFirstPhase35ApplyPriorityRelay32Depth22Stage1(actualPrefixConsumer, kb, priorityRelay, q, valid);
    }

    __aicore__ inline void RunfactorPhase36(const bool &actualPrefixConsumer, uint32_t &kb, uint32_t &owner,
                                            uint32_t &q, const bool &valid) {
        return LdetSharedOperations::RunfactorPhase12SharedPhase4<CyclicG4PriorityRelay, ClassPlan46::HETERO5>(
            *this, actualPrefixConsumer, kb, owner, q, valid);
    }
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunDuplicatePhase37(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                               const bool &valid, GM_ADDR &workspace) {

        static_assert(!PREFIX_STREAM || !WAVED, "prefix requires original single wave q0 freshness barrier");
        recycleState = 0U;
        n = NMAX;
        RunGetBlockIdxPhase34<WAVED>(input, matrixIndex, output, valid, workspace);
        sign = 1.0f;
        prod = 1.0f;
        ex = 0;
        singular = 0;
        if constexpr (!ClassPlan46::EXTRA && !ClassPlan46::PANEL_MASK)
            Duplicate(mat, 0.0f, NMAX * S);
        LdetSharedOperations::SharedOwner15DuplicateStage5<CyclicG4PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS,
                                                           ClassPlan46::HETERO3, ClassPlan46::HETERO5,
                                                           ClassPlan46::PANEL_MASK>(*this);
    }

    __aicore__ inline void RunDataCopyPhase38(const bool &priorityRelay, uint32_t &q, const bool &valid) {

        if (valid && !singular) {
            if constexpr (G == 5U) {
                if (q == 2U && rank == 3U) {
                    VM();
                    DataCopy(relay, mat[64U * S + 32U], {192U, 4U, (S - 32U) / 8U, 0U});
                    MS();
                }
            } else {
                if (!priorityRelay && q == 1U && rank == 1U) {
                    VM();
                    DataCopy(relay, mat[32U * S + 32U], {224U, 4U, (S - 32U) / 8U, 0U});
                    MS();
                }
                if (!priorityRelay && q == 3U && rank == 2U) {
                    VM();
                    DataCopy(relay, mat[96U * S + 32U], {160U, 4U, (S - 32U) / 8U, 0U});
                    MS();
                }
            }
        }
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunDataCopyPhase39(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                              const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase37<WAVED>(input, matrixIndex, output, valid, workspace);
        LdetSharedOperations::SharedOwnerDataCopyStage14<CyclicG4PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS,
                                                         ClassPlan46::HETERO3, ClassPlan46::HETERO5, ONE_INPUT_DMA>(
            *this, valid);
    }
    __aicore__ inline void RunDataCopyPhase40(uint32_t &q) {
        return LdetSharedOperations::RunDataCopyPhase16SharedPhase5<CyclicG4PriorityRelay>(*this, q);
    }

    __aicore__ inline void RunDataCopyPhase41DataCopyDepth22Stage1(uint32_t &q) {

        if constexpr (G == 5U) {
            LdetSharedOperations::RecycleFiveWay56<CyclicG4PriorityRelay>(*this, q);
        } else {
            RunDataCopyPhase40(q);
        }
    }

    __aicore__ inline void RunDataCopyPhase41RunfactorPhase36Depth22Stage2(const bool &actualPrefixConsumer,
                                                                           uint32_t &kb, uint32_t &owner,
                                                                           const bool &priorityRelay, uint32_t &q,
                                                                           const bool &valid) {

        RunfactorPhase36(actualPrefixConsumer, kb, owner, q, valid);
        if (valid) {
            RunDataCopyPhase41DataCopyDepth22Stage1(q);
        }
        RunbandFirstPhase35(actualPrefixConsumer, kb, owner, priorityRelay, q, valid);
    }

    __aicore__ inline auto RunDataCopyPhase41IndexSelect22Stage7(uint32_t &q)
        -> decltype(ClassPlan46::HETERO5 ? (q == 4U ? 0U : (q == 5U ? 4U : q))
                                         : (q == 1U ? 1U : (q == 2U ? 0U : (q == 3U ? 2U : (q == 4U ? 1U : 3U))))) {
        return LdetSharedOperations::SelectPrefixConsumer32<ClassPlan46::HETERO5>(q);
    }

    __aicore__ inline void RunDataCopyPhase41(uint32_t &kb, uint32_t &owner, const bool &priorityRelay, uint32_t &q,
                                              const bool &valid) {

        RunDataCopyPhase38(priorityRelay, q, valid);
        const bool actualPrefixConsumer = PREFIX_STREAM && q > 0U && rank == (RunDataCopyPhase41IndexSelect22Stage7(q));
        RunDataCopyPhase41RunfactorPhase36Depth22Stage2(actualPrefixConsumer, kb, owner, priorityRelay, q, valid);
    }
    template <bool WAVED> __aicore__ inline void RunDataCopyPadPhase42(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase19SharedPhase6<
            CyclicG4PriorityRelay, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA, ClassPlan46::FINAL_NO_EXPORT,
            WAVED>(*this, rounds, valid);
    }
    inline void RunPipeBarrierPhase43() {
        return LdetSharedOperations::RunPipeBarrierPhase20SharedPhase7<CyclicG4PriorityRelay>(*this);
    }
    __aicore__ inline void RunPipeBarrierPhase44(const bool &valid) {

        LdetSharedOperations::SharedOwner26ScalarUpdateStage9<CyclicG4PriorityRelay, ClassPlan46::HETERO5>(*this,
                                                                                                           valid);
        if constexpr (ClassPlan46::EARLY_TAIL_INDEX) {
            RunPipeBarrierPhase43();
        } else {
            LdetSharedOperations::SharedOwnerPipeBarrierStage9<CyclicG4PriorityRelay>(*this);
            LDET_MASK_RST();
        }
        MV();
        if constexpr ((ClassPlan46::GROUP != 5U && !ClassPlan46::PHASE_TRIM) || !ClassPlan46::PANEL_MASK)
            VS();
    }
    template <bool WAVED> __aicore__ inline void RunDataCopyPadPhase45(const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase22SharedPhase8<CyclicG4PriorityRelay, ClassPlan46::HETERO3,
                                                                       ClassPlan46::HETERO5, WAVED>(*this, valid);
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class Depth22Arg21_1>
    __aicore__ inline void RunRunDataCopyPhase39Depth22Stage2(GM_ADDR &input, Depth22Arg21_1 &matrixIndex,
                                                              GM_ADDR &output, const bool &valid, GM_ADDR &workspace) {

        RunDataCopyPhase39<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        RunPipeBarrierPhase44(valid);
        if constexpr (!ClassPlan46::COMPACT && !ClassPlan46::PRUNE_STAGE_WAITS)
            SyncAll<true>();
    }

    __aicore__ inline auto RunIndexSelect22Stage8(uint32_t &q)
        -> decltype(G == 5U ? (q < 2U ? 0U : (q < 5U ? q - 1U : 0U))
                            : (q < 2U ? 0U : (q == 2U ? 1U : (q == 3U ? 0U : (q == 4U ? 2U : 1U))))) {
        return LdetSharedOperations::SelectPanelOwner32<G>(q);
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = !WAVED || active;
        const bool priorityRelay = GetBlockNum() / G >= 10U;
        RunRunDataCopyPhase39Depth22Stage2<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        uint32_t rounds = ClassPlan46::UB_FINISH64 ? 7U : blocks;
        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            uint32_t kb = q * B, owner = RunIndexSelect22Stage8(q);
            RunDataCopyPhase41(kb, owner, priorityRelay, q, valid);
        }
        if constexpr (ClassPlan46::UB_FINISH64) {
            return RunDataCopyPadPhase45<WAVED>(valid);
        }
        RunDataCopyPadPhase42<WAVED>(rounds, valid);
    }
