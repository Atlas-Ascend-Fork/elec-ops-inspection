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
// Included in the CyclicRecycled45 class; preserves its member scope and access.
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunGetBlockIdxPhase11(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                                 const bool &valid, GM_ADDR &workspace) {
        return LdetSharedOperations::RunGetBlockIdxPhase11SharedPhase3<
            CyclicRecycled45, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::GROUP, ClassPlan46::HETERO3,
            ClassPlan46::HETERO5, ClassPlan46::PRUNE_STAGE_WAITS, WAVED, PhaseArg1>(*this, input, matrixIndex, output,
                                                                                    valid, workspace);
    }

    __aicore__ inline void RunfactorPhase12(const bool &actualPrefixConsumer, uint32_t &kb, uint32_t &owner,
                                            uint32_t &q, const bool &valid) {
        return LdetSharedOperations::RunfactorPhase12SharedPhase4<CyclicRecycled45, ClassPlan46::HETERO5>(
            *this, actualPrefixConsumer, kb, owner, q, valid);
    }
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunDuplicatePhase13(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                               const bool &valid, GM_ADDR &workspace) {

        static_assert(!PREFIX_STREAM || !WAVED, "prefix requires original single wave q0 freshness barrier");
        recycleState = 0U;
        n = NMAX;
        RunGetBlockIdxPhase11<WAVED>(input, matrixIndex, output, valid, workspace);
        LdetSharedOperations::SharedOwnerDuplicateStage17<CyclicRecycled45, ClassPlan46::CONTIGUOUS_BANDS,
                                                          ClassPlan46::EXTRA, ClassPlan46::HETERO3,
                                                          ClassPlan46::HETERO5, ClassPlan46::PANEL_MASK>(*this);
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunDataCopyPhase14(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                              const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase13<WAVED>(input, matrixIndex, output, valid, workspace);
        LdetSharedOperations::SharedOwnerDataCopyStage14<CyclicRecycled45, ClassPlan46::CONTIGUOUS_BANDS,
                                                         ClassPlan46::HETERO3, ClassPlan46::HETERO5, ONE_INPUT_DMA>(
            *this, valid);
    }

    __aicore__ inline void RunbandFirstPhase15(const bool &actualPrefixConsumer, uint32_t &kb, uint32_t &owner,
                                               uint32_t &q, const bool &valid) {

        LdetSharedOperations::SharedOwner26ScalarUpdateStage8<CyclicRecycled45, ClassPlan46::CONTIGUOUS_BANDS>(
            *this, actualPrefixConsumer, kb, owner, q, valid);
        if (valid && !singular) {
            LdetSharedOperations::PrefixAwarePanelUpdate64<CyclicRecycled45, PREFIX_STREAM>(*this, kb, q,
                                                                                            actualPrefixConsumer);
        }
    }
    __aicore__ inline void RunDataCopyPhase16(uint32_t &q) {
        return LdetSharedOperations::RunDataCopyPhase16SharedPhase5<CyclicRecycled45>(*this, q);
    }

    __aicore__ inline void RunDataCopyPhase17DataCopyDepth22Stage1(uint32_t &q) {

        if constexpr (G == 5U) {
            LdetSharedOperations::RecycleFiveWay56<CyclicRecycled45>(*this, q);
        } else {
            RunDataCopyPhase16(q);
        }
    }

    __aicore__ inline void RunDataCopyPhase17RunfactorPhase12Depth22Stage2(const bool &actualPrefixConsumer,
                                                                           uint32_t &kb, uint32_t &owner, uint32_t &q,
                                                                           const bool &valid) {

        RunfactorPhase12(actualPrefixConsumer, kb, owner, q, valid);
        if (valid) {
            RunDataCopyPhase17DataCopyDepth22Stage1(q);
        }
        RunbandFirstPhase15(actualPrefixConsumer, kb, owner, q, valid);
    }

    __aicore__ inline auto RunDataCopyPhase17IndexSelect22Stage3(uint32_t &q)
        -> decltype(ClassPlan46::HETERO5 ? (q == 4U ? 0U : (q == 5U ? 4U : q))
                                         : (q == 1U ? 1U : (q == 2U ? 0U : (q == 3U ? 2U : (q == 4U ? 1U : 3U))))) {
        return LdetSharedOperations::SelectPrefixConsumer32<ClassPlan46::HETERO5>(q);
    }

    __aicore__ inline void RunDataCopyPhase17(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        const bool actualPrefixConsumer = PREFIX_STREAM && q > 0U && rank == (RunDataCopyPhase17IndexSelect22Stage3(q));
        RunDataCopyPhase17RunfactorPhase12Depth22Stage2(actualPrefixConsumer, kb, owner, q, valid);
    }

    __aicore__ inline void RunDataCopyPhase18(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        if (valid && !singular) {
            if constexpr (G == 5U) {
                if (q == 2U && rank == 3U) {
                    VM();
                    DataCopy(relay, mat[64U * S + 32U], {192U, 4U, (S - 32U) / 8U, 0U});
                    MS();
                }
            } else {
                LdetSharedOperations::SharedOwnerDataCopyStage32<CyclicRecycled45>(*this, q);
            }
        }
        RunDataCopyPhase17(kb, owner, q, valid);
    }
    template <bool WAVED> __aicore__ inline void RunDataCopyPadPhase19(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase19SharedPhase6<
            CyclicRecycled45, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA, ClassPlan46::FINAL_NO_EXPORT, WAVED>(
            *this, rounds, valid);
    }
    inline void RunPipeBarrierPhase20() {
        return LdetSharedOperations::RunPipeBarrierPhase20SharedPhase7<CyclicRecycled45>(*this);
    }
    __aicore__ inline void RunPipeBarrierPhase21(const bool &valid) {

        LdetSharedOperations::SharedOwner26ScalarUpdateStage9<CyclicRecycled45, ClassPlan46::HETERO5>(*this, valid);
        if constexpr (ClassPlan46::EARLY_TAIL_INDEX) {
            RunPipeBarrierPhase20();
        } else {
            LdetSharedOperations::SharedOwnerPipeBarrierStage9<CyclicRecycled45>(*this);
            LDET_MASK_RST();
        }
        MV();
        if constexpr ((ClassPlan46::GROUP != 5U && !ClassPlan46::PHASE_TRIM) || !ClassPlan46::PANEL_MASK)
            VS();
    }
    template <bool WAVED> __aicore__ inline void RunDataCopyPadPhase22(const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase22SharedPhase8<CyclicRecycled45, ClassPlan46::HETERO3,
                                                                       ClassPlan46::HETERO5, WAVED>(*this, valid);
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class Depth22Arg13_1>
    __aicore__ inline void RunRunDataCopyPhase14Depth22Stage1(GM_ADDR &input, Depth22Arg13_1 &matrixIndex,
                                                              GM_ADDR &output, const bool &valid, GM_ADDR &workspace) {

        RunDataCopyPhase14<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        RunPipeBarrierPhase21(valid);
        if constexpr (!ClassPlan46::COMPACT && !ClassPlan46::PRUNE_STAGE_WAITS)
            SyncAll<true>();
    }

    __aicore__ inline auto RunIndexSelect22Stage4(uint32_t &q)
        -> decltype(G == 5U ? (q < 2U ? 0U : (q < 5U ? q - 1U : 0U))
                            : (q < 2U ? 0U : (q == 2U ? 1U : (q == 3U ? 0U : (q == 4U ? 2U : 1U))))) {
        return LdetSharedOperations::SelectPanelOwner32<G>(q);
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = !WAVED || active;
        RunRunDataCopyPhase14Depth22Stage1<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        uint32_t rounds = ClassPlan46::UB_FINISH64 ? 7U : blocks;
        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            uint32_t kb = q * B, owner = RunIndexSelect22Stage4(q);
            RunDataCopyPhase18(kb, owner, q, valid);
        }
        if constexpr (ClassPlan46::UB_FINISH64) {
            return RunDataCopyPadPhase22<WAVED>(valid);
        }
        RunDataCopyPadPhase19<WAVED>(rounds, valid);
    }
