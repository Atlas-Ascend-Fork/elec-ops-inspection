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
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunDuplicatePhase62(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                               const bool &valid, GM_ADDR &workspace) {

        matrix = WAVED ? matrixIndex : GetBlockIdx() / G;
        blocks = n / B;
        owned = blocks > rank ? (blocks - 1 - rank) / G + 1 : 0;
        if constexpr (ClassPlan46::CONTIGUOUS_BANDS) {
            const uint32_t first = bandFirst(rank);
            owned = (ClassPlan46::HETERO6 && rank < 6U)
                        ? 1U
                        : (blocks > first ? (blocks - first < 2U ? blocks - first : 2U) : 0U);
        }
        LdetSharedOperations::SharedOwnerScalarUpdateStage2<CyclicN256Seven, ClassPlan46::GROUP,
                                                            ClassPlan46::PRUNE_STAGE_WAITS>(*this);
        LdetSharedOperations::SharedOwnerDuplicateStage30<CyclicN256Seven, ClassPlan46::EXTRA, ClassPlan46::PANEL_MASK>(
            *this, input, output, valid, workspace);
    }

    __aicore__ inline void RunconstexprPhase63(uint32_t &kb, uint32_t &owner, const bool &prefixConsumer, uint32_t &q,
                                               const bool &valid) {
        return LdetSharedOperations::RunconstexprPhase63SharedPhase17<CyclicN256Seven, ClassPlan46::CONTIGUOUS_BANDS>(
            *this, kb, owner, prefixConsumer, q, valid);
    }
    template <bool ONE_INPUT_DMA>
    __aicore__ inline void RunDataCopyPhase64DataCopyDepth22Stage1(const bool &splitNextInput, const bool &valid) {

        if constexpr (ONE_INPUT_DMA) {
            const uint32_t inputWidth = ClassPlan46::HETERO6 ? owned * B : 2U * B;
            if (valid) {
                if (splitNextInput)
                    DataCopy(mat[8U], in[bandFirst(rank) * B + 8U],
                             {static_cast<uint16_t>(n), 3U, static_cast<uint16_t>((n - 24U) / 8U),
                              static_cast<uint16_t>((S - 24U) / 8U)});
                else
                    DataCopy(mat, in[bandFirst(rank) * B],
                             {static_cast<uint16_t>(n), static_cast<uint16_t>(inputWidth / 8U),
                              static_cast<uint16_t>((n - inputWidth) / 8U),
                              static_cast<uint16_t>((S - inputWidth) / 8U)});
            }
        } else {
            LdetSharedOperations::CopyOwnedBands60<CyclicN256Seven, ClassPlan46::CONTIGUOUS_BANDS>(*this, valid);
        }
        MV();
        VS();
    }
    template <bool ONE_INPUT_DMA>
    __aicore__ inline void RunDataCopyPhase64(const bool &firstInputOwner, const bool &splitNextInput,
                                              const bool &valid) {

        LdetSharedOperations::SharedOwnerScalarUpdateStage34<CyclicN256Seven>(*this, firstInputOwner);
        if (!firstInputOwner) {
            RunDataCopyPhase64DataCopyDepth22Stage1<ONE_INPUT_DMA>(splitNextInput, valid);
        }
    }
    template <bool ONE_INPUT_DMA>
    __aicore__ inline void RunDataCopyPhase65(const bool &firstInputOwner, const bool &splitNextInput,
                                              const bool &valid) {

        if (splitNextInput) {
            if (valid)
                DataCopy(mat, in[bandFirst(rank) * B],
                         {static_cast<uint16_t>(n), 1U, static_cast<uint16_t>((n - 8U) / 8U),
                          static_cast<uint16_t>((S - 8U) / 8U)});
        }
        if (firstInputOwner) {
            LdetSharedOperations::SharedOwnerDataCopyStage18<CyclicN256Seven, ClassPlan46::CONTIGUOUS_BANDS,
                                                             ClassPlan46::HETERO6, ClassPlan46::THIN_H6, ONE_INPUT_DMA>(
                *this, valid);
        }
    }
    template <bool ONE_INPUT_DMA> __aicore__ inline void RunPipeBarrierPhase66(const bool &valid) {

        const bool firstInputOwner =
            GetBlockNum() / G > 3U || ((!ClassPlan46::HETERO6 || ClassPlan46::THIN_H6) && rank == 0U);
        const bool splitNextInput = ClassPlan46::THIN_H6 && rank == 1U && GetBlockNum() / G <= 3U;
        RunDataCopyPhase65<ONE_INPUT_DMA>(firstInputOwner, splitNextInput, valid);
        LdetSharedOperations::SharedOwnerPipeBarrierStage23<CyclicN256Seven, ClassPlan46::HETERO6,
                                                            ClassPlan46::THIN_H6>(*this, valid);
        PipeBarrier<PIPE_V>();
        Adds(tab[128U], tab, int32_t(128U * P * 4U), 128);
        PipeBarrier<PIPE_V>();
        LDET_MASK_SET(n);
        RunDataCopyPhase64<ONE_INPUT_DMA>(firstInputOwner, splitNextInput, valid);
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunDuplicatePhase67(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                               const bool &valid, GM_ADDR &workspace) {

        static_assert(!PREFIX_STREAM || !WAVED, "G6 prefix singlewave q0 freshness");
        n = NMAX;
        rank = GetBlockIdx() % G;
        RunDuplicatePhase62<WAVED>(input, matrixIndex, output, valid, workspace);
        Duplicate(fac, 0.0f, B * NMAX);
        LdetSharedOperations::SharedOwner15DuplicateStage6<CyclicN256Seven, ClassPlan46::CONTIGUOUS_BANDS,
                                                           ClassPlan46::HETERO6, ClassPlan46::PANEL_MASK,
                                                           ClassPlan46::THIN_H6>(*this);
        RunPipeBarrierPhase66<ONE_INPUT_DMA>(valid);
    }

    __aicore__ inline void RunconstexprPhase68constexprDepth22Stage1(uint32_t &q, const bool &valid) {

        uint32_t kb = q * B, owner = ClassPlan46::CONTIGUOUS_BANDS ? (ClassPlan46::HETERO6 ? q : q / 2U) : q % G;
        const bool prefixConsumer = PREFIX_STREAM && rank == q + 1U;
        if constexpr (!ClassPlan46::HETERO6 || ClassPlan46::THIN_H6) {
            LdetSharedOperations::FactorAndPublish64<CyclicN256Seven, PREFIX_STREAM>(*this, kb, q, owner, valid);
        }
        RunconstexprPhase63(kb, owner, prefixConsumer, q, valid);
    }

    __aicore__ inline void RunconstexprPhase68(uint32_t &rounds, const bool &valid) {

        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            RunconstexprPhase68constexprDepth22Stage1(q, valid);
        }
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = !WAVED || active;
        RunDuplicatePhase67<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        uint32_t rounds = ClassPlan46::UB_FINISH64 ? 7U : blocks;
        RunconstexprPhase68(rounds, valid);
        if constexpr (ClassPlan46::UB_FINISH64) {
            LdetSharedOperations::SharedOwner26DataCopyPadStage7<CyclicN256Seven, ClassPlan46::HETERO6,
                                                                 ClassPlan46::THIN_H6, WAVED>(*this, valid);
            return;
        }
        LdetSharedOperations::SharedOwnerDataCopyPadStage21<CyclicN256Seven, ClassPlan46::CONTIGUOUS_BANDS,
                                                            ClassPlan46::EXTRA, ClassPlan46::FINAL_NO_EXPORT, WAVED>(
            *this, rounds, valid);
    }
