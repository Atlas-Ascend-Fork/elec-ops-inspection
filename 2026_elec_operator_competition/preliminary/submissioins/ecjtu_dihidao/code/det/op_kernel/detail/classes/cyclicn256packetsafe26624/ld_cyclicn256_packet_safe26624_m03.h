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
// Included in the CyclicN256PacketSafe26624 class; preserves its member scope and access.
    __aicore__ inline void RunconstexprPhase90(GM_ADDR &input, GM_ADDR &output, const bool &valid, GM_ADDR &workspace) {

        LdetSharedOperations::SharedOwnerScalarUpdateStage6<CyclicN256PacketSafe26624, ClassPlan46::CONTIGUOUS_BANDS,
                                                            ClassPlan46::GROUP, ClassPlan46::HETERO6,
                                                            ClassPlan46::PRUNE_STAGE_WAITS>(*this);
        fu = fac.template ReinterpretCast<uint32_t>();
        bindPayloadMeta(0U);
        LdetSharedOperations::SharedOwnerScalarUpdateStage7<CyclicN256PacketSafe26624, ClassPlan46::HETERO6,
                                                            ClassPlan46::THIN_H6>(*this, input, output, valid,
                                                                                  workspace);
    }
    template <bool ONE_INPUT_DMA>
    __aicore__ inline void RunDataCopyPhase91(const bool &firstInputOwner, const bool &splitNextInput,
                                              const bool &valid) {
        return LdetSharedOperations::RunDataCopyPhase70SharedPhase19<
            CyclicN256PacketSafe26624, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::HETERO6, ClassPlan46::THIN_H6,
            ONE_INPUT_DMA>(*this, firstInputOwner, splitNextInput, valid);
    }
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunDuplicatePhase92(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                               const bool &valid, GM_ADDR &workspace) {

        n = NMAX;
        rank = GetBlockIdx() % G;
        matrix = WAVED ? matrixIndex : GetBlockIdx() / G;
        blocks = n / B;
        owned = blocks > rank ? (blocks - 1 - rank) / G + 1 : 0;
        RunconstexprPhase90(input, output, valid, workspace);
        prod = 1.0f;
        ex = 0;
        singular = 0;
        if constexpr (ClassPlan46::THIN_H6) {
            prefixReady = LocalTensor<uint32_t>(TPosition::VECCALC, END, 32U);
            for (uint32_t j = 0; j < 4U; ++j) {
                const uint32_t t = j == 3U ? 0x6A1F8000U : 0x6A1F0000U + 8U * (j + 1U);
                for (uint32_t i = 0; i < 8U; ++i)
                    prefixReady.SetValue(j * 8U + i, t);
            }
        }
        if constexpr (!ClassPlan46::EXTRA && !ClassPlan46::PANEL_MASK)
            Duplicate(mat, 0.0f, NMAX * S);
        Duplicate(fac, 0.0f, B * NMAX + META_WORDS);
        LdetSharedOperations::SharedOwner15DuplicateStage6<CyclicN256PacketSafe26624, ClassPlan46::CONTIGUOUS_BANDS,
                                                           ClassPlan46::HETERO6, ClassPlan46::PANEL_MASK,
                                                           ClassPlan46::THIN_H6>(*this);
    }

    __aicore__ inline void RunconstexprPhase93(uint32_t &kb, uint32_t &owner, const bool &prefixConsumer, uint32_t &q,
                                               const bool &valid) {

        if constexpr (!ClassPlan46::HETERO6 || ClassPlan46::THIN_H6) {
            if (valid && rank == owner) {
                if (!singular)
                    factor(kb, q);
                else if constexpr (PREFIX_STREAM)
                    publishAbort(kb);
                publish(kb);
            }
        }
        if constexpr (PREFIX_STREAM) {
            if (valid && prefixConsumer) {
                bool aborted = false;
                consumePacket<0U>(kb, q, aborted);
                consumePacket<1U>(kb, q, aborted);
                consumePacket<2U>(kb, q, aborted);
            }
        }
        CrossCoreSetFlag<0, PIPE_MTE3>(15U);
        CrossCoreWaitFlag<0>(15U);
        if (valid && rank != owner &&
            (ClassPlan46::CONTIGUOUS_BANDS ? bandFirst(rank) + owned - 1U : rank + (owned - 1U) * G) > q) {
            if constexpr (PREFIX_STREAM) {
                if (prefixConsumer)
                    receiveFinalPacket(kb);
                else
                    receive(kb);
            } else
                receive(kb);
        }
    }
    template <bool ONE_INPUT_DMA>
    __aicore__ inline void RunDataCopyPhase94(const bool &firstInputOwner, const bool &splitNextInput,
                                              const bool &valid) {
        return LdetSharedOperations::RunDataCopyPhase73SharedPhase20<
            CyclicN256PacketSafe26624, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::HETERO6, ONE_INPUT_DMA>(
            *this, firstInputOwner, splitNextInput, valid);
    }
    template <bool WAVED> __aicore__ inline void RunDataCopyPadPhase95(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase19SharedPhase6<
            CyclicN256PacketSafe26624, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA, ClassPlan46::FINAL_NO_EXPORT,
            WAVED>(*this, rounds, valid);
    }

    __aicore__ inline void RunconstexprPhase96(uint32_t &rounds, const bool &valid) {

        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            uint32_t kb = q * B, owner = ClassPlan46::CONTIGUOUS_BANDS ? (ClassPlan46::HETERO6 ? q : q / 2U) : q % G;
            const bool prefixConsumer = PREFIX_STREAM && rank == q + 1U;
            RunconstexprPhase93(kb, owner, prefixConsumer, q, valid);
            if (valid && !singular) {
                if constexpr (PREFIX_STREAM) {
                    if (prefixConsumer)
                        ApplyPanelUpdates<24U, 32U>(kb, q);
                    else
                        ApplyPanelUpdates(kb, q);
                } else
                    ApplyPanelUpdates(kb, q);
            }
        }
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = !WAVED || active;
        static_assert(!PREFIX_STREAM || !WAVED, "G6 prefix singlewave q0 freshness");
        RunDuplicatePhase92<WAVED>(input, matrixIndex, output, valid, workspace);
        const bool firstInputOwner =
            GetBlockNum() / G > 3U || ((!ClassPlan46::HETERO6 || ClassPlan46::THIN_H6) && rank == 0U);
        const bool splitNextInput = ClassPlan46::THIN_H6 && rank == 1U && GetBlockNum() / G <= 3U;
        RunDataCopyPhase91<ONE_INPUT_DMA>(firstInputOwner, splitNextInput, valid);
        Adds(tab[128U], tab, int32_t(128U * P * 4U), 128);
        PipeBarrier<PIPE_V>();
        LDET_MASK_SET(n);
        for (uint32_t r = 0; r < 8; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[r * NMAX].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
        RunDataCopyPhase94<ONE_INPUT_DMA>(firstInputOwner, splitNextInput, valid);
        uint32_t rounds = ClassPlan46::UB_FINISH64 ? 7U : blocks;
        RunconstexprPhase96(rounds, valid);
        if constexpr (ClassPlan46::UB_FINISH64) {
            LdetSharedOperations::SharedOwner26DataCopyPadStage7<CyclicN256PacketSafe26624, ClassPlan46::HETERO6,
                                                                 ClassPlan46::THIN_H6, WAVED>(*this, valid);
            return;
        }
        RunDataCopyPadPhase95<WAVED>(rounds, valid);
    }
