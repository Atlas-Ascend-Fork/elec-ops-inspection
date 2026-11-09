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
    __aicore__ inline void RunDuplicatePhase97(GM_ADDR &input, GM_ADDR &output, const bool &valid, GM_ADDR &workspace) {

        row = LocalTensor<float>(TPosition::VECCALC, ROW, S);
        prow = LocalTensor<float>(TPosition::VECCALC, PROW, B);
        tmp = LocalTensor<float>(TPosition::VECCALC, TMP, 256);
        tab = LocalTensor<int32_t>(TPosition::VECCALC, TAB, NMAX);
        lanes = LocalTensor<int32_t>(TPosition::VECCALC, LANE, 8 * NMAX);
        meta = LocalTensor<float>(TPosition::VECCALC, META, META_WORDS);
        mu = meta.template ReinterpretCast<uint32_t>();
        LdetSharedOperations::SharedOwnerDuplicateStage30<CyclicN256ReuseFirstTail, ClassPlan46::EXTRA,
                                                          ClassPlan46::PANEL_MASK>(*this, input, output, valid,
                                                                                   workspace);
        LdetSharedOperations::SharedOwnerDuplicateStage31<CyclicN256ReuseFirstTail, ClassPlan46::CONTIGUOUS_BANDS,
                                                          ClassPlan46::HETERO6, ClassPlan46::PANEL_MASK,
                                                          ClassPlan46::THIN_H6>(*this);
    }
    __aicore__ inline void Shared36DataCopyStage5(const bool &valid) {
        const uint32_t inputWidth = B;
        if (valid)
            DataCopy(mat, in[bandFirst(rank) * B],
                     {static_cast<uint16_t>(n), static_cast<uint16_t>(inputWidth / 8U),
                      static_cast<uint16_t>((n - inputWidth) / 8U), static_cast<uint16_t>((S - inputWidth) / 8U)});
    }
    template <bool ONE_INPUT_DMA> __aicore__ inline void RunPipeBarrierPhase98DataCopyDepth22Stage1(const bool &valid) {

        if constexpr (ONE_INPUT_DMA) {
            Shared36DataCopyStage5(valid);
        } else {
            LdetSharedOperations::CopyOwnedBands60<CyclicN256ReuseFirstTail, ClassPlan46::CONTIGUOUS_BANDS>(*this,
                                                                                                            valid);
        }
        MV();
        VS();
    }
    template <bool ONE_INPUT_DMA>
    __aicore__ inline void RunPipeBarrierPhase98(const bool &firstInputOwner, const bool &valid) {

        PipeBarrier<PIPE_V>();
        Adds(tab[128U], tab, int32_t(128U * P * 4U), 128);
        PipeBarrier<PIPE_V>();
        LDET_MASK_SET(n);
        LdetSharedOperations::SharedOwnerScalarUpdateStage34<CyclicN256ReuseFirstTail>(*this, firstInputOwner);
        if (valid && rank == 0U)
            DataCopy(mat[B], in[6U * B], {static_cast<uint16_t>(n), 8U, 24U, static_cast<uint16_t>((S - 64U) / 8U)});
        if (!firstInputOwner) {
            RunPipeBarrierPhase98DataCopyDepth22Stage1<ONE_INPUT_DMA>(valid);
        }
    }

    __aicore__ inline void RunSyncAllPhase99(uint32_t &kb, uint32_t &owner, const bool &prefixConsumer, uint32_t &q,
                                             const bool &valid) {

        SyncAll<true>();
        if (valid && rank != owner && (rank == 0U || rank > q)) {
            if constexpr (PREFIX_STREAM) {
                if (prefixConsumer)
                    receiveRange<16U, 32U>(kb);
                else
                    receive(kb);
            } else
                receive(kb);
        }
        if (valid && rank == 0U && q == 0U)
            MV();
        if (valid && !singular) {
            if constexpr (PREFIX_STREAM) {
                if (prefixConsumer)
                    ApplyPanelUpdates<16U, 32U>(kb, q);
                else
                    ApplyPanelUpdates(kb, q);
            } else
                ApplyPanelUpdates(kb, q);
        }
    }

    __aicore__ inline void RunconstexprPhase100constexprDepth22Stage1(uint32_t &q, const bool &valid) {

        uint32_t kb = q * B, owner = ClassPlan46::CONTIGUOUS_BANDS ? (ClassPlan46::HETERO6 ? q : q / 2U) : q % G;
        const bool prefixConsumer = PREFIX_STREAM && rank == (q == 5U ? 0U : q + 1U);
        if constexpr (!ClassPlan46::HETERO6 || ClassPlan46::THIN_H6) {
            LdetSharedOperations::FactorAndPublish64<CyclicN256ReuseFirstTail, PREFIX_STREAM>(*this, kb, q, owner,
                                                                                              valid);
        }
        if constexpr (PREFIX_STREAM) {
            if (valid && prefixConsumer) {
                LuPollN256Prefix(prefixTicket(q), 0x6A1F0010U);
                receiveRange<0U, 16U>(kb);
                if (!singular)
                    ApplyPanelUpdates<0U, 16U>(kb, q);
            }
        }
        RunSyncAllPhase99(kb, owner, prefixConsumer, q, valid);
    }

    __aicore__ inline void RunconstexprPhase100(uint32_t &rounds, const bool &valid) {

        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            RunconstexprPhase100constexprDepth22Stage1(q, valid);
        }
    }
    template <bool ONE_INPUT_DMA> __aicore__ inline void RunDataCopyPhase101DataCopyDepth22Stage1(const bool &valid) {

        if constexpr (ONE_INPUT_DMA) {
            static_assert(ClassPlan46::CONTIGUOUS_BANDS && (G == 4U || ClassPlan46::HETERO6) && B == 32U &&
                              (S == 64U || ClassPlan46::THIN_H6),
                          "single input DMA scope");
            Shared36DataCopyStage5(valid);
        } else {
            LdetSharedOperations::CopyOwnedBands60<CyclicN256ReuseFirstTail, ClassPlan46::CONTIGUOUS_BANDS>(*this,
                                                                                                            valid);
        }
    }
    template <bool ONE_INPUT_DMA>
    __aicore__ inline void RunDataCopyPhase101(const bool &firstInputOwner, const bool &valid) {

        if (firstInputOwner) {
            RunDataCopyPhase101DataCopyDepth22Stage1<ONE_INPUT_DMA>(valid);
        }
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunPipeBarrierPhase102(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                                  const bool &valid, GM_ADDR &workspace) {

        n = NMAX;
        rank = GetBlockIdx() % G;
        matrix = WAVED ? matrixIndex : GetBlockIdx() / G;
        blocks = n / B;
        owned = blocks > rank ? (blocks - 1 - rank) / G + 1 : 0;
        if constexpr (ClassPlan46::CONTIGUOUS_BANDS) {
            owned = rank == 0U ? 3U : 1U;
        }
        mat = LocalTensor<float>(TPosition::VECCALC, MAT, NMAX * S);
        panel = LocalTensor<float>(TPosition::VECCALC,
                                   (ClassPlan46::GROUP == 8U && ClassPlan46::PRUNE_STAGE_WAITS) ? MAT : PANEL,
                                   P * (NMAX + GUARD_ROWS));
        fac = LocalTensor<float>(TPosition::VECCALC, FAC, B * NMAX + META_WORDS);
        brc = LocalTensor<float>(TPosition::VECCALC, BRC, 8 * (NMAX + GUARD_COEFF));
        col = LocalTensor<float>(TPosition::VECCALC, COL, NMAX);
        ab = LocalTensor<float>(TPosition::VECCALC, ABS, NMAX);
        RunDuplicatePhase97(input, output, valid, workspace);
        const bool firstInputOwner =
            GetBlockNum() / G > 3U || ((!ClassPlan46::HETERO6 || ClassPlan46::THIN_H6) && rank == 0U);
        RunDataCopyPhase101<ONE_INPUT_DMA>(firstInputOwner, valid);
        LdetSharedOperations::SharedOwnerPipeBarrierStage23<CyclicN256ReuseFirstTail, ClassPlan46::HETERO6,
                                                            ClassPlan46::THIN_H6>(*this, valid);
        RunPipeBarrierPhase98<ONE_INPUT_DMA>(firstInputOwner, valid);
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = !WAVED || active;
        static_assert(!PREFIX_STREAM || !WAVED, "G6 prefix singlewave q0 freshness");
        RunPipeBarrierPhase102<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        uint32_t rounds = ClassPlan46::UB_FINISH64 ? 7U : blocks;
        RunconstexprPhase100(rounds, valid);
        if constexpr (ClassPlan46::UB_FINISH64) {
            if (valid && rank == 0U) {
                LdetSharedOperations::FinishOrWriteZero64(*this);
            }
            if constexpr (WAVED)
                SyncAll<true>();
            return;
        }
        LdetSharedOperations::SharedOwnerDataCopyPadStage21<CyclicN256ReuseFirstTail, ClassPlan46::CONTIGUOUS_BANDS,
                                                            ClassPlan46::EXTRA, ClassPlan46::FINAL_NO_EXPORT, WAVED>(
            *this, rounds, valid);
    }
