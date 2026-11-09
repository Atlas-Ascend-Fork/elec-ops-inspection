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
// Included in the CyclicN256Inplace120 class; preserves its member scope and access.
    __aicore__ inline void RunDataCopyPhase46(uint32_t &q, const bool &valid) {

        if (valid && q == 2U) {
            rebalanced = 2U;
            owned = 2U;
            if (rank == 0U && !singular && (!LOOKAHEAD_Q3 || !prefactored3)) {
                DataCopy(mat[64U * S + 32U], relay[224U * 32U], {192U, 4U, 0U, (S - 32U) / 8U});
                MV();
            }
            if (rank == 1U) {
                matShift = 64U;
                mat = LocalTensor<float>(TPosition::VECCALC, MAT + matShift * 4U, NMAX * S);
                if (!singular) {
                    DataCopy(mat[64U * S + 32U], relay, {192U, 4U, 0U, (S - 32U) / 8U});
                    MV();
                }
            }
            if (rank == 2U) {
                matShift = 32U;
                mat = LocalTensor<float>(TPosition::VECCALC, MAT + matShift * 4U, NMAX * S);
            }
        }
        if (valid && q == 3U) {
            rebalanced = 3U;
            if (rank == 0U) {
                owned = 1U;
                matShift = 0U;
                mat = LocalTensor<float>(TPosition::VECCALC, MAT, NMAX * S);
                if (!singular) {
                    DataCopy(mat[96U * S], relay, {160U, 4U, 0U, (S - 32U) / 8U});
                    MV();
                }
            } else if (rank == 1U) {
                owned = 1U;
            }
        }
    }

    __aicore__ inline void RunDataCopyPhase47(uint32_t &q, const bool &valid) {

        if (LOOKAHEAD_Q3 && valid && q == 2U && rank == 0U && !singular) {
            CrossCoreSetFlag<0, PIPE_MTE3>(14);
            waitTags();
            rebalanced = 2U;
            owned = 2U;
            DataCopy(mat[64U * S + 32U], relay[224U * 32U], {192U, 4U, 0U, (S - 32U) / 8U});
            MV();
            ApplyPanelUpdates(64U, 2U);
            factor(96U, 3U);
            publish(96U);
            prefactored3 = true;
            CrossCoreWaitFlag<0>(14);
        } else {
            if constexpr (READY_ALL_TRIM) {
                CrossCoreSetFlag<0, PIPE_MTE3>(14);
                CrossCoreWaitFlag<0>(14);
            } else
                SyncAll<true>();
        }
        if (valid && q == 1U) {
            rebalanced = 1U;
            owned = rank == 0U ? 1U : (rank == 1U ? 2U : 3U);
            matShift = rank == 1U ? 32U : 0U;
            mat = LocalTensor<float>(TPosition::VECCALC, MAT + matShift * 4U, NMAX * S);
            if (rank == 0U && !singular) {
                DataCopy(mat[32U * S], relay, {224U, 4U, 0U, (S - 32U) / 8U});
                MV();
            }
        }
        RunDataCopyPhase46(q, valid);
    }

    __aicore__ inline void RunDuplicatePhase48(GM_ADDR &input, GM_ADDR &output, const bool &valid, GM_ADDR &workspace) {

        meta = LocalTensor<float>(TPosition::VECCALC, META, META_WORDS);
        mu = meta.template ReinterpretCast<uint32_t>();
        const uint32_t addressMatrix = valid ? matrix : 0U;
        in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input) +
                           (ACTIVE12_B11 && matrix == 11U ? 0U : addressMatrix) * n * n);
        ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + addressMatrix * (FAC_SLOTS * GM_WORDS));
        wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) + addressMatrix * (FAC_SLOTS * GM_WORDS));
        out.SetGlobalBuffer((ACTIVE12_B11 && matrix == 11U)
                                ? reinterpret_cast<__gm__ float *>(workspace) +
                                      (GetBlockNum() / G) * (FAC_SLOTS * GM_WORDS + 224U * 64U)
                                : reinterpret_cast<__gm__ float *>(output) + addressMatrix);
        relay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) +
                              (GetBlockNum() / G) * (FAC_SLOTS * GM_WORDS) + addressMatrix * (224U * 64U));
        sign = 1.0f;
        prod = 1.0f;
        ex = 0;
        singular = 0;
        if constexpr (LOOKAHEAD_Q3) {
            prefactored3 = false;
            if constexpr (DISJOINT_SIGNAL) {
                signalBuf = LocalTensor<uint32_t>(TPosition::VECCALC, END, 16U);
                static_assert(END + 64U <= 196352, "disjoint signal UB");
            }
            tags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) +
                                 (GetBlockNum() / G) * (FAC_SLOTS * GM_WORDS) + addressMatrix * (224U * 64U) + 14304U);
        }
        if constexpr (!EXTRA && !PANEL_MASK)
            Duplicate(mat, 0.0f, NMAX * S);
        Duplicate(fac, 0.0f, B * NMAX);
        Duplicate(col, 0.0f, NMAX);
        if constexpr (GUARDED) {
            Duplicate(panel[NMAX * P], 0.0f, GUARD_ROWS * P);
            Duplicate(brc, 0.0f, 8 * (NMAX + GUARD_COEFF));
        }
    }
    template <bool WAVED, class PhaseArg1>
    __aicore__ inline void RunPipeBarrierPhase49(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                                 const bool &valid, GM_ADDR &workspace) {

        rebalanced = false;
        matShift = 0U;
        n = NMAX;
        rank = GetBlockIdx() % G;
        matrix = WAVED ? matrixIndex : GetBlockIdx() / G;
        blocks = n / B;
        owned = blocks > rank ? (blocks - 1 - rank) / G + 1 : 0;
        owned = rank == 0U ? 2U : 3U;
        mat = LocalTensor<float>(TPosition::VECCALC, MAT, NMAX * S);
        panel = LocalTensor<float>(TPosition::VECCALC, (GROUP == 8U && PRUNE_STAGE_WAITS) ? MAT : PANEL,
                                   P * (NMAX + GUARD_ROWS));
        LdetSharedOperations::SharedOwnerScalarUpdateStage19<CyclicN256Inplace120>(*this);
        RunDuplicatePhase48(input, output, valid, workspace);
        if constexpr (!PANEL_MASK) {
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(4);
            WaitFlag<HardEvent::V_MTE2>(4);
        } else {
            static_assert((G == 4U || HETERO5 || HETERO3) && S == 120U && B == 32U && CONTIGUOUS_BANDS,
                          "validownedcolumnsfullyoverwritten");
        }
    }

    __aicore__ inline void RunDataCopyPhase50(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        if (valid && q == 2U && rank == 2U && !singular) {
            VM();
            DataCopy(relay, mat[64U * S], {192U, 4U, (S - 32U) / 8U, 0U});
            MS();
        }
        if (valid && q == 3U && rank == 1U && !singular) {
            VM();
            DataCopy(relay, mat[96U * S + 32U], {160U, 4U, (S - 32U) / 8U, 0U});
            MS();
        }
        if (valid && rank == owner && !(LOOKAHEAD_Q3 && q == 3U && prefactored3)) {
            if (!singular)
                factor(kb, q);
            publish(kb);
        }
        RunDataCopyPhase47(q, valid);
    }

    __aicore__ inline void RunbandFirstPhase51(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        RunDataCopyPhase50(kb, owner, q, valid);
        if (valid && rank != owner && (CONTIGUOUS_BANDS ? bandFirst(rank) + owned - 1U : rank + (owned - 1U) * G) > q)
            receive(kb);
        if (LOOKAHEAD_Q3 && FAC_SLOTS == 2U && valid && rank == 2U && q == 1U)
            notifyTag(1U, 1U);
        if (valid && !singular && !(LOOKAHEAD_Q3 && q == 2U && rank == 0U && prefactored3))
            ApplyPanelUpdates(kb, q);
    }

    __aicore__ inline void RunDataCopyPhase52(uint32_t &rounds, const bool &valid) {

        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            uint32_t kb = q * B, owner = q < 4U ? 0U : (q == 4U ? 1U : 0U);
            if (valid && q == 1U && rank == 1U && !singular) {
                VM();
                DataCopy(relay, mat[32U * S], {224U, 4U, (S - 32U) / 8U, 0U});
                MS();
            }
            if (valid && q == 2U && rank == 1U && !singular) {
                VM();
                DataCopy(relay[224U * 32U], mat[64U * S], {192U, 4U, (S - 32U) / 8U, 0U});
                MS();
            }
            if (LOOKAHEAD_Q3 && valid && q == 2U && rank == 1U)
                notifyTag(0U, 1U);
            RunbandFirstPhase51(kb, owner, q, valid);
        }
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunDataCopyPhase53(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                              const bool &valid, GM_ADDR &workspace) {

        RunPipeBarrierPhase49<WAVED>(input, matrixIndex, output, valid, workspace);
        if constexpr (ONE_INPUT_DMA) {
            static_assert(CONTIGUOUS_BANDS && (G == 4U || HETERO5 || HETERO3) && B == 32U && S == 120U,
                          "single input DMA scope");
            const uint32_t inputWidth = owned * B;
            if (valid)
                DataCopy(mat, in[bandFirst(rank) * B],
                         {static_cast<uint16_t>(n), static_cast<uint16_t>(inputWidth / 8U),
                          static_cast<uint16_t>((n - inputWidth) / 8U), static_cast<uint16_t>((S - inputWidth) / 8U)});
        } else {
            if (valid)
                for (uint32_t j = 0; j < owned; ++j)
                    DataCopy(mat[j * B], in[(CONTIGUOUS_BANDS ? bandFirst(rank) + j : rank + j * G) * B],
                             {static_cast<uint16_t>(n), B / 8U, static_cast<uint16_t>((n - B) / 8), (S - B) / 8U});
        }
    }
    template <bool WAVED> __aicore__ inline void RunDataCopyPadPhase54(uint32_t &rounds, const bool &valid) {

        const uint32_t last = rounds - 1U;
        if (valid && rank == (CONTIGUOUS_BANDS ? last / 2U : last % G)) {
            if (!singular) {
                if constexpr (EXTRA || FINAL_NO_EXPORT)
                    factor<false>(last * B, last);
                else
                    factor(last * B, last);
            }
            meta.SetValue(0, singular ? 0.0f : sign * final());
            SM();
            DataCopyPad(out, meta, DataCopyExtParams{1, 4, 0, 0, 0});
            MS();
        }
        if constexpr (WAVED)
            SyncAll<true>();
    }
    inline void RunPipeBarrierPhase55() {

        if (rank == 2U) {
            PrepareTail64Indices();
        } else {
            LdetSharedOperations::SharedOwner26PipeBarrierStage2<CyclicN256Inplace120>(*this);
            Adds(tab[128U], tab, int32_t(128U * P * 4U), 128);
            PipeBarrier<PIPE_V>();
            LDET_MASK_SET(n);
            for (uint32_t r = 0; r < 8; ++r)
                vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[r * NMAX].GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
            LDET_MASK_RST();
        }
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunPipeBarrierPhase56(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                                 const bool &valid, GM_ADDR &workspace) {

        RunDataCopyPhase53<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        if constexpr (EARLY_TAIL_INDEX) {
            RunPipeBarrierPhase55();
        } else {
            LdetSharedOperations::SharedOwner15PipeBarrierStage1<CyclicN256Inplace120>(*this);
            LDET_MASK_RST();
        }
        MV();
        VS();
        if (LOOKAHEAD_Q3 && valid && (rank == 1U || (FAC_SLOTS == 2U && rank == 2U)))
            notifyTag(rank - 1U, 0U);
        if constexpr (!COMPACT && !PRUNE_STAGE_WAITS)
            SyncAll<true>();
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false, bool PAD11 = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = PAD11 ? (GetBlockIdx() / G < 11U) : (!WAVED || active);
        RunPipeBarrierPhase56<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        uint32_t rounds = UB_FINISH64 ? 7U : blocks;
        RunDataCopyPhase52(rounds, valid);
        if constexpr (UB_FINISH64) {
            if (valid && rank == (HETERO3 ? 2U : (HETERO5 ? 4U : 3U))) {
                LdetSharedOperations::FinishOrWriteZero64(*this);
            }
            if constexpr (WAVED)
                SyncAll<true>();
            return;
        }
        RunDataCopyPadPhase54<WAVED>(rounds, valid);
    }
