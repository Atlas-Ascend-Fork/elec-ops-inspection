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
// Included in the CyclicRowLUMedium class; preserves its member scope and access.
    template <bool WAVED>
    __aicore__ inline void RunDuplicatePhase112(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                const bool &valid, GM_ADDR &workspace) {

        in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input) +
                           static_cast<uint64_t>(addressMatrix) * inputDim * inputDim);
        ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + addressMatrix * (2U * GM_WORDS));
        wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) + addressMatrix * (2U * GM_WORDS));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + addressMatrix);
        if constexpr (PREFIX_STREAM) {
            static_assert(!WAVED, "prefix ticket freshness relies on the original first ready join");
            prefixFlag =
                workspace + ((3U * 2U * GM_WORDS * 4U + 63U) & ~63U) + addressMatrix * (SPLIT_OWNERS ? 256U : 128U);
            if (valid && rank == 0U) {
                for (uint32_t i = 0U; i < (SPLIT_OWNERS ? 4U : 2U); ++i)
                    LuPublish(prefixFlag + i * 64U, 0U);
            }
        }
        sign = 1.0f;
        prod = 1.0f;
        ex = 0;
        singular = 0;
        if constexpr (!EXTRA)
            Duplicate(mat, 0.0f, NMAX * S);
        Duplicate(fac, 0.0f, B * NMAX);
        Duplicate(col, 0.0f, NMAX);
        if constexpr ((DIM == 192U || DIM == 200U || DIM == 208U)) {
            Duplicate(panel[NMAX * PANEL_PITCH], 0.0f, GUARD_ROWS * PANEL_PITCH);
            if constexpr (INPLACE_PANEL || (BALANCED200 && LOCAL_TAIL))
                Duplicate(mat[NMAX * S], 0.0f, GUARD_ROWS * S);
            Duplicate(brc, 0.0f, 8 * (NMAX + GUARD_COEFF));
        }
        PipeBarrier<PIPE_V>();
        if constexpr (INPUT_OVERLAP)
            copyInput(valid);
    }

    __aicore__ inline void RunPipeBarrierPhase113(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {
        return LdetSharedOperations::RunPipeBarrierPhase106SharedPhase25<CyclicRowLUMedium, EXTRA, PAIR_LOCAL,
                                                                         PREFIX_STREAM, RETIRE_RECEIVE, SPLIT_OWNERS>(
            *this, kb, owner, q, valid);
    }
    __aicore__ inline void RunLDET_MASK_SETPhase114(const bool &valid) {

        LDET_MASK_SET(n);
        for (uint32_t r = 0; r < 8; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[r * NMAX].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        if constexpr (!INPUT_OVERLAP)
            copyInput(valid);
        MV();
        VS();
        if constexpr (PAD_INPUT) {
            static_assert(DIM == 208U && !CONTIGUOUS_BANDS && !EXTRA && !INPUT_OVERLAP, "identity padding scope");
            for (uint32_t j = 0; j < owned; ++j) {
                const uint32_t first = (rank + j * G) * B;
                for (uint32_t r = inputDim; r < NMAX; ++r)
                    if (r >= first && r < first + B)
                        mat.SetValue(r * S + j * B + r - first, 1.0f);
            }
            SV();
        }
        if constexpr (!COMPACT || SPLIT_OWNERS)
            SyncAll<true>();
    }
    template <bool WAVED, class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ inline void RunGetBlockIdxPhase115(PhaseArg0 &matrixIndex, PhaseArg1 &originalDim,
                                                  PhaseArg2 &physicalGroup) {

        inputDim = PAD_INPUT ? originalDim : NMAX;
        n = NMAX;
        rank = GetBlockIdx() % physicalGroup;
        matrix = WAVED ? matrixIndex : GetBlockIdx() / physicalGroup;
        blocks = (n + B - 1U) / B;
        owned = blocks > rank ? (blocks - 1 - rank) / G + 1 : 0;
        if constexpr (CONTIGUOUS_BANDS) {
            const uint32_t first = bandFirst(rank), count = bandCount(rank);
            owned = blocks > first ? (blocks - first < count ? blocks - first : count) : 0U;
        }
        mat = LocalTensor<float>(TPosition::VECCALC, MAT,
                                 (NMAX + ((INPLACE_PANEL || (BALANCED200 && LOCAL_TAIL)) ? GUARD_ROWS : 0U)) * S);
        panel = LocalTensor<float>(TPosition::VECCALC, PANEL, PANEL_PITCH * (NMAX + GUARD_ROWS));
        fac = LocalTensor<float>(TPosition::VECCALC, FAC, B * NMAX);
        brc = LocalTensor<float>(TPosition::VECCALC, BRC, 8 * (NMAX + GUARD_COEFF));
        col = LocalTensor<float>(TPosition::VECCALC, COL, NMAX);
        ab = LocalTensor<float>(TPosition::VECCALC, ABS, NMAX);
        row = LocalTensor<float>(TPosition::VECCALC, ROW, S);
        prow = LocalTensor<float>(TPosition::VECCALC, PROW, B);
        tmp = LocalTensor<float>(TPosition::VECCALC, TMP, TMP_BYTES / 4U);
        tab = LocalTensor<int32_t>(TPosition::VECCALC, TAB, NMAX);
        lanes = LocalTensor<int32_t>(TPosition::VECCALC, LANE, 8 * NMAX);
        meta = LocalTensor<float>(TPosition::VECCALC, META, META_WORDS);
        mu = meta.template ReinterpretCast<uint32_t>();
    }
    __aicore__ inline void Runstatic_assertPhase116(const bool &valid) {

        static_assert(CONTIGUOUS_BANDS && (((GROUP == 3U || SPLIT_OWNERS) && LOCAL_CUT == 128U &&
                                            (DIM == 192U || DIM == 200U || NATIVE208)) ||
                                           (GROUP == 2U && ((DIM == 192U && LOCAL_CUT == 96U) || PAIR200))),
                      "local tail route scope");
        if (valid && rank == G - 1U)
            FinishLocalTail();
        return;
    }

    __aicore__ inline void RunconstexprPhase117(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        LdetSharedOperations::SharedOwner26ScalarUpdateStage3<CyclicRowLUMedium, PREFIX_STREAM, SPLIT_OWNERS>(*this, kb,
                                                                                                              q, valid);
        RunPipeBarrierPhase113(kb, owner, q, valid);
    }

    __aicore__ inline void RunSetGlobalBufferPhase118(const uint32_t &addressMatrix, uint32_t &kb, uint32_t &owner,
                                                      uint32_t &q, const bool &valid, GM_ADDR &workspace) {

        LdetSharedOperations::SharedOwnerScalarUpdateStage28<CyclicRowLUMedium, PREFIX_STREAM, SPLIT_OWNERS>(
            *this, addressMatrix, kb, owner, q, valid, workspace);
        RunconstexprPhase117(kb, owner, q, valid);
        if (valid && !singular) {
            LdetSharedOperations::SharedOwner26ScalarUpdateStage5<CyclicRowLUMedium, PREFIX_STREAM, SPLIT_OWNERS>(
                *this, kb, q);
        }
    }
    template <bool WAVED>
    __aicore__ inline void RunPipeBarrierPhase119(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                  const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase112<WAVED>(addressMatrix, input, output, valid, workspace);
        LdetSharedOperations::SharedOwnerPipeBarrierStage13<CyclicRowLUMedium, INDEX_SEEDS>(*this);
        RunLDET_MASK_SETPhase114(valid);
    }

    __aicore__ inline void RunDataCopyPadPhase120(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase111SharedPhase28<CyclicRowLUMedium, CONTIGUOUS_BANDS, EXTRA>(
            *this, rounds, valid);
    }
    template <bool WAVED = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true, uint32_t originalDim = NMAX, uint32_t physicalGroup = G) {
        if constexpr (DIM == 200U && GROUP == 3U && COMPACT && LOCAL_TAIL && LOCAL_CUT == 128U) {
            if (physicalGroup > G && GetBlockIdx() % physicalGroup >= G) {
                for (uint32_t q = 0U; q < LOCAL_CUT / B; ++q)
                    SyncAll<true>();
                return;
            }
        }
        const bool valid = !WAVED || active;
        RunGetBlockIdxPhase115<WAVED>(matrixIndex, originalDim, physicalGroup);
        const uint32_t addressMatrix = valid ? matrix : 0U;
        RunPipeBarrierPhase119<WAVED>(addressMatrix, input, output, valid, workspace);
        uint32_t rounds = blocks;
        for (uint32_t q = 0; q < (LOCAL_TAIL ? LOCAL_CUT / B : rounds - 1U); ++q) {
            uint32_t kb = q * B, owner = CONTIGUOUS_BANDS ? ownerOf(q) : q % G;
            RunSetGlobalBufferPhase118(addressMatrix, kb, owner, q, valid, workspace);
        }
        if constexpr (LOCAL_TAIL) {
            return Runstatic_assertPhase116(valid);
        }
        RunDataCopyPadPhase120(rounds, valid);
    }
