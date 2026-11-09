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
// Included in the CyclicRowLU class; preserves its member scope and access.
    template <bool ONE_INPUT_DMA> __aicore__ inline void RunDataCopyPhase7(const bool &valid) {

        LdetSharedOperations::SharedOwner15DataCopyStage3<CyclicRowLU, CONTIGUOUS_BANDS, HETERO3, HETERO5,
                                                          ONE_INPUT_DMA>(*this, valid);
        LdetSharedOperations::SharedOwner15PipeBarrierStage4<CyclicRowLU>(*this);
        for (uint32_t r = 0; r < 8; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(lanes[r * NMAX].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        MV();
        VS();
    }

    __aicore__ inline void RunDuplicatePhase8(GM_ADDR &input, GM_ADDR &output, const bool &valid, GM_ADDR &workspace) {

        blocks = n / B;
        owned = blocks > rank ? (blocks - 1 - rank) / G + 1 : 0;
        if constexpr (CONTIGUOUS_BANDS) {
            const uint32_t first = bandFirst(rank);
            const uint32_t maxOwned = HETERO3 ? 3U : 2U;
            owned = (HETERO5 && (rank == 1U || rank == 2U))
                        ? 1U
                        : (blocks > first ? (blocks - first < maxOwned ? blocks - first : maxOwned) : 0U);
        }
        mat = LocalTensor<float>(TPosition::VECCALC, MAT, NMAX * S);
        panel = LocalTensor<float>(TPosition::VECCALC, (GROUP == 8U && PRUNE_STAGE_WAITS) ? MAT : PANEL,
                                   P * (NMAX + GUARD_ROWS));
        LdetSharedOperations::SharedOwnerScalarUpdateStage19<CyclicRowLU>(*this);
        meta = LocalTensor<float>(TPosition::VECCALC, META, META_WORDS);
        mu = meta.template ReinterpretCast<uint32_t>();
        const uint32_t addressMatrix = valid ? matrix : 0U;
        in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input) + addressMatrix * n * n);
        ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + addressMatrix * (2U * GM_WORDS));
        wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) + addressMatrix * (2U * GM_WORDS));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + addressMatrix);
        sign = 1.0f;
        prod = 1.0f;
        ex = 0;
        singular = 0;
        if constexpr (!EXTRA && !PANEL_MASK)
            Duplicate(mat, 0.0f, NMAX * S);
    }

    __aicore__ inline void RunfactorPhase9factorDepth22Stage1(uint32_t &kb, uint32_t &owner, uint32_t &q,
                                                              const bool &valid) {

        if (valid && rank == owner) {
            if (!singular)
                factor(kb, q);
            publish(kb);
        }
        SyncAll<true>();
        if (valid && rank != owner && (CONTIGUOUS_BANDS ? bandFirst(rank) + owned - 1U : rank + (owned - 1U) * G) > q)
            receive(kb);
        if (valid && !singular)
            ApplyPanelUpdates(kb, q);
    }

    __aicore__ inline auto RunfactorPhase9IndexSelect22Stage2(uint32_t &q)
        -> decltype(CONTIGUOUS_BANDS ? (HETERO3 ? q / 3U : (HETERO5 ? (q < 2U ? 0U : (q < 4U ? q - 1U : 3U)) : q / 2U))
                                     : q % G) {
        if (CONTIGUOUS_BANDS) {
            if (HETERO3) {
                return q / 3U;
            }
            if (HETERO5) {
                if (q < 2U) {
                    return 0U;
                }
                if (q < 4U) {
                    return q - 1U;
                }
                return 3U;
            }
            return q / 2U;
        }
        return q % G;
    }

    __aicore__ inline void RunfactorPhase9(uint32_t &rounds, const bool &valid) {

        for (uint32_t q = 0; q + 1U < rounds; ++q) {
            uint32_t kb = q * B, owner = RunfactorPhase9IndexSelect22Stage2(q);
            RunfactorPhase9factorDepth22Stage1(kb, owner, q, valid);
        }
    }
    template <bool WAVED, bool ONE_INPUT_DMA, class PhaseArg1>
    __aicore__ inline void RunDuplicatePhase10(GM_ADDR &input, PhaseArg1 &matrixIndex, GM_ADDR &output,
                                               const bool &valid, GM_ADDR &workspace) {

        n = NMAX;
        rank = GetBlockIdx() % G;
        matrix = WAVED ? matrixIndex : GetBlockIdx() / G;
        RunDuplicatePhase8(input, output, valid, workspace);
        LdetSharedOperations::SharedOwner15DuplicateStage5<CyclicRowLU, CONTIGUOUS_BANDS, HETERO3, HETERO5, PANEL_MASK>(
            *this);
        RunDataCopyPhase7<ONE_INPUT_DMA>(valid);
        if constexpr (!COMPACT && !PRUNE_STAGE_WAITS)
            SyncAll<true>();
    }
    template <bool WAVED = false, bool ONE_INPUT_DMA = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true) {
        const bool valid = !WAVED || active;
        RunDuplicatePhase10<WAVED, ONE_INPUT_DMA>(input, matrixIndex, output, valid, workspace);
        uint32_t rounds = UB_FINISH64 ? 7U : blocks;
        RunfactorPhase9(rounds, valid);
        if constexpr (UB_FINISH64) {
            LdetSharedOperations::SharedOwner26DataCopyPadStage10<CyclicRowLU, HETERO3, HETERO5, WAVED>(*this, valid);
            return;
        }
        LdetSharedOperations::SharedOwnerDataCopyPadStage21<CyclicRowLU, CONTIGUOUS_BANDS, EXTRA, FINAL_NO_EXPORT,
                                                            WAVED>(*this, rounds, valid);
    }
