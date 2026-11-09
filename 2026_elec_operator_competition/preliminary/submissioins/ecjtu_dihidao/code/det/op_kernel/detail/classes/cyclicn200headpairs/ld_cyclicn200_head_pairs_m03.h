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
// Included in the CyclicN200HeadPairs class; preserves its member scope and access.
    template <bool WAVED>
    __aicore__ inline void RunDuplicatePhase121(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                const bool &valid, GM_ADDR &workspace) {
        return LdetSharedOperations::RunDuplicatePhase121SharedPhase29<
            CyclicN200HeadPairs, ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPLACE_PANEL,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::LOCAL_TAIL, ClassPlan46::PANEL_PITCH, ClassPlan46::PREFIX_STREAM,
            ClassPlan46::SPLIT_OWNERS, WAVED>(*this, addressMatrix, input, output, valid, workspace);
    }
    template <bool WAVED> __aicore__ inline void RunLDET_MASK_SETPhase122(const bool &valid) {
        return LdetSharedOperations::RunLDET_MASK_SETPhase122SharedPhase30<
            CyclicN200HeadPairs, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT, ClassPlan46::PREFIX_STREAM, ClassPlan46::SPLIT_OWNERS,
            WAVED>(*this, valid);
    }
    template <bool WAVED>
    __aicore__ inline void RunPipeBarrierPhase123(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                  const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase121<WAVED>(addressMatrix, input, output, valid, workspace);
        LdetSharedOperations::SharedOwnerPipeBarrierStage35<CyclicN200HeadPairs, ClassPlan46::INDEX_SEEDS>(*this);
        RunLDET_MASK_SETPhase122<WAVED>(valid);
        LdetSharedOperations::SharedOwnerScalarUpdateStage36<
            CyclicN200HeadPairs, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::COMPACT, ClassPlan46::CONTIGUOUS_BANDS,
            ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT,
            ClassPlan46::SPLIT_OWNERS>(*this);
    }
    template <bool WAVED, class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ inline void RunGetBlockIdxPhase124(PhaseArg0 &matrixIndex, PhaseArg1 &originalDim,
                                                  PhaseArg2 &physicalGroup) {
        return LdetSharedOperations::RunGetBlockIdxPhase107SharedPhase26<
            CyclicN200HeadPairs, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::INPLACE_PANEL, ClassPlan46::LOCAL_TAIL,
            ClassPlan46::PAD_INPUT, ClassPlan46::PANEL_PITCH, WAVED, PhaseArg0, PhaseArg1, PhaseArg2>(
            *this, matrixIndex, originalDim, physicalGroup);
    }

    __aicore__ inline void RunSetGlobalBufferPhase125(const uint32_t &addressMatrix, const bool &valid,
                                                      GM_ADDR &workspace) {
        return LdetSharedOperations::RunSetGlobalBufferPhase125SharedPhase31<CyclicN200HeadPairs,
                                                                             ClassPlan46::LOCAL_CUT>(
            *this, addressMatrix, valid, workspace);
    }
    inline void Runstatic_assertPhase126() {
        return LdetSharedOperations::Runstatic_assertPhase126SharedPhase32<
            CyclicN200HeadPairs, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::LOCAL_CUT, ClassPlan46::SPLIT_OWNERS>(*this);
    }

    __aicore__ inline void RunDataCopyPadPhase127(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase111SharedPhase28<
            CyclicN200HeadPairs, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA>(*this, rounds, valid);
    }
    template <bool WAVED = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true, uint32_t originalDim = NMAX, uint32_t physicalGroup = G) {
        if (LdetSharedOperations::SkipInactivePhysicalGroup51<CyclicN200HeadPairs, false>(*this, physicalGroup))
            return;
        const bool valid = !WAVED || active;
        RunGetBlockIdxPhase124<WAVED>(matrixIndex, originalDim, physicalGroup);
        const uint32_t addressMatrix = valid ? matrix : 0U;
        RunPipeBarrierPhase123<WAVED>(addressMatrix, input, output, valid, workspace);
        uint32_t rounds = blocks;
        RunSetGlobalBufferPhase125(addressMatrix, valid, workspace);
        if constexpr (ClassPlan46::LOCAL_TAIL) {
            Runstatic_assertPhase126();
            if (valid && rank == G - 1U) {
                if (singular != 3U)
                    FinishLocalTail();
                LuPublish(prefixFlag + (G - 2U) * 64U, singular == 3U ? 0x6B1F0002U : 0x6B1F0001U);
            }
            return;
        }
        RunDataCopyPadPhase127(rounds, valid);
    }
