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
// Included in the CyclicRowLUN200C22 class; preserves its member scope and access.
    template <bool WAVED>
    __aicore__ inline void RunDuplicatePhase103(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                const bool &valid, GM_ADDR &workspace) {
        return LdetSharedOperations::RunDuplicatePhase103SharedPhase23<
            CyclicRowLUN200C22, ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPLACE_PANEL,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::LOCAL_TAIL, ClassPlan46::PANEL_PITCH, ClassPlan46::PREFIX_STREAM,
            ClassPlan46::SPLIT_OWNERS, WAVED>(*this, addressMatrix, input, output, valid, workspace);
    }
    template <bool WAVED> __aicore__ inline void RunLDET_MASK_SETPhase104(const bool &valid) {
        return LdetSharedOperations::RunLDET_MASK_SETPhase104SharedPhase24<
            CyclicRowLUN200C22, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT, ClassPlan46::PREFIX_STREAM, ClassPlan46::SPLIT_OWNERS,
            WAVED>(*this, valid);
    }
    template <bool WAVED>
    __aicore__ inline void RunPipeBarrierPhase105(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                  const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase103<WAVED>(addressMatrix, input, output, valid, workspace);
        LdetSharedOperations::SharedOwnerPipeBarrierStage13<CyclicRowLUN200C22, ClassPlan46::INDEX_SEEDS>(*this);
        RunLDET_MASK_SETPhase104<WAVED>(valid);
        LdetSharedOperations::SharedOwnerScalarUpdateStage27<
            CyclicRowLUN200C22, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::COMPACT, ClassPlan46::CONTIGUOUS_BANDS,
            ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT,
            ClassPlan46::SPLIT_OWNERS>(*this);
    }

    __aicore__ inline void RunPipeBarrierPhase106(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {
        return LdetSharedOperations::RunPipeBarrierPhase106SharedPhase25<
            CyclicRowLUN200C22, ClassPlan46::EXTRA, ClassPlan46::PAIR_LOCAL, ClassPlan46::PREFIX_STREAM,
            ClassPlan46::RETIRE_RECEIVE, ClassPlan46::SPLIT_OWNERS>(*this, kb, owner, q, valid);
    }
    template <bool WAVED, class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ inline void RunGetBlockIdxPhase107(PhaseArg0 &matrixIndex, PhaseArg1 &originalDim,
                                                  PhaseArg2 &physicalGroup) {
        return LdetSharedOperations::RunGetBlockIdxPhase107SharedPhase26<
            CyclicRowLUN200C22, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::INPLACE_PANEL, ClassPlan46::LOCAL_TAIL,
            ClassPlan46::PAD_INPUT, ClassPlan46::PANEL_PITCH, WAVED, PhaseArg0, PhaseArg1, PhaseArg2>(
            *this, matrixIndex, originalDim, physicalGroup);
    }
    __aicore__ inline void Runstatic_assertPhase108(const bool &valid) {
        return LdetSharedOperations::Runstatic_assertPhase108SharedPhase27<
            CyclicRowLUN200C22, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::LOCAL_CUT, ClassPlan46::SPLIT_OWNERS>(*this, valid);
    }

    __aicore__ inline void RunconstexprPhase109(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        LdetSharedOperations::SharedOwner26ScalarUpdateStage3<CyclicRowLUN200C22, ClassPlan46::PREFIX_STREAM,
                                                              ClassPlan46::SPLIT_OWNERS>(*this, kb, q, valid);
        RunPipeBarrierPhase106(kb, owner, q, valid);
    }

    __aicore__ inline void RunSetGlobalBufferPhase110(const uint32_t &addressMatrix, uint32_t &kb, uint32_t &owner,
                                                      uint32_t &q, const bool &valid, GM_ADDR &workspace) {

        LdetSharedOperations::SharedOwnerScalarUpdateStage28<CyclicRowLUN200C22, ClassPlan46::PREFIX_STREAM,
                                                             ClassPlan46::SPLIT_OWNERS>(*this, addressMatrix, kb, owner,
                                                                                        q, valid, workspace);
        RunconstexprPhase109(kb, owner, q, valid);
        if (valid && !singular) {
            LdetSharedOperations::SharedOwner26ScalarUpdateStage5<CyclicRowLUN200C22, ClassPlan46::PREFIX_STREAM,
                                                                  ClassPlan46::SPLIT_OWNERS>(*this, kb, q);
        }
    }

    __aicore__ inline void RunDataCopyPadPhase111(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase111SharedPhase28<
            CyclicRowLUN200C22, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA>(*this, rounds, valid);
    }
    template <bool WAVED = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true, uint32_t originalDim = NMAX, uint32_t physicalGroup = G) {
        if (LdetSharedOperations::SkipInactivePhysicalGroup51<CyclicRowLUN200C22, false>(*this, physicalGroup))
            return;
        const bool valid = !WAVED || active;
        RunGetBlockIdxPhase107<WAVED>(matrixIndex, originalDim, physicalGroup);
        const uint32_t addressMatrix = valid ? matrix : 0U;
        RunPipeBarrierPhase105<WAVED>(addressMatrix, input, output, valid, workspace);
        uint32_t rounds = blocks;
        for (uint32_t q = 0; q < (ClassPlan46::LOCAL_TAIL ? ClassPlan46::LOCAL_CUT / B : rounds - 1U); ++q) {
            uint32_t kb = q * B, owner = ClassPlan46::CONTIGUOUS_BANDS ? ownerOf(q) : q % G;
            RunSetGlobalBufferPhase110(addressMatrix, kb, owner, q, valid, workspace);
        }
        if constexpr (ClassPlan46::LOCAL_TAIL) {
            return Runstatic_assertPhase108(valid);
        }
        RunDataCopyPadPhase111(rounds, valid);
    }
