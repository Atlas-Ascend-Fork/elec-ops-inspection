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
// Included in the FullFallbackHead102 class; preserves its member scope and access.
    template <bool WAVED>
    __aicore__ inline void RunDuplicatePhase150(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                const bool &valid, GM_ADDR &workspace) {
        return LdetSharedOperations::RunDuplicatePhase121SharedPhase29<
            FullFallbackHead102, ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPLACE_PANEL,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::LOCAL_TAIL, ClassPlan46::PANEL_PITCH, ClassPlan46::PREFIX_STREAM,
            ClassPlan46::SPLIT_OWNERS, WAVED>(*this, addressMatrix, input, output, valid, workspace);
    }
    template <bool WAVED> __aicore__ inline void RunLDET_MASK_SETPhase151(const bool &valid) {
        return LdetSharedOperations::RunLDET_MASK_SETPhase122SharedPhase30<
            FullFallbackHead102, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT, ClassPlan46::PREFIX_STREAM, ClassPlan46::SPLIT_OWNERS,
            WAVED>(*this, valid);
    }
    template <bool WAVED>
    __aicore__ inline void RunPipeBarrierPhase152(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                  const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase150<WAVED>(addressMatrix, input, output, valid, workspace);
        LdetSharedOperations::SharedOwnerPipeBarrierStage35<FullFallbackHead102, ClassPlan46::INDEX_SEEDS>(*this);
        RunLDET_MASK_SETPhase151<WAVED>(valid);
        LdetSharedOperations::SharedOwnerScalarUpdateStage36<
            FullFallbackHead102, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::COMPACT, ClassPlan46::CONTIGUOUS_BANDS,
            ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT,
            ClassPlan46::SPLIT_OWNERS>(*this);
    }
    template <bool WAVED, class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ inline void RunFullFallbackVirtual102Phase153(PhaseArg0 &matrixIndex, PhaseArg1 &originalDim,
                                                             PhaseArg2 &physicalGroup) {

        inputDim = ClassPlan46::PAD_INPUT ? originalDim : NMAX;
        n = NMAX;
        rank = (FullFallbackVirtual102() % physicalGroup) / 2U + ((FullFallbackVirtual102() % physicalGroup) & 1U) * 3U;
        matrix = WAVED ? matrixIndex : FullFallbackVirtual102() / physicalGroup;
        blocks = (n + B - 1U) / B;
        LdetSharedOperations::SharedOwnerScalarUpdateStage3<FullFallbackHead102, ClassPlan46::CONTIGUOUS_BANDS,
                                                            ClassPlan46::INPLACE_PANEL, ClassPlan46::LOCAL_TAIL,
                                                            ClassPlan46::PANEL_PITCH>(*this);
        mu = meta.template ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void RunSetGlobalBufferPhase154(const uint32_t &addressMatrix, const bool &valid,
                                                      GM_ADDR &workspace) {
        return LdetSharedOperations::RunSetGlobalBufferPhase125SharedPhase31<FullFallbackHead102,
                                                                             ClassPlan46::LOCAL_CUT>(
            *this, addressMatrix, valid, workspace);
    }
    inline void Runstatic_assertPhase155() {
        return LdetSharedOperations::Runstatic_assertPhase126SharedPhase32<
            FullFallbackHead102, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::LOCAL_CUT, ClassPlan46::SPLIT_OWNERS>(*this);
    }

    __aicore__ inline void RunDataCopyPadPhase156(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase111SharedPhase28<
            FullFallbackHead102, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA>(*this, rounds, valid);
    }
    template <bool WAVED = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true, uint32_t originalDim = NMAX, uint32_t physicalGroup = G) {
        if (LdetSharedOperations::SkipInactivePhysicalGroup51<FullFallbackHead102, true>(*this, physicalGroup))
            return;
        const bool valid = !WAVED || active;
        RunFullFallbackVirtual102Phase153<WAVED>(matrixIndex, originalDim, physicalGroup);
        const uint32_t addressMatrix = valid ? matrix : 0U;
        RunPipeBarrierPhase152<WAVED>(addressMatrix, input, output, valid, workspace);
        uint32_t rounds = blocks;
        RunSetGlobalBufferPhase154(addressMatrix, valid, workspace);
        if constexpr (ClassPlan46::LOCAL_TAIL) {
            Runstatic_assertPhase155();
            if (valid && rank == G - 1U) {
                if (singular != 3U)
                    FinishLocalTail();
                LuPublish(prefixFlag + (G - 2U) * 64U, singular == 3U ? 0x6B1F0002U : 0x6B1F0001U);
            }
            return;
        }
        RunDataCopyPadPhase156(rounds, valid);
    }
