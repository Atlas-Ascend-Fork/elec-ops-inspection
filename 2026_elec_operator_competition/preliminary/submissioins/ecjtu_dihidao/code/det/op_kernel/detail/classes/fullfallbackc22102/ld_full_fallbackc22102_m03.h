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
// Included in the FullFallbackC22102 class; preserves its member scope and access.
    template <bool WAVED>
    __aicore__ inline void RunDuplicatePhase141(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                const bool &valid, GM_ADDR &workspace) {
        return LdetSharedOperations::RunDuplicatePhase103SharedPhase23<
            FullFallbackC22102, ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPLACE_PANEL,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::LOCAL_TAIL, ClassPlan46::PANEL_PITCH, ClassPlan46::PREFIX_STREAM,
            ClassPlan46::SPLIT_OWNERS, WAVED>(*this, addressMatrix, input, output, valid, workspace);
    }
    template <bool WAVED> __aicore__ inline void RunLDET_MASK_SETPhase142(const bool &valid) {
        return LdetSharedOperations::RunLDET_MASK_SETPhase104SharedPhase24<
            FullFallbackC22102, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT, ClassPlan46::PREFIX_STREAM, ClassPlan46::SPLIT_OWNERS,
            WAVED>(*this, valid);
    }
    template <bool WAVED>
    __aicore__ inline void RunPipeBarrierPhase143(const uint32_t &addressMatrix, GM_ADDR &input, GM_ADDR &output,
                                                  const bool &valid, GM_ADDR &workspace) {

        RunDuplicatePhase141<WAVED>(addressMatrix, input, output, valid, workspace);
        if constexpr (ClassPlan46::INDEX_SEEDS) {
            LdetSharedOperations::SharedOwner26PipeBarrierStage2<FullFallbackC22102>(*this);
            Adds(tab[128U], tab, int32_t(128U * P * 4U), NMAX - 128U);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t i = 0; i < n; ++i)
                tab.SetValue(i, int32_t(i * P * 4));
            SV();
        }
        RunLDET_MASK_SETPhase142<WAVED>(valid);
        LdetSharedOperations::SharedOwnerScalarUpdateStage27<
            FullFallbackC22102, ClassPlan46::ASYNC_FRESHNESS, ClassPlan46::COMPACT, ClassPlan46::CONTIGUOUS_BANDS,
            ClassPlan46::DIM, ClassPlan46::EXTRA, ClassPlan46::INPUT_OVERLAP, ClassPlan46::PAD_INPUT,
            ClassPlan46::SPLIT_OWNERS>(*this);
    }

    __aicore__ inline void RunPipeBarrierPhase144(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {
        return LdetSharedOperations::RunPipeBarrierPhase106SharedPhase25<
            FullFallbackC22102, ClassPlan46::EXTRA, ClassPlan46::PAIR_LOCAL, ClassPlan46::PREFIX_STREAM,
            ClassPlan46::RETIRE_RECEIVE, ClassPlan46::SPLIT_OWNERS>(*this, kb, owner, q, valid);
    }
    template <bool WAVED, class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ inline void RunFullFallbackVirtual102Phase145(PhaseArg0 &matrixIndex, PhaseArg1 &originalDim,
                                                             PhaseArg2 &physicalGroup) {

        inputDim = ClassPlan46::PAD_INPUT ? originalDim : NMAX;
        n = NMAX;
        rank = (FullFallbackVirtual102() % physicalGroup) / 2U + ((FullFallbackVirtual102() % physicalGroup) & 1U) * 3U;
        matrix = WAVED ? matrixIndex : FullFallbackVirtual102() / physicalGroup;
        blocks = (n + B - 1U) / B;
        LdetSharedOperations::SharedOwnerScalarUpdateStage3<FullFallbackC22102, ClassPlan46::CONTIGUOUS_BANDS,
                                                            ClassPlan46::INPLACE_PANEL, ClassPlan46::LOCAL_TAIL,
                                                            ClassPlan46::PANEL_PITCH>(*this);
        mu = meta.template ReinterpretCast<uint32_t>();
    }
    __aicore__ inline void Runstatic_assertPhase146(const bool &valid) {
        return LdetSharedOperations::Runstatic_assertPhase108SharedPhase27<
            FullFallbackC22102, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::DIM, ClassPlan46::GROUP,
            ClassPlan46::LOCAL_CUT, ClassPlan46::SPLIT_OWNERS>(*this, valid);
    }

    __aicore__ inline void RunconstexprPhase147(uint32_t &kb, uint32_t &owner, uint32_t &q, const bool &valid) {

        if constexpr (ClassPlan46::PREFIX_STREAM) {
            if (valid &&
                (ClassPlan46::SPLIT_OWNERS ? rank == q + 1U : ((rank == 1U && q == 1U) || (rank == 2U && q == 3U)))) {
                LuPoll(prefixFlag + (ClassPlan46::SPLIT_OWNERS ? q * 64U : (q == 3U ? 64U : 0U)), 0x6A1F0010U);
                receiveRange<0U, 16U>(kb);
                if (!singular)
                    ApplyPanelUpdates<0U, 16U>(kb, q);
            }
        }
        RunPipeBarrierPhase144(kb, owner, q, valid);
    }

    __aicore__ inline void RunSetGlobalBufferPhase148(const uint32_t &addressMatrix, uint32_t &kb, uint32_t &owner,
                                                      uint32_t &q, const bool &valid, GM_ADDR &workspace) {

        const uint32_t slotWords = (addressMatrix * 2U + (q & 1U)) * GM_WORDS;
        ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + slotWords);
        wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) + slotWords);
        if (valid && rank == owner) {
            if (!singular)
                factor(kb, q);
            else if constexpr (ClassPlan46::PREFIX_STREAM) {
                if (ClassPlan46::SPLIT_OWNERS || q == 1U || q == 3U)
                    publishPrefix16(kb);
            }
            publish(kb);
        }
        RunconstexprPhase147(kb, owner, q, valid);
        if (valid && !singular) {
            if constexpr (ClassPlan46::PREFIX_STREAM) {
                if (ClassPlan46::SPLIT_OWNERS ? rank == q + 1U : ((rank == 1U && q == 1U) || (rank == 2U && q == 3U)))
                    ApplyPanelUpdates<16U, 32U>(kb, q);
                else
                    ApplyPanelUpdates(kb, q);
            } else
                ApplyPanelUpdates(kb, q);
        }
    }

    __aicore__ inline void RunDataCopyPadPhase149(uint32_t &rounds, const bool &valid) {
        return LdetSharedOperations::RunDataCopyPadPhase111SharedPhase28<
            FullFallbackC22102, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::EXTRA>(*this, rounds, valid);
    }
    template <bool WAVED = false>
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, uint32_t matrixIndex = 0U,
                               bool active = true, uint32_t originalDim = NMAX, uint32_t physicalGroup = G) {
        if (LdetSharedOperations::SkipInactivePhysicalGroup51<FullFallbackC22102, true>(*this, physicalGroup))
            return;
        const bool valid = !WAVED || active;
        RunFullFallbackVirtual102Phase145<WAVED>(matrixIndex, originalDim, physicalGroup);
        const uint32_t addressMatrix = valid ? matrix : 0U;
        RunPipeBarrierPhase143<WAVED>(addressMatrix, input, output, valid, workspace);
        uint32_t rounds = blocks;
        for (uint32_t q = 0; q < (ClassPlan46::LOCAL_TAIL ? ClassPlan46::LOCAL_CUT / B : rounds - 1U); ++q) {
            uint32_t kb = q * B, owner = ClassPlan46::CONTIGUOUS_BANDS ? ownerOf(q) : q % G;
            RunSetGlobalBufferPhase148(addressMatrix, kb, owner, q, valid, workspace);
        }
        if constexpr (ClassPlan46::LOCAL_TAIL) {
            return Runstatic_assertPhase146(valid);
        }
        RunDataCopyPadPhase149(rounds, valid);
    }
