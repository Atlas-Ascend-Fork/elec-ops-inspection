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
// Included in the KernelLinalgDet class; preserves its member scope and access.
    __aicore__ inline void ProcessN15StaticProductEntry() {
        if (cnt_ == 0U)
            return;
        ProcessFamilyColumns<15U, 16U, true, true>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, 4U, 0, 0, 0});
    }
    __aicore__ inline void ProcessUnifiedSmallEntry() {
        if (cnt_ == 0U)
            return;
        if (n_ == 32U && batch_ == 1U)
            ProcessSmallFixed<32U, true, true, true, true, true>();
        else if (n_ == 9U)
            ProcessFamilyColumns<9U, 16U>();
        else if (n_ == 18U)
            ProcessFamilyColumns<18U, 24U>();
        else if (n_ == 19U)
            ProcessFamilyColumns<19U, 24U>();
        else if (n_ == 30U)
            ProcessSmallRawUB<false, true, true, true, true, true>();
        else if (n_ == 20U && batch_ == 1U)
            ProcessFamilyColumns<>();
        else if (n_ >= 20U && n_ <= 31U && n_ != 24U)
            ProcessSmallRawUB<false, true, true, true>();
        else if (n_ == 16U)
            ProcessFamilyColumns<16U, 24U>();
        else if (n_ == 24U)
            ProcessFamilyColumns<24U, 24U>();
        else if (n_ == 32U)
            ProcessFamilyColumns<32U, 40U>();
        else
            ProcessFamilyColumns<>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, 4U, 0, 0, 0});
    }
    __aicore__ inline void ProcessN16BatchTail4Entry() {
        if (cnt_ == 0U)
            return;
        ProcessFamilyColumns<16U, 24U, false, false, true>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, 4U, 0, 0, 0});
    }
    template <uint32_t SPR_MODE = 0U> __aicore__ inline void ProcessRaggedMidRecurrenceEntry() {
        if (cnt_ == 0U)
            return;
        ProcessSmallRawUB<false, true, true, true, false, false, SPR_MODE>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, 4U, 0, 0, 0});
    }
    __aicore__ inline void ProcessN18B8FixedEntry() {
        if (cnt_ == 0U)
            return;
        ProcessFamilyColumns<18U, 24U>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, 4U, 0, 0, 0});
    }
    __aicore__ inline void ProcessN19B8FixedEntry() {
        if (cnt_ == 0U)
            return;
        ProcessFamilyColumns<19U, 24U>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(detGm_[start_], res_, {1, 4U, 0, 0, 0});
    }
