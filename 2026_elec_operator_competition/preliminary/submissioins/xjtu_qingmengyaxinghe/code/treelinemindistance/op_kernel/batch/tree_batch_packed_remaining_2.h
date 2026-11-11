// Copyright 2026 Qianqiuer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CommonVectorStage55(SharedArgument0& row) {
Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
                Adds(offset_,offset_,int32_t(row*4),16);
                Duplicate(W(0),__builtin_inff(),64);
                Brcb(W(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                Copy(W(2),W(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
                Gather(W(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(t_.segments+63)/64);PipeBarrier<PIPE_V>();
                Compare(smallMask_,W(0),W(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
}
