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

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::SharedSegmentMetadata16() {
MakeIndices(outputOffsets_,int32_t(0),8);PipeBarrier<PIPE_V>();
            Muls(so_,outputOffsets_,int32_t(24),8);
            if constexpr(!SINGLE_TILE || sizeof(T)==2)Muls(outputOffsets_,outputOffsets_,int32_t(8),8);
            PipeBarrier<PIPE_V>();
            Copy(so_[8],so_,8,7,CopyRepeatParams(1,1,1,0));
            const uint32_t bits=((1U<<t_.segments)-1U)*0x01010101U;
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),bits,2);
}

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
template<class SharedArgument0, class SharedArgument1>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::SharedSegmentPadding9(SharedArgument0& index, SharedArgument1& tmp) {
MakeIndices(index, int32_t(0), count_);
        PipeBarrier<PIPE_V>();
        ShiftRight(po_.template ReinterpretCast<uint32_t>(), index.template ReinterpretCast<uint32_t>(),
                   t_.segmentLog, count_);
        PipeBarrier<PIPE_V>();
        Muls(tmp, po_, static_cast<int32_t>(stride_), count_);
        PipeBarrier<PIPE_V>();
        Sub(so_, index, tmp, count_);
        PipeBarrier<PIPE_V>();
        Cast(W(8), so_, RoundMode::CAST_NONE, count_);
        Muls(so_, so_, int32_t(24), count_);
        Muls(po_, po_, int32_t(12), count_);
        PipeBarrier<PIPE_V>();
        if constexpr(MODE==8) {
            Copy(po_,index,64,count_/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        }
        CompareScalar(valid_, W(8), static_cast<float>(static_cast<int32_t>(t_.segments)), CMPMODE::LT, count_);
        Muls(outputOffsets_, index, int32_t(8), rows_);
}
