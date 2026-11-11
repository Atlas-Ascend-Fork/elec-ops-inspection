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
template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CommonVectorStage52(SharedArgument0& index) {
if (stride_ > 64) {
            const uint32_t parts = stride_ / 64;
            const uint32_t partStride = parts < 8 ? 8 : parts;
            if(parts>=8) {
                Muls(reduceOffsets_,index,int32_t(8),rows_*partStride);
            } else {
                Duplicate(reduceOffsets_, int32_t(0), rows_ * partStride);
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < rows_; ++row)
                    MakeIndices(reduceOffsets_[row * partStride], static_cast<int32_t>(row * parts), parts);
                PipeBarrier<PIPE_V>();
                Muls(reduceOffsets_, reduceOffsets_, int32_t(8), rows_ * partStride);
            }
            Muls(reduceBases_, index, static_cast<int32_t>(parts * 8), rows_);
            PipeBarrier<PIPE_V>();
            Adds(reduceBases_, reduceBases_, int32_t(4), rows_);
        }
}
