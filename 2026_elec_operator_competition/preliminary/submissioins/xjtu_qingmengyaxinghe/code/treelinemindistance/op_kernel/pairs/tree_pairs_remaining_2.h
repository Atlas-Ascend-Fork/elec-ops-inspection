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
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CommonVectorStage56() {
MakeIndices(po_,int32_t(0),count_);PipeBarrier<PIPE_V>();
            Muls(so_,po_,int32_t(24),count_);
            Cast(W(8),po_,RoundMode::CAST_NONE,count_);
            Duplicate(outputOffsets_,int32_t(0),8);
            Muls(reduceOffsets_,po_,int32_t(8),32);
            Duplicate(reduceBases_,int32_t(4),8);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(8),float(int32_t(t_.segments)),CMPMODE::LT,count_);
}
