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
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedResourceBlock4(SharedArgument0& sparse, SharedArgument1& first, SharedArgument2& hits, SharedArgument3& second, SharedArgument4& candidates, SharedArgument5& row, SharedArgument6& searchColumns, SharedArgument7& searchCp) {
if(sparse) {
                    Adds(W(7),segmentIndices_,float(int32_t(first*64)),64);
                    if(hits==2)Adds(W(7)[64],segmentIndices_,float(int32_t(second*64)),64);
                } else {
                    MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
                    Cast(W(7),W(6).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);
                }
                PipeBarrier<PIPE_V>();
                CompareScalar(valid_,W(7),float(int32_t(t_.segments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                this->template Compensated<true>();
                WholeReduceMin(W(6),W(0),64,candidates/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                Gather(W(7),W(6),outputOffset_.template ReinterpretCast<uint32_t>(),0U,candidates/64);PipeBarrier<PIPE_V>();
                WholeReduceMin(record_,W(7),candidates/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);ProcessMulsStage(row, sparse, searchColumns, searchCp);
}
