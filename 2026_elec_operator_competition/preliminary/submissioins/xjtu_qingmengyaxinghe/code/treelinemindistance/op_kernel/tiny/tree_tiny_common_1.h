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

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2>
__aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::SharedResourceBlock6(SharedArgument0& same, SharedArgument1& bounds, SharedArgument2& unary) {
SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,16,0));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum<float,false>(lengths_,zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(bounds[64],lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
}

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2>
__aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::SharedResourceBlock8(SharedArgument0& unary, SharedArgument1& bounds, SharedArgument2& same) {
Abs<float,false>(zeroGeometry_[256],zeroGeometry_[256],uint64_t(64),1,unary);
                Maxs<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(bounds[128],zeroGeometry_[256],zeroGeometry_[384],CMPMODE::LE,uint64_t(64),1,same);
                Mins<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);
                PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
                WholeReduceSum<float,false>(scores_,W(2),3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
}
