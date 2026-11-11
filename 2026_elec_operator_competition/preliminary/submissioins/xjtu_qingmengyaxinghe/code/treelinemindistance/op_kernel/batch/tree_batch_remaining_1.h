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
template<class SharedArgument0, class SharedArgument1, class SharedArgument2>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CommonVectorStage51(SharedArgument0& chunks, SharedArgument1& same, SharedArgument2& unary) {
if(chunks<16) {
            Duplicate(chunkLower_[chunks*64],__builtin_inff(),(16-chunks)*64);
            PipeBarrier<PIPE_V>();
        }
        // Multi-repeat Compare packs sixteen64-bit results into128 bytes.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,
            uint64_t(64),16,BinaryRepeatParams{1,1,1,8,8,0});
        PipeBarrier<PIPE_V>();
        auto sourceBits=mask_.template ReinterpretCast<uint16_t>();
        auto unions=W(0).template ReinterpretCast<uint16_t>();
        auto collisions=W(1).template ReinterpretCast<uint16_t>();
        auto nextUnions=W(2).template ReinterpretCast<uint16_t>();
        auto nextCollisions=W(3).template ReinterpretCast<uint16_t>();
        auto mergedCollisions=W(4).template ReinterpretCast<uint16_t>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(32));
        Or<uint16_t,false>(unions,sourceBits,sourceBits[32],uint64_t(128),1,same);
        SharedChunkUnion3(collisions, sourceBits, same, nextUnions, unions, nextCollisions, mergedCollisions, unary);
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CommonVectorStage53(SharedArgument0& j) {
if constexpr(DEFER_CHUNK_CERTIFICATE) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if constexpr(FULL_FIXED_MASK)
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
            } else {
                GatherMask(W(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                GatherMask(W(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
            }
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CommonVectorStage54(SharedArgument0& first, SharedArgument1& second, SharedArgument2& hits, SharedArgument3& sparse, SharedArgument4& candidates, SharedArgument5& row, SharedArgument6& logicalSegments) {
RefineFullFloatFallbackBodyCopyStage(first, second, hits, sparse);
                    count_=candidates;wp_=sparse?160:m_+32;
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
                        Brcb(W(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                        Copy(P(axis),W(0)[(row%8)*8],64,candidates/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
                    }
                    SharedSparseWinner2(sparse, first, hits, second, logicalSegments, candidates);
                    Gather(offset_,W(6).template ReinterpretCast<int32_t>(),offset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    Add(tileIdx_,tileIdx_,offset_,1);PipeBarrier<PIPE_V>();
}


template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
__aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::TryCompactSearchTail(uint32_t j) {
    if constexpr(COMPACT_TAIL) {
        if(t_.segments-j<=8) {
            CompactSearchTail<false>(j,t_.segments-j);MergeSearchMinimum<false>(j);return true;
        }
    }
    return false;
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<bool FIXED>
__aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompensatedW(uint32_t axis) { if constexpr(FIXED)return FixedFewW(axis);else return PrimaryW(axis);
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<bool FIXED>
__aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompensatedP(uint32_t axis) { if constexpr(FIXED)return FixedFewP(axis);else return PrimaryP(axis);
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<bool FIXED>
__aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompensatedC(uint32_t axis) { if constexpr(FIXED)return FixedFewC(axis);else return PrimaryC(axis);
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<bool FIXED,class Repeats>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompensatedDeferred(Repeats repeats,const BinaryRepeatParams& binary) {


#pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(CompensatedW<FIXED>(axis),CompensatedP<FIXED>(axis),CompensatedC<FIXED>(axis),uint64_t(64),repeats,binary);
                Sub<float,false>(CompensatedW<FIXED>(axis+16),CompensatedC<FIXED>(axis+3),CompensatedW<FIXED>(axis+9),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
#pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(CompensatedW<FIXED>(axis+3),CompensatedP<FIXED>(axis),CompensatedW<FIXED>(axis),uint64_t(64),repeats,binary);
                Add<float,false>(CompensatedW<FIXED>(axis+13),CompensatedW<FIXED>(axis+9),CompensatedW<FIXED>(axis+16),uint64_t(64),repeats,binary);
                Sub<float,false>(CompensatedC<FIXED>(axis+10),CompensatedW<FIXED>(axis+16),CompensatedC<FIXED>(axis),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
#pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Add<float,false>(CompensatedW<FIXED>(axis+6),CompensatedW<FIXED>(axis),CompensatedW<FIXED>(axis+3),uint64_t(64),repeats,binary);
                Sub<float,false>(CompensatedW<FIXED>(axis+3),CompensatedW<FIXED>(axis+3),CompensatedC<FIXED>(axis),uint64_t(64),repeats,binary);
                Sub<float,false>(CompensatedW<FIXED>(axis+13),CompensatedC<FIXED>(axis+3),CompensatedW<FIXED>(axis+13),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
#pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Sub<float,false>(CompensatedW<FIXED>(axis+6),CompensatedP<FIXED>(axis),CompensatedW<FIXED>(axis+6),uint64_t(64),repeats,binary);
                Add<float,false>(CompensatedC<FIXED>(axis+10),CompensatedC<FIXED>(axis+10),CompensatedW<FIXED>(axis+13),uint64_t(64),repeats,binary);
            }
            PipeBarrier<PIPE_V>();
#pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Add<float,false>(CompensatedW<FIXED>(axis+3),CompensatedW<FIXED>(axis+3),CompensatedW<FIXED>(axis+6),uint64_t(64),repeats,binary);
            PipeBarrier<PIPE_V>();
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<bool FIXED,class Repeats>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompensatedImmediate(Repeats repeats,const BinaryRepeatParams& binary) {

#pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(CompensatedW<FIXED>(axis), CompensatedP<FIXED>(axis), CompensatedC<FIXED>(axis), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
#pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(CompensatedW<FIXED>(axis+3), CompensatedP<FIXED>(axis), CompensatedW<FIXED>(axis), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
#pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                Add<float, false>(CompensatedW<FIXED>(axis+6), CompensatedW<FIXED>(axis), CompensatedW<FIXED>(axis+3), uint64_t(64), repeats, binary);
                Sub<float, false>(CompensatedW<FIXED>(axis+3), CompensatedW<FIXED>(axis+3), CompensatedC<FIXED>(axis), uint64_t(64), repeats, binary);
            }
            PipeBarrier<PIPE_V>();
#pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(CompensatedW<FIXED>(axis+6), CompensatedP<FIXED>(axis), CompensatedW<FIXED>(axis+6), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
#pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(CompensatedW<FIXED>(axis+3), CompensatedW<FIXED>(axis+3), CompensatedW<FIXED>(axis+6), uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
}
