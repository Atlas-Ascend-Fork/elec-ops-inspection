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

// Select the compensated residual independently for each coordinate plane.
__aicore__ __attribute__((always_inline)) inline void SelectResidualPlanes(
    LocalTensor<float> x0, LocalTensor<float> x1, LocalTensor<float> x2,
    LocalTensor<float> y0, LocalTensor<float> y1, LocalTensor<float> y2,
    LocalTensor<uint8_t> mask, bool singleRegister, uint8_t repeats,
    const BinaryRepeatParams &binary) {
    if (singleRegister) {
        SetCmpMask(mask);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(x0,x0,y0,1,binary);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(x1,x1,y1,1,binary);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(x2,x2,y2,1,binary);
    } else {
        Select<float,uint8_t,false>(x0,mask,x0,y0,SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
        Select<float,uint8_t,false>(x1,mask,x1,y1,SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
        Select<float,uint8_t,false>(x2,mask,x2,y2,SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
    }
}

// Compare the two endpoint triples lexicographically before canonicalization.
__aicore__ __attribute__((always_inline)) inline void CompareEndpointOrder(
    LocalTensor<uint8_t> keep, LocalTensor<uint8_t> equal, LocalTensor<uint8_t> less,
    LocalTensor<uint8_t> equalY, LocalTensor<float> c0, LocalTensor<float> c1,
    LocalTensor<float> c2, LocalTensor<float> c3, LocalTensor<float> c4,
    LocalTensor<float> c5, uint32_t count) {
    Compare(keep,c0,c3,CMPMODE::LT,count);
    Compare(equal,c0,c3,CMPMODE::EQ,count);
    Compare(less,c1,c4,CMPMODE::LT,count);
    Compare(equalY,c1,c4,CMPMODE::EQ,count);
    PipeBarrier<PIPE_V>();
    And(less.ReinterpretCast<uint16_t>(),less.ReinterpretCast<uint16_t>(),equal.ReinterpretCast<uint16_t>(),count/16);
    And(equal.ReinterpretCast<uint16_t>(),equal.ReinterpretCast<uint16_t>(),equalY.ReinterpretCast<uint16_t>(),count/16);
    PipeBarrier<PIPE_V>();
    Or(keep.ReinterpretCast<uint16_t>(),keep.ReinterpretCast<uint16_t>(),less.ReinterpretCast<uint16_t>(),count/16);
    Compare(less,c2,c5,CMPMODE::LE,count);
    PipeBarrier<PIPE_V>();
    And(less.ReinterpretCast<uint16_t>(),less.ReinterpretCast<uint16_t>(),equal.ReinterpretCast<uint16_t>(),count/16);
    PipeBarrier<PIPE_V>();
    Or(keep.ReinterpretCast<uint16_t>(),keep.ReinterpretCast<uint16_t>(),less.ReinterpretCast<uint16_t>(),count/16);
    PipeBarrier<PIPE_V>();
}

__aicore__ __attribute__((always_inline)) inline bool HasIdenticalDyadicPairs(LocalTensor<uint64_t> words, uint32_t livePairs) {
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint64_t forward=words.GetValue(0)&words.GetValue(4),reverse=words.GetValue(8)&words.GetValue(12);
    const uint64_t identical=((forward&(forward>>1)&(forward>>2))|(reverse&(reverse>>1)&(reverse>>2)))&0x0101010101010101ULL;
    const uint32_t identicalPairs=uint32_t((identical*0x0102040810204080ULL)>>56);
    return (identicalPairs&livePairs)==livePairs;
}

__aicore__ __attribute__((always_inline)) inline bool IsCertifiedResidualRange(LocalTensor<float> lengths, LocalTensor<float> scores) {
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t norm=lengths.ReinterpretCast<uint32_t>().GetValue(0);
    const uint32_t score=scores.ReinterpretCast<uint32_t>().GetValue(0);
    return norm>=0x39800000U && norm<=0x44800000U && score>=0x3f800000U && score<=0x44800000U;
}

__aicore__ __attribute__((always_inline)) inline bool HasAmbiguousMinimum(LocalTensor<float> record, LocalTensor<uint8_t> smallMask) {
    Compare(smallMask,record[64],record,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return smallMask.GetValue(0)&1U;
}

__aicore__ __attribute__((always_inline)) inline void ProjectSingleResidual(LocalTensor<float> broadcast, LocalTensor<float> projection,
    LocalTensor<float> residual, LocalTensor<float> direction, LocalTensor<float> scores,
    const UnaryRepeatParams &unary, const BinaryRepeatParams &same) {
    Brcb(broadcast,projection,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
    Muls<float,false>(broadcast,broadcast,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
    MulAddDst<float,float,false>(residual,broadcast,direction,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
    Mul<float,false>(residual,residual,residual,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
    WholeReduceSum(scores,residual,3,1,1,1,1);PipeBarrier<PIPE_V>();
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}

__aicore__ __attribute__((always_inline)) inline void CompensateDirectionPlanes(LocalTensor<float> w0, LocalTensor<float> w1, LocalTensor<float> w2, LocalTensor<float> w3, LocalTensor<float> w4, LocalTensor<float> w5, LocalTensor<float> c0, LocalTensor<float> c1, LocalTensor<float> c2, LocalTensor<float> c3, LocalTensor<float> c4, LocalTensor<float> c5, LocalTensor<float> c6, LocalTensor<float> c7, LocalTensor<float> c8, LocalTensor<float> c10, LocalTensor<float> c11, LocalTensor<float> c12, uint32_t count) {
    Sub(w0,c3,c6,count);
    Sub(w1,c4,c7,count);
    Sub(w2,c5,c8,count);
    PipeBarrier<PIPE_V>();
    Add(w3,c6,w0,count);Sub(c10,w0,c0,count);
    Add(w4,c7,w1,count);Sub(c11,w1,c1,count);
    Add(w5,c8,w2,count);Sub(c12,w2,c2,count);
    PipeBarrier<PIPE_V>();
    Sub(w3,c3,w3,count);
    Sub(w4,c4,w4,count);
    Sub(w5,c5,w5,count);
    PipeBarrier<PIPE_V>();
    Add(c10,c10,w3,count);
    Add(c11,c11,w4,count);
    Add(c12,c12,w5,count);
}

template<bool HALF_RESOURCE>
struct TreePrefixSearchState {
    LocalTensor<float> certHigh_;
    LocalTensor<float> cert_;
    LocalTensor<float> chunkLower_;
    LocalTensor<float> hi_;
    LocalTensor<int32_t> idx_;
    LocalTensor<float> lo_;
    LocalTensor<uint8_t> mask_;
    LocalTensor<int32_t> offset_;
    LocalTensor<int32_t> oneHot_;
    LocalTensor<float> record_;
    LocalTensor<float> refined_;
    LocalTensor<int32_t> rowBase_;
    LocalTensor<float> second_;
    LocalTensor<float> segmentIndices_;
    LocalTensor<float> tileHi_;
    LocalTensor<int32_t> tileIdx_;
    LocalTensor<float> tileLo_;
    LocalTensor<uint8_t> valid_;
    LocalTensor<float> xyz_;
    const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
    const UnaryRepeatParams unary{1,1,8,8};
    __aicore__ __attribute__((always_inline)) inline TreePrefixSearchState() :
        certHigh_(TPosition::VECCALC,184256U+(HALF_RESOURCE?384U:0U),64U),
        cert_(TPosition::VECCALC,184000U+(HALF_RESOURCE?384U:0U),64U),
        chunkLower_(TPosition::VECCALC,174208U+(HALF_RESOURCE?384U:0U),1024U),
        hi_(TPosition::VECCALC,183488U+(HALF_RESOURCE?384U:0U),64U),
        idx_(TPosition::VECCALC,183744U+(HALF_RESOURCE?384U:0U),64U),
        lo_(TPosition::VECCALC,182976U+(HALF_RESOURCE?384U:0U),64U),
        mask_(TPosition::VECCALC,187072U+(HALF_RESOURCE?256U:0U),512U),
        offset_(TPosition::VECCALC,185792U+(HALF_RESOURCE?384U:0U),64U),
        oneHot_(TPosition::VECCALC,184512U+(HALF_RESOURCE?384U:0U),64U),
        record_(TPosition::VECCALC,182400U+(HALF_RESOURCE?384U:0U),128U),
        refined_(TPosition::VECCALC,179072U+(HALF_RESOURCE?384U:0U),832U),
        rowBase_(TPosition::VECCALC,186048U+(HALF_RESOURCE?384U:0U),64U),
        second_(TPosition::VECCALC,183232U+(HALF_RESOURCE?384U:0U),64U),
        segmentIndices_(TPosition::VECCALC,173184U,64U),
        tileHi_(TPosition::VECCALC,185280U+(HALF_RESOURCE?384U:0U),64U),
        tileIdx_(TPosition::VECCALC,185536U+(HALF_RESOURCE?384U:0U),64U),
        tileLo_(TPosition::VECCALC,184768U+(HALF_RESOURCE?384U:0U),64U),
        valid_(TPosition::VECCALC,187584U+(HALF_RESOURCE?256U:0U),512U),
        xyz_(TPosition::VECCALC,59136U,3168U) {
    }
};

template<class T,uint32_t ROWS,uint32_t CAPACITY>
struct TreeFewRefinementState {
    static constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
    static constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
    static constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
    LocalTensor<float> record;
    LocalTensor<uint8_t> mask,valid,smallMask;
    __aicore__ __attribute__((always_inline)) inline TreeFewRefinementState() :
        record(TPosition::VECCALC,stateBase*4,128),
        mask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4)*4,ROWS*8),
        valid(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8),
        smallMask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32) {
    }
};

struct TreeDyadicScratch {
    LocalTensor<float> left;
    LocalTensor<float> right;
    LocalTensor<float> delta;
    LocalTensor<float> relA;
    LocalTensor<float> relC;
    LocalTensor<float> temp;
    LocalTensor<float> rounded;
    LocalTensor<float> product;
    LocalTensor<float> norm;
    LocalTensor<float> dot;
    LocalTensor<float> endpoint;
    LocalTensor<float> box;
    LocalTensor<float> upper;
    LocalTensor<uint8_t> flags;
    __aicore__ __attribute__((always_inline)) inline TreeDyadicScratch(LocalTensor<float> base) :
        left(base),
        right(base[64]),
        delta(base[128]),
        relA(base[192]),
        relC(base[256]),
        temp(base[320]),
        rounded(base[384]),
        product(base[448]),
        norm(base[512]),
        dot(base[576]),
        endpoint(base[640]),
        box(base[704]),
        upper(base[768]),
        flags(base[832].ReinterpretCast<uint8_t>()) {
    }
};

struct TreeExactCorrectionViews {
    LocalTensor<float> scores;
    LocalTensor<float> points;
    LocalTensor<float> endpoints;
    LocalTensor<float> storage;
    LocalTensor<float> limit;
    LocalTensor<uint8_t> predicates;
    __aicore__ __attribute__((always_inline)) inline TreeExactCorrectionViews(uint32_t scoreAddress, uint32_t pointAddress, uint32_t pointPitch, uint32_t endpointAddress, uint32_t endpointPitch, uint32_t storageAddress, uint32_t limitAddress, uint32_t predicateAddress, uint32_t count) :
        scores(TPosition::VECCALC,scoreAddress,count),
        points(TPosition::VECCALC,pointAddress,3*pointPitch),
        endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch),
        storage(TPosition::VECCALC,storageAddress,45*64),
        limit(TPosition::VECCALC,limitAddress,64),
        predicates(TPosition::VECCALC,predicateAddress,96) {
    }
};

template<bool FULL_FIXED_MASK,bool DEFER_SECOND>
__aicore__ __attribute__((always_inline)) inline void MergeGuardedChunkMinimum(
    LocalTensor<float> lo, LocalTensor<float> hi, LocalTensor<float> second,
    LocalTensor<float> record, LocalTensor<int32_t> tileIdx,
    LocalTensor<float> chunkMinimum, LocalTensor<float> chunkUpper, uint32_t j,
    const BinaryRepeatParams& same, const UnaryRepeatParams& unary) {
    if constexpr(!FULL_FIXED_MASK) {
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    }
    Compare<float,false>(chunkMinimum,lo,CMPMODE::LT,uint64_t(64),same);
    if constexpr(!DEFER_SECOND)Max<float,false>(record,chunkMinimum,lo,uint64_t(64),1,same);
    Adds<int32_t,false>(tileIdx,tileIdx,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
    if constexpr(!DEFER_SECOND)Min<float,false>(second,record,second,uint64_t(64),1,same);
    Min<float,false>(lo,lo,chunkMinimum,uint64_t(64),1,same);
    Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi,chunkUpper,hi,1,same);
}

__aicore__ __attribute__((always_inline)) inline void FilterZeroMagnitudeExponents(LocalTensor<float> scratch,LocalTensor<float> temporary,LocalTensor<float> sf,LocalTensor<float> exponent,LocalTensor<uint8_t> mask,uint32_t coords) {
Duplicate(scratch.ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),64);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<coords;k+=64) {
        And(temporary.ReinterpretCast<uint16_t>(),sf[k].ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        CompareScalar(mask,temporary.ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Not(mask.ReinterpretCast<uint16_t>(),mask.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Select(exponent[k],mask,exponent[k],255.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
    }
}

struct TreeInteriorTieViews {
    LocalTensor<float> scores,endpoints,storage,winnerEndpoints,tmp,record;
    LocalTensor<uint8_t> predicates,certified,test,other;
    __aicore__ __attribute__((always_inline)) inline TreeInteriorTieViews(
        uint32_t scoreAddress,uint32_t endpointAddress,uint32_t endpointPitch,
        uint32_t storageAddress,uint32_t predicateAddress) :
        scores(TPosition::VECCALC,scoreAddress,8),
        endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch),
        storage(TPosition::VECCALC,storageAddress,65*64),
        winnerEndpoints(storage[45*64]),tmp(storage[51*64]),record(storage[64*64]),
        predicates(TPosition::VECCALC,predicateAddress,96),
        certified(predicates),test(predicates[32]),other(predicates[64]) {
    }
};

__aicore__ __attribute__((always_inline)) inline void RefinePotentialCollinearMask(LocalTensor<float> tmp,LocalTensor<float> winnerEndpoints,
    LocalTensor<float> endpoints,uint32_t endpointPitch,LocalTensor<uint8_t> certified,LocalTensor<uint8_t> test) {
Sub(tmp,winnerEndpoints[3*64],winnerEndpoints,8);
    Sub(tmp[64],winnerEndpoints[4*64],winnerEndpoints[64],8);
    Sub(tmp[2*64],endpoints,winnerEndpoints,8);
    Sub(tmp[3*64],endpoints[endpointPitch],winnerEndpoints[64],8);PipeBarrier<PIPE_V>();
    Mul(tmp[4*64],tmp[2*64],tmp[64],8);
    Mul(tmp[5*64],tmp[3*64],tmp,8);PipeBarrier<PIPE_V>();
    Sub(tmp[6*64],tmp[4*64],tmp[5*64],8);
    Abs(tmp[4*64],tmp[4*64],8);Abs(tmp[5*64],tmp[5*64],8);PipeBarrier<PIPE_V>();
    Abs(tmp[6*64],tmp[6*64],8);
    Add(tmp[7*64],tmp[4*64],tmp[5*64],8);PipeBarrier<PIPE_V>();
    Muls(tmp[7*64],tmp[7*64],3.0517578125e-5F,8);PipeBarrier<PIPE_V>();
    Compare<float,uint8_t,true>(test,tmp[6*64],tmp[7*64],CMPMODE::LE,uint64_t(8),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
    And(certified.ReinterpretCast<uint16_t>(),certified.ReinterpretCast<uint16_t>(),test.ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
}
