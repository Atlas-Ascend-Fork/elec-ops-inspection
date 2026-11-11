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

// A2 vector multiply-add rounds its product before accumulation. Split both
// operands into twelve-bit significands to recover the exact product error.
// Callers certify normal, finite operands and products before accepting zero.
template<uint32_t COUNT>
__aicore__ __attribute__((always_inline)) inline void ExactProductError(
    const LocalTensor<float>& error,const LocalTensor<float>& x,
    const LocalTensor<float>& y,const LocalTensor<float>& leading,
    const LocalTensor<float>& scratch) {
    auto xh=scratch,xl=scratch[64],yh=scratch[2*64],yl=scratch[3*64];
    auto lowProduct=scratch[4*64],cross=scratch[5*64];
    Duplicate(cross.template ReinterpretCast<uint32_t>(),uint32_t(0xfffff000U),COUNT);PipeBarrier<PIPE_V>();
    And(xh.template ReinterpretCast<uint16_t>(),x.template ReinterpretCast<uint16_t>(),cross.template ReinterpretCast<uint16_t>(),2*COUNT);
    And(yh.template ReinterpretCast<uint16_t>(),y.template ReinterpretCast<uint16_t>(),cross.template ReinterpretCast<uint16_t>(),2*COUNT);PipeBarrier<PIPE_V>();
    Sub(xl,x,xh,COUNT);Sub(yl,y,yh,COUNT);PipeBarrier<PIPE_V>();
    Mul(error,xh,yh,COUNT);Mul(lowProduct,xl,yl,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,leading,error,COUNT);PipeBarrier<PIPE_V>();
    Mul(cross,xl,yh,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,error,cross,COUNT);PipeBarrier<PIPE_V>();
    Mul(cross,xh,yl,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,error,cross,COUNT);PipeBarrier<PIPE_V>();
    Sub(error,lowProduct,error,COUNT);PipeBarrier<PIPE_V>();
}
#include "tree_geometry_common.h"


// Raw coordinate comparisons certify geometric zero without a tolerance.
// A rounded projection can otherwise leave a residual on an axis-aligned line.
template<bool BROADCAST_ENDPOINTS=false>
__aicore__ __attribute__((noinline)) void CorrectAxialZeroScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t scratchAddress,uint32_t scratchPitch,
    uint32_t predicateAddress,uint32_t predicatePitch,
    uint32_t count) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,count);
    const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> scratch(TPosition::VECCALC,scratchAddress,3*scratchPitch);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,3*predicatePitch);
    SetMaskNorm();
    auto a=predicates,b=predicates[predicatePitch];
    auto low=scratch,high=scratch[scratchPitch],zero=scratch[2*scratchPitch];
    const uint8_t repeats=count/64;
    const uint8_t coefficientStride=BROADCAST_ENDPOINTS?0:8;
    const BinaryRepeatParams both{1,1,1,8,coefficientStride,coefficientStride};
    const BinaryRepeatParams coefficient{1,1,1,8,8,coefficientStride};
    Duplicate(zero,0.0F,count);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t axis=0;axis<3;++axis) {
        const uint32_t u=(axis+1)%3,v=(axis+2)%3;
        Compare(a,endpoints[u*endpointPitch],endpoints[(u+3)*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,both);
        Compare(b,endpoints[v*endpointPitch],endpoints[(v+3)*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,both);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Compare(b,points[u*pointPitch],endpoints[u*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,coefficient);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Compare(b,points[v*pointPitch],endpoints[v*endpointPitch],CMPMODE::EQ,uint64_t(64),repeats,coefficient);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Min(low,endpoints[axis*endpointPitch],endpoints[(axis+3)*endpointPitch],uint64_t(64),repeats,both);
        Max(high,endpoints[axis*endpointPitch],endpoints[(axis+3)*endpointPitch],uint64_t(64),repeats,both);PipeBarrier<PIPE_V>();
        Compare(b,points[axis*pointPitch],low,CMPMODE::GE,count);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Compare(b,points[axis*pointPitch],high,CMPMODE::LE,count);PipeBarrier<PIPE_V>();
        And(a.ReinterpretCast<uint16_t>(),a.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
        Select(scores,a,zero,scores,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();
    }
    SetMaskNorm();
}

// Exact floating expansions are used only when a result can be geometric zero.
// Binary16 coordinates and their differences have a 2^-24 quantum, so products
// and their exact product errors stay in the normal binary32 range. Wider binary32 inputs
// receive an explicit operand-range certificate before a zero can be accepted.

template<bool BOUNDS_CERTIFIED, bool REFINE_NEAR, class StageGeometryType, class StageDeltaType, class StageTmpType, class StageExpansionType, class StageCertifiedType, class StageTestType, class StageSameType, class StageUnaryType, class StagePointComponentsType, class StageDirectionComponentsType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresSetMaskNormStage(StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageExpansionType& expansion, StageCertifiedType& certified, StageTestType& test, StageSameType& same, StageUnaryType& unary, StagePointComponentsType& pointComponents, StageDirectionComponentsType& directionComponents) {
    {
    for(uint32_t axis=0;axis<3;++axis) {
        const uint32_t next=(axis+1)%3;
        uint32_t limbs=0;
        SharedCrossProductExpansion1(pointComponents, directionComponents, delta, next, axis, tmp, same, unary, limbs, expansion, test, certified);
        if constexpr(BOUNDS_CERTIFIED || REFINE_NEAR) {
            Duplicate(geometry[axis*64],0.0F,64);PipeBarrier<PIPE_V>();
            for(uint32_t k=0;k<limbs;++k) {
                Add(geometry[axis*64],geometry[axis*64],expansion[k*64],64);PipeBarrier<PIPE_V>();
            }
        }
    }
    }
}

template<bool BOUNDS_CERTIFIED, bool REFINE_NEAR, class StageScoresType, class StageGeometryType, class StageDeltaType, class StageTmpType, class StageTestType, class StageOtherType, class StageStartType, class StageSafeCandidatesType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresAddStage(StageScoresType& scores, StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageTestType& test, StageOtherType& other, StageStartType& start, StageSafeCandidatesType& safeCandidates) {
    if constexpr(BOUNDS_CERTIFIED || REFINE_NEAR) {
        constexpr uint32_t scoreLanes=REFINE_NEAR?64:8;
        // Interior projection predicates permit the compensated cross
        // product avoids subtracting a rounded closest point from P.
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Add(tmp[(2+axis)*64],delta[(3+axis)*64],delta[(9+axis)*64],scoreLanes);
        PipeBarrier<PIPE_V>();
        Mul(tmp,tmp[2*64],tmp[2*64],scoreLanes);
        Mul(tmp[64],geometry,geometry,scoreLanes);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=1;axis<3;++axis) {
            MulAddDst(tmp,tmp[(2+axis)*64],tmp[(2+axis)*64],scoreLanes);
            MulAddDst(tmp[64],geometry[axis*64],geometry[axis*64],scoreLanes);PipeBarrier<PIPE_V>();
        }
        CompareScalar(test,tmp,0.0F,CMPMODE::GT,64);PipeBarrier<PIPE_V>();
        Select(tmp,test,tmp,1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,scoreLanes);PipeBarrier<PIPE_V>();
        Div(tmp[64],tmp[64],tmp,scoreLanes);PipeBarrier<PIPE_V>();
        Duplicate(other.template ReinterpretCast<uint32_t>(),uint32_t(0),2);PipeBarrier<PIPE_V>();
        Duplicate(other.template ReinterpretCast<uint32_t>(),uint32_t(safeCandidates),REFINE_NEAR?2:1);PipeBarrier<PIPE_V>();
        if constexpr(REFINE_NEAR) {
            SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(2));
            Duplicate<uint32_t,false>(other.template ReinterpretCast<uint32_t>(),uint32_t(safeCandidates>>32),uint64_t(64),1,1,8);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
        }
        CompareScalar(test,tmp[64],__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
        And(other.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),REFINE_NEAR?4:1);PipeBarrier<PIPE_V>();
        Select(scores[start],other,tmp[64],scores[start],SELMODE::VSEL_TENSOR_TENSOR_MODE,scoreLanes);PipeBarrier<PIPE_V>();
    }
}

template<bool REFINE_NEAR, class StageGeometryType, class StageDeltaType, class StageTmpType, class StageCertifiedType, class StageTestType, class StageOtherType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresDuplicateStage(StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageCertifiedType& certified, StageTestType& test, StageOtherType& other) {
    if constexpr(REFINE_NEAR) {
        // Test the projection against both endpoints without subtracting
        // two large, almost equal squared lengths.
        Duplicate(tmp,0.0F,64);Duplicate(tmp[64],0.0F,64);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Mul(tmp[2*64],delta[axis*64],delta[(3+axis)*64],64);
            Mul(tmp[3*64],delta[(6+axis)*64],delta[(3+axis)*64],64);PipeBarrier<PIPE_V>();
            MulAddDst(tmp[2*64],delta[axis*64],delta[(9+axis)*64],64);
            MulAddDst(tmp[3*64],delta[(6+axis)*64],delta[(9+axis)*64],64);PipeBarrier<PIPE_V>();
            Add(tmp[2*64],tmp[2*64],tmp[3*64],64);PipeBarrier<PIPE_V>();
            Add(tmp,tmp,tmp[2*64],64);
            Sub(tmp[4*64],geometry[axis*64],geometry[(6+axis)*64],64);PipeBarrier<PIPE_V>();
            Sub(tmp[6*64],geometry[axis*64],tmp[4*64],64);PipeBarrier<PIPE_V>();
            Add(tmp[7*64],tmp[4*64],tmp[6*64],64);
            Sub(tmp[5*64],tmp[6*64],geometry[(6+axis)*64],64);PipeBarrier<PIPE_V>();
            Sub(tmp[7*64],geometry[axis*64],tmp[7*64],64);PipeBarrier<PIPE_V>();
            Add(tmp[5*64],tmp[5*64],tmp[7*64],64);PipeBarrier<PIPE_V>();
            Mul(tmp[2*64],tmp[4*64],delta[(3+axis)*64],64);
            Mul(tmp[3*64],tmp[5*64],delta[(3+axis)*64],64);PipeBarrier<PIPE_V>();
            MulAddDst(tmp[2*64],tmp[4*64],delta[(9+axis)*64],64);
            MulAddDst(tmp[3*64],tmp[5*64],delta[(9+axis)*64],64);PipeBarrier<PIPE_V>();
            Add(tmp[2*64],tmp[2*64],tmp[3*64],64);PipeBarrier<PIPE_V>();
            Add(tmp[64],tmp[64],tmp[2*64],64);PipeBarrier<PIPE_V>();
        }
        CompareScalar(test,tmp,0.0F,CMPMODE::GE,64);
        CompareScalar(other,tmp[64],0.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),4);
        Mul(tmp,delta[3*64],delta[3*64],64);PipeBarrier<PIPE_V>();
        MulAddDst(tmp,delta[4*64],delta[4*64],64);PipeBarrier<PIPE_V>();
        MulAddDst(tmp,delta[5*64],delta[5*64],64);PipeBarrier<PIPE_V>();
        CompareScalar(test,tmp,0.0F,CMPMODE::GT,64);PipeBarrier<PIPE_V>();
        And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
    }
}

template<bool BOUNDS_CERTIFIED, bool REFINE_NEAR, class StageGeometryType, class StageDeltaType, class StageTmpType, class StageExpansionType, class StageCertifiedType, class StageTestType, class StageOtherType, class StageSameType, class StageUnaryType, class StagePointComponentsType, class StageDirectionComponentsType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresMulStage(StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageExpansionType& expansion, StageCertifiedType& certified, StageTestType& test, StageOtherType& other, StageSameType& same, StageUnaryType& unary, StagePointComponentsType& pointComponents, StageDirectionComponentsType& directionComponents) {
    if(pointComponents==1 && directionComponents==1) {
        // With exact coordinate differences, each cross-product has two
        // products. Equality of both rounded products and exact errors
        // is an exact equality certificate, avoiding expansion merging.
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            const uint32_t next=(axis+1)%3;
            Mul(tmp,delta[axis*64],delta[(3+next)*64],64);
            Mul(tmp[2*64],delta[next*64],delta[(3+axis)*64],64);PipeBarrier<PIPE_V>();
            ExactProductError<64>(tmp[64],delta[axis*64],delta[(3+next)*64],tmp,expansion);
            ExactProductError<64>(tmp[3*64],delta[next*64],delta[(3+axis)*64],tmp[2*64],expansion);
            if constexpr(BOUNDS_CERTIFIED || REFINE_NEAR) {
                Sub(geometry[axis*64],tmp,tmp[2*64],64);
                Sub(tmp[4*64],tmp[64],tmp[3*64],64);PipeBarrier<PIPE_V>();
                Add(geometry[axis*64],geometry[axis*64],tmp[4*64],64);PipeBarrier<PIPE_V>();
            }
            Compare(test,tmp,tmp[2*64],CMPMODE::EQ,64);
            Compare(other,tmp[64],tmp[3*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(test.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    } else { CorrectCollinearZeroScoresSetMaskNormStage<BOUNDS_CERTIFIED, REFINE_NEAR>(geometry, delta, tmp, expansion, certified, test, same, unary, pointComponents, directionComponents); }
}

template<bool HALF_INPUT, class StageDeltaType, class StageTmpType, class StageCertifiedType, class StageTestType, class StageOtherType, class StagePointComponentsType, class StageDirectionComponentsType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresAbsStage(StageDeltaType& delta, StageTmpType& tmp, StageCertifiedType& certified, StageTestType& test, StageOtherType& other, StagePointComponentsType& pointComponents, StageDirectionComponentsType& directionComponents) {
    if constexpr(!HALF_INPUT) {
        SharedCrossProductScaling2(pointComponents, directionComponents, tmp, delta, test, other, certified);
    }
}

template<class StageGeometryType, class StageDeltaType, class StageTmpType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresSubStage(StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp) {
SharedCrossProductRounding4(geometry, delta, tmp);
}

template<bool BOUNDS_CERTIFIED, bool REFINE_NEAR, class StageGeometryType, class StageTmpType, class StageCertifiedType, class StageTestType, class StageOtherType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresMinStage(StageGeometryType& geometry, StageTmpType& tmp, StageCertifiedType& certified, StageTestType& test, StageOtherType& other) {
    if constexpr(!BOUNDS_CERTIFIED && !REFINE_NEAR) {
#pragma unroll
    for(uint32_t axis=0;axis<3;++axis) {
        Min(tmp,geometry[(axis+3)*64],geometry[(axis+6)*64],64);
        Max(tmp[64],geometry[(axis+3)*64],geometry[(axis+6)*64],64);PipeBarrier<PIPE_V>();
        Compare(test,geometry[axis*64],tmp,CMPMODE::GE,64);
        Compare(other,geometry[axis*64],tmp[64],CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        And(test.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
    }
    }
}


template<bool BOUNDS_CERTIFIED, class StageCandidateBitsType, class StageScoresType, class StageLimitType, class StageTmpType, class StageCertifiedType, class StageTestType, class StageStartType>
__aicore__ __attribute__((always_inline)) inline bool CorrectCollinearZeroScoresSetMaskNormStage2DuplicateStage(StageCandidateBitsType& candidateBits, StageScoresType& scores, StageLimitType& limit, StageTmpType& tmp, StageCertifiedType& certified, StageTestType& test, StageStartType& start) {
    if constexpr(BOUNDS_CERTIFIED) {
        Duplicate(certified.template ReinterpretCast<uint32_t>(),uint32_t(0),2);PipeBarrier<PIPE_V>();
        Duplicate(certified.template ReinterpretCast<uint32_t>(),uint32_t(candidateBits),1);PipeBarrier<PIPE_V>();
    } else {
    WholeReduceMin(tmp,scores[start],64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Compare(test,tmp,limit,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if(!(test.GetValue(0)&1U))return true;
    CompareScalar(certified,scores[start],0.0F,CMPMODE::GT,64);
    CompareScalar(test,scores[start],__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
    And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if(!certified.template ReinterpretCast<uint64_t>().GetValue(0))return true;
    }
    return false;
}
template<bool HALF_INPUT, bool BOUNDS_CERTIFIED, bool REFINE_NEAR, class StagePointPitchType, class StageEndpointPitchType, class StageCountType, class StageBroadcastEndpointsType, class StageCandidateBitsType, class StageScoresType, class StagePointsType, class StageEndpointsType, class StageLimitType, class StageGeometryType, class StageDeltaType, class StageTmpType, class StageExpansionType, class StageCertifiedType, class StageTestType, class StageOtherType, class StageSameType, class StageUnaryType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearZeroScoresSetMaskNormStage2(StagePointPitchType& pointPitch, StageEndpointPitchType& endpointPitch, StageCountType& count, StageBroadcastEndpointsType& broadcastEndpoints, StageCandidateBitsType& candidateBits, StageScoresType& scores, StagePointsType& points, StageEndpointsType& endpoints, StageLimitType& limit, StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageExpansionType& expansion, StageCertifiedType& certified, StageTestType& test, StageOtherType& other, StageSameType& same, StageUnaryType& unary) {
    for(uint32_t start=0;start<count;start+=64) {
        SetMaskNorm();
        if (CorrectCollinearZeroScoresSetMaskNormStage2DuplicateStage<BOUNDS_CERTIFIED>(candidateBits, scores, limit, tmp, certified, test, start)) continue;
        SetMaskNorm();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Copy(geometry[axis*64],points[axis*pointPitch+start],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+3)*64],endpoints[axis*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+6)*64],endpoints[(axis+3)*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        CorrectCollinearZeroScoresMinStage<BOUNDS_CERTIFIED, REFINE_NEAR>(geometry, tmp, certified, test, other);
        // A point outside the segment's coordinate bounds cannot be collinear
        // within that segment. Subsequent predicates only clear these bits.
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(!certified.template ReinterpretCast<uint64_t>().GetValue(0))continue;
        CorrectCollinearZeroScoresSubStage(geometry, delta, tmp);
        CorrectCollinearZeroScoresDuplicateStage<REFINE_NEAR>(geometry, delta, tmp, certified, test, other);
        SharedScaledPointDifference3(tmp, delta, certified, test);
        const uint32_t componentBits=test.template ReinterpretCast<uint16_t>().GetValue(0);
        const uint32_t pointComponents=(componentBits&1U)?1:2;
        const uint32_t directionComponents=(componentBits&256U)?1:2;
        CorrectCollinearZeroScoresAbsStage<HALF_INPUT>(delta, tmp, certified, test, other, pointComponents, directionComponents);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t safeCandidates=certified.template ReinterpretCast<uint64_t>().GetValue(0);
        if(!safeCandidates)continue;
        CorrectCollinearZeroScoresMulStage<BOUNDS_CERTIFIED, REFINE_NEAR>(geometry, delta, tmp, expansion, certified, test, other, same, unary, pointComponents, directionComponents);
        Duplicate(tmp,0.0F,64);PipeBarrier<PIPE_V>();
        Select(scores[start],certified,tmp,scores[start],SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        CorrectCollinearZeroScoresAddStage<BOUNDS_CERTIFIED, REFINE_NEAR>(scores, geometry, delta, tmp, test, other, start, safeCandidates);
    }
    SetMaskNorm();
}
template<bool HALF_INPUT,bool BOUNDS_CERTIFIED=false,bool REFINE_NEAR=HALF_INPUT && !BOUNDS_CERTIFIED>
__aicore__ __attribute__((noinline)) void CorrectCollinearZeroScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t limitAddress,
    uint32_t predicateAddress,uint32_t count,bool broadcastEndpoints=false,uint64_t candidateBits=~uint64_t(0)) {
    TreeExactCorrectionViews arena(scoreAddress,pointAddress,pointPitch,endpointAddress,endpointPitch,storageAddress,limitAddress,predicateAddress,count);
    auto geometry=arena.storage,delta=arena.storage[9*64],tmp=arena.storage[21*64],expansion=arena.storage[29*64];
    auto certified=arena.predicates,test=arena.predicates[32],other=arena.predicates[64];
    const BinaryRepeatParams same{1,1,1,8,8,8};
    const UnaryRepeatParams unary{1,1,8,8};
    CorrectCollinearZeroScoresSetMaskNormStage2<HALF_INPUT, BOUNDS_CERTIFIED, REFINE_NEAR>(pointPitch, endpointPitch, count, broadcastEndpoints, candidateBits, arena.scores, arena.points, arena.endpoints, arena.limit, geometry, delta, tmp, expansion, certified, test, other, same, unary);
}
