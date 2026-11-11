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

template<class StageGeometryType, class StageDeltaType, class StageTmpType, class StageExpansionType, class StageCertifiedType, class StageTestType, class StageSameType, class StageUnaryType, class StagePointComponentsType, class StageDirectionComponentsType>
__aicore__ __attribute__((always_inline)) inline void CorrectExactLineScoresSetMaskNormStage(StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageExpansionType& expansion, StageCertifiedType& certified, StageTestType& test, StageSameType& same, StageUnaryType& unary, StagePointComponentsType& pointComponents, StageDirectionComponentsType& directionComponents) {
    {
    for(uint32_t axis=0;axis<3;++axis) {
        const uint32_t next=(axis+1)%3;
        uint32_t limbs=0;
        SharedCrossProductExpansion1(pointComponents, directionComponents, delta, next, axis, tmp, same, unary, limbs, expansion, test, certified);
        {
            Duplicate(geometry[axis*64],0.0F,64);PipeBarrier<PIPE_V>();
            for(uint32_t k=0;k<limbs;++k) {
                Add(geometry[axis*64],geometry[axis*64],expansion[k*64],64);PipeBarrier<PIPE_V>();
            }
        }
    }
    }
}

template<class StageGeometryType, class StageDeltaType, class StageTmpType, class StageExpansionType, class StageCertifiedType, class StageTestType, class StageOtherType, class StageSameType, class StageUnaryType, class StagePointComponentsType, class StageDirectionComponentsType>
__aicore__ __attribute__((always_inline)) inline void CorrectExactLineScoresMulStage(StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageExpansionType& expansion, StageCertifiedType& certified, StageTestType& test, StageOtherType& other, StageSameType& same, StageUnaryType& unary, StagePointComponentsType& pointComponents, StageDirectionComponentsType& directionComponents) {
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
            {
                Sub(geometry[axis*64],tmp,tmp[2*64],64);
                Sub(tmp[4*64],tmp[64],tmp[3*64],64);PipeBarrier<PIPE_V>();
                Add(geometry[axis*64],geometry[axis*64],tmp[4*64],64);PipeBarrier<PIPE_V>();
            }
            Compare(test,tmp,tmp[2*64],CMPMODE::EQ,64);
            Compare(other,tmp[64],tmp[3*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(test.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    } else { CorrectExactLineScoresSetMaskNormStage(geometry, delta, tmp, expansion, certified, test, same, unary, pointComponents, directionComponents); }
}

template<class StageDeltaType, class StageTmpType, class StageCertifiedType, class StageTestType, class StageOtherType, class StagePointComponentsType, class StageDirectionComponentsType>
__aicore__ __attribute__((always_inline)) inline void CorrectExactLineScoresAbsStage(StageDeltaType& delta, StageTmpType& tmp, StageCertifiedType& certified, StageTestType& test, StageOtherType& other, StagePointComponentsType& pointComponents, StageDirectionComponentsType& directionComponents) {

        SharedCrossProductScaling2(pointComponents, directionComponents, tmp, delta, test, other, certified);

}

template<class StageGeometryType, class StageDeltaType, class StageTmpType>
__aicore__ __attribute__((always_inline)) inline void CorrectExactLineScoresSubStage(StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp) {
SharedCrossProductRounding4(geometry, delta, tmp);
}

template<class StagePointPitchType, class StageEndpointPitchType, class StageCountType, class StageBroadcastEndpointsType, class StageCandidateBitsType, class StageScoresType, class StagePointsType, class StageEndpointsType, class StageGeometryType, class StageDeltaType, class StageTmpType, class StageExpansionType, class StageCertifiedType, class StageTestType, class StageOtherType, class StageSameType, class StageUnaryType>
__aicore__ __attribute__((always_inline)) inline void CorrectExactLineScoresSetMaskNormStage2(StagePointPitchType& pointPitch, StageEndpointPitchType& endpointPitch, StageCountType& count, StageBroadcastEndpointsType& broadcastEndpoints, StageCandidateBitsType& candidateBits, StageScoresType& scores, StagePointsType& points, StageEndpointsType& endpoints, StageGeometryType& geometry, StageDeltaType& delta, StageTmpType& tmp, StageExpansionType& expansion, StageCertifiedType& certified, StageTestType& test, StageOtherType& other, StageSameType& same, StageUnaryType& unary) {
    for(uint32_t start=0;start<count;start+=64) {
        SetMaskNorm();
        {
            Duplicate(certified.template ReinterpretCast<uint32_t>(),uint32_t(0),2);PipeBarrier<PIPE_V>();
            Duplicate(certified.template ReinterpretCast<uint32_t>(),uint32_t(candidateBits),1);PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Copy(geometry[axis*64],points[axis*pointPitch+start],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+3)*64],endpoints[axis*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
            Copy(geometry[(axis+6)*64],endpoints[(axis+3)*endpointPitch+(broadcastEndpoints?0:start)],64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        // Incoming certified lanes are narrowed by exact line predicates.
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(!certified.template ReinterpretCast<uint64_t>().GetValue(0))continue;
        CorrectExactLineScoresSubStage(geometry, delta, tmp);
        SharedScaledPointDifference3(tmp, delta, certified, test);
        const uint32_t componentBits=test.template ReinterpretCast<uint16_t>().GetValue(0);
        const uint32_t pointComponents=(componentBits&1U)?1:2;
        const uint32_t directionComponents=(componentBits&256U)?1:2;
          CorrectExactLineScoresAbsStage(delta, tmp, certified, test, other, pointComponents, directionComponents);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t safeCandidates=certified.template ReinterpretCast<uint64_t>().GetValue(0);
        if(!safeCandidates)continue;
        CorrectExactLineScoresMulStage(geometry, delta, tmp, expansion, certified, test, other, same, unary, pointComponents, directionComponents);
        Duplicate(tmp,0.0F,64);PipeBarrier<PIPE_V>();
        Select(scores[start],certified,tmp,scores[start],SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();

    }
}
__aicore__ __attribute__((noinline)) inline void CorrectExactLineScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t limitAddress,
    uint32_t predicateAddress,uint32_t count,bool broadcastEndpoints=false,uint64_t candidateBits=~uint64_t(0)) {
    TreeExactCorrectionViews arena(scoreAddress,pointAddress,pointPitch,endpointAddress,endpointPitch,storageAddress,limitAddress,predicateAddress,count);
    auto geometry=arena.storage,delta=arena.storage[9*64],tmp=arena.storage[21*64],expansion=arena.storage[29*64];
    auto certified=arena.predicates,test=arena.predicates[32],other=arena.predicates[64];
    const BinaryRepeatParams same{1,1,1,8,8,8};
    const UnaryRepeatParams unary{1,1,8,8};
    CorrectExactLineScoresSetMaskNormStage2(pointPitch, endpointPitch, count, broadcastEndpoints, candidateBits, arena.scores, arena.points, arena.endpoints, geometry, delta, tmp, expansion, certified, test, other, same, unary);
    SetMaskNorm();
}
