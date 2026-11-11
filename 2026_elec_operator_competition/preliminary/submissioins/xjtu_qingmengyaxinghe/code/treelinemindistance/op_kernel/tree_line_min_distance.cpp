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

#define TREE_NATIVE_FLOAT_TOKEN_float 1
#define TREE_NATIVE_FLOAT_TOKEN_half 0
#define TREE_NATIVE_FLOAT_TOKEN_IMPL(token) TREE_NATIVE_FLOAT_TOKEN_##token
#define TREE_NATIVE_FLOAT_TOKEN(token) TREE_NATIVE_FLOAT_TOKEN_IMPL(token)
#define TREE_NATIVE_HALF_TOKEN_float 0
#define TREE_NATIVE_HALF_TOKEN_half 1
#define TREE_NATIVE_HALF_TOKEN_IMPL(token) TREE_NATIVE_HALF_TOKEN_##token
#define TREE_NATIVE_HALF_TOKEN(token) TREE_NATIVE_HALF_TOKEN_IMPL(token)
#include "kernel_operator.h"

#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

using AscendC::Abs;
using AscendC::Add;
using AscendC::Adds;
using AscendC::And;
using AscendC::Axpy;
using AscendC::BinaryRepeatParams;
using AscendC::Brcb;
using AscendC::BrcbRepeatParams;
using AscendC::Cast;
using AscendC::CMPMODE;
using AscendC::Compare;
using AscendC::CompareScalar;
using AscendC::Copy;
using AscendC::CopyRepeatParams;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::Div;
using AscendC::Duplicate;
using AscendC::FusedMulAdd;
using AscendC::Gather;
using AscendC::GatherMask;
using AscendC::GatherMaskParams;
using AscendC::GetBlockIdx;
using AscendC::GetSFFValue;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::ICachePreLoad;
using AscendC::InitSocState;
using AscendC::LocalTensor;
using AscendC::MaskMode;
using AscendC::Max;
using AscendC::Maxs;
using AscendC::Min;
using AscendC::Mins;
using AscendC::MrgSort;
using AscendC::MrgSort4Info;
using AscendC::MrgSortSrcList;
using AscendC::Mul;
using AscendC::MulAddDst;
using AscendC::Muls;
using AscendC::Not;
using AscendC::Or;
using AscendC::PipeBarrier;
using AscendC::ReduceOrder;
using AscendC::ResetMask;
using AscendC::RoundMode;
using AscendC::SELMODE;
using AscendC::Select;
using AscendC::SetCmpMask;
using AscendC::SetFlag;
using AscendC::SetMaskCount;
using AscendC::SetMaskNorm;
using AscendC::SetVectorMask;
using AscendC::ShiftLeft;
using AscendC::ShiftRight;
using AscendC::Sort32;
using AscendC::Sqrt;
using AscendC::Sub;
using AscendC::SyncAll;
using AscendC::TPosition;
using AscendC::UnaryRepeatParams;
using AscendC::WaitFlag;
using AscendC::WholeReduceMax;
using AscendC::WholeReduceMin;
using AscendC::WholeReduceSum;
#include "tree_vector_geometry.h"
#if defined(DTYPE_TREE_POINTS)
// All arithmetic payloads are sixteen-lane vector packets. Scalar loops
// traverse radix digits, axes and segment packets, never coordinate values.
#include "tree_float_rational16.h"


#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
// Called after refinement and index remapping, when all six work planes are dead.
// Columns, compactP, prefetched rawP, and output rows remain outside this arena.

template<class StageSegmentsType, class StageWorkType, class StageSgType, class StagePointType, class StagePacketType, class StageLaneType, class StageOffsetsType, class StageExactType>
__aicore__ __attribute__((always_inline)) inline void RefineWideFloatRowsBodyDuplicateStage(StageSegmentsType& segments, StageWorkType& work, StageSgType& sg, StagePointType& point, StagePacketType& packet, StageLaneType& lane, StageOffsetsType& offsets, StageExactType& exact) {
    for(uint32_t start=0;start<segments;start+=16) {
        const uint32_t count=segments-start<16?segments-start:16;
        Duplicate(packet,0.0F,96);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        DataCopyPad(packet,sg[start*6],DataCopyExtParams{1,count*24,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        exact.MakeScores(point,packet,offsets.template ReinterpretCast<uint32_t>(),segments,start);
        exact.Tournament(lane.template ReinterpretCast<uint32_t>());exact.Merge(work[20000],start==0);
    }
}

template<class StageSegmentsType, class StageWorkType, class StageIndicesType, class StageDistancesType, class StageSgType, class StagePointType, class StagePacketType, class StageTempType, class StageLaneType, class StageOffsetsType, class StageRowType, class StageReducedType, class StageMaxExpType>
__aicore__ __attribute__((always_inline)) inline void RefineWideFloatRowsBodyCastStage(uint32_t workAddress, StageSegmentsType& segments, StageWorkType& work, StageIndicesType& indices, StageDistancesType& distances, StageSgType& sg, StagePointType& point, StagePacketType& packet, StageTempType& temp, StageLaneType& lane, StageOffsetsType& offsets, StageRowType& row, StageReducedType& reduced, StageMaxExpType& maxExp) {
    PipeBarrier<PIPE_V>();
    Cast(reduced[8].template ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);uint32_t minExp=uint32_t(reduced[8].template ReinterpretCast<int32_t>().GetValue(0));
    if(minExp>maxExp)minExp=maxExp;
    const uint32_t required=(maxExp-minExp+28+11)/12,limbs=required<4?4:required;

    TreeFloatRational16 exact(workAddress,limbs,minExp);exact.Init();
    Duplicate(lane,int32_t(0),16);PipeBarrier<PIPE_V>();
    const uint64_t laneMasks[4]={0xaaaa,0xcccc,0xf0f0,0xff00};
    for(uint32_t k=0;k<4;++k) {
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),laneMasks[k]);
        Adds<int32_t,false>(lane,lane,int32_t(1U<<k),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
    Muls(offsets,lane,int32_t(24),16);PipeBarrier<PIPE_V>();
    RefineWideFloatRowsBodyDuplicateStage(segments, work, sg, point, packet, lane, offsets, exact);
    exact.WriteDistance(temp,lane.template ReinterpretCast<uint32_t>());
    Cast(temp[16].template ReinterpretCast<int32_t>(),exact.Plane(24*limbs),RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(0),uint64_t(1)<<(row%8));
    Copy<float,false>(distances[(row/8)*8],temp,uint64_t(64),1,CopyRepeatParams(1,0,8,0));
    Copy<int32_t,false>(indices[(row/8)*8],temp[16].template ReinterpretCast<int32_t>(),uint64_t(64),1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

}

template<class StageSegmentsType, class StageLiveRowsType, class StageWideRowsType, class StageWorkType, class StagePointsType, class StageIndicesType, class StageDistancesType, class StageSgType, class StagePaddedType, class StageSfType, class StagePointType, class StagePacketType, class StageTempType, class StageLaneType, class StageOffsetsType>
__aicore__ __attribute__((always_inline)) inline void RefineWideFloatRowsBodyDuplicateStage2(uint32_t workAddress, StageSegmentsType& segments, StageLiveRowsType& liveRows, StageWideRowsType& wideRows, StageWorkType& work, StagePointsType& points, StageIndicesType& indices, StageDistancesType& distances, StageSgType& sg, StagePaddedType& padded, StageSfType& sf, StagePointType& point, StagePacketType& packet, StageTempType& temp, StageLaneType& lane, StageOffsetsType& offsets) {
    for(uint32_t row=0;row<liveRows;++row) {
    if((wideRows&(uint64_t(1)<<row))==0)continue;
    PipeBarrier<PIPE_ALL>();
    Duplicate(sf,0.0F,padded+64);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    DataCopyPad(sf,sg,DataCopyExtParams{1,segments*24,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    for(uint32_t axis=0;axis<3;++axis) {
        Brcb(temp,points[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(point[axis*16],temp[(row%8)*8],16,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
    }
    Copy(sf[padded],point,48,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    const uint32_t coords=padded+64;
    auto exponent=work,scratch=work[6208],record=work[8192],reduced=work[8320];
    auto mask=work[7000].template ReinterpretCast<uint8_t>();
    ShiftRight(exponent.template ReinterpretCast<uint32_t>(),sf.template ReinterpretCast<uint32_t>(),uint32_t(23),coords);
    Duplicate(scratch.template ReinterpretCast<uint32_t>(),uint32_t(255),64);PipeBarrier<PIPE_V>();
    // Broadcast bit mask to all coordinates without scalar payload reads.
    for(uint32_t k=0;k<coords;k+=64)And(exponent[k].template ReinterpretCast<uint16_t>(),exponent[k].template ReinterpretCast<uint16_t>(),scratch.template ReinterpretCast<uint16_t>(),128);
    PipeBarrier<PIPE_V>();Cast(exponent,exponent.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,coords);PipeBarrier<PIPE_V>();
    Maxs(exponent,exponent,1.0F,coords);PipeBarrier<PIPE_V>();
    Duplicate(record,0.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMax(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(reduced[8].template ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);const uint32_t maxExp=uint32_t(reduced[8].template ReinterpretCast<int32_t>().GetValue(0));
    // Magnitude bits distinguish subnormals from signed zeros without FTZ.
    FilterZeroMagnitudeExponents(scratch,work[6272],sf,exponent,mask,coords);
    Duplicate(record,255.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMin(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);RefineWideFloatRowsBodyCastStage(workAddress, segments, work, indices, distances, sg, point, packet, temp, lane, offsets, row, reduced, maxExp);}
}
__aicore__ __attribute__((noinline)) inline void RefineWideFloatRowsBody(
    GM_ADDR segmentAddress,uint32_t segments,uint32_t workAddress,
    uint32_t pointAddress,uint32_t indexAddress,uint32_t distanceAddress,
    uint32_t liveRows,uint64_t wideRows) {
    LocalTensor<float> work(TPosition::VECCALC,workAddress,25344);
    LocalTensor<float> points(TPosition::VECCALC,pointAddress,192);
    LocalTensor<int32_t> indices(TPosition::VECCALC,indexAddress,64);
    LocalTensor<float> distances(TPosition::VECCALC,distanceAddress,64);
    GlobalTensor<float> sg;sg.SetGlobalBuffer((__gm__ float*)segmentAddress);
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

    // 79,872 B arithmetic + 9,280 B best score + a 384 B streamed packet.
    // Range detection temporarily stages all coefficients before arithmetic starts.
    const uint32_t padded=(6*segments+63)/64*64;
    auto sf=work[12000],point=work[22528],packet=work[22656],temp=work[22800];
    auto lane=work[22912].ReinterpretCast<int32_t>();
    auto offsets=work[22928].ReinterpretCast<int32_t>();
    RefineWideFloatRowsBodyDuplicateStage2(workAddress, segments, liveRows, wideRows, work, points, indices, distances, sg, padded, sf, point, packet, temp, lane, offsets);
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}

__aicore__ __attribute__((always_inline)) inline void RefineWideFloatRows(
    GM_ADDR segmentAddress,uint32_t segments,uint32_t workAddress,
    uint32_t pointAddress,uint32_t indexAddress,uint32_t distanceAddress,
    uint32_t liveRows,bool coefficientSafe) {
    LocalTensor<float> work(TPosition::VECCALC,workAddress,25344);
    LocalTensor<float> points(TPosition::VECCALC,pointAddress,192);
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    const uint64_t active=(~uint64_t(0))>>(64-liveRows);
    uint64_t wideRows=coefficientSafe?uint64_t(0):active;
    if(coefficientSafe) {
        // Three contiguous coordinate planes share the same vector predicate pass.
        // Integer zero equality retains nonzero subnormals; one V_S joins all axes.
        auto masks=work[384].ReinterpretCast<uint8_t>();
        Duplicate(work[192].ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),192);PipeBarrier<PIPE_V>();
        And(work.ReinterpretCast<uint16_t>(),points.ReinterpretCast<uint16_t>(),work[192].ReinterpretCast<uint16_t>(),384);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,true>(masks,work,1099511627776.0F,CMPMODE::GT,uint64_t(64),uint8_t(3),UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,true>(masks[32],work,9.5367431640625e-7F,CMPMODE::LT,uint64_t(64),uint8_t(3),UnaryRepeatParams(1,1,8,8));
        CompareScalar(masks[64],work.ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,192);PipeBarrier<PIPE_V>();
        Not(masks[64].ReinterpretCast<uint16_t>(),masks[64].ReinterpretCast<uint16_t>(),12);PipeBarrier<PIPE_V>();
        And(masks[32].ReinterpretCast<uint16_t>(),masks[32].ReinterpretCast<uint16_t>(),masks[64].ReinterpretCast<uint16_t>(),12);PipeBarrier<PIPE_V>();
        Or(masks.ReinterpretCast<uint16_t>(),masks.ReinterpretCast<uint16_t>(),masks[32].ReinterpretCast<uint16_t>(),12);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto packed=masks.ReinterpretCast<uint64_t>();
        wideRows=(packed.GetValue(0)|packed.GetValue(1)|packed.GetValue(2))&active;
    }

    if(wideRows)RefineWideFloatRowsBody(segmentAddress,segments,workAddress,
        pointAddress,indexAddress,distanceAddress,liveRows,wideRows);
}


#endif // TREE_STREAM_EXACT

#if defined(DTYPE_TREE_POINTS)
#include "tree_packet_rational64.h"

#include "tree_packet_integer_rational64.h"
#endif

#include "tree_zero_score_correction.h"

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))

#include "tree_exact_line_scores.h"

// Equal supporting lines with strictly interior projections have the same
// exact distance. Only certified lanes reuse the incumbent's computed score.
// Reuse the original necessary-condition rejection before entering the exact
// certificate. A surviving predicate always executes the complete correction.
template<bool NORMAL_MASK>
__aicore__ __attribute__((always_inline)) inline bool CheckCollinearInteriorTies8(
    uint32_t scoreAddress,uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t predicateAddress) {
    TreeInteriorTieViews arena(scoreAddress,endpointAddress,endpointPitch,storageAddress,predicateAddress);
    PipeBarrier<PIPE_V>();SetMaskNorm();
    WholeReduceMin(arena.record,arena.scores,8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t winner=arena.record.ReinterpretCast<uint32_t>().GetValue(1);
    if(winner>=8){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return false;}
    CompareScalar<float,uint8_t,true>(arena.certified,arena.scores,__builtin_inff(),CMPMODE::LT,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    Duplicate(arena.tmp[12*64].ReinterpretCast<uint32_t>(),winner*4,64);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t col=0;col<6;++col)if(col!=2 && col!=5)
        Gather(arena.winnerEndpoints[col*64],arena.endpoints[col*endpointPitch],arena.tmp[12*64].ReinterpretCast<uint32_t>(),0U,64);
    PipeBarrier<PIPE_V>();
    // A necessary common-line condition only. The exact certificate below
    // still decides equality; a rounded cross product never accepts a tie.
    if constexpr(NORMAL_MASK) {
SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
    const BinaryRepeatParams binary{1,1,1,8,8,8};
    const UnaryRepeatParams unary{1,1,8,8};
    Sub<float,false>(arena.tmp,arena.winnerEndpoints[3*64],arena.winnerEndpoints,uint64_t(8),1,binary);
    Sub<float,false>(arena.tmp[64],arena.winnerEndpoints[4*64],arena.winnerEndpoints[64],uint64_t(8),1,binary);
    Sub<float,false>(arena.tmp[2*64],arena.endpoints,arena.winnerEndpoints,uint64_t(8),1,binary);
    Sub<float,false>(arena.tmp[3*64],arena.endpoints[endpointPitch],arena.winnerEndpoints[64],uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    Mul<float,false>(arena.tmp[4*64],arena.tmp[2*64],arena.tmp[64],uint64_t(8),1,binary);
    Mul<float,false>(arena.tmp[5*64],arena.tmp[3*64],arena.tmp,uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    Sub<float,false>(arena.tmp[6*64],arena.tmp[4*64],arena.tmp[5*64],uint64_t(8),1,binary);
    Abs<float,false>(arena.tmp[4*64],arena.tmp[4*64],uint64_t(8),1,unary);Abs<float,false>(arena.tmp[5*64],arena.tmp[5*64],uint64_t(8),1,unary);PipeBarrier<PIPE_V>();
    Abs<float,false>(arena.tmp[6*64],arena.tmp[6*64],uint64_t(8),1,unary);
    Add<float,false>(arena.tmp[7*64],arena.tmp[4*64],arena.tmp[5*64],uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    Muls<float,false>(arena.tmp[7*64],arena.tmp[7*64],3.0517578125e-5F,uint64_t(8),1,unary);PipeBarrier<PIPE_V>();
    Compare<float,uint8_t,false>(arena.test,arena.tmp[6*64],arena.tmp[7*64],CMPMODE::LE,uint64_t(8),1,binary);PipeBarrier<PIPE_V>();
    SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(1));
    And<uint16_t,false>(arena.certified.ReinterpretCast<uint16_t>(),arena.certified.ReinterpretCast<uint16_t>(),arena.test.ReinterpretCast<uint16_t>(),uint64_t(1),1,binary);PipeBarrier<PIPE_V>();

} else {
RefinePotentialCollinearMask(arena.tmp,arena.winnerEndpoints,arena.endpoints,endpointPitch,arena.certified,arena.test);
}
SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t possible=arena.certified.ReinterpretCast<uint16_t>().GetValue(0)&255U;
    const bool possibleTie=(possible&(1U<<winner)) && (possible&(possible-1));
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return possibleTie;
}

__aicore__ __attribute__((always_inline)) inline bool MayHaveCollinearInteriorTies8(
    uint32_t scoreAddress,uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t predicateAddress) { return CheckCollinearInteriorTies8<false>(scoreAddress,endpointAddress,endpointPitch,storageAddress,predicateAddress);
}

__aicore__ __attribute__((always_inline)) inline bool MayHaveCollinearInteriorTies8NormalMask(
    uint32_t scoreAddress,uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t predicateAddress) { return CheckCollinearInteriorTies8<true>(scoreAddress,endpointAddress,endpointPitch,storageAddress,predicateAddress);
}


template<class StageEndpointPitchType, class StageStorageAddressType, class StagePredicateAddressType, class StageEndpointsType, class StageWinnerEndpointsType, class StageTmpType, class StageRecordType, class StageTestType, class StageBitsType>
__aicore__ __attribute__((always_inline)) inline bool CorrectCollinearInteriorTies8DuplicateStage(StageEndpointPitchType& endpointPitch, StageStorageAddressType& storageAddress, StagePredicateAddressType& predicateAddress, StageEndpointsType& endpoints, StageWinnerEndpointsType& winnerEndpoints, StageTmpType& tmp, StageRecordType& record, StageTestType& test, StageBitsType& bits) {
    for(uint32_t end=0;end<2;++end) {
        Duplicate(tmp,0.0F,3*64);Duplicate(tmp[11*64],1.0F,64);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Copy(tmp[axis*64],endpoints[(end*3+axis)*endpointPitch],8,1,CopyRepeatParams(1,1,1,1));
        PipeBarrier<PIPE_V>();
        CorrectExactLineScores(uint32_t(tmp[11*64].GetPhyAddr()),uint32_t(tmp.GetPhyAddr()),64,
            uint32_t(winnerEndpoints.GetPhyAddr()),64,storageAddress,uint32_t(record.GetPhyAddr()),
            predicateAddress,64,false,uint64_t(bits));
        CompareScalar<float,uint8_t,true>(test,tmp[11*64],0.0F,CMPMODE::EQ,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        bits&=test.template ReinterpretCast<uint16_t>().GetValue(0)&255U;
        if(!(bits&(bits-1))){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return true;}
    }
        return false;
}

template<class StageTmpType, class StageCertifiedType, class StageTestType, class StageOtherType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearInteriorTies8AbsStage(StageTmpType& tmp, StageCertifiedType& certified, StageTestType& test, StageOtherType& other) {
#pragma unroll

    for(uint32_t k=0;k<6;++k) {
        Abs(tmp[9*64],tmp[k*64],8);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,true>(test,tmp[9*64],9.313225746154785e-10F,CMPMODE::GE,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});
        CompareScalar<float,uint8_t,true>(other,tmp[9*64],0.0F,CMPMODE::EQ,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Or(test.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
        And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),1);
        CompareScalar<float,uint8_t,true>(test,tmp[9*64],9.223372036854776e18F,CMPMODE::LE,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    }
}

template<class StagePointPitchType, class StageEndpointPitchType, class StageStorageAddressType, class StagePredicateAddressType, class StageScoresType, class StagePointsType, class StageEndpointsType, class StageWinnerEndpointsType, class StageTmpType, class StageRecordType, class StageCertifiedType, class StageTestType, class StageOtherType, class StageWinnerType>
__aicore__ __attribute__((always_inline)) inline void CorrectCollinearInteriorTies8SubStage(StagePointPitchType& pointPitch, StageEndpointPitchType& endpointPitch, StageStorageAddressType& storageAddress, StagePredicateAddressType& predicateAddress, StageScoresType& scores, StagePointsType& points, StageEndpointsType& endpoints, StageWinnerEndpointsType& winnerEndpoints, StageTmpType& tmp, StageRecordType& record, StageCertifiedType& certified, StageTestType& test, StageOtherType& other, StageWinnerType& winner) {
#pragma unroll
    for(uint32_t axis=0;axis<3;++axis) {
        Sub(tmp[axis*64],endpoints[(axis+3)*endpointPitch],endpoints[axis*endpointPitch],8);
        Sub(tmp[(axis+3)*64],points[axis*pointPitch],endpoints[axis*endpointPitch],8);
    }
    PipeBarrier<PIPE_V>();
    // The normal-operand range keeps every product and rounding bound finite
    // or rejects the lane; it also avoids underflow in the dot certificate.
    CorrectCollinearInteriorTies8AbsStage(tmp, certified, test, other);
    Mul(tmp[6*64],tmp[3*64],tmp,8);
    Mul(tmp[7*64],tmp,tmp,8);PipeBarrier<PIPE_V>();
#pragma unroll
    for(uint32_t axis=1;axis<3;++axis) {
        MulAddDst(tmp[6*64],tmp[(axis+3)*64],tmp[axis*64],8);
        MulAddDst(tmp[7*64],tmp[axis*64],tmp[axis*64],8);PipeBarrier<PIPE_V>();
    }
#pragma unroll
    for(uint32_t k=0;k<6;++k)Abs(tmp[k*64],tmp[k*64],8);
    PipeBarrier<PIPE_V>();
    Mul(tmp[8*64],tmp[3*64],tmp,8);PipeBarrier<PIPE_V>();
#pragma unroll
    for(uint32_t axis=1;axis<3;++axis) {
        MulAddDst(tmp[8*64],tmp[(axis+3)*64],tmp[axis*64],8);PipeBarrier<PIPE_V>();
    }
    Add(tmp[10*64],tmp[8*64],tmp[7*64],8);
    Sub(tmp[9*64],tmp[7*64],tmp[6*64],8);PipeBarrier<PIPE_V>();
    Muls(tmp[10*64],tmp[10*64],3.0517578125e-5F,8);PipeBarrier<PIPE_V>();
    Compare<float,uint8_t,true>(test,tmp[6*64],tmp[10*64],CMPMODE::GT,uint64_t(8),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});
    Compare<float,uint8_t,true>(other,tmp[9*64],tmp[10*64],CMPMODE::GT,uint64_t(8),uint8_t(1),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
    And(test.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t bits=certified.template ReinterpretCast<uint16_t>().GetValue(0)&255U;
    if(!(bits&(1U<<winner)) || !(bits&(bits-1))){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return;}
    // The XY rejection and projection certificate do not read winner z.
    // Its byte-offset vector remains live in tmp[12*64] until this point.
    Gather(winnerEndpoints[2*64],endpoints[2*endpointPitch],tmp[12*64].template ReinterpretCast<uint32_t>(),0U,64);
    Gather(winnerEndpoints[5*64],endpoints[5*endpointPitch],tmp[12*64].template ReinterpretCast<uint32_t>(),0U,64);
    PipeBarrier<PIPE_V>();
    if (CorrectCollinearInteriorTies8DuplicateStage(endpointPitch, storageAddress, predicateAddress, endpoints, winnerEndpoints, tmp, record, test, bits)) return;
    Duplicate(certified.template ReinterpretCast<uint32_t>(),bits,1);PipeBarrier<PIPE_V>();
    Brcb(tmp[3*64],record,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    Select(scores,certified,tmp[3*64],scores,SELMODE::VSEL_TENSOR_TENSOR_MODE,8);PipeBarrier<PIPE_V>();
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}
__aicore__ __attribute__((noinline)) inline void CorrectCollinearInteriorTies8(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t endpointAddress,uint32_t endpointPitch,
    uint32_t storageAddress,uint32_t predicateAddress) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,8);
    const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
    const LocalTensor<float> endpoints(TPosition::VECCALC,endpointAddress,6*endpointPitch);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,65*64);
    const LocalTensor<uint8_t> predicates(TPosition::VECCALC,predicateAddress,96);
    auto winnerEndpoints=storage[45*64],tmp=storage[51*64],record=storage[64*64];
    auto certified=predicates,test=predicates[32],other=predicates[64];
    PipeBarrier<PIPE_V>();SetMaskNorm();
    WholeReduceMin(record,scores,8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t winner=record.ReinterpretCast<uint32_t>().GetValue(1);
    if(winner>=8){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return;}
    CompareScalar<float,uint8_t,true>(certified,scores,__builtin_inff(),CMPMODE::LT,uint64_t(8),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    Duplicate(tmp[12*64].ReinterpretCast<uint32_t>(),winner*4,64);PipeBarrier<PIPE_V>();
    #pragma unroll
    for(uint32_t col=0;col<6;++col)if(col!=2 && col!=5)
        Gather(winnerEndpoints[col*64],endpoints[col*endpointPitch],tmp[12*64].ReinterpretCast<uint32_t>(),0U,64);
    PipeBarrier<PIPE_V>();
    // A necessary common-line condition only. The exact certificate below
    // still decides equality; a rounded cross product never accepts a tie.
    RefinePotentialCollinearMask(tmp,winnerEndpoints,endpoints,endpointPitch,certified,test);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t possible=certified.ReinterpretCast<uint16_t>().GetValue(0)&255U;
    if(!(possible&(1U<<winner)) || !(possible&(possible-1))){SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));return;}
CorrectCollinearInteriorTies8SubStage(pointPitch, endpointPitch, storageAddress, predicateAddress, scores, points, endpoints, winnerEndpoints, tmp, record, certified, test, other, winner);
}
#endif


template<bool HALF_INPUT, uint32_t RECORDS, class StageScoresType, class StagePType, class StageAType, class StageCType, class StageWorkType, class StageGeometryType, class StageExactType, class StageMasksType, class StageOffsetsType, class StageCandidatesType>
__aicore__ __attribute__((always_inline)) inline void CorrectTinyZeroScoresDuplicateStage(StageScoresType& scores, StagePType& p, StageAType& a, StageCType& c, StageWorkType& work, StageGeometryType& geometry, StageExactType& exact, StageMasksType& masks, StageOffsetsType& offsets, StageCandidatesType& candidates) {
    if(candidates) {
        Duplicate(geometry,0.0F,9*64);PipeBarrier<PIPE_V>();
        Duplicate(offsets,int32_t(0),8);PipeBarrier<PIPE_V>();
        SetMaskNorm();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
        Adds<int32_t,false>(offsets,offsets,int32_t(1),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
        Adds<int32_t,false>(offsets,offsets,int32_t(2),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
        Adds<int32_t,false>(offsets,offsets,int32_t(4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(32),8);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Gather(geometry[axis*64],p,offsets.template ReinterpretCast<uint32_t>(),axis*4,RECORDS);
            Gather(geometry[(axis+3)*64],a,offsets.template ReinterpretCast<uint32_t>(),axis*4,RECORDS);
            Gather(geometry[(axis+6)*64],c,offsets.template ReinterpretCast<uint32_t>(),axis*4,RECORDS);
        }
        PipeBarrier<PIPE_V>();
        CorrectCollinearZeroScores<HALF_INPUT,true,HALF_INPUT>(uint32_t(scores.GetPhyAddr()),uint32_t(geometry.GetPhyAddr()),64,
            uint32_t(geometry[3*64].GetPhyAddr()),64,uint32_t(exact.GetPhyAddr()),uint32_t(work[5*64].GetPhyAddr()),
            uint32_t(masks.GetPhyAddr()),64,false,candidates);
    }
}
template<bool HALF_INPUT,uint32_t RECORDS=8>
__aicore__ __attribute__((noinline)) void CorrectTinyZeroScores(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t aAddress,uint32_t cAddress,
    uint32_t workAddress,uint32_t geometryAddress,uint32_t exactAddress,
    uint32_t predicateAddress,uint32_t offsetAddress,uint64_t activePairs) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,64);
    const LocalTensor<float> p(TPosition::VECCALC,pointAddress,64);
    const LocalTensor<float> a(TPosition::VECCALC,aAddress,64);
    const LocalTensor<float> c(TPosition::VECCALC,cAddress,64);
    const LocalTensor<float> work(TPosition::VECCALC,workAddress,8*64);
    const LocalTensor<float> geometry(TPosition::VECCALC,geometryAddress,9*64);
    const LocalTensor<float> exact(TPosition::VECCALC,exactAddress,45*64);
    const LocalTensor<uint8_t> masks(TPosition::VECCALC,predicateAddress,512);
    const LocalTensor<int32_t> offsets(TPosition::VECCALC,offsetAddress,64);
    constexpr uint32_t coordinates=RECORDS*8;
                // Certify axis-aligned zero and coordinate bounds directly
                // from the existing eight-coordinate records.
                Min(work[0*64],a,c,coordinates);Max(work[1*64],a,c,coordinates);
                CompareScalar(masks,scores,0.0F,CMPMODE::GT,64);
                CompareScalar(masks[320],scores,__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
                Compare(masks[64],p,work[0*64],CMPMODE::GE,coordinates);
                Compare(masks[128],p,work[1*64],CMPMODE::LE,coordinates);
                Compare(masks[192],a,c,CMPMODE::EQ,coordinates);
                Compare(masks[256],p,a,CMPMODE::EQ,coordinates);PipeBarrier<PIPE_V>();
                And(masks[64].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),4);
                And(masks[192].template ReinterpretCast<uint16_t>(),masks[192].template ReinterpretCast<uint16_t>(),masks[256].template ReinterpretCast<uint16_t>(),4);
                And(masks.template ReinterpretCast<uint16_t>(),masks.template ReinterpretCast<uint16_t>(),masks[320].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint64_t coordinateBounds=masks[64].template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t fixedCoordinates=masks[192].template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t positiveFinite=masks.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t bounded=((coordinateBounds&(coordinateBounds>>1)&(coordinateBounds>>2))&0x0101010101010101ULL);
                const uint64_t axial=((fixedCoordinates&(fixedCoordinates>>1))|(fixedCoordinates&(fixedCoordinates>>2))|((fixedCoordinates>>1)&(fixedCoordinates>>2)))&bounded;
                const uint64_t boundedPairs=(bounded*0x0102040810204080ULL)>>56;
                const uint64_t axialPairs=(axial*0x0102040810204080ULL)>>56;
                Duplicate(masks[256].template ReinterpretCast<uint32_t>(),uint32_t(~axialPairs),2);PipeBarrier<PIPE_V>();
                Select(scores,masks[256],scores,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                const uint64_t candidates=positiveFinite&~axialPairs&activePairs&(HALF_INPUT?~uint64_t(0):boundedPairs);
                CorrectTinyZeroScoresDuplicateStage<HALF_INPUT, RECORDS>(scores, p, a, c, work, geometry, exact, masks, offsets, candidates);
                Select(scores,masks[384],scores,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
    SetMaskNorm();
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
// Finite binary16 coordinates are integral after multiplication by 2^24.
// Base4096 digits fit exact binary32 multiply/add packets: the maximum
// normalized product plus destination and carry is 2^24-1.
using TreeHalfRational64 = TreePacketRational64<4, 24>;

__aicore__ __attribute__((always_inline)) inline uint32_t TreeHalfPendingRows(
    uint32_t scoreAddress,uint32_t storageAddress,uint32_t validRows,uint32_t limitAddress,
    uint32_t recordsAddress,uint32_t offsetsAddress) {
    const LocalTensor<float> scores(TPosition::VECCALC,scoreAddress,32*64);
    const LocalTensor<float> storage(TPosition::VECCALC,storageAddress,8*2080);
    const LocalTensor<float> limit(TPosition::VECCALC,limitAddress,64);
    const LocalTensor<float> records(TPosition::VECCALC,recordsAddress,32*8+64);
    const LocalTensor<uint32_t> offsets(TPosition::VECCALC,offsetsAddress,32+64);
    auto masks=storage[220*64].ReinterpretCast<uint8_t>();
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    Gather(storage[2048],records,offsets,0U,32);PipeBarrier<PIPE_V>();
    Muls(storage[2048],storage[2048],1.00006103515625F,32);
    Muls(storage[2400],limit,4.0F,32);PipeBarrier<PIPE_V>();
    Add(storage[2048],storage[2048],storage[2400],32);PipeBarrier<PIPE_V>();
    Brcb(storage[2080],storage[2048],4,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    Compare(masks,scores,storage[2080],CMPMODE::LE,uint64_t(64),uint8_t(32),
        BinaryRepeatParams(1,1,0,1,8,1));PipeBarrier<PIPE_V>();
    auto bits=masks.ReinterpretCast<int32_t>();
    auto minus=storage.ReinterpretCast<int32_t>();
    Adds(minus,bits,int32_t(-1),64);PipeBarrier<PIPE_V>();
    And(storage[64].ReinterpretCast<uint16_t>(),bits.ReinterpretCast<uint16_t>(),minus.ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
    auto empty=storage[128].ReinterpretCast<uint8_t>();
    auto one=storage[136].ReinterpretCast<uint8_t>();
    auto signOnly=storage[144].ReinterpretCast<uint8_t>();
    CompareScalar(empty,bits,int32_t(0),CMPMODE::EQ,64);
    CompareScalar(one,storage[64].ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);
    // INT_MIN is a singleton. Exclude it independently of integer overflow mode.
    CompareScalar(signOnly,bits,int32_t(-2147483647-1),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint64_t any=~empty.ReinterpretCast<uint64_t>().GetValue(0);
    const uint64_t multiple=~one.ReinterpretCast<uint64_t>().GetValue(0)&~signOnly.ReinterpretCast<uint64_t>().GetValue(0);
    uint64_t packed=(multiple|(multiple>>1)|(any&(any>>1)))&uint64_t(0x5555555555555555ULL);
    packed=(packed|(packed>>1))&uint64_t(0x3333333333333333ULL);
    packed=(packed|(packed>>2))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
    packed=(packed|(packed>>4))&uint64_t(0x00ff00ff00ff00ffULL);
    packed=(packed|(packed>>8))&uint64_t(0x0000ffff0000ffffULL);
    packed=(packed|(packed>>16))&uint64_t(0x00000000ffffffffULL);
    uint32_t pending=uint32_t(packed);
    pending&=validRows==32?~uint32_t(0):(uint32_t(1)<<validRows)-1U;
    return pending;
}

__aicore__ __attribute__((noinline)) inline bool TreeHalfRefineNearTies(
    uint32_t scoreAddress,uint32_t pointAddress,uint32_t pointPitch,
    uint32_t sfAddress,uint32_t soAddress,uint32_t storageAddress,
    uint32_t indexAddress,uint32_t validRows,uint32_t segments,uint32_t pending) {
    const LocalTensor<int32_t> indices(TPosition::VECCALC,indexAddress,32);
    if(!pending)return false;
    TreeHalfRational64 exact(storageAddress);exact.Init();
    while(pending) {
        const uint32_t row=uint32_t(__builtin_ctz(pending));pending&=pending-1U;
        exact.MakeScores(row,pointAddress,pointPitch,sfAddress,soAddress,segments);exact.Tournament();
        Cast(exact.Plane(214).ReinterpretCast<int32_t>(),exact.Plane(96),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(1U<<(row&7U)));
        Copy<int32_t,false>(indices[(row/8)*8],exact.Plane(214).ReinterpretCast<int32_t>(),uint64_t(8),1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    }
    return true;
}
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeGeneralExactFallback(
    GM_ADDR pointAddress,GM_ADDR segmentAddress,GM_ADDR distanceAddress,GM_ADDR indexAddress,uint32_t segments);
#endif

#include "tree_pairs.h"

template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactFallback(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,
    uint32_t rows,uint32_t segmentCount);

#include "tree_tiny.h"

// Independent vector implementation for the single-point, single-segment resource.
#include "tree_single_point_single_segment.h"

#define TREE_LINE_MIN_DISTANCE_DEVICE_BODY 1
#include "tree_line_min_distance_tiling.h"
#undef TREE_LINE_MIN_DISTANCE_DEVICE_BODY

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
constexpr uint32_t SORT_CAP=4096;
__aicore__ inline void SpatialIndices(LocalTensor<int32_t> values,uint32_t count) {
    Duplicate(values,int32_t(0),8);PipeBarrier<PIPE_V>();SetMaskNorm();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
    Adds<int32_t,false>(values,values,int32_t(1),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
    Adds<int32_t,false>(values,values,int32_t(2),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
    Adds<int32_t,false>(values,values,int32_t(4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
    for(uint32_t filled=8;filled<count;filled*=2) {
        uint32_t n=count-filled<filled?count-filled:filled;
        Adds(values[filled],values,int32_t(filled),n);PipeBarrier<PIPE_V>();
    }
}
__aicore__ __attribute__((always_inline)) inline void SpatialSort(LocalTensor<float> pairs,LocalTensor<float> scratch,
    LocalTensor<float> key,LocalTensor<uint32_t> index,uint32_t count) {
    Sort32(pairs,key,index,count/32);PipeBarrier<PIPE_V>();
    bool inPairs=true;
    for(uint32_t length=32;length<count;length*=4) {
        auto src=inPairs?pairs:scratch;auto dst=inPairs?scratch:pairs;
        const uint32_t lists=count/length<4?count/length:4;
        const uint16_t sizes[4]={uint16_t(length),uint16_t(length),uint16_t(lists>2?length:0),uint16_t(lists>3?length:0)};
        for(uint32_t begin=0;begin<count;begin+=4*length) {
            MrgSortSrcList<float> sources(src[2*begin],src[2*(begin+length)],
                src[2*(begin+(lists>2?2:0)*length)],src[2*(begin+(lists>3?3:0)*length)]);
            MrgSort(dst[2*begin],sources,MrgSort4Info(sizes,false,lists==4?15:3,1));
        }
        PipeBarrier<PIPE_V>();inPairs=!inPairs;
    }
    if(!inPairs) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));Copy<float,false>(pairs,scratch,uint64_t(64),count/32,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
}
// Hilbert axes-to-transpose, using exact bounded integer vector arithmetic.
// Algorithm reference: Skilling; https://github.com/galtay/hilbertcurve .
__aicore__ __attribute__((always_inline)) inline void SpatialBitAndValue(
    const LocalTensor<int32_t>& dst,const LocalTensor<int32_t>& src,int32_t value,
    const LocalTensor<float>& constants,uint32_t count) {
    Duplicate(constants.ReinterpretCast<int32_t>(),value,64);PipeBarrier<PIPE_V>();
    SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
    And<uint16_t,false>(dst.ReinterpretCast<uint16_t>(),src.ReinterpretCast<uint16_t>(),constants.ReinterpretCast<uint16_t>(),uint64_t(128),count/64,BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
}

// The 24-state Hilbert automaton is shared by both sorting strategies.


#include "tree_hilbert_keys.h"


template<bool NORMAL_KEY_PACKETS, class StageCountType, class StageIdsType, class StageMaskType, class StageQuantizedType, class StageStateType, class StageHilbertTableType, class StageKeyIntegerType, class StageDigitType>
__aicore__ __attribute__((always_inline)) inline void TreeSpatialArrangeSpatialHilbertNormalPacketsStage(StageCountType& count, StageIdsType& ids, StageMaskType& mask, StageQuantizedType& quantized, StageStateType& state, StageHilbertTableType& hilbertTable, StageKeyIntegerType& keyInteger, StageDigitType& digit) {
    if constexpr(NORMAL_KEY_PACKETS) {
        SpatialHilbertNormalPackets(quantized,state,keyInteger,digit,ids,mask,hilbertTable,count,3);
    } else {
    Duplicate(state,uint32_t(0),count);Duplicate(keyInteger,uint32_t(0),count);PipeBarrier<PIPE_V>();
    for(uint32_t remaining=7;remaining>3;--remaining) {
        Duplicate(mask,uint32_t(0),count);PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            ShiftRight(digit,quantized[axis*SORT_CAP].template ReinterpretCast<uint32_t>(),remaining-1,count);PipeBarrier<PIPE_V>();
            SpatialBitAndValue(digit.template ReinterpretCast<int32_t>(),digit.template ReinterpretCast<int32_t>(),1,ids.template ReinterpretCast<float>(),count);
            ShiftLeft(digit,digit,2-axis,count);PipeBarrier<PIPE_V>();
            Add(mask.template ReinterpretCast<int32_t>(),mask.template ReinterpretCast<int32_t>(),digit.template ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
        }
        ShiftLeft(ids,state,uint32_t(3),count);PipeBarrier<PIPE_V>();
        Add(ids.template ReinterpretCast<int32_t>(),ids.template ReinterpretCast<int32_t>(),mask.template ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
        ShiftLeft(ids,ids,uint32_t(2),count);PipeBarrier<PIPE_V>();
        Gather(mask.template ReinterpretCast<float>(),hilbertTable,ids,uint32_t(0),count);PipeBarrier<PIPE_V>();
        ShiftRight(state,mask,uint32_t(3),count);PipeBarrier<PIPE_V>();
        SpatialBitAndValue(digit.template ReinterpretCast<int32_t>(),mask.template ReinterpretCast<int32_t>(),7,ids.template ReinterpretCast<float>(),count);
        ShiftLeft(keyInteger,keyInteger,uint32_t(3),count);PipeBarrier<PIPE_V>();
        Add(keyInteger.template ReinterpretCast<int32_t>(),keyInteger.template ReinterpretCast<int32_t>(),digit.template ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
    }
    }
}

template<bool INPUT_SOA, class StageCountType, class StageRawType, class StageTempType, class StageIdsType, class StageMaskType, class StageRecordType, class StageReducedType>
__aicore__ __attribute__((always_inline)) inline void TreeSpatialArrangeCopyStage(StageCountType& count, StageRawType& raw, StageTempType& temp, StageIdsType& ids, StageMaskType& mask, StageRecordType& record, StageReducedType& reduced) {
    for(uint32_t axis=0;axis<3;++axis) {
        if constexpr(INPUT_SOA)
            Copy(temp,raw[axis*SORT_CAP],64,count/64,CopyRepeatParams(1,1,8,8));
        else Gather(temp,raw,ids,axis*4,count);
        PipeBarrier<PIPE_V>();auto coord=temp;
        WholeReduceMin(record,coord,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float low=reduced.GetValue(0);
        WholeReduceMax(record,coord,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float high=reduced.GetValue(0);
        const float span=high-low;
        float scale=span>=9.5367431640625e-7F?127.0F/span:0.0F;
        Adds(temp,coord,-low,count);PipeBarrier<PIPE_V>();Muls(temp,temp,scale,count);PipeBarrier<PIPE_V>();
        Maxs(temp,temp,0.0F,count);PipeBarrier<PIPE_V>();Mins(temp,temp,127.0F,count);PipeBarrier<PIPE_V>();
        Cast(mask.template ReinterpretCast<int32_t>(),temp,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
        LocalTensor<float> quantized(TPosition::VECCALC,49152,3*SORT_CAP);
        Copy(quantized[axis*SORT_CAP],mask.template ReinterpretCast<float>(),64,count/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
}

template<bool INPUT_SOA, class StageValidType, class StageCountType, class StageGyType, class StageRawType, class StagePairsType, class StageScratchType, class StageKeyType, class StageTempType, class StageIdsType>
__aicore__ __attribute__((always_inline)) inline void TreeSpatialArrangeSpatialSortStage(StageValidType& valid, StageCountType& count, StageGyType& gy, StageRawType& raw, StagePairsType& pairs, StageScratchType& scratch, StageKeyType& key, StageTempType& temp, StageIdsType& ids) {
    SpatialSort(pairs,scratch,key,ids,count);
    GatherMask(ids.template ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    Muls(ids.template ReinterpretCast<int32_t>(),ids.template ReinterpretCast<int32_t>(),int32_t(INPUT_SOA?4:12),count);PipeBarrier<PIPE_V>();
    for(uint32_t axis=0;axis<3;++axis) {
        Gather(temp,raw[INPUT_SOA?axis*SORT_CAP:0],ids,INPUT_SOA?0:axis*4,count);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(gy[axis*SORT_CAP],temp,count);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    GatherMask(ids.template ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    Cast(key,ids.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();Muls(key,key,-1.0F,count);PipeBarrier<PIPE_V>();
    SpatialIndices(ids.template ReinterpretCast<int32_t>(),count);SpatialSort(pairs,scratch,key,ids,count);
    GatherMask(ids.template ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopy(gy[5*SORT_CAP],ids.template ReinterpretCast<float>(),count);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

    // The sorting arena is now dead. Reuse 144 KiB for the vector gather
    // from sorted SoA to the existing operator's contiguous AoS input format.
    PipeBarrier<PIPE_ALL>();
    for(uint32_t axis=0;axis<3;++axis)DataCopy(raw[axis*SORT_CAP],gy[axis*SORT_CAP],count);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<float> aos(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<int32_t> gatherOffsets(TPosition::VECCALC,98304,3*SORT_CAP);
    SpatialIndices(gatherOffsets,3*count);
    Cast(aos,gatherOffsets,RoundMode::CAST_NONE,3*count);PipeBarrier<PIPE_V>();
    Muls(aos,aos,0.3333333432674408F,3*count);PipeBarrier<PIPE_V>();
    Cast(aos.template ReinterpretCast<int32_t>(),aos,RoundMode::CAST_FLOOR,3*count);PipeBarrier<PIPE_V>();
    // 4*((i-3*q)*SORT_CAP+q) = (4*SORT_CAP)*i + (4-12*SORT_CAP)*q.
    Muls(gatherOffsets,gatherOffsets,int32_t(4*SORT_CAP),3*count);
    Muls(aos.template ReinterpretCast<int32_t>(),aos.template ReinterpretCast<int32_t>(),int32_t(4)-int32_t(12*SORT_CAP),3*count);PipeBarrier<PIPE_V>();
    Add(gatherOffsets,gatherOffsets,aos.template ReinterpretCast<int32_t>(),3*count);PipeBarrier<PIPE_V>();
    Gather(aos,raw,gatherOffsets.template ReinterpretCast<uint32_t>(),uint32_t(0),3*count);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gy,aos,DataCopyExtParams{1,3*valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}
template<bool NORMAL_KEY_PACKETS=false,bool INPUT_SOA=false>
__aicore__ __attribute__((always_inline)) inline void TreeSpatialArrange(
    GM_ADDR input,GM_ADDR workspace,uint32_t valid,uint32_t count) {
    GlobalTensor<float> gx,gy;gx.SetGlobalBuffer((__gm__ float*)input);gy.SetGlobalBuffer((__gm__ float*)workspace);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP);
    LocalTensor<float> pairs(TPosition::VECCALC,49152,2*SORT_CAP),scratch(TPosition::VECCALC,81920,2*SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),temp(TPosition::VECCALC,147456,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP),mask(TPosition::VECCALC,163840,SORT_CAP);
    LocalTensor<float> record(TPosition::VECCALC,180224,128),reduced(TPosition::VECCALC,180736,64);
    Duplicate(raw,0.0F,3*SORT_CAP);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if constexpr(INPUT_SOA) {
        // The global bucket already stores these exact three coordinate planes.
        // Zero padding matches the previous AoS reload; live values are bit copies.
        for(uint32_t axis=0;axis<3;++axis)
            DataCopyPad(raw[axis*SORT_CAP],gx[axis*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    } else {
        DataCopyPad(raw,gx,DataCopyExtParams{1,3*valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SpatialIndices(ids.ReinterpretCast<int32_t>(),count);
    Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
    TreeSpatialArrangeCopyStage<INPUT_SOA>(count, raw, temp, ids, mask, record, reduced);
    LocalTensor<int32_t> quantized(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<uint32_t> state(TPosition::VECCALC,98304,SORT_CAP);
    // Fixed Hilbert orientation automaton:24 states,8 input octants per state.
    // The packed output is3 key bits plus the next orientation state.
    // These stores initialize192 algorithm constants, never input elements.
    LocalTensor<uint64_t> hilbertConstants(TPosition::VECCALC,NORMAL_KEY_PACKETS?181248U:180224U,96);
    if constexpr(!NORMAL_KEY_PACKETS) {
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    InitHilbertConstants(hilbertConstants);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    auto hilbertTable=hilbertConstants.ReinterpretCast<float>();
    auto keyInteger=key.ReinterpretCast<uint32_t>(),digit=temp.ReinterpretCast<uint32_t>();
    TreeSpatialArrangeSpatialHilbertNormalPacketsStage<NORMAL_KEY_PACKETS>(count, ids, mask, quantized, state, hilbertTable, keyInteger, digit);
    Cast(key,keyInteger.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
    SpatialIndices(ids.ReinterpretCast<int32_t>(),count);
    if(valid<count) {
        Cast(temp,ids.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        CompareScalar(mask.ReinterpretCast<uint8_t>(),temp,float(int32_t(valid)),CMPMODE::LT,count);PipeBarrier<PIPE_V>();
        Select(key,mask.ReinterpretCast<uint8_t>(),key,-__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count);PipeBarrier<PIPE_V>();
    }
    TreeSpatialArrangeSpatialSortStage<INPUT_SOA>(valid, count, gy, raw, pairs, scratch, key, temp, ids);
}
__aicore__ __attribute__((always_inline)) inline void TreeSpatialRestore(
    GM_ADDR workspace,GM_ADDR distance,GM_ADDR index,uint32_t valid) {
    GlobalTensor<float> gx,gd,gi;gx.SetGlobalBuffer((__gm__ float*)workspace);
    gd.SetGlobalBuffer((__gm__ float*)distance);gi.SetGlobalBuffer((__gm__ float*)index);
    GlobalTensor<uint32_t> gxInverse;gxInverse.SetGlobalBuffer((__gm__ uint32_t*)workspace+5*SORT_CAP);
    LocalTensor<float> raw(TPosition::VECCALC,0,2*SORT_CAP),outD(TPosition::VECCALC,32768,SORT_CAP),outI(TPosition::VECCALC,49152,SORT_CAP);
    LocalTensor<uint32_t> inverse(TPosition::VECCALC,65536,SORT_CAP);
    DataCopyPad(raw,gx[3*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(raw[SORT_CAP],gx[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(inverse,gxInverse,DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<uint32_t>{false,0,0,uint32_t(0)});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    ShiftLeft(inverse,inverse,uint32_t(2),valid);PipeBarrier<PIPE_V>();
    Gather(outD,raw,inverse,uint32_t(0),valid);Gather(outI,raw[SORT_CAP],inverse,uint32_t(0),valid);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gd,outD,DataCopyExtParams{1,valid*4,0,0,0});DataCopyPad(gi,outI,DataCopyExtParams{1,valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))

// Batch-wide geometric buckets reduce the spatial diameter of a query tile.
// Admission is derived from available cores, UB capacity and measured occupancy.
constexpr uint32_t TREE_BUCKET_CAPACITY=sizeof(DTYPE_TREE_POINTS)==2?16U:8U;
constexpr uint32_t TREE_BUCKET_BOUND_BASE=188160U+4U*TREE_BUCKET_CAPACITY;
__aicore__ __attribute__((noinline)) inline void TreeBucketKeys(uint32_t count,uint32_t owners) {
    const LocalTensor<float> low(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE,3),scale(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE+64,3);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),temp(TPosition::VECCALC,147456,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP),mask(TPosition::VECCALC,163840,SORT_CAP);
    LocalTensor<float> record(TPosition::VECCALC,180224,128);
    SpatialIndices(ids.ReinterpretCast<int32_t>(),count);
    Muls(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
    LocalTensor<float> axes(TPosition::VECCALC,49152,3*SORT_CAP);
    for(uint32_t axis=0;axis<3;++axis) {
        Gather(temp,raw,ids,axis*4,count);PipeBarrier<PIPE_V>();
        Adds(temp,temp,-low.GetValue(axis),count);PipeBarrier<PIPE_V>();
        Muls(temp,temp,scale.GetValue(axis),count);PipeBarrier<PIPE_V>();
        Maxs(temp,temp,0.0F,count);PipeBarrier<PIPE_V>();Mins(temp,temp,127.0F,count);PipeBarrier<PIPE_V>();
        Cast(axes[axis*SORT_CAP].ReinterpretCast<int32_t>(),temp,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
    }
    LocalTensor<int32_t> quantized(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<uint32_t> state(TPosition::VECCALC,98304,SORT_CAP);
    // Fixed Hilbert orientation automaton:24 states,8 input octants per state.
    // The packed output is3 key bits plus the next orientation state.
    // These stores initialize192 algorithm constants, never input elements.
    LocalTensor<uint64_t> hilbertConstants(TPosition::VECCALC,181248,96);
    {
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    InitHilbertConstants(hilbertConstants);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    auto hilbertTable=hilbertConstants.ReinterpretCast<float>();
    auto keyInteger=key.ReinterpretCast<uint32_t>(),digit=temp.ReinterpretCast<uint32_t>();
    const uint32_t lowest=((owners & (owners-1))==0 && (sizeof(DTYPE_TREE_POINTS)==4 || owners<=8))?6:4;
    SpatialHilbertNormalPackets(quantized,state,keyInteger,digit,ids,mask,hilbertTable,count,lowest);
    Cast(key,keyInteger.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
__aicore__ __attribute__((always_inline)) inline void TreeBucketSoAToAoS(
    GM_ADDR workspace,uint32_t valid,uint32_t count) {
    GlobalTensor<float> gy;gy.SetGlobalBuffer((__gm__ float*)workspace);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP),aos(TPosition::VECCALC,49152,3*SORT_CAP);
    LocalTensor<int32_t> offsets(TPosition::VECCALC,98304,3*SORT_CAP);
    PipeBarrier<PIPE_ALL>();
    for(uint32_t axis=0;axis<3;++axis)DataCopy(raw[axis*SORT_CAP],gy[axis*SORT_CAP],count);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SpatialIndices(offsets,3*count);Cast(aos,offsets,RoundMode::CAST_NONE,3*count);PipeBarrier<PIPE_V>();
    Muls(aos,aos,0.3333333432674408F,3*count);PipeBarrier<PIPE_V>();
    Cast(aos.ReinterpretCast<int32_t>(),aos,RoundMode::CAST_FLOOR,3*count);PipeBarrier<PIPE_V>();
    Muls(offsets,offsets,int32_t(4*SORT_CAP),3*count);
    Muls(aos.ReinterpretCast<int32_t>(),aos.ReinterpretCast<int32_t>(),int32_t(4)-int32_t(12*SORT_CAP),3*count);PipeBarrier<PIPE_V>();
    Add(offsets,offsets,aos.ReinterpretCast<int32_t>(),3*count);PipeBarrier<PIPE_V>();
    Gather(aos,raw,offsets.ReinterpretCast<uint32_t>(),uint32_t(0),3*count);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gy,aos,DataCopyExtParams{1,3*valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}

#endif

__aicore__ __attribute__((always_inline)) inline void TreeBucketCopy(
    const LocalTensor<float>& destination,const LocalTensor<float>& input,uint32_t count) {
    const uint32_t full=count/64,tail=count%64;
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    if(full)Copy<float,false>(destination,input,uint64_t(64),full,CopyRepeatParams(1,1,8,8));
    if(tail) {
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(tail));
        Copy<float,false>(destination[full*64],input[full*64],uint64_t(tail),1,CopyRepeatParams(1,1,8,8));
    }
    PipeBarrier<PIPE_V>();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}

__aicore__ __attribute__((always_inline)) inline void TreeBucketRestore(
    GM_ADDR workspace,uint32_t valid) {
    GlobalTensor<float> gx;gx.SetGlobalBuffer((__gm__ float*)workspace);
    LocalTensor<float> raw(TPosition::VECCALC,0,2*SORT_CAP),outD(TPosition::VECCALC,32768,SORT_CAP),outI(TPosition::VECCALC,49152,SORT_CAP);
    LocalTensor<uint32_t> inverse(TPosition::VECCALC,65536,SORT_CAP);
    LocalTensor<float> constant(TPosition::VECCALC,81920,64);
    DataCopyPad(raw,gx[3*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(raw[SORT_CAP],gx[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    DataCopyPad(inverse.ReinterpretCast<float>(),gx[5*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SpatialBitAndValue(inverse.ReinterpretCast<int32_t>(),inverse.ReinterpretCast<int32_t>(),4095,constant,(valid+63)/64*64);
    ShiftLeft(inverse,inverse,uint32_t(2),valid);PipeBarrier<PIPE_V>();
    Gather(outD,raw,inverse,uint32_t(0),valid);Gather(outI,raw[SORT_CAP],inverse,uint32_t(0),valid);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gx[3*SORT_CAP],outD,DataCopyExtParams{1,valid*4,0,0,0});
    DataCopyPad(gx[4*SORT_CAP],outI,DataCopyExtParams{1,valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
}

template<class ControlArgument0, class ControlArgument1, class ControlArgument2, class ControlArgument3, class ControlArgument4, class ControlArgument5, class ControlArgument6, class ControlArgument7, class ControlArgument8, class ControlArgument9, class ControlArgument10>
__aicore__ __attribute__((always_inline)) inline void TreeBucketWriteOriginalOrderControlStage31(ControlArgument0& live, ControlArgument1& assembledD, ControlArgument2& position, ControlArgument3& packed, ControlArgument4& sourceI, ControlArgument5& predicate, ControlArgument6& gatherExtent, ControlArgument7& kept, ControlArgument8& assembledI, ControlArgument9& originalId, ControlArgument10& assembledIds) {
if(live) {
            TreeBucketCopy(assembledD[position],packed,live);
            GatherMask(packed,sourceI,predicate.template ReinterpretCast<uint32_t>(),true,gatherExtent,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            TreeBucketCopy(assembledI[position],packed,live);
            GatherMask(packed,originalId,predicate.template ReinterpretCast<uint32_t>(),true,gatherExtent,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            TreeBucketCopy(assembledIds[position],packed,live);position+=(live+7)/8*8;
        }
}

__aicore__ __attribute__((always_inline)) inline void TreeBucketWriteOriginalOrder(
    GM_ADDR workspace,GM_ADDR distance,GM_ADDR index,const LocalTensor<uint32_t>& bucketCount,
    uint32_t batch,uint32_t owners,uint32_t owner,uint32_t points) {
    constexpr uint32_t stride=6*SORT_CAP+14*1056+1024;
    const uint32_t begin=points*owner/owners,end=points*(owner+1)/owners;
    LocalTensor<float> assembledD(TPosition::VECCALC,0,SORT_CAP),assembledI(TPosition::VECCALC,16384,SORT_CAP),assembledIds(TPosition::VECCALC,32768,SORT_CAP);
    LocalTensor<float> sourceD(TPosition::VECCALC,49152,SORT_CAP),sourceI(TPosition::VECCALC,65536,SORT_CAP);
    LocalTensor<uint32_t> encoded(TPosition::VECCALC,81920,SORT_CAP),ordinal(TPosition::VECCALC,131072,SORT_CAP);
    LocalTensor<float> originalId(TPosition::VECCALC,98304,SORT_CAP),packed(TPosition::VECCALC,114688,SORT_CAP);
    LocalTensor<uint8_t> predicate(TPosition::VECCALC,147456,512),upper(TPosition::VECCALC,147968,512),validMask(TPosition::VECCALC,148480,512);
    Duplicate(assembledIds,__builtin_inff(),SORT_CAP);PipeBarrier<PIPE_V>();uint32_t position=0;
    for(uint32_t other=0;other<owners;++other) {
        const uint32_t valid=bucketCount.GetValue(other);uint32_t count=(valid+63)/64*64;
        GlobalTensor<float> gs;gs.SetGlobalBuffer((__gm__ float*)(workspace+(batch*owners+other)*stride*4));
        DataCopyPad(sourceD,gs[3*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(sourceI,gs[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(encoded.ReinterpretCast<float>(),gs[5*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ShiftRight(encoded,encoded,uint32_t(12),count);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        Cast(originalId,encoded.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        const uint32_t gatherExtent=valid;
#else
        Cast(originalId,encoded.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);SpatialIndices(ordinal.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
        Cast(packed,ordinal.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        CompareScalar(validMask,packed,float(int32_t(valid)),CMPMODE::LT,count);
        const uint32_t gatherExtent=count;
#endif
        CompareScalar(predicate,originalId,float(int32_t(begin)),CMPMODE::GE,count);
        CompareScalar(upper,originalId,float(int32_t(end)),CMPMODE::LT,count);PipeBarrier<PIPE_V>();
        And(predicate.ReinterpretCast<uint16_t>(),predicate.ReinterpretCast<uint16_t>(),upper.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
#if !defined(DTYPE_TREE_POINTS) || !TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        And(predicate.ReinterpretCast<uint16_t>(),predicate.ReinterpretCast<uint16_t>(),validMask.ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
#endif
        uint64_t kept=0;GatherMask(packed,sourceD,predicate.ReinterpretCast<uint32_t>(),true,gatherExtent,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);uint32_t live=uint32_t(kept);
        TreeBucketWriteOriginalOrderControlStage31(live, assembledD, position, packed, sourceI, predicate, gatherExtent, kept, assembledI, originalId, assembledIds);
    }
    LocalTensor<float> pairs(TPosition::VECCALC,49152,2*SORT_CAP),scratch(TPosition::VECCALC,81920,2*SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),outD(TPosition::VECCALC,147456,SORT_CAP),outI(TPosition::VECCALC,163840,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP);uint32_t count=64;while(count<position)count*=2;
    Muls(key,assembledIds,-1.0F,count);SpatialIndices(ids.ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
    SpatialSort(pairs,scratch,key,ids,count);GatherMask(ids.ReinterpretCast<float>(),pairs,uint8_t(2),GatherMaskParams(1,count/32,8,0));PipeBarrier<PIPE_V>();
    ShiftLeft(ids,ids,uint32_t(2),count);PipeBarrier<PIPE_V>();
    Gather(outD,assembledD,ids,uint32_t(0),end-begin);Gather(outI,assembledI,ids,uint32_t(0),end-begin);PipeBarrier<PIPE_V>();
    GlobalTensor<float> gd,gi;gd.SetGlobalBuffer((__gm__ float*)distance+batch*points+begin);gi.SetGlobalBuffer((__gm__ float*)index+batch*points+begin);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gd,outD,DataCopyExtParams{1,(end-begin)*4,0,0,0});DataCopyPad(gi,outI,DataCopyExtParams{1,(end-begin)*4,0,0,0});PipeBarrier<PIPE_ALL>();
}


template<auto stride, class StageWorkspaceType, class StageTType, class StageOwnersType, class StageBatchType, class StageOwnerType, class StageInputType, class StageStoreType, class StageRawType, class StageAxesType, class StageOriginalIdType, class StageKeyType, class StageTempType, class StageIdsType, class StageRecordType, class StageBucketCountType, class StagePieceCountsType, class StageAllPointCoordinatesSafeType, class StageProducerMaxType, class StagePreviousStartType>
__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcessSetGlobalBufferStage(StageWorkspaceType& workspace, StageTType& t, StageOwnersType& owners, StageBatchType& batch, StageOwnerType& owner, StageInputType& input, StageStoreType& store, StageRawType& raw, StageAxesType& axes, StageOriginalIdType& originalId, StageKeyType& key, StageTempType& temp, StageIdsType& ids, StageRecordType& record, StageBucketCountType& bucketCount, StagePieceCountsType& pieceCounts, StageAllPointCoordinatesSafeType& allPointCoordinatesSafe, StageProducerMaxType& producerMax, StagePreviousStartType& previousStart) {
    for(uint32_t piece=0;piece<owners;++piece) {
        const uint32_t start=t.points*piece/owners;
        const uint32_t valid=t.points*(piece+1)/owners-start;
        const uint32_t count=(valid+63U)&~63U;
        GlobalTensor<float> cachedKeys;
        cachedKeys.SetGlobalBuffer((__gm__ float*)(workspace+(batch*owners+piece)*stride*4)+6*SORT_CAP);
        Duplicate(raw,0.0F,3*SORT_CAP);Duplicate(key,0.0F,count);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(raw,input[3*start],DataCopyExtParams{1,3*valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(key,cachedKeys,DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopy(pieceCounts.template ReinterpretCast<float>(),cachedKeys[SORT_CAP],TREE_BUCKET_CAPACITY);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
#if !defined(DTYPE_TREE_POINTS) || !TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        SpatialIndices(ids.template ReinterpretCast<int32_t>(),count);PipeBarrier<PIPE_V>();
        Cast(temp,ids.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
#endif
        // Every payload GatherMask uses counter-mode valid; padded
        // lanes need no explicit validity Compare/Select.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // All IDs stay below16*4096+4096, so every integer increment is exact in FP32.
        if(start!=previousStart) {
            Adds(originalId,originalId,float(int32_t(start-previousStart)),producerMax);PipeBarrier<PIPE_V>();
            previousStart=start;
        }
#else
        Adds(originalId,temp,float(int32_t(start)),count);Muls(ids.template ReinterpretCast<int32_t>(),ids.template ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
    for(uint32_t axis=0;axis<3;++axis)Gather(axes[axis*SORT_CAP],raw,ids,axis*4,count);PipeBarrier<PIPE_V>();
        {
            CompareScalar(record.template ReinterpretCast<uint8_t>(),key,float(int32_t(owner)),CMPMODE::EQ,count);PipeBarrier<PIPE_V>();
            uint64_t kept=0;
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_15
            allPointCoordinatesSafe=allPointCoordinatesSafe && ((pieceCounts.GetValue(0)&0x80000000U)!=0);
            uint32_t live=pieceCounts.GetValue(owner)&0x7fffffffU,prefix=bucketCount.GetValue(owner);
#else // TREE_HALF_COORD_CERT_15
            uint32_t live=pieceCounts.GetValue(owner),prefix=bucketCount.GetValue(owner);
#endif // TREE_HALF_COORD_CERT_15
            for(uint32_t bucket=0;bucket<owners;++bucket)
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_17
                bucketCount.SetValue(bucket,bucketCount.GetValue(bucket)+(pieceCounts.GetValue(bucket)&0x7fffffffU));
#else // TREE_HALF_COORD_CERT_17
                bucketCount.SetValue(bucket,bucketCount.GetValue(bucket)+pieceCounts.GetValue(bucket));
#endif // TREE_HALF_COORD_CERT_17
            if(live && prefix+live<=SORT_CAP) {
                // All coordinate gathers consumed raw before this point.
                // Its three disjoint planes can hold the compacted payload.
        #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
    for(uint32_t axis=0;axis<3;++axis) {
                    GatherMask(raw[axis*SORT_CAP],axes[axis*SORT_CAP],record.template ReinterpretCast<uint32_t>(),true,valid,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
                }
                GatherMask(temp,originalId,record.template ReinterpretCast<uint32_t>(),true,valid,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
    for(uint32_t axis=0;axis<3;++axis)
                    DataCopyPad(store[axis*SORT_CAP+prefix],raw[axis*SORT_CAP],DataCopyExtParams{1,live*4,0,0,0});
                DataCopyPad(store[4*SORT_CAP+prefix],temp,DataCopyExtParams{1,live*4,0,0,0});
                // Drain before the next producer clears raw or reuses temp.
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }
}

template<class StageOwnersType, class StageStoreType, class StageRawType, class StageKeyType, class StageTempType, class StageIdsType, class StageRecordType, class StageBucketCountType, class StageBoundBeginType, class StageBoundEndType, class StageLocalPointCoordinatesSafeType, class StageLocalKeySafeType>
__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcessTreeBucketKeysStage(StageOwnersType& owners, StageStoreType& store, StageRawType& raw, StageKeyType& key, StageTempType& temp, StageIdsType& ids, StageRecordType& record, StageBucketCountType& bucketCount, StageBoundBeginType& boundBegin, StageBoundEndType& boundEnd, StageLocalPointCoordinatesSafeType& localPointCoordinatesSafe, StageLocalKeySafeType& localKeySafe) {
    {
        const uint32_t valid=boundEnd-boundBegin;
        const uint32_t count=(valid+63U)&~63U;
        if(localKeySafe) TreeBucketKeys(count,owners);
        else {Duplicate(key,-1.0F,count);PipeBarrier<PIPE_V>();}
        // Compute this producer's counts once. The same multiplication and
        // FLOOR cast define bucket membership in all original consumers.
        Muls(temp,key,float(int32_t(owners))*(((owners & (owners-1))==0 && (sizeof(DTYPE_TREE_POINTS)==4 || owners<=8))?0.125F:0.001953125F),count);PipeBarrier<PIPE_V>();
        Cast(ids.template ReinterpretCast<int32_t>(),temp,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
        Cast(temp,ids.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        // Publish the canonical bucket ID once, using the original FP32
        // multiplication and FLOOR cast. Consumers need no reconstruction.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(store[6*SORT_CAP],temp,DataCopyExtParams{1,valid*4,0,0,0});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        for(uint32_t bucket=0;bucket<owners;++bucket) {
            CompareScalar(record.template ReinterpretCast<uint8_t>(),temp,float(int32_t(bucket)),CMPMODE::EQ,count);PipeBarrier<PIPE_V>();
            // Counter mode excludes the padded tail directly.
            uint64_t kept=0;GatherMask(raw,temp,record.template ReinterpretCast<uint32_t>(),true,valid,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            bucketCount.SetValue(bucket,uint32_t(kept));
        }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_9
        // Counts are at most SORT_CAP. Bit31 of the first word carries a
        // conservative certificate without changing the metadata transfer size.
        bucketCount.SetValue(0,bucketCount.GetValue(0)|((localPointCoordinatesSafe && localKeySafe)?0x80000000U:0U));
#else // TREE_HALF_COORD_CERT_9
#endif // TREE_HALF_COORD_CERT_9
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_11
        // The unchanged second global barrier precedes coefficient-cache reuse.
#else // TREE_HALF_COORD_CERT_11
        // This temporary count slot follows the complete key region. The
        // unchanged second global barrier precedes coefficient-cache reuse.
#endif // TREE_HALF_COORD_CERT_11
        DataCopy(store[7*SORT_CAP],bucketCount.template ReinterpretCast<float>(),TREE_BUCKET_CAPACITY);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for(uint32_t bucket=0;bucket<TREE_BUCKET_CAPACITY;++bucket)bucketCount.SetValue(bucket,uint32_t(0));
    }
}

template<class StageInputType, class StageRawType, class StageKeyType, class StageTempType, class StageIdsType, class StageRecordType, class StageReducedType, class StageLowType, class StageHighType, class StageBoundBeginType, class StageBoundEndType, class StageLocalPointCoordinatesSafeType>
__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcessDuplicateStage(StageInputType& input, StageRawType& raw, StageKeyType& key, StageTempType& temp, StageIdsType& ids, StageRecordType& record, StageReducedType& reduced, StageLowType& low, StageHighType& high, StageBoundBeginType& boundBegin, StageBoundEndType& boundEnd, StageLocalPointCoordinatesSafeType& localPointCoordinatesSafe) {
    {
        const uint32_t start=boundBegin;
        uint32_t valid=boundEnd-start;uint32_t count=64;while(count<valid)count*=2;
        Duplicate(raw,0.0F,3*SORT_CAP);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(raw,input[3*start],DataCopyExtParams{1,3*valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SpatialIndices(ids.template ReinterpretCast<int32_t>(),count);Muls(ids.template ReinterpretCast<int32_t>(),ids.template ReinterpretCast<int32_t>(),int32_t(12),count);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis) {
            Gather(temp,raw,ids,axis*4,count);PipeBarrier<PIPE_V>();
            WholeReduceMin(record,temp,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float lo=reduced.GetValue(0);if(lo<low.GetValue(axis))low.SetValue(axis,lo);
            WholeReduceMax(record,temp,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float hi=reduced.GetValue(0);if(hi>high.GetValue(axis))high.SetValue(axis,hi);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_7
            // Finite inputs are required by the operator contract. Existing
            // axis bounds certify the upper magnitude; zero is admissible.
            auto nonzero=key.template ReinterpretCast<uint8_t>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            // Test original bits before Abs: FTZ must not turn a nonzero
            // Float subnormal into an admissible mathematical zero. Negative
            // zero conservatively rejects the certificate and uses fallback.
            CompareScalar(nonzero,temp.template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,count);PipeBarrier<PIPE_V>();
            Not(nonzero.template ReinterpretCast<uint16_t>(),nonzero.template ReinterpretCast<uint16_t>(),count/16);PipeBarrier<PIPE_V>();
            Abs(temp,temp,count);PipeBarrier<PIPE_V>();
#else
            Abs(temp,temp,count);PipeBarrier<PIPE_V>();
            CompareScalar(nonzero,temp,0.0F,CMPMODE::NE,count);PipeBarrier<PIPE_V>();
#endif
            Select(temp,nonzero,temp,9.5367431640625e-7F,SELMODE::VSEL_TENSOR_SCALAR_MODE,count);PipeBarrier<PIPE_V>();
            WholeReduceMin(record,temp,64,count/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced,record,count/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if(!(reduced.GetValue(0)>=9.5367431640625e-7F))localPointCoordinatesSafe=false;
#else // TREE_HALF_COORD_CERT_7
#endif // TREE_HALF_COORD_CERT_7
        }
    }
}

template<auto stride, class StageWorkspaceType, class StageTType, class StageOwnersType, class StageBatchType, class StageOwnerType, class StageInputType, class StageStoreType, class StageRawType, class StageAxesType, class StageOriginalIdType, class StageKeyType, class StageTempType, class StageIdsType, class StageRecordType, class StageBucketCountType, class StagePieceCountsType, class StageEligibleType, class StageAllPointCoordinatesSafeType>
__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcessDefinedStage(StageWorkspaceType& workspace, StageTType& t, StageOwnersType& owners, StageBatchType& batch, StageOwnerType& owner, StageInputType& input, StageStoreType& store, StageRawType& raw, StageAxesType& axes, StageOriginalIdType& originalId, StageKeyType& key, StageTempType& temp, StageIdsType& ids, StageRecordType& record, StageBucketCountType& bucketCount, StagePieceCountsType& pieceCounts, StageEligibleType& eligible, StageAllPointCoordinatesSafeType& allPointCoordinatesSafe) {
    if(eligible) {
        // Traverse the original point order, reading each producer interval.
        // Grouping boundaries change; stable per-bucket point order does not.
        // Every producer uses a prefix of these unchanged gather offsets.
        const uint32_t producerMax=((t.points+owners-1U)/owners+63U)&~63U;
        uint32_t previousStart=0U;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        SpatialIndices(ids.template ReinterpretCast<int32_t>(),producerMax);PipeBarrier<PIPE_V>();
        Cast(originalId,ids.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,producerMax);PipeBarrier<PIPE_V>();
        Muls(ids.template ReinterpretCast<int32_t>(),ids.template ReinterpretCast<int32_t>(),int32_t(12),producerMax);PipeBarrier<PIPE_V>();
#endif
        TreeGlobalSpatialProcessSetGlobalBufferStage<stride>(workspace, t, owners, batch, owner, input, store, raw, axes, originalId, key, temp, ids, record, bucketCount, pieceCounts, allPointCoordinatesSafe, producerMax, previousStart);
        uint32_t sum=0;
        for(uint32_t bucket=0;bucket<owners;++bucket) {uint32_t live=bucketCount.GetValue(bucket);sum+=live;if(live>SORT_CAP || live<t.points/(2*owners))eligible=false;}
        if(sum!=t.points)eligible=false;
    }
}

template<class StagePType, class StageSType, class StageDType, class StageIType, class StageTType, class StageBatchType, class StageOwnerType, class StageStoreType, class StageLocalType, class StageBucketCountType, class StageEligibleType, class StageAllPointCoordinatesSafeType, class StageOpType>
__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcessGetValueStage(StagePType& p, StageSType& s, StageDType& d, StageIType& i, StageTType& t, StageBatchType& batch, StageOwnerType& owner, StageStoreType& store, StageLocalType& local, StageBucketCountType& bucketCount, StageEligibleType& eligible, StageAllPointCoordinatesSafeType& allPointCoordinatesSafe, StageOpType& op) {
    if(eligible) {
        uint32_t valid=bucketCount.GetValue(owner),count=64;while(count<valid)count*=2;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // Consume the existing coordinate planes directly; preserve sorting and fallback.
        TreeSpatialArrange<true,true>(local,local,valid,count);
#else
        TreeSpatialArrange<true,true>(local,local,valid,count);
#endif
        LocalTensor<float> originals(TPosition::VECCALC,0,SORT_CAP),inverse(TPosition::VECCALC,16384,SORT_CAP);
        DataCopyPad(originals,store[4*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(inverse,store[5*SORT_CAP],DataCopyExtParams{1,valid*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(originals.template ReinterpretCast<int32_t>(),originals,RoundMode::CAST_FLOOR,valid);PipeBarrier<PIPE_V>();
        ShiftLeft(originals.template ReinterpretCast<uint32_t>(),originals.template ReinterpretCast<uint32_t>(),uint32_t(12),valid);PipeBarrier<PIPE_V>();
        Add(inverse.template ReinterpretCast<int32_t>(),inverse.template ReinterpretCast<int32_t>(),originals.template ReinterpretCast<int32_t>(),valid);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);DataCopyPad(store[5*SORT_CAP],inverse,DataCopyExtParams{1,valid*4,0,0,0});PipeBarrier<PIPE_ALL>();
        auto part=t;part.batches=1;part.points=valid;part.blocks=1;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_19
        op.Init(local,s+batch*t.segments*24,local+3*SORT_CAP*4,local+4*SORT_CAP*4,part);op.template Process<true>(0,valid,local,allPointCoordinatesSafe);
#else // TREE_HALF_COORD_CERT_19
        op.Init(local,s+batch*t.segments*24,local+3*SORT_CAP*4,local+4*SORT_CAP*4,part);op.template Process<true>(0,valid,local);
#endif // TREE_HALF_COORD_CERT_19
        TreeBucketRestore(local,valid);
    } else {op.Init(p,s,d,i,t);op.template Process<true>();}
}
template<class ControlArgument0, class ControlArgument1, class ControlArgument2, class ControlArgument3>
__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcessControlStage32(ControlArgument0& low, ControlArgument1& high, ControlArgument2& localKeySafe, ControlArgument3& scale) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif

for(uint32_t axis=0;axis<3;++axis) {
        float lo=low.GetValue(axis),hi=high.GetValue(axis);
        if(!(lo>=-1099511627776.0F && hi<=1099511627776.0F))localKeySafe=false;
        float span=hi-lo;scale.SetValue(axis,span>=9.5367431640625e-7F?127.0F/span:0.0F);
    }
}

__aicore__ __attribute__((always_inline)) inline void TreeGlobalSpatialProcess(
    GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,const TreeLineMinDistanceTilingData& t) {
    constexpr uint32_t stride=6*SORT_CAP+14*1056+1024;
    const uint32_t owners=t.blocks/t.batches,batch=GetBlockIdx()/owners,owner=GetBlockIdx()%owners;
    GlobalTensor<float> input,store;input.SetGlobalBuffer((__gm__ float*)p+batch*t.points*3);
    GM_ADDR local=workspace+GetBlockIdx()*stride*4;store.SetGlobalBuffer((__gm__ float*)local);
    LocalTensor<float> raw(TPosition::VECCALC,0,3*SORT_CAP),axes(TPosition::VECCALC,49152,3*SORT_CAP),originalId(TPosition::VECCALC,98304,SORT_CAP);
    LocalTensor<float> key(TPosition::VECCALC,114688,SORT_CAP),temp(TPosition::VECCALC,147456,SORT_CAP);
    LocalTensor<uint32_t> ids(TPosition::VECCALC,131072,SORT_CAP);
    LocalTensor<float> record(TPosition::VECCALC,180224,128),reduced(TPosition::VECCALC,180736,64);
    // Native R64 search uses UB bytes[0,188128); retain bounded metadata
    // in the aligned padding below184 KiB, outside all payload arenas.
    LocalTensor<uint32_t> bucketCount(TPosition::VECCALC,188160,TREE_BUCKET_CAPACITY);
    // Bounds are dead after key generation. Reuse their reserved UB padding
    // for one producer's bounded counts, outside the search payload arena.
    LocalTensor<uint32_t> pieceCounts(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE,TREE_BUCKET_CAPACITY);
    LocalTensor<float> low(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE,3),high(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE+32,3),scale(TPosition::VECCALC,TREE_BUCKET_BOUND_BASE+64,3);
    PipeBarrier<PIPE_ALL>();
    #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#pragma unroll 1
#endif
        for(uint32_t axis=0;axis<3;++axis){low.SetValue(axis,__builtin_inff());high.SetValue(axis,-__builtin_inff());}
    for(uint32_t bucket=0;bucket<TREE_BUCKET_CAPACITY;++bucket)bucketCount.SetValue(bucket,uint32_t(0));
    // Quantization is only a spatial ordering heuristic. Each key producer
    // may normalize its own disjoint interval; all consumers read that key.
    const uint32_t boundBegin=t.points*owner/owners;
    const uint32_t boundEnd=t.points*(owner+1)/owners;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_5
    bool localPointCoordinatesSafe=true;
#else // TREE_HALF_COORD_CERT_5
#endif // TREE_HALF_COORD_CERT_5
      TreeGlobalSpatialProcessDuplicateStage(input, raw, key, temp, ids, record, reduced, low, high, boundBegin, boundEnd, localPointCoordinatesSafe);
    bool localKeySafe=true;
        TreeGlobalSpatialProcessControlStage32(low, high, localKeySafe, scale);
    bool eligible=true;
    // Unsafe producers publish a negative key for each real query. After
    // the existing key barrier every consumer observes the same missing
    // bucket membership, so the original total-count check rejects globally.
      TreeGlobalSpatialProcessTreeBucketKeysStage(owners, store, raw, key, temp, ids, record, bucketCount, boundBegin, boundEnd, localPointCoordinatesSafe, localKeySafe);
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    SetMaskNorm();ResetMask();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_13
    bool allPointCoordinatesSafe=true;
#else // TREE_HALF_COORD_CERT_13
#endif // TREE_HALF_COORD_CERT_13
    TreeGlobalSpatialProcessDefinedStage<stride>(workspace, t, owners, batch, owner, input, store, raw, axes, originalId, key, temp, ids, record, bucketCount, pieceCounts, eligible, allPointCoordinatesSafe);
    // Complete every key read before any owner reuses its coefficient cache.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    AscendC::ICachePreLoad(int64_t(6));
#endif
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    SetMaskNorm();ResetMask();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    KernelTreeBatch<float,64,1024,1024,false,false> op;
    TreeGlobalSpatialProcessGetValueStage(p, s, d, i, t, batch, owner, store, local, bucketCount, eligible, allPointCoordinatesSafe, op);
    SyncAll<true>();SetMaskNorm();ResetMask();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    if(eligible)TreeBucketWriteOriginalOrder(workspace,d,i,bucketCount,batch,owners,owner,t.points);
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
__aicore__ __attribute__((noinline,aligned(128))) inline void TreeGlobalSpatialCompactProcess(
    GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,GM_ADDR tiling) {
    // Launch-owned metadata remains immutable in GM until the kernel ends.
    const auto words=(__gm__ uint32_t*)tiling;
    TreeLineMinDistanceTilingData t;
    t.batches=words[0];t.points=words[1];t.segments=words[2];t.paddedSegments=words[3];
    t.pointTile=words[4];t.blocks=words[5];t.schedule=words[6];t.segmentLog=words[7];
    TreeGlobalSpatialProcess(p,s,d,i,workspace,t);
}
#endif

__aicore__ __attribute__((always_inline)) inline void TreeSpatialProcess(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,
    GM_ADDR workspace,const TreeLineMinDistanceTilingData& t) {
    KernelTreeBatch<float,64,1024,1024,false,false> op;
    op.Init(p,s,d,i,t);
    const uint32_t total=t.batches*t.points,begin=total*GetBlockIdx()/t.blocks,end=total*(GetBlockIdx()+1)/t.blocks;
    if(!op.SpatialSortPointCoordinatesSafe(begin,end)) {op.Process();}
    else {
        constexpr uint32_t stride=6*SORT_CAP+14*1056+1024;
        GM_ADDR local=workspace+GetBlockIdx()*stride*4;
        for(uint32_t position=begin;position<end;) {
            uint32_t batch=position/t.points,start=position%t.points;
            uint32_t valid=end-position;if(valid>t.points-start)valid=t.points-start;if(valid>SORT_CAP)valid=SORT_CAP;
            uint32_t count=64;while(count<valid)count*=2;
            TreeSpatialArrange(p+position*12,local,valid,count);
            auto part=t;part.batches=1;part.points=valid;part.blocks=1;
            op.Init(local,s+batch*t.segments*24,local+3*SORT_CAP*4,local+4*SORT_CAP*4,part);
            op.Process(0,valid,local);
            TreeSpatialRestore(local,d+position*4,i+position*4,valid);
            position+=valid;
        }
    }
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
// Promotion is exact for every binary16 value; only the final distance narrows.
// The intermediate GM planes are private to the owner and always computed here.
__aicore__ __attribute__((always_inline)) inline void PromoteSpatialHalf(
    GM_ADDR input,GM_ADDR output,uint32_t count) {
    GlobalTensor<half> source;source.SetGlobalBuffer((__gm__ half*)input);
    GlobalTensor<float> destination;destination.SetGlobalBuffer((__gm__ float*)output);
    LocalTensor<half> raw(TPosition::VECCALC,32768U,4096U);
    LocalTensor<float> promoted(TPosition::VECCALC,0U,4096U);
    SetMaskNorm();ResetMask();
    for(uint32_t start=0;start<count;start+=4096U) {
        const uint32_t live=count-start<4096U?count-start:4096U;
        const uint32_t aligned=(live+63U)/64U*64U;
        Duplicate(raw,half(0),aligned);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(raw,source[start],DataCopyExtParams{1,live*2U,0,0,0},DataCopyPadExtParams<half>{false,0,0,half(0)});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(promoted,raw,RoundMode::CAST_NONE,aligned);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(destination[start],promoted,DataCopyExtParams{1,live*4U,0,0,0});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}
__aicore__ __attribute__((always_inline)) inline void WriteSpatialHalf(
    GM_ADDR sourceDistance,GM_ADDR sourceIndex,GM_ADDR outputDistance,GM_ADDR outputIndex,
    uint32_t count,bool copyIndex) {
    GlobalTensor<float> source;source.SetGlobalBuffer((__gm__ float*)sourceDistance);
    GlobalTensor<half> destination;destination.SetGlobalBuffer((__gm__ half*)outputDistance);
    GlobalTensor<int32_t> indices,indexOut;
    indices.SetGlobalBuffer((__gm__ int32_t*)sourceIndex);indexOut.SetGlobalBuffer((__gm__ int32_t*)outputIndex);
    LocalTensor<float> raw(TPosition::VECCALC,0U,4096U);
    LocalTensor<half> rounded(TPosition::VECCALC,32768U,4096U);
    LocalTensor<int32_t> indexBuffer(TPosition::VECCALC,65536U,4096U);
    SetMaskNorm();ResetMask();
    for(uint32_t start=0;start<count;start+=4096U) {
        const uint32_t live=count-start<4096U?count-start:4096U;
        DataCopyPad(raw,source[start],DataCopyExtParams{1,live*4U,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        if(copyIndex)DataCopyPad(indexBuffer,indices[start],DataCopyExtParams{1,live*4U,0,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(rounded,raw,RoundMode::CAST_RINT,live);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(destination[start],rounded,DataCopyExtParams{1,live*2U,0,0,0});
        if(copyIndex)DataCopyPad(indexOut[start],indexBuffer,DataCopyExtParams{1,live*4U,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ __attribute__((noinline)) inline void TreeHalfGlobalSpatialProcess(
    GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,GM_ADDR workspace,const TreeLineMinDistanceTilingData& t) {
    constexpr uint32_t spatialWords=6*SORT_CAP+14*1056+1024;
    const uint32_t total=t.batches*t.points;
    const uint32_t owners=t.blocks/t.batches,batch=GetBlockIdx()/owners,owner=GetBlockIdx()%owners;
    const uint32_t begin=total*GetBlockIdx()/t.blocks,end=total*(GetBlockIdx()+1)/t.blocks;
    GM_ADDR promotedPoints=workspace+t.blocks*spatialWords*4U;
    GM_ADDR promotedSegments=promotedPoints+total*12U;
    GM_ADDR promotedDistances=promotedSegments+t.batches*t.segments*24U;
    PromoteSpatialHalf(p+begin*6U,promotedPoints+begin*12U,(end-begin)*3U);
    if(owner==0)PromoteSpatialHalf(s+batch*t.segments*12U,
        promotedSegments+batch*t.segments*24U,t.segments*6U);
    PipeBarrier<PIPE_ALL>();SyncAll<true>();SetMaskNorm();ResetMask();
    TreeGlobalSpatialProcess(promotedPoints,promotedSegments,promotedDistances,i,workspace,t);
    // Global restore writes this owner's original interval before it returns.
    WriteSpatialHalf(promotedDistances+begin*4U,i+begin*4U,d+begin*2U,i+begin*4U,end-begin,false);
}
#endif

#include "tree_batch_packed.h"

template <typename T, uint32_t FIXED_ROWS>
__aicore__ __attribute__((noinline)) void TreeM1FallbackTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    GM_ADDR tilingAddress) {
    TreeLineMinDistanceTilingData tiling;
    const auto words=reinterpret_cast<__gm__ const uint32_t*>(tilingAddress);
    tiling.batches=words[0];tiling.points=words[1];
    tiling.segments=1;tiling.paddedSegments=1;
    tiling.pointTile=words[4];tiling.blocks=words[5];
    tiling.schedule=3;tiling.segmentLog=0;
    // Keep each complete multi-tile implementation outside the bounded owner entry.
    KernelTreePairs<T,3,false,FIXED_ROWS,(FIXED_ROWS ? 1 : 0)> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
    if constexpr(sizeof(T)==4 && FIXED_ROWS==0) {

    }
}

// Fixed-capacity M1 resource. All addresses match KernelTreePairs<float,3,true,1024,1>.
#include "tree_m1_fixed_arena.h"

template <typename T>
__aicore__ __attribute__((noinline)) void TreeM1SingleTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    uint32_t pointCount) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    if constexpr(sizeof(T)==4)AscendC::ICachePreLoad(int64_t(1));
#endif
    TreeM1FixedArena::Process(points,segments,distance,indices,pointCount);

}

template <typename T>
__aicore__ __attribute__((noinline)) void TreeTinyPairTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    const TreeLineMinDistanceTilingData &tiling) {
    // Preserve the existing one-query, two-segment vector resource in a helper.
    KernelTreeTiny<T,1,8,true,true,false,false,2> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
}

template <typename T>
__aicore__ __attribute__((noinline)) void TreeMidSingleTailTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    const TreeLineMinDistanceTilingData &tiling) {
    // Keep the bounded coefficient-cache resource in a separate device function.
    KernelTreeBatch<T,64,256,192,true,true> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
}

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template <typename T, uint32_t FIXED_POINTS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeShortPairTile(
    GM_ADDR points, GM_ADDR segments, GM_ADDR distance, GM_ADDR indices,
    uint32_t pointCount, uint32_t segmentCount) {
    // Fixed resource: only the actual N and M are live in this instantiation.
    // Process uses the existing bounded owner mapping; neither B nor blocks
    // is read by the single-tile class. All generic resources stay in the entry.
    TreeLineMinDistanceTilingData resource{0,FIXED_POINTS?FIXED_POINTS:pointCount,segmentCount,32,4,0,1,5};
    KernelTreePairs<T,1,true,4,32> op;
    op.Init(points,segments,distance,indices,resource);op.Process();
}
#endif

#if defined(DTYPE_TREE_POINTS)
template<typename T,uint32_t OWNERS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyOneTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices) {
    AscendC::ICachePreLoad(int64_t(1));
    const TreeLineMinDistanceTilingData resource{0,1,2,8,1,0,16,3};
    KernelTreeTiny<T,0,8,(sizeof(T)!=4),true,false,false,2,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(OWNERS==1U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<typename T>
__aicore__ __attribute__((noinline)) void TreeOneQueryTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,
    uint32_t batches,uint32_t segmentsCount,uint32_t blocks) {
    const TreeLineMinDistanceTilingData tiling{batches,1,segmentsCount,1024,1,blocks,8,10};
    KernelTreePairs<T,8,false,1,1024> op;
    op.Init(points,segments,distance,indices,tiling);op.Process();
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<typename T,uint32_t FIXED_POINTS=0>
__aicore__ __attribute__((always_inline)) inline void TreeFewPointTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t pointCount) {
    const TreeLineMinDistanceTilingData resource{0,FIXED_POINTS?FIXED_POINTS:pointCount,1,8,4,0,15,3};
    KernelTreeTiny<T,4,1,false,false,true,true,0,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(FIXED_POINTS>0U && FIXED_POINTS<=4U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<typename T,uint32_t FIXED_POINTS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeNativePairPointTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t pointCount) {
    AscendC::ICachePreLoad(int64_t(1));
    const TreeLineMinDistanceTilingData resource{0,FIXED_POINTS?FIXED_POINTS:pointCount,2,8,4,0,23,3};
    KernelTreeTiny<T,4,2,false,false,false,false,2,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(FIXED_POINTS>0U && FIXED_POINTS<=4U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<typename T,uint32_t FIXED_SEGMENTS=0>
__aicore__ __attribute__((noinline,aligned(128))) void TreeBoundedSingleQueryTile(
    GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t segmentCount) {
    const TreeLineMinDistanceTilingData resource{0,1,FIXED_SEGMENTS?FIXED_SEGMENTS:segmentCount,8,1,0,24,3};
    KernelTreeTiny<T,0,8,false,true,false,false,FIXED_SEGMENTS,false,true> op;
    op.Init(points,segments,distance,indices,resource);op.Process(FIXED_SEGMENTS>0U?0U:GetBlockIdx());
}
#endif

#if defined(DTYPE_TREE_POINTS)
template<uint32_t RESERVED>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactCore(
    GM_ADDR distanceAddress,GM_ADDR indexAddress,uint32_t rows,uint32_t segments,bool halfOutput);
template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactFallback(
    GM_ADDR pointAddress,GM_ADDR segmentAddress,GM_ADDR distanceAddress,GM_ADDR indexAddress,
    uint32_t rows,uint32_t segments) {
    // All previous Tiny buffers are dead. This disjoint arena holds exact
    // radix-4096 rational packets; no host numerical fallback is involved.
    PipeBarrier<PIPE_ALL>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    LocalTensor<T> raw(TPosition::VECCALC,100000,128);
    LocalTensor<float> pf(TPosition::VECCALC,100512,64),sf(TPosition::VECCALC,100768,128);
    LocalTensor<float> point(TPosition::VECCALC,101280,48),temp(TPosition::VECCALC,101536,64);
    LocalTensor<int32_t> lanes(TPosition::VECCALC,101792,16),offsets(TPosition::VECCALC,101856,16);
    LocalTensor<float> exponents(TPosition::VECCALC,101920,192),scratch(TPosition::VECCALC,102688,64);
    LocalTensor<float> record(TPosition::VECCALC,102944,64);
    LocalTensor<uint8_t> mask(TPosition::VECCALC,103200,256);
    GlobalTensor<T> pg,sg,dg;GlobalTensor<int32_t> ig;
    pg.SetGlobalBuffer((__gm__ T*)pointAddress);sg.SetGlobalBuffer((__gm__ T*)segmentAddress);
    dg.SetGlobalBuffer((__gm__ T*)distanceAddress);ig.SetGlobalBuffer((__gm__ int32_t*)indexAddress);
    Duplicate(pf,0.0F,64);Duplicate(sf,0.0F,128);
    if constexpr(sizeof(T)==2)Duplicate(raw,T(0),128);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    if constexpr(sizeof(T)==2) {
        DataCopyPad(raw,pg,DataCopyExtParams{1,rows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
        DataCopyPad(raw[64],sg,DataCopyExtParams{1,segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
    } else {
        DataCopyPad(pf,pg,DataCopyExtParams{1,rows*12,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        DataCopyPad(sf,sg,DataCopyExtParams{1,segments*24,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    if constexpr(sizeof(T)==2) {
        Cast(pf,raw,RoundMode::CAST_NONE,64);Cast(sf,raw[64],RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
    }
    TreeTinyExactCore<0>(distanceAddress,indexAddress,rows,segments,sizeof(T)==2);
}
#endif

#if defined(DTYPE_TREE_POINTS)

template<class T, class StageSegmentsType, class StageWorkType, class StagePointType, class StagePacketType, class StageLaneType, class StageOffsetsType, class StagePacketRawType, class StageSgType, class StageExactType>
__aicore__ __attribute__((always_inline)) inline void TreeGeneralExactFallbackDuplicateStage(StageSegmentsType& segments, StageWorkType& work, StagePointType& point, StagePacketType& packet, StageLaneType& lane, StageOffsetsType& offsets, StagePacketRawType& packetRaw, StageSgType& sg, StageExactType& exact) {
    for(uint32_t start=0;start<segments;start+=16) {
        const uint32_t count=segments-start<16?segments-start:16;
        Duplicate(packet,0.0F,96);
        if constexpr(sizeof(T)==2)Duplicate(packetRaw,T(0),96);
        PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        if constexpr(sizeof(T)==2)
            DataCopyPad(packetRaw,sg[start*6],DataCopyExtParams{1,count*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
        else
            DataCopyPad(packet,sg[start*6],DataCopyExtParams{1,count*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if constexpr(sizeof(T)==2){Cast(packet,packetRaw,RoundMode::CAST_NONE,96);PipeBarrier<PIPE_V>();}
        exact.MakeScores(point,packet,offsets.template ReinterpretCast<uint32_t>(),segments,start);
        exact.Tournament(lane.template ReinterpretCast<uint32_t>());exact.Merge(work[20000],start==0);
    }
}


template<class T, class StageSegmentsType, class StageWorkType, class StagePointType, class StagePacketType, class StageTempType, class StageLaneType, class StageOffsetsType, class StagePacketRawType, class StageSgType, class StageDgType, class StageIgType, class StageLimbsType, class StageExactType>
__aicore__ __attribute__((always_inline)) inline void TreeGeneralExactFallbackMulsStage(StageSegmentsType& segments, StageWorkType& work, StagePointType& point, StagePacketType& packet, StageTempType& temp, StageLaneType& lane, StageOffsetsType& offsets, StagePacketRawType& packetRaw, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageLimbsType& limbs, StageExactType& exact) {
    Muls(offsets,lane,int32_t(24),16);PipeBarrier<PIPE_V>();
    TreeGeneralExactFallbackDuplicateStage<T>(segments, work, point, packet, lane, offsets, packetRaw, sg, exact);
    exact.WriteDistance(temp,lane.template ReinterpretCast<uint32_t>());
    Cast(temp[16].template ReinterpretCast<int32_t>(),exact.Plane(24*limbs),RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();
    LocalTensor<T> out=temp.template ReinterpretCast<T>();
    if constexpr(sizeof(T)==2){out=packetRaw;Cast(out,temp,RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(dg,out,DataCopyExtParams{1,uint32_t(sizeof(T)),0,0,0});
    DataCopyPad(ig,temp[16].template ReinterpretCast<int32_t>(),DataCopyExtParams{1,4,0,0,0});
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
template<class T, class StageSegmentsType, class StageWorkType, class StageSfType, class StagePointType, class StagePacketType, class StageTempType, class StageLaneType, class StageOffsetsType, class StagePacketRawType, class StageSgType, class StageDgType, class StageIgType, class StageCoordsType, class StageExponentType, class StageScratchType, class StageRecordType, class StageReducedType, class StageMaskType>
__aicore__ __attribute__((always_inline)) inline void TreeGeneralExactFallbackAndStage(StageSegmentsType& segments, StageWorkType& work, StageSfType& sf, StagePointType& point, StagePacketType& packet, StageTempType& temp, StageLaneType& lane, StageOffsetsType& offsets, StagePacketRawType& packetRaw, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageCoordsType& coords, StageExponentType& exponent, StageScratchType& scratch, StageRecordType& record, StageReducedType& reduced, StageMaskType& mask) {
    PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<coords;k+=64)
        And(exponent[k].template ReinterpretCast<uint16_t>(),exponent[k].template ReinterpretCast<uint16_t>(),scratch.template ReinterpretCast<uint16_t>(),128);
    PipeBarrier<PIPE_V>();Cast(exponent,exponent.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,coords);PipeBarrier<PIPE_V>();
    Maxs(exponent,exponent,1.0F,coords);PipeBarrier<PIPE_V>();
    Duplicate(record,0.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMax(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(reduced[8].template ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t maxExp=uint32_t(reduced[8].template ReinterpretCast<int32_t>().GetValue(0));
    FilterZeroMagnitudeExponents(scratch,work[6272],sf,exponent,mask,coords);
    Duplicate(record,255.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMin(record,exponent,64,coords/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(reduced,record,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(reduced,reduced,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(reduced[8].template ReinterpretCast<int32_t>(),reduced,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t minExp=uint32_t(reduced[8].template ReinterpretCast<int32_t>().GetValue(0));
    if(minExp>maxExp)minExp=maxExp;
    const uint32_t required=(maxExp-minExp+39)/12,limbs=required<4?4:required;
    TreeFloatRational16 exact(0,limbs,minExp);exact.Init();
    Duplicate(lane,int32_t(0),16);PipeBarrier<PIPE_V>();
    const uint64_t laneMasks[4]={0xaaaa,0xcccc,0xf0f0,0xff00};
    for(uint32_t k=0;k<4;++k) {
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),laneMasks[k]);
        Adds<int32_t,false>(lane,lane,int32_t(1U<<k),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
    TreeGeneralExactFallbackMulsStage<T>(segments, work, point, packet, temp, lane, offsets, packetRaw, sg, dg, ig, limbs, exact);
}
template<class T>
__aicore__ __attribute__((noinline,aligned(128))) void TreeGeneralExactFallback(
    GM_ADDR pointAddress,GM_ADDR segmentAddress,GM_ADDR distanceAddress,GM_ADDR indexAddress,
    uint32_t segments) {
    PipeBarrier<PIPE_ALL>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    LocalTensor<float> work(TPosition::VECCALC,0,25344);
    auto sf=work[12000],point=work[22528],packet=work[22656],temp=work[22800];
    auto lane=work[22912].template ReinterpretCast<int32_t>();
    auto offsets=work[22928].template ReinterpretCast<int32_t>();
    auto pointRaw=work[23104].template ReinterpretCast<T>();
    auto packetRaw=work[23232].template ReinterpretCast<T>();
    GlobalTensor<T> pg,sg,dg;GlobalTensor<int32_t> ig;
    pg.SetGlobalBuffer((__gm__ T*)pointAddress);sg.SetGlobalBuffer((__gm__ T*)segmentAddress);
    dg.SetGlobalBuffer((__gm__ T*)distanceAddress);ig.SetGlobalBuffer((__gm__ int32_t*)indexAddress);
    const uint32_t padded=(6*segments+63)/64*64;
    Duplicate(sf,0.0F,padded+64);Duplicate(pointRaw,T(0),64);
    if constexpr(sizeof(T)==2)Duplicate(work[8496].template ReinterpretCast<T>(),T(0),padded);
    PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    DataCopyPad(pointRaw,pg,DataCopyExtParams{1,3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
    if constexpr(sizeof(T)==2)
        DataCopyPad(work[8496].template ReinterpretCast<T>(),sg,DataCopyExtParams{1,segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
    else
        DataCopyPad(sf,sg,DataCopyExtParams{1,segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    if constexpr(sizeof(T)==2) {
        Cast(work[23136],pointRaw,RoundMode::CAST_NONE,64);
        Cast(sf,work[8496].template ReinterpretCast<T>(),RoundMode::CAST_NONE,padded);PipeBarrier<PIPE_V>();
    }
    auto pointFloat=sizeof(T)==2?work[23136]:work[23104];
    Brcb(temp,pointFloat,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    for(uint32_t axis=0;axis<3;++axis)
        Copy(point[axis*16],temp[axis*8],16,1,CopyRepeatParams(1,0,8,0));
    PipeBarrier<PIPE_V>();
    Copy(sf[padded],point,48,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();

    const uint32_t coords=padded+64;
    auto exponent=work,scratch=work[6208],record=work[8192],reduced=work[8320];
    auto mask=work[7000].template ReinterpretCast<uint8_t>();
    ShiftRight(exponent.template ReinterpretCast<uint32_t>(),sf.template ReinterpretCast<uint32_t>(),uint32_t(23),coords);
    Duplicate(scratch.template ReinterpretCast<uint32_t>(),uint32_t(255),64);TreeGeneralExactFallbackAndStage<T>(segments, work, sf, point, packet, temp, lane, offsets, packetRaw, sg, dg, ig, coords, exponent, scratch, record, reduced, mask);
}
#endif


template<typename DT_TREE_POINTS, uint32_t SCHEDULE, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageTiling_dataType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceInitSocStateStage(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageTiling_dataType& tiling_data) {
    if constexpr(SCHEDULE==4 && sizeof(DT_TREE_POINTS)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        ) {
        InitSocState();
        const uint32_t tiles=(tiling_data.points+tiling_data.pointTile-1U)/tiling_data.pointTile;
        if(tiling_data.pointTile==16U && tiling_data.batches*tiles<=tiling_data.blocks) {
            KernelTreePairs<DT_TREE_POINTS,4,true,16,64> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
        if(tiling_data.batches*tiles<=tiling_data.blocks) {
            KernelTreePairs<DT_TREE_POINTS,4,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreePairs<DT_TREE_POINTS,4> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
        }


    } else if constexpr(SCHEDULE==3 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        InitSocState();
        // A bounded1K-query arena makes all UB pitches constant. The class
        // covers a continuous range of query tile sizes, with a dynamic fallback.
        if(tiling_data.pointTile>512U && tiling_data.pointTile<=1024U) {
            const uint32_t tiles=(tiling_data.points+1023U)/1024U;
            const uint32_t owners=tiling_data.batches*tiles;
            if(owners<=tiling_data.blocks) {
                if(GetBlockIdx()<owners) {
                    TreeM1SingleTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data.points);
                }
            } else {
                KernelTreePairs<DT_TREE_POINTS,3,false,1024,1> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else {
            KernelTreePairs<DT_TREE_POINTS,3> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS, SCHEDULE> op;
        op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data);
        op.Process();
    }
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageWorkspaceType, class StageTiling_dataType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceInitStage(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageWorkspaceType& workspace, StageTiling_dataType& tiling_data) {
    if constexpr(SCHEDULE==6 && sizeof(DT_TREE_POINTS)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
    ) {
        if(tiling_data.segments>576) {
            KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreeBatch<DT_TREE_POINTS,64,1024,576,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else {
    if(tiling_data.segments<=576) {
        if constexpr(sizeof(DT_TREE_POINTS)==4) {
            AscendC::ICachePreLoad(int64_t(6));
            KernelTreeBatchPacked<DT_TREE_POINTS,64,1024,576,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreeBatch<DT_TREE_POINTS,64,1024,576,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(SCHEDULE==6 && sizeof(DT_TREE_POINTS)==4) {
            if(tiling_data.segments==1024 && tiling_data.batches*tiling_data.points/tiling_data.blocks>=1024)
                TreeSpatialProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling_data);
            else {
                KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,false,false> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else
#endif
        {
        KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,(SCHEDULE==11||SCHEDULE==13),SCHEDULE==11> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);
        op.Process();
        }
    }
    }
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS, class StageWordsType, class StageTiling_dataType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceStateStage(StageWordsType& words, StageTiling_dataType& tiling_data) {
    if constexpr((SCHEDULE==18 || SCHEDULE==19) && sizeof(DT_TREE_POINTS)==2) {
        tiling_data.segments=words[2];tiling_data.paddedSegments=SCHEDULE==18?32U:64U;
        tiling_data.pointTile=SCHEDULE==18?8U:32U;tiling_data.blocks=words[5];
        tiling_data.schedule=SCHEDULE;tiling_data.segmentLog=SCHEDULE==18?5U:6U;
    } else if constexpr(SCHEDULE==17 && sizeof(DT_TREE_POINTS)==4) {
        // The one-tail 192-coefficient resource contains two full 64-lane
        // groups and one live tail. B, N and the active grid remain dynamic.
        tiling_data.segments=2U*64U+1U;tiling_data.paddedSegments=256U;
        tiling_data.pointTile=16U;tiling_data.blocks=words[5];
        tiling_data.schedule=17U;tiling_data.segmentLog=8U;
    } else if constexpr(SCHEDULE==3 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        // Schedule 3 is the host's single-segment domain. These four fields
        // are invariants of that domain for every admitted shape and value.
        tiling_data.segments=1;tiling_data.paddedSegments=1;
        tiling_data.pointTile=words[4];tiling_data.blocks=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        tiling_data.schedule=3;tiling_data.segmentLog=0;
    } else {
        tiling_data.segments=words[2];tiling_data.paddedSegments=words[3];
        tiling_data.pointTile=words[4];tiling_data.blocks=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        tiling_data.schedule=words[6];tiling_data.segmentLog=words[7];
    }
}

template<typename DT_TREE_POINTS, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageTiling_dataType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceInitSocStateStage2(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageTiling_dataType& tiling_data) {
    {
        InitSocState();
        if(tiling_data.pointTile==4) {
            if(tiling_data.segments==1) {
                KernelTreeTiny<DT_TREE_POINTS,4,1,false,false,true> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else if(tiling_data.segments==2 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
                      && sizeof(DTYPE_TREE_POINTS)==4
#endif
                      ) {
                KernelTreeTiny<DT_TREE_POINTS,4,2,false,false,false,false,2> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else {
                KernelTreeTiny<DT_TREE_POINTS,4,2> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else if(tiling_data.segments==2 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
                  && sizeof(DTYPE_TREE_POINTS)==4
#endif
                  ) {
            TreeTinyPairTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);
        } else if(tiling_data.segments>1) {
            KernelTreeTiny<DT_TREE_POINTS,1,8,true,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreeTiny<DT_TREE_POINTS,1,8,false,false,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    }
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageTiling_dataType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceInitSocStateStage3(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageTiling_dataType& tiling_data) {
    if constexpr(SCHEDULE==8 && sizeof(DT_TREE_POINTS)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        ) {
        InitSocState();
        if(tiling_data.paddedSegments==1024U) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
            TreeOneQueryTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data.batches,tiling_data.segments,tiling_data.blocks);
#endif
        } else {
            KernelTreePairs<DT_TREE_POINTS,8> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else if constexpr (SCHEDULE == 5) { tree_line_min_distanceInitSocStateStage2<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data); } else if constexpr (SCHEDULE == 1) {
        InitSocState();
        uint32_t tiles;
        if constexpr(sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
                     && sizeof(DTYPE_TREE_POINTS)==4
#endif
                     ) {
            tiles=tiling_data.points<=tiling_data.pointTile?1U:
                (tiling_data.points+tiling_data.pointTile-1)/tiling_data.pointTile;
        } else tiles=(tiling_data.points+tiling_data.pointTile-1)/tiling_data.pointTile;
        if(tiling_data.paddedSegments<=32 && tiling_data.batches*tiles<=tiling_data.blocks) {
            if(tiling_data.pointTile==4 && tiling_data.paddedSegments==32) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                if constexpr(sizeof(DT_TREE_POINTS)==4) {
                    TreeShortPairTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,
                        tiling_data.points,tiling_data.segments);
                } else {
#endif
                KernelTreePairs<DT_TREE_POINTS,1,true,4,32> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                }
#endif
            } else {
                KernelTreePairs<DT_TREE_POINTS,1,true> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else {
            KernelTreePairs<DT_TREE_POINTS,1> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else tree_line_min_distanceInitSocStateStage<DT_TREE_POINTS, SCHEDULE>(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data);
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageTiling_dataType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceInitSocStateStage4(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageTiling_dataType& tiling_data) {
    if constexpr (SCHEDULE == 15) {
        InitSocState();
        KernelTreeTiny<DT_TREE_POINTS,4,1,false,false,true,true> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 16) {
        InitSocState();
        KernelTreeTiny<DT_TREE_POINTS,0,8,true,true,false,false,2> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 17) {
        InitSocState();
        if constexpr(sizeof(DT_TREE_POINTS)==4) {
            AscendC::ICachePreLoad(int64_t(4));
            KernelTreeBatch<DT_TREE_POINTS,64,256,192,true,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            KernelTreeTiny<DT_TREE_POINTS,1,8,false,false,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        }
    } else if constexpr(SCHEDULE==18 && sizeof(DT_TREE_POINTS)==2) {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS,1,true,8,32> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr(SCHEDULE==19 && sizeof(DT_TREE_POINTS)==2) {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS,4,true,32,64> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr(SCHEDULE==18 || SCHEDULE==19 || SCHEDULE==22) {
        InitSocState();
        AscendC::ICachePreLoad(int64_t(2));
        KernelTreeBatchPacked<DT_TREE_POINTS,64,1024,576,SCHEDULE!=18,SCHEDULE==19> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 20) {
        if constexpr(sizeof(DT_TREE_POINTS)==4) {
            InitSocState();
            KernelTreeSinglePointSingleSegment<DT_TREE_POINTS,1,8,false,false,true,false,0,true> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
        } else {
            InitSocState();
            KernelTreeTiny<DT_TREE_POINTS,1,8,false,false,true,false,0,true,false> op;
            op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);
            op.Process(RESIDENT_OWNERS==1U?0U:GetBlockIdx());
        }
    } else tree_line_min_distanceInitSocStateStage3<DT_TREE_POINTS, SCHEDULE>(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data);
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageWorkspaceType, class StageTiling_dataType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceInitSocStateStage5(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageWorkspaceType& workspace, StageTiling_dataType& tiling_data) {
    if constexpr (SCHEDULE == 6 || SCHEDULE == 11 || SCHEDULE == 13) {
        InitSocState();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(SCHEDULE==6 && sizeof(DT_TREE_POINTS)==4) {
            // Emit the complete-cache resource before the bounded prefix resource.
            if(tiling_data.segments>576) {
                if(tiling_data.segments==1024 && tiling_data.batches*tiling_data.points/tiling_data.blocks>=1024)
                    TreeSpatialProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling_data);
                else {
                    KernelTreeBatch<DT_TREE_POINTS,64,1024,1024,false,false> op;
                    op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
                }
            } else {
                KernelTreeBatchPacked<DT_TREE_POINTS,64,1024,576,false,false> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else
#endif
        tree_line_min_distanceInitStage<DT_TREE_POINTS, SCHEDULE>(tree_points, line_segments, min_distance, nearest_segment_idx, workspace, tiling_data);
    } else if constexpr (SCHEDULE == 9) {
        InitSocState();
        KernelTreePairs<DT_TREE_POINTS,2,true> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else if constexpr (SCHEDULE == 14) {
        InitSocState();
        KernelTreeTiny<DT_TREE_POINTS,0,8,true,true> op;
        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
    } else tree_line_min_distanceInitSocStateStage4<DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS>(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data);
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, class ControlArgument0, class ControlArgument1, class ControlArgument2, class ControlArgument3, class ControlArgument4>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceTree_line_min_distanceStateStageStageControlStage10(ControlArgument0& tiling_data, ControlArgument1& tree_points, ControlArgument2& line_segments, ControlArgument3& min_distance, ControlArgument4& nearest_segment_idx) {
if(tiling_data.segments<=192) {
                        if constexpr(sizeof(DT_TREE_POINTS)==4)AscendC::ICachePreLoad(int64_t(6));
                        KernelTreeBatch<DT_TREE_POINTS,64,256,192,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
                        op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
                } else {
                    KernelTreeBatch<DT_TREE_POINTS,64,256,256,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
                    op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
                }
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageWorkspaceType, class StageTilingType, class StageWordsType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceTree_line_min_distanceStateStageStage(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageWorkspaceType& workspace, StageTilingType& tiling, StageWordsType& words) {
    {
        TreeLineMinDistanceTilingData tiling_data;
        tiling_data.batches=words[0];tiling_data.points=words[1];
        tree_line_min_distanceStateStage<DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS>(words, tiling_data);
        if (GetBlockIdx() < tiling_data.blocks) {
        if constexpr (SCHEDULE == 21) {
            InitSocState();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeGlobalSpatialCompactProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling);


#elif defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
            TreeHalfGlobalSpatialProcess(tree_points,line_segments,min_distance,nearest_segment_idx,GetUserWorkspace(workspace),tiling_data);
#endif
        } else if constexpr (SCHEDULE == 7 || SCHEDULE == 10 || SCHEDULE == 12) {
            InitSocState();
            if(tiling_data.paddedSegments==128) {
                KernelTreeBatch<DT_TREE_POINTS,64,128,128,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            } else if(tiling_data.paddedSegments==256) {
                tree_line_min_distanceTree_line_min_distanceStateStageStageControlStage10<DT_TREE_POINTS, SCHEDULE>(tiling_data, tree_points, line_segments, min_distance, nearest_segment_idx);
            } else {
                KernelTreeBatch<DT_TREE_POINTS,64,512,512,(SCHEDULE==10||SCHEDULE==12),SCHEDULE==10> op;
                op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,tiling_data);op.Process();
            }
        } else tree_line_min_distanceInitSocStateStage5<DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS>(tree_points, line_segments, min_distance, nearest_segment_idx, workspace, tiling_data);
    }
    }
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageWorkspaceType, class StageTilingType, class StageWordsType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceGetBlockIdxStage(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageWorkspaceType& workspace, StageTilingType& tiling, StageWordsType& words) {
    if constexpr(SCHEDULE==23 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        if constexpr(RESIDENT_OWNERS>0) {
            // A complete single-batch resident packet encodes N and owns four points per core.
            if(GetBlockIdx()<(RESIDENT_OWNERS+3U)/4U) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                TreeNativePairPointTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,nearest_segment_idx,RESIDENT_OWNERS);
#endif
            }
        } else {
        const uint32_t pointCount=words[1],owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeNativePairPointTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,pointCount);
#endif
        }
        }
    } else if constexpr(SCHEDULE==24 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        if constexpr(RESIDENT_OWNERS>0) {
            // The complete single-batch point resource has exactly one owner.
            if(GetBlockIdx()==0) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                TreeBoundedSingleQueryTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,nearest_segment_idx,RESIDENT_OWNERS);
#endif
            }
        } else {
        const uint32_t segmentCount=words[2],owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeBoundedSingleQueryTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,segmentCount);
#endif
        }
        }
    } else { tree_line_min_distanceTree_line_min_distanceStateStageStage<DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS>(tree_points, line_segments, min_distance, nearest_segment_idx, workspace, tiling, words); }
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS, class StageTree_pointsType, class StageLine_segmentsType, class StageMin_distanceType, class StageNearest_segment_idxType, class StageWorkspaceType, class StageTilingType, class StageWordsType>
__aicore__ __attribute__((always_inline)) inline void tree_line_min_distanceGetBlockIdxStage2(StageTree_pointsType& tree_points, StageLine_segmentsType& line_segments, StageMin_distanceType& min_distance, StageNearest_segment_idxType& nearest_segment_idx, StageWorkspaceType& workspace, StageTilingType& tiling, StageWordsType& words) {
    if constexpr(SCHEDULE>=25 && SCHEDULE<=32
#ifdef DTYPE_TREE_POINTS
        && sizeof(DT_TREE_POINTS)==sizeof(DTYPE_TREE_POINTS)
#else
        && false
#endif
        ) {
        constexpr uint32_t residentOwners=SCHEDULE-24U;
        if(GetBlockIdx()<residentOwners) {
#if defined(DTYPE_TREE_POINTS)
            TreeTinyOneTile<DT_TREE_POINTS,residentOwners>(tree_points,line_segments,min_distance,nearest_segment_idx);
#endif
        }
    } else if constexpr(SCHEDULE==15 && sizeof(DT_TREE_POINTS)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) {
        if constexpr(RESIDENT_OWNERS>0) {
            constexpr uint32_t pointCount=RESIDENT_OWNERS,owners=(pointCount+3U)/4U;
            if(GetBlockIdx()<owners) {
                // Native packet initializes Vector state after the input DMA is queued.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                TreeFewPointTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,nearest_segment_idx,pointCount);
#endif
            }
        } else {
        const uint32_t pointCount=words[1],owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
            // Native packet initializes Vector state after the input DMA is queued.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeFewPointTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx,pointCount);
#endif
        }
        }
    } else if constexpr(SCHEDULE==16
#ifdef DTYPE_TREE_POINTS
        && sizeof(DT_TREE_POINTS)==sizeof(DTYPE_TREE_POINTS)
#else
        && false
#endif
        ) {
        const uint32_t owners=RESIDENT_OWNERS?RESIDENT_OWNERS:words[5];
        if(GetBlockIdx()<owners) {
#if defined(DTYPE_TREE_POINTS)
            TreeTinyOneTile<DT_TREE_POINTS>(tree_points,line_segments,min_distance,nearest_segment_idx);
#endif
        }
    } else tree_line_min_distanceGetBlockIdxStage<DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS>(tree_points, line_segments, min_distance, nearest_segment_idx, workspace, tiling, words);
}

template<typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS>
__aicore__ constexpr bool UsesScheduleStage() { return (SCHEDULE==1 && sizeof(DT_TREE_POINTS)==4 && RESIDENT_OWNERS>0
#ifdef DTYPE_TREE_POINTS
    && sizeof(DTYPE_TREE_POINTS)==4
#endif
    );
}
template <typename DT_TREE_POINTS, uint32_t SCHEDULE, uint32_t RESIDENT_OWNERS>
 __global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments, GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    const auto words=reinterpret_cast<__gm__ const uint32_t*>(tiling);
    if constexpr(UsesScheduleStage<DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS>()) {
        if(GetBlockIdx()<words[5]) {
            InitSocState();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            TreeShortPairTile<DT_TREE_POINTS,RESIDENT_OWNERS>(tree_points,line_segments,min_distance,
                nearest_segment_idx,RESIDENT_OWNERS,words[2]);
#endif
        }
    } else tree_line_min_distanceGetBlockIdxStage2<DT_TREE_POINTS, SCHEDULE, RESIDENT_OWNERS>(tree_points, line_segments, min_distance, nearest_segment_idx, workspace, tiling, words);
 }

#undef TREE_NATIVE_HALF_TOKEN
#undef TREE_NATIVE_HALF_TOKEN_IMPL
#undef TREE_NATIVE_HALF_TOKEN_half
#undef TREE_NATIVE_HALF_TOKEN_float

#undef TREE_NATIVE_FLOAT_TOKEN
#undef TREE_NATIVE_FLOAT_TOKEN_IMPL
#undef TREE_NATIVE_FLOAT_TOKEN_half
#undef TREE_NATIVE_FLOAT_TOKEN_float

#if defined(DTYPE_TREE_POINTS)

template<class StageRowsType, class StageSegmentsType, class StageHalfOutputType, class StageRawType, class StagePfType, class StageSfType, class StagePointType, class StageTempType, class StageLanesType, class StageOffsetsType, class StageDgType, class StageDhType, class StageIgType, class StageLimbsType, class StageExactType>
__aicore__ __attribute__((always_inline)) inline void TreeTinyExactCoreBrcbStage(StageRowsType& rows, StageSegmentsType& segments, StageHalfOutputType& halfOutput, StageRawType& raw, StagePfType& pf, StageSfType& sf, StagePointType& point, StageTempType& temp, StageLanesType& lanes, StageOffsetsType& offsets, StageDgType& dg, StageDhType& dh, StageIgType& ig, StageLimbsType& limbs, StageExactType& exact) {
    for(uint32_t row=0;row<rows;++row) {
        for(uint32_t axis=0;axis<3;++axis) {
            const uint32_t pos=row*3+axis;
            Brcb(temp,pf[(pos/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(point[axis*16],temp[(pos%8)*8],16,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
        }
        exact.MakeScores(point,sf,offsets.template ReinterpretCast<uint32_t>(),segments);
        exact.Tournament(lanes.template ReinterpretCast<uint32_t>());
        exact.WriteDistance(temp,lanes.template ReinterpretCast<uint32_t>());
        Cast(temp[16].template ReinterpretCast<int32_t>(),exact.Plane(24*limbs),RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();
        if(halfOutput)Cast(raw,temp,RoundMode::CAST_RINT,1);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if(halfOutput)DataCopyPad(dh[row],raw,DataCopyExtParams{1,2,0,0,0});
        else DataCopyPad(dg[row],temp,DataCopyExtParams{1,4,0,0,0});
        DataCopyPad(ig[row],temp[16].template ReinterpretCast<int32_t>(),DataCopyExtParams{1,4,0,0,0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}

template<class StageRowsType, class StageSegmentsType, class StageHalfOutputType, class StageRawType, class StagePfType, class StageSfType, class StagePointType, class StageTempType, class StageLanesType, class StageOffsetsType, class StageDgType, class StageDhType, class StageIgType, class StageLimbsType, class StageExactType>
__aicore__ __attribute__((always_inline)) inline void TreeTinyExactCoreInitStage(StageRowsType& rows, StageSegmentsType& segments, StageHalfOutputType& halfOutput, StageRawType& raw, StagePfType& pf, StageSfType& sf, StagePointType& point, StageTempType& temp, StageLanesType& lanes, StageOffsetsType& offsets, StageDgType& dg, StageDhType& dh, StageIgType& ig, StageLimbsType& limbs, StageExactType& exact) {
    exact.Init();
    Duplicate(lanes,int32_t(0),16);PipeBarrier<PIPE_V>();
    const uint64_t bits[4]={0xaaaa,0xcccc,0xf0f0,0xff00};
    for(uint32_t k=0;k<4;++k) {
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),bits[k]);
        Adds<int32_t,false>(lanes,lanes,int32_t(1U<<k),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }
    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    Muls(offsets,lanes,int32_t(24),16);PipeBarrier<PIPE_V>();
    TreeTinyExactCoreBrcbStage(rows, segments, halfOutput, raw, pf, sf, point, temp, lanes, offsets, dg, dh, ig, limbs, exact);
    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}
template<uint32_t RESERVED>
__aicore__ __attribute__((noinline,aligned(128))) void TreeTinyExactCore(
    GM_ADDR distanceAddress,GM_ADDR indexAddress,uint32_t rows,uint32_t segments,bool halfOutput) {
    LocalTensor<half> raw(TPosition::VECCALC,100000,128);
    LocalTensor<float> pf(TPosition::VECCALC,100512,64),sf(TPosition::VECCALC,100768,128);
    LocalTensor<float> point(TPosition::VECCALC,101280,48),temp(TPosition::VECCALC,101536,64);
    LocalTensor<int32_t> lanes(TPosition::VECCALC,101792,16),offsets(TPosition::VECCALC,101856,16);
    LocalTensor<float> exponents(TPosition::VECCALC,101920,192),scratch(TPosition::VECCALC,102688,64);
    LocalTensor<float> record(TPosition::VECCALC,102944,64);
    LocalTensor<uint8_t> mask(TPosition::VECCALC,103200,256);
    GlobalTensor<float> dg;GlobalTensor<half> dh;GlobalTensor<int32_t> ig;
    dg.SetGlobalBuffer((__gm__ float*)distanceAddress);dh.SetGlobalBuffer((__gm__ half*)distanceAddress);
    ig.SetGlobalBuffer((__gm__ int32_t*)indexAddress);
    Copy(sf[64],pf,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    ShiftRight(exponents.ReinterpretCast<uint32_t>(),sf.ReinterpretCast<uint32_t>(),uint32_t(23),128);
    Duplicate(scratch.ReinterpretCast<uint32_t>(),uint32_t(255),64);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<128;k+=64)And(exponents[k].ReinterpretCast<uint16_t>(),exponents[k].ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),128);
    PipeBarrier<PIPE_V>();Cast(exponents,exponents.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
    Maxs(exponents,exponents,1.0F,128);PipeBarrier<PIPE_V>();
    WholeReduceMax(record,exponents,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMax(temp,record,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(temp[8].ReinterpretCast<int32_t>(),temp,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const uint32_t maxExp=uint32_t(temp[8].ReinterpretCast<int32_t>().GetValue(0));
    Duplicate(scratch.ReinterpretCast<uint32_t>(),uint32_t(0x7fffffff),64);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<128;k+=64) {
        And(temp.ReinterpretCast<uint16_t>(),sf[k].ReinterpretCast<uint16_t>(),scratch.ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        CompareScalar(mask,temp.ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Not(mask.ReinterpretCast<uint16_t>(),mask.ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Select(exponents[k],mask,exponents[k],255.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
    }
    WholeReduceMin(record,exponents,64,2,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    WholeReduceMin(temp,record,2,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
    Cast(temp[8].ReinterpretCast<int32_t>(),temp,RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t minExp=uint32_t(temp[8].ReinterpretCast<int32_t>().GetValue(0));
    if(minExp>maxExp)minExp=maxExp;
    const uint32_t required=(maxExp-minExp+39)/12,limbs=required<4?4:required;
    TreeFloatRational16 exact(0,limbs,minExp);TreeTinyExactCoreInitStage(rows, segments, halfOutput, raw, pf, sf, point, temp, lanes, offsets, dg, dh, ig, limbs, exact);
}
#endif
