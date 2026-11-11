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
// Inline implementation of KernelTreePairs.

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
template<auto maskWords, class ControlArgument0, class ControlArgument1, class ControlArgument2, class ControlArgument4>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::LoadSegmentsReinterpretCastStageControlStage2(ControlArgument0& lanes, ControlArgument1& bitmap, ControlArgument2& axis, ControlArgument4& words) {
                #pragma unroll
                        for(uint32_t phase=0;phase<3;++phase) {
                            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),lanes[phase]);
                            Duplicate<uint32_t,false>(bitmap[axis*maskWords],words[(axis+phase)%3],uint64_t(64),1,1,8);
                        }
}

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StagePrepareMetadataType, class StageExpandedCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::LoadSegmentsReinterpretCastStage(StagePrepareMetadataType& prepareMetadata, StageExpandedCountType& expandedCount) {
        if constexpr(MODE==3) {
            if(prepareMetadata) {
                const uint32_t coefficientCount=count_;
                count_=expandedCount;
                if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
                    // Each bitmap selects one coordinate in an arbitrary AoS3
                    // query array. No input values are involved in its construction.
                    auto bitmap=po_.template ReinterpretCast<uint32_t>();
                    constexpr uint32_t maskWords=3U*FIXED_ROWS/32U;
                    constexpr uint32_t words[3]={0x49249249U,0x92492492U,0x24924924U};
                    constexpr uint64_t lanes[3]={0x249249ULL,0x492492ULL,0x924924ULL};
                    SetMaskNorm();
            #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
LoadSegmentsReinterpretCastStageControlStage2<maskWords>(lanes, bitmap, axis, words);
                    }
                    PipeBarrier<PIPE_V>();
                    SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
            #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis)
                        Copy(bitmap[axis*maskWords+24],bitmap[axis*maskWords],24,3,CopyRepeatParams(1,1,3,0));
                    PipeBarrier<PIPE_V>();
                } else {
                    MakeIndices(po_,int32_t(0),count_);PipeBarrier<PIPE_V>();
                    Muls(po_,po_,int32_t(12),count_);
                }
                if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
                    // With one segment, every valid nearest-segment index is0.
                    // The index buffer has no other writer in this specialization.
                    Duplicate(stageIdx_,int32_t(0),rows_);
                }
                count_=coefficientCount;
            }
        } else if constexpr(MODE!=4 || SINGLE_TILE) {
            if(prepareMetadata) {
                const uint32_t coefficientCount=count_;count_=expandedCount;
                PrepareMetadata();count_=coefficientCount;
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckExactTileDifferences(uint32_t pointCount) {
        if constexpr(PACKED_HALF_ROW_OUTPUT) {
            const uint32_t segmentCount=6*stride_;
            const uint32_t values=segmentCount+pointCount,repeats=values/128U;
            auto packet=W(0).template ReinterpretCast<T>();packet.SetSize(values);
            auto reduced=W(3).template ReinterpretCast<T>();
            auto extrema=W(5).template ReinterpretCast<T>();
            Abs(packet,rawS_,segmentCount);Abs(packet[segmentCount],rawP_,pointCount);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(reduced,extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            CompareScalar(mask_,packet,T(0),CMPMODE::GT,values);PipeBarrier<PIPE_V>();
            Select(packet,mask_,packet,T(__builtin_inff()),SELMODE::VSEL_TENSOR_SCALAR_MODE,values);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced[16],extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                C(0),C(3),CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t largest=reduced.template ReinterpretCast<uint16_t>().GetValue(0);
            const uint32_t smallest=reduced.template ReinterpretCast<uint16_t>().GetValue(16);
            const uint32_t highField=largest>>10U,lowField=smallest>>10U;
            // Normal Half exponents translate by 112. A nonzero subnormal
            // mantissa m translates to 103+floor(log2(m)); OR 1 defines zero.
            const uint32_t highExponent=highField==31U?255U:highField?highField+112U:134U-uint32_t(__builtin_clz(largest|1U));
            const uint32_t lowExponent=lowField==31U?255U:lowField?lowField+112U:134U-uint32_t(__builtin_clz(smallest|1U));
            exactDifferences_=largest==0U || highExponent-lowExponent<=13U;
            const uint32_t validBits=t_.segments==32U?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
        } else { CheckExactTileDifferencesWStage(pointCount); }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StagePointCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckExactTileDifferencesWStage(StagePointCountType& pointCount) {
        {
        // This runtime certificate covers all input coordinates in the tile.
        // For two eleven-bit significands, gaps <=12 fit 24 bits even with
        // carry. At gap 13 the smaller operand is below the larger operand's
        // gap to its next power of two, so the signed difference still fits.
        // Effective subnormal significands are zero-padded subsets of this rule.
        const uint32_t segmentCount=6*stride_;
        const uint32_t values=(segmentCount+pointCount+63)/64*64,repeats=values/64;
        auto coordinates=W(0);coordinates.SetSize(values);
        if constexpr(PACKED_HALF_ROW_OUTPUT) {
            // Abs writes the complete packed packet directly from the two
            // resident inputs; no intermediate copy or clearing is needed.
            Abs(coordinates,sf_,segmentCount);
            Abs(coordinates[segmentCount],pf_,pointCount);PipeBarrier<PIPE_V>();
        } else {
        Duplicate(coordinates,0.0F,values);PipeBarrier<PIPE_V>();
        if(segmentCount>=64)Copy(coordinates,sf_,64,segmentCount/64,CopyRepeatParams(1,1,8,8));
        if(segmentCount%64)Copy(coordinates[(segmentCount/64)*64],sf_[(segmentCount/64)*64],segmentCount%64,1,CopyRepeatParams(1,1,8,8));
        Copy(coordinates[segmentCount],pf_,64,pointCount/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        Abs(coordinates,coordinates,values);PipeBarrier<PIPE_V>();
        }
        WholeReduceMax(W(3),coordinates,64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(W(4),W(3),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
        CompareScalar(mask_,coordinates,0.0F,CMPMODE::GT,values);PipeBarrier<PIPE_V>();
        Select(coordinates,mask_,coordinates,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,values);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(3),coordinates,64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(5),W(3),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            // W0 is dead after the extrema reduction. Share its predicate
            // and scalar synchronization with the exact-difference certificate.
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                C(0),C(3),CMPMODE::EQ,uint64_t(64),1,
                BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t largest=W(4).template ReinterpretCast<uint32_t>().GetValue(0);
        const uint32_t smallest=W(5).template ReinterpretCast<uint32_t>().GetValue(0);
        const uint32_t highExponent=(largest>>23)&255U,lowExponent=(smallest>>23)&255U;
        exactDifferences_=largest==0 || highExponent-lowExponent<=13;
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            const uint32_t validBits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
        }
            }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckExactDifferences(uint32_t pointCount) {
        if constexpr(DIRECT_HALF_ROW_POINTS) { CheckExactDifferencesWStage(pointCount); } else {
        // A binary16 value has at most eleven significant bits. An exponent
        // spread of at most twelve makes every signed difference fit binary32.
        Copy(W(0),sf_,64,6,CopyRepeatParams(1,1,8,8));
        Copy(W(0)[384],pf_,64,pointCount/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        const uint32_t values=384+pointCount,repeats=values/64;
        Abs(W(0),W(0),values);PipeBarrier<PIPE_V>();
        WholeReduceMax(W(1),W(0),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(W(2),W(1),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
        CompareScalar(mask_,W(0),0.0F,CMPMODE::GT,values);PipeBarrier<PIPE_V>();
        Select(W(0),mask_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,values);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(1),W(0),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(3),W(1),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(W(3),W(3),4096.0F,1);PipeBarrier<PIPE_V>();
        Compare(mask_,W(2),W(3),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            // Endpoint equality is exact for the promoted Half coordinates.
            // W(0) is dead after the min reduction and remains the equality mask
            // until PrepareTileGeometry consumes it. Share the existing V_S wait.
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                RawC(0),RawC(3),CMPMODE::EQ,uint64_t(64),1,
                BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        exactDifferences_=(mask_.GetValue(0)&1U)!=0;
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER) {
            const uint64_t validBits=t_.segments==64?~uint64_t(0):(uint64_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint64_t>().GetValue(0)&validBits)!=0;
        }
            }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StagePointCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckExactDifferencesWStage(StagePointCountType& pointCount) {
        {
            // The resident inputs are Half. Reduce their magnitudes before
            // promotion, then keep the original FP32 ratio certificate.
            const uint32_t values=384U+pointCount;
            const uint32_t padded=(values+127U)/128U*128U,repeats=padded/128U;
            auto packet=W(0).template ReinterpretCast<T>();packet.SetSize(padded);
            auto extrema=W(1).template ReinterpretCast<T>();
            auto reduced=W(2).template ReinterpretCast<T>();
            Abs(packet,rawS_,384U);Abs(packet[384],rawP_,pointCount);
            if(padded>values)Duplicate(packet[values],T(0),padded-values);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(reduced,extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            CompareScalar(mask_,packet,T(0),CMPMODE::GT,padded);PipeBarrier<PIPE_V>();
            Select(packet,mask_,packet,T(__builtin_inff()),SELMODE::VSEL_TENSOR_SCALAR_MODE,padded);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(extrema,packet,128,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(reduced[16],extrema,repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            Cast(W(3),reduced,RoundMode::CAST_NONE,1);
            Cast(W(4),reduced[16],RoundMode::CAST_NONE,1);PipeBarrier<PIPE_V>();
            Muls(W(4),W(4),4096.0F,1);PipeBarrier<PIPE_V>();
            Compare(mask_,W(3),W(4),CMPMODE::LE,1);
            Compare<float,uint8_t,false>(W(0).template ReinterpretCast<uint8_t>(),
                RawC(0),RawC(3),CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            exactDifferences_=(mask_.GetValue(0)&1U)!=0;
            const uint64_t validBits=t_.segments==64U?~uint64_t(0):(uint64_t(1)<<t_.segments)-1U;
            nativeHalfXTies_=(W(0).template ReinterpretCast<uint64_t>().GetValue(0)&validBits)!=0;
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometry() {
        constexpr uint64_t lanes=64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const uint8_t axisPitch=uint8_t(MODE==4?12:allocCount_/8);
        const BinaryRepeatParams axes{1,1,1,axisPitch,axisPitch,axisPitch};
        // The score certificate handles arbitrary endpoint orientation.
        // Keep mapped layouts canonical, because their raw endpoint storage aliases directions.
        if constexpr(!(MODE==2 && SINGLE_TILE && !MAPPED_STARTS)) { PrepareTileGeometryWStage<lanes>(unary, same, axisPitch, axes); }
        if constexpr(PACKED_HALF_GEOMETRY_AXES) {
            Sub<float,false>(C(6),C(3),C(0),lanes,3,BinaryRepeatParams(1,1,1,12,12,12));
        } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
            Sub<float,false>(C(6),C(3),C(0),lanes,3,axes);
        } else {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(C(axis+6),C(axis+3),C(axis),lanes,1,same);
        }
        PipeBarrier<PIPE_V>();
        if(!(sizeof(T)==2 && exactDifferences_)) { PrepareTileGeometrySubStage<lanes>(same, axes); } else {
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Duplicate<float,false>(C(10),0.0F,lanes,3,1,12);
            } else {
                #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Duplicate<float,false>(C(axis+10),0.0F,lanes,1,1,8);
            }
        }
        Mul<float,false>(C(9),C(6),C(6),lanes,1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(7),C(7),lanes,1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(8),C(8),lanes,1,same);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask_,C(9),0.0F,CMPMODE::GT,lanes,1,unary);PipeBarrier<PIPE_V>();
        Select<float,uint8_t,false>(C(9),mask_,C(9),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,lanes,1,same);PipeBarrier<PIPE_V>();
        // One complete coefficient register needs one reduction, without
        // reducing the single result a second time.
        WholeReduceMax(zeroBound_,C(9),64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(zeroBound_,zeroBound_,1.8189894035458565e-12F,1);PipeBarrier<PIPE_V>();
        Brcb(zeroBound_[8],zeroBound_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(zeroBound_[32],zeroBound_[8],64,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<auto lanes, class StageUnaryType, class StageSameType, class StageAxisPitchType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometryWStage(StageUnaryType& unary, StageSameType& same, StageAxisPitchType& axisPitch, StageAxesType& axes) {
        {
        auto keep=W(8).template ReinterpretCast<uint8_t>();
        auto equal=W(0).template ReinterpretCast<uint8_t>();
        auto less=W(1).template ReinterpretCast<uint8_t>();
        auto equalY=W(2).template ReinterpretCast<uint8_t>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(lanes);
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(keep,RawC(0),RawC(3),CMPMODE::LT,lanes,1,same);
        else Compare<float,uint8_t,false>(keep,C(0),C(3),CMPMODE::LT,lanes,1,same);
        if constexpr(!NATIVE_HALF_ENDPOINT_ORDER) {
            if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(equal,RawC(0),RawC(3),CMPMODE::EQ,lanes,1,same);
            else Compare<float,uint8_t,false>(equal,C(0),C(3),CMPMODE::EQ,lanes,1,same);
        }
        bool hasXTies=true;
        if constexpr(NATIVE_HALF_ENDPOINT_ORDER)hasXTies=nativeHalfXTies_;
        if constexpr(UsesTileCoefficientsStage2()) {
            // Only equal first coordinates require the remaining lexicographic axes.
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t validBits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
            hasXTies=(equal.template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
        }
        PrepareTileGeometryReinterpretCastStage<lanes>(same, equal, hasXTies);
        PrepareTileGeometryCompareStage<lanes>(same, keep, equal, less, equalY, hasXTies);
        SetVectorMask<float,MaskMode::NORMAL>(lanes);SetCmpMask(keep);
        PrepareTileGeometrySelectStage(same, axes);
        PipeBarrier<PIPE_V>();
        PrepareTileGeometrySelectStage2(same, axes);
        PipeBarrier<PIPE_V>();
        if constexpr(!MAPPED_STARTS) {
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Adds<float,false>(C(0),W(0),0.0F,lanes,3,UnaryRepeatParams(1,1,12,8));
            } else if constexpr(OWNED_AXES8 || PACKED_HALF_ROW_OUTPUT) {
                Adds<float,false>(C(0),W(0),0.0F,lanes,3,UnaryRepeatParams(1,1,axisPitch,axisPitch));
            } else {
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Adds<float,false>(C(axis),W(axis),0.0F,lanes,1,unary);
            }
            PipeBarrier<PIPE_V>();
        }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ constexpr bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::UsesTileCoefficientsStage2() { return (sizeof(T)==2 && TILE_COEFFICIENTS && FIXED_STRIDE==32 && !NATIVE_HALF_ENDPOINT_ORDER);
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageSameType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometrySelectStage2(StageSameType& same, StageAxesType& axes) {
        if constexpr(MAPPED_STARTS) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(3),RawC(3),RawC(0),3,axes);
        } else if constexpr(PACKED_HALF_GEOMETRY_AXES) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(3),C(3),C(0),3,BinaryRepeatParams(1,1,1,12,12,12));
        } else if constexpr(OWNED_AXES8 || PACKED_HALF_ROW_OUTPUT) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(3),C(3),C(0),3,axes);
        } else {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(axis+3),C(axis+3),C(axis),1,same);
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageSameType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometrySelectStage(StageSameType& same, StageAxesType& axes) {
        if constexpr(MAPPED_STARTS) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(0),RawC(0),RawC(3),3,axes);
        } else if constexpr(PACKED_HALF_GEOMETRY_AXES) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(0),C(0),C(3),3,BinaryRepeatParams(1,1,1,8,12,12));
        } else if constexpr(OWNED_AXES8 || PACKED_HALF_ROW_OUTPUT) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(0),C(0),C(3),3,axes);
        } else {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(axis),C(axis),C(axis+3),1,same);
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<auto lanes, class StageSameType, class StageKeepType, class StageEqualType, class StageLessType, class StageEqualYType, class StageHasXTiesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometryCompareStage(StageSameType& same, StageKeepType& keep, StageEqualType& equal, StageLessType& less, StageEqualYType& equalY, StageHasXTiesType& hasXTies) {
        if(hasXTies) {
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(less,RawC(1),RawC(4),CMPMODE::LT,lanes,1,same);
        else Compare<float,uint8_t,false>(less,C(1),C(4),CMPMODE::LT,lanes,1,same);
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(equalY,RawC(1),RawC(4),CMPMODE::EQ,lanes,1,same);
        else Compare<float,uint8_t,false>(equalY,C(1),C(4),CMPMODE::EQ,lanes,1,same);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
        And<uint16_t,false>(less.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);
        And<uint16_t,false>(equal.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),equalY.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep.template ReinterpretCast<uint16_t>(),keep.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);
        SetVectorMask<float,MaskMode::NORMAL>(lanes);
        if constexpr(MAPPED_STARTS) Compare<float,uint8_t,false>(less,RawC(2),RawC(5),CMPMODE::LE,lanes,1,same);
        else Compare<float,uint8_t,false>(less,C(2),C(5),CMPMODE::LE,lanes,1,same);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
        And<uint16_t,false>(less.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep.template ReinterpretCast<uint16_t>(),keep.template ReinterpretCast<uint16_t>(),less.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<auto lanes, class StageSameType, class StageEqualType, class StageHasXTiesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometryReinterpretCastStage(StageSameType& same, StageEqualType& equal, StageHasXTiesType& hasXTies) {
        if constexpr(sizeof(T)==4) {
            if constexpr(UsesModeStage2()) {
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint32_t validBits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1U;
                hasXTies=(equal.template ReinterpretCast<uint32_t>().GetValue(0)&validBits)!=0;
            } else {
                PipeBarrier<PIPE_V>();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
                And<uint16_t,false>(equal.template ReinterpretCast<uint16_t>(),equal.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,same);PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                hasXTies=equal.template ReinterpretCast<uint64_t>().GetValue(0)!=0;
                SetVectorMask<float,MaskMode::NORMAL>(lanes);
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ constexpr bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::UsesModeStage2() { return (MODE==1 && SINGLE_TILE && FIXED_ROWS==4 && FIXED_STRIDE==32 && SHORT_FP32_METADATA);
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
template<auto lanes, class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometrySubStageControlStage61(ControlArgument0& axes, ControlArgument1& same) {
if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Sub<float,false>(W(0),C(3),C(6),lanes,3,BinaryRepeatParams(1,1,1,8,12,12));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Sub<float,false>(W(0),C(3),C(6),lanes,3,axes);
            } else {
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),C(axis+3),C(axis+6),lanes,1,same);
            }
}

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<auto lanes, class StageSameType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareTileGeometrySubStage(StageSameType& same, StageAxesType& axes) {
        {
            PrepareTileGeometrySubStageControlStage61<lanes>(axes, same);
            PipeBarrier<PIPE_V>();
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Add<float,false>(W(3),C(6),W(0),lanes,3,BinaryRepeatParams(1,1,1,8,12,8));
                Sub<float,false>(C(10),W(0),C(0),lanes,3,BinaryRepeatParams(1,1,1,12,8,12));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Add<float,false>(W(3),C(6),W(0),lanes,3,axes);
                Sub<float,false>(C(10),W(0),C(0),lanes,3,axes);
            } else {
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis) {
                    Add<float,false>(W(axis+3),C(axis+6),W(axis),lanes,1,same);
                    Sub<float,false>(C(axis+10),W(axis),C(axis),lanes,1,same);
                }
            }
            PipeBarrier<PIPE_V>();
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Sub<float,false>(W(3),C(3),W(3),lanes,3,BinaryRepeatParams(1,1,1,8,12,8));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Sub<float,false>(W(3),C(3),W(3),lanes,3,axes);
            } else {
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis+3),C(axis+3),W(axis+3),lanes,1,same);
            }
            PipeBarrier<PIPE_V>();
            if constexpr(PACKED_HALF_GEOMETRY_AXES) {
                Add<float,false>(C(10),C(10),W(3),lanes,3,BinaryRepeatParams(1,1,1,12,12,8));
            } else if constexpr(BATCH_COORDINATE_AXES || PACKED_HALF_ROW_OUTPUT) {
                Add<float,false>(C(10),C(10),W(3),lanes,3,axes);
            } else {
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)Add<float,false>(C(axis+10),C(axis+10),W(axis+3),lanes,1,same);
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareSingleSegmentGeometryNative() {
        constexpr uint64_t lanes=64;
        const uint8_t workPitch=uint8_t(allocCount_/8);
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(lanes);
        Sub<float,false>(C(6),C(3),C(0),lanes,3,BinaryRepeatParams(1,1,1,13,13,13));
        PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),C(3),C(6),lanes,3,BinaryRepeatParams(1,1,1,workPitch,13,13));
        PipeBarrier<PIPE_V>();
        Add<float,false>(W(3),C(6),W(0),lanes,3,BinaryRepeatParams(1,1,1,workPitch,13,workPitch));
        Sub<float,false>(C(10),W(0),C(0),lanes,3,BinaryRepeatParams(1,1,1,13,workPitch,13));
        PipeBarrier<PIPE_V>();
        Sub<float,false>(W(3),C(3),W(3),lanes,3,BinaryRepeatParams(1,1,1,workPitch,13,workPitch));
        PipeBarrier<PIPE_V>();
        Add<float,false>(C(10),C(10),W(3),lanes,3,BinaryRepeatParams(1,1,1,13,13,workPitch));
        Mul<float,false>(C(9),C(6),C(6),lanes,1,same);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(7),C(7),lanes,1,same);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(8),C(8),lanes,1,same);
        PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask_,C(9),0.0F,CMPMODE::GT,lanes,1,unary);
        PipeBarrier<PIPE_V>();
        Select<float,uint8_t,false>(C(9),mask_,C(9),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,lanes,1,same);
    }
