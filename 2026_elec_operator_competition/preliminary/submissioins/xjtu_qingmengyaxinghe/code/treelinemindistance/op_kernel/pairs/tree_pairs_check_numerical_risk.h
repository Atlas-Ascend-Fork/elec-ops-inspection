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
    template<bool FAST_FLOAT_DIFFERENCES>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckNumericalRisk() {
#if defined(DTYPE_TREE_POINTS)
        if constexpr(MODE==3 && sizeof(T)==4) { CheckNumericalRiskCompareScalarStage(); } else if constexpr(MODE!=3) {
            if (CheckNumericalRiskGetValueStage3()) return;
            auto possible=zeroExact_.template ReinterpretCast<uint8_t>();
            if (CheckNumericalRiskMulsStage<FAST_FLOAT_DIFFERENCES>(possible)) return;

            if constexpr(MODE==2) {
                Brcb(W(8),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            } else if constexpr(MODE==8) {
                Brcb(W(6),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                Brcb(W(7),W(6),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                Copy(W(8),W(7),64,count_/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
            } else {
                MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,count_);PipeBarrier<PIPE_V>();
                uint32_t shift=0;for(uint32_t s=stride_;s>1;s>>=1)++shift;
                ShiftRight(W(6).template ReinterpretCast<uint32_t>(),W(6).template ReinterpretCast<uint32_t>(),shift,count_);PipeBarrier<PIPE_V>();
                Muls(W(6).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),int32_t(4),count_);PipeBarrier<PIPE_V>();
                Gather(W(8),out_,W(6).template ReinterpretCast<uint32_t>(),0U,count_);PipeBarrier<PIPE_V>();
            }
            const uint8_t repeats=count_/64;
            const UnaryRepeatParams unary{1,1,8,8};
            const BinaryRepeatParams repeatBound{1,1,1,8,8,0};
            Muls<float,false>(W(7),W(0),0.999935150146484375F,uint64_t(64),repeats,unary);
            Muls<float,false>(W(8),W(8),1.000064849853515625F,uint64_t(64),repeats,unary);
            Muls(W(6),zeroBound_[32],131072.0F,64);PipeBarrier<PIPE_V>();
            Sub<float,false>(W(7),W(7),W(6),uint64_t(64),repeats,repeatBound);
            Add<float,false>(W(8),W(8),W(6),uint64_t(64),repeats,repeatBound);PipeBarrier<PIPE_V>();
            Compare(possible,W(7),W(8),CMPMODE::LE,count_);
            if constexpr(MODE!=8)Compare(mask_,out_,zeroBound_[32],CMPMODE::LE,64);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if constexpr(MODE!=8) {
                const uint64_t live=validComputeRows_==64?~uint64_t(0):(uint64_t(1)<<validComputeRows_)-1;
                if(mask_.template ReinterpretCast<uint64_t>().GetValue(0)&live)needsExact_=true;
            }
            auto bits=possible.template ReinterpretCast<uint64_t>();
            const float normBound=zeroBound_[32].GetValue(0);
            if(!(normBound>=0.0F && normBound<=2097152.0F))needsExact_=true;
            CheckNumericalRiskGetValueStage(possible, bits);
        }
#endif
    }

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckNumericalRiskGetValueStage3() {
        if constexpr(MODE==8) {
                if(selectedCount_==1) {
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const float score=out_.GetValue(0),norm=C(9).GetValue(0);
                    if(!(score>=0.0F && score<=1152921504606846976.0F &&
                         norm>=0.0F && norm<=1152921504606846976.0F))needsExact_=true;
                    return true;
                }
            }
        return false;
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckNumericalRiskCompareScalarStage() {
        {
            CompareScalar(mask_,W(0),1152921504606846976.0F,CMPMODE::LE,count_);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            auto bits=mask_.template ReinterpretCast<uint64_t>();
            for(uint32_t first=0;first<validComputeRows_;first+=64) {
                const uint32_t count=validComputeRows_-first<64?validComputeRows_-first:64;
                const uint64_t active=count==64?~uint64_t(0):(uint64_t(1)<<count)-1;
                if((bits.GetValue(first/64)&active)!=active)needsExact_=true;
            }
            const float norm=C(9).GetValue(0);
            if(!(norm>=0.0F && norm<=1152921504606846976.0F))needsExact_=true;
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool FAST_FLOAT_DIFFERENCES, class StagePossibleType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckNumericalRiskMulsStage(StagePossibleType& possible) {
        if constexpr(MODE!=8) {
                // Enlarging both one-sided score intervals gives a conservative
                // upper threshold: the old candidate set is a subset of this one.
                Muls(W(7),out_,1.000244140625F,64);
                Muls(W(6),zeroBound_[32],262272.0F,64);PipeBarrier<PIPE_V>();
                Add(W(7),W(7),W(6),64);PipeBarrier<PIPE_V>();
                if constexpr(MODE==2) {
                    Brcb(W(8),W(7),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                } else if(stride_<=64) {
                    Brcb(W(6),W(7),(rows_+7U)/8U,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                    Copy(W(8),W(6),stride_,rows_,CopyRepeatParams(1,0,stride_/8U,1));PipeBarrier<PIPE_V>();
                } else {
                    MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,count_);PipeBarrier<PIPE_V>();
                    uint32_t shift=0;for(uint32_t n=stride_;n>1;n>>=1)++shift;
                    ShiftRight(W(6).template ReinterpretCast<uint32_t>(),W(6).template ReinterpretCast<uint32_t>(),shift,count_);PipeBarrier<PIPE_V>();
                    Muls(W(6).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),int32_t(4),count_);PipeBarrier<PIPE_V>();
                    Gather(W(8),W(7),W(6).template ReinterpretCast<uint32_t>(),0U,count_);PipeBarrier<PIPE_V>();
                }
                Compare(possible,W(0),W(8),CMPMODE::LE,count_);PipeBarrier<PIPE_V>();
                if(stride_==32 || stride_==64) {
                    if(!PackedRowsAreUnique<FAST_FLOAT_DIFFERENCES>(possible))needsExact_=true;
                    return true;
                }
                auto safe=zeroExact_[128].template ReinterpretCast<uint8_t>();
                if constexpr(FAST_FLOAT_DIFFERENCES) {
                    // Require squared distance > maximum segment norm / 4096.
                    // Near/cancellation-sensitive results rerun the unchanged
                    // compensated vector calculation before any GM output write.
                    Muls(W(8),zeroBound_[32],134217728.0F,64);PipeBarrier<PIPE_V>();
                    Compare(safe,out_,W(8),CMPMODE::GT,64);
                } else Compare(safe,out_,zeroBound_[32],CMPMODE::GT,64);
                if constexpr(sizeof(T)==4) {
                    CompareScalar(mask_,zeroBound_[32],2097152.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
                    And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint64_t live=validComputeRows_==64?~uint64_t(0):(uint64_t(1)<<validComputeRows_)-1;
                if((safe.template ReinterpretCast<uint64_t>().GetValue(0)&live)!=live)needsExact_=true;
                auto bits=possible.template ReinterpretCast<uint64_t>();
                CheckNumericalRiskGetValueStage2(bits);
                return true;
            }
        return false;
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageBitsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckNumericalRiskGetValueStage2(StageBitsType& bits) {
        if constexpr(MODE==2) {
            uint64_t counts=bits.GetValue(0);
            counts-=((counts>>1)&uint64_t(0x5555555555555555ULL));
            counts=(counts&uint64_t(0x3333333333333333ULL))+((counts>>2)&uint64_t(0x3333333333333333ULL));
            counts=(counts+(counts>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
            const uint64_t active=validComputeRows_==8?~uint64_t(0):(uint64_t(1)<<(validComputeRows_*8))-1;
            if(((counts^uint64_t(0x0101010101010101ULL))&active)!=0)needsExact_=true;
        } else {
            for(uint32_t row=0;row<validComputeRows_;++row) {
                uint32_t candidates=0;
                if(stride_<=64) {
                    const uint32_t first=row*stride_;
                    const uint64_t active=stride_==64?~uint64_t(0):(uint64_t(1)<<stride_)-1;
                    const uint64_t selected=(bits.GetValue(first/64)>>(first%64))&active;
                    candidates=selected!=0 && (selected&(selected-1))==0?1:0;
                } else for(uint32_t k=0;k<stride_/64;++k)candidates+=uint32_t(__builtin_popcountll(bits.GetValue(row*(stride_/64)+k)));
                if(candidates!=1)needsExact_=true;
            }
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StagePossibleType, class StageBitsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::CheckNumericalRiskGetValueStage(StagePossibleType& possible, StageBitsType& bits) {
        if constexpr(MODE==2) {
            uint64_t counts=bits.GetValue(0);
            counts-=((counts>>1)&uint64_t(0x5555555555555555ULL));
            counts=(counts&uint64_t(0x3333333333333333ULL))+((counts>>2)&uint64_t(0x3333333333333333ULL));
            counts=(counts+(counts>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
            const uint64_t active=validComputeRows_==8?~uint64_t(0):
                (uint64_t(1)<<(validComputeRows_*8))-1;
            if(((counts^uint64_t(0x0101010101010101ULL))&active)!=0)needsExact_=true;
        } else if constexpr(MODE==8) {
            uint64_t candidateCount=0;
            GatherMask(W(6),po_.template ReinterpretCast<float>(),possible.template ReinterpretCast<uint32_t>(),
                       true,count_,GatherMaskParams(1,1,8,1),candidateCount);
            PipeBarrier<PIPE_V>();
            if(candidateCount!=1)needsExact_=true;
        } else {
        for(uint32_t row=0;row<validComputeRows_;++row) {
            uint32_t candidates=0;
            if(stride_<=64) {
                const uint32_t first=row*stride_;
                const uint64_t active=stride_==64?~uint64_t(0):(uint64_t(1)<<stride_)-1;
                const uint64_t selected=(bits.GetValue(first/64)>>(first%64))&active;
                candidates=selected!=0 && (selected&(selected-1))==0?1:0;
            } else {
                for(uint32_t k=0;k<stride_/64;++k)
                    candidates+=uint32_t(__builtin_popcountll(bits.GetValue(row*(stride_/64)+k)));
            }
            if(candidates!=1)needsExact_=true;
        }
        }
    }
#endif

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareMetadata() {
        if constexpr (MODE == 3) {
            MakeIndices(po_,int32_t(0),count_);PipeBarrier<PIPE_V>();
            Muls(po_,po_,int32_t(12),count_);
        } else if constexpr(MODE == 4) {
            MakeIndices(so_,int32_t(0),64);PipeBarrier<PIPE_V>();
            if constexpr((!SINGLE_TILE || sizeof(T)==2) && !PACKED_HALF_ROW_OUTPUT)Muls(outputOffsets_,so_,int32_t(8),rows_);
            Muls(so_,so_,int32_t(24),64);
            const uint32_t high=t_.segments==64?~uint32_t(0):(uint32_t(1)<<(t_.segments-32))-1;
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),~uint32_t(0),count_/32);PipeBarrier<PIPE_V>();
            if(high!=~uint32_t(0)) {
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL & (count_/32==64?~uint64_t(0):((uint64_t(1)<<(count_/32))-1))));
                Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
                SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
            }
        } else if constexpr(MODE == 2) {
            SharedSegmentMetadata16();
        } else if constexpr(MODE==8) {
            CommonVectorStage56();
        } else { PrepareMetadataWStage(); }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareMetadataWStage() {
        {
        auto index = W(6).template ReinterpretCast<int32_t>();
        auto tmp = W(7).template ReinterpretCast<int32_t>();
        if constexpr(UsesTileCoefficientsStage()) { InitMakeIndicesStage(index, tmp); } else {
        SharedSegmentPadding9(index, tmp);
        }
        CommonVectorStage52(index);
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ constexpr bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::UsesTileCoefficientsStage() { return (sizeof(T)==4 || (TILE_COEFFICIENTS && sizeof(T)==2 && FIXED_STRIDE==32 && FIXED_ROWS<=8));
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::MakeIndices(const LocalTensor<int32_t>& values, int32_t first, uint32_t count) {
        if constexpr(UsesModeStage()) {
            // Input-independent gather metadata, with no scalar data-processing loop.
            auto seed=values.template ReinterpretCast<uint64_t>();
            seed.SetValue(0,(uint64_t(uint32_t(first+1))<<32)|uint32_t(first));
            seed.SetValue(1,(uint64_t(uint32_t(first+3))<<32)|uint32_t(first+2));
            seed.SetValue(2,(uint64_t(uint32_t(first+5))<<32)|uint32_t(first+4));
            seed.SetValue(3,(uint64_t(uint32_t(first+7))<<32)|uint32_t(first+6));
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(values, first, 8);
        CommonDyadicGeometry52(values);
        }
        for (uint32_t filled = 8; filled < count; filled *= 2) {
            const uint32_t n = count-filled < filled ? count-filled : filled;
            Adds(values[filled], values, static_cast<int32_t>(filled), n);
            PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ constexpr bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::UsesModeStage() { return ((MODE==1 && SINGLE_TILE && FIXED_ROWS==4 && FIXED_STRIDE==32 && SHORT_FP32_METADATA) || OWNED_AXES8 ||
            (sizeof(T)==2 && SINGLE_TILE && ((MODE==1 && FIXED_STRIDE==32) || (MODE==4 && FIXED_STRIDE==64))));
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template <class U>
    __aicore__ inline LocalTensor<U> KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::Allocate(uint32_t &address, uint32_t elements) {
        const uint32_t bytes = (elements * sizeof(U) + 31) / 32 * 32;
        LocalTensor<U> result(TPosition::VECCALC, address, bytes / sizeof(U));
        address += bytes;
        return result;
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline LocalTensor<float> KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::RawC(uint32_t col) { return columns_[col * (MODE==3?104:MODE==4?96:allocCount_)];
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline LocalTensor<float> KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::C(uint32_t col) {
        if constexpr(MAPPED_STARTS) {
            constexpr uint32_t columns[13]={6,7,8,3,4,5,0,1,2,9,10,11,12};
            return RawC(columns[col]);
        } else return columns_[col * (MODE==3?104:MODE==4?96:allocCount_)];
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline LocalTensor<float> KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::W(uint32_t col) { return work_[col * allocCount_];
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline LocalTensor<float> KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::P(uint32_t col) {
        if constexpr(DIRECT_HALF_ROW_POINTS || DIRECT_SINGLE_QUERY_POINTS)return pointSlots_[col*8];
        else return xyz_[col*allocCount_];
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::LoadSegmentInputs(
        uint32_t batch,uint32_t pointStart,uint32_t validRows) {
        if constexpr(MODE==3 && sizeof(T)==2) validPointRows_=validRows;
        if constexpr(MODE==3 || MODE==4 || SINGLE_TILE)count_=64;
        else if(stride_<=64 && count_>=(sizeof(T)==4?128:256))count_=64;
        Duplicate(rawS_, T(0), stride_ * 6);
        // A native Half32-row tile fits in the existing 160-element arena.
        // Zero padding to 128 coordinates keeps the range certificate unchanged.
        constexpr bool fixedHalfCoordinatePacket=MODE==4 && SINGLE_TILE && sizeof(T)==2 &&
            (FIXED_ROWS==0 || FIXED_ROWS==32);
        const uint32_t transferCount=fixedHalfCoordinatePacket?128U:(validRows*3+63)/64*64;
        Duplicate(rawP_,T(0),transferCount);
        if constexpr(SINGLE_TILE) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        } else PipeBarrier<PIPE_ALL>();
        DataCopyPad(rawS_, sg_[batch * t_.segments * 6],
                    DataCopyExtParams{1, t_.segments * 6 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, T(0)});
        DataCopyPad(rawP_,pg_[(batch*t_.points+pointStart)*3],
                        DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
        if constexpr(SINGLE_TILE)SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        return transferCount;
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::LoadSegments(uint32_t batch,uint32_t pointStart,uint32_t validRows,bool prepareMetadata) {
        const uint32_t expandedCount=count_;
        const uint32_t transferCount=LoadSegmentInputs(batch,pointStart,validRows);
        LoadSegmentsReinterpretCastStage(prepareMetadata, expandedCount);
        if constexpr(SINGLE_TILE)WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        else PipeBarrier<PIPE_ALL>();
        if constexpr(sizeof(T)==2)Cast(pf_,rawP_,RoundMode::CAST_NONE,transferCount);
        if constexpr (sizeof(T) == 2) Cast(sf_, rawS_, RoundMode::CAST_NONE, stride_ * 6);
        PipeBarrier<PIPE_V>();
        LoadSegmentsBrcbStage();
        PipeBarrier<PIPE_V>();
        if constexpr(MODE==4 && sizeof(T)==2)CheckExactDifferences(transferCount);
        if constexpr(TILE_COEFFICIENTS && sizeof(T)==2)CheckExactTileDifferences(transferCount);
        if constexpr(SINGLE_TILE)PrepareTileGeometry();else PrepareGeometry<MODE!=8>();
        LoadSegmentsCopyStage(expandedCount);
        if constexpr(MODE != 3)Brcb(pointSlots_,pf_,(validRows*3+7)/8,BrcbRepeatParams(1,8));
        if constexpr(SINGLE_TILE)PipeBarrier<PIPE_V>();else PipeBarrier<PIPE_ALL>();
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageExpandedCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::LoadSegmentsCopyStage(StageExpandedCountType& expandedCount) {
        if(count_!=expandedCount) {
            if constexpr(MODE != 3 && MODE != 4 && !TILE_COEFFICIENTS) {
        #pragma unroll
                for(uint32_t col=0;col<13;++col)
                    Copy(C(col),C(col),64,expandedCount/64,CopyRepeatParams(1,1,8,0));
            }
            PipeBarrier<PIPE_V>();
            count_=expandedCount;
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::LoadSegmentsBrcbStage() {
        if constexpr (MODE == 3) {
            Brcb(W(0),sf_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
    #pragma unroll
            for(uint32_t axis=0;axis<6;++axis)
                Copy(C(axis),W(0)[axis*8],64,count_/64,CopyRepeatParams(1,0,8,0));
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 6; ++axis)
                if constexpr(MAPPED_STARTS)
                    Gather(RawC(axis), sf_, so_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
                else Gather(C(axis), sf_, so_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
        }
    }
