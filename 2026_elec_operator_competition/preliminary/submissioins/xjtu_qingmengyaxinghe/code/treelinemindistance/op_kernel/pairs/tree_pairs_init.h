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
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData &t) {
        t_ = t;
        exactDifferences_=false;needsExact_=false;
        stride_ = FIXED_STRIDE ? FIXED_STRIDE : MODE == 2 ? 8 : MODE == 4 ? 64 : t.paddedSegments;
        rows_ = FIXED_ROWS ? FIXED_ROWS : MODE == 2 ? 8 : MODE == 4 ? 32 : MODE == 8 ? 1 : t.pointTile;
        count_ = stride_ * rows_;
        allocCount_ = count_ < 64 ? 64 : count_;
        count_ = allocCount_;
        group_ = (MODE == 4 || MODE == 8) ? 1 : stride_ > 64 ? 32 : stride_ == 64 ? 16 : 1;
        allocCount_ += 32;
        pg_.SetGlobalBuffer((__gm__ T *)p);
        sg_.SetGlobalBuffer((__gm__ T *)s);
        dg_.SetGlobalBuffer((__gm__ T *)d);
        ig_.SetGlobalBuffer((__gm__ int32_t *)i);
        uint32_t address = 0;
        sf_ = Allocate<float>(address, stride_ * 6);
        if constexpr(sizeof(T)==4)rawS_=sf_.template ReinterpretCast<T>();
        else rawS_=Allocate<T>(address,stride_*6);
        columns_ = Allocate<float>(address, (MODE==3?104:MODE==4?96:allocCount_) * 13);
        pf_ = Allocate<float>(address, rows_ * group_ * 3 + 64);
        if constexpr(sizeof(T)==4)rawP_=pf_.template ReinterpretCast<T>();
        else rawP_=Allocate<T>(address,rows_*group_*3+64);
        InitAllocateStage(address);
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageAddressType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::InitAllocateStage(StageAddressType& address) {
        if constexpr (MODE != 3) pointSlots_ = Allocate<float>(address, (rows_ * group_ * 3 + 64) * 8);
        xyz_ = Allocate<float>(address, allocCount_ * 3);
        work_ = Allocate<float>(address, allocCount_ * 9);
        po_ = Allocate<int32_t>(address, allocCount_);
        so_ = Allocate<int32_t>(address, allocCount_);
        mask_ = Allocate<uint8_t>(address, (allocCount_ + 255) / 256 * 32);
        valid_ = Allocate<uint8_t>(address, (allocCount_ + 255) / 256 * 32);
        records_ = Allocate<float>(address, rows_ * 8 + 64);
        reduceWork_ = Allocate<float>(address, allocCount_);
        outputOffsets_ = Allocate<int32_t>(address, rows_ + 64);
        out_ = Allocate<float>(address, rows_ + 64);
        stageIdx_ = Allocate<int32_t>(address, ((rows_ + 7) / 8 * 8) * group_);
        outIdx_ = stageIdx_;
        stageOut_ = Allocate<T>(address, ((rows_ * sizeof(T) + 31) / 32 * 32 / sizeof(T)) * group_);
        outT_ = stageOut_;
        if constexpr(MODE!=3 || sizeof(T)==4) {
            reduceOffsets_ = Allocate<int32_t>(address, rows_ * 16 + 64);
            reduceBases_ = Allocate<int32_t>(address, rows_ + 64);
        }
        if constexpr(MODE!=3 || sizeof(T)==2) {
            zeroBound_=Allocate<float>(address,96);
            zeroExact_=Allocate<float>(address,45*64);
        }
        InitMakeIndicesStage2();

    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::InitMakeIndicesStage2() {
        if constexpr(MODE==3 || (MODE==4 && !SINGLE_TILE)) {
        if constexpr (MODE == 3) {
            // Point gather offsets are prepared while the first input DMA is in flight.
        } else if constexpr(MODE == 4) {
            MakeIndices(so_,int32_t(0),64);PipeBarrier<PIPE_V>();
            if constexpr((!SINGLE_TILE || sizeof(T)==2) && !PACKED_HALF_ROW_OUTPUT)Muls(outputOffsets_,so_,int32_t(8),rows_);
            Muls(so_,so_,int32_t(24),64);
            const uint32_t high=t_.segments==64?~uint32_t(0):(uint32_t(1)<<(t_.segments-32))-1;
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),~uint32_t(0),count_/32);PipeBarrier<PIPE_V>();
            if(high!=~uint32_t(0)) {
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL &
                    (count_/32==64?~uint64_t(0):((uint64_t(1)<<(count_/32))-1))));
                Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
                SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
            }
        } else if constexpr(MODE == 2) {
            SharedSegmentMetadata16();
        } else if constexpr(MODE==8) {
            CommonVectorStage56();
        } else { InitWStage(); }
            PipeBarrier<PIPE_ALL>();
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::InitWStage() {
        {
        auto index = W(6).template ReinterpretCast<int32_t>();
        auto tmp = W(7).template ReinterpretCast<int32_t>();
        if constexpr(sizeof(T)==4) { InitMakeIndicesStage(index, tmp); } else {
        SharedSegmentPadding9(index, tmp);
        }
        CommonVectorStage52(index);
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageIndexType, class StageTmpType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::InitMakeIndicesStage(StageIndexType& index, StageTmpType& tmp) {
        {
        if(stride_<=64) {
            MakeIndices(so_,int32_t(0),stride_);PipeBarrier<PIPE_V>();
            if constexpr((!SINGLE_TILE || sizeof(T)==2) && !PACKED_HALF_ROW_OUTPUT)Muls(outputOffsets_,so_,int32_t(8),rows_);
            if(count_>stride_)Copy(so_[stride_],so_,stride_,count_/stride_-1,CopyRepeatParams(1,1,stride_/8,0));
            PipeBarrier<PIPE_V>();
            Muls(so_,so_,int32_t(24),count_);
            if(stride_<=32) {
                uint32_t bits=t_.segments==32?~uint32_t(0):(uint32_t(1)<<t_.segments)-1;
                if(stride_==8)bits*=0x01010101U;
                else if(stride_==16)bits|=bits<<16;
                Duplicate(valid_.template ReinterpretCast<uint32_t>(),bits,count_/32);
            } else {
                const uint32_t high=t_.segments==64?~uint32_t(0):(uint32_t(1)<<(t_.segments-32))-1;
                Duplicate(valid_.template ReinterpretCast<uint32_t>(),~uint32_t(0),count_/32);PipeBarrier<PIPE_V>();
                if(high!=~uint32_t(0)) {
                    SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL &
                        (count_/32==64?~uint64_t(0):((uint64_t(1)<<(count_/32))-1))));
                    Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
                    SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                }
            }
        } else {
            MakeIndices(index,int32_t(0),count_);PipeBarrier<PIPE_V>();
            ShiftRight(po_.template ReinterpretCast<uint32_t>(),index.template ReinterpretCast<uint32_t>(),t_.segmentLog,count_);PipeBarrier<PIPE_V>();
            Muls(tmp,po_,static_cast<int32_t>(stride_),count_);PipeBarrier<PIPE_V>();
            Sub(so_,index,tmp,count_);PipeBarrier<PIPE_V>();
            Cast(W(8),so_,RoundMode::CAST_NONE,count_);
            Muls(so_,so_,int32_t(24),count_);Muls(po_,po_,int32_t(12),count_);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(8),static_cast<float>(static_cast<int32_t>(t_.segments)),CMPMODE::LT,count_);
            Muls(outputOffsets_,index,int32_t(8),rows_);
        }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::Process() {
        if constexpr(SINGLE_TILE) { ProcessGetBlockIdxStage(); } else if constexpr(MODE==8) {
            const uint32_t batch=GetBlockIdx();
            validComputeRows_=1;needsExact_=false;
            LoadSegments(batch,0,1,true);
            if constexpr(!DIRECT_SINGLE_QUERY_POINTS) {
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                if(stride_<=64)Copy(P(axis),pointSlots_[axis*8],stride_,1,CopyRepeatParams(1,0,8,0));
                else Copy(P(axis),pointSlots_[axis*8],64,stride_/64,CopyRepeatParams(1,0,8,0));
            }
            }
            PipeBarrier<PIPE_V>();SelectCandidates();
            if constexpr(sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
                && sizeof(DTYPE_TREE_POINTS)==2
#endif
            ) {
                if(count_==64)Compute<false,true>();
                else Compute();
            } else Compute();
            DataCopyPad(dg_[batch],outT_,DataCopyExtParams{1,uint32_t(sizeof(T)),0,0,0});
            DataCopyPad(ig_[batch],outIdx_,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
            if(needsExact_)RepairRange(batch,0,1);
        } else { ProcessGetBlockIdxStage2(); }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ProcessGetBlockIdxStage2() {
        {
        const uint32_t tiles = (t_.points + rows_ - 1) / rows_;
        const uint32_t total = t_.batches * tiles;
        const uint32_t begin = total * GetBlockIdx() / t_.blocks;
        const uint32_t end = total * (GetBlockIdx() + 1) / t_.blocks;
        uint32_t cached = 0xffffffffU;
        for (uint32_t task = begin; task < end;) {
            const uint32_t batch = task / tiles;
            const uint32_t start = task % tiles * rows_;
            uint32_t groups = end - task < group_ ? end - task : group_;
            if (groups > tiles - task % tiles) groups = tiles - task % tiles;
            const uint32_t validRows = t_.points - start < rows_ * groups ? t_.points - start : rows_ * groups;
            needsExact_=false;
            const bool changed = batch != cached;
            if(changed) { LoadSegments(batch,start,validRows,cached==0xffffffffU);cached=batch; }
            if(!changed)LoadPoints(batch,start,validRows);
            const uint32_t distanceStride = (rows_ * sizeof(T) + 31) / 32 * 32 / sizeof(T);
            const uint32_t indexStride = (rows_ + 7) / 8 * 8;
            ProcessGatherMaskStage(groups, validRows, distanceStride, indexStride);
            const uint32_t base = batch * t_.points + start;
            const uint32_t full = validRows / rows_;
            const uint32_t tail = validRows % rows_;
            if (full) {
                DataCopyPad(dg_[base], stageOut_, DataCopyExtParams{static_cast<uint16_t>(full), rows_ * uint32_t(sizeof(T)), 0, 0, 0});
                DataCopyPad(ig_[base], stageIdx_, DataCopyExtParams{static_cast<uint16_t>(full), rows_ * 4, 0, 0, 0});
            }
            if (tail) {
                DataCopyPad(dg_[base + full * rows_], stageOut_[full * distanceStride], DataCopyExtParams{1, tail * uint32_t(sizeof(T)), 0, 0, 0});
                DataCopyPad(ig_[base + full * rows_], stageIdx_[full * indexStride], DataCopyExtParams{1, tail * 4, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
            if(needsExact_) {
                RepairRange(batch,start,validRows);
                if(task+groups<end) {
                    const TreeLineMinDistanceTilingData t=t_;
                    Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
                         (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
                    cached=0xffffffffU;
                }
            }
            task += groups;
        }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageGroupsType, class StageValidRowsType, class StageDistanceStrideType, class StageIndexStrideType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ProcessGatherMaskStage(StageGroupsType& groups, StageValidRowsType& validRows, StageDistanceStrideType& distanceStride, StageIndexStrideType& indexStride) {
        for (uint32_t slot = 0; slot < groups; ++slot) {
            const uint32_t offset=slot*rows_;
            validComputeRows_=validRows-offset<rows_?validRows-offset:rows_;
            outT_ = stageOut_[slot * distanceStride];
            outIdx_ = stageIdx_[slot * indexStride];
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
                    uint64_t selected=0;
                    GatherMask(P(axis),pf_,po_[axis*(3U*FIXED_ROWS/32U)].template ReinterpretCast<uint32_t>(),
                        true,3U*count_,GatherMaskParams(1,1,8,1),selected);
                } else if constexpr (MODE == 3) {
                    Gather(P(axis), pf_, po_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
                } else if (stride_ <= 64) {
                    Copy(P(axis), pointSlots_[(slot * rows_ * 3 + axis) * 8],
                         stride_, rows_, CopyRepeatParams(1, 0, stride_ / 8, 3));
                } else {
                    for (uint32_t row = 0; row < rows_; ++row)
                        Copy(P(axis)[row * stride_], pointSlots_[((slot * rows_ + row) * 3 + axis) * 8],
                             64, stride_ / 64, CopyRepeatParams(1, 0, 8, 0));
                }
            }
            PipeBarrier<PIPE_V>();
            if constexpr(MODE==4 && sizeof(T)==2) {
                if(exactDifferences_)Compute<true>();else Compute();
            } else Compute();
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ProcessGetBlockIdxStage() {
        {
            const uint32_t tiles=(t_.points+rows_-1)/rows_;
            const uint32_t batch=GetBlockIdx()/tiles;
            const uint32_t start=(GetBlockIdx()-batch*tiles)*rows_;
            const uint32_t validRows=t_.points-start<rows_?t_.points-start:rows_;
            validComputeRows_=validRows;needsExact_=false;
            LoadSegments(batch,start,validRows,true);
            if constexpr(!DIRECT_HALF_ROW_POINTS) {
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis)
                    Copy(P(axis),pointSlots_[axis*8],stride_,rows_,CopyRepeatParams(1,0,stride_/8,3));
                PipeBarrier<PIPE_V>();
            }
            if constexpr((TILE_COEFFICIENTS || MODE==4) && sizeof(T)==2) {
                if(exactDifferences_)Compute<true>();else Compute<false>();
            } else if constexpr(SHORT_FP32_METADATA && SINGLE_TILE && MODE==2) {
                Compute<true>();
                if(needsExact_) {
                    needsExact_=false;
                    Compute<false>();
                }
            } else Compute();
            DataCopyPad(dg_[batch*t_.points+start],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
            DataCopyPad(ig_[batch*t_.points+start],outIdx_,DataCopyExtParams{1,validRows*4,0,0,0});
            if(needsExact_) {
                // The exact repair overwrites the same GM interval. Ordinary
                // single-tile drains have no subsequent UB writer or GM overwrite.
                SetFlag<HardEvent::MTE3_S>(EVENT_ID3);WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
                RepairRange(batch,start,validRows);
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::RepairRange(uint32_t batch,uint32_t start,uint32_t rows) {
#if defined(DTYPE_TREE_POINTS)
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=0;row<rows;++row) {
            const uint32_t pos=batch*t_.points+start+row;
            TreeGeneralExactFallback<T>((GM_ADDR)pg_[pos*3].GetPhyAddr(),
                (GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),
                (GM_ADDR)dg_[pos].GetPhyAddr(),(GM_ADDR)ig_[pos].GetPhyAddr(),t_.segments);
        }
#endif
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool FAST_FLOAT_DIFFERENCES>
    __aicore__ inline bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PackedRowsAreUnique(LocalTensor<uint8_t> possible) {
        if constexpr(PACKED_HALF_ROW_OUTPUT) {
            // Only the first eight predicate words represent this resident arena.
            // Keep one normal vector mask across arithmetic and comparisons.
            auto safe=zeroExact_[128].template ReinterpretCast<uint8_t>();
            auto combined=possible.template ReinterpretCast<int32_t>();
            const UnaryRepeatParams unary{1,1,8,8};
            const BinaryRepeatParams binary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Adds<int32_t,false>(W(6).template ReinterpretCast<int32_t>(),combined,int32_t(-1),uint64_t(64),1,unary);
            PipeBarrier<PIPE_V>();
            And<uint16_t,false>(W(6).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),combined.template ReinterpretCast<uint16_t>(),uint64_t(64),1,binary);
            PipeBarrier<PIPE_V>();
            CompareScalar<int32_t,uint8_t,false>(safe,W(6).template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,uint64_t(64),1,unary);
            Compare<float,uint8_t,false>(mask_,out_,zeroBound_[32],CMPMODE::GT,uint64_t(64),1,binary);
            PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(4));
            And<uint16_t,false>(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),uint64_t(4),1,binary);
            PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t live=(uint64_t(1)<<validComputeRows_)-1U;
            return (safe.template ReinterpretCast<uint64_t>().GetValue(0)&live)==live;
        } else { bool packedRowsAreUniqueReinterpretCastStageResult{};
        if (PackedRowsAreUniqueReinterpretCastStage<FAST_FLOAT_DIFFERENCES>(possible, packedRowsAreUniqueReinterpretCastStageResult)) return packedRowsAreUniqueReinterpretCastStageResult; }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool FAST_FLOAT_DIFFERENCES, class StagePossibleType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PackedRowsAreUniqueReinterpretCastStage(StagePossibleType& possible, bool& stageResult) {
        {
        auto source=possible.template ReinterpretCast<int32_t>();
        auto combined=source;
        if(stride_==64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            GatherMask(W(6),possible.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(W(7),possible.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Or(W(8).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),W(7).template ReinterpretCast<uint16_t>(),128);
            And(W(7).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),W(7).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            combined=W(8).template ReinterpretCast<int32_t>();
        }
        Adds(W(6).template ReinterpretCast<int32_t>(),combined,int32_t(-1),64);PipeBarrier<PIPE_V>();
        And(W(6).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),combined.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        if(stride_==64) {
            Or(W(6).template ReinterpretCast<uint16_t>(),W(6).template ReinterpretCast<uint16_t>(),W(7).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        auto safe=zeroExact_[128].template ReinterpretCast<uint8_t>();
        CompareScalar(safe,W(6).template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);
        if constexpr(sizeof(T)==4) {
            CompareScalar(mask_,combined,int32_t(0),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            Not(mask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        if constexpr(FAST_FLOAT_DIFFERENCES) {
            Muls(W(7),zeroBound_[32],134217728.0F,64);PipeBarrier<PIPE_V>();
            Compare(mask_,out_,W(7),CMPMODE::GT,64);
        } else Compare(mask_,out_,zeroBound_[32],CMPMODE::GT,64);
        PipeBarrier<PIPE_V>();
        And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        // Finite binary16 endpoints satisfy norm <= 3*(2*65504)^2,
        // hence norm*2^-39 < 1. The wider float domain still needs its check.
        if constexpr(sizeof(T)==4) {
            CompareScalar(mask_,zeroBound_[32],2097152.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(safe.template ReinterpretCast<uint16_t>(),safe.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t live=validComputeRows_==64?~uint64_t(0):(uint64_t(1)<<validComputeRows_)-1;
        { stageResult = (safe.template ReinterpretCast<uint64_t>().GetValue(0)&live)==live; return true; }
            }
        return false;
    }
