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
// Inline implementation of KernelTreeBatch.

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageCompactP_Type, class StageRecord_Type, class StageTileHi_Type, class StageThresholdType, class StageBroadcastType, class StageQueryRadiusType, class StageSavedCenterScoresType, class StageOtherCenterScoresType, class StagePredicateType, class StageWitnessType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateSetMaskNormStage2(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageCompactP_Type& compactP_, StageRecord_Type& record_, StageTileHi_Type& tileHi_, StageThresholdType& threshold, StageBroadcastType& broadcast, StageQueryRadiusType& queryRadius, StageSavedCenterScoresType& savedCenterScores, StageOtherCenterScoresType& otherCenterScores, StagePredicateType& predicate, StageWitnessType& witness) {
        auto witnessError=work_[14336];
        constexpr uint64_t witnessLanes=64;
        // Matrix rows are16 witnesses; columns are all64 real queries.
        const BinaryRepeatParams witnessSame{1,1,1,8,8,8};
        const BinaryRepeatParams witnessSegment{1,1,0,8,8,1};
        const BinaryRepeatParams witnessPoint{1,1,0,8,0,1};
        const UnaryRepeatParams witnessUnary{1,1,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(witnessLanes);
        Sub<float,false>(SpatialPruneW(0),compactP_[0*64],witness[0*128],witnessLanes,16,witnessPoint);
        Sub<float,false>(SpatialPruneW(1),compactP_[1*64],witness[1*128],witnessLanes,16,witnessPoint);
        Mul<float,false>(SpatialPruneW(4),SpatialPruneW(0),SpatialPruneW(0),witnessLanes,16,witnessSame);
        Mul<float,false>(SpatialPruneW(3),SpatialPruneW(0),witness[3*128],witnessLanes,16,witnessSegment);
        Sub<float,false>(SpatialPruneW(2),compactP_[2*64],witness[2*128],witnessLanes,16,witnessPoint);
        MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(1),SpatialPruneW(1),witnessLanes,16,witnessSame);
        MulAddDst<float,float,false>(SpatialPruneW(3),SpatialPruneW(1),witness[4*128],witnessLanes,16,witnessSegment);
        MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(2),SpatialPruneW(2),witnessLanes,16,witnessSame);
        FusedMulAdd<float,false>(SpatialPruneW(2),witness[5*128],SpatialPruneW(3),witnessLanes,16,BinaryRepeatParams{1,0,1,8,1,8});
        Maxs<float,false>(SpatialPruneW(3),SpatialPruneW(2),0.0F,witnessLanes,16,witnessUnary);
        Min<float,false>(SpatialPruneW(3),SpatialPruneW(3),witness[6*128],witnessLanes,16,witnessSegment);
        Muls<float,false>(witnessError,SpatialPruneW(4),1.52587890625e-5F,witnessLanes,16,witnessUnary);
        FusedMulAdd<float,false>(SpatialPruneW(2),SpatialStateC(4)[CAPACITY+8],SpatialPruneW(3),witnessLanes,16,BinaryRepeatParams{1,0,1,8,0,8});
        MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(3),SpatialPruneW(2),witnessLanes,16,witnessSame);PipeBarrier<PIPE_V>();
        Maxs<float,false>(SpatialPruneW(4),SpatialPruneW(4),0.0F,witnessLanes,16,witnessUnary);PipeBarrier<PIPE_V>();
        Muls<float,false>(witnessError,witnessError,2.25F,witnessLanes,16,witnessUnary);PipeBarrier<PIPE_V>();
        Add<float,false>(SpatialPruneW(4),SpatialPruneW(4),witnessError,witnessLanes,16,witnessSame);PipeBarrier<PIPE_V>();
        // Fold witness rows while retaining the64 query coordinates.
        Min<float,false>(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(4)[512],witnessLanes,8,witnessSame);PipeBarrier<PIPE_V>();
        Min<float,false>(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(4)[256],witnessLanes,4,witnessSame);PipeBarrier<PIPE_V>();
        Min<float,false>(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(4)[128],witnessLanes,2,witnessSame);PipeBarrier<PIPE_V>();
        Min<float,false>(tileHi_,SpatialPruneW(4),SpatialPruneW(4)[64],witnessLanes,1,witnessSame);PipeBarrier<PIPE_V>();
        Maxs(tileHi_,tileHi_,0.0F,64);PipeBarrier<PIPE_V>();
        Sqrt(tileHi_,tileHi_,64);PipeBarrier<PIPE_V>();
        Add(tileHi_,tileHi_,queryRadius,64);PipeBarrier<PIPE_V>();
        Muls(tileHi_,tileHi_,1.0000152587890625F,64);PipeBarrier<PIPE_V>();
        Mul(tileHi_,tileHi_,tileHi_,64);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,tileHi_,32,2,8,1,4,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Copy(threshold,record_,16,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        Brcb(broadcast,threshold,2,BrcbRepeatParams(1,8));return PruneSpatialStaticStateReinterpretCastStage(spatialIndices, segments, pruned, logicalSegments, work_, broadcast, savedCenterScores, otherCenterScores, predicate);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageBroadcastType, class StageSavedCenterScoresType, class StageOtherCenterScoresType, class StagePredicateType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateReinterpretCastStage(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageBroadcastType& broadcast, StageSavedCenterScoresType& savedCenterScores, StageOtherCenterScoresType& otherCenterScores, StagePredicateType& predicate) {
        PipeBarrier<PIPE_V>();
        auto otherPredicate=work_[10080].template ReinterpretCast<uint8_t>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(predicate,savedCenterScores,broadcast,CMPMODE::LE,uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});
        Compare<float,uint8_t,false>(otherPredicate,otherCenterScores,broadcast[64],CMPMODE::LE,uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
        Or(predicate.template ReinterpretCast<uint16_t>(),predicate.template ReinterpretCast<uint16_t>(),otherPredicate.template ReinterpretCast<uint16_t>(),COEFFICIENTS/16);PipeBarrier<PIPE_V>();
        // No ordinal coefficient is forced into the surviving set.
        SpatialPruneIndices(SpatialPruneW(0).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Cast(SpatialPruneW(1),SpatialPruneW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);PipeBarrier<PIPE_V>();
        auto genuinePredicate=work_[10080].template ReinterpretCast<uint8_t>();
        CompareScalar(genuinePredicate,SpatialPruneW(1),float(int32_t(segments)),CMPMODE::LT,COEFFICIENTS);PipeBarrier<PIPE_V>();
        And(predicate.template ReinterpretCast<uint16_t>(),predicate.template ReinterpretCast<uint16_t>(),genuinePredicate.template ReinterpretCast<uint16_t>(),COEFFICIENTS/16);PipeBarrier<PIPE_V>();
        auto packed=work_[8000];auto offsets=work_[9024].template ReinterpretCast<int32_t>();
        uint64_t kept=0;
        GatherMask(packed,SpatialPruneW(1),predicate.template ReinterpretCast<uint32_t>(),true,COEFFICIENTS,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>=960) {return COEFFICIENTS;}
        logicalSegments=uint32_t(kept);
        // Every padded coefficient duplicates the final genuine retained segment.
        // Stable packing preserves the original argmin tie order.
        Mins(SpatialPruneW(1),SpatialPruneW(1),float(int32_t(kept-1)),COEFFICIENTS);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(1),SpatialPruneW(1),4.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Cast(offsets,SpatialPruneW(1),RoundMode::CAST_RINT,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Gather(SpatialPruneW(0),packed,offsets.template ReinterpretCast<uint32_t>(),0U,COEFFICIENTS);PipeBarrier<PIPE_V>();
        auto originalIndices=packed.template ReinterpretCast<int32_t>();
        Cast(originalIndices,SpatialPruneW(0),RoundMode::CAST_RINT,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Muls(offsets,originalIndices,int32_t(4),COEFFICIENTS);PipeBarrier<PIPE_V>();
        // Preserve the inverse map before reusing the dead packed-ID arena.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // The inverse map survives search/refinement above their UB arenas.
        const LocalTensor<int32_t> pendingIndices(TPosition::VECCALC,192512U,1024U);
        Copy(pendingIndices,originalIndices,64,(uint32_t(kept)+63U)/64U,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
#else
        auto pendingIndices=work_[5*(64*64+128)].template ReinterpretCast<int32_t>();
        Copy(pendingIndices,originalIndices,64,COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
#endif
        // Every live search chunk is gathered once. The remaining payload
        // duplicates the final retained coefficient exactly, as before.
        const uint32_t compacted=(uint32_t(kept)+63U)/64U*64U;
        for(uint32_t col=0;col<14;++col) {
            const uint32_t stage=col<8?col*1024U:10240U+(col-8U)*1024U;
            Gather(work_[stage],SpatialStateC(col),offsets.template ReinterpretCast<uint32_t>(),0U,compacted);
        }
        return PruneSpatialStaticStateDefinedStage3(spatialIndices, pruned, work_, kept, pendingIndices, compacted);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StagePrunedType, class StageWork_Type, class StageKeptType, class StagePendingIndicesType, class StageCompactedType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateDefinedStage3(GlobalTensor<int32_t>& spatialIndices, StagePrunedType& pruned, StageWork_Type& work_, StageKeptType& kept, StagePendingIndicesType& pendingIndices, StageCompactedType& compacted) {
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        for(uint32_t col=0;col<14;++col) {
            const uint32_t stage=col<8?col*1024U:10240U+(col-8U)*1024U;
            Copy(SpatialStateC(col),work_[stage],64,compacted/64U,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
#else
        for(uint32_t col=0;col<14;++col) {
            const uint32_t stage=col<8?col*1024U:10240U+(col-8U)*1024U;
            Brcb(work_[16384U+col*64U],work_[stage+compacted-8U],1,BrcbRepeatParams(1,8));
            Copy(SpatialStateC(col),work_[stage],64,compacted/64U,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t col=0;col<14;++col)
            Copy(SpatialStateC(col)[compacted],work_[16384U+col*64U+56U],64,(COEFFICIENTS-compacted)/64U,CopyRepeatParams(1,0,8,0));
        PipeBarrier<PIPE_V>();
#endif
#if !defined(DTYPE_TREE_POINTS) || !TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(spatialIndices,pendingIndices,COEFFICIENTS);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
#endif
        pruned=true;
        return (uint32_t(kept)+63)/64*64;


    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#else
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageWork_Type, class StageCenterBroadcastType, class StageCenterErrorBroadcastType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateSetMaskNormStage(StageWork_Type& work_, StageCenterBroadcastType& centerBroadcast, StageCenterErrorBroadcastType& centerErrorBroadcast) {
        for(uint32_t sphere=0;sphere<2;++sphere) {
            // Center lower scores select genuine witnesses and filter candidates.
            // Center-to-segment scores use the retained13-instruction dot/norm order.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            for(uint32_t axis=0;axis<3;++axis)
                Sub<float,false>(SpatialPruneW(axis),centerBroadcast[(2*axis+sphere)*64],SpatialStateC(axis),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
            PipeBarrier<PIPE_V>();
            Mul(SpatialPruneW(4),SpatialPruneW(0),SpatialPruneW(0),COEFFICIENTS);Mul(SpatialPruneW(3),SpatialPruneW(0),SpatialStateC(6),COEFFICIENTS);PipeBarrier<PIPE_V>();
            MulAddDst(SpatialPruneW(4),SpatialPruneW(1),SpatialPruneW(1),COEFFICIENTS);MulAddDst(SpatialPruneW(3),SpatialPruneW(1),SpatialStateC(7),COEFFICIENTS);PipeBarrier<PIPE_V>();
            MulAddDst(SpatialPruneW(4),SpatialPruneW(2),SpatialPruneW(2),COEFFICIENTS);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(SpatialPruneW(2),SpatialStateC(8),SpatialPruneW(3),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Maxs(SpatialPruneW(3),SpatialPruneW(2),0.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();Min(SpatialPruneW(3),SpatialPruneW(3),SpatialStateC(9),COEFFICIENTS);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(SpatialPruneW(2),SpatialStateC(4)[CAPACITY+8],SpatialPruneW(3),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            MulAddDst(SpatialPruneW(4),SpatialPruneW(3),SpatialPruneW(2),COEFFICIENTS);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Sub<float,false>(SpatialPruneW(4),SpatialPruneW(4),centerErrorBroadcast[sphere*64],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            Copy(work_[12288+sphere*1024],SpatialPruneW(4),64,COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        }
    }
#endif
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::MakeIndices(const LocalTensor<int32_t>& values, int32_t first, uint32_t count) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(UsesNativeFloatPrefixSparseStage()) {
            // The following masked additions consume the complete seed packet.
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
            Duplicate<int32_t,false>(values,first,uint64_t(64),1,1,8);
            PipeBarrier<PIPE_V>();
        } else
#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(sizeof(T)==4) {
            // The promoted arena may contain sorted pairs. Initialize all eight
            // seeds and finish the write before the masked read-modify-writes.
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
            Duplicate<int32_t,false>(values,first,uint64_t(64),1,1,8);
            PipeBarrier<PIPE_V>();
        } else
#endif
        Duplicate(values, first, 8);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        if constexpr(CAPACITY==256 && sizeof(T)==4)PipeBarrier<PIPE_V>();
#endif
        CommonDyadicGeometry52(values);
        for (uint32_t filled = 8; filled < count; filled *= 2) {
            const uint32_t n = count-filled < filled ? count-filled : filled;
            Adds(values[filled], values, static_cast<int32_t>(filled), n);
            PipeBarrier<PIPE_V>();
        }
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesNativeFloatPrefixSparseStage() { return (NATIVE_FLOAT_PREFIX_SPARSE || (CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL));
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template <class U>
    __aicore__ inline LocalTensor<U> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::Allocate(uint32_t &address, uint32_t elements) {
        const uint32_t bytes = (elements * sizeof(U) + 31) / 32 * 32;
        LocalTensor<U> result(TPosition::VECCALC, address, bytes / sizeof(U));
        address += bytes;
        return result;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::MaterializeDirectionLow() {
        if constexpr(DEFER_DIRECTION_LOW) {
            if(directionLowReady_)return;
            // Coarse and selected scratch is dead here; cache once per batch.
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub(FewWork(axis),C(axis+3),C(axis),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub(FewWork(axis+3),C(axis+3),FewWork(axis),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Add(FewWork(axis+6),FewWork(axis),FewWork(axis+3),COEFFICIENTS);
                Sub(C(axis+10),FewWork(axis+3),C(axis),COEFFICIENTS);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub(FewWork(axis+6),C(axis+3),FewWork(axis+6),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Add(C(axis+10),C(axis+10),FewWork(axis+6),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
            directionLowReady_=true;
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::LoadPrefixResidentSegments(
        GM_ADDR source,uint32_t segmentCount) {
        static_assert(NATIVE_FLOAT_PREFIX_SPARSE,"Fixed native Float prefix arena required");
        AscendC::ICachePreLoad(int64_t(2));
        constexpr uint32_t pitch=CAPACITY+32;
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*pitch+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        // Reconstruct only the handles read by the unchanged coefficient loader.
        // The local object never escapes this function, so its fields fold to constants.
        KernelTreeBatch op;
        SharedResidentCoefficients19<pitch, pointPitch, workBase>(op, segmentCount, source);
        op.so_=op.work_[13*CAPACITY].template ReinterpretCast<int32_t>();op.ones_=op.work_[12*CAPACITY];
        op.rowBase_=LocalTensor<int32_t>(TPosition::VECCALC,(stateBase+128+16+12*64)*4,64);
        op.mask_=LocalTensor<uint8_t>(TPosition::VECCALC,(stateBase+128+16+16*64)*4,ROWS*8);
        op.LoadSegments(0,false);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::LoadBoundedResidentSegments(
        GM_ADDR source,uint32_t segmentCount,bool initialInputs) {
        static_assert(DEFER_DIRECTION_LOW && ROWS==64,"Fixed native Float bounded arena required");
        constexpr uint32_t pitch=CAPACITY+32;
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*pitch+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        KernelTreeBatch op;
        SharedResidentCoefficients19<pitch, pointPitch, workBase>(op, segmentCount, source);
        op.ones_=op.work_[12*CAPACITY];
        op.so_=LocalTensor<int32_t>(TPosition::VECCALC,(stateBase+128+16+16*64+ROWS*4+8)*4,COEFFICIENTS);
        op.rowBase_=LocalTensor<int32_t>(TPosition::VECCALC,(stateBase+128+16+12*64)*4,64);
        op.mask_=LocalTensor<uint8_t>(TPosition::VECCALC,(stateBase+128+16+16*64)*4,ROWS*8);
        op.LoadSegments(0,initialInputs);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::LoadResidentSegments(uint32_t batch,bool initialInputs) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(2));
        static_assert(DEFER_DIRECTION_LOW || NATIVE_FLOAT_PREFIX_SPARSE,"Resident native Float cache required");
        LoadSegments(batch,initialInputs);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::LoadSegments(uint32_t batch,bool initialInputs) {
        if constexpr(DEFER_DIRECTION_LOW)directionLowReady_=false;
        if constexpr(!(CAPACITY==256 && sizeof(T)==4)) {
            ShiftRight(so_,rowBase_,int32_t(8),64);PipeBarrier<PIPE_V>();
            for(uint32_t filled=64;filled<COEFFICIENTS;filled*=2) {
                const uint32_t n=COEFFICIENTS-filled<filled?COEFFICIENTS-filled:filled;
                Adds(so_[filled],so_,int32_t(filled),n);PipeBarrier<PIPE_V>();
            }
            Muls(so_,so_,int32_t(24),COEFFICIENTS);
        }
        if(!initialInputs) {
            Duplicate(rawS_, T(0), COEFFICIENTS * 6);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(rawS_, sg_[batch * t_.segments * 6],
                        DataCopyExtParams{1, t_.segments * 6 * uint32_t(sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>{false, 0, 0, T(0)});
            PipeBarrier<PIPE_ALL>();
        } else PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == 2) Cast(sf_, rawS_, RoundMode::CAST_NONE, COEFFICIENTS * 6);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 6; ++axis)
            Gather(C(axis), sf_, so_.template ReinterpretCast<uint32_t>(), axis * 4, count_);
        PipeBarrier<PIPE_V>();
        LoadSegmentsWStage();
        LoadSegmentsSetMaskNormStage();
        CompareScalar(mask_, C(9), 0.0F, CMPMODE::GT, count_);
        PipeBarrier<PIPE_V>();
        Select(C(9), mask_, C(9), 1.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count_);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE || DEFER_DIRECTION_LOW)PipeBarrier<PIPE_V>();
#endif
        NormalizeDirections();
        if constexpr((CLEAN_PRIMARY64 && sizeof(T)==2) || NATIVE_HALF_PREFIX_PRIMARY) {
            Abs(W(0),C(10),COEFFICIENTS);
            Abs(W(1),C(11),COEFFICIENTS);
            Abs(W(2),C(12),COEFFICIENTS);PipeBarrier<PIPE_V>();
            Max(W(0),W(0),W(1),COEFFICIENTS);PipeBarrier<PIPE_V>();
            Max(W(0),W(0),W(2),COEFFICIENTS);PipeBarrier<PIPE_V>();
            WholeReduceMax(record_,W(0),64,COEFFICIENTS/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(tileSecond_,record_,COEFFICIENTS/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            exactDirectionLow_=tileSecond_.template ReinterpretCast<uint32_t>().GetValue(0)==0;
        }
        PipeBarrier<PIPE_ALL>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::LoadSegmentsWStage() {
        if constexpr (true) {
            // A segment and its reversed representation must follow identical
            // floating-point operations, including the exact-index tie rule.
            auto keep = W(8).template ReinterpretCast<uint8_t>();
            auto equal = W(0).template ReinterpretCast<uint8_t>();
            auto less = W(1).template ReinterpretCast<uint8_t>();
            auto equalY = W(2).template ReinterpretCast<uint8_t>();
            CompareEndpointOrder(keep, equal, less, equalY, C(0), C(1), C(2), C(3), C(4), C(5), count_);
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis)
                Select(W(axis), keep, C(axis), C(axis+3), SELMODE::VSEL_TENSOR_TENSOR_MODE, count_);
            PipeBarrier<PIPE_V>();
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis)
                Select(C(axis+3), keep, C(axis+3), C(axis), SELMODE::VSEL_TENSOR_TENSOR_MODE, count_);
            PipeBarrier<PIPE_V>();
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Adds(C(axis), W(axis), 0.0F, count_);
            PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::LoadSegmentsSetMaskNormStage() {
        if constexpr(DEFER_DIRECTION_LOW) {
            constexpr uint64_t lanes=64;
            constexpr uint8_t repeats=COEFFICIENTS/64;
            const BinaryRepeatParams same{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(lanes);
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub<float,false>(C(axis+6),C(axis+3),C(axis),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
            Mul<float,false>(C(9),C(6),C(6),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(C(9),C(7),C(7),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(C(9),C(8),C(8),lanes,repeats,same);
            PipeBarrier<PIPE_V>();
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub(C(axis+6), C(axis+3), C(axis), count_);
        PipeBarrier<PIPE_V>();
        if constexpr(!DEFER_DIRECTION_LOW) {
    CompensateDirectionPlanes(W(0), W(1), W(2), W(3), W(4), W(5), C(0), C(1), C(2), C(3), C(4), C(5), C(6), C(7), C(8), C(10), C(11), C(12), count_);
        }
        Mul(C(9), C(6), C(6), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(7), C(7), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(8), C(8), count_);
        PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::NormalizeDirections() {
        if constexpr(PREFIX_NORMALIZATION) {
            Adds(C(13),C(9),0.0F,COEFFICIENTS);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE || DEFER_DIRECTION_LOW)PipeBarrier<PIPE_V>();
#endif
            Sqrt(C(9),C(9),COEFFICIENTS);
            Duplicate(ones_,1.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
            Div(W(0),ones_,C(9),COEFFICIENTS);PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Mul(C(axis+6),C(axis+6),W(0),COEFFICIENTS);
            PipeBarrier<PIPE_V>();
        } else {
        Adds(W(0),C(9),0.0F,COEFFICIENTS);
        Sqrt(C(9),C(9),COEFFICIENTS);
        Duplicate(ones_,1.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
        Div(C(13),ones_,C(9),COEFFICIENTS);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Mul(C(axis+6),C(axis+6),C(13),COEFFICIENTS);
        PipeBarrier<PIPE_V>();
        Adds(C(13),W(0),0.0F,COEFFICIENTS);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryW(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(workBase+k*96)*4,96);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryC(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t refinedBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192;
        return LocalTensor<float>(TPosition::VECCALC,(refinedBase+k*64)*4,64);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryP(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t pointBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024;
        return LocalTensor<float>(TPosition::VECCALC,(pointBase+k*64)*4,64);
    }
