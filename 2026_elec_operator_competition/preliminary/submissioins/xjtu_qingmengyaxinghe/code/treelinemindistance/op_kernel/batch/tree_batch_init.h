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

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::Init(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,const TreeLineMinDistanceTilingData& t) {
        t_=t;m_=CAPACITY;cp_=m_+32;wp_=ROWS*64+128;pp_=(m_>ROWS*8?m_:ROWS*8)+32;count_=COEFFICIENTS;
        pg_.SetGlobalBuffer((__gm__ T*)p);sg_.SetGlobalBuffer((__gm__ T*)s);
        dg_.SetGlobalBuffer((__gm__ T*)d);ig_.SetGlobalBuffer((__gm__ int32_t*)i);
        uint32_t address=0;
        columns_=Allocate<float>(address,14*cp_);xyz_=Allocate<float>(address,3*pp_);
        work_=Allocate<float>(address,6*wp_);
        sf_=work_[6*m_];
        if constexpr(sizeof(T)==4)rawS_=sf_.template ReinterpretCast<T>();
        else rawS_=work_.template ReinterpretCast<T>();
        so_=work_[13*m_].template ReinterpretCast<int32_t>();ones_=work_[12*m_];
        segmentIndices_=Allocate<float>(address,64);
        pf_=Allocate<float>(address,192);
        if constexpr(sizeof(T)==4)rawP_=pf_.template ReinterpretCast<T>();
        else rawP_=Allocate<T>(address,192);
        chunkLower_=Allocate<float>(address,1024);
        compactP_=Allocate<float>(address,192);refined_=Allocate<float>(address,13*64);
        record_=Allocate<float>(address,128);record2_=Allocate<float>(address,16);
        lo_=Allocate<float>(address,64);second_=Allocate<float>(address,64);hi_=Allocate<float>(address,64);
        idx_=Allocate<int32_t>(address,64);cert_=Allocate<float>(address,64);certHigh_=Allocate<float>(address,64);oneHot_=Allocate<int32_t>(address,64);
        tileLo_=Allocate<float>(address,64);tileSecond_=Allocate<float>(address,64);tileHi_=Allocate<float>(address,64);
        tileIdx_=Allocate<int32_t>(address,64);offset_=Allocate<int32_t>(address,64);
        rowBase_=Allocate<int32_t>(address,64);pointOffset_=Allocate<int32_t>(address,64);
        outputOffset_=Allocate<int32_t>(address,64);outT_=Allocate<T>(address,64);
        mask_=Allocate<uint8_t>(address,ROWS*8);valid_=Allocate<uint8_t>(address,ROWS*8);smallMask_=Allocate<uint8_t>(address,32);
        if constexpr(CAPACITY==256 && sizeof(T)==4)so_=Allocate<int32_t>(address,COEFFICIENTS);
        if constexpr(!DEFER_DIRECTION_LOW) {
            Duplicate(rawP_,T(0),192);Duplicate(compactP_,0.0F,192);
        }
        InitDuplicateStage();
        MakeIndices(offset_,0,64);
        InitShiftLeftStage();
        InitCStage();
        Muls(rowBase_,offset_,int32_t(256),64);Muls(pointOffset_,offset_,int32_t(12),64);
        Muls(outputOffset_,offset_,int32_t(8),64);InitCastStage();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::InitCastStage() {
        Cast(segmentIndices_,offset_,RoundMode::CAST_NONE,64);
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            // Persistent segment gather indices survive scratch reuse and batch changes.
            Muls(so_,offset_,int32_t(24),64);PipeBarrier<PIPE_V>();
    #pragma unroll
            for(uint32_t chunk=64;chunk<COEFFICIENTS;chunk+=64)
                Adds(so_[chunk],so_,int32_t(chunk*24),64);
        }
        Duplicate(tileLo_,__builtin_inff(),64);
        if constexpr(!PREFIX_NORMALIZATION)Duplicate(tileSecond_,__builtin_inff(),64);
        Duplicate(tileHi_,__builtin_inff(),64);
        Duplicate(tileIdx_,int32_t(0),64);
        if constexpr(UsesNativeHalfPrefixPrimaryStage())Duplicate(C(3)[CAPACITY+8],-2.0F,8);
        if constexpr(NATIVE_HALF_PREFIX_PRIMARY || NATIVE_FLOAT_PREFIX_SPARSE)Duplicate(xyz_[pp_-16],2.1250002384185791015625F,8);
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
            Duplicate(C(3)[CAPACITY+8],-2.0F,8);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(UsesRowsStage())
                Duplicate(C(4)[CAPACITY+8],-2.0F,8);
#endif
            Duplicate(xyz_[pp_-16],2.125F,8);
        }
        PipeBarrier<PIPE_ALL>();

    }

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesRowsStage() { return (ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL);
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesNativeHalfPrefixPrimaryStage() { return (NATIVE_HALF_PREFIX_PRIMARY || NATIVE_FLOAT_PREFIX_SPARSE || (CLEAN_PRIMARY64 && sizeof(T)==2));
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::InitDuplicateStage() {
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            // First-tile input buffers are independent of the index/state arena.
            Duplicate(rawS_,T(0),COEFFICIENTS*6);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            const uint32_t firstPosition=t_.batches*t_.points*GetBlockIdx()/t_.blocks;
            const uint32_t firstEnd=t_.batches*t_.points*(GetBlockIdx()+1)/t_.blocks;
            const uint32_t firstBatch=firstPosition/t_.points,firstPoint=firstPosition%t_.points;
            const uint32_t firstRows=Smallest(Smallest(firstEnd-firstPosition,t_.points-firstPoint),ROWS);
            DataCopyPad(rawS_,sg_[firstBatch*t_.segments*6],DataCopyExtParams{1,t_.segments*6*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            DataCopyPad(rawP_,pg_[firstPosition*3],DataCopyExtParams{1,firstRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            // Warm fixed-arena callees while the first input records are in flight.
            AscendC::PreLoadImpl(reinterpret_cast<void*>(&KernelTreeBatch::LoadBoundedResidentSegments),int64_t(1));
            AscendC::PreLoadImpl(reinterpret_cast<void*>(&KernelTreeBatch::PrimaryCompensated16),int64_t(1));
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::InitShiftLeftStage() {
        if constexpr(CAPACITY==1024) {
        ShiftLeft(oneHot_.template ReinterpretCast<uint32_t>(),offset_.template ReinterpretCast<uint32_t>(),uint32_t(23),32);PipeBarrier<PIPE_V>();
        Adds(oneHot_,oneHot_,int32_t(0x3f800000),32);PipeBarrier<PIPE_V>();
        Cast(oneHot_,oneHot_.template ReinterpretCast<float>(),RoundMode::CAST_RINT,31);PipeBarrier<PIPE_V>();
        SetMaskNorm();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x80000000ULL));
        Duplicate<int32_t,false>(oneHot_,int32_t(-2147483647-1),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            // Three coordinate padding gaps and dead refinement storage hold
            // four groups of four complete 64-row comparison predicates.
    #pragma unroll
            for(uint32_t group=0;group<4;++group) {
                const uint32_t base=group<3?uint32_t(xyz_[group*pp_+ROWS*8].GetPhyAddr()):uint32_t(refined_.GetPhyAddr());
                Duplicate(oneHot_[32+group*8],int32_t(base),8);
            }
        } else Copy(oneHot_[32],oneHot_,32,1,CopyRepeatParams(1,1,4,4));
        PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::InitCStage() {
        if constexpr(UsesRowsStage2()) {
            // Unused column padding preserves two aligned bitset rotations.
            auto rotate4=C(3)[CAPACITY+16].template ReinterpretCast<int32_t>();
            auto rotate2=C(3)[CAPACITY+24].template ReinterpretCast<int32_t>();
            Duplicate(tileLo_.template ReinterpretCast<int32_t>(),int32_t(7),8);
            Adds(rotate4,offset_,int32_t(4),8);
            Adds(rotate2,offset_,int32_t(2),8);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
            const BinaryRepeatParams bitsSame{1,1,1,8,8,8};
            And<uint16_t,false>(rotate4.template ReinterpretCast<uint16_t>(),
                rotate4.template ReinterpretCast<uint16_t>(),tileLo_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,bitsSame);
            And<uint16_t,false>(rotate2.template ReinterpretCast<uint16_t>(),
                rotate2.template ReinterpretCast<uint16_t>(),tileLo_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,bitsSame);
            PipeBarrier<PIPE_V>();
            Muls(rotate4,rotate4,int32_t(4),8);Muls(rotate2,rotate2,int32_t(4),8);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesRowsStage2() { return (CLEAN_PRIMARY64 && (sizeof(T)==4 || sizeof(T)==2) && ROWS==64);
    }

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SpatialSortPointCoordinatesSafe(uint32_t begin,uint32_t end) {
        if(end-begin<ROWS*8)return false;
        auto raw=work_,absolute=work_[4224];
        for(uint32_t coordinate=begin*3;coordinate<end*3;coordinate+=4096) {
            const uint32_t available=Smallest(end*3-coordinate,4096);
            const uint32_t aligned=(available+63)/64*64;
            if(aligned!=available)Duplicate(raw,0.0F,aligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            DataCopyPad(raw,pg_[coordinate],DataCopyExtParams{1,available*4,0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Abs(absolute,raw,aligned);PipeBarrier<PIPE_V>();
            WholeReduceMax(record_,absolute,64,aligned/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(tileLo_,record_,aligned/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            CompareScalar(smallMask_,tileLo_,1099511627776.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if((smallMask_.GetValue(0)&1U)==0) {PipeBarrier<PIPE_ALL>();return false;}
        }
        PipeBarrier<PIPE_ALL>();return true;
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CommitRefinedCandidate(uint32_t row) {
        Brcb(W(0),record_,1,BrcbRepeatParams(1,8));
        Brcb(W(1),tileIdx_.template ReinterpretCast<float>(),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(W(2),W(0),64,1,CopyRepeatParams(1,0,8,0));
        Copy(W(3),W(1),64,1,CopyRepeatParams(1,0,8,0));
        CompareScalar(smallMask_,segmentIndices_,float(int32_t(row)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetCmpMask(smallMask_);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,W(2),tileSecond_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),W(3),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
    }

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFullFloatFallbackBody(uint32_t row,uint32_t chunks) {
        static_assert(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full resource required");
                    uint32_t first=0,second=0,hits=0;
                    for(uint32_t chunk=0;chunk<(t_.segments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
                        if(hits==0)first=chunk;else if(hits==1)second=chunk;++hits;
                    }
                    const bool sparse=hits>0&&hits<=2;
                    const uint32_t candidates=sparse?hits*64:COEFFICIENTS;
                    auto searchColumns=columns_;const uint32_t searchCp=cp_;
                    RefineFullFloatFallbackBodyCopyStage(first, second, hits, sparse);
                    SharedCandidateProjection11(candidates, sparse, row, first, hits, second);
                    CompareScalar(valid_,W(7),float(int32_t(t_.segments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                    Compensated<true>();
                    SharedSparseProjection7(candidates, sparse);
                    const bool unique=CertifyWideScore(candidates);
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    CommitRefinedCandidate(row);
                    return unique;
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageFirstType, class StageSecondType, class StageHitsType, class StageSparseType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFullFloatFallbackBodyCopyStage(StageFirstType& first, StageSecondType& second, StageHitsType& hits, StageSparseType& sparse) {
        if(sparse) {
            auto selectedColumns=work_[16*1056];
    #pragma unroll
            for(uint32_t col=0;col<14;++col)if(col<6||col>9) {
                Copy(selectedColumns[col*128],C(col)[first*64],64,1,CopyRepeatParams(1,1,8,8));
                if(hits==2)Copy(selectedColumns[col*128+64],C(col)[second*64],64,1,CopyRepeatParams(1,1,8,8));
            }
            PipeBarrier<PIPE_V>();columns_=selectedColumns;cp_=128;
            Duplicate(record2_.template ReinterpretCast<int32_t>(),int32_t(first*64),8);
            Duplicate(record2_.template ReinterpretCast<int32_t>()[8],int32_t(second*64),8);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::InitFullFloatFrame(
        KernelTreeBatch& frame,uint32_t segmentCount) {
        frame.t_={};frame.t_.segments=segmentCount;
        frame.m_=1024U;frame.cp_=1056U;frame.pp_=1056U;frame.wp_=1056U;frame.count_=1024U;frame.liveRows_=64U;
        frame.directionLowReady_=false;
        frame.columns_=LocalTensor<float>(TPosition::VECCALC,0U,14784U);
        frame.xyz_=LocalTensor<float>(TPosition::VECCALC,59136U,3168U);
        frame.work_=LocalTensor<float>(TPosition::VECCALC,71808U,25344U);
        frame.segmentIndices_=LocalTensor<float>(TPosition::VECCALC,173184U,64U);
        frame.pf_=LocalTensor<float>(TPosition::VECCALC,173440U,192U);
        frame.chunkLower_=LocalTensor<float>(TPosition::VECCALC,174208U,1024U);
        frame.compactP_=LocalTensor<float>(TPosition::VECCALC,178304U,192U);
        frame.refined_=LocalTensor<float>(TPosition::VECCALC,179072U,832U);
        frame.record_=LocalTensor<float>(TPosition::VECCALC,182400U,128U);
        frame.record2_=LocalTensor<float>(TPosition::VECCALC,182912U,16U);
        frame.lo_=LocalTensor<float>(TPosition::VECCALC,182976U,64U);
        frame.second_=LocalTensor<float>(TPosition::VECCALC,183232U,64U);
        frame.hi_=LocalTensor<float>(TPosition::VECCALC,183488U,64U);
        frame.idx_=LocalTensor<int32_t>(TPosition::VECCALC,183744U,64U);
        frame.cert_=LocalTensor<float>(TPosition::VECCALC,184000U,64U);
        frame.certHigh_=LocalTensor<float>(TPosition::VECCALC,184256U,64U);
        frame.oneHot_=LocalTensor<int32_t>(TPosition::VECCALC,184512U,64U);
        frame.tileLo_=LocalTensor<float>(TPosition::VECCALC,184768U,64U);
        frame.tileSecond_=LocalTensor<float>(TPosition::VECCALC,185024U,64U);
        frame.tileHi_=LocalTensor<float>(TPosition::VECCALC,185280U,64U);
        frame.tileIdx_=LocalTensor<int32_t>(TPosition::VECCALC,185536U,64U);
        frame.offset_=LocalTensor<int32_t>(TPosition::VECCALC,185792U,64U);
        frame.rowBase_=LocalTensor<int32_t>(TPosition::VECCALC,186048U,64U);
        frame.pointOffset_=LocalTensor<int32_t>(TPosition::VECCALC,186304U,64U);
        frame.outputOffset_=LocalTensor<int32_t>(TPosition::VECCALC,186560U,64U);
        frame.outT_=LocalTensor<T>(TPosition::VECCALC,186816U,64U);
        frame.mask_=LocalTensor<uint8_t>(TPosition::VECCALC,187072U,512U);
        frame.valid_=LocalTensor<uint8_t>(TPosition::VECCALC,187584U,512U);
        frame.smallMask_=LocalTensor<uint8_t>(TPosition::VECCALC,188096U,32U);
        frame.sf_=frame.work_[6144];
        frame.rawS_=frame.sf_.template ReinterpretCast<T>();frame.rawP_=frame.pf_.template ReinterpretCast<T>();
        frame.so_=frame.work_[13312].template ReinterpretCast<int32_t>();frame.ones_=frame.work_[12288];
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFullFloatFallback(uint32_t row,uint32_t chunks,uint32_t segmentCount) {
        static_assert(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full resource required");
        KernelTreeBatch frame;
        InitFullFloatFrame(frame,segmentCount);
        return frame.RefineFullFloatFallbackBody(row,chunks);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::LoadFullFloatSegmentsStatic(GM_ADDR segments,uint32_t batch,uint32_t segmentCount) {
        static_assert(CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full resource required");
        static_assert(!DEFER_DIRECTION_LOW && !NATIVE_HALF_PREFIX_PRIMARY,"No mutable metadata is produced by this Float loader");
        KernelTreeBatch frame;
        InitFullFloatFrame(frame,segmentCount);
        frame.sg_.SetGlobalBuffer((__gm__ T*)segments);
        frame.LoadSegments(batch,false);
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineBoundedFullRow(
        uint32_t row,uint32_t chunks,uint32_t logicalSegments) {
        static_assert(DEFER_DIRECTION_LOW,"Bounded native Float cache required");
        KernelTreeBatch op;
        op.m_=CAPACITY;op.cp_=CAPACITY+32;op.wp_=ROWS*64+128;
        op.pp_=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;op.count_=COEFFICIENTS;
        op.t_.segments=logicalSegments;
        uint32_t address=0;
        op.columns_=op.template Allocate<float>(address,14*op.cp_);op.xyz_=op.template Allocate<float>(address,3*op.pp_);
        op.work_=op.template Allocate<float>(address,6*op.wp_);
        op.sf_=op.work_[6*op.m_];
        if constexpr(sizeof(T)==4)op.rawS_=op.sf_.template ReinterpretCast<T>();
        else op.rawS_=op.work_.template ReinterpretCast<T>();
        op.so_=op.work_[13*op.m_].template ReinterpretCast<int32_t>();op.ones_=op.work_[12*op.m_];
        op.segmentIndices_=op.template Allocate<float>(address,64);
        op.pf_=op.template Allocate<float>(address,192);
        if constexpr(sizeof(T)==4)op.rawP_=op.pf_.template ReinterpretCast<T>();
        else op.rawP_=op.template Allocate<T>(address,192);
        op.chunkLower_=op.template Allocate<float>(address,1024);
        op.compactP_=op.template Allocate<float>(address,192);op.refined_=op.template Allocate<float>(address,13*64);
        op.record_=op.template Allocate<float>(address,128);op.record2_=op.template Allocate<float>(address,16);
        op.lo_=op.template Allocate<float>(address,64);op.second_=op.template Allocate<float>(address,64);op.hi_=op.template Allocate<float>(address,64);
        op.idx_=op.template Allocate<int32_t>(address,64);op.cert_=op.template Allocate<float>(address,64);op.certHigh_=op.template Allocate<float>(address,64);op.oneHot_=op.template Allocate<int32_t>(address,64);
        op.tileLo_=op.template Allocate<float>(address,64);op.tileSecond_=op.template Allocate<float>(address,64);op.tileHi_=op.template Allocate<float>(address,64);
        op.tileIdx_=op.template Allocate<int32_t>(address,64);op.offset_=op.template Allocate<int32_t>(address,64);
        op.rowBase_=op.template Allocate<int32_t>(address,64);op.pointOffset_=op.template Allocate<int32_t>(address,64);
        op.outputOffset_=op.template Allocate<int32_t>(address,64);op.outT_=op.template Allocate<T>(address,64);
        op.mask_=op.template Allocate<uint8_t>(address,ROWS*8);op.valid_=op.template Allocate<uint8_t>(address,ROWS*8);op.smallMask_=op.template Allocate<uint8_t>(address,32);
        if constexpr(CAPACITY==256 && sizeof(T)==4)op.so_=op.template Allocate<int32_t>(address,COEFFICIENTS);
        op.wp_=CAPACITY+32;
        return op.RefineBoundedFullRowBody(row,chunks,logicalSegments);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineBoundedFullRowBody(
        uint32_t row,uint32_t chunks,uint32_t logicalSegments) {
        static_assert(DEFER_DIRECTION_LOW,"Bounded native Float cache required");
                    uint32_t first=0,second=0,hits=0;
                    for(uint32_t chunk=0;chunk<(logicalSegments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
                        if(hits==0)first=chunk;else if(hits==1)second=chunk;++hits;
                    }
                    const bool sparse=hits>0&&hits<=2;
                    const uint32_t candidates=sparse?hits*64:COEFFICIENTS;
                    auto searchColumns=columns_;const uint32_t searchCp=cp_;
                    CommonVectorStage54(first, second, hits, sparse, candidates, row, logicalSegments);
                    const bool unique=CertifyWideScore(candidates);
                    columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
                    CommitRefinedCandidate(row);
                    return unique;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RestartBoundedAfterExact() {
        static_assert(DEFER_DIRECTION_LOW || NATIVE_FLOAT_PREFIX_SPARSE,"Resident native Float cache required");
        const TreeLineMinDistanceTilingData t=t_;
        Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
             (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RestartBoundedInlineAfterExact() {
        static_assert(DEFER_DIRECTION_LOW || NATIVE_FLOAT_PREFIX_SPARSE,"Resident native Float cache required");
        const TreeLineMinDistanceTilingData t=t_;
        Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
             (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::Process(uint32_t rangeBegin,uint32_t rangeEnd,GM_ADDR spatialWorkspace,bool spatialPointCoordinatesSafe) {
        const uint32_t total=t_.batches*t_.points;
        const uint32_t begin=rangeEnd?rangeBegin:total*GetBlockIdx()/t_.blocks,end=rangeEnd?rangeEnd:total*(GetBlockIdx()+1)/t_.blocks;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        bool spatialCoefficientDirty=false,spatialPruned=false;
        uint32_t spatialDirtyCount=COEFFICIENTS;
        GlobalTensor<float> spatialCoefficients;
        if(spatialWorkspace)spatialCoefficients.SetGlobalBuffer((__gm__ float*)spatialWorkspace+6*4096);
#endif
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        bool corePointSafe=false,sharedCoefficientCoordinatesSafe=false;
        if constexpr(UsesRowsStage3())
            // Sorting has a wider range domain than the numerical certificate.
            // Validate each sorted full tile before using the shared bound.
            corePointSafe=spatialWorkspace?false:CertifyCorePointCoordinates(begin,end);
#endif
        uint32_t cached=0xffffffffU;
        bool prefetchedPoint=false,outputInFlight=false;
        ProcessSmallestStage2<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, spatialPointCoordinatesSafe, begin, end, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, spatialCoefficients, corePointSafe, sharedCoefficientCoordinatesSafe, cached, prefetchedPoint, outputInFlight);
    }

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesRowsStage3() { return (CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL);
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageBeginType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSpatialCoefficientsType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessSmallestStage2(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageBeginType& begin, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSpatialCoefficientsType& spatialCoefficients, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight) {
        for(uint32_t position=begin;position<end;) {
            const uint32_t batch=position/t_.points,start=position%t_.points;
            const uint32_t validRows=Smallest(Smallest(end-position,t_.points-start),ROWS);
            uint32_t logicalSegments=t_.segments;
            liveRows_=validRows;
            bool exactPointSafe=corePointSafe;
            count_=COEFFICIENTS;
            bool initialInputs=false;
            if constexpr(CAPACITY==256 && sizeof(T)==4)initialInputs=cached==0xffffffffU && position==begin;
            ProcessDefinedStage3<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, spatialPointCoordinatesSafe, begin, end, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, spatialCoefficients, corePointSafe, sharedCoefficientCoordinatesSafe, cached, prefetchedPoint, outputInFlight, position, batch, start, validRows, logicalSegments, exactPointSafe, initialInputs);}
        if constexpr(OVERLAP_POINT_INPUT) {
            if(outputInFlight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
            PipeBarrier<PIPE_ALL>();
        }

    }
