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
    template<class StageMasksType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfEndpointPacketTiesEndpointEncodeStage(StageMasksType& masks) {
#pragma unroll

        for(uint32_t axis=0;axis<3;++axis) {
            EndpointEncode(FixedFewP(axis),9);EndpointEncode(EndpointF(6+axis),12);
            Sub(EndpointI(31),EndpointI(9),EndpointI(12),64);PipeBarrier<PIPE_V>();EndpointSplit(15);
            Sub(EndpointI(31),EndpointI(10),EndpointI(13),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(16);
            Sub(EndpointI(17),EndpointI(11),EndpointI(14),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(17),EndpointI(17),EndpointI(30),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(31),EndpointI(15),int32_t(-1),64);PipeBarrier<PIPE_V>();EndpointSplit(18);
            Muls(EndpointI(31),EndpointI(16),int32_t(-1),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(19);
            Muls(EndpointI(20),EndpointI(17),int32_t(-1),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(20),EndpointI(20),EndpointI(30),64);
            Cast(EndpointF(38),EndpointI(17),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(masks[192],EndpointF(38),0.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
    #pragma unroll
            for(uint32_t k=0;k<3;++k)
                Select(EndpointF(21+k),masks[192],EndpointF(18+k),EndpointF(15+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
            PipeBarrier<PIPE_V>();
            Mul(EndpointI(31),EndpointI(21),EndpointI(21),64);PipeBarrier<PIPE_V>();EndpointSplit(24);
            Mul(EndpointI(31),EndpointI(21),EndpointI(22),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(31),EndpointI(31),int32_t(2),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(25);
            Mul(EndpointI(31),EndpointI(22),EndpointI(22),64);
            Mul(EndpointI(37),EndpointI(21),EndpointI(23),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(37),EndpointI(37),int32_t(2),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(37),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(26);
            Mul(EndpointI(31),EndpointI(22),EndpointI(23),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(31),EndpointI(31),int32_t(2),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(27);
            Mul(EndpointI(31),EndpointI(23),EndpointI(23),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(28);
            Copy(EndpointI(29),EndpointI(30),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            Duplicate(EndpointI(30),int32_t(0),64);PipeBarrier<PIPE_V>();
    #pragma unroll 1
            for(uint32_t k=0;k<6;++k) {
                Add(EndpointI(31),EndpointI(k),EndpointI(24+k),64);PipeBarrier<PIPE_V>();
                Add(EndpointI(31),EndpointI(31),EndpointI(30),64);PipeBarrier<PIPE_V>();EndpointSplit(k);
            }
        }
    }

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ResolveExactPacketBody(
        uint64_t validLanes,uint32_t pending) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        const LocalTensor<float> laneIds(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<float> packetState(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr()),1024);
        const LocalTensor<float> backup(TPosition::VECCALC,(workBase+20*1056)*4,1728);
        uint64_t representable=~uint64_t(0);
        // Both encodings are selected only after testing every active coordinate.
        const auto certificate=backup[4*64].template ReinterpretCast<uint8_t>();
        const auto shortCertificate=backup[5*64].template ReinterpretCast<uint8_t>();
        Duplicate(certificate.template ReinterpretCast<uint16_t>(),uint16_t(65535),4);
        Duplicate(shortCertificate.template ReinterpretCast<uint16_t>(),uint16_t(65535),4);
        Duplicate(backup[7*64].template ReinterpretCast<uint32_t>(),uint32_t(0x7fffffffU),64);PipeBarrier<PIPE_V>();
                ResolveExactPacketBodyFixedFewPStage(backup, certificate, shortCertificate);
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
#if TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        representable=certificate.template ReinterpretCast<uint64_t>().GetValue(0);
#endif
        const uint64_t shortBits=shortCertificate.template ReinterpretCast<uint64_t>().GetValue(0);
        uint32_t resolved=0;
        uint64_t exactLanes=0,groupMask=0;
        for(uint32_t group=0;group<8;++group) {
            const uint64_t active=validLanes&(uint64_t(255)<<(8*group));
            if((pending&(1U<<group)) && active && (active&representable)==active) {
                resolved|=1U<<group;
                exactLanes|=active;
                groupMask|=uint64_t(255)<<(8*group);
            }
        }
        if(!resolved)return 0;
        Copy(backup,packetState,64,16,CopyRepeatParams(1,1,8,8));
        Copy(backup[1024],FixedFewW(0),64,1,CopyRepeatParams(1,1,8,8));
        #pragma unroll 1
        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            Copy(backup[1088+coordinate*64],x,64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        const auto activeMask=backup[1664].template ReinterpretCast<uint8_t>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        activeMask.template ReinterpretCast<uint64_t>().SetValue(0,exactLanes);
        return ResolveExactPacketBodyFixedFewPStage2(laneIds, packetState, backup, shortBits, resolved, exactLanes, groupMask, activeMask);
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageLaneIdsType, class StagePacketStateType, class StageBackupType, class StageShortBitsType, class StageResolvedType, class StageExactLanesType, class StageGroupMaskType, class StageActiveMaskType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ResolveExactPacketBodyFixedFewPStage2(StageLaneIdsType& laneIds, StagePacketStateType& packetState, StageBackupType& backup, StageShortBitsType& shortBits, StageResolvedType& resolved, StageExactLanesType& exactLanes, StageGroupMaskType& groupMask, StageActiveMaskType& activeMask) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;

        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
#pragma unroll 1
        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            Select(x,activeMask,x,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        }
        PipeBarrier<PIPE_V>();
        const LocalTensor<float> arena(TPosition::VECCALC,uint32_t(FewWork(0).GetPhyAddr()),224*64);
        ResolveExactPacketBodyExactStage<pointPitch>(laneIds, backup, shortBits, exactLanes, arena);
        const LocalTensor<uint8_t> predicates(TPosition::VECCALC,uint32_t(arena[220*64].GetPhyAddr()),512);
        Compare(predicates,backup,arena[96*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Select(arena[216*64],predicates,backup[1024],__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        predicates[32].template ReinterpretCast<uint64_t>().SetValue(0,groupMask);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        Select(FixedFewW(0),predicates[32],arena[216*64],backup[1024],SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        Copy(packetState,backup,64,16,CopyRepeatParams(1,1,8,8));
#pragma unroll 1
        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            Copy(x,backup[1088+coordinate*64],64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        return resolved;

    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto pointPitch, class StageLaneIdsType, class StageBackupType, class StageShortBitsType, class StageExactLanesType, class StageArenaType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ResolveExactPacketBodyExactStage(StageLaneIdsType& laneIds, StageBackupType& backup, StageShortBitsType& shortBits, StageExactLanesType& exactLanes, StageArenaType& arena) {
        if((exactLanes&shortBits)==exactLanes) {
            TreePacketIntegerRational64<> exact(uint32_t(arena.GetPhyAddr()));exact.Init();
            exact.template MakeScores<true>(0,uint32_t(FixedFewP(0).GetPhyAddr()),pointPitch,
                uint32_t(FixedFewC(0).GetPhyAddr()),uint32_t(laneIds.GetPhyAddr()),CAPACITY,
                uint32_t(backup.GetPhyAddr()),exactLanes);
            exact.template Tournament<3>();
        } else {
            TreePacketRational64<> exact(uint32_t(arena.GetPhyAddr()));exact.Init();
            exact.template MakeScores<true>(0,uint32_t(FixedFewP(0).GetPhyAddr()),pointPitch,
                uint32_t(FixedFewC(0).GetPhyAddr()),uint32_t(laneIds.GetPhyAddr()),CAPACITY,
                uint32_t(backup.GetPhyAddr()),exactLanes);
            exact.template Tournament<3>();
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageBackupType, class StageCertificateType, class StageShortCertificateType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ResolveExactPacketBodyFixedFewPStage(StageBackupType& backup, StageCertificateType& certificate, StageShortCertificateType& shortCertificate) {
#pragma unroll

        for(uint32_t coordinate=0;coordinate<9;++coordinate) {
            const auto x=coordinate<3?FixedFewP(coordinate):FixedFewC(coordinate-3);
            And(backup[6*64].template ReinterpretCast<uint16_t>(),x.template ReinterpretCast<uint16_t>(),
                backup[7*64].template ReinterpretCast<uint16_t>(),128);
            Abs(backup[3*64],x,64);PipeBarrier<PIPE_V>();
            CompareScalar(certificate[64],backup[3*64],128.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            CompareScalar(certificate[96],backup[6*64].template ReinterpretCast<int32_t>(),int32_t(0),CMPMODE::EQ,64);
            CompareScalar(shortCertificate[64],backup[3*64],9.5367431640625e-7F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            Or(shortCertificate[64].template ReinterpretCast<uint16_t>(),shortCertificate[64].template ReinterpretCast<uint16_t>(),
                certificate[96].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Select(backup[0],certificate[64],x,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            Muls(backup[0],backup[0],1048576.0F,64);PipeBarrier<PIPE_V>();
            Cast(backup[64].template ReinterpretCast<int32_t>(),backup[0],RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
            Cast(backup[2*64],backup[64].template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Compare(shortCertificate[32],backup[0],backup[2*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate.template ReinterpretCast<uint16_t>(),certificate[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate.template ReinterpretCast<uint16_t>(),shortCertificate[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
#if TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            CompareScalar(certificate[128],backup[3*64],5.960464477539063e-8F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            Or(certificate[128].template ReinterpretCast<uint16_t>(),certificate[128].template ReinterpretCast<uint16_t>(),
                certificate[96].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Muls(backup[0],backup[0],16.0F,64);PipeBarrier<PIPE_V>();
            Cast(backup[64].template ReinterpretCast<int32_t>(),backup[0],RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
            Cast(backup[2*64],backup[64].template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Compare(certificate[32],backup[0],backup[2*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(certificate.template ReinterpretCast<uint16_t>(),certificate.template ReinterpretCast<uint16_t>(),certificate[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certificate.template ReinterpretCast<uint16_t>(),certificate.template ReinterpretCast<uint16_t>(),certificate[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certificate.template ReinterpretCast<uint16_t>(),certificate.template ReinterpretCast<uint16_t>(),certificate[128].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
#endif
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ResolveExactPacket(
        uint64_t validLanes,uint32_t safeGroups) {
        uint32_t pending=0;
        for(uint32_t group=0;group<8;++group)
            if(((validLanes>>(group*8))&255U) && !(safeGroups&(1U<<group)))pending|=1U<<group;
        if(!pending)return 0;

        return ResolveExactPacketBody(validLanes,pending);
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ResolveExactSingle(
        uint32_t kept,LocalTensor<float> candidates,LocalTensor<float> record,
        LocalTensor<int32_t> index,LocalTensor<int32_t> offset,LocalTensor<int32_t> outputOffset) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        const LocalTensor<float> saved(TPosition::VECCALC,(workBase+20*1056+1696)*4,8);
        Copy(saved,BatchIds(),8,1,CopyRepeatParams(1,1,1,1));
        Copy(BatchIds(),candidates,8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        const bool resolved=(ResolveExactPacket((uint64_t(1)<<kept)-1,0)&1U)!=0;
        Copy(BatchIds(),saved,8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
        if(!resolved)return false;
        WholeReduceMin(record,FixedFewW(0),8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
        Gather(offset,record.template ReinterpretCast<int32_t>(),outputOffset.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
        Muls(offset,offset,int32_t(4),1);PipeBarrier<PIPE_V>();
        Gather(index.template ReinterpretCast<float>(),candidates,offset.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
        Cast(index,index.template ReinterpretCast<float>(),RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
        return true;
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectPendingCollinearGroups(
        uint64_t validLanes,uint32_t safeGroups,uint32_t groups) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        const LocalTensor<uint8_t> valid(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        bool touched=false;
        for(uint32_t group=0;group<groups;++group)if(!(safeGroups&(1U<<group))) {
            const uint32_t bits=uint32_t(validLanes>>(group*8))&255U;
            if(bits && (bits&(bits-1)) && MayHaveCollinearInteriorTies8NormalMask(
                uint32_t(FixedFewW(0)[group*8].GetPhyAddr()),
                uint32_t(FixedFewC(0)[group*8].GetPhyAddr()),128,
                uint32_t(FewWork(0)[20*1056].GetPhyAddr()),uint32_t(valid.GetPhyAddr())-ROWS*8)) {
                CorrectCollinearInteriorTies8(uint32_t(FixedFewW(0)[group*8].GetPhyAddr()),
                    uint32_t(FixedFewP(0)[group*8].GetPhyAddr()),pointPitch,
                    uint32_t(FixedFewC(0)[group*8].GetPhyAddr()),128,
                    uint32_t(FewWork(0)[20*1056].GetPhyAddr()),uint32_t(valid.GetPhyAddr())-ROWS*8);
                touched=true;
            }
        }
        return touched;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineCandidateBatch(uint64_t validLanes,uint64_t rows,uint64_t& uniqueRows) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(1));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> compactP_(TPosition::VECCALC,(stateBase-13*64-192)*4,192);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<float> record_(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<float> tileSecond_(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        const LocalTensor<float> distances(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+160*4,8);
        const LocalTensor<int32_t> indices(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+192*4,8);
        Cast(FewWork(7).template ReinterpretCast<int32_t>(),BatchIds(),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        Muls(FewWork(7).template ReinterpretCast<int32_t>(),FewWork(7).template ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t col=0;col<14;++col)if(col<6||col>9)
            Gather(FixedFewC(col),FewCoefficients(col),FewWork(7).template ReinterpretCast<uint32_t>(),0U,64);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Gather(FewWork(axis),compactP_[axis*64],BatchPointIds(),0U,8);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Brcb(FixedFewP(axis),FewWork(axis),1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        valid_.template ReinterpretCast<uint64_t>().SetValue(0,validLanes);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        if(!BatchFewCompensated())return false;
        // Eight independent8-lane minima preserve each point's increasing global indices.
        WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS)
        RefineCandidateBatchGatherStage(validLanes, rows, uniqueRows, outputOffset_, record_, distances);
#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        bool refineCandidateBatchEndpointIStageResult{};
        if (RefineCandidateBatchEndpointIStage(validLanes, rows, uniqueRows, outputOffset_, record_, idx_, tileSecond_, smallMask_, distances, indices, refineCandidateBatchEndpointIStageResult)) return refineCandidateBatchEndpointIStageResult;
#endif
        Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);
        Gather(FewWork(0).template ReinterpretCast<int32_t>(),record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,8);PipeBarrier<PIPE_V>();
        Muls(FewWork(0).template ReinterpretCast<int32_t>(),FewWork(0).template ReinterpretCast<int32_t>(),int32_t(4),8);
        Muls(FewWork(1).template ReinterpretCast<int32_t>(),outputOffset_,int32_t(4),8);PipeBarrier<PIPE_V>();
        Add(FewWork(0).template ReinterpretCast<int32_t>(),FewWork(0).template ReinterpretCast<int32_t>(),FewWork(1).template ReinterpretCast<int32_t>(),8);PipeBarrier<PIPE_V>();
        Gather(indices.template ReinterpretCast<float>(),BatchIds(),FewWork(0).template ReinterpretCast<uint32_t>(),0U,8);return RefineCandidateBatchCastStage(rows, idx_, tileSecond_, smallMask_, distances, indices);
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageValidLanesType, class StageRowsType, class StageUniqueRowsType, class StageOutputOffset_Type, class StageRecord_Type, class StageIdx_Type, class StageTileSecond_Type, class StageSmallMask_Type, class StageDistancesType, class StageIndicesType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineCandidateBatchEndpointIStage(StageValidLanesType& validLanes, StageRowsType& rows, StageUniqueRowsType& uniqueRows, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageSmallMask_Type& smallMask_, StageDistancesType& distances, StageIndicesType& indices, bool& stageResult) {
        if constexpr(NATIVE_HALF_PREFIX_PRIMARY) {
            const auto localOffset=EndpointI(61);
            const auto packetOffset=EndpointI(62);
            const auto mappedDistance=EndpointF(63);
            const auto mappedIndex=EndpointI(64);
            uint32_t safeGroups=0,pendingGroups=0;
            for(uint32_t group=0;group<8;++group) {
                const uint32_t active=uint32_t(validLanes>>(group*8))&255U;
                if(active && !(active&(active-1U)))safeGroups|=1U<<group;
                else if(active)pendingGroups|=1U<<group;
            }
            if(pendingGroups) {
                Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
                safeGroups|=CertifyRefined(distances,validLanes);
                const uint32_t exactGroups=ResolveExactPacket(validLanes,safeGroups);
                if(exactGroups) {
                    safeGroups|=exactGroups;
                    WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                }
            }
            for(uint32_t group=0;group<8;++group)if(safeGroups&(1U<<group)) {
                const uint32_t row=BatchPointIds().GetValue(group)/4;
                uniqueRows|=(uint64_t(1)<<row)&rows;
            }
            Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);
            Gather(localOffset,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,8);PipeBarrier<PIPE_V>();
            Muls(localOffset,localOffset,int32_t(4),8);
            Muls(packetOffset,outputOffset_,int32_t(4),8);PipeBarrier<PIPE_V>();
            Add(localOffset,localOffset,packetOffset,8);PipeBarrier<PIPE_V>();
            Gather(indices.template ReinterpretCast<float>(),BatchIds(),localOffset.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Cast(indices,indices.template ReinterpretCast<float>(),RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
            Gather(mappedDistance,distances,BatchPointMap(),0U,64);
            Gather(mappedIndex,indices,BatchPointMap(),0U,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            smallMask_.template ReinterpretCast<uint64_t>().SetValue(0,rows);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));SetCmpMask(smallMask_);
            const BinaryRepeatParams same{1,1,1,8,8,8};
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,mappedDistance,tileSecond_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),mappedIndex.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            { stageResult = true; return true; }
        }
        return false;
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRowsType, class StageIdx_Type, class StageTileSecond_Type, class StageSmallMask_Type, class StageDistancesType, class StageIndicesType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineCandidateBatchCastStage(StageRowsType& rows, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageSmallMask_Type& smallMask_, StageDistancesType& distances, StageIndicesType& indices) {
        PipeBarrier<PIPE_V>();
        Cast(indices,indices.template ReinterpretCast<float>(),RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
        Gather(FewWork(0),distances,BatchPointMap(),0U,64);
        Gather(FewWork(1).template ReinterpretCast<int32_t>(),indices,BatchPointMap(),0U,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        smallMask_.template ReinterpretCast<uint64_t>().SetValue(0,rows);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));SetCmpMask(smallMask_);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,FewWork(0),tileSecond_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),FewWork(1),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        return true;

    }

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageValidLanesType, class StageRowsType, class StageUniqueRowsType, class StageOutputOffset_Type, class StageRecord_Type, class StageDistancesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineCandidateBatchGatherStage(StageValidLanesType& validLanes, StageRowsType& rows, StageUniqueRowsType& uniqueRows, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageDistancesType& distances) {
        if constexpr(sizeof(T)==4) {
            Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            uint32_t safeGroups=CertifyRefined(distances,validLanes);
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
                // A strictly separated interval cannot contain an exact tie.
                // Only unresolved groups enter the supporting-line certificate.
                const uint32_t groups=PredicatePopulation(rows);
                if(safeGroups!=((1U<<groups)-1U) &&
                   CorrectPendingCollinearGroups(validLanes,safeGroups,groups)) {
                    WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
                    safeGroups|=CertifyRefined(distances,validLanes);
                }
            }
            const uint32_t exactGroups=ResolveExactPacket(validLanes,safeGroups);
            if(exactGroups) {
                safeGroups|=exactGroups;
                WholeReduceMin(record_,FixedFewW(0),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                Gather(distances,record_,outputOffset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            }
            for(uint32_t group=0;group<8;++group)if(safeGroups&(1U<<group)) {
                const uint32_t row=BatchPointIds().GetValue(group)/4;
                uniqueRows|=(uint64_t(1)<<row)&rows;
            }
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ClearCandidateBatch() {
        Duplicate(BatchIds(),0.0F,64);
        Duplicate(BatchPointIds(),uint32_t(0),8);
        Duplicate(BatchPointMap(),uint32_t(0),64);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<int32_t> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SavedOverlapChunks() {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+13*1056+256;
        return LocalTensor<int32_t>(TPosition::VECCALC,base*4,64);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BuildOverlapChunks(uint32_t segments) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> chunkLower(TPosition::VECCALC,(workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0))*4,1024);
        const LocalTensor<float> hi(TPosition::VECCALC,(stateBase+128+16+2*64)*4,64);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(SavedOverlapChunks(),int32_t(0),uint64_t(64),1,1,8);
        Duplicate<float,false>(FewWork(1),0.0F,uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        for(uint32_t chunk=0;chunk<(segments+63)/64;++chunk) {
            Compare<float,false>(chunkLower[chunk*64],hi,CMPMODE::LE,uint64_t(64),same);
            Duplicate<int32_t,false>(FewWork(0).template ReinterpretCast<int32_t>(),int32_t(uint32_t(1)<<chunk),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(FewWork(0),FewWork(0),FewWork(1),1,same);PipeBarrier<PIPE_V>();
            // Distinct power-of-two bits sum exactly to their union; no floating arithmetic.
            Add<int32_t,false>(SavedOverlapChunks(),SavedOverlapChunks(),FewWork(0).template ReinterpretCast<int32_t>(),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline uint64_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineSparseRows(uint64_t rows,uint32_t validRows,uint32_t segments,uint64_t& uniqueRows) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(2));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> chunkLower_(TPosition::VECCALC,(workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0))*4,1024);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<int32_t> offset_(TPosition::VECCALC,(stateBase+128+16+11*64)*4,64);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<float> hi_(TPosition::VECCALC,(stateBase+128+16+2*64)*4,64);
        const LocalTensor<float> segmentIndices_(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        uint32_t count=0;
        uint64_t validLanes=0,groupRows=0,accepted=0;
        ClearCandidateBatch();
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE || NATIVE_HALF_PREFIX_PRIMARY) {
            // Keep uncertain geometric-zero rows on their exact fallback.
            // A zero row must not reject seven independent nonzero queries.
            const LocalTensor<float> primaryScores(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
            const LocalTensor<float> primaryNorm(TPosition::VECCALC,(stateBase-4*64)*4,64);
            Muls(FewWork(0),primaryNorm,1.8189894035458565e-12F,64);PipeBarrier<PIPE_V>();
            Compare(smallMask_,primaryScores,FewWork(0),CMPMODE::GT,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            rows&=smallMask_.template ReinterpretCast<uint64_t>().GetValue(0);
        }
#endif
        const uint64_t active=(~uint64_t(0))>>(64-validRows);
        return RefineSparseRowsGetSFFValueStage2(rows, segments, uniqueRows, chunkLower_, outputOffset_, offset_, idx_, hi_, segmentIndices_, valid_, smallMask_, count, validLanes, groupRows, accepted, active);
    }
