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

template<class StageHilbertConstantsType>
__aicore__ __attribute__((always_inline)) inline void InitHilbertConstantsSetValueStage2(StageHilbertConstantsType& hilbertConstants) {
    hilbertConstants.SetValue(72,uint64_t(0x0000009b00000092ULL));
    hilbertConstants.SetValue(73,uint64_t(0x0000002400000095ULL));
    hilbertConstants.SetValue(74,uint64_t(0x0000001800000089ULL));
    hilbertConstants.SetValue(75,uint64_t(0x000000a7000000beULL));
    hilbertConstants.SetValue(76,uint64_t(0x0000009d0000009aULL));
    hilbertConstants.SetValue(77,uint64_t(0x0000002c00000093ULL));
    hilbertConstants.SetValue(78,uint64_t(0x000000b6000000a9ULL));
    hilbertConstants.SetValue(79,uint64_t(0x0000008700000078ULL));
    hilbertConstants.SetValue(80,uint64_t(0x000000a5000000a2ULL));
    hilbertConstants.SetValue(81,uint64_t(0x0000008600000079ULL));
    hilbertConstants.SetValue(82,uint64_t(0x0000006c000000bbULL));
    hilbertConstants.SetValue(83,uint64_t(0x000000b7000000a8ULL));
    hilbertConstants.SetValue(84,uint64_t(0x000000ad00000076ULL));
    hilbertConstants.SetValue(85,uint64_t(0x0000007c00000017ULL));
    hilbertConstants.SetValue(86,uint64_t(0x000000aa00000099ULL));
    hilbertConstants.SetValue(87,uint64_t(0x0000003b00000028ULL));
    hilbertConstants.SetValue(88,uint64_t(0x00000071000000b2ULL));
    hilbertConstants.SetValue(89,uint64_t(0x0000005800000083ULL));
    hilbertConstants.SetValue(90,uint64_t(0x0000009e000000b5ULL));
    hilbertConstants.SetValue(91,uint64_t(0x0000009700000044ULL));
    hilbertConstants.SetValue(92,uint64_t(0x000000a3000000baULL));
    hilbertConstants.SetValue(93,uint64_t(0x0000007000000059ULL));
    hilbertConstants.SetValue(94,uint64_t(0x00000064000000bdULL));
    hilbertConstants.SetValue(95,uint64_t(0x0000009f00000096ULL));
}
template<class StageHilbertConstantsType>
__aicore__ __attribute__((always_inline)) inline void InitHilbertConstantsSetValueStage(StageHilbertConstantsType& hilbertConstants) {
    hilbertConstants.SetValue(36,uint64_t(0x0000007b00000010ULL));
    hilbertConstants.SetValue(37,uint64_t(0x0000004a00000009ULL));
    hilbertConstants.SetValue(38,uint64_t(0x0000003c0000002fULL));
    hilbertConstants.SetValue(39,uint64_t(0x0000004d00000026ULL));
    hilbertConstants.SetValue(40,uint64_t(0x0000005f00000084ULL));
    hilbertConstants.SetValue(41,uint64_t(0x0000000e00000055ULL));
    hilbertConstants.SetValue(42,uint64_t(0x0000009000000043ULL));
    hilbertConstants.SetValue(43,uint64_t(0x0000002100000052ULL));
    hilbertConstants.SetValue(44,uint64_t(0x000000370000008eULL));
    hilbertConstants.SetValue(45,uint64_t(0x00000060000000b9ULL));
    hilbertConstants.SetValue(46,uint64_t(0x000000740000005dULL));
    hilbertConstants.SetValue(47,uint64_t(0x0000000b0000005aULL));
    hilbertConstants.SetValue(48,uint64_t(0x0000006b000000bcULL));
    hilbertConstants.SetValue(49,uint64_t(0x000000b0000000afULL));
    hilbertConstants.SetValue(50,uint64_t(0x0000006200000065ULL));
    hilbertConstants.SetValue(51,uint64_t(0x000000410000003eULL));
    hilbertConstants.SetValue(52,uint64_t(0x0000006d000000a4ULL));
    hilbertConstants.SetValue(53,uint64_t(0x0000001600000077ULL));
    hilbertConstants.SetValue(54,uint64_t(0x0000006a00000063ULL));
    hilbertConstants.SetValue(55,uint64_t(0x0000002900000098ULL));
    hilbertConstants.SetValue(56,uint64_t(0x000000b1000000aeULL));
    hilbertConstants.SetValue(57,uint64_t(0x000000400000003fULL));
    hilbertConstants.SetValue(58,uint64_t(0x0000007200000075ULL));
    hilbertConstants.SetValue(59,uint64_t(0x000000130000005cULL));
    hilbertConstants.SetValue(60,uint64_t(0x0000007d0000001eULL));
    hilbertConstants.SetValue(61,uint64_t(0x0000007a000000a1ULL));
    hilbertConstants.SetValue(62,uint64_t(0x000000ac00000007ULL));
    hilbertConstants.SetValue(63,uint64_t(0x0000004b00000068ULL));
    hilbertConstants.SetValue(64,uint64_t(0x0000001900000082ULL));
    hilbertConstants.SetValue(65,uint64_t(0x000000a600000085ULL));
    hilbertConstants.SetValue(66,uint64_t(0x00000088000000b3ULL));
    hilbertConstants.SetValue(67,uint64_t(0x000000bf00000054ULL));
    hilbertConstants.SetValue(68,uint64_t(0x0000000f0000005eULL));
    hilbertConstants.SetValue(69,uint64_t(0x0000001c0000008dULL));
    hilbertConstants.SetValue(70,uint64_t(0x0000002000000091ULL));
    hilbertConstants.SetValue(71,uint64_t(0x000000330000008aULL));
    InitHilbertConstantsSetValueStage2(hilbertConstants);
}
__aicore__ __attribute__((always_inline)) inline void InitHilbertConstants(
    const LocalTensor<uint64_t>& hilbertConstants) {
    hilbertConstants.SetValue(0,uint64_t(0x0000001100000008ULL));
    hilbertConstants.SetValue(1,uint64_t(0x000000020000001bULL));
    hilbertConstants.SetValue(2,uint64_t(0x0000002e00000027ULL));
    hilbertConstants.SetValue(3,uint64_t(0x0000000500000034ULL));
    hilbertConstants.SetValue(4,uint64_t(0x0000004700000038ULL));
    hilbertConstants.SetValue(5,uint64_t(0x0000005600000049ULL));
    hilbertConstants.SetValue(6,uint64_t(0x000000140000005bULL));
    hilbertConstants.SetValue(7,uint64_t(0x0000000d0000000aULL));
    hilbertConstants.SetValue(8,uint64_t(0x0000000100000030ULL));
    hilbertConstants.SetValue(9,uint64_t(0x0000006e00000067ULL));
    hilbertConstants.SetValue(10,uint64_t(0x0000001200000073ULL));
    hilbertConstants.SetValue(11,uint64_t(0x000000150000000cULL));
    hilbertConstants.SetValue(12,uint64_t(0x000000810000007eULL));
    hilbertConstants.SetValue(13,uint64_t(0x0000001a0000001dULL));
    hilbertConstants.SetValue(14,uint64_t(0x000000500000004fULL));
    hilbertConstants.SetValue(15,uint64_t(0x000000030000008cULL));
    hilbertConstants.SetValue(16,uint64_t(0x0000002b00000094ULL));
    hilbertConstants.SetValue(17,uint64_t(0x0000002200000025ULL));
    hilbertConstants.SetValue(18,uint64_t(0x000000800000007fULL));
    hilbertConstants.SetValue(19,uint64_t(0x000000510000004eULL));
    hilbertConstants.SetValue(20,uint64_t(0x0000002d0000009cULL));
    hilbertConstants.SetValue(21,uint64_t(0x0000002a00000023ULL));
    hilbertConstants.SetValue(22,uint64_t(0x000000060000001fULL));
    hilbertConstants.SetValue(23,uint64_t(0x00000069000000a0ULL));
    hilbertConstants.SetValue(24,uint64_t(0x0000005700000048ULL));
    hilbertConstants.SetValue(25,uint64_t(0x000000040000008bULL));
    hilbertConstants.SetValue(26,uint64_t(0x0000004600000039ULL));
    hilbertConstants.SetValue(27,uint64_t(0x0000003500000032ULL));
    hilbertConstants.SetValue(28,uint64_t(0x000000ab00000000ULL));
    hilbertConstants.SetValue(29,uint64_t(0x0000004c0000006fULL));
    hilbertConstants.SetValue(30,uint64_t(0x0000003a00000031ULL));
    hilbertConstants.SetValue(31,uint64_t(0x0000003d00000066ULL));
    hilbertConstants.SetValue(32,uint64_t(0x0000008f000000b4ULL));
    hilbertConstants.SetValue(33,uint64_t(0x000000b800000053ULL));
    hilbertConstants.SetValue(34,uint64_t(0x0000003600000045ULL));
    hilbertConstants.SetValue(35,uint64_t(0x0000006100000042ULL));
    InitHilbertConstantsSetValueStage(hilbertConstants);
}
// Hilbert keys use complete64-word tiles; no point arithmetic is changed.
__aicore__ __attribute__((always_inline)) inline void SpatialKeyAndNormal(
    const LocalTensor<uint32_t>& dst,const LocalTensor<uint32_t>& src,uint32_t value,
    const LocalTensor<uint32_t>& constants,uint8_t repeats) {
    Duplicate<uint32_t,false>(constants,value,uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
    SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
    And<uint16_t,false>(dst.ReinterpretCast<uint16_t>(),src.ReinterpretCast<uint16_t>(),
        constants.ReinterpretCast<uint16_t>(),uint64_t(128),repeats,
        BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
    SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
}
__aicore__ __attribute__((always_inline)) inline void SpatialHilbertNormalPackets(
    const LocalTensor<int32_t>& quantized,const LocalTensor<uint32_t>& state,
    const LocalTensor<uint32_t>& keyInteger,const LocalTensor<uint32_t>& digit,
    const LocalTensor<uint32_t>& ids,const LocalTensor<uint32_t>& mask,
    const LocalTensor<float>& hilbertTable,uint32_t count,uint32_t lowest) {
    const uint8_t repeats=uint8_t(count/64);
    const UnaryRepeatParams unary{1,1,8,8};
    const BinaryRepeatParams binary{1,1,1,8,8,8};
    SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
    Duplicate<uint32_t,false>(state,uint32_t(0),uint64_t(64),repeats,1,8);
    Duplicate<uint32_t,false>(keyInteger,uint32_t(0),uint64_t(64),repeats,1,8);PipeBarrier<PIPE_V>();
    for(uint32_t remaining=7;remaining>lowest;--remaining) {
        Duplicate<uint32_t,false>(mask,uint32_t(0),uint64_t(64),repeats,1,8);PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            ShiftRight<uint32_t,false>(digit,quantized[axis*SORT_CAP].ReinterpretCast<uint32_t>(),remaining-1,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
            SpatialKeyAndNormal(digit,digit,1,ids,repeats);
            ShiftLeft<uint32_t,false>(digit,digit,2-axis,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
            Add<int32_t,false>(mask.ReinterpretCast<int32_t>(),mask.ReinterpretCast<int32_t>(),digit.ReinterpretCast<int32_t>(),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        }
        ShiftLeft<uint32_t,false>(ids,state,uint32_t(3),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(ids.ReinterpretCast<int32_t>(),ids.ReinterpretCast<int32_t>(),mask.ReinterpretCast<int32_t>(),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        ShiftLeft<uint32_t,false>(ids,ids,uint32_t(2),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Gather(mask.ReinterpretCast<float>(),hilbertTable,ids,uint32_t(0),count);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
        ShiftRight<uint32_t,false>(state,mask,uint32_t(3),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        SpatialKeyAndNormal(digit,mask,7,ids,repeats);
        ShiftLeft<uint32_t,false>(keyInteger,keyInteger,uint32_t(3),uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(keyInteger.ReinterpretCast<int32_t>(),keyInteger.ReinterpretCast<int32_t>(),digit.ReinterpretCast<int32_t>(),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
    }
}
