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
// Resource declaration and inline implementation modules.

template<uint32_t DIGITS=2,bool SIGNED_COORDINATES=true>
class TreePacketIntegerRational64 {
    static_assert(DIGITS==2 && SIGNED_COORDINATES,"Signed base-32768 rational format");
    // With |coordinate| < 128 and scale 2^20, dot products fit 59 signed
    // bits, Gram numerators fit 120 bits, and cross products fit 180 bits.
    static constexpr uint32_t NUMERATOR_DIGITS=8U;
    static constexpr uint32_t CROSS_DIGITS=NUMERATOR_DIGITS+2*DIGITS;
    LocalTensor<int32_t> f_;
    __aicore__ __attribute__((always_inline)) inline LocalTensor<int32_t> F(uint32_t p);
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint8_t> M(uint32_t p);
    // Each multiply-add is at most 2^30-1, safely within signed int32.
    __aicore__ __attribute__((always_inline)) inline void Split(LocalTensor<int32_t> dst);
public:
    __aicore__ __attribute__((always_inline)) inline TreePacketIntegerRational64(uint32_t address);
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> Plane(uint32_t p);

    __aicore__ __attribute__((always_inline)) inline void Init();
    __aicore__ __attribute__((always_inline)) inline void Zero(uint32_t dst,uint32_t n);
    __aicore__ __attribute__((always_inline)) inline void CopyNumber(uint32_t dst,uint32_t src,uint32_t n);
    // Digit values are in [0,32767]. Bias only their representation so VSEL
    // sees normal float values, then undo the bias using exact int32 arithmetic.
    __aicore__ __attribute__((always_inline)) inline void SelectDigit(
        LocalTensor<int32_t> dst,LocalTensor<int32_t> a,LocalTensor<int32_t> b,uint32_t mask);
    __aicore__ __attribute__((always_inline)) inline void NegateIf(uint32_t dst,uint32_t n,uint32_t mask);
    __aicore__ __attribute__((always_inline)) inline void Encode(uint32_t dst,LocalTensor<float> x);
    __aicore__ __attribute__((always_inline)) inline void SubCoordinate(uint32_t dst,uint32_t a,uint32_t b);
    __aicore__ __attribute__((always_inline)) inline void CoordinateProduct(uint32_t dst,uint32_t a,uint32_t b);
    __aicore__ __attribute__((always_inline)) inline void AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n);
    __aicore__ __attribute__((always_inline)) inline void SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n);
    __aicore__ __attribute__((always_inline)) inline void Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n);
    // Return masks a<b and a==b; callers retain any signed predicates elsewhere.
    __aicore__ __attribute__((always_inline)) inline void CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal);
    __aicore__ __attribute__((always_inline)) inline void SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask);
    template<bool PACKET=false>
    __aicore__ __attribute__((always_inline)) inline void MakeScores(uint32_t row,uint32_t pointAddress,uint32_t pointPitch,uint32_t sfAddress,uint32_t soAddress,uint32_t segments,
        uint32_t indexAddress=0,uint64_t validLanes=0);
    template<bool PACKET, class StageRowType, class StagePointPitchType, class StagePointsType, class StageSfType, class StageSoType>
    __aicore__ __attribute__((always_inline)) inline void MakeScoresCopyStage2(StageRowType& row, StagePointPitchType& pointPitch, StagePointsType& points, StageSfType& sf, StageSoType& so);

    template<uint32_t STAGES=6>
    __aicore__ __attribute__((always_inline)) inline void Tournament();
};

#include "tree_packet_integer_rational64_f.h"
