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

template<uint32_t DIGITS=4,uint32_t SCALE_EXP=24>
class TreePacketRational64 {
    static_assert((DIGITS==4 && SCALE_EXP==24) || (DIGITS==2 && SCALE_EXP==14),"Certified rational formats");
    LocalTensor<float> f_;
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> F(uint32_t p);
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint8_t> M(uint32_t p);
    __aicore__ __attribute__((always_inline)) inline void Split(LocalTensor<float> dst);
public:
    __aicore__ __attribute__((always_inline)) inline TreePacketRational64(uint32_t address);
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> Plane(uint32_t p);

    __aicore__ __attribute__((always_inline)) inline void Init();
    __aicore__ __attribute__((always_inline)) inline void Zero(uint32_t dst,uint32_t n);
    __aicore__ __attribute__((always_inline)) inline void CopyNumber(uint32_t dst,uint32_t src,uint32_t n);
    __aicore__ __attribute__((always_inline)) inline void NegateIf(uint32_t dst,uint32_t n,uint32_t mask);
    __aicore__ __attribute__((always_inline)) inline void Encode(uint32_t dst,LocalTensor<float> x);
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
    __aicore__ __attribute__((always_inline)) inline void MakeScoresCopyStage(StageRowType& row, StagePointPitchType& pointPitch, StagePointsType& points, StageSfType& sf, StageSoType& so);

    template<uint32_t STAGES=6>
    __aicore__ __attribute__((always_inline)) inline void Tournament();
};

#include "tree_packet_rational64_f.h"
