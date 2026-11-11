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

class TreeFloatRational16 {
    LocalTensor<float> f_;
    uint32_t l_, baseExp_;
    static constexpr uint32_t V=16;
    __aicore__ inline LocalTensor<float> F(uint32_t p);
    __aicore__ inline LocalTensor<float> S(uint32_t p);
    __aicore__ inline LocalTensor<uint8_t> M(uint32_t p);
    __aicore__ inline void CmpScalar(LocalTensor<uint8_t> dst,LocalTensor<float> x,float value,CMPMODE mode);
    __aicore__ inline void Cmp(LocalTensor<uint8_t> dst,LocalTensor<float> x,LocalTensor<float> y,CMPMODE mode);
    __aicore__ inline void Split(LocalTensor<float> dst);
    __aicore__ inline void MaskAnd(uint32_t d,uint32_t a,uint32_t b);
    __aicore__ inline void MaskOr(uint32_t d,uint32_t a,uint32_t b);
public:
    static constexpr uint32_t MAX_PLANES=51*24+24;
    static constexpr uint32_t ARENA_BYTES=MAX_PLANES*16*4;
    __aicore__ inline TreeFloatRational16(uint32_t address,uint32_t limbs,uint32_t baseExp);
    __aicore__ inline LocalTensor<float> Plane(uint32_t p);
    __aicore__ inline void Init();
    __aicore__ inline void Zero(uint32_t dst,uint32_t n);
    __aicore__ inline void CopyNumber(uint32_t dst,uint32_t src,uint32_t n);
    __aicore__ inline void NegateIf(uint32_t dst,uint32_t n,uint32_t mask);
    // Decode IEEE bits. No float multiply by 2^149, denormal arithmetic, or
    // approximate exponential is involved in the exact integer encoding.
    __aicore__ inline void Encode(uint32_t dst,LocalTensor<float> x);
    __aicore__ inline void AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n);
    __aicore__ inline void SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n);
    __aicore__ inline void Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n);
    __aicore__ inline void CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal);
    __aicore__ inline void SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask);
    __aicore__ inline void MakeScores(LocalTensor<float> point,LocalTensor<float> segments,LocalTensor<uint32_t> offsets,uint32_t segmentCount,uint32_t baseSegment=0);
    __aicore__ inline void ChooseGathered();
    __aicore__ inline void Tournament(LocalTensor<uint32_t> laneOffsets);
    __aicore__ inline void Merge(LocalTensor<float> best,bool first);
    __aicore__ inline void Normalized(uint32_t src,uint32_t digits,LocalTensor<float> value,LocalTensor<float> exponent,LocalTensor<uint32_t> lanes);
    // Recover distance from the winning exact rational using its leading
    // three radix digits. Indices remain exact; FP32 distance uses the normal
    // operator tolerance. Split powers avoid avoidable intermediate overflow.
    __aicore__ inline void WriteDistance(LocalTensor<float> dst,LocalTensor<uint32_t> lanes);
};

#include "tree_float_rational16_f.h"
