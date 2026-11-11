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
// Inline implementation of TreePacketIntegerRational64.

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<int32_t> TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::F(uint32_t p) { return f_[p*64];
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint8_t> TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::M(uint32_t p) { return F(220).template ReinterpretCast<uint8_t>()[p*32];
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::Split(LocalTensor<int32_t> dst) {
        ShiftRight(F(205),F(204),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(208),F(205),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(dst,F(204),F(208),64);PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::TreePacketIntegerRational64(uint32_t address)
      : f_(TPosition::VECCALC,address,224*64) {
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::Plane(uint32_t p) {return F(p).template ReinterpretCast<float>();
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::Init() {
        Duplicate(F(216),int32_t(0),64);Duplicate(F(217),int32_t(1),64);PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::Zero(uint32_t dst,uint32_t n) {
        if constexpr(DIGITS==2) {
            Duplicate(F(dst),int32_t(0),n*64);
        } else {
            #pragma unroll 1
            for(uint32_t k=0;k<n;++k)Duplicate(F(dst+k),int32_t(0),64);
        }
        PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::CopyNumber(uint32_t dst,uint32_t src,uint32_t n) {
        if constexpr(DIGITS==2) {
            Copy(F(dst),F(src),64,uint8_t(n),CopyRepeatParams(1,1,8,8));
        } else {
            #pragma unroll 1
            for(uint32_t k=0;k<n;++k)Copy(F(dst+k),F(src+k),64,1,CopyRepeatParams(1,1,8,8));
        }
        PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::SelectDigit(
        LocalTensor<int32_t> dst,LocalTensor<int32_t> a,LocalTensor<int32_t> b,uint32_t mask) {
        Adds(F(206),a,int32_t(1073741824),64);
        Adds(F(212),b,int32_t(1073741824),64);PipeBarrier<PIPE_V>();
        Select(F(215).template ReinterpretCast<float>(),M(mask),F(206).template ReinterpretCast<float>(),F(212).template ReinterpretCast<float>(),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        Adds(dst,F(215),int32_t(-1073741824),64);PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::NegateIf(uint32_t dst,uint32_t n,uint32_t mask) {
        Duplicate(F(205),int32_t(1),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Muls(F(204),F(dst+k),int32_t(-1),64);PipeBarrier<PIPE_V>();
            Adds(F(204),F(204),int32_t(32767),64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(209));
            SelectDigit(F(dst+k),F(209),F(dst+k),mask);
        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::Encode(uint32_t dst,LocalTensor<float> x) {
        if constexpr(SIGNED_COORDINATES) {
        // The certificate preserves every admitted coordinate. Bound speculative
        // rejected lanes as well, so all signed digit intermediates stay in range.
        Maxs(F(210).template ReinterpretCast<float>(),x,-128.0F,64);PipeBarrier<PIPE_V>();
        Mins(F(210).template ReinterpretCast<float>(),F(210).template ReinterpretCast<float>(),128.0F,64);PipeBarrier<PIPE_V>();
        Muls(F(210).template ReinterpretCast<float>(),F(210).template ReinterpretCast<float>(),1048576.0F,64);PipeBarrier<PIPE_V>();
        Cast(F(207),F(210).template ReinterpretCast<float>(),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        ShiftRight(F(dst+1),F(207),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(211),F(dst+1),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(F(dst),F(207),F(211),64);PipeBarrier<PIPE_V>();

        } else {
        // Convert the signed value once, then form its two base32768 digits.
        // All finite Half values times2^14 fit int32; admitted values are exact.
        Muls(F(210).template ReinterpretCast<float>(),x,16384.0F,64);PipeBarrier<PIPE_V>();
        Cast(F(207),F(210).template ReinterpretCast<float>(),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        ShiftRight(F(dst+1),F(207),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(211),F(dst+1),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(F(dst),F(207),F(211),64);PipeBarrier<PIPE_V>();
        ShiftRight(F(205),F(dst+1),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(F(211),F(205),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(F(dst+1),F(dst+1),F(211),64);PipeBarrier<PIPE_V>();

        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::SubCoordinate(uint32_t dst,uint32_t a,uint32_t b) {
        Sub(F(204),F(a),F(b),64);PipeBarrier<PIPE_V>();
        Split(F(dst));
        Sub(F(dst+1),F(a+1),F(b+1),64);PipeBarrier<PIPE_V>();
        Add(F(dst+1),F(dst+1),F(205),64);PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::CoordinateProduct(uint32_t dst,uint32_t a,uint32_t b) {
        // Low digits are unsigned15-bit; differences have |high digit| <= 8192.
        Mul(F(204),F(a),F(b),64);PipeBarrier<PIPE_V>();
        Split(F(dst));
        Mul(F(210),F(a),F(b+1),64);
        Mul(F(211),F(a+1),F(b),64);PipeBarrier<PIPE_V>();
        Add(F(204),F(210),F(211),64);PipeBarrier<PIPE_V>();
        Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
        Split(F(dst+1));
        Mul(F(204),F(a+1),F(b+1),64);PipeBarrier<PIPE_V>();
        Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
        Split(F(dst+2));
        Copy(F(204),F(205),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        Split(F(dst+3));
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Add(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(dst+k));
        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        if constexpr(SIGNED_COORDINATES) {
        Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Sub(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            // Signed carry is0 or-1. Split normalizes every digit exactly.
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(dst+k));
        }

        } else {
        Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Sub(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Sub(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Cast(F(206).template ReinterpretCast<float>(),F(204),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(0),F(206).template ReinterpretCast<float>(),0.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            SelectDigit(F(205),F(217),F(216),0);
            Muls(F(208),F(205),int32_t(32768),64);PipeBarrier<PIPE_V>();
            Add(F(dst+k),F(204),F(208),64);PipeBarrier<PIPE_V>();
        }

        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n) {
        Zero(dst,n);
        #pragma unroll 1
        for(uint32_t i=0;i<na;++i) {
            Duplicate(F(205),int32_t(0),64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t j=0;j<nb && i+j<n;++j) {
                Mul(F(204),F(a+i),F(b+j),64);PipeBarrier<PIPE_V>();
                Add(F(204),F(204),F(dst+i+j),64);PipeBarrier<PIPE_V>();
                Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
                Split(F(dst+i+j));
            }
            if(i+nb<n) {Copy(F(dst+i+nb),F(205),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal) {
        Duplicate(M(less).template ReinterpretCast<uint16_t>(),uint16_t(0),4);
        Duplicate(M(equal).template ReinterpretCast<uint16_t>(),uint16_t(65535),4);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=int32_t(n)-1;k>=0;--k) {
            // int32 Compare supports EQ only on this device. The biased
            // normal float encodings preserve the order of all15-bit digits.
            Adds(F(206),F(a+uint32_t(k)),int32_t(1073741824),64);
            Adds(F(212),F(b+uint32_t(k)),int32_t(1073741824),64);PipeBarrier<PIPE_V>();
            Compare(M(0),F(206).template ReinterpretCast<float>(),F(212).template ReinterpretCast<float>(),CMPMODE::LT,64);
            Compare(M(1),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(less).template ReinterpretCast<uint16_t>(),M(less).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);
            And(M(equal).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),M(1).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask) {
        if(dst==96) {
            // Lane IDs already contain normal float indices or exact zero.
            Select(F(96).template ReinterpretCast<float>(),M(mask),F(a).template ReinterpretCast<float>(),F(b).template ReinterpretCast<float>(),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        } else {
            #pragma unroll 1
            for(uint32_t k=0;k<n;++k)SelectDigit(F(dst+k),F(a+k),F(b+k),mask);
        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    template<bool PACKET>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::MakeScores(uint32_t row,uint32_t pointAddress,uint32_t pointPitch,uint32_t sfAddress,uint32_t soAddress,uint32_t segments,
        uint32_t indexAddress,uint64_t validLanes) {
        const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
        const LocalTensor<float> sf(TPosition::VECCALC,sfAddress,PACKET?768:384);
        const LocalTensor<uint32_t> so(TPosition::VECCALC,soAddress,64);
        if constexpr(DIGITS!=4)Zero(0,24); // Keep unused numerator/denominator digits zero.
        Zero(100,32); // dot, |P-A|^2, |C-A|^2, |P-C|^2
                MakeScoresCopyStage2<PACKET>(row, pointPitch, points, sf, so);
        Multiply(172,108,2*DIGITS,116,2*DIGITS,NUMERATOR_DIGITS);Multiply(188,100,2*DIGITS,100,2*DIGITS,NUMERATOR_DIGITS);
        SubNumber(0,172,188,NUMERATOR_DIGITS);CopyNumber(16,116,2*DIGITS);
        CompareNumber(100,116,2*DIGITS,6,7);
        Not(M(12).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>(); // dot>=vv
        Cast(F(206).template ReinterpretCast<float>(),F(100+2*DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(M(13),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>(); // dot<0
        Zero(156,NUMERATOR_DIGITS);CompareNumber(100,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CompareNumber(116,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CopyNumber(156,124,2*DIGITS);SelectNumber(0,156,0,NUMERATOR_DIGITS,12);
        Zero(156,NUMERATOR_DIGITS);CopyNumber(156,108,2*DIGITS);SelectNumber(0,156,0,NUMERATOR_DIGITS,13);
        Or(M(12).template ReinterpretCast<uint16_t>(),M(12).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Zero(156,2*DIGITS);Duplicate(F(156),int32_t(1),64);PipeBarrier<PIPE_V>();SelectNumber(16,156,16,2*DIGITS,12);
        if constexpr(PACKET) {
            const LocalTensor<float> ids(TPosition::VECCALC,indexAddress,64);
            Copy(F(99),so.template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));
            Copy(F(96),ids.template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            M(12).template ReinterpretCast<uint64_t>().SetValue(0,~validLanes);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        } else {
        Cast(F(99).template ReinterpretCast<float>(),so.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(99).template ReinterpretCast<float>(),F(99).template ReinterpretCast<float>(),1.0F/24.0F,64);PipeBarrier<PIPE_V>();
        // The fixed offset vector consists exactly of 0,24,...,1512.
        Cast(F(98),F(99).template ReinterpretCast<float>(),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        Cast(F(99).template ReinterpretCast<float>(),F(98),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(M(12),F(99).template ReinterpretCast<float>(),float(int32_t(segments)),CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        }
        Zero(156,NUMERATOR_DIGITS);Duplicate(F(156),int32_t(1),64);PipeBarrier<PIPE_V>();SelectNumber(0,156,0,NUMERATOR_DIGITS,12);
        Zero(156,2*DIGITS);SelectNumber(16,156,16,2*DIGITS,12);if constexpr(!PACKET)CopyNumber(96,99,1);
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    template<bool PACKET, class StageRowType, class StagePointPitchType, class StagePointsType, class StageSfType, class StageSoType>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::MakeScoresCopyStage2(StageRowType& row, StagePointPitchType& pointPitch, StagePointsType& points, StageSfType& sf, StageSoType& so) {
#pragma unroll

        for(uint32_t axis=0;axis<3;++axis) {
            if constexpr(PACKET) {
                Copy(F(218),sf[axis*128].template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));
                Copy(F(219),sf[(axis+3)*128].template ReinterpretCast<int32_t>(),64,1,CopyRepeatParams(1,1,8,8));
            } else {
                Gather(F(218),sf.template ReinterpretCast<int32_t>(),so,axis*4,64);Gather(F(219),sf.template ReinterpretCast<int32_t>(),so,(axis+3)*4,64);
            }
            PipeBarrier<PIPE_V>();
            Encode(132,points[axis*pointPitch+row*64]);Encode(136,F(218).template ReinterpretCast<float>());Encode(140,F(219).template ReinterpretCast<float>());
            if constexpr(SIGNED_COORDINATES) {
            SubCoordinate(144,132,136);SubCoordinate(148,140,136);SubCoordinate(152,132,140);
            CoordinateProduct(156,144,144);AddNumber(108,108,156,2*DIGITS);
            CoordinateProduct(156,148,148);AddNumber(116,116,156,2*DIGITS);
            CoordinateProduct(156,152,152);AddNumber(124,124,156,2*DIGITS);
            CoordinateProduct(156,144,148);AddNumber(100,100,156,2*DIGITS);
            } else {
            SubNumber(144,132,136,DIGITS);SubNumber(148,140,136,DIGITS);SubNumber(152,132,140,DIGITS);
            Cast(F(206).template ReinterpretCast<float>(),F(144+DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(8),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);
            Cast(F(206).template ReinterpretCast<float>(),F(148+DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(9),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);
            Or(M(11).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Not(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(M(11).template ReinterpretCast<uint16_t>(),M(11).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            NegateIf(144,DIGITS,8);NegateIf(148,DIGITS,9);
            Cast(F(206).template ReinterpretCast<float>(),F(152+DIGITS-1),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(M(8),F(206).template ReinterpretCast<float>(),16384.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();NegateIf(152,DIGITS,8);
            Multiply(156,144,DIGITS,144,DIGITS,2*DIGITS);AddNumber(108,108,156,2*DIGITS);
            Multiply(156,148,DIGITS,148,DIGITS,2*DIGITS);AddNumber(116,116,156,2*DIGITS);
            Multiply(156,152,DIGITS,152,DIGITS,2*DIGITS);AddNumber(124,124,156,2*DIGITS);
            Multiply(156,144,DIGITS,148,DIGITS,2*DIGITS);NegateIf(156,2*DIGITS,11);AddNumber(100,100,156,2*DIGITS);
            }
        }
    }

template<uint32_t DIGITS, bool SIGNED_COORDINATES>
    template<uint32_t STAGES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketIntegerRational64<DIGITS, SIGNED_COORDINATES>::Tournament() {
        #pragma unroll 1
        for(uint32_t stage=0;stage<STAGES;++stage) {
            Cast(F(98),F(99).template ReinterpretCast<float>(),RoundMode::CAST_RINT,64);
            Duplicate(F(214).template ReinterpretCast<int32_t>(),int32_t(1U<<stage),64);PipeBarrier<PIPE_V>();
            And(F(213).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);
            Or(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Not(F(213).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            And(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(F(98).template ReinterpretCast<int32_t>(),F(98).template ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();
            if constexpr(DIGITS==2) {
                #pragma unroll 1
                for(uint32_t k=0;k<NUMERATOR_DIGITS;++k)Gather(F(24+k),F(k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
                #pragma unroll 1
                for(uint32_t k=0;k<4;++k)Gather(F(40+k),F(16+k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                #pragma unroll 1
                for(uint32_t k=0;k<24;++k)Gather(F(24+k),F(k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
            }
            Gather(F(97),F(96),F(98).template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            Multiply(48,24,NUMERATOR_DIGITS,16,2*DIGITS,CROSS_DIGITS);Multiply(72,0,NUMERATOR_DIGITS,40,2*DIGITS,CROSS_DIGITS);
            CompareNumber(48,72,CROSS_DIGITS,6,7);
            Compare(M(0),F(97).template ReinterpretCast<float>(),F(96).template ReinterpretCast<float>(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(6).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            if constexpr(DIGITS==2) {
                SelectNumber(0,24,0,NUMERATOR_DIGITS,6);SelectNumber(16,40,16,4,6);
            } else {
                SelectNumber(0,24,0,24,6);
            }
            SelectNumber(96,97,96,1,6);
        }
    }
