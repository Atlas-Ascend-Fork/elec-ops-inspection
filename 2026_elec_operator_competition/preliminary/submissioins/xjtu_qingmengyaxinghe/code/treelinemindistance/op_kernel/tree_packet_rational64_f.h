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
// Inline implementation of TreePacketRational64.

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> TreePacketRational64<DIGITS, SCALE_EXP>::F(uint32_t p) { return f_[p*64];
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint8_t> TreePacketRational64<DIGITS, SCALE_EXP>::M(uint32_t p) { return F(220).template ReinterpretCast<uint8_t>()[p*32];
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::Split(LocalTensor<float> dst) {
        Muls(F(208),F(204),0.000244140625F,64);PipeBarrier<PIPE_V>();
        Cast(F(207).template ReinterpretCast<int32_t>(),F(208),RoundMode::CAST_FLOOR,64);PipeBarrier<PIPE_V>();
        Cast(F(205),F(207).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(208),F(205),4096.0F,64);PipeBarrier<PIPE_V>();
        Sub(dst,F(204),F(208),64);PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline TreePacketRational64<DIGITS, SCALE_EXP>::TreePacketRational64(uint32_t address)
      : f_(TPosition::VECCALC,address,224*64) {
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> TreePacketRational64<DIGITS, SCALE_EXP>::Plane(uint32_t p) {return F(p);
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::Init() {
        Duplicate(F(216),0.0F,64);Duplicate(F(217),1.0F,64);PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::Zero(uint32_t dst,uint32_t n) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Duplicate(F(dst+k),0.0F,64);
        PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::CopyNumber(uint32_t dst,uint32_t src,uint32_t n) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Copy(F(dst+k),F(src+k),64,1,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::NegateIf(uint32_t dst,uint32_t n,uint32_t mask) {
        Duplicate(F(205),1.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Muls(F(204),F(dst+k),-1.0F,64);PipeBarrier<PIPE_V>();
            Adds(F(204),F(204),4095.0F,64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(209));
            Select(F(dst+k),M(mask),F(209),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        }
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::Encode(uint32_t dst,LocalTensor<float> x) {
        CompareScalar(M(10),x,0.0F,CMPMODE::LT,64);
        Abs(F(210),x,64);PipeBarrier<PIPE_V>();
        Muls(F(210),F(210),SCALE_EXP==14?16384.0F:16777216.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=int32_t(DIGITS)-1;k>=0;--k) {
            const float inverse=k==3?1.4551915228366852e-11F:k==2?5.960464477539063e-8F:k==1?0.000244140625F:1.0F;
            const float power=k==3?68719476736.0F:k==2?16777216.0F:k==1?4096.0F:1.0F;
            Muls(F(211),F(210),inverse,64);PipeBarrier<PIPE_V>();
            Cast(F(207).template ReinterpretCast<int32_t>(),F(211),RoundMode::CAST_FLOOR,64);PipeBarrier<PIPE_V>();
            Cast(F(dst+uint32_t(k)),F(207).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Muls(F(211),F(dst+uint32_t(k)),power,64);PipeBarrier<PIPE_V>();
            Sub(F(210),F(210),F(211),64);PipeBarrier<PIPE_V>();
        }
        NegateIf(dst,DIGITS,10);
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Add(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            Split(F(dst+k));
        }
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k) {
            Sub(F(204),F(a+k),F(b+k),64);PipeBarrier<PIPE_V>();
            Sub(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
            CompareScalar(M(0),F(204),0.0F,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            Select(F(205),M(0),F(217),F(216),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
            Muls(F(208),F(205),4096.0F,64);PipeBarrier<PIPE_V>();
            Add(F(dst+k),F(204),F(208),64);PipeBarrier<PIPE_V>();
        }
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n) {
        Zero(dst,n);
        #pragma unroll 1
        for(uint32_t i=0;i<na;++i) {
            Duplicate(F(205),0.0F,64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t j=0;j<nb && i+j<n;++j) {
                if constexpr(DIGITS==2) {
                    Add(F(204),F(dst+i+j),F(205),64);PipeBarrier<PIPE_V>();
                    MulAddDst(F(204),F(a+i),F(b+j),64);PipeBarrier<PIPE_V>();
                } else {
                    Mul(F(204),F(a+i),F(b+j),64);PipeBarrier<PIPE_V>();
                    Add(F(204),F(204),F(dst+i+j),64);PipeBarrier<PIPE_V>();
                    Add(F(204),F(204),F(205),64);PipeBarrier<PIPE_V>();
                }
                Split(F(dst+i+j));
            }
            if(i+nb<n) {Copy(F(dst+i+nb),F(205),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
        }
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal) {
        Duplicate(M(less).template ReinterpretCast<uint16_t>(),uint16_t(0),4);
        Duplicate(M(equal).template ReinterpretCast<uint16_t>(),uint16_t(65535),4);PipeBarrier<PIPE_V>();
        #pragma unroll 1
        for(int32_t k=int32_t(n)-1;k>=0;--k) {
            Compare(M(0),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::LT,64);
            Compare(M(1),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(less).template ReinterpretCast<uint16_t>(),M(less).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);
            And(M(equal).template ReinterpretCast<uint16_t>(),M(equal).template ReinterpretCast<uint16_t>(),M(1).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask) {
        #pragma unroll 1
        for(uint32_t k=0;k<n;++k)Select(F(dst+k),M(mask),F(a+k),F(b+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        PipeBarrier<PIPE_V>();
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    template<bool PACKET>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::MakeScores(uint32_t row,uint32_t pointAddress,uint32_t pointPitch,uint32_t sfAddress,uint32_t soAddress,uint32_t segments,
        uint32_t indexAddress,uint64_t validLanes) {
        const LocalTensor<float> points(TPosition::VECCALC,pointAddress,3*pointPitch);
        const LocalTensor<float> sf(TPosition::VECCALC,sfAddress,PACKET?768:384);
        const LocalTensor<uint32_t> so(TPosition::VECCALC,soAddress,64);
        if constexpr(DIGITS!=4)Zero(0,24); // Keep unused numerator/denominator digits zero.
        Zero(100,32); // dot, |P-A|^2, |C-A|^2, |P-C|^2
                MakeScoresCopyStage<PACKET>(row, pointPitch, points, sf, so);
        Multiply(172,108,2*DIGITS,116,2*DIGITS,4*DIGITS);Multiply(188,100,2*DIGITS,100,2*DIGITS,4*DIGITS);
        SubNumber(0,172,188,4*DIGITS);CopyNumber(16,116,2*DIGITS);
        CompareNumber(100,116,2*DIGITS,6,7);
        Not(M(12).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>(); // dot>=vv
        CompareScalar(M(13),F(100+2*DIGITS-1),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>(); // dot<0
        Zero(156,16);CompareNumber(100,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CompareNumber(116,156,2*DIGITS,6,7);PipeBarrier<PIPE_V>();
        Or(M(13).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        CopyNumber(156,124,2*DIGITS);SelectNumber(0,156,0,16,12);
        Zero(156,16);CopyNumber(156,108,2*DIGITS);SelectNumber(0,156,0,16,13);
        Or(M(12).template ReinterpretCast<uint16_t>(),M(12).template ReinterpretCast<uint16_t>(),M(13).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Zero(156,8);Duplicate(F(156),1.0F,64);PipeBarrier<PIPE_V>();SelectNumber(16,156,16,8,12);
        if constexpr(PACKET) {
            const LocalTensor<float> ids(TPosition::VECCALC,indexAddress,64);
            Copy(F(99),so.template ReinterpretCast<float>(),64,1,CopyRepeatParams(1,1,8,8));
            Copy(F(96),ids,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            M(12).template ReinterpretCast<uint64_t>().SetValue(0,~validLanes);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        } else {
        Cast(F(99),so.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(F(99),F(99),1.0F/24.0F,64);PipeBarrier<PIPE_V>();
        // The fixed offset vector consists exactly of 0,24,...,1512.
        Cast(F(98).template ReinterpretCast<int32_t>(),F(99),RoundMode::CAST_RINT,64);PipeBarrier<PIPE_V>();
        Cast(F(99),F(98).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(M(12),F(99),float(int32_t(segments)),CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        }
        Zero(156,16);Duplicate(F(156),1.0F,64);PipeBarrier<PIPE_V>();SelectNumber(0,156,0,16,12);
        Zero(156,8);SelectNumber(16,156,16,8,12);if constexpr(!PACKET)CopyNumber(96,99,1);
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    template<bool PACKET, class StageRowType, class StagePointPitchType, class StagePointsType, class StageSfType, class StageSoType>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::MakeScoresCopyStage(StageRowType& row, StagePointPitchType& pointPitch, StagePointsType& points, StageSfType& sf, StageSoType& so) {
#pragma unroll

        for(uint32_t axis=0;axis<3;++axis) {
            if constexpr(PACKET) {
                Copy(F(218),sf[axis*128],64,1,CopyRepeatParams(1,1,8,8));
                Copy(F(219),sf[(axis+3)*128],64,1,CopyRepeatParams(1,1,8,8));
            } else {
                Gather(F(218),sf,so,axis*4,64);Gather(F(219),sf,so,(axis+3)*4,64);
            }
            PipeBarrier<PIPE_V>();
            Encode(132,points[axis*pointPitch+row*64]);Encode(136,F(218));Encode(140,F(219));
            SubNumber(144,132,136,DIGITS);SubNumber(148,140,136,DIGITS);SubNumber(152,132,140,DIGITS);
            CompareScalar(M(8),F(144+DIGITS-1),2048.0F,CMPMODE::GE,64);
            CompareScalar(M(9),F(148+DIGITS-1),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);
            Or(M(11).template ReinterpretCast<uint16_t>(),M(8).template ReinterpretCast<uint16_t>(),M(9).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Not(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(M(11).template ReinterpretCast<uint16_t>(),M(11).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            NegateIf(144,DIGITS,8);NegateIf(148,DIGITS,9);
            CompareScalar(M(8),F(152+DIGITS-1),2048.0F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();NegateIf(152,DIGITS,8);
            Multiply(156,144,DIGITS,144,DIGITS,2*DIGITS);AddNumber(108,108,156,2*DIGITS);
            Multiply(156,148,DIGITS,148,DIGITS,2*DIGITS);AddNumber(116,116,156,2*DIGITS);
            Multiply(156,152,DIGITS,152,DIGITS,2*DIGITS);AddNumber(124,124,156,2*DIGITS);
            Multiply(156,144,DIGITS,148,DIGITS,2*DIGITS);NegateIf(156,2*DIGITS,11);AddNumber(100,100,156,2*DIGITS);
        }
    }

template<uint32_t DIGITS, uint32_t SCALE_EXP>
    template<uint32_t STAGES>
    __aicore__ __attribute__((always_inline)) inline void TreePacketRational64<DIGITS, SCALE_EXP>::Tournament() {
        #pragma unroll 1
        for(uint32_t stage=0;stage<STAGES;++stage) {
            Cast(F(98).template ReinterpretCast<int32_t>(),F(99),RoundMode::CAST_RINT,64);
            Duplicate(F(214).template ReinterpretCast<int32_t>(),int32_t(1U<<stage),64);PipeBarrier<PIPE_V>();
            And(F(213).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);
            Or(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(214).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Not(F(213).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            And(F(98).template ReinterpretCast<uint16_t>(),F(98).template ReinterpretCast<uint16_t>(),F(213).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(F(98).template ReinterpretCast<int32_t>(),F(98).template ReinterpretCast<int32_t>(),int32_t(4),64);PipeBarrier<PIPE_V>();
            #pragma unroll 1
            for(uint32_t k=0;k<24;++k)Gather(F(24+k),F(k),F(98).template ReinterpretCast<uint32_t>(),0U,64);
            Gather(F(97),F(96),F(98).template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            Multiply(48,24,4*DIGITS,16,2*DIGITS,6*DIGITS);Multiply(72,0,4*DIGITS,40,2*DIGITS,6*DIGITS);
            CompareNumber(48,72,6*DIGITS,6,7);
            Compare(M(0),F(97),F(96),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            And(M(0).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),M(7).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(M(6).template ReinterpretCast<uint16_t>(),M(6).template ReinterpretCast<uint16_t>(),M(0).template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            SelectNumber(0,24,0,24,6);SelectNumber(96,97,96,1,6);
        }
    }
