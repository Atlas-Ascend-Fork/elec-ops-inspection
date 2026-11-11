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
// Inline implementation of TreeFloatRational16.

    __aicore__ inline LocalTensor<float> TreeFloatRational16::F(uint32_t p) { return f_[p*V];
    }

    __aicore__ inline LocalTensor<float> TreeFloatRational16::S(uint32_t p) { return F(51*l_+p);
    }

    __aicore__ inline LocalTensor<uint8_t> TreeFloatRational16::M(uint32_t p) { return S(16).ReinterpretCast<uint8_t>()[p*32];
    }

    __aicore__ inline void TreeFloatRational16::CmpScalar(LocalTensor<uint8_t> dst,LocalTensor<float> x,float value,CMPMODE mode) {
        CompareScalar<float,uint8_t,true>(dst,x,value,mode,uint64_t(V),uint8_t(1),UnaryRepeatParams(1,1,8,8));
    }

    __aicore__ inline void TreeFloatRational16::Cmp(LocalTensor<uint8_t> dst,LocalTensor<float> x,LocalTensor<float> y,CMPMODE mode) {
        Compare<float,uint8_t,true>(dst,x,y,mode,uint64_t(V),uint8_t(1),BinaryRepeatParams(1,1,1,8,8,8));
    }

    __aicore__ inline void TreeFloatRational16::Split(LocalTensor<float> dst) {
        Muls(S(4),S(0),1.0F/4096.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(3).ReinterpretCast<int32_t>(),S(4),RoundMode::CAST_FLOOR,V);PipeBarrier<PIPE_V>();
        Cast(S(1),S(3).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(S(4),S(1),4096.0F,V);PipeBarrier<PIPE_V>();
        Sub(dst,S(0),S(4),V);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeFloatRational16::MaskAnd(uint32_t d,uint32_t a,uint32_t b) {
        And(M(d).ReinterpretCast<uint16_t>(),M(a).ReinterpretCast<uint16_t>(),M(b).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeFloatRational16::MaskOr(uint32_t d,uint32_t a,uint32_t b) {
        Or(M(d).ReinterpretCast<uint16_t>(),M(a).ReinterpretCast<uint16_t>(),M(b).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline TreeFloatRational16::TreeFloatRational16(uint32_t address,uint32_t limbs,uint32_t baseExp)
      : f_(TPosition::VECCALC,address,MAX_PLANES*16),l_(limbs),baseExp_(baseExp) {
    }

    __aicore__ inline LocalTensor<float> TreeFloatRational16::Plane(uint32_t p) {return F(p);
    }

    __aicore__ inline void TreeFloatRational16::Init() {
        Duplicate(S(12),0.0F,V);Duplicate(S(13),1.0F,V);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeFloatRational16::Zero(uint32_t dst,uint32_t n) {
        for(uint32_t k=0;k<n;++k)Duplicate(F(dst+k),0.0F,V);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeFloatRational16::CopyNumber(uint32_t dst,uint32_t src,uint32_t n) {
        for(uint32_t k=0;k<n;++k)Copy(F(dst+k),F(src+k),V,1,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeFloatRational16::NegateIf(uint32_t dst,uint32_t n,uint32_t mask) {
        Duplicate(S(1),1.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<n;++k) {
            Muls(S(0),F(dst+k),-1.0F,V);PipeBarrier<PIPE_V>();
            Adds(S(0),S(0),4095.0F,V);PipeBarrier<PIPE_V>();
            Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();
            Split(S(5));
            Select(F(dst+k),M(mask),S(5),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void TreeFloatRational16::Encode(uint32_t dst,LocalTensor<float> x) {
        auto bits=x.ReinterpretCast<uint32_t>();
        ShiftRight(S(2).ReinterpretCast<uint32_t>(),bits,uint32_t(31),V);PipeBarrier<PIPE_V>();
        Cast(S(6),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        CmpScalar(M(10),S(6),0.0F,CMPMODE::GT);
        ShiftRight(S(2).ReinterpretCast<uint32_t>(),bits,uint32_t(23),V);
        Duplicate(S(3).ReinterpretCast<uint32_t>(),uint32_t(255),V);PipeBarrier<PIPE_V>();
        And(S(2).ReinterpretCast<uint16_t>(),S(2).ReinterpretCast<uint16_t>(),S(3).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
        Cast(S(6),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);
        Duplicate(S(3).ReinterpretCast<uint32_t>(),uint32_t(0x7fffff),V);PipeBarrier<PIPE_V>();
        And(S(2).ReinterpretCast<uint16_t>(),bits.ReinterpretCast<uint16_t>(),S(3).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
        Cast(S(7),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);
        CmpScalar(M(0),S(6),0.0F,CMPMODE::GT);PipeBarrier<PIPE_V>();
        Adds(S(8),S(7),8388608.0F,V);PipeBarrier<PIPE_V>();
        Select(S(7),M(0),S(8),S(7),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);
        Select(S(6),M(0),S(6),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,V);PipeBarrier<PIPE_V>();
        Adds(S(6),S(6),-float(int32_t(baseExp_)),V);PipeBarrier<PIPE_V>();
        // Zero may carry a negative shift when every nonzero input is normal.
        // Its mantissa is zero, so clamping that unused exponent is exact.
        Maxs(S(6),S(6),0.0F,V);PipeBarrier<PIPE_V>();
        Muls(S(8),S(6),1.0F/12.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(2).ReinterpretCast<int32_t>(),S(8),RoundMode::CAST_FLOOR,V);PipeBarrier<PIPE_V>();
        Cast(S(9),S(2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(S(8),S(9),12.0F,V);PipeBarrier<PIPE_V>();
        Sub(S(8),S(6),S(8),V);PipeBarrier<PIPE_V>();
        Adds(S(8),S(8),127.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(2).ReinterpretCast<int32_t>(),S(8),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
        Muls(S(2).ReinterpretCast<int32_t>(),S(2).ReinterpretCast<int32_t>(),int32_t(8388608),V);PipeBarrier<PIPE_V>();
        // Split the 24-bit mantissa before multiplying by 2^remainder.
        Muls(S(8),S(7),1.0F/4096.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(3).ReinterpretCast<int32_t>(),S(8),RoundMode::CAST_FLOOR,V);PipeBarrier<PIPE_V>();
        Cast(S(11),S(3).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(S(8),S(11),4096.0F,V);PipeBarrier<PIPE_V>();
        Sub(S(7),S(7),S(8),V);PipeBarrier<PIPE_V>();
        Mul(S(0),S(7),S(2),V);PipeBarrier<PIPE_V>();Split(S(6));
        Mul(S(0),S(11),S(2),V);PipeBarrier<PIPE_V>();
        Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();Split(S(7));
        Copy(S(8),S(1),V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<l_;++k) {
            CmpScalar(M(0),S(9),float(int32_t(k)),CMPMODE::EQ);
            CmpScalar(M(1),S(9),float(int32_t(k)-1),CMPMODE::EQ);
            CmpScalar(M(2),S(9),float(int32_t(k)-2),CMPMODE::EQ);PipeBarrier<PIPE_V>();
            Select(F(dst+k),M(0),S(6),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,V);PipeBarrier<PIPE_V>();
            Select(F(dst+k),M(1),S(7),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
            Select(F(dst+k),M(2),S(8),F(dst+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
        }
        NegateIf(dst,l_,10);
    }

    __aicore__ inline void TreeFloatRational16::AddNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(S(1),0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<n;++k) {
            Add(S(0),F(a+k),F(b+k),V);PipeBarrier<PIPE_V>();
            Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();Split(F(dst+k));
        }
    }

    __aicore__ inline void TreeFloatRational16::SubNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n) {
        Duplicate(S(1),0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<n;++k) {
            Sub(S(0),F(a+k),F(b+k),V);PipeBarrier<PIPE_V>();
            Sub(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();
            CmpScalar(M(0),S(0),0.0F,CMPMODE::LT);PipeBarrier<PIPE_V>();
            Select(S(1),M(0),S(13),S(12),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
            Muls(S(4),S(1),4096.0F,V);PipeBarrier<PIPE_V>();
            Add(F(dst+k),S(0),S(4),V);PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void TreeFloatRational16::Multiply(uint32_t dst,uint32_t a,uint32_t na,uint32_t b,uint32_t nb,uint32_t n) {
        Zero(dst,n);
        for(uint32_t i=0;i<na;++i) {
            Duplicate(S(1),0.0F,V);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<nb && i+j<n;++j) {
                Mul(S(0),F(a+i),F(b+j),V);PipeBarrier<PIPE_V>();
                Add(S(0),S(0),F(dst+i+j),V);PipeBarrier<PIPE_V>();
                Add(S(0),S(0),S(1),V);PipeBarrier<PIPE_V>();Split(F(dst+i+j));
            }
            if(i+nb<n){Copy(F(dst+i+nb),S(1),V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
        }
    }

    __aicore__ inline void TreeFloatRational16::CompareNumber(uint32_t a,uint32_t b,uint32_t n,uint32_t less,uint32_t equal) {
        Duplicate(M(less).ReinterpretCast<uint16_t>(),uint16_t(0),1);
        Duplicate(M(equal).ReinterpretCast<uint16_t>(),uint16_t(65535),1);PipeBarrier<PIPE_V>();
        for(int32_t k=int32_t(n)-1;k>=0;--k) {
            Cmp(M(0),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::LT);
            Cmp(M(1),F(a+uint32_t(k)),F(b+uint32_t(k)),CMPMODE::EQ);PipeBarrier<PIPE_V>();
            MaskAnd(0,0,equal);MaskOr(less,less,0);MaskAnd(equal,equal,1);
        }
    }

    __aicore__ inline void TreeFloatRational16::SelectNumber(uint32_t dst,uint32_t a,uint32_t b,uint32_t n,uint32_t mask) {
        for(uint32_t k=0;k<n;++k)Select(F(dst+k),M(mask),F(a+k),F(b+k),SELMODE::VSEL_TENSOR_TENSOR_MODE,V);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeFloatRational16::MakeScores(LocalTensor<float> point,LocalTensor<float> segments,LocalTensor<uint32_t> offsets,uint32_t segmentCount,uint32_t baseSegment) {
        Zero(25*l_,8*l_);
        for(uint32_t axis=0;axis<3;++axis) {
            Gather(S(14),segments,offsets,axis*4,V);
            Gather(S(15),segments,offsets,(axis+3)*4,V);PipeBarrier<PIPE_V>();
            Encode(33*l_,point[axis*V]);Encode(34*l_,S(14));Encode(35*l_,S(15));
            SubNumber(36*l_,33*l_,34*l_,l_);SubNumber(37*l_,35*l_,34*l_,l_);SubNumber(38*l_,33*l_,35*l_,l_);
            CmpScalar(M(8),F(37*l_-1),2048.0F,CMPMODE::GE);
            CmpScalar(M(9),F(38*l_-1),2048.0F,CMPMODE::GE);PipeBarrier<PIPE_V>();
            MaskAnd(0,8,9);MaskOr(11,8,9);
            Not(M(0).ReinterpretCast<uint16_t>(),M(0).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();MaskAnd(11,11,0);
            NegateIf(36*l_,l_,8);NegateIf(37*l_,l_,9);
            CmpScalar(M(8),F(39*l_-1),2048.0F,CMPMODE::GE);PipeBarrier<PIPE_V>();NegateIf(38*l_,l_,8);
            Multiply(39*l_,36*l_,l_,36*l_,l_,2*l_);AddNumber(27*l_,27*l_,39*l_,2*l_);
            Multiply(39*l_,37*l_,l_,37*l_,l_,2*l_);AddNumber(29*l_,29*l_,39*l_,2*l_);
            Multiply(39*l_,38*l_,l_,38*l_,l_,2*l_);AddNumber(31*l_,31*l_,39*l_,2*l_);
            Multiply(39*l_,36*l_,l_,37*l_,l_,2*l_);NegateIf(39*l_,2*l_,11);AddNumber(25*l_,25*l_,39*l_,2*l_);
        }
        Multiply(43*l_,27*l_,2*l_,29*l_,2*l_,4*l_);
        Multiply(47*l_,25*l_,2*l_,25*l_,2*l_,4*l_);
        SubNumber(0,43*l_,47*l_,4*l_);CopyNumber(4*l_,29*l_,2*l_);
        CompareNumber(25*l_,29*l_,2*l_,6,7);
        Not(M(12).ReinterpretCast<uint16_t>(),M(6).ReinterpretCast<uint16_t>(),1);PipeBarrier<PIPE_V>();
        CmpScalar(M(13),F(27*l_-1),2048.0F,CMPMODE::GE);PipeBarrier<PIPE_V>();
        Zero(39*l_,4*l_);CompareNumber(25*l_,39*l_,2*l_,6,7);MaskOr(13,13,7);
        CompareNumber(29*l_,39*l_,2*l_,6,7);MaskOr(13,13,7);
        CopyNumber(39*l_,31*l_,2*l_);SelectNumber(0,39*l_,0,4*l_,12);
        Zero(39*l_,4*l_);CopyNumber(39*l_,27*l_,2*l_);SelectNumber(0,39*l_,0,4*l_,13);
        MaskOr(12,12,13);Zero(39*l_,2*l_);Duplicate(F(39*l_),1.0F,V);PipeBarrier<PIPE_V>();
        SelectNumber(4*l_,39*l_,4*l_,2*l_,12);
        Cast(F(24*l_+3),offsets.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Muls(F(24*l_+3),F(24*l_+3),1.0F/24.0F,V);PipeBarrier<PIPE_V>();
        Cast(F(24*l_+2).ReinterpretCast<int32_t>(),F(24*l_+3),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
        Cast(F(24*l_+3),F(24*l_+2).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,V);PipeBarrier<PIPE_V>();
        Adds(F(24*l_+3),F(24*l_+3),float(int32_t(baseSegment)),V);PipeBarrier<PIPE_V>();
        CmpScalar(M(12),F(24*l_+3),float(int32_t(segmentCount)),CMPMODE::GE);PipeBarrier<PIPE_V>();
        Zero(39*l_,4*l_);Duplicate(F(39*l_),1.0F,V);PipeBarrier<PIPE_V>();SelectNumber(0,39*l_,0,4*l_,12);
        Zero(39*l_,2*l_);SelectNumber(4*l_,39*l_,4*l_,2*l_,12);CopyNumber(24*l_,24*l_+3,1);
    }

    __aicore__ inline void TreeFloatRational16::ChooseGathered() {
        Multiply(12*l_,6*l_,4*l_,4*l_,2*l_,6*l_);
        Multiply(18*l_,0,4*l_,10*l_,2*l_,6*l_);
        CompareNumber(12*l_,18*l_,6*l_,6,7);
        Cmp(M(0),F(24*l_+1),F(24*l_),CMPMODE::LT);PipeBarrier<PIPE_V>();
        MaskAnd(0,0,7);MaskOr(6,6,0);
        SelectNumber(0,6*l_,0,6*l_,6);SelectNumber(24*l_,24*l_+1,24*l_,1,6);
    }

    __aicore__ inline void TreeFloatRational16::Tournament(LocalTensor<uint32_t> laneOffsets) {
        // laneOffsets initially contains 0..15, independent of global segment ID.
        for(uint32_t stage=0;stage<4;++stage) {
            Duplicate(S(10).ReinterpretCast<uint32_t>(),uint32_t(1U<<stage),V);PipeBarrier<PIPE_V>();
            And(S(9).ReinterpretCast<uint16_t>(),laneOffsets.ReinterpretCast<uint16_t>(),S(10).ReinterpretCast<uint16_t>(),2*V);
            Or(F(24*l_+2).ReinterpretCast<uint16_t>(),laneOffsets.ReinterpretCast<uint16_t>(),S(10).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
            Not(S(9).ReinterpretCast<uint16_t>(),S(9).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
            And(F(24*l_+2).ReinterpretCast<uint16_t>(),F(24*l_+2).ReinterpretCast<uint16_t>(),S(9).ReinterpretCast<uint16_t>(),2*V);PipeBarrier<PIPE_V>();
            Muls(F(24*l_+2).ReinterpretCast<int32_t>(),F(24*l_+2).ReinterpretCast<int32_t>(),int32_t(4),V);PipeBarrier<PIPE_V>();
            for(uint32_t k=0;k<6*l_;++k)Gather(F(6*l_+k),F(k),F(24*l_+2).ReinterpretCast<uint32_t>(),0U,V);
            Gather(F(24*l_+1),F(24*l_),F(24*l_+2).ReinterpretCast<uint32_t>(),0U,V);PipeBarrier<PIPE_V>();ChooseGathered();
        }
    }

    __aicore__ inline void TreeFloatRational16::Merge(LocalTensor<float> best,bool first) {
        if(!first) {
            for(uint32_t k=0;k<6*l_;++k)Copy(F(6*l_+k),best[k*V],V,1,CopyRepeatParams(1,1,8,8));
            Copy(F(24*l_+1),best[6*l_*V],V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();ChooseGathered();
        }
        for(uint32_t k=0;k<6*l_;++k)Copy(best[k*V],F(k),V,1,CopyRepeatParams(1,1,8,8));
        Copy(best[6*l_*V],F(24*l_),V,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeFloatRational16::Normalized(uint32_t src,uint32_t digits,LocalTensor<float> value,LocalTensor<float> exponent,LocalTensor<uint32_t> lanes) {
        Duplicate(exponent,0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<digits;++k) {
            CmpScalar(M(0),F(src+k),0.0F,CMPMODE::GT);
            Duplicate(S(0),float(int32_t(k)),V);PipeBarrier<PIPE_V>();
            Select(exponent,M(0),S(0),exponent,SELMODE::VSEL_TENSOR_TENSOR_MODE,V);PipeBarrier<PIPE_V>();
        }
        Duplicate(value,0.0F,V);PipeBarrier<PIPE_V>();
        for(uint32_t k=0;k<3;++k) {
            Adds(S(0),exponent,-float(int32_t(k)),V);PipeBarrier<PIPE_V>();
            CmpScalar(M(0),S(0),0.0F,CMPMODE::GE);
            Maxs(S(0),S(0),0.0F,V);PipeBarrier<PIPE_V>();
            Cast(S(1).ReinterpretCast<int32_t>(),S(0),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
            Muls(S(1).ReinterpretCast<int32_t>(),S(1).ReinterpretCast<int32_t>(),int32_t(V),V);PipeBarrier<PIPE_V>();
            Add(S(1).ReinterpretCast<int32_t>(),S(1).ReinterpretCast<int32_t>(),lanes.ReinterpretCast<int32_t>(),V);PipeBarrier<PIPE_V>();
            Muls(S(1).ReinterpretCast<int32_t>(),S(1).ReinterpretCast<int32_t>(),int32_t(4),V);PipeBarrier<PIPE_V>();
            Gather(S(2),F(src),S(1).ReinterpretCast<uint32_t>(),0U,V);PipeBarrier<PIPE_V>();
            Select(S(2),M(0),S(2),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,V);PipeBarrier<PIPE_V>();
            Muls(S(2),S(2),k==0?1.0F:k==1?0.000244140625F:5.960464477539063e-8F,V);PipeBarrier<PIPE_V>();
            Add(value,value,S(2),V);PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void TreeFloatRational16::WriteDistance(LocalTensor<float> dst,LocalTensor<uint32_t> lanes) {
        Normalized(0,4*l_,S(6),S(7),lanes);
        Normalized(4*l_,2*l_,S(8),S(9),lanes);
        Div(S(0),S(6),S(8),V);PipeBarrier<PIPE_V>();Sqrt(S(0),S(0),V);PipeBarrier<PIPE_V>();
        Sub(S(1),S(7),S(9),V);PipeBarrier<PIPE_V>();
        Muls(S(1),S(1),6.0F,V);PipeBarrier<PIPE_V>();
        Adds(S(1),S(1),float(int32_t(baseExp_)-150),V);PipeBarrier<PIPE_V>();
        Maxs(S(2),S(1),-126.0F,V);PipeBarrier<PIPE_V>();Mins(S(2),S(2),127.0F,V);PipeBarrier<PIPE_V>();
        Sub(S(3),S(1),S(2),V);PipeBarrier<PIPE_V>();
        Maxs(S(3),S(3),-126.0F,V);PipeBarrier<PIPE_V>();Mins(S(3),S(3),127.0F,V);PipeBarrier<PIPE_V>();
        Adds(S(2),S(2),127.0F,V);Adds(S(3),S(3),127.0F,V);PipeBarrier<PIPE_V>();
        Cast(S(2).ReinterpretCast<int32_t>(),S(2),RoundMode::CAST_RINT,V);
        Cast(S(3).ReinterpretCast<int32_t>(),S(3),RoundMode::CAST_RINT,V);PipeBarrier<PIPE_V>();
        Muls(S(2).ReinterpretCast<int32_t>(),S(2).ReinterpretCast<int32_t>(),int32_t(8388608),V);
        Muls(S(3).ReinterpretCast<int32_t>(),S(3).ReinterpretCast<int32_t>(),int32_t(8388608),V);PipeBarrier<PIPE_V>();
        Mul(S(0),S(0),S(2),V);PipeBarrier<PIPE_V>();Mul(dst,S(0),S(3),V);PipeBarrier<PIPE_V>();
    }
