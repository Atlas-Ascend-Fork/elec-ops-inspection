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

template<class ControlArgument0, class ControlArgument1, class ControlArgument2, class ControlArgument3>
__aicore__ __attribute__((always_inline)) inline void SharedCrossProductExpansion1ControlStage1(ControlArgument0& limbs, ControlArgument1& tmp, ControlArgument2& expansion, ControlArgument3& same) {
for(uint32_t k=0;k<limbs;++k) {
                            Add<float,false>(tmp[3*64],tmp[2*64],expansion[k*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                            Sub<float,false>(tmp[4*64],tmp[3*64],tmp[2*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                            Sub<float,false>(tmp[5*64],tmp[3*64],tmp[4*64],uint64_t(64),1,same);
                            Sub<float,false>(tmp[6*64],expansion[k*64],tmp[4*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                            Sub<float,false>(tmp[7*64],tmp[2*64],tmp[5*64],uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                            Add<float,false>(expansion[k*64],tmp[7*64],tmp[6*64],uint64_t(64),1,same);
                            Copy(tmp[2*64],tmp[3*64],64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                        }
}

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7, class SharedArgument8, class SharedArgument9, class SharedArgument10, class SharedArgument11>
__aicore__ __attribute__((always_inline)) inline void SharedCrossProductExpansion1(SharedArgument0& pointComponents, SharedArgument1& directionComponents, SharedArgument2& delta, SharedArgument3& next, SharedArgument4& axis, SharedArgument5& tmp, SharedArgument6& same, SharedArgument7& unary, SharedArgument8& limbs, SharedArgument9& expansion, SharedArgument10& test, SharedArgument11& certified) {
for(uint32_t sign=0;sign<2;++sign)
            for(uint32_t al=0;al<pointComponents;++al)
                for(uint32_t dl=0;dl<directionComponents;++dl) {
                    auto x=delta[(al*6+(sign?next:axis))*64];
                    auto y=delta[(3+dl*6+(sign?axis:next))*64];
                    SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Mul<float,false>(tmp,x,y,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    ExactProductError<64>(tmp[64],x,y,tmp,tmp[2*64]);
                    if(sign) {
                        Muls<float,false>(tmp,tmp,-1.0F,uint64_t(64),1,unary);
                        Muls<float,false>(tmp[64],tmp[64],-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                    }
                    for(uint32_t part=0;part<2;++part) {
                        Copy(tmp[2*64],tmp[(1-part)*64],64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
                        SharedCrossProductExpansion1ControlStage1(limbs, tmp, expansion, same);
                        Copy(expansion[limbs*64],tmp[2*64],64,1,CopyRepeatParams(1,1,8,8));++limbs;PipeBarrier<PIPE_V>();
                    }
                }
        for(uint32_t k=0;k<limbs;++k) {
            CompareScalar(test,expansion[k*64],0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
}

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6>
__aicore__ __attribute__((always_inline)) inline void SharedCrossProductScaling2(SharedArgument0& pointComponents, SharedArgument1& directionComponents, SharedArgument2& tmp, SharedArgument3& delta, SharedArgument4& test, SharedArgument5& other, SharedArgument6& certified) {
const uint32_t checkedColumns=6+3*(pointComponents-1)+3*(directionComponents-1);
        for(uint32_t k=0;k<checkedColumns;++k) {
            const uint32_t col=k>=6 && pointComponents==1?k+3:k;
            Abs(tmp,delta[col*64],64);PipeBarrier<PIPE_V>();
            CompareScalar(test,tmp,9.313225746154785e-10F,CMPMODE::GE,64);
            CompareScalar(other,tmp,0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            Or(test.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);
            CompareScalar(test,tmp,9.223372036854776e18F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(certified.template ReinterpretCast<uint16_t>(),certified.template ReinterpretCast<uint16_t>(),test.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
}

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3>
__aicore__ __attribute__((always_inline)) inline void SharedScaledPointDifference3(SharedArgument0& tmp, SharedArgument1& delta, SharedArgument2& certified, SharedArgument3& test) {
Abs(tmp,delta[6*64],6*64);PipeBarrier<PIPE_V>();
        // Only surviving candidates affect the expansion's component counts.
#pragma unroll
        for(uint32_t axis=0;axis<6;++axis)
            Select(tmp[axis*64],certified,tmp[axis*64],0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        WholeReduceMax(tmp[6*64],tmp,64,3,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(tmp[6*64+8],tmp[3*64],64,3,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(tmp[7*64],tmp[6*64],3,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(tmp[7*64+8],tmp[6*64+8],3,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        CompareScalar(test,tmp[7*64],0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
}

template<class SharedArgument0, class SharedArgument1, class SharedArgument2>
__aicore__ __attribute__((always_inline)) inline void SharedCrossProductRounding4(SharedArgument0& geometry, SharedArgument1& delta, SharedArgument2& tmp) {
#pragma unroll

    for(uint32_t axis=0;axis<3;++axis) {
#pragma unroll
        for(uint32_t kind=0;kind<2;++kind) {
            auto x=geometry[(kind==0?axis:axis+6)*64],y=geometry[(axis+3)*64];
            auto hi=delta[(kind*3+axis)*64],lo=delta[(6+kind*3+axis)*64];
            Sub(hi,x,y,64);PipeBarrier<PIPE_V>();
            Sub(tmp,x,hi,64);PipeBarrier<PIPE_V>();
            Add(tmp[64],hi,tmp,64);Sub(lo,tmp,y,64);PipeBarrier<PIPE_V>();
            Sub(tmp[64],x,tmp[64],64);PipeBarrier<PIPE_V>();
            Add(lo,lo,tmp[64],64);PipeBarrier<PIPE_V>();
        }
    }
}
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7, class SharedArgument8, class SharedArgument9, class SharedArgument10, class SharedArgument11>
__aicore__ __attribute__((always_inline)) inline void CommonDyadicGeometry51(SharedArgument0& rotate1, SharedArgument1& rotate2, SharedArgument2& coordinates, SharedArgument3& u, SharedArgument4& relA, SharedArgument5& v, SharedArgument6& delta, SharedArgument7& cross, SharedArgument8& crossNorm, SharedArgument9& pairs, SharedArgument10& product, SharedArgument11& endpoint) {
Adds<int32_t,false>(rotate1,rotate1,int32_t(4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Adds<int32_t,false>(rotate2,rotate2,int32_t(-8),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x0404040404040404ULL));
            Adds<int32_t,false>(rotate1,rotate1,int32_t(-4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Adds<int32_t,false>(rotate2,rotate2,int32_t(-4),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Gather(u,relA,rotate1.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(v,delta,rotate2.template ReinterpretCast<uint32_t>(),0U,coordinates);PipeBarrier<PIPE_V>();
            Mul(cross,u,v,coordinates);PipeBarrier<PIPE_V>();
            Gather(u,relA,rotate2.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(v,delta,rotate1.template ReinterpretCast<uint32_t>(),0U,coordinates);PipeBarrier<PIPE_V>();
            Mul(u,u,v,coordinates);PipeBarrier<PIPE_V>();
            Sub(cross,cross,u,coordinates);PipeBarrier<PIPE_V>();
            Mul(cross,cross,cross,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(crossNorm,cross,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            Mul(product,relA,relA,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(endpoint,product,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
}

template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void CommonDyadicGeometry52(SharedArgument0& values) {
SetMaskNorm();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(0), uint64_t(0xaa));
        Adds<int32_t, false>(values, values, int32_t(1), uint64_t(64), 1, UnaryRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(0), uint64_t(0xcc));
        Adds<int32_t, false>(values, values, int32_t(2), uint64_t(64), 1, UnaryRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(0), uint64_t(0xf0));
        Adds<int32_t, false>(values, values, int32_t(4), uint64_t(64), 1, UnaryRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t, MaskMode::NORMAL>(uint64_t(64));
}

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7, class SharedArgument8, class SharedArgument9>
__aicore__ __attribute__((always_inline)) inline void CommonDyadicGeometry53(SharedArgument0& temp, SharedArgument1& relC, SharedArgument2& coordinates, SharedArgument3& upper, SharedArgument4& pairs, SharedArgument5& endpoint, SharedArgument6& norm, SharedArgument7& crossNorm, SharedArgument8& flags, SharedArgument9& endpoints) {
Mul(temp,relC,relC,coordinates);PipeBarrier<PIPE_V>();
            WholeReduceSum(upper,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            Min(endpoint,endpoint,upper,pairs);Maxs(norm,norm,0.00390625F,pairs);PipeBarrier<PIPE_V>();
            Div(upper,crossNorm,norm,pairs);PipeBarrier<PIPE_V>();
            Duplicate(flags.template ReinterpretCast<uint32_t>(),endpoints,2);PipeBarrier<PIPE_V>();
}

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7, class SharedArgument8, class SharedArgument9, class SharedArgument10>
__aicore__ __attribute__((always_inline)) inline void CommonDyadicGeometry54(SharedArgument0& product, SharedArgument1& relA, SharedArgument2& coordinates, SharedArgument3& temp, SharedArgument4& relC, SharedArgument5& endpoint, SharedArgument6& pairs, SharedArgument7& upper, SharedArgument8& box, SharedArgument9& left, SharedArgument10& right) {
Mul(product,relA,relA,coordinates);Mul(temp,relC,relC,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(endpoint,product,3,pairs,1,1,1);
        WholeReduceSum(upper,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        Min(box,left,right,coordinates);Max(product,left,right,coordinates);PipeBarrier<PIPE_V>();
}
