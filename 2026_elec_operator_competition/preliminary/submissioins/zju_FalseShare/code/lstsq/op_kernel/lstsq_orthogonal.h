/*
 * Copyright (c) 2026 FalseShare contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "lstsq_cube.h"

namespace Orthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,Q32=266240,
 BP=270336,XP=278528,R=286720,DX=294912,REDUCE=303104,NT=303360,NH=307456,AD=319744,XD=418048,PRODUCTS=430336,EA=438528,EX=438784,STRIDE=kOrthogonalWorkspaceFloats;
// Phase measurements put one packing chunk near four direct-QR rows.
// Move only the uncovered packing work into the longest prior Cube phases.
__aicore__ inline uint32_t EarlyPackingChunks(uint32_t n) {
    const uint32_t directRows=(n-1)%16+1,available=(n-1)/64;
    const uint32_t wanted=(24-directRows+7)/8;
    return wanted<available?wanted:available;
}
static_assert(NH+12288==AD);
static_assert(EX+16==STRIDE);
class Vector {
    AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    AscendC::LocalTensor<float> w;
    uint32_t lane,teams;float inputScale=1.0f;
public:
    uint32_t batch;
    __aicore__ inline void Init(uint32_t batches) {
        if (batches == 0) return;
        uint32_t id=AscendC::GetBlockIdx();batch=id%batches;lane=id/batches;teams=(2*AscendC::GetBlockNum()+batches-1-batch)/batches;
        pipe.InitBuffer(buf,192*1024);w=buf.Get<float>();
        if(lane<8) {
            auto identity=w[36864];AscendC::Duplicate(identity,0.0f,512);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<8;++j)identity.SetValue(j*64+lane*8+j,1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void TailGuard32(AscendC::GlobalTensor<float> defect,AscendC::GlobalTensor<int32_t> decision) {
        if(lane)return;
        AscendC::DataCopy(w,defect,1024);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<32;++r)w.SetValue(r*32+r,1.0f+w.GetValue(r*32+r));AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Abs(w[1024],w,1024);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(w[2048],w[1024],32,32,1,1,4);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(w[2112],w[2048],32,1,1,1,8);AscendC::WholeReduceSum(w[2120],w[2048],32,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
        const float bound=w.GetValue(2112),total=w.GetValue(2120);
        if(!(bound<=0.015625f && total>=0.0f && total<=0.5f)) {
            auto flag=w[2176].ReinterpretCast<int32_t>();AscendC::Duplicate(flag,(int32_t)0,8);AscendC::PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetAtomicAdd<int32_t>();AscendC::DataCopy(decision,flag,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetAtomicNone();
        }
    }
    __aicore__ inline void TailGuard16(AscendC::GlobalTensor<float> defect,AscendC::GlobalTensor<int32_t> decision) {
        if(lane)return;
        AscendC::DataCopy(w,defect,256);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<16;++r)w.SetValue(r*16+r,1.0f+w.GetValue(r*16+r));AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Abs(w[1024],w,256);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(w[2048],w[1024],16,16,1,1,2);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(w[2112],w[2048],16,1,1,1,8);AscendC::WholeReduceSum(w[2120],w[2048],16,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
        const float bound=w.GetValue(2112),total=w.GetValue(2120);
        if(!(bound<=0.015625f && total>=0.0f && total<=0.25f)) {
            auto flag=w[2176].ReinterpretCast<int32_t>();AscendC::Duplicate(flag,(int32_t)0,8);AscendC::PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetAtomicAdd<int32_t>();AscendC::DataCopy(decision,flag,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetAtomicNone();
        }
    }
    __aicore__ inline void PackTail16(AscendC::GlobalTensor<float> inverse,AscendC::GlobalTensor<float> cross) {
        if(lane!=1)return;
        AscendC::DataCopy(w,cross,256);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(inverse[16],w,AscendC::DataCopyExtParams{16,64,0,64,0});AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void PackTail32(AscendC::GlobalTensor<float> inverse,AscendC::GlobalTensor<float> cross) {
        if(lane!=1)return;
        AscendC::DataCopy(w,cross,1024);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(inverse[32],w,AscendC::DataCopyExtParams{32,128,0,128,0});AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void TailQR(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> src,uint32_t real) {
        if(lane)return;
        AscendC::DataCopy(w,src,real*256);
        OrthogonalTailQR(w,w,real,2);
        AscendC::DataCopy(dst,w,real*256);AscendC::PipeBarrier<PIPE_ALL>();
    }
    template<bool Rhs> __aicore__ inline float Split6(AscendC::LocalTensor<float> v,AscendC::LocalTensor<int8_t> digits) {
        auto q=w[256],ab=w[512],mx=w[768];auto hd=w[1024].ReinterpretCast<half>();
        AscendC::Abs(ab,v,256);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(mx,ab,64,4,1,1,8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);uint32_t mb,mi;AscendC::GetReduceRepeatMaxMinSpr<float>(mb,mi);
        int32_t exponent=(int32_t)((mb>>23)&255)-127-5;
        if((mb&0x7fffffff)==0){exponent=0;AscendC::Duplicate(v,0.0f,256);AscendC::PipeBarrier<PIPE_V>();}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;AscendC::Muls(v,v,scale.value,256);AscendC::PipeBarrier<PIPE_V>();}
        for(uint32_t d=0;d<6;++d) {
            AscendC::Adds(q,v,12582912.0f,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Adds(q,q,-12582912.0f,256);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(hd,q,AscendC::RoundMode::CAST_NONE,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(digits[(Rhs?5-d:d)*256],hd,AscendC::RoundMode::CAST_TRUNC,256);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(v,v,q,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(v,v,128.0f,256);AscendC::PipeBarrier<PIPE_V>();
        }
        return (float)exponent;
    }
    __aicore__ __attribute__((noinline)) void SplitAChunk(AscendC::GlobalTensor<int8_t> digits,AscendC::GlobalTensor<float> exponents,AscendC::GlobalTensor<float> matrix,uint32_t panel,uint32_t n) {
        SplitOrthogonalInput<6,1536>(w,lane,teams,inputScale,digits,exponents,matrix,panel,n);
    }
    __aicore__ __attribute__((noinline)) void SplitXPacked4(AscendC::GlobalTensor<int8_t> digits,AscendC::GlobalTensor<float> exponents,AscendC::GlobalTensor<float> solution,uint32_t k) {
        auto input=w[10000],e=w[8000];auto d=w[1536].ReinterpretCast<int8_t>();auto ix=w[19000].ReinterpretCast<int32_t>();auto packed=w[20000].ReinterpretCast<int8_t>();
        AscendC::CreateVecIndex(ix,(int32_t)0,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(ix,ix,(int32_t)128,256);AscendC::PipeBarrier<PIPE_ALL>();
        if(lane<k){AscendC::DataCopy(input,solution,8192);AscendC::PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<4;c+=teams) {
            if(c<k)AscendC::Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else AscendC::Duplicate(w,0.0f,256);AscendC::PipeBarrier<PIPE_ALL>();
            float exponent=0;if(c<k)exponent=Split6<true>(w,d);else AscendC::Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,384);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Duplicate(packed.ReinterpretCast<int32_t>(),(int32_t)0,3072);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight)AscendC::DataCopy(packed[weight*1536],d[(5-weight)*256],(weight+1)*256);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(digits[c*1536],packed,AscendC::DataCopyExtParams{8,1536,0,4608,0});AscendC::PipeBarrier<PIPE_ALL>();
            e.SetValue(0,exponent);e.SetValue(8,0.0f);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(exponents[c],e,AscendC::DataCopyExtParams{1,4,0,0,0});AscendC::DataCopyPad(exponents[c+4],e[8],AscendC::DataCopyExtParams{1,4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6Packed4(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> initial,AscendC::GlobalTensor<float> reserve,AscendC::GlobalTensor<float> next,AscendC::GlobalTensor<int32_t> products,AscendC::GlobalTensor<float> expA,AscendC::GlobalTensor<float> expX,AscendC::GlobalTensor<float> rhs) {
        constexpr uint32_t L=128;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];auto packed=w[7000],allProducts=w[9000],rhsTile=w[10400];auto ix=w[4608].ReinterpretCast<int32_t>();
        AscendC::DataCopy(ae,expA,256);AscendC::DataCopy(xe,expX,16);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(ix,(int32_t)0,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftRight(ix,ix,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(ix,ix,(int32_t)2,L);AscendC::PipeBarrier<PIPE_ALL>();
        auto index=w[11000].ReinterpretCast<int32_t>();auto columns=w[11504].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(index,(int32_t)0,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftRight(columns,index,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(columns,columns,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(columns,index,columns,L);AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(index,columns,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(ex,xe,index.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
        auto outputIndex=w[32000].ReinterpretCast<int32_t>(),outputRows=w[33024].ReinterpretCast<int32_t>(),rhsIndex=w[34048].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(rhsIndex,(int32_t)0,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(rhsIndex,rhsIndex,ix,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(rhsIndex,rhsIndex,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();
        AscendC::CreateVecIndex(outputIndex,(int32_t)0,1024);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftRight(outputRows,outputIndex,(int32_t)5,1024);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(outputRows,outputRows,(int32_t)28,1024);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(outputIndex,outputIndex,outputRows,1024);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(outputIndex,outputIndex,(int32_t)2,1024);AscendC::PipeBarrier<PIPE_ALL>();
        uint64_t outputMask[2]={0x0000000f0000000fULL,0};
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            uint32_t row=tile*32;AscendC::Gather(ea,ae[row],ix.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(low,0.0f,L);AscendC::DataCopyPad(rhsTile,rhs[row*32],AscendC::DataCopyExtParams{32,32,96,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();AscendC::Gather(high,rhsTile,rhsIndex.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(allProducts.ReinterpretCast<int32_t>(),products[row*32],1024);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CreateVecIndex(columns,(int32_t)0,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftRight(index,columns,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(gs.ReinterpretCast<int32_t>(),index,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(columns,columns,gs.ReinterpretCast<int32_t>(),L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(index,index,(int32_t)7,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(columns,columns,(int32_t)2,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(index,index,columns,L);AscendC::PipeBarrier<PIPE_ALL>();
            auto q0=w[14000].ReinterpretCast<int32_t>(),q1=w[14256].ReinterpretCast<int32_t>(),q2=w[14512].ReinterpretCast<int32_t>(),q3=w[14768].ReinterpretCast<int32_t>(),q4=w[15024].ReinterpretCast<int32_t>(),q5=w[15280].ReinterpretCast<int32_t>();
            AscendC::Adds(columns,index,(int32_t)(0*16),L);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(q0,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(columns,index,(int32_t)(1*16),L);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(q1,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(columns,index,(int32_t)(2*16),L);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(q2,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(columns,index,(int32_t)(3*16),L);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(q3,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(columns,index,(int32_t)(4*16),L);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(q4,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(columns,index,(int32_t)(5*16),L);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(q5,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(q0,q0,(int32_t)7,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(q0,q0,q1,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(q2,q2,(int32_t)7,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(q2,q2,q3,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(q4,q4,(int32_t)7,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(q4,q4,q5,L);AscendC::PipeBarrier<PIPE_V>();
            // Canonical base-16384 digits. All intermediate magnitudes are below 2^30.
            AscendC::ShiftRight(q5,q4,(int32_t)14,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(q2,q2,q5,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(q5,q5,(int32_t)14,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(q4,q4,q5,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(q3,q2,(int32_t)14,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(q0,q0,q3,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(q3,q3,(int32_t)14,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(q2,q2,q3,L);AscendC::PipeBarrier<PIPE_V>();
            // Split the leading int32 at its FP32 rounding boundary; pack its exact remainder with the middle digit.
            AscendC::Cast(gs,q0,AscendC::RoundMode::CAST_NONE,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(q1,gs,AscendC::RoundMode::CAST_RINT,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(q1,q0,q1,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(q1,q1,(int32_t)14,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(q1,q1,q2,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(q2.ReinterpretCast<float>(),q1,AscendC::RoundMode::CAST_NONE,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(q4.ReinterpretCast<float>(),q4,AscendC::RoundMode::CAST_NONE,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(factor,ea,ex,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Adds(factor,factor,120.0f,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(factor.ReinterpretCast<int32_t>(),factor,AscendC::RoundMode::CAST_RINT,L);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(factor.ReinterpretCast<int32_t>(),factor.ReinterpretCast<int32_t>(),(int32_t)23,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(term,gs,factor,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(tmp,high,term,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,tmp,high,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,tmp,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,high,v,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,term,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(u,u,v,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(low,low,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(high,tmp,1.0f,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(factor,factor,0.00006103515625f,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(term,q2.ReinterpretCast<float>(),factor,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(tmp,high,term,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,tmp,high,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,tmp,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,high,v,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,term,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(u,u,v,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(low,low,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(high,tmp,1.0f,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(factor,factor,0.00006103515625f,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(term,q4.ReinterpretCast<float>(),factor,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(tmp,high,term,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,tmp,high,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,tmp,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,high,v,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,term,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(u,u,v,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(low,low,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(high,tmp,1.0f,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(high,high,low,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Duplicate(packed,0.0f,1024);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(packed,high,outputIndex.ReinterpretCast<uint32_t>(),0,outputMask,16,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(dst[row*32],packed,1024);AscendC::DataCopy(initial[row*32],packed,1024);AscendC::DataCopy(reserve[row*32],packed,1024);AscendC::DataCopy(next[row*32],packed,1024);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void AccurateResidual(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> matrix,AscendC::GlobalTensor<float> solution,AscendC::GlobalTensor<float> rhs,AscendC::GlobalTensor<float> mirror,AscendC::GlobalTensor<float> reserve,AscendC::GlobalTensor<float> next,uint32_t k) {
        AccurateRows<256,16> dot;dot.Init(w);
        auto packedX=w[45312],rhsTile=w[47648],result=w[48160];auto xix=w[47360].ReinterpretCast<uint32_t>(),bix=w[47616].ReinterpretCast<uint32_t>();
        AscendC::DataCopyPad(packedX,solution,AscendC::DataCopyExtParams{256,k*4,(32-k)*4,0,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)(8-k),0});for(uint32_t j=0;j<256;++j)xix.SetValue(j,j*8*4);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            AscendC::DataCopy(dot.raw,matrix[row*256],4096);AscendC::DataCopy(rhsTile,rhs[row*32],512);AscendC::Duplicate(result,0.0f,512);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t c=0;c<k;++c) {
                AscendC::Gather(dot.x,packedX,xix,c*4,256);AscendC::PipeBarrier<PIPE_V>();
                for(uint32_t r=1;r<16;++r)AscendC::DataCopy(dot.x[r*256],dot.x,256);AscendC::PipeBarrier<PIPE_ALL>();dot.AccurateDot(16);
                for(uint32_t r=0;r<16;++r)bix.SetValue(r,(r*32+c)*4);AscendC::PipeBarrier<PIPE_ALL>();AscendC::Gather(dot.a,rhsTile,bix,0,16);AscendC::PipeBarrier<PIPE_V>();
                dot.A(dot.h,dot.h,dot.a,16);dot.A(dot.h,dot.h,dot.l,16);AscendC::PipeBarrier<PIPE_ALL>();
                for(uint32_t r=0;r<16;++r)result.SetValue(r*32+c,dot.h.GetValue(r));AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::DataCopy(dst[row*32],result,512);AscendC::DataCopy(mirror[row*32],result,512);AscendC::DataCopy(reserve[row*32],result,512);AscendC::DataCopy(next[row*32],result,512);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackCachedB(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> initial,uint32_t k) {
        // Normalize retained the unscaled complete RHS at w[8448].
        // Offset r*32+c maps to r*k+c; only c<k is gathered.
        auto ix=w[16384].ReinterpretCast<int32_t>(),rows=w[16896].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(ix,(int32_t)0,512);AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(rows,ix,(int32_t)5,512);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(rows,rows,(int32_t)(32-k),512);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(ix,ix,rows,512);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(ix,ix,(int32_t)2,512);AscendC::PipeBarrier<PIPE_V>();
        uint64_t columns=(1ULL<<k)-1,mask[2]={columns|(columns<<32),0};
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            AscendC::Duplicate(w,0.0f,512);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(w,w[8448],ix.ReinterpretCast<uint32_t>(),row*k*4,mask,8,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(w,w,-inputScale,512);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(dst[row*32],w,512);AscendC::DataCopy(initial[row*32],w,512);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Normalize(AscendC::GlobalTensor<float> normalized,AscendC::GlobalTensor<float> current,AscendC::GlobalTensor<float> src,AscendC::GlobalTensor<float> rhs,uint32_t k,uint32_t n) {
        NormalizeOrthogonalInput(w,lane,teams,inputScale,normalized,current,src,rhs,k,n);
    }
    __aicore__ inline void Finish(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> x,AscendC::GlobalTensor<float> dx,uint32_t k,uint32_t n,GM_ADDR rr,bool hasResidual) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*32+c)*4);
        AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            AscendC::DataCopy(w,x[row*32],512);AscendC::DataCopy(w[4096],dx[row*32],512);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(w,w,w[4096],512);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Gather(w[8192],w,ix,0,16*k);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(dst[row*k],w[8192],AscendC::DataCopyExtParams{1,(n-row<16?n-row:16)*k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
        }
        if(lane==0 && hasResidual) {
            AscendC::GlobalTensor<float> gr;gr.SetGlobalBuffer((__gm__ float*)rr);
            AscendC::Duplicate(w,0.0f,8);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gr[(uint64_t)batch*k],w,AscendC::DataCopyExtParams{1,k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t B=32,uint32_t S=64> __aicore__ inline void InvertSmall(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> src,bool negative=false) {
        if(lane)return;
        constexpr uint32_t R=2*B+8;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        auto recip=w[4160],minusOne=w[4224];
        AscendC::Duplicate(minusOne,-1.0f,8);
        AscendC::Duplicate(panel,0.0f,B*R);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(panel,src,AscendC::DataCopyExtParams{B,B*4,(S-B)*4,(R-B)/8,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
        AscendC::PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,1.0f);ix.SetValue(r,r*R*4);}AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            AscendC::Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)B,1,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(weights,col,B/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(recip,minusOne,weights[j*8],(uint64_t)8,1,AscendC::BinaryRepeatParams{1,1,1,8,8,8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(pivot,panel[j*R],recip,(uint64_t)(2*B),1,AscendC::BinaryRepeatParams{1,1,0,8,8,0});AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(panel,pivot,weights,(uint64_t)(2*B),(uint8_t)B,AscendC::BinaryRepeatParams{1,1,0,R/8,0,1});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(panel[j*R],pivot,-1.0f,(uint64_t)(2*B),1,AscendC::UnaryRepeatParams{1,1,8,8});AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if(negative){AscendC::Muls(panel,panel,-1.0f,B*R);AscendC::PipeBarrier<PIPE_ALL>();}
        AscendC::DataCopyPad(dst,panel[B],AscendC::DataCopyExtParams{B,B*4,(R-B)/8,(S-B)*4,0});AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void NegateBlock(AscendC::GlobalTensor<float> src,uint32_t owner=1) {
        if(lane!=owner)return;
        AscendC::DataCopyPad(w,src,AscendC::DataCopyExtParams{32,128,128,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(w,w,-1.0f,1024);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(src,w,AscendC::DataCopyExtParams{32,128,0,128,0});AscendC::PipeBarrier<PIPE_ALL>();
    }


    __aicore__ inline void NewtonPrepare(AscendC::GlobalTensor<half> packed,AscendC::GlobalTensor<float> gram) {
        if(lane>=8)return;
        auto value=w,ab=w[4096],sum=w[8192],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        AscendC::DataCopy(value,gram,4096);AscendC::PipeBarrier<PIPE_ALL>();AscendC::Abs(ab,value,4096);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(sum,ab,64,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(sum[128],sum,64,1,1,1,8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();float alpha=1.8f/sum.GetValue(128);
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            AscendC::DataCopy(value,gram[row*64],512);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(h,value,AscendC::RoundMode::CAST_RINT,512);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ab,h,AscendC::RoundMode::CAST_NONE,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(ab,value,ab,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[2048],ab,AscendC::RoundMode::CAST_RINT,512);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(eye,w[36864],1.0f,512);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(h[1024],eye,AscendC::RoundMode::CAST_RINT,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(ab,value,-alpha,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(ab,eye,ab,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[1536],ab,AscendC::RoundMode::CAST_RINT,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(value,eye,-alpha,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[512],value,AscendC::RoundMode::CAST_RINT,512);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(value,eye,2.0f,512);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[2560],value,AscendC::RoundMode::CAST_RINT,512);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(packed[row*64],h,AscendC::DataCopyParams{4,32,0,224});AscendC::DataCopy(packed[16384+row*64],h[2048],512);AscendC::DataCopy(packed[20480+row*64],h[2560],512);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ZeroDecision(AscendC::GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();AscendC::Duplicate(flag,(int32_t)0,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(decision,flag,8);AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuard(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<int32_t> decision,bool product=false) {
        auto ab=w[4096],sum=w[8192];
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            AscendC::DataCopy(w,dst[row*64],512);AscendC::PipeBarrier<PIPE_ALL>();
            if(product){AscendC::Add(w,w,w[36864],512);AscendC::PipeBarrier<PIPE_V>();}
            AscendC::Abs(ab,w,512);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(sum,ab,64,8,1,1,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceMax(sum[128],sum,8,1,1,1,8);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(sum[192],sum,8,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
            float bound=sum.GetValue(128),total=sum.GetValue(192);
            if(!(bound<=0.015625f && total>=0.0f && total<=2.0f)) {
                auto flag=sum[128].ReinterpretCast<int32_t>();AscendC::Duplicate(flag,(int32_t)0,8);AscendC::PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetAtomicAdd<int32_t>();AscendC::DataCopy(decision,flag,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetAtomicNone();
            }
        }
    }
    __aicore__ inline void NewtonDefect(AscendC::GlobalTensor<float> dst) {
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            AscendC::DataCopy(w,dst[row*64],512);AscendC::PipeBarrier<PIPE_ALL>();AscendC::Muls(w,w,-1.0f,512);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<8;++j){uint32_t off=j*64+row+j;w.SetValue(off,2.0f+w.GetValue(off));}
            AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(dst[row*64],w,512);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

};
template<bool Cube> __aicore__ __attribute__((noinline)) bool CheckRefined(
    std::conditional_t<Cube,NativeMM,Vector> &op,AscendC::GlobalTensor<float> mem,
    AscendC::GlobalTensor<int32_t> decisions,uint32_t batches,uint32_t giBase) {
    // The first decision remains immutable while other cores read it.
    auto actual=decisions[8];
    if constexpr(!Cube)op.ZeroDecision(actual);
    if constexpr(Cube)op.template Run<64,64,64>(mem[G],mem[giBase],mem[NT],batches,STRIDE,STRIDE,STRIDE);
    AscendC::SyncAll<false>();
    if constexpr(!Cube)op.NewtonGuard(mem[NT],actual,true);
    AscendC::SyncAll<false>();
    AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(actual);
    return actual.GetValue(0)==0;
}
template<bool Cube> __aicore__ inline void Tail(std::conditional_t<Cube,NativeMM,Vector> &op,
    AscendC::GlobalTensor<float> mem,AscendC::GlobalTensor<float> A,AscendC::GlobalTensor<int8_t> ad,GM_ADDR workspace,
    uint32_t offset,uint32_t real,uint32_t n,uint32_t k,uint32_t batches) {
    constexpr uint32_t INV=D+32768,GRAM=D+33792,CROSS=D+34304,DEFECT=D+34560;
    bool split=false;
    if(real>16 && real<=32) {
        AscendC::GlobalTensor<int32_t> decision;decision.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+224);
        if constexpr(!Cube)op.ZeroDecision(decision);
        if constexpr(Cube)op.template Run<16,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GRAM],batches,STRIDE,STRIDE,STRIDE,256,256,32);
        AscendC::SyncAll<false>();
        if constexpr(!Cube)op.template InvertSmall<16,32>(mem[INV],mem[GRAM],true);
        AscendC::SyncAll<false>();
        if constexpr(Cube)op.template Run<16,16,16,false,false,false>(mem[GRAM],mem[INV],mem[DEFECT],batches,STRIDE,STRIDE,STRIDE,32,32,16);
        AscendC::SyncAll<false>();
        if constexpr(!Cube)op.TailGuard16(mem[DEFECT],decision);
        AscendC::SyncAll<false>();
        AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(decision);
        split=decision.GetValue(0)==0;
        if(split) {
            if constexpr(Cube)op.template Run<16,16,16,false,false,false>(mem[INV],mem[GRAM+16],mem[CROSS],batches,STRIDE,STRIDE,STRIDE,32,32,16);
            if constexpr(Cube)SyncCube();
            if constexpr(Cube)op.template Run<16,256,16,true,false,true>(mem[CROSS],mem[CT+offset*256],mem[CT+(offset+16)*256],batches,STRIDE,STRIDE,STRIDE,16,256,256);
            AscendC::SyncAll<false>();
        }
    }
    if constexpr(!Cube) {
        op.TailQR(mem[CT+(offset+(split?16:0))*256],mem[CT+(offset+(split?16:0))*256],real-(split?16:0));
        if(split)op.PackTail16(mem[INV],mem[CROSS]);
        if(k<=4)for(uint32_t panel=EarlyPackingChunks(n)*32;panel<160;panel+=32)op.SplitAChunk(ad,mem[EA],A,panel,n);
    }
    AscendC::SyncAll<false>();
    if(split) {
        if constexpr(Cube)op.template Run<16,256,32,false,false,false>(mem[INV],mem[CT+offset*256],mem[CT+offset*256],batches,STRIDE,STRIDE,STRIDE,32,256,256);
        AscendC::SyncAll<false>();
    }
}
template<bool Cube> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgLstsqTilingData &t,GM_ADDR rr) {
    // Uniform metadata guard: every Cube and Vector exits before any barrier.
    if (t.batches == 0) return;
    const uint32_t batches=t.batches,k=t.k,n=t.n;
    AscendC::GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    std::conditional_t<Cube,NativeMM,Vector> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*n*n];B=B[(uint64_t)batch*n*k];X=X[(uint64_t)batch*n*k];}
    AscendC::GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    AscendC::GlobalTensor<int8_t> ad,xd;AscendC::GlobalTensor<int32_t> products;
    ad.SetGlobalBuffer((__gm__ int8_t*)workspace+((uint64_t)batch*STRIDE+AD)*4);
    xd.SetGlobalBuffer((__gm__ int8_t*)workspace+((uint64_t)batch*STRIDE+XD)*4);
    products.SetGlobalBuffer((__gm__ int32_t*)workspace+(uint64_t)batch*STRIDE+PRODUCTS);
    if constexpr(!Cube){op.Normalize(mem[AN],mem[CT],A,B,k,n);op.PackCachedB(mem[BP],mem[DX],k);}
    AscendC::SyncAll<false>();
    const uint32_t lastPanel=(n-1)/64*64;
    // The fixed capacity has at most three regular panels followed by one real tail.
    for(uint32_t p=0;p<lastPanel && p<192;p+=64) {
        const uint32_t giBase=D+p*64;
        if constexpr(Cube)op.template Run<64,64,256,false,true>(mem[CT+p*256],mem[CT+p*256],mem[G],batches,STRIDE,STRIDE,STRIDE);
        AscendC::SyncAll<false>();
        bool useExact=false;
        {
            AscendC::GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128+(p/64)*32);
            if constexpr(!Cube)op.ZeroDecision(decisions);
            if constexpr(!Cube)op.NewtonPrepare(halfMem,mem[G]);
            AscendC::SyncAll<false>();
            if constexpr(Cube){// Narrower remaining row space makes the Gram spectrum less favorable.
            // This is an iteration budget only; the same defect guard decides use.
            const uint32_t remaining=n-p,steps=remaining>=256?6:(remaining>=192?7:(remaining>=128?9:13));
            op.Newton64(halfMem,mem[giBase],mem[NT],batches,STRIDE,steps);op.CrossGramImplicit256(mem[CT],mem[D+16384],batches,STRIDE,p);}
            if constexpr(!Cube)if(k<=4) {
                const uint32_t first=(n-1)/64-EarlyPackingChunks(n);
                if(p/64>=first)op.SplitAChunk(ad,mem[EA],A,(p/64-first)*32,n);
            }
            AscendC::SyncAll<false>();
            if constexpr(!Cube)op.NewtonGuard(mem[NT],decisions);
            if constexpr(Cube)op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);
            AscendC::SyncAll<false>();
            AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(decisions);
            useExact=decisions.GetValue(0)!=0;
            // The first guard precedes the final refinement. Validate the
            // returned inverse with the same bound before exact elimination.
            if(useExact)useExact=!CheckRefined<Cube>(op,mem,decisions,batches,giBase);
        }
        if(useExact) {
        if constexpr(!Cube){op.template InvertSmall<>(mem[giBase],mem[G]);op.NegateBlock(mem[G+2048]);}
        AscendC::SyncAll<false>();
        if constexpr(Cube)op.Schur32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        AscendC::SyncAll<false>();
        if constexpr(!Cube){op.template InvertSmall<>(mem[giBase+2080],mem[G+2080],true);op.NegateBlock(mem[G+32]);op.NegateBlock(mem[giBase],2);}
        AscendC::SyncAll<false>();
        if constexpr(Cube)op.Assemble32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        if constexpr(Cube)SyncCube();
        if constexpr(Cube){op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);SyncCube();}
        }
        if(p+64<256) {
            if constexpr(Cube) {
                if(p==0)op.template Run<192,256,64,true,false,true>(mem[H+64],mem[CT],mem[CT+64*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
                else if(p==64)op.template Run<128,256,64,true,false,true>(mem[H+p*256+128],mem[CT+p*256],mem[CT+128*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
                else op.template Run<64,256,64,true,false,true>(mem[H+p*256+192],mem[CT+p*256],mem[CT+192*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
            }
            if constexpr(Cube)SyncCube();
        }
        }
    {
        const uint32_t p=lastPanel;
            if constexpr(Cube)op.ReprojectTail(mem[CT],mem[D],mem[H],mem[D+28672],mem[D+16384],batches,STRIDE,p);
            AscendC::SyncAll<false>();
            const uint32_t real=n-p;
            if(real>32) {
                AscendC::GlobalTensor<int32_t> decision;decision.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+224);
                if constexpr(!Cube)op.ZeroDecision(decision);
                if constexpr(Cube)op.TailProducts32(mem[CT+p*256],mem[G],mem[Q32],batches,STRIDE);
                AscendC::SyncAll<false>();
                if constexpr(!Cube)op.template InvertSmall<>(mem[D+12288],mem[G],true);
                if constexpr(Cube)op.ComposeForward(mem[H],batches,STRIDE,p);
                AscendC::SyncAll<false>();
                if constexpr(Cube)op.template Run<32,32,32,false,false,false>(mem[G],mem[D+12288],mem[NT],batches,STRIDE,STRIDE,STRIDE,64,64,32);
                AscendC::SyncAll<false>();
                if constexpr(!Cube)op.TailGuard32(mem[NT],decision);
                AscendC::SyncAll<false>();
                AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(decision);
                const bool fallback=decision.GetValue(0)!=0;
                if(fallback) {
                    Tail<Cube>(op,mem,A,ad,workspace,p,real,n,k,batches);
                } else {
                    if constexpr(Cube)op.template Run<32,32,32,false,false,false>(mem[D+12288],mem[Q32],mem[Q32],batches,STRIDE,STRIDE,STRIDE,64,32,32);
                    if constexpr(Cube)SyncCube();
                    if constexpr(Cube)op.template Run<32,256,32,true,false,true>(mem[Q32],mem[CT+p*256],mem[CT+(p+32)*256],batches,STRIDE,STRIDE,STRIDE,32,256,256);
                    AscendC::SyncAll<false>();
                    Tail<Cube>(op,mem,A,ad,workspace,p+32,real-32,n,k,batches);
                    if constexpr(!Cube)op.PackTail32(mem[D+12288],mem[Q32]);
                    AscendC::SyncAll<false>();
                    // [negative inverse, negative projection] applies [W0; negative tail factor].
                    // Each output tile reads and overwrites only its own columns of W0.
                    if constexpr(Cube)op.template Run<32,256,64,false,false,false>(mem[D+12288],mem[CT+p*256],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE,64,256,256);
                    if constexpr(Cube)SyncCube();
                }
            } else {
                if constexpr(Cube)op.ComposeForward(mem[H],batches,STRIDE,p);
                Tail<Cube>(op,mem,A,ad,workspace,p,real,n,k,batches);
            }

            }
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[BP],mem[DX],batches,STRIDE,(n+63)/64);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);
    if constexpr(Cube)SyncCube();
    constexpr uint32_t R0=D+32768,R1=D+49152;
    AscendC::SyncAll<false>();
    if(k<=4) {
        if constexpr(!Cube)op.SplitXPacked4(xd,mem[EX],mem[XP],k);
        AscendC::SyncAll<false>();
        if constexpr(Cube)op.Int8Packed4(ad,xd,products,batches,STRIDE);
        AscendC::SyncAll<false>();
        if constexpr(!Cube)op.Assemble6Packed4(mem[R],mem[DX],mem[R0],mem[R1],products,mem[EA],mem[EX],mem[BP]);
    } else {
        if constexpr(!Cube)op.AccurateResidual(mem[R],mem[AN],mem[XP],mem[BP],mem[DX],mem[R0],mem[R1],k);
    }
    AscendC::SyncAll<false>();
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[R],mem[DX],batches,STRIDE,(n+63)/64);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[R],batches,STRIDE,STRIDE,STRIDE);
    if constexpr(Cube)SyncCube();
    // Both copies retain q0 while the first forward solve consumes DX.
    // One product updates immutable input R0 and the next forward destination R1.
    if constexpr(Cube)op.template Run<256,32,256,false,false,true>(mem[AN],mem[R],mem[R0],batches,STRIDE,STRIDE,STRIDE,256,32,32,R1-R0);
    if constexpr(Cube)SyncCube();
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[R0],mem[R1],batches,STRIDE,(n+63)/64);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true,false,true>(mem[CT],mem[R1],mem[R],batches,STRIDE,STRIDE,STRIDE);
    AscendC::SyncAll<false>();
    if constexpr(!Cube)op.Finish(X,mem[XP],mem[R],k,n,rr,rr && t.residualElements>=(uint64_t)batches*k);
}
}

// Row Gram panels, adapted from solve bfdd2d4, with accurate residual refinement.
