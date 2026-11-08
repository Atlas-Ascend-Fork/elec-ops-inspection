/*
 * Copyright (c) 2026 LittleFairy.
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
#include "linalg_solve_matmul.h"

namespace AscendC {
namespace SolveDetail {
namespace NarrowOrthogonal {
class Vector : public OrthogonalVectorState {
public:
    __aicore__ inline void Init(uint32_t batches) {
        if(batches==0)return;
        InitializeState(batches);
        if(lane<6) {
            auto identity=w[36864];Duplicate(identity,0.0f,1536);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)identity.SetValue(j*96+lane*16+j,1.0f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackCachedB(GlobalTensor<float> dst,GlobalTensor<float> initial,uint32_t k) {
        // NormalizeSeed retained unscaled RHS, zero-padded to 256 rows, at w[8448].
        // Offset r*16+c maps to r*k+c; only c<k is gathered.
        auto ix=w[16384].ReinterpretCast<int32_t>(),rows=w[16896].ReinterpretCast<int32_t>();
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();
        ShiftRight(rows,ix,(int32_t)4,256);PipeBarrier<PIPE_V>();
        Muls(rows,rows,(int32_t)(16-k),256);PipeBarrier<PIPE_V>();
        Sub(ix,ix,rows,256);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,256);PipeBarrier<PIPE_V>();
        uint64_t columns=(1ULL<<k)-1,mask[2]={columns|(columns<<16)|(columns<<32)|(columns<<48),0};
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            Duplicate(w,0.0f,256);PipeBarrier<PIPE_V>();
            Gather(w,w[8448],ix.ReinterpretCast<uint32_t>(),row*k*4,mask,4,8);PipeBarrier<PIPE_V>();
            Muls(w,w,-inputScale,256);PipeBarrier<PIPE_ALL>();
            DataCopy(dst[row*16],w,256);DataCopy(initial[row*16],w,256);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackB(GlobalTensor<float> dst,GlobalTensor<float> initial,GlobalTensor<float> src,uint32_t k,uint32_t n) {
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            Duplicate(w,0.0f,512);PipeBarrier<PIPE_ALL>();
            if(row<n)DataCopyPad(w,src[row*k],DataCopyExtParams{(uint16_t)(n-row<16?n-row:16),k*4,0,3,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-k),0});
            PipeBarrier<PIPE_ALL>();Muls(w,w,-inputScale,512);PipeBarrier<PIPE_ALL>();
            DataCopy(dst[row*32],w,512);DataCopy(initial[row*32],w,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Normalize(GlobalTensor<float> normalized,GlobalTensor<float> src,GlobalTensor<float> tmp,uint32_t n) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams){
            uint32_t row=tile*16,rows=row<n?(n-row<16?n-row:16):0;
            Duplicate(w[count],0.0f,4096);PipeBarrier<PIPE_ALL>();
            if(rows)DataCopyPad(w[count],src[row*n],DataCopyExtParams{(uint16_t)rows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});count+=4096;
        }
        PipeBarrier<PIPE_ALL>();uint32_t mb=0,mi=0;
        if(count) {
            Abs(w[8192],w,count);PipeBarrier<PIPE_V>();WholeReduceMax(w[16384],w[8192],64,(uint8_t)(count/64),1,1,8);
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);GetReduceRepeatMaxMinSpr<float>(mb,mi);
        }
        w[20000].ReinterpretCast<uint32_t>().SetValue(0,mb);PipeBarrier<PIPE_ALL>();DataCopy(tmp[lane*8],w[20000],8);PipeBarrier<PIPE_ALL>();
        SyncAll<true>();DataCopy(w[22000],tmp,teams*8);PipeBarrier<PIPE_ALL>();
        uint32_t maxBits=0;for(uint32_t i=0;i<teams;++i){uint32_t v=w[22000].ReinterpretCast<uint32_t>().GetValue(i*8);maxBits=v>maxBits?v:maxBits;}
        int32_t exponent=(int32_t)((maxBits>>23)&255)-127;exponent=exponent>126?126:(exponent<-126?-126:exponent);
        union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;inputScale=scale.value;
        if(count){Muls(w,w,inputScale,count);PipeBarrier<PIPE_ALL>();}
        count=0;for(uint32_t tile=lane;tile<16;tile+=teams){
            for(uint32_t row=tile*16;row<tile*16+16;++row)if(row>=n)w.SetValue(count+(row%16)*256+row,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(normalized[tile*4096],w[count],4096);count+=4096;}PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void NormalizeSeed(GlobalTensor<float> normalized,GlobalTensor<float> src,GlobalTensor<float> rhs,uint32_t k,uint32_t n) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams){
            uint32_t row=tile*16,rows=row<n?(n-row<16?n-row:16):0;
            Duplicate(w[count],0.0f,4096);PipeBarrier<PIPE_ALL>();
            if(rows)DataCopyPad(w[count],src[row*n],DataCopyExtParams{(uint16_t)rows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});count+=4096;
        }
        PipeBarrier<PIPE_ALL>();
        // The common seed row and complete RHS are identical on all eight lanes.
        Duplicate(w[8192],0.0f,1280);PipeBarrier<PIPE_ALL>();
        DataCopyPad(w[8192],src,DataCopyExtParams{1,n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        DataCopyPad(w[8448],rhs,DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();Abs(w[10000],w[8192],1280);PipeBarrier<PIPE_V>();
        WholeReduceMax(w[16384],w[10000],64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t aBits,mi;GetReduceRepeatMaxMinSpr<float>(aBits,mi);
        WholeReduceMax(w[16416],w[10256],64,16,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t bBits;GetReduceRepeatMaxMinSpr<float>(bBits,mi);
        inputScale=BoundedInputScale(aBits,bBits);
        if(count){Muls(w,w,inputScale,count);PipeBarrier<PIPE_ALL>();}
        count=0;for(uint32_t tile=lane;tile<16;tile+=teams){
            for(uint32_t row=tile*16;row<tile*16+16;++row)if(row>=n)w.SetValue(count+(row%16)*256+row,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(normalized[tile*4096],w[count],4096);count+=4096;}PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Finish(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k,uint32_t n) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*32+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            DataCopy(w,x[row*32],512);DataCopy(w[4096],dx[row*32],512);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],512);PipeBarrier<PIPE_ALL>();
            uint32_t rows=n-row<16?n-row:16;
            Gather(w[8192],w,ix,0,rows*k);PipeBarrier<PIPE_ALL>();
            DataCopyPad(dst[row*k],w[8192],DataCopyExtParams{1,rows*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Finish16(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k,uint32_t n) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*16+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            DataCopy(w,x[row*16],256);DataCopy(w[4096],dx[row*16],256);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],256);PipeBarrier<PIPE_ALL>();
            uint32_t rows=n-row<16?n-row:16;
            Gather(w[8192],w,ix,0,rows*k);PipeBarrier<PIPE_ALL>();
            DataCopyPad(dst[row*k],w[8192],DataCopyExtParams{1,rows*k*4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    template<bool Rhs> __aicore__ inline float Split6(LocalTensor<float> v,LocalTensor<int8_t> digits) {
        auto q=w[256],ab=w[512],mx=w[768];auto hd=w[1024].ReinterpretCast<half>();
        Abs(ab,v,256);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);
        int32_t exponent=(int32_t)((mb>>23)&255)-127-5;
        if((mb&0x7fffffff)==0){exponent=0;Duplicate(v,0.0f,256);PipeBarrier<PIPE_V>();}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Muls(v,v,scale.value,256);PipeBarrier<PIPE_V>();}
        for(uint32_t d=0;d<6;++d) {
            Adds(q,v,12582912.0f,256);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,256);PipeBarrier<PIPE_V>();
            Cast(hd,q,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();Cast(digits[(Rhs?5-d:d)*256],hd,RoundMode::CAST_TRUNC,256);PipeBarrier<PIPE_V>();
            Sub(v,v,q,256);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,256);PipeBarrier<PIPE_V>();
        }
        return (float)exponent;
    }
    __aicore__ inline float Split6Batched(LocalTensor<float> v,LocalTensor<int8_t> digits) {
        auto allDigits=w[2048],ab=w[512],mx=w[768];auto hd=w[4096].ReinterpretCast<half>();
        Abs(ab,v,256);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);
        int32_t exponent=(int32_t)((mb>>23)&255)-127-5;
        if((mb&0x7fffffff)==0){exponent=0;Duplicate(v,0.0f,256);PipeBarrier<PIPE_V>();}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Muls(v,v,scale.value,256);PipeBarrier<PIPE_V>();}
        for(uint32_t d=0;d<6;++d) {
            auto q=allDigits[(5-d)*256];
            Adds(q,v,12582912.0f,256);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,256);PipeBarrier<PIPE_V>();
            Sub(v,v,q,256);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,256);PipeBarrier<PIPE_V>();
        }
        Cast(hd,allDigits,RoundMode::CAST_NONE,1536);PipeBarrier<PIPE_V>();
        Cast(digits,hd,RoundMode::CAST_TRUNC,1536);PipeBarrier<PIPE_V>();
        return (float)exponent;
    }
    __aicore__ __attribute__((noinline)) void SplitAChunk(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,GlobalTensor<float> matrix,uint32_t panel,uint32_t n) {
        if(lane==0)return;
        constexpr uint32_t L=2048;auto v=w,q=w[2048],ab=w[4096],mx=w[6144],e=w[12000];auto hd=w[8192].ReinterpretCast<half>();auto bd=w[10000].ReinterpretCast<int8_t>();
        uint32_t start=((panel/32)*(teams-1)+lane-1)*8;
        if(start<256) {
            uint32_t rows=start<n?(n-start<8?n-start:8):0;
            Duplicate(v,0.0f,L);PipeBarrier<PIPE_ALL>();
            if(rows)DataCopyPad(v,matrix[start*n],DataCopyExtParams{(uint16_t)rows,n*4,0,(256-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            PipeBarrier<PIPE_ALL>();Muls(v,v,inputScale,L);PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<8;++r)if(start+r>=n)v.SetValue(r*256+start+r,1.0f);
            PipeBarrier<PIPE_ALL>();Abs(ab,v,L);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<8;++r) {
                WholeReduceMax(mx,ab[r*256],64,4,1,1,8);SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
                uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);int32_t exponent=(int32_t)((mb>>23)&255)-132;
                if((mb&0x7fffffff)==0){exponent=0;Duplicate(v[r*256],0.0f,256);}
                else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Muls(v[r*256],v[r*256],scale.value,256);}
                PipeBarrier<PIPE_ALL>();e.SetValue(r,(float)exponent);
            }
            PipeBarrier<PIPE_ALL>();DataCopy(exponents[start],e,8);PipeBarrier<PIPE_ALL>();
            for(uint32_t d=0;d<6;++d) {
                Adds(q,v,12582912.0f,L);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,L);PipeBarrier<PIPE_V>();
                Cast(hd,q,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();Cast(bd,hd,RoundMode::CAST_TRUNC,L);PipeBarrier<PIPE_V>();
                Sub(v,v,q,L);PipeBarrier<PIPE_V>();Muls(v,v,128.0f,L);PipeBarrier<PIPE_ALL>();
                DataCopyPad(digits[start*1536+d*256],bd,DataCopyExtParams{8,256,0,1280,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ __attribute__((noinline)) void SplitXPacked4(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,GlobalTensor<float> solution,uint32_t k) {
        auto input=w[10000],e=w[8000];auto d=w[1536].ReinterpretCast<int8_t>();auto ix=w[19000].ReinterpretCast<int32_t>();auto packed=w[20000].ReinterpretCast<int8_t>();
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)64,256);PipeBarrier<PIPE_ALL>();
        if(lane<k){DataCopy(input,solution,4096);PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<4;c+=teams) {
            if(c<k)Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else Duplicate(w,0.0f,256);PipeBarrier<PIPE_ALL>();
            float exponent=0;if(c<k)exponent=Split6Batched(w,d);else Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,384);PipeBarrier<PIPE_ALL>();
            Duplicate(packed.ReinterpretCast<int32_t>(),(int32_t)0,3072);PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight)DataCopy(packed[weight*1536],d[(5-weight)*256],(weight+1)*256);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(digits[c*1536],packed,DataCopyExtParams{8,1536,0,4608,0});PipeBarrier<PIPE_ALL>();
            e.SetValue(0,exponent);e.SetValue(8,0.0f);PipeBarrier<PIPE_ALL>();
            DataCopyPad(exponents[c],e,DataCopyExtParams{1,4,0,0,0});DataCopyPad(exponents[c+4],e[8],DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SplitX(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,GlobalTensor<float> solution,uint32_t k) {
        auto input=w[10000],e=w[8000];auto d=w[1536].ReinterpretCast<int8_t>();auto ix=w[19000].ReinterpretCast<int32_t>();
        CreateVecIndex(ix,(int32_t)0,256);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)128,256);PipeBarrier<PIPE_ALL>();
        if(lane<k){DataCopy(input,solution,8192);PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<16;c+=teams) {
            if(c<k)Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else Duplicate(w,0.0f,256);PipeBarrier<PIPE_ALL>();
            float exponent=0;
            if(c<k)exponent=Split6<true>(w,d);else Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,384);PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight)DataCopy(digits[(weight*(weight+1)/2*16+c*(weight+1))*256],d[(5-weight)*256],(weight+1)*256);
            e.SetValue(0,exponent);PipeBarrier<PIPE_ALL>();DataCopyPad(exponents[c],e,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6Packed4(GlobalTensor<float> dst,GlobalTensor<float> initial,GlobalTensor<int32_t> products,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> rhs) {
        constexpr uint32_t L=128;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];auto packed=w[7000],allProducts=w[9000],rhsTile=w[10400];auto ix=w[4608].ReinterpretCast<int32_t>();
        DataCopy(ae,expA,256);DataCopy(xe,expX,16);PipeBarrier<PIPE_ALL>();
        CreateVecIndex(ix,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_ALL>();
        auto index=w[11000].ReinterpretCast<int32_t>();auto columns=w[11504].ReinterpretCast<int32_t>();
        CreateVecIndex(index,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(columns,index,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(columns,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Sub(columns,index,columns,L);PipeBarrier<PIPE_V>();
        ShiftLeft(index,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Gather(ex,xe,index.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
        auto outputIndex=w[16000].ReinterpretCast<int32_t>(),outputRows=w[16512].ReinterpretCast<int32_t>(),rhsIndex=w[17024].ReinterpretCast<int32_t>();
        CreateVecIndex(rhsIndex,(int32_t)0,L);PipeBarrier<PIPE_V>();Add(rhsIndex,rhsIndex,ix,L);PipeBarrier<PIPE_V>();ShiftLeft(rhsIndex,rhsIndex,(int32_t)2,L);PipeBarrier<PIPE_V>();
        CreateVecIndex(outputIndex,(int32_t)0,512);PipeBarrier<PIPE_V>();ShiftRight(outputRows,outputIndex,(int32_t)4,512);PipeBarrier<PIPE_V>();
        Muls(outputRows,outputRows,(int32_t)12,512);PipeBarrier<PIPE_V>();Sub(outputIndex,outputIndex,outputRows,512);PipeBarrier<PIPE_V>();ShiftLeft(outputIndex,outputIndex,(int32_t)2,512);PipeBarrier<PIPE_ALL>();
        uint64_t outputMask[2]={0x000f000f000f000fULL,0};
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            uint32_t row=tile*32;Gather(ea,ae[row],ix.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Duplicate(low,0.0f,L);DataCopyPad(rhsTile,rhs[row*16],DataCopyExtParams{32,32,32,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Gather(high,rhsTile,rhsIndex.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            DataCopy(allProducts.ReinterpretCast<int32_t>(),products[row*32],1024);PipeBarrier<PIPE_ALL>();
            CreateVecIndex(columns,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(index,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();ShiftLeft(gs.ReinterpretCast<int32_t>(),index,(int32_t)2,L);PipeBarrier<PIPE_V>();Sub(columns,columns,gs.ReinterpretCast<int32_t>(),L);PipeBarrier<PIPE_V>();ShiftLeft(index,index,(int32_t)7,L);PipeBarrier<PIPE_V>();ShiftLeft(columns,columns,(int32_t)2,L);PipeBarrier<PIPE_V>();Add(index,index,columns,L);PipeBarrier<PIPE_ALL>();
            auto q0=w[14000].ReinterpretCast<int32_t>(),q1=w[14256].ReinterpretCast<int32_t>(),q2=w[14512].ReinterpretCast<int32_t>(),q3=w[14768].ReinterpretCast<int32_t>(),q4=w[15024].ReinterpretCast<int32_t>(),q5=w[15280].ReinterpretCast<int32_t>();
            Adds(columns,index,(int32_t)(0*16),L);PipeBarrier<PIPE_V>();Gather(q0,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(1*16),L);PipeBarrier<PIPE_V>();Gather(q1,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(2*16),L);PipeBarrier<PIPE_V>();Gather(q2,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(3*16),L);PipeBarrier<PIPE_V>();Gather(q3,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(4*16),L);PipeBarrier<PIPE_V>();Gather(q4,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Adds(columns,index,(int32_t)(5*16),L);PipeBarrier<PIPE_V>();Gather(q5,allProducts.ReinterpretCast<int32_t>(),columns.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            ShiftLeft(q0,q0,(int32_t)7,L);PipeBarrier<PIPE_V>();Add(q0,q0,q1,L);PipeBarrier<PIPE_V>();
            ShiftLeft(q2,q2,(int32_t)7,L);PipeBarrier<PIPE_V>();Add(q2,q2,q3,L);PipeBarrier<PIPE_V>();
            ShiftLeft(q4,q4,(int32_t)7,L);PipeBarrier<PIPE_V>();Add(q4,q4,q5,L);PipeBarrier<PIPE_V>();
            // Canonical base-16384 digits. All intermediate magnitudes are below 2^30.
            ShiftRight(q5,q4,(int32_t)14,L);PipeBarrier<PIPE_V>();Add(q2,q2,q5,L);PipeBarrier<PIPE_V>();ShiftLeft(q5,q5,(int32_t)14,L);PipeBarrier<PIPE_V>();Sub(q4,q4,q5,L);PipeBarrier<PIPE_V>();
            ShiftRight(q3,q2,(int32_t)14,L);PipeBarrier<PIPE_V>();Add(q0,q0,q3,L);PipeBarrier<PIPE_V>();ShiftLeft(q3,q3,(int32_t)14,L);PipeBarrier<PIPE_V>();Sub(q2,q2,q3,L);PipeBarrier<PIPE_V>();
            // Split the leading int32 at its FP32 rounding boundary; pack its exact remainder with the middle digit.
            Cast(gs,q0,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();Cast(q1,gs,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();Sub(q1,q0,q1,L);PipeBarrier<PIPE_V>();ShiftLeft(q1,q1,(int32_t)14,L);PipeBarrier<PIPE_V>();Add(q1,q1,q2,L);PipeBarrier<PIPE_V>();
            Cast(q2.ReinterpretCast<float>(),q1,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();Cast(q4.ReinterpretCast<float>(),q4,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();
            Add(factor,ea,ex,L);PipeBarrier<PIPE_V>();Adds(factor,factor,120.0f,L);PipeBarrier<PIPE_V>();Cast(factor.ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();ShiftLeft(factor.ReinterpretCast<int32_t>(),factor.ReinterpretCast<int32_t>(),(int32_t)23,L);PipeBarrier<PIPE_V>();
            Mul(term,gs,factor,L);PipeBarrier<PIPE_V>();
            Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            Muls(factor,factor,0.00006103515625f,L);PipeBarrier<PIPE_V>();
            Mul(term,q2.ReinterpretCast<float>(),factor,L);PipeBarrier<PIPE_V>();
            Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            Muls(factor,factor,0.00006103515625f,L);PipeBarrier<PIPE_V>();
            Mul(term,q4.ReinterpretCast<float>(),factor,L);PipeBarrier<PIPE_V>();
            Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            Add(high,high,low,L);PipeBarrier<PIPE_V>();Duplicate(packed,0.0f,512);PipeBarrier<PIPE_V>();Gather(packed,high,outputIndex.ReinterpretCast<uint32_t>(),0,outputMask,8,8);PipeBarrier<PIPE_ALL>();DataCopy(dst[row*16],packed,512);DataCopy(initial[row*16],packed,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6(GlobalTensor<float> dst,GlobalTensor<int32_t> products,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> rhs) {
        constexpr uint32_t L=256;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];auto packed=w[7000];auto ix=w[4608].ReinterpretCast<int32_t>();
        DataCopy(ae,expA,256);DataCopy(xe,expX,16);PipeBarrier<PIPE_ALL>();
        CreateVecIndex(ix,(int32_t)0,L);PipeBarrier<PIPE_V>();ShiftRight(ix,ix,(int32_t)4,L);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,L);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<16;++r)DataCopy(ex[r*16],xe,16);PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            uint32_t row=tile*16;Gather(ea,ae[row],ix.ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();
            Duplicate(low,0.0f,L);DataCopyPad(high,rhs[row*32],DataCopyExtParams{16,64,64,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<6;++weight) {
                DataCopy(gs.ReinterpretCast<int32_t>(),products[weight*4096+row*16],L);PipeBarrier<PIPE_ALL>();
                Cast(gs,gs.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();
                Add(factor,ea,ex,L);PipeBarrier<PIPE_V>();Adds(factor,factor,127.0f-7.0f*weight,L);PipeBarrier<PIPE_V>();
                Cast(factor.ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();ShiftLeft(factor.ReinterpretCast<int32_t>(),factor.ReinterpretCast<int32_t>(),(int32_t)23,L);PipeBarrier<PIPE_V>();
                Mul(term,gs,factor,L);PipeBarrier<PIPE_V>();
                Add(tmp,high,term,L);PipeBarrier<PIPE_V>();Sub(u,tmp,high,L);PipeBarrier<PIPE_V>();Sub(v,tmp,u,L);PipeBarrier<PIPE_V>();Sub(v,high,v,L);PipeBarrier<PIPE_V>();Sub(u,term,u,L);PipeBarrier<PIPE_V>();Add(u,u,v,L);PipeBarrier<PIPE_V>();Add(low,low,u,L);PipeBarrier<PIPE_V>();Muls(high,tmp,1.0f,L);PipeBarrier<PIPE_V>();
            }
            Add(high,high,low,L);PipeBarrier<PIPE_ALL>();Duplicate(packed,0.0f,512);PipeBarrier<PIPE_V>();Muls(packed,high,1.0f,(uint64_t)16,(uint8_t)16,UnaryRepeatParams{1,1,4,2});PipeBarrier<PIPE_ALL>();DataCopy(dst[row*32],packed,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SaveForwardIdentity(GlobalTensor<float> dst) {
        // Init prepared these six disjoint 16x96 identity row tiles.
        if(lane<6){DataCopy(dst[lane*1536],w[36864],1536);PipeBarrier<PIPE_ALL>();}
    }
    __aicore__ inline void ZeroUnusedFactorRows(GlobalTensor<float> dst) {
        if(lane<4) {Duplicate(w,0.0f,4096);PipeBarrier<PIPE_ALL>();DataCopy(dst[(12+lane)*4096],w,4096);PipeBarrier<PIPE_ALL>();}
    }
    __aicore__ inline void Defect(GlobalTensor<float> dst,GlobalTensor<float> src) {
        if(lane)return;
        DataCopy(w,src,1024);PipeBarrier<PIPE_ALL>();Muls(w,w,-1.0f,1024);PipeBarrier<PIPE_ALL>();
        for(uint32_t i=0;i<32;++i)w.SetValue(i*33,1.0f+w.GetValue(i*33));
        PipeBarrier<PIPE_ALL>();DataCopy(dst,w,1024);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ReprojectCoefficients(GlobalTensor<float> h,GlobalTensor<float> coefficient,uint32_t cut) {
        for(uint32_t row=lane*16;row<cut;row+=teams*16) {
            DataCopy(w,coefficient[row*32],512);
            DataCopyPad(w[4096],h[row*256+cut],DataCopyExtParams{16,128,896,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            Add(w[4096],w[4096],w,512);PipeBarrier<PIPE_ALL>();
            DataCopy(coefficient[row*32],w,512);DataCopyPad(h[row*256+cut],w[4096],DataCopyExtParams{16,128,0,896,0});PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void TailQR(GlobalTensor<float> dst,GlobalTensor<float> src,uint32_t real) {
        if(lane)return;
        auto panel=w,triangular=w[8192],product=w[9216],partial=w[13312],sum=w[13568],coeff=w[13632],weights=w[13760];
        auto ix=w[14080].ReinterpretCast<int32_t>();
        DataCopy(panel,src,real*256);Duplicate(triangular,0.0f,1024);Duplicate(partial,0.0f,256);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<real;++j) {
            Mul(product,panel[j*256],panel[j*256],256);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,product,64,4,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(sum,sum,4,1,1,1,8);PipeBarrier<PIPE_V>();Sqrt(sum,sum,1);PipeBarrier<PIPE_ALL>();
            float norm=sum.GetValue(0);triangular.SetValue(j*32+31,norm);PipeBarrier<PIPE_ALL>();
            Muls(panel[j*256],panel[j*256],CheckedReciprocal(norm),256);PipeBarrier<PIPE_V>();
            uint32_t rows=real-j-1;
            if(rows)for(uint32_t pass=0;pass<2;++pass) {
                for(uint32_t q=0;q<4;++q)Mul(product[q*64],panel[(j+1)*256+q*64],panel[j*256+q*64],(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,1,32,32,0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(partial,product,64,rows*4,2,1,8);PipeBarrier<PIPE_V>();
                WholeReduceSum(coeff,partial,8,rows,1,1,1);PipeBarrier<PIPE_V>();
                Add(triangular[j*32],triangular[j*32],coeff,rows);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,rows);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((rows+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[(j+1)*256+q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)rows,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        for(int32_t j=(int32_t)real-1;j>=0;--j) {
            float norm=triangular.GetValue(j*32+31);
            Muls(panel[j*256],panel[j*256],CheckedReciprocal(norm),256);PipeBarrier<PIPE_V>();
            if(j) {
                for(int32_t r=0;r<j;++r)ix.SetValue(r,(r*32+j-r-1)*4);PipeBarrier<PIPE_ALL>();
                Gather(coeff,triangular,ix.ReinterpretCast<uint32_t>(),0,(uint32_t)j);PipeBarrier<PIPE_V>();
                Muls(coeff,coeff,-1.0f,(uint32_t)j);PipeBarrier<PIPE_V>();Brcb(weights,coeff,(uint8_t)((j+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)MulAddDst(panel[q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)j,BinaryRepeatParams{1,1,0,32,0,1});
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_V>();Muls(panel,panel,-1.0f,real*256);PipeBarrier<PIPE_ALL>();DataCopy(dst,panel,real*256);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void Invert(GlobalTensor<float> dst,GlobalTensor<float> src) {
        if(lane)return;
        constexpr uint32_t B=32,R=72;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,src,DataCopyExtParams{B,128,0,5,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,-1.0f);ix.SetValue(r,r*R*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            float d=panel.GetValue(j*R+j);
            Muls(pivot,panel[j*R],-CheckedReciprocal(d),(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});
            Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)32,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(panel,pivot,weights,(uint64_t)64,(uint8_t)32,BinaryRepeatParams{1,1,0,9,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[j*R],pivot,-1.0f,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(dst,panel[B],DataCopyExtParams{B,128,5,0,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void Invert32(GlobalTensor<float> dst,GlobalTensor<float> src,bool negative=false) {
        if(lane)return;
        constexpr uint32_t B=32,R=72;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,src,DataCopyExtParams{B,128,384,5,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,1.0f);ix.SetValue(r,r*R*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            float d=panel.GetValue(j*R+j);
            Muls(pivot,panel[j*R],-CheckedReciprocal(d),(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});
            Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)32,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            MulAddDst(panel,pivot,weights,(uint64_t)64,(uint8_t)32,BinaryRepeatParams{1,1,0,9,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[j*R],pivot,-1.0f,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_ALL>();
        }
        if(negative){Muls(panel,panel,-1.0f,B*R);PipeBarrier<PIPE_ALL>();}
        DataCopyPad(dst,panel[B],DataCopyExtParams{B,128,5,384,0});PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t N> __aicore__ inline void NegateBlock(GlobalTensor<float> src) {
        if(lane!=1)return;
        DataCopyPad(w,src,DataCopyExtParams{N,N*4,(128-N)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        Muls(w,w,-1.0f,N*N);PipeBarrier<PIPE_ALL>();DataCopyPad(src,w,DataCopyExtParams{N,N*4,0,(128-N)*4,0});PipeBarrier<PIPE_ALL>();
    }


    __aicore__ inline void PadGram96(GlobalTensor<float> gram) {
        uint32_t row=lane*16;
        if(row<96) {
            Duplicate(w,0.0f,512);PipeBarrier<PIPE_ALL>();
            DataCopyPad(gram[row*128+96],w,DataCopyExtParams{16,128,0,384,0});
        } else if(row<128) {
            Duplicate(w,0.0f,2048);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)w.SetValue(j*128+row+j,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(gram[row*128],w,2048);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void NewtonPrepare96(GlobalTensor<half> packed,GlobalTensor<float> gram) {
        if(lane>=6)return;
        auto value=w,ab=w[16384],sum=w[32768];DataCopy(value,gram,8192);PipeBarrier<PIPE_ALL>();Abs(ab,value,8192);Duplicate(sum,0.0f,128);PipeBarrier<PIPE_V>();
        Add(ab,ab,ab[64],(uint64_t)64,(uint8_t)64,BinaryRepeatParams{1,1,1,16,16,16});PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,64,64,1,1,16);PipeBarrier<PIPE_V>();WholeReduceMax(sum[256],sum,64,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceMax(sum[512],sum[256],2,1,1,1,8);PipeBarrier<PIPE_ALL>();
        float alpha=1.8f/sum.GetValue(512);auto temp=w[4096],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        uint32_t row=lane*16;
        DataCopyPad(value,gram[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Cast(h,value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Cast(temp,h,RoundMode::CAST_NONE,1536);PipeBarrier<PIPE_V>();Sub(temp,value,temp,1536);PipeBarrier<PIPE_V>();Cast(h[6144],temp,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(eye,w[36864],1.0f,1536);PipeBarrier<PIPE_V>();Cast(h[3072],eye,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(temp,value,-alpha,1536);PipeBarrier<PIPE_V>();Add(temp,temp,eye,1536);PipeBarrier<PIPE_V>();Cast(h[7680],temp,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(value,eye,-alpha,1536);PipeBarrier<PIPE_V>();Cast(h[1536],value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(value,eye,2.0f,1536);PipeBarrier<PIPE_V>();Cast(h[4608],value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_V>();
        Muls(value,eye,-1.0f,1536);PipeBarrier<PIPE_V>();Cast(h[9216],value,RoundMode::CAST_RINT,1536);PipeBarrier<PIPE_ALL>();
        DataCopy(packed[row*96],h,DataCopyParams{7,96,0,480});PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SaveFallbackGram96(GlobalTensor<float> dst,GlobalTensor<float> src) {
        if(lane>=6)return;
        const uint32_t row=lane*16;
        DataCopyPad(w,src[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[row*128],w,DataCopyExtParams{16,384,0,128,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void FallbackDefect96(GlobalTensor<float> dst) {
        if(lane>=6)return;
        const uint32_t row=lane*16;
        DataCopyPad(w,dst[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        Add(w,w,w[36864],1536);PipeBarrier<PIPE_ALL>();
        DataCopyPad(dst[row*128],w,DataCopyExtParams{16,384,0,128,0});PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ZeroDecision(GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuard96(GlobalTensor<float> dst,GlobalTensor<int32_t> decision) {
        if(lane>=6)return;auto ab=w[4096],sum=w[8192];uint32_t row=lane*16;
        DataCopyPad(w,dst[row*128],DataCopyExtParams{16,384,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();Abs(ab,w,1536);PipeBarrier<PIPE_V>();
        Add(ab,ab,ab[32],(uint64_t)32,(uint8_t)16,BinaryRepeatParams{1,1,1,12,12,12});PipeBarrier<PIPE_V>();
        Add(ab,ab,ab[64],(uint64_t)32,(uint8_t)16,BinaryRepeatParams{1,1,1,12,12,12});PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,32,16,1,1,12);PipeBarrier<PIPE_V>();WholeReduceMax(sum[128],sum,16,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(sum[192],sum,16,1,1,1,8);PipeBarrier<PIPE_ALL>();
        float bound=sum.GetValue(128),total=sum.GetValue(192);
        if(!(bound<=0.0625f && total>=0.0f && total<=2.0f)) {
            auto flag=sum[128].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
        }
    }
    __aicore__ inline void NewtonDefect(GlobalTensor<float> dst) {
        NewtonDefect64(w,dst,lane,teams);
    }

};
}

} // namespace SolveDetail
} // namespace AscendC
