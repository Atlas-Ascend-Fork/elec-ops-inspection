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

#include "lstsq_narrow_cube.h"

namespace NarrowOrthogonal {
template<uint32_t Panel> class Vector {
    AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    AscendC::LocalTensor<float> w;
    uint32_t lane,teams;float inputScale=1.0f;
public:
    uint32_t batch;
    __aicore__ inline void Init(uint32_t batches) {
        if (batches == 0) return;
        uint32_t id=AscendC::GetBlockIdx();batch=id%batches;lane=id/batches;teams=(2*AscendC::GetBlockNum()+batches-1-batch)/batches;
        pipe.InitBuffer(buf,160*1024);w=buf.Get<float>();
        if(lane<Panel/16) {
            auto identity=w[36864];AscendC::Duplicate(identity,0.0f,16*Panel);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)identity.SetValue(j*Panel+lane*16+j,1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackCachedB(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> initial,uint32_t k) {
        // NormalizeSeed retained unscaled RHS, zero-padded to 256 rows, at w[8448].
        // Offset r*16+c maps to r*k+c; only c<k is gathered.
        auto ix=w[16384].ReinterpretCast<int32_t>(),rows=w[16896].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(ix,(int32_t)0,256);AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(rows,ix,(int32_t)4,256);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(rows,rows,(int32_t)(16-k),256);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(ix,ix,rows,256);AscendC::PipeBarrier<PIPE_V>();AscendC::ShiftLeft(ix,ix,(int32_t)2,256);AscendC::PipeBarrier<PIPE_V>();
        uint64_t columns=(1ULL<<k)-1,mask[2]={columns|(columns<<16)|(columns<<32)|(columns<<48),0};
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            AscendC::Duplicate(w,0.0f,256);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(w,w[8448],ix.ReinterpretCast<uint32_t>(),row*k*4,mask,4,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(w,w,-inputScale,256);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(dst[row*16],w,256);AscendC::DataCopy(initial[row*16],w,256);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Normalize(AscendC::GlobalTensor<float> normalized,AscendC::GlobalTensor<float> current,AscendC::GlobalTensor<float> src,AscendC::GlobalTensor<float> rhs,uint32_t k,uint32_t n) {
        NormalizeOrthogonalInput(w,lane,teams,inputScale,normalized,current,src,rhs,k,n);
    }
    __aicore__ inline void Finish(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> x,AscendC::GlobalTensor<float> dx,uint32_t k,uint32_t n,GM_ADDR rr,bool hasResidual) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*16+c)*4);
        AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<n;row+=teams*16) {
            AscendC::DataCopy(w,x[row*16],256);AscendC::DataCopy(w[4096],dx[row*16],256);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(w,w,w[4096],256);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Gather(w[8192],w,ix,0,16*k);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(dst[row*k],w[8192],AscendC::DataCopyExtParams{1,(n-row<16?n-row:16)*k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
        }
        if(lane==0 && hasResidual) {
            AscendC::GlobalTensor<float> gr;gr.SetGlobalBuffer((__gm__ float*)rr);
            AscendC::Duplicate(w,0.0f,8);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gr[(uint64_t)batch*k],w,AscendC::DataCopyExtParams{1,k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void SplitAChunk(AscendC::GlobalTensor<int8_t> digits,AscendC::GlobalTensor<float> exponents,AscendC::GlobalTensor<float> matrix,uint32_t panel,uint32_t n) {
        SplitOrthogonalInput<kOperandDigits,kResidualWidth>(w,lane,teams,inputScale,digits,exponents,matrix,panel,n);
    }
    __aicore__ inline void ZeroUnusedFactorRows(AscendC::GlobalTensor<float> dst) {
        if(lane<(256-2*Panel)/16) {AscendC::Duplicate(w,0.0f,4096);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(dst[(2*Panel/16+lane)*4096],w,4096);AscendC::PipeBarrier<PIPE_ALL>();}
    }
    template<uint32_t W> __aicore__ inline void ReprojectCoefficients(AscendC::GlobalTensor<float> h,AscendC::GlobalTensor<float> coefficient,uint32_t cut) {
        for(uint32_t row=lane*16;row<cut;row+=teams*16) {
            AscendC::DataCopy(w,coefficient[row*W],16*W);
            AscendC::DataCopyPad(w[4096],h[row*256+cut],AscendC::DataCopyExtParams{16,W*4,(256-W)*4,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(w[4096],w[4096],w,16*W);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(coefficient[row*W],w,512);AscendC::DataCopyPad(h[row*256+cut],w[4096],AscendC::DataCopyExtParams{16,W*4,0,(256-W)*4,0});AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ __attribute__((noinline)) void TailQRDirect(AscendC::LocalTensor<float> panel,uint32_t real) {
        OrthogonalTailQR(w,panel,real,real<=4?1u:2u);
    }
    __aicore__ __attribute__((noinline)) bool TailFour(AscendC::LocalTensor<float> panel,uint32_t real) {
        // These temporaries avoid the persistent outer 16/24-row tail factors
        // and the direct QR scratch used by at most four remaining rows.
        auto gram=w[24576],aug=w[24640],pivot=w[24704],col=w[24720],weights=w[24736];
        auto ix=w[24800].ReinterpretCast<uint32_t>();auto factor=w[24832],first=w[25088];
        auto product=w[28672],partial=w[36864],coeff=w[37440];
        // Each 8-row slab holds one independent prefix-row product group.
        // Pad unused rows before the combined reductions for shorter tails.
        if(real<8)AscendC::Duplicate(product,0.0f,8192);
        AscendC::Duplicate(partial,0.0f,512);AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t j=0;j<4;++j)
            for(uint32_t q=0;q<4;++q)AscendC::Mul(product[j*2048+q*64],panel[q*64],panel[j*256+q*64],(uint64_t)64,(uint8_t)real,AscendC::BinaryRepeatParams{1,1,1,32,32,0});
        AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(partial,product,64,128,2,1,8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(gram,partial,8,32,1,1,1);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(aug,0.0f,64);AscendC::Duplicate(factor,0.0f,32);AscendC::Duplicate(first,0.0f,1024);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(aug,gram,1.0f,(uint64_t)4,4,AscendC::UnaryRepeatParams{1,1,2,1});AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<4;++r){aug.SetValue(r*16+8+r,1.0f);ix.SetValue(r,r*16*4);}AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<4;++j) {
            const float inverse=1.0f/aug.GetValue(j*16+j);
            AscendC::Muls(pivot,aug[j*16],-inverse,16);AscendC::Gather(col,aug,ix,j*4,4);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(weights,col,1,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(aug,pivot,weights,(uint64_t)16,4,AscendC::BinaryRepeatParams{1,1,0,2,0,1});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(aug[j*16],pivot,-1.0f,16);AscendC::PipeBarrier<PIPE_ALL>();
        }
        for(uint32_t j=0;j<4;++j) {
            AscendC::Gather(col,aug,ix,(8+j)*4,4);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(col,col,-1.0f,4);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(weights,col,1,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(factor,gram[j*8],weights,(uint64_t)8,4,AscendC::BinaryRepeatParams{1,1,0,1,0,1});AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopy(product,factor,32);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<4;++r)product.SetValue(r*8+r,1.0f+product.GetValue(r*8+r));AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Abs(product,product,32);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(partial,product,4,4,1,1,1);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(coeff,partial,4,1,1,1,8);AscendC::WholeReduceSum(coeff[8],partial,4,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
        const float defect=coeff.GetValue(0),total=coeff.GetValue(8);
        if(!(defect<=0.015625f && total>=0.0f && total<=0.25f))return false;
        const uint32_t remaining=real-4;
        for(uint32_t j=0;j<4;++j) {
            AscendC::Brcb(weights,factor[j*8],1,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(panel[1024+q*64],panel[j*256+q*64],weights[32],(uint64_t)64,(uint8_t)remaining,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
            AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(col,aug,ix,(8+j)*4,4);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(col,col,-1.0f,4);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(weights,col,1,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(first[q*64],panel[j*256+q*64],weights,(uint64_t)64,4,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();TailQRDirect(panel[1024],remaining);
        for(uint32_t r=0;r<4;++r)ix.SetValue(r,r*8*4);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<remaining;++j) {
            AscendC::Gather(col,factor,ix,(4+j)*4,4);AscendC::PipeBarrier<PIPE_V>();AscendC::Brcb(weights,col,1,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(first[q*64],panel[(4+j)*256+q*64],weights,(uint64_t)64,4,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(panel,first,1024);AscendC::PipeBarrier<PIPE_ALL>();return true;
    }
    __aicore__ inline void TailQRResident(AscendC::LocalTensor<float> panel,uint32_t real) {
        if(real>4 && real<=8 && TailFour(panel,real))return;
        TailQRDirect(panel,real);
    }
    __aicore__ inline void TailQR(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> src,uint32_t real) {
        if(lane)return;AscendC::DataCopy(w,src,real*256);AscendC::PipeBarrier<PIPE_ALL>();
        TailQRResident(w,real);AscendC::DataCopy(dst,w,real*256);AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void PublishTailReady(AscendC::GlobalTensor<int32_t> dst,int32_t value) {
        auto flag=w[9216].ReinterpretCast<int32_t>();AscendC::Duplicate(flag,value,8);
        AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(dst,flag,8);AscendC::PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Prefix> struct TailViews {
        AscendC::LocalTensor<float> panel,first,gram,aug,factor,weights,pivot,col,product,partial,coeff;
        AscendC::LocalTensor<uint32_t> index,factorIndex;
    };
    template<uint32_t Prefix,bool CubeApply>
    __aicore__ inline void BindTailViews(TailViews<Prefix> &v) {
        constexpr uint32_t B=Prefix,S=2*B;
        v.panel=w;v.first=w[8192];v.gram=w[B==16?12288:20480];v.aug=w[B==16?12800:22528];v.factor=w[B==16?13312:14336];
        v.weights=w[B==16?13824:15104];v.pivot=w[B==16?13952:15360];v.col=w[B==16?14016:15424];
        v.index=w[B==16?14080:15456].ReinterpretCast<uint32_t>();
        v.factorIndex=w[B==16?14080:15520].ReinterpretCast<uint32_t>();
        v.product=w[20480];v.partial=w[36864];v.coeff=w[37440];
    }
    template<uint32_t Prefix,bool CubeApply>
    __aicore__ inline void FactorTail(TailViews<Prefix> &v,AscendC::GlobalTensor<float> src,AscendC::GlobalTensor<float> gramSource,uint32_t real) {
        constexpr uint32_t B=Prefix,S=2*B;
        AscendC::Duplicate(v.panel,0.0f,8192);AscendC::Duplicate(v.partial,0.0f,512);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(v.panel,src,real*256);AscendC::DataCopy(v.gram,gramSource,B*32);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Duplicate(v.aug,0.0f,B*S);AscendC::Duplicate(v.factor,0.0f,B*32);AscendC::Duplicate(v.first,0.0f,CubeApply?1024:B*256);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(v.aug,v.gram,AscendC::DataCopyParams{B,B/8,(32-B)/8,B/8});AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<B;++r) {
            v.aug.SetValue(r*S+B+r,1.0f);v.index.SetValue(r,r*S*4);
            if constexpr(B!=16 && !CubeApply)v.factorIndex.SetValue(r,r*32*4);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            const float inverse=1.0f/v.aug.GetValue(j*S+j);
            AscendC::Muls(v.pivot,v.aug[j*S],-inverse,S);AscendC::Gather(v.col,v.aug,v.index,j*4,B);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(v.weights,v.col,B/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(v.aug,v.pivot,v.weights,(uint64_t)S,(uint8_t)B,AscendC::BinaryRepeatParams{1,1,0,S/8,0,1});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(v.aug[j*S],v.pivot,-1.0f,S);AscendC::PipeBarrier<PIPE_ALL>();
        }
        for(uint32_t j=0;j<B;++j) {
            AscendC::Gather(v.col,v.aug,v.index,(B+j)*4,B);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(v.col,v.col,-1.0f,B);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(v.weights,v.col,B/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(v.factor,v.gram[j*32],v.weights,(uint64_t)32,(uint8_t)B,AscendC::BinaryRepeatParams{1,1,0,4,0,1});AscendC::PipeBarrier<PIPE_V>();
        }
    }
    template<uint32_t Prefix,bool CubeApply>
    __aicore__ inline bool CheckTailFactor(TailViews<Prefix> &v) {
        constexpr uint32_t B=Prefix,S=2*B;
        // Gram is dead here; the 24-row variant reuses that space for the defect.
        AscendC::DataCopy(v.product,v.factor,AscendC::DataCopyParams{B,B/8,(32-B)/8,0});AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<B;++r)v.product.SetValue(r*B+r,1.0f+v.product.GetValue(r*B+r));AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Abs(v.product,v.product,B*B);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(v.partial,v.product,B,B,1,1,B/8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(v.coeff,v.partial,B,1,1,1,8);AscendC::WholeReduceSum(v.coeff[8],v.partial,B,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
        const float defect=v.coeff.GetValue(0),total=v.coeff.GetValue(8);
        return defect<=0.015625f && total>=0.0f && total<=0.25f;
    }
    template<uint32_t Prefix,bool CubeApply>
    __aicore__ inline void PrepareTailTransform(TailViews<Prefix> &v,uint32_t real) {
        constexpr uint32_t B=Prefix,S=2*B;
        if constexpr(CubeApply) {
            // T = [-Ginv, H; 0, I]. Its bottom rows preserve the completed QR.
            AscendC::Muls(v.aug,v.aug,-1.0f,B*S);AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(v.first,v.aug[B],AscendC::DataCopyParams{B,B/8,B/8,(32-B)/8});
            AscendC::DataCopy(v.first[B],v.factor[B],AscendC::DataCopyParams{B,(32-B)/8,B/8,B/8});AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t r=B;r<real;++r)v.first.SetValue(r*32+r,1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t Prefix,bool CubeApply>
    __aicore__ inline void ProjectTail(TailViews<Prefix> &v,uint32_t real) {
        constexpr uint32_t B=Prefix,S=2*B;
        const uint32_t remaining=real-B;
        for(uint32_t j=0;j<B;++j) {
            AscendC::Brcb(v.weights,v.factor[j*32+B],(remaining+7)/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(v.panel[B*256+q*64],v.panel[j*256+q*64],v.weights,(uint64_t)64,(uint8_t)remaining,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr(!CubeApply) {
                AscendC::Gather(v.col,v.aug,v.index,(B+j)*4,B);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(v.col,v.col,-1.0f,B);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Brcb(v.weights,v.col,B/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(v.first[q*64],v.panel[j*256+q*64],v.weights,(uint64_t)64,(uint8_t)B,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }
    template<uint32_t Prefix,bool CubeApply>
    __aicore__ inline void StoreTail(TailViews<Prefix> &v,AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> src,AscendC::GlobalTensor<float> transform,AscendC::GlobalTensor<int32_t> ready,uint32_t remaining) {
        constexpr uint32_t B=Prefix,S=2*B;
        if constexpr(CubeApply) {
            AscendC::DataCopy(transform,v.first,1024);AscendC::DataCopy(src[B*256],v.panel[B*256],remaining*256);
            PublishTailReady(ready,1);
        } else {
            for(uint32_t j=0;j<remaining;++j) {
                AscendC::Gather(v.col,v.factor,v.factorIndex,(B+j)*4,B);AscendC::PipeBarrier<PIPE_V>();AscendC::Brcb(v.weights,v.col,B/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(v.first[q*64],v.panel[(B+j)*256+q*64],v.weights,(uint64_t)64,(uint8_t)B,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(dst,v.first,B*256);AscendC::DataCopy(dst[B*256],v.panel[B*256],remaining*256);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t Prefix=16,bool CubeApply=false>
    __aicore__ __attribute__((noinline)) void LocalTail32(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> src,AscendC::GlobalTensor<float> gramSource,uint32_t real,GM_ADDR scratch=nullptr) {
        static_assert(Prefix==16 || Prefix==24);
        static_assert(!CubeApply || (Panel==112 && Prefix==24));
        if(lane)return;
        AscendC::GlobalTensor<float> transform;AscendC::GlobalTensor<int32_t> ready;
        if constexpr(CubeApply) {
            transform.SetGlobalBuffer((__gm__ float*)scratch+GI);
            ready.SetGlobalBuffer((__gm__ int32_t*)scratch+REDUCE+96);
        }
        TailViews<Prefix> v;BindTailViews<Prefix,CubeApply>(v);
        FactorTail<Prefix,CubeApply>(v,src,gramSource,real);
        if(!CheckTailFactor<Prefix,CubeApply>(v)) {
            TailQRResident(v.panel,real);AscendC::DataCopy(dst,v.panel,real*256);
            if constexpr(CubeApply)PublishTailReady(ready,0);else AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
        PrepareTailTransform<Prefix,CubeApply>(v,real);
        ProjectTail<Prefix,CubeApply>(v,real);
        const uint32_t remaining=real-Prefix;
        AscendC::PipeBarrier<PIPE_ALL>();TailQRResident(v.panel[Prefix*256],remaining);
        StoreTail<Prefix,CubeApply>(v,dst,src,transform,ready,remaining);
    }
    __aicore__ __attribute__((noinline)) void Invert32(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> src,bool negative=false) {
        if(lane)return;
        constexpr uint32_t B=32,R=72;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        AscendC::Duplicate(panel,0.0f,B*R);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(panel,src,AscendC::DataCopyExtParams{B,128,384,5,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
        AscendC::PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,1.0f);ix.SetValue(r,r*R*4);}AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            float d=panel.GetValue(j*R+j);
            AscendC::Muls(pivot,panel[j*R],-CheckedReciprocal(d),(uint64_t)64,1,AscendC::UnaryRepeatParams{1,1,8,8});
            AscendC::Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)32,1,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(weights,col,4,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(panel,pivot,weights,(uint64_t)64,(uint8_t)32,AscendC::BinaryRepeatParams{1,1,0,9,0,1});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(panel[j*R],pivot,-1.0f,(uint64_t)64,1,AscendC::UnaryRepeatParams{1,1,8,8});AscendC::PipeBarrier<PIPE_ALL>();
        }
        if(negative){AscendC::Muls(panel,panel,-1.0f,B*R);AscendC::PipeBarrier<PIPE_ALL>();}
        AscendC::DataCopyPad(dst,panel[B],AscendC::DataCopyExtParams{B,128,5,384,0});AscendC::PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t N> __aicore__ inline void NegateBlock(AscendC::GlobalTensor<float> src) {
        if(lane!=1)return;
        AscendC::DataCopyPad(w,src,AscendC::DataCopyExtParams{N,N*4,(128-N)*4,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(w,w,-1.0f,N*N);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(src,w,AscendC::DataCopyExtParams{N,N*4,0,(128-N)*4,0});AscendC::PipeBarrier<PIPE_ALL>();
    }


    __aicore__ inline void PadGramPanel(AscendC::GlobalTensor<float> gram) {
        uint32_t row=lane*16;
        if(row<Panel) {
            AscendC::Duplicate(w,0.0f,16*(128-Panel));AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gram[row*128+Panel],w,AscendC::DataCopyExtParams{16,(128-Panel)*4,0,Panel*4,0});
        } else if(row<128) {
            AscendC::Duplicate(w,0.0f,2048);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<16;++j)w.SetValue(j*128+row+j,1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(gram[row*128],w,2048);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void NewtonPreparePanel(AscendC::GlobalTensor<half> packed,AscendC::GlobalTensor<float> gram) {
        if(lane>=Panel/16)return;
        auto value=w,ab=w[16384],sum=w[32768];AscendC::DataCopy(value,gram,8192);AscendC::PipeBarrier<PIPE_ALL>();AscendC::Abs(ab,value,8192);AscendC::Duplicate(sum,0.0f,128);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(ab,ab,ab[64],(uint64_t)64,(uint8_t)64,AscendC::BinaryRepeatParams{1,1,1,16,16,16});AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(sum,ab,64,64,1,1,16);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(sum[256],sum,64,1,1,1,8);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(sum[512],sum[256],2,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
        float alpha=1.8f/sum.GetValue(512);auto temp=w[4096],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        uint32_t row=lane*16;
        AscendC::DataCopyPad(value,gram[row*128],AscendC::DataCopyExtParams{16,Panel*4,(128-Panel)*4,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();AscendC::Cast(h,value,AscendC::RoundMode::CAST_RINT,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(temp,h,AscendC::RoundMode::CAST_NONE,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(temp,value,temp,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[(4*16*Panel)],temp,AscendC::RoundMode::CAST_RINT,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(eye,w[36864],1.0f,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[(2*16*Panel)],eye,AscendC::RoundMode::CAST_RINT,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(temp,value,-alpha,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Add(temp,temp,eye,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[(5*16*Panel)],temp,AscendC::RoundMode::CAST_RINT,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(value,eye,-alpha,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[(1*16*Panel)],value,AscendC::RoundMode::CAST_RINT,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(value,eye,2.0f,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[(3*16*Panel)],value,AscendC::RoundMode::CAST_RINT,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(value,eye,-1.0f,(1*16*Panel));AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(h[(6*16*Panel)],value,AscendC::RoundMode::CAST_RINT,(1*16*Panel));AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(packed[row*Panel],h,AscendC::DataCopyParams{7,Panel,0,Panel*Panel/16-Panel});AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ZeroDecision(AscendC::GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();AscendC::Duplicate(flag,(int32_t)0,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(decision,flag,8);AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuardPanel(AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<int32_t> decision) {
        if constexpr(Panel==96) {

            if(lane>=6)return;auto ab=w[4096],sum=w[8192];uint32_t row=lane*16;
            AscendC::DataCopyPad(w,dst[row*128],AscendC::DataCopyExtParams{16,384,128,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();AscendC::Abs(ab,w,1536);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ab,ab,ab[32],(uint64_t)32,(uint8_t)16,AscendC::BinaryRepeatParams{1,1,1,12,12,12});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ab,ab,ab[64],(uint64_t)32,(uint8_t)16,AscendC::BinaryRepeatParams{1,1,1,12,12,12});AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(sum,ab,32,16,1,1,12);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(sum[128],sum,16,1,1,1,8);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceSum(sum[192],sum,16,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
            float bound=sum.GetValue(128),total=sum.GetValue(192);
            if(!(bound<=0.0625f && total>=0.0f && total<=2.0f)) {
                auto flag=sum[128].ReinterpretCast<int32_t>();AscendC::Duplicate(flag,(int32_t)0,8);AscendC::PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetAtomicAdd<int32_t>();AscendC::DataCopy(decision,flag,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetAtomicNone();
            }

        } else {

            if(lane>=Panel/16)return;auto ab=w[4096],sum=w[8192];uint32_t row=lane*16;
            AscendC::Duplicate(w,0.0f,2048);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(w,dst[row*128],AscendC::DataCopyExtParams{16,Panel*4,(128-Panel)*4,(128-Panel)/8,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();AscendC::Abs(ab,w,2048);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ab,ab,ab[64],(uint64_t)64,(uint8_t)16,AscendC::BinaryRepeatParams{1,1,1,16,16,16});AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(sum,ab,64,16,1,1,16);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(sum[128],sum,16,1,1,1,8);AscendC::WholeReduceSum(sum[192],sum,16,1,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
            float bound=sum.GetValue(128),total=sum.GetValue(192);
            if(!(bound<=0.0625f && total>=0.0f && total<=2.0f)) {
                auto flag=sum[128].ReinterpretCast<int32_t>();AscendC::Duplicate(flag,(int32_t)0,8);AscendC::PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetAtomicAdd<int32_t>();AscendC::DataCopy(decision,flag,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetAtomicNone();
            }
        }
    }
    template<bool Rhs> __aicore__ inline float SplitOperandDigits(AscendC::LocalTensor<float> v,AscendC::LocalTensor<int8_t> digits) {
        auto q=w[256],ab=w[512],mx=w[768];auto hd=w[1024].ReinterpretCast<half>();
        AscendC::Abs(ab,v,256);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(mx,ab,64,4,1,1,8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);uint32_t mb,mi;AscendC::GetReduceRepeatMaxMinSpr<float>(mb,mi);
        int32_t exponent=(int32_t)((mb>>23)&255)-127-5;
        if((mb&0x7fffffff)==0){exponent=0;AscendC::Duplicate(v,0.0f,256);AscendC::PipeBarrier<PIPE_V>();}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;AscendC::Muls(v,v,scale.value,256);AscendC::PipeBarrier<PIPE_V>();}
        for(uint32_t d=0;d<kOperandDigits;++d) {
            AscendC::Adds(q,v,12582912.0f,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Adds(q,q,-12582912.0f,256);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(hd,q,AscendC::RoundMode::CAST_NONE,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(digits[(Rhs?kOperandDigits-1-d:d)*256],hd,AscendC::RoundMode::CAST_TRUNC,256);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(v,v,q,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(v,v,128.0f,256);AscendC::PipeBarrier<PIPE_V>();
        }
        return (float)exponent;
    }
    __aicore__ __attribute__((noinline)) void SplitXPacked4(AscendC::GlobalTensor<int8_t> digits,AscendC::GlobalTensor<float> exponents,AscendC::GlobalTensor<float> solution,uint32_t k) {
        auto input=w[10000],e=w[8000];auto d=w[1536].ReinterpretCast<int8_t>();auto ix=w[19000].ReinterpretCast<int32_t>();auto packed=w[20000].ReinterpretCast<int8_t>();
        AscendC::CreateVecIndex(ix,(int32_t)0,256);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(ix,ix,(int32_t)64,256);AscendC::PipeBarrier<PIPE_ALL>();
        if(lane<k){AscendC::DataCopy(input,solution,4096);AscendC::PipeBarrier<PIPE_ALL>();}
        for(uint32_t c=lane;c<8;c+=teams) {
            if(c<k)AscendC::Gather(w,input[c],ix.ReinterpretCast<uint32_t>(),0,256);else AscendC::Duplicate(w,0.0f,256);AscendC::PipeBarrier<PIPE_ALL>();
            float exponent=0;if(c<k)exponent=SplitOperandDigits<true>(w,d);else AscendC::Duplicate(d.ReinterpretCast<int32_t>(),(int32_t)0,kResidualWidth/4);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Duplicate(packed.ReinterpretCast<int32_t>(),(int32_t)0,6*kResidualWidth/4);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t weight=0;weight<kOperandDigits;++weight)AscendC::DataCopy(packed[weight*kResidualWidth],d[(kOperandDigits-1-weight)*256],(weight+1)*256);
            // Degree five retains A1*X4 + ... + A4*X1. A0*X5 and A5*X0 are absent.
            AscendC::DataCopy(packed[5*kResidualWidth+256],d,(kOperandDigits-1)*256);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(digits[c*kResidualWidth],packed,AscendC::DataCopyExtParams{6,kResidualWidth,0,7*kResidualWidth,0});AscendC::PipeBarrier<PIPE_ALL>();
            if(c<4) {
                e.SetValue(0,exponent);e.SetValue(8,0.0f);AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyPad(exponents[c],e,AscendC::DataCopyExtParams{1,4,0,0,0});AscendC::DataCopyPad(exponents[c+4],e[8],AscendC::DataCopyExtParams{1,4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ __attribute__((noinline)) void Assemble6Packed4(AscendC::GlobalTensor<float> initial,AscendC::GlobalTensor<int32_t> products,AscendC::GlobalTensor<float> expA,AscendC::GlobalTensor<float> expX,AscendC::GlobalTensor<float> rhs) {
        constexpr uint32_t L=256;
        auto ae=w,xe=w[256],gs=w[512],high=w[1024],low=w[1536],term=w[2048],tmp=w[2560],u=w[3072],v=w[3584],factor=w[4096],ea=w[5120],ex=w[5632];
        auto packed=w[7000];auto allProducts=w[9000].ReinterpretCast<int32_t>();
        AscendC::DataCopy(ae,expA,256);AscendC::DataCopy(xe,expX,16);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            const uint32_t row=tile*32;
            AscendC::Brcb(ea,ae[row],4,AscendC::BrcbRepeatParams{1,8});
            AscendC::Muls(ex,xe,1.0f,(uint64_t)8,32,AscendC::UnaryRepeatParams{1,1,1,0});
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(high,rhs[row*16],AscendC::DataCopyParams{32,1,1,0});
            AscendC::DataCopy(allProducts,products[row*48],1536);AscendC::PipeBarrier<PIPE_ALL>();
            auto q0=w[14000].ReinterpretCast<int32_t>(),q1=w[14256].ReinterpretCast<int32_t>(),q2=w[14512].ReinterpretCast<int32_t>(),q3=w[14768].ReinterpretCast<int32_t>(),q4=w[15024].ReinterpretCast<int32_t>(),q5=w[15280].ReinterpretCast<int32_t>();
            // Each coefficient is one aligned eight-lane block per product row.
            AscendC::DataCopy(q0,allProducts,AscendC::DataCopyParams{32,1,5,0});
            AscendC::DataCopy(q1,allProducts[8],AscendC::DataCopyParams{32,1,5,0});
            AscendC::DataCopy(q2,allProducts[16],AscendC::DataCopyParams{32,1,5,0});
            AscendC::DataCopy(q3,allProducts[24],AscendC::DataCopyParams{32,1,5,0});
            AscendC::DataCopy(q4,allProducts[32],AscendC::DataCopyParams{32,1,5,0});
            AscendC::DataCopy(q5,allProducts[40],AscendC::DataCopyParams{32,1,5,0});AscendC::PipeBarrier<PIPE_V>();
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
            // The canonical middle limb is smaller than a nonzero leading
            // limb; a zero leading limb also makes this FastTwoSum exact.
            // Form AX as two floats before its potentially cancelling sum with B.
            AscendC::Mul(term,gs,factor,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(factor,factor,0.00006103515625f,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(u,q2.ReinterpretCast<float>(),factor,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(tmp,term,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,tmp,term,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(low,u,v,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(factor,factor,0.00006103515625f,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(v,q4.ReinterpretCast<float>(),factor,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(low,low,v,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(term,high,tmp,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,term,high,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,term,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(v,high,v,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Sub(u,tmp,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(u,u,v,L);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(low,low,u,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(high,term,low,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Duplicate(packed,0.0f,512);AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(packed,high,AscendC::DataCopyParams{32,1,0,1});AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(initial[row*16],packed,512);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

};
}
