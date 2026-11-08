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
#include "linalg_solve_cube_panel.h"

namespace AscendC {
namespace SolveDetail {
// Fixed-block normal-equation inverse: 256 narrow RHS and 512-capacity
// single-batch wide RHS. Padding and workspace follow the shared capacity.
// Normalize A by a power of two, form C^T C, and eliminate fixed SPD blocks
// against C^T. Apply the resulting C inverse to scaled RHS and residuals.
// The first correction uses FP32 residuals; the last uses compensated products.
template<uint32_t N,uint32_t B=32,uint32_t Refine=2,bool Grouped=(N>256)> class NormalInverse : public CubePanel<N> {
    static_assert((N==256 || N==384 || N==512) && B==32 && Refine==2);
    using Base=CubePanel<N>;
    using Base::SplitRow;
    using Base::pipe;using Base::buf;using Base::CAP;using Base::P;using Base::T;using Base::SZ;using Base::CP;using Base::IT;using Base::ARRAYS;
    using Base::a;using Base::x;using Base::h;using Base::l;using Base::ah;using Base::al;using Base::xh;using Base::xl;using Base::s;
    using Base::raw;using Base::ahi;using Base::alo;using Base::ixBase;using Base::rhsIx;
    using Base::A;using Base::Scale;using Base::D;using Base::LoadInput;using Base::LoadColumn;using Base::AccurateDot;
    __aicore__ inline void Invert(GlobalTensor<float> gram,GlobalTensor<float> inv,uint32_t batch,uint32_t base) {
        constexpr uint32_t R=2*B+8;
        auto w=buf.template Get<float>(),panel=w,pivot=w[B*R],col=pivot[2*B],weights=col[B];
        auto ix=weights[8*B].template ReinterpretCast<int32_t>();
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,gram[((uint64_t)batch*N+base)*N+base],DataCopyExtParams{B,B*4,(N-B)*4,(R-B)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r)panel.SetValue(r*R+B+r,1.0f);
        for(uint32_t r=0;r<B;++r)ix.SetValue(r,r*R*4);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            const float d=panel.GetValue(j*R+j);
            Muls(pivot,panel[j*R],-CheckedReciprocal(d),(uint64_t)(2*B),1,UnaryRepeatParams{1,1,8,8});Gather(col,panel[j],ix.template ReinterpretCast<uint32_t>(),0,(uint64_t)B,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t c=0;c<2*B;c+=64)MulAddDst(panel[c],pivot[c],weights,(uint64_t)(2*B-c<64?2*B-c:64),(uint8_t)B,BinaryRepeatParams{1,1,0,R/8,0,1});
            PipeBarrier<PIPE_V>();Muls(panel[j*R],pivot,-1.0f,(uint64_t)(2*B),1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_ALL>();
        }
        Muls(panel,panel,-1.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(inv[((uint64_t)batch*N+base)*N+base],panel[B],DataCopyExtParams{B,B*4,(R-B)/8,(N-B)*4,0});PipeBarrier<PIPE_ALL>();
        // Subtract I from the current Gram block. Its product with -Gjj^-1
        // then gives Gjj^-1-I on the pivot rows, so the following accumulated
        // updates need no separate row-zeroing or coefficient override.
        DataCopyPad(panel,gram[((uint64_t)batch*N+base)*N+base],DataCopyExtParams{B,B*4,(N-B)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<B;++r)panel.SetValue(r*B+r,panel.GetValue(r*B+r)-1.0f);PipeBarrier<PIPE_ALL>();
        DataCopyPad(gram[((uint64_t)batch*N+base)*N+base],panel,DataCopyExtParams{B,B*4,0,(N-B)*4,0});PipeBarrier<PIPE_ALL>();
    }
    // Four signed base-128 digits. A stores ascending digits per row;
    // each X prefix is reversed so one int8 GEMM sums a common weight.
    template<bool Rhs,bool Chunk=false> __aicore__ inline void SplitRowInt8(GlobalTensor<int8_t> digits,GlobalTensor<float> exponents,uint32_t row,uint32_t rowCount,uint32_t chunkRow=0) {
        static_assert(!Chunk || (!Rhs && 40*N+16<=CAP));
        auto w=buf.template Get<float>(),v=Chunk?w[8*N+chunkRow*N]:w,q=w[N],ab=w[2*N],mx=w[3*N];
        auto hd=w[4*N].template ReinterpretCast<half>();auto bd=w[7*N].template ReinterpretCast<int8_t>();
        Abs(ab,v,N);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,N/64,1,1,8);Base::template Sync<HardEvent::V_S>();
        uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);const int32_t e=(int32_t)((mb>>23)&255)-127;int32_t exponent=e-5;
        if((mb&0x7fffffff)==0){exponent=0;Duplicate(v,0.0f,N);}
        else if(e<-100 || e>100){exponent=1000;Duplicate(v,0.0f,N);}
        else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;Scale(v,v,scale.value,N);}
        for(uint32_t d=0;d<4;++d){
            Adds(q,v,12582912.0f,N);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,N);PipeBarrier<PIPE_V>();
            Cast(hd,q,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();Cast(bd[(Rhs?3-d:d)*N],hd,RoundMode::CAST_TRUNC,N);PipeBarrier<PIPE_V>();
            D(v,v,q,N);Scale(v,v,128.0f,N);
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr(Chunk) {
            DataCopy(w[24*N].template ReinterpretCast<int8_t>()[chunkRow*4*N],bd,4*N);
            w[40*N].SetValue(chunkRow,(float)exponent);PipeBarrier<PIPE_ALL>();
        } else {
            if constexpr(Rhs){for(uint32_t d=0;d<4;++d)DataCopy(digits[((uint64_t)d*(d+1)/2*rowCount+(uint64_t)row*(d+1))*N],bd[(3-d)*N],(d+1)*N);}
            else DataCopy(digits[(uint64_t)row*4*N],bd,4*N);
            PipeBarrier<PIPE_ALL>();mx.SetValue(0,(float)exponent);PipeBarrier<PIPE_ALL>();DataCopyPad(exponents[row],mx,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t G> __aicore__ inline bool SplitAssembleGrouped(GlobalTensor<int32_t> pairs,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> gb,GlobalTensor<float> residual,const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams,float inputScale) {
        constexpr uint32_t L=G*N;static_assert(19*L<=CAP);
        auto w=buf.template Get<float>(),ae=w,xe=w[L],high=w[2*L],low=w[3*L],gs=w[4*L],tmp=w[5*L],term=w[6*L],u=w[7*L],v=w[8*L],factor=w[9*L],br=w[10*L];
        auto ix=w[18*L].template ReinterpretCast<int32_t>();
        auto integerSum=w[11*L].template ReinterpretCast<int32_t>(),rounded=w[12*L].template ReinterpretCast<int32_t>(),remainder=w[13*L].template ReinterpretCast<int32_t>();
        auto groupLow=w[14*L];
        DataCopy(ae,expA,N);DataCopy(xe,expX,N);PipeBarrier<PIPE_ALL>();
        const int32_t amin=Base::template ExponentExtreme<true>(ae,high,N),amax=Base::template ExponentExtreme<false>(ae,high,N);
        const int32_t xmin=Base::template ExponentExtreme<true>(xe,high,t.k),xmax=Base::template ExponentExtreme<false>(xe,high,t.k);
        if(amax>=1000 || xmax>=1000 || amin+xmin<-50 || amax+xmax>100)return false;
        for(uint32_t c=1;c<G;++c)DataCopy(ae[c*N],ae,N);PipeBarrier<PIPE_ALL>();
        CreateVecIndex(ix,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)32,N);PipeBarrier<PIPE_V>();
        for(uint32_t c=1;c<G;++c)Adds(ix[c*N],ix,(int32_t)(c*4),N);PipeBarrier<PIPE_V>();
        for(uint32_t task=lane;task<(t.k+G-1)/G;task+=teams) {
            const uint32_t k=task*G,width=t.k-k<G?t.k-k:G;
            Duplicate(br,0.0f,N*8);Duplicate(low,0.0f,L);PipeBarrier<PIPE_ALL>();
            DataCopyPad(br,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+k],DataCopyExtParams{(uint16_t)t.n,width*4,(t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});
            PipeBarrier<PIPE_ALL>();Gather(high,br,ix.template ReinterpretCast<uint32_t>(),0,L);PipeBarrier<PIPE_V>();Muls(high,high,inputScale,L);PipeBarrier<PIPE_V>();
            for(uint32_t d=0;d<4;++d) {
                if(width<G){Duplicate(integerSum,(int32_t)0,L);PipeBarrier<PIPE_ALL>();}
                DataCopy(integerSum,pairs[((uint64_t)d*t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();
                // At most 4*512*64^2 < 2^24: conversion preserves the exact integer sum.
                Cast(gs,integerSum,RoundMode::CAST_NONE,L);PipeBarrier<PIPE_V>();
                for(uint32_t c=0;c<G;++c)Adds(factor[c*N],ae[c*N],xe.GetValue(k+c)-7.0f*d+127.0f,N);PipeBarrier<PIPE_V>();
                Cast(factor.template ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,L);PipeBarrier<PIPE_V>();
                ShiftLeft(factor.template ReinterpretCast<int32_t>(),factor.template ReinterpretCast<int32_t>(),(int32_t)23,L);PipeBarrier<PIPE_V>();
                Mul(term,gs,factor,L);PipeBarrier<PIPE_V>();Scale(term,term,-1.0f,L);
                A(tmp,high,term,L);D(u,tmp,high,L);D(v,tmp,u,L);D(v,high,v,L);D(u,term,u,L);A(u,u,v,L);A(low,low,u,L);Scale(high,tmp,1.0f,L);
            }
            A(high,high,low,L);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],high,width*N);PipeBarrier<PIPE_ALL>();
        }
        return true;
    }
public:
    struct NormalServiceWork {
        GM_ADDR workspace;
        GM_ADDR sourceA;
        __gm__ float *gm;
        uint32_t batches,order,rhsCount,hmM,hmN,hmR,hmC,hmTiles,tm,tn,mr,nr;
        uint64_t splitStart,adWords,xdWords;
        GlobalTensor<float> input,gram,factors,inv,initial,residual,normalRhs,selectedGram,selectedInverse,originalA;
        GlobalTensor<int8_t> ad,xd;
        GlobalTensor<int32_t> products;
    };
    template<class MM> static __aicore__ inline void FormNormalGram(NormalServiceWork &work,MM &mm) {
        SyncAll<false>();
        for(uint32_t task=GetBlockIdx();task<work.batches*work.mr*work.nr;task+=GetBlockNum()) {
            const uint32_t batch=task/(work.mr*work.nr),tile=task%(work.mr*work.nr),row=tile/work.nr*work.tm,col=tile%work.nr*work.tn;
            mm.SetOrgShape(N,N,N);mm.SetSingleShape(work.tm,work.tn,N);mm.SetTensorA(work.input[(uint64_t)batch*N*N+row],true);mm.SetTensorB(work.input[(uint64_t)batch*N*N+col],false);mm.SetTail(N-row<work.tm?N-row:work.tm,N-col<work.tn?N-col:work.tn,N);mm.IterateAll(work.gram[((uint64_t)batch*N+row)*N+col],0);mm.End();
        }
        SyncAll<false>();
    }
    template<class MM> static __aicore__ inline void EliminateNormalBlocks(NormalServiceWork &work,MM &mm) {
        for(uint32_t base=0;base<work.order;base+=B) {
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<work.batches*work.mr;task+=GetBlockNum()) {
                const uint32_t batch=task/work.mr,row=task%work.mr*work.tm;
                mm.SetOrgShape(N,N,N);mm.SetSingleShape(work.tm,work.tn,B);mm.SetTensorA(work.gram[((uint64_t)batch*N+row)*N+base],false);mm.SetTensorB(work.inv[((uint64_t)batch*N+base)*N+base],false);mm.SetTail(N-row<work.tm?N-row:work.tm,B,B);mm.IterateAll(work.factors[((uint64_t)batch*N+row)*N+base],0);mm.End();
            }
            SyncAll<false>();
            const uint32_t first=base+B<N?(base+B)/work.tn*work.tn:N,cc=(N-first+work.tn-1)/work.tn;
            for(uint32_t task=GetBlockIdx();task<work.batches*work.mr*cc;task+=GetBlockNum()) {
                const uint32_t batch=task/(work.mr*cc),tile=task%(work.mr*cc),row=tile/cc*work.tm,col=first+tile%cc*work.tn;
                mm.SetOrgShape(N,N,N);mm.SetSingleShape(work.tm,work.tn,B);mm.SetTensorA(work.factors[((uint64_t)batch*N+row)*N+base],false);mm.SetTensorB(work.selectedGram[(uint64_t)batch*B*N+col],false);mm.SetTail(N-row<work.tm?N-row:work.tm,N-col<work.tn?N-col:work.tn,B);mm.IterateAll(work.gram[((uint64_t)batch*N+row)*N+col],1);mm.End();
            }
            for(uint32_t task=GetBlockIdx();task<work.batches*work.mr*work.nr;task+=GetBlockNum()) {
                const uint32_t batch=task/(work.mr*work.nr),tile=task%(work.mr*work.nr),row=tile/work.nr*work.tm,col=tile%work.nr*work.tn;
                mm.SetOrgShape(N,N,N);mm.SetSingleShape(work.tm,work.tn,B);mm.SetTensorA(work.factors[((uint64_t)batch*N+row)*N+base],false);mm.SetTensorB(work.selectedInverse[(uint64_t)batch*B*N+col],false);mm.SetTail(N-row<work.tm?N-row:work.tm,N-col<work.tn?N-col:work.tn,B);mm.IterateAll(work.input[((uint64_t)batch*N+row)*N+col],1);mm.End();
            }
            SyncAll<false>();
        }
    }
    template<class MM,class HM> static __aicore__ inline void RefineNormalService(NormalServiceWork &work,MM &mm,HM &hm) {
        const uint32_t kr=(work.rhsCount+work.tm-1)/work.tm;
        for(uint32_t pass=0;pass<=Refine;++pass) {
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<work.batches*kr*work.nr;task+=GetBlockNum()) {
                const uint32_t batch=task/(kr*work.nr),tile=task%(kr*work.nr),row=tile/work.nr*work.tm,col=tile%work.nr*work.tn;
                mm.SetOrgShape(work.rhsCount,N,N);mm.SetSingleShape(work.tm,work.tn,N);mm.SetTensorA(work.residual[((uint64_t)batch*work.rhsCount+row)*N],false);mm.SetTensorB(work.input[((uint64_t)batch*N+col)*N],true);mm.SetTail(work.rhsCount-row<work.tm?work.rhsCount-row:work.tm,N-col<work.tn?N-col:work.tn,N);mm.IterateAll(work.normalRhs[((uint64_t)batch*work.rhsCount+row)*N+col],0);mm.End();
            }
            SyncAll<false>();
            if(pass<Refine) {
                SyncAll<false>();
                if(pass==0) {
                    for(uint32_t task=GetBlockIdx();task<work.batches*kr*work.nr;task+=GetBlockNum()) {
                        const uint32_t batch=task/(kr*work.nr),tile=task%(kr*work.nr),row=tile/work.nr*work.tm,col=tile%work.nr*work.tn;
                        mm.SetOrgShape(work.rhsCount,N,N);mm.SetSingleShape(work.tm,work.tn,N);mm.SetTensorA(work.initial[((uint64_t)batch*work.rhsCount+row)*N],false);mm.SetTensorB(work.originalA[((uint64_t)batch*N+col)*N],true);mm.SetTail(work.rhsCount-row<work.tm?work.rhsCount-row:work.tm,N-col<work.tn?N-col:work.tn,N);mm.IterateAll(work.normalRhs[((uint64_t)batch*work.rhsCount+row)*N+col],0);mm.End();
                    }
                } else for(uint32_t task=GetBlockIdx();task<work.batches*4*work.hmTiles;task+=GetBlockNum()) {
                    const uint32_t bt=task/(4*work.hmTiles),tile=task%work.hmTiles,row=tile/work.hmC*work.hmM,column=tile%work.hmC*work.hmN,d=task/work.hmTiles%4,K=(d+1)*N;
                    hm.SetOrgShape(work.rhsCount,N,K,4*N);hm.SetSingleShape(work.hmM,work.hmN,K);
                    hm.SetTensorA(work.xd[((uint64_t)bt*10*work.rhsCount+(uint64_t)d*(d+1)/2*work.rhsCount+(uint64_t)row*(d+1))*N],false);
                    hm.SetTensorB(work.ad[((uint64_t)bt*N+column)*4*N],true);
                    hm.SetTail(work.rhsCount-row<work.hmM?work.rhsCount-row:work.hmM,N-column<work.hmN?N-column:work.hmN,K);hm.IterateAll(work.products[((uint64_t)bt*4*work.rhsCount+(uint64_t)d*work.rhsCount+row)*N+column],0);hm.End();
                }
                SyncAll<false>();
            }
        }
    }
public:
    static __aicore__ inline void Service(GM_ADDR workspace,GM_ADDR system,const tiling::TCubeTiling &tc,uint32_t batches,uint32_t order,uint32_t rhsCount,const tiling::TCubeTiling *halfTiling=nullptr,GM_ADDR sourceA=nullptr) {
        using Fp=MatmulType<TPosition::GM,CubeFormat::ND,float>;
        using Hp=MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
        using Ip=MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
        TPipe cubePipe;Matmul<Fp,Fp,Fp> mm;Matmul<Hp,Hp,Ip> hm;REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc,hm,halfTiling);mm.SetHF32(false);
        NormalServiceWork work;
        work.workspace=workspace;work.sourceA=sourceA;work.batches=batches;work.order=order;work.rhsCount=rhsCount;
        work.gm=(__gm__ float*)work.workspace;

        work.input.SetGlobalBuffer(work.gm);work.gram.SetGlobalBuffer(work.gm+(uint64_t)work.batches*N*N);work.factors.SetGlobalBuffer(work.gm+(uint64_t)work.batches*2*N*N);work.inv.SetGlobalBuffer(work.gm+(uint64_t)work.batches*3*N*N);
        work.initial.SetGlobalBuffer(work.gm+(uint64_t)work.batches*4*N*N);work.residual.SetGlobalBuffer(work.gm+(uint64_t)work.batches*(4*N*N+work.rhsCount*N));work.normalRhs.SetGlobalBuffer(work.gm+(uint64_t)work.batches*(4*N*N+2*work.rhsCount*N));

        work.selectedGram.SetGlobalBuffer(work.gm+(uint64_t)work.batches*(4*N*N+3*work.rhsCount*N));work.selectedInverse.SetGlobalBuffer(work.gm+(uint64_t)work.batches*(4*N*N+3*work.rhsCount*N+B*N));
        work.splitStart=(uint64_t)work.batches*(4*N*N+3*work.rhsCount*N+2*B*N);work.adWords=(uint64_t)work.batches*4*N*N/4;work.xdWords=(uint64_t)work.batches*10*work.rhsCount*N/4;

        work.ad.SetGlobalBuffer((__gm__ int8_t*)(work.gm+work.splitStart));work.xd.SetGlobalBuffer((__gm__ int8_t*)(work.gm+work.splitStart+work.adWords));work.products.SetGlobalBuffer((__gm__ int32_t*)(work.gm+work.splitStart+work.adWords+work.xdWords));
        const auto &ht=*halfTiling;work.hmM=ht.singleCoreM;work.hmN=ht.singleCoreN;work.hmR=(work.rhsCount+work.hmM-1)/work.hmM;work.hmC=(N+work.hmN-1)/work.hmN;work.hmTiles=work.hmR*work.hmC;
        work.originalA.SetGlobalBuffer(work.order<N?work.gm+(uint64_t)work.batches*N*N:(__gm__ float*)work.sourceA);
        work.tm=tc.singleCoreM;work.tn=tc.singleCoreN;work.mr=(N+work.tm-1)/work.tm;work.nr=(N+work.tn-1)/work.tn;
        FormNormalGram(work,mm);
        EliminateNormalBlocks(work,mm);
        RefineNormalService(work,mm,hm);
    }
    static constexpr uint32_t RG=8;
    struct NormalWork {
        LinalgSolveTilingData t;
        uint32_t cores,id,batch,lane,teams,tiles;
        uint64_t splitStart,adWords,xdWords,pairWords;
        __gm__ float *gm;
        float inputScale;
        LocalTensor<float> mat,pivot,col,active,outputTile;
        LocalTensor<int32_t> groupIx,tileIndices;
        GlobalTensor<float> ga,gb,gx,input,gram,factors,inv,initial,residual,normalRhs,selectedGram,selectedInverse,expA,expX;
        GlobalTensor<int8_t> ad,xd;
        GlobalTensor<int32_t> products;
    };
    __aicore__ inline void InitializeNormalWork(NormalWork &state,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace) {
        state.cores=GetBlockNum()*2;state.id=GetBlockIdx();state.batch=state.id%state.t.batches;state.lane=state.id/state.t.batches;
        state.teams=(state.cores+state.t.batches-1-state.batch)/state.t.batches;state.tiles=(N+T-1)/T;
        pipe.InitBuffer(buf,(CAP+128+2*CP+28*IT)*4);
        state.mat=buf.template Get<float>();state.pivot=state.mat[CAP];state.col=state.pivot[128];state.active=state.col[CP];
        ixBase=state.active[CP].template ReinterpretCast<uint32_t>();rhsIx=state.col.template ReinterpretCast<uint32_t>();
        a=state.mat;x=state.mat[SZ];h=state.mat[2*SZ];l=state.mat[3*SZ];ah=state.mat[4*SZ];al=state.mat[5*SZ];xh=state.mat[6*SZ];xl=state.mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=state.mat[(4+i)*SZ];raw=state.mat[8*SZ];ahi=state.mat[9*SZ];alo=state.mat[10*SZ];state.outputTile=state.mat[ARRAYS*SZ];
        state.gm=(__gm__ float*)workspace;
        state.ga,state.gb,state.gx,state.input,state.gram,state.factors,state.inv,state.initial,state.residual,state.normalRhs;
        state.ga.SetGlobalBuffer((__gm__ float*)aa);state.gb.SetGlobalBuffer((__gm__ float*)bb);state.gx.SetGlobalBuffer((__gm__ float*)xx);
        state.input.SetGlobalBuffer(state.gm);state.gram.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*N*N);state.factors.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*2*N*N);state.inv.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*3*N*N);
        state.initial.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*4*N*N);state.residual.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*(4*N*N+state.t.k*N));state.normalRhs.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*(4*N*N+2*state.t.k*N));

        state.selectedGram.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*(4*N*N+3*state.t.k*N));state.selectedInverse.SetGlobalBuffer(state.gm+(uint64_t)state.t.batches*(4*N*N+3*state.t.k*N+B*N));
        state.splitStart=(uint64_t)state.t.batches*(4*N*N+3*state.t.k*N+2*B*N);state.adWords=(uint64_t)state.t.batches*4*N*N/4;state.xdWords=(uint64_t)state.t.batches*10*state.t.k*N/4;state.pairWords=(uint64_t)state.t.batches*4*state.t.k*N;

        state.ad.SetGlobalBuffer((__gm__ int8_t*)(state.gm+state.splitStart)+(uint64_t)state.batch*4*N*N);state.xd.SetGlobalBuffer((__gm__ int8_t*)(state.gm+state.splitStart+state.adWords)+(uint64_t)state.batch*10*state.t.k*N);
        state.products.SetGlobalBuffer((__gm__ int32_t*)(state.gm+state.splitStart+state.adWords+state.xdWords)+(uint64_t)state.batch*4*state.t.k*N);state.expA.SetGlobalBuffer(state.gm+state.splitStart+state.adWords+state.xdWords+state.pairWords+(uint64_t)state.batch*2*N);state.expX.SetGlobalBuffer(state.gm+state.splitStart+state.adWords+state.xdWords+state.pairWords+(uint64_t)state.batch*2*N+N);
        if(state.lane==0){Duplicate(state.mat,0.0f,N);PipeBarrier<PIPE_ALL>();DataCopy(state.expX,state.mat,N);PipeBarrier<PIPE_ALL>();}
    }
    __aicore__ inline void LoadAndScaleNormalMatrix(NormalWork &state) {
        uint32_t laneMax=0;
        for(uint32_t tile=state.lane;tile<state.tiles;tile+=state.teams) {
            const uint32_t row=tile*T;LoadInput(state.ga,(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n,T,row<state.t.n?(state.t.n-row<T?state.t.n-row:T):0,state.t.n,state.mat);
            DataCopy(state.input[((uint64_t)state.batch*N+row)*N],state.mat,T*N);PipeBarrier<PIPE_ALL>();
            Abs(x,state.mat,T*N);PipeBarrier<PIPE_V>();WholeReduceMax(h,x,64,T*N/64,1,1,8);Base::template Sync<HardEvent::V_S>();
            uint32_t bits,index;GetReduceRepeatMaxMinSpr<float>(bits,index);laneMax=bits>laneMax?bits:laneMax;
        }
        Duplicate(h,0.0f,8);PipeBarrier<PIPE_ALL>();h.template ReinterpretCast<uint32_t>().SetValue(0,laneMax);PipeBarrier<PIPE_ALL>();
        DataCopy(state.factors[(uint64_t)state.batch*N*N+state.lane*8],h,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
        DataCopy(a,state.factors[(uint64_t)state.batch*N*N],state.teams*8);PipeBarrier<PIPE_ALL>();
        uint32_t maxBits=0;for(uint32_t r=0;r<state.teams;++r){uint32_t value=a.template ReinterpretCast<uint32_t>().GetValue(r*8);maxBits=value>maxBits?value:maxBits;}
        int32_t exponent=(int32_t)((maxBits>>23)&255)-127;exponent=exponent>126?126:(exponent<-126?-126:exponent);
        union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;state.inputScale=scale.value;
        for(uint32_t tile=state.lane;tile<state.tiles;tile+=state.teams){const uint32_t row=tile*T;DataCopy(a,state.input[((uint64_t)state.batch*N+row)*N],T*N);PipeBarrier<PIPE_ALL>();Muls(a,a,state.inputScale,T*N);PipeBarrier<PIPE_ALL>();if(state.t.n<N){for(uint32_t r=0;r<T;++r)if(row+r>=state.t.n)a.SetValue(r*N+row+r,1.0f);PipeBarrier<PIPE_ALL>();}DataCopy(state.input[((uint64_t)state.batch*N+row)*N],a,T*N);PipeBarrier<PIPE_ALL>();}
    }
    __aicore__ inline void LoadNormalRhs(NormalWork &state) {
        CreateVecIndex(rhsIx.template ReinterpretCast<int32_t>(),(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rhsIx.template ReinterpretCast<int32_t>(),rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(Grouped?32:state.t.k*4),N);PipeBarrier<PIPE_ALL>();
        state.groupIx=state.mat[3*SZ].template ReinterpretCast<int32_t>();
        if constexpr(!Grouped) {
        for(uint32_t k=state.lane;k<state.t.k;k+=state.teams) {
            DataCopy(state.mat,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*N*state.t.k],N*state.t.k);PipeBarrier<PIPE_ALL>();Gather(x,state.mat,rhsIx,k*4,N);PipeBarrier<PIPE_ALL>();DataCopy(state.residual[((uint64_t)state.batch*state.t.k+k)*N],x,N);PipeBarrier<PIPE_ALL>();
        }
        } else {
        for(uint32_t c=0;c<RG;++c)Adds(state.groupIx[c*N],rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(c*4),N);PipeBarrier<PIPE_ALL>();
        for(uint32_t task=state.lane;task<(state.t.k+RG-1)/RG;task+=state.teams) {
            const uint32_t k=task*RG,width=state.t.k-k<RG?state.t.k-k:RG;
            Duplicate(state.mat,0.0f,N*8);PipeBarrier<PIPE_ALL>();
            DataCopyPad(state.mat,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+k],DataCopyExtParams{(uint16_t)state.t.n,width*4,(state.t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});
            PipeBarrier<PIPE_ALL>();Gather(x,state.mat,state.groupIx.template ReinterpretCast<uint32_t>(),0,RG*N);PipeBarrier<PIPE_ALL>();
            DataCopy(state.residual[((uint64_t)state.batch*state.t.k+k)*N],x,width*N);PipeBarrier<PIPE_ALL>();
        }
        }
        SyncAll<false>();SyncAll<false>();
    }
    __aicore__ inline void TransposeNormalMatrix(NormalWork &state) {
        state.tileIndices=state.mat[4*SZ].template ReinterpretCast<int32_t>();
        CreateVecIndex(state.col.template ReinterpretCast<int32_t>(),(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(state.col.template ReinterpretCast<int32_t>(),state.col.template ReinterpretCast<int32_t>(),(int32_t)(T*4),N);PipeBarrier<PIPE_V>();
        for(uint32_t r=0;r<T;++r)Adds(state.tileIndices[r*N],state.col.template ReinterpretCast<int32_t>(),(int32_t)(r*4),N);PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=state.lane;tile<state.tiles;tile+=state.teams) {
            const uint32_t row=tile*T;
            const uint32_t width=row<state.t.n?(state.t.n-row<T?state.t.n-row:T):0,paddedWidth=(width+7)/8*8;
            if(state.t.n<N){Duplicate(a,0.0f,T*N);PipeBarrier<PIPE_ALL>();}
            if(width)DataCopyPad(a,state.ga[(uint64_t)state.batch*state.t.n*state.t.n+row],DataCopyExtParams{(uint16_t)state.t.n,width*4,(state.t.n-width)*4,(T-paddedWidth)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(paddedWidth-width),0});PipeBarrier<PIPE_ALL>();
            Gather(x,a,state.tileIndices.template ReinterpretCast<uint32_t>(),0,T*N);PipeBarrier<PIPE_V>();Muls(x,x,state.inputScale,T*N);PipeBarrier<PIPE_ALL>();
            if(state.t.n<N){for(uint32_t r=0;r<T;++r)if(row+r>=state.t.n)x.SetValue(r*N+row+r,1.0f);PipeBarrier<PIPE_ALL>();}
            DataCopy(state.input[((uint64_t)state.batch*N+row)*N],x,T*N);PipeBarrier<PIPE_ALL>();
        }
        SyncAll<true>();
    }
    __aicore__ inline void FactorNormalPanel(NormalWork &state,uint32_t base) {
            if(state.lane==0)Invert(state.gram,state.inv,state.batch,base);
            else {
                const uint32_t share=(B+state.teams-2)/(state.teams-1),start=(state.lane-1)*share,count=start<B?(B-start<share?B-start:share):0;
                if(count){
                    // A four-state.lane team publishes up to ceil(32/3) rows per state.lane.
                    // Keep the two row groups disjoint at the larger capacities.
                    auto selectedData=state.mat[N>256?2*SZ:SZ];
                    DataCopy(a,state.gram[((uint64_t)state.batch*N+base+start)*N],count*N);DataCopy(selectedData,state.input[((uint64_t)state.batch*N+base+start)*N],count*N);PipeBarrier<PIPE_ALL>();
                    DataCopy(state.selectedGram[((uint64_t)state.batch*B+start)*N],a,count*N);DataCopy(state.selectedInverse[((uint64_t)state.batch*B+start)*N],selectedData,count*N);PipeBarrier<PIPE_ALL>();
                }
            }
            if(state.lane>0) {
                const uint32_t slots=((state.t.n+B-1)/B)*(state.teams-1),group=(N+slots-1)/slots;
                const uint32_t start=((base/B)*(state.teams-1)+state.lane-1)*group,count=start<N?(N-start<group?N-start:group):0;
                if(count) {
                    auto staged=state.mat[8*N],exponents=state.mat[40*N];auto packed=state.mat[24*N].template ReinterpretCast<int8_t>();
                    const uint32_t validRows=start<state.t.n?(state.t.n-start<count?state.t.n-start:count):0;
                    Duplicate(staged,0.0f,count*N);PipeBarrier<PIPE_ALL>();
                    if(validRows)DataCopyPad(staged,state.ga[((uint64_t)state.batch*state.t.n+start)*state.t.n],DataCopyExtParams{(uint16_t)validRows,state.t.n*4,0,(N-(state.t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-state.t.n%8)%8),0});
                    PipeBarrier<PIPE_ALL>();if constexpr(Grouped){Muls(staged,staged,state.inputScale,count*N);PipeBarrier<PIPE_V>();}
                    for(uint32_t row=0;row<count;++row)SplitRowInt8<false,true>(state.ad,state.expA,start+row,N,row);
                    PipeBarrier<PIPE_ALL>();DataCopy(state.ad[(uint64_t)start*4*N],packed,count*4*N);
                    DataCopyPad(state.expA[start],exponents,DataCopyExtParams{1,count*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
            SyncAll<false>();SyncAll<false>();SyncAll<false>();
    }
    __aicore__ inline void PrepareNormalReduction(NormalWork &state) {
        // Gram is no longer needed after factorization; reuse it for padded A
        // in the first FP32 state.residual, avoiding an extra workspace matrix.
        if(state.t.n<N)for(uint32_t tile=state.lane;tile<state.tiles;tile+=state.teams) {
            const uint32_t row=tile*T;LoadInput(state.ga,(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n,T,row<state.t.n?(state.t.n-row<T?state.t.n-row:T):0,state.t.n,state.mat);
            DataCopy(state.gram[((uint64_t)state.batch*N+row)*N],state.mat,T*N);PipeBarrier<PIPE_ALL>();
        }
        for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
            auto ix0=ixBase[4*IT*(4-half)],ix1=ix0[IT*half];for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j){ix0.SetValue(row*half+j,(row*stride+j)*4);ix1.SetValue(row*half+j,(row*stride+half+j)*4);}
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ScaleNormalResidual(NormalWork &state) {
            if constexpr(!Grouped) {
            for(uint32_t k=state.lane;k<state.t.k;k+=state.teams){DataCopy(a,state.residual[((uint64_t)state.batch*state.t.k+k)*N],N);PipeBarrier<PIPE_ALL>();Muls(a,a,state.inputScale,N);PipeBarrier<PIPE_ALL>();DataCopy(state.residual[((uint64_t)state.batch*state.t.k+k)*N],a,N);PipeBarrier<PIPE_ALL>();}
            } else {
            for(uint32_t task=state.lane;task<(state.t.k+RG-1)/RG;task+=state.teams){const uint32_t k=task*RG,width=state.t.k-k<RG?state.t.k-k:RG;DataCopy(a,state.residual[((uint64_t)state.batch*state.t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();Muls(a,a,state.inputScale,width*N);PipeBarrier<PIPE_ALL>();DataCopy(state.residual[((uint64_t)state.batch*state.t.k+k)*N],a,width*N);PipeBarrier<PIPE_ALL>();}
            }
    }
    __aicore__ inline void UpdateNormalSolution(NormalWork &state,uint32_t pass) {
            if constexpr(!Grouped) {
            for(uint32_t k=state.lane;k<state.t.k;k+=state.teams) {
                DataCopy(a,state.normalRhs[((uint64_t)state.batch*state.t.k+k)*N],N);PipeBarrier<PIPE_ALL>();
                if(pass){DataCopy(x,state.initial[((uint64_t)state.batch*state.t.k+k)*N],N);PipeBarrier<PIPE_ALL>();Add(a,a,x,N);PipeBarrier<PIPE_ALL>();}
                DataCopy(state.initial[((uint64_t)state.batch*state.t.k+k)*N],a,N);PipeBarrier<PIPE_ALL>();
            }
            } else {
            for(uint32_t task=state.lane;task<(state.t.k+RG-1)/RG;task+=state.teams) {
                const uint32_t k=task*RG,width=state.t.k-k<RG?state.t.k-k:RG;
                DataCopy(a,state.normalRhs[((uint64_t)state.batch*state.t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();
                if(pass){DataCopy(x,state.initial[((uint64_t)state.batch*state.t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();Add(a,a,x,width*N);PipeBarrier<PIPE_ALL>();}
                DataCopy(state.initial[((uint64_t)state.batch*state.t.k+k)*N],a,width*N);PipeBarrier<PIPE_ALL>();
            }
            }
    }
    __aicore__ inline bool AssembleNormalResidual(NormalWork &state,uint32_t pass) {
            bool assembled=false;
            if(pass==0) {
                CreateVecIndex(rhsIx.template ReinterpretCast<int32_t>(),(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rhsIx.template ReinterpretCast<int32_t>(),rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(Grouped?32:state.t.k*4),N);PipeBarrier<PIPE_ALL>();
                if constexpr(!Grouped) {
                for(uint32_t k=state.lane;k<state.t.k;k+=state.teams){DataCopy(a,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*N*state.t.k],N*state.t.k);DataCopy(h,state.normalRhs[((uint64_t)state.batch*state.t.k+k)*N],N);PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,k*4,N);PipeBarrier<PIPE_V>();Sub(x,x,h,N);PipeBarrier<PIPE_ALL>();DataCopy(state.residual[((uint64_t)state.batch*state.t.k+k)*N],x,N);PipeBarrier<PIPE_ALL>();}
                } else {
                for(uint32_t c=0;c<RG;++c)Adds(state.groupIx[c*N],rhsIx.template ReinterpretCast<int32_t>(),(int32_t)(c*4),N);PipeBarrier<PIPE_ALL>();
                for(uint32_t task=state.lane;task<(state.t.k+RG-1)/RG;task+=state.teams) {
                    const uint32_t k=task*RG,width=state.t.k-k<RG?state.t.k-k:RG;
                    Duplicate(a,0.0f,N*8);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(a,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+k],DataCopyExtParams{(uint16_t)state.t.n,width*4,(state.t.k-width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-width),0});
                    DataCopy(h,state.normalRhs[((uint64_t)state.batch*state.t.k+k)*N],width*N);PipeBarrier<PIPE_ALL>();
                    Gather(x,a,state.groupIx.template ReinterpretCast<uint32_t>(),0,RG*N);PipeBarrier<PIPE_V>();Sub(x,x,h,width*N);PipeBarrier<PIPE_ALL>();
                    DataCopy(state.residual[((uint64_t)state.batch*state.t.k+k)*N],x,width*N);PipeBarrier<PIPE_ALL>();
                }
                }
                assembled=true;
            } else assembled=SplitAssembleGrouped<4>(state.products,state.expA,state.expX,state.gb,state.residual,state.t,state.batch,state.lane,state.teams,Grouped?state.inputScale:1.0f);
        return assembled;
    }
    __aicore__ inline void ComputeNormalResidual(NormalWork &state) {
            for(uint32_t tile=state.lane;tile<state.tiles;tile+=state.teams) {
                const uint32_t row=tile*T,rows=T;
                LoadInput(state.ga,(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n,rows,row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0,state.t.n,raw);Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
                for(uint32_t k=0;k<state.t.k;++k) {
                    LoadColumn(state.initial,((uint64_t)state.batch*state.t.k+k)*P,rows);AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                    const uint32_t rb=k/8*8,kw=state.t.k-rb<8?state.t.k-rb:8,validRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
                    Duplicate(a,0.0f,T*8);PipeBarrier<PIPE_ALL>();if(validRows)DataCopyPad(a,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+row*state.t.k+rb],DataCopyExtParams{(uint16_t)validRows,kw*4,(state.t.k-kw)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-kw),0});PipeBarrier<PIPE_ALL>();
                    for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k-rb)*4);PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_V>();D(h,x,h,rows);D(h,h,l,rows);PipeBarrier<PIPE_ALL>();DataCopyPad(state.residual[((uint64_t)state.batch*state.t.k+k)*P+row],h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
    }
    __aicore__ inline void StoreNormalSolution(NormalWork &state) {
        if constexpr(!Grouped) {
        for(uint32_t tile=state.lane;tile<state.tiles;tile+=state.teams) {
            const uint32_t row=tile*T,rows=T;
            for(uint32_t rb=0;rb<state.t.k;rb+=8) {
                const uint32_t kw=state.t.k-rb<8?state.t.k-rb:8;
                for(uint32_t k=rb;k<rb+kw;++k){DataCopyPad(x,state.initial[((uint64_t)state.batch*state.t.k+k)*P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();for(uint32_t i=0;i<rows;++i)state.outputTile.SetValue(i*8+k-rb,x.GetValue(i));}
                PipeBarrier<PIPE_ALL>();const uint32_t validRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;if(validRows)DataCopyPad(state.gx[(uint64_t)state.batch*state.t.n*state.t.k+row*state.t.k+rb],state.outputTile,DataCopyExtParams{(uint16_t)validRows,kw*4,0,(state.t.k-kw)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }
        } else {
        constexpr uint32_t OW=32;
        auto packed=state.mat[OW*IT];auto transposeIx=state.mat[2*OW*IT].template ReinterpretCast<int32_t>();
        CreateVecIndex(transposeIx,(int32_t)0,OW);PipeBarrier<PIPE_V>();Muls(transposeIx,transposeIx,(int32_t)(IT*4),OW);PipeBarrier<PIPE_V>();
        for(uint32_t r=1;r<T;++r)Adds(transposeIx[r*OW],transposeIx,(int32_t)(r*4),OW);PipeBarrier<PIPE_ALL>();
        for(uint32_t tile=state.lane;tile<state.tiles;tile+=state.teams) {
            const uint32_t row=tile*T,rows=T,validRows=row<state.t.n?(state.t.n-row<T?state.t.n-row:T):0;if(!validRows)continue;
            for(uint32_t rb=0;rb<state.t.k;rb+=OW) {
                const uint32_t width=state.t.k-rb<OW?state.t.k-rb:OW;
                Duplicate(a,0.0f,OW*IT);PipeBarrier<PIPE_ALL>();
                DataCopyPad(a,state.initial[((uint64_t)state.batch*state.t.k+rb)*P+row],DataCopyExtParams{(uint16_t)width,rows*4,(P-rows)*4,(IT-(rows+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-rows%8)%8),0});
                PipeBarrier<PIPE_ALL>();Gather(packed,a,transposeIx.template ReinterpretCast<uint32_t>(),0,rows*OW);PipeBarrier<PIPE_ALL>();
                DataCopyPad(state.gx[(uint64_t)state.batch*state.t.n*state.t.k+row*state.t.k+rb],packed,DataCopyExtParams{(uint16_t)validRows,width*4,(OW-(width+7)/8*8)/8,(state.t.k-width)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }
        }    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        NormalWork state;state.t=t;
        InitializeNormalWork(state,aa,bb,xx,workspace);
        LoadAndScaleNormalMatrix(state);LoadNormalRhs(state);TransposeNormalMatrix(state);
        for(uint32_t base=0;base<t.n;base+=B)FactorNormalPanel(state,base);
        PrepareNormalReduction(state);
        bool scaledResidual=false;
        for(uint32_t pass=0;pass<=Refine;++pass) {
            SyncAll<true>();
            if(!scaledResidual)ScaleNormalResidual(state);
            SyncAll<false>();SyncAll<false>();
            UpdateNormalSolution(state,pass);SyncAll<true>();
            if(pass==Refine)break;
            if(pass>0)for(uint32_t k=state.lane;k<t.k;k+=state.teams) {
                DataCopy(state.mat,state.initial[((uint64_t)state.batch*t.k+k)*N],N);PipeBarrier<PIPE_ALL>();
                SplitRowInt8<true>(state.xd,state.expX,k,t.k);
            }
            SyncAll<false>();SyncAll<false>();
            bool assembled=AssembleNormalResidual(state,pass);
            if(!assembled)ComputeNormalResidual(state);
            scaledResidual=Grouped && pass>0 && assembled;
        }
        StoreNormalSolution(state);
    }

};

} // namespace SolveDetail
} // namespace AscendC
