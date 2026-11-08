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
// Apply the same per-panel elimination maps to the correction RHS.
// All factor data is produced from this invocation's A in workspace.
template<uint32_t N,uint32_t DotN=N,bool SplitResidual=false> class DirectPanel : public CubePanel<DotN> {
    using Base=CubePanel<DotN>;
    static_assert(!SplitResidual || (N<=256 && DotN==256));
    using Base::SplitRow;using Base::SplitAssembleRows;
    using Base::P;using Base::T;using Base::SZ;using Base::CP;using Base::IT;
    static constexpr uint32_t B=Base::B,PL=2*B+8,PP=Base::PP;using Base::ARRAYS;
    static constexpr uint32_t FactorCols=(N+B-1)/B*B;
    static constexpr uint32_t INPUT_INDEX=Base::INPUT_INDEX,PANEL_SCRATCH=Base::PANEL_SCRATCH;using Base::CAP;
    static_assert(N*PL+((B+3)/4)*(2*P)<=CAP-2*P);
    using Base::pipe;using Base::buf;using Base::a;using Base::x;
    using Base::ah;using Base::al;using Base::xh;using Base::xl;using Base::h;using Base::l;using Base::s;
    using Base::raw;using Base::ahi;using Base::alo;using Base::ixBase;using Base::rhsIx;
    using Base::Scale;using Base::D;using Base::LoadInput;using Base::LoadColumn;using Base::AccurateDot;
    template<HardEvent event> __aicore__ inline void Sync() {Base::template Sync<event>();}
    __aicore__ inline void ApplyFactors(GlobalTensor<float> factors,GlobalTensor<float> ids,GlobalTensor<float> initial,GlobalTensor<float> residual,
        const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams,const uint32_t *order) {
        static_assert(2*PP*B+4*N+10*B<=CAP);
        if(lane>=t.k)return;
        auto w=buf.template Get<float>(),input=w,prod=w[PP*B],sol=w[2*PP*B],sums=sol[N],picked=sums[N],coeff=picked[B];
        auto perm=coeff[8*B].template ReinterpretCast<int32_t>();
        for(uint32_t row=0;row<N;++row)perm.SetValue(row,order[row]*4);PipeBarrier<PIPE_ALL>();
        if constexpr(FactorCols>N){Duplicate(perm[N],(int32_t)0,FactorCols-N);PipeBarrier<PIPE_ALL>();}
        for(uint32_t k=lane;k<t.k;k+=teams) {
            Duplicate(prod,0.0f,PP*B);PipeBarrier<PIPE_ALL>();
            DataCopy(sol,residual[((uint64_t)batch*t.k+k)*P],N);PipeBarrier<PIPE_ALL>();
            for(uint32_t base=0;base<t.n;base+=B) {
                const uint32_t pw=t.n-base<B?t.n-base:B;
                DataCopyPad(input,factors[((uint64_t)batch*FactorCols+base)*N],DataCopyExtParams{B,N*4,0,(PP-N)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                Gather(picked,sol,perm[base].template ReinterpretCast<uint32_t>(),0,B);PipeBarrier<PIPE_V>();
                Brcb(coeff,picked,B/8,BrcbRepeatParams{1,8});
                Sync<HardEvent::V_S>();for(uint32_t j=0;j<pw;++j)sol.SetValue(order[base+j],0.0f);Sync<HardEvent::S_V>();
                for(uint32_t off=0;off<N;off+=64)Mul(prod[off],input[off],coeff,(uint64_t)64,(uint8_t)B,BinaryRepeatParams{1,1,0,PP/8,PP/8,1});
                PipeBarrier<PIPE_V>();
                for(uint32_t half=B/2;half;half/=2){Add(prod,prod,prod[half*PP],half*PP);PipeBarrier<PIPE_V>();}
                Add(sol,sol,prod,N);PipeBarrier<PIPE_ALL>();
            }
            Gather(sums,sol,perm.template ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_ALL>();
            DataCopy(sol,initial[((uint64_t)batch*t.k+k)*P],N);PipeBarrier<PIPE_ALL>();
            Add(sol,sol,sums,N);PipeBarrier<PIPE_ALL>();DataCopy(initial[((uint64_t)batch*t.k+k)*P],sol,N);PipeBarrier<PIPE_ALL>();
        }
    }

public:
    static __aicore__ inline void Service(GM_ADDR workspace,GM_ADDR system,const tiling::TCubeTiling &tc,uint32_t batches,uint32_t order,uint32_t rhsCount,const tiling::TCubeTiling *halfTiling=nullptr) {
        using Fp=MatmulType<TPosition::GM,CubeFormat::ND,float>;
        using Hp=MatmulType<TPosition::GM,CubeFormat::ND,half>;
        TPipe cubePipe;Matmul<Fp,Fp,Fp> mm;Matmul<Hp,Hp,Fp> hm;
        if constexpr(SplitResidual){REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc,hm,halfTiling);}else{REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc);}mm.SetHF32(false);
        const uint32_t L=2*P,usedColumns=P+(rhsCount+7)/8*8;

        GlobalTensor<float> matrix,factors,selected;
        matrix.SetGlobalBuffer((__gm__ float*)workspace);
        factors.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batches*N*L);
        selected.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)batches*(N*L+FactorCols*N));
        const uint32_t tm=tc.singleCoreM,tn=tc.singleCoreN,mr=(N+tm-1)/tm;
        for(uint32_t base=0;base<order;base+=B) {

            SyncAll<false>();
            const uint32_t first=N>=512?(base+B)/tn*tn:base+B,nr=(usedColumns-first+tn-1)/tn,per=mr*nr;
            for(uint32_t task=GetBlockIdx();task<batches*per;task+=GetBlockNum()) {
                const uint32_t batch=task/per,tile=task%per,row=tile/nr*tm,column=first+tile%nr*tn;
                mm.SetOrgShape(N,L,B);mm.SetSingleShape(tm,tn,B);
                mm.SetTensorA(factors[((uint64_t)batch*FactorCols+base)*N+row],true);
                mm.SetTensorB(selected[(uint64_t)batch*B*L+column],false);
                mm.SetTail(N-row<tm?N-row:tm,usedColumns-column<tn?usedColumns-column:tn,B);
                mm.IterateAll(matrix[(uint64_t)batch*N*L+row*L+column],1);mm.End();
            }
            SyncAll<false>();
        }
        if constexpr(SplitResidual) {
            const uint64_t splitStart=(uint64_t)batches*(N*L+FactorCols*N+B*L+FactorCols+2*rhsCount*P);
            const uint64_t adWords=(uint64_t)batches*2*P*P,xdWords=(uint64_t)batches*2*rhsCount*P;
            GlobalTensor<half> ad,xd;GlobalTensor<float> products;
            ad.SetGlobalBuffer((__gm__ half*)((__gm__ float*)workspace+splitStart));
            xd.SetGlobalBuffer((__gm__ half*)((__gm__ float*)workspace+splitStart+adWords));
            products.SetGlobalBuffer((__gm__ float*)workspace+splitStart+adWords+xdWords);
            const auto &ht=*halfTiling;const uint32_t hmM=ht.singleCoreM,hmN=ht.singleCoreN,hmR=(rhsCount+hmM-1)/hmM,hmC=(P+hmN-1)/hmN,hmTiles=hmR*hmC;
            constexpr uint32_t pairs=N<P?13:10;
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<batches*pairs*hmTiles;task+=GetBlockNum()) {
                const uint32_t bt=task/(pairs*hmTiles),tile=task%hmTiles,row=tile/hmC*hmM,column=tile%hmC*hmN,pair=task/hmTiles%pairs;
                uint32_t i=0,j=pair;while(j>=(N<P?(i<2?4:5-i):4-i)){j-=(N<P?(i<2?4:5-i):4-i);++i;}
                hm.SetOrgShape(rhsCount,P,P);hm.SetSingleShape(hmM,hmN,P);
                hm.SetTensorA(xd[((uint64_t)bt*4+j)*rhsCount*P+row*P],false);
                hm.SetTensorB(ad[((uint64_t)bt*4+i)*P*P+column*P],true);
                hm.SetTail(rhsCount-row<hmM?rhsCount-row:hmM,P-column<hmN?P-column:hmN,P);
                hm.IterateAll(products[(((uint64_t)bt*16+i*4+j)*rhsCount+row)*P+column],0);hm.End();
            }
            SyncAll<false>();
        }
    }
    struct DirectWork {
        LinalgSolveTilingData t;
        uint32_t cores,id,batch,lane,teams,chunk,begin,end,L,cw,firstColumn,width;
        uint64_t factorStart,selectedStart,idsStart,initialStart,residualStart;
        __gm__ float *gm;
        uint32_t labels[N],order[N];
        LocalTensor<float> mat,pivot,col,active,outputTile,matrixZeros,panel,selected,transform,panelScratch;
        LocalTensor<int32_t> inputIndex;
        GlobalTensor<float> ga,gb,gx,matrix,factors,selectedGM,ids,initial,residual,products,expA,expX;
        GlobalTensor<half> ad,xd;
    };
    __aicore__ inline void InitializeDirectWork(DirectWork &state,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace) {
        state.cores=GetBlockNum()*2;state.id=GetBlockIdx();state.batch=state.id%state.t.batches;state.lane=state.id/state.t.batches;
        state.teams=(state.cores+state.t.batches-1-state.batch)/state.t.batches;state.chunk=(N+state.teams-1)/state.teams;state.begin=state.lane*state.chunk;state.end=state.begin+state.chunk<N?state.begin+state.chunk:N;
        state.L=2*P;state.cw=(state.L+state.teams*8-1)/(state.teams*8)*8;state.firstColumn=state.lane*state.cw;state.width=state.firstColumn<state.L?(state.L-state.firstColumn<state.cw?state.L-state.firstColumn:state.cw):0;
        pipe.InitBuffer(buf,(CAP+128+2*CP+28*IT)*4);
        state.mat=buf.template Get<float>();state.pivot=state.mat[CAP];state.col=state.pivot[128];state.active=state.col[CP];
        ixBase=state.active[CP].template ReinterpretCast<uint32_t>();rhsIx=state.col.template ReinterpretCast<uint32_t>();
        a=state.mat;x=state.mat[SZ];h=state.mat[2*SZ];l=state.mat[3*SZ];ah=state.mat[4*SZ];al=state.mat[5*SZ];xh=state.mat[6*SZ];xl=state.mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=state.mat[(4+i)*SZ];raw=state.mat[8*SZ];ahi=state.mat[9*SZ];alo=state.mat[10*SZ];state.outputTile=state.mat[ARRAYS*SZ];

        state.ga.SetGlobalBuffer((__gm__ float*)aa);state.gb.SetGlobalBuffer((__gm__ float*)bb);state.gx.SetGlobalBuffer((__gm__ float*)xx);
        state.factorStart=(uint64_t)state.t.batches*N*state.L;state.selectedStart=state.factorStart+(uint64_t)state.t.batches*FactorCols*N;
        state.idsStart=state.selectedStart+(uint64_t)state.t.batches*B*state.L;state.initialStart=state.idsStart+state.t.batches*FactorCols;state.residualStart=state.initialStart+(uint64_t)state.t.batches*state.t.k*P;
        state.gm=(__gm__ float*)workspace;state.matrix.SetGlobalBuffer(state.gm);state.factors.SetGlobalBuffer(state.gm+state.factorStart);state.selectedGM.SetGlobalBuffer(state.gm+state.selectedStart);state.ids.SetGlobalBuffer(state.gm+state.idsStart);state.initial.SetGlobalBuffer(state.gm+state.initialStart);state.residual.SetGlobalBuffer(state.gm+state.residualStart);

        if constexpr(SplitResidual) {
            const uint64_t splitStart=state.residualStart+(uint64_t)state.t.batches*state.t.k*P;
            const uint64_t adWords=(uint64_t)state.t.batches*2*P*P,xdWords=(uint64_t)state.t.batches*2*state.t.k*P,pairWords=(uint64_t)state.t.batches*16*state.t.k*P;
            state.ad.SetGlobalBuffer((__gm__ half*)(state.gm+splitStart)+(uint64_t)state.batch*4*P*P);
            state.xd.SetGlobalBuffer((__gm__ half*)(state.gm+splitStart+adWords)+(uint64_t)state.batch*4*state.t.k*P);
            state.products.SetGlobalBuffer(state.gm+splitStart+adWords+xdWords+(uint64_t)state.batch*16*state.t.k*P);
            state.expA.SetGlobalBuffer(state.gm+splitStart+adWords+xdWords+pairWords+(uint64_t)state.batch*2*P);
            state.expX.SetGlobalBuffer(state.gm+splitStart+adWords+xdWords+pairWords+(uint64_t)state.batch*2*P+P);
            if(state.lane==0){Duplicate(state.mat,0.0f,P);PipeBarrier<PIPE_ALL>();DataCopy(state.expX,state.mat,P);PipeBarrier<PIPE_ALL>();}
        }

    }
    __aicore__ inline void LoadDirectMatrix(DirectWork &state) {
        for(uint32_t i=0;i<N;++i){state.labels[i]=i;state.order[i]=i;}
        for(uint32_t row=state.begin;row<state.end;row+=T) {
            const uint32_t rows=state.end-row<T?state.end-row:T,realRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
            Duplicate(state.mat,0.0f,rows*state.L);PipeBarrier<PIPE_ALL>();
            if(realRows) {
                DataCopyPad(state.mat,state.ga[(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n],DataCopyExtParams{(uint16_t)realRows,state.t.n*4,0,(state.L-(state.t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-state.t.n%8)%8),0});
                DataCopyPad(state.mat[P],state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+row*state.t.k],DataCopyExtParams{(uint16_t)realRows,state.t.k*4,0,(state.L-(state.t.k+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-state.t.k%8)%8),0});
            }
            PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<rows;++r)if(row+r>=state.t.n)state.mat.SetValue(r*state.L+row+r,1.0f);
            PipeBarrier<PIPE_ALL>();DataCopy(state.matrix[(uint64_t)state.batch*N*state.L+row*state.L],state.mat,rows*state.L);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PrepareDirectPanel(DirectWork &state) {
        state.matrixZeros=state.mat[CAP-2*P];Duplicate(state.matrixZeros,0.0f,state.L);Duplicate(state.active,0.0f,CP);PipeBarrier<PIPE_V>();Duplicate(state.active,1.0f,state.t.n);PipeBarrier<PIPE_ALL>();SyncAll<true>();
        state.panel=state.mat[N*B];state.selected=state.mat[N*PL];state.transform=state.mat[N*B+2*PP*B];
        state.inputIndex=state.mat[INPUT_INDEX].template ReinterpretCast<int32_t>();state.panelScratch=state.mat[PANEL_SCRATCH];
        if(state.lane==0) {

            auto rowIndex=state.panelScratch.template ReinterpretCast<int32_t>();CreateVecIndex(rowIndex,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rowIndex,rowIndex,(int32_t)(B*4),N);PipeBarrier<PIPE_V>();
            Duplicate(state.inputIndex,(int32_t)0,PP*B);PipeBarrier<PIPE_V>();for(uint32_t j=0;j<B;++j)Adds(state.inputIndex[j*PP],rowIndex,(int32_t)(j*4),N);

        }
    }
    __aicore__ inline void FactorDirectPanel(DirectWork &state,uint32_t base) {
            uint32_t panelRows[B]={};const uint32_t pw=state.t.n-base<B?state.t.n-base:B;

            if(state.lane==0) {
                DataCopyPad(state.mat,state.matrix[(uint64_t)state.batch*N*state.L+base],DataCopyExtParams{(uint16_t)N,B*4,(state.L-B)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
                Gather(state.panel,state.mat,state.inputIndex.template ReinterpretCast<uint32_t>(),0,PP*B);Duplicate(state.panel[PP*B],0.0f,PP*B);PipeBarrier<PIPE_V>();
                ColumnPanel<N,PP,B>::Run(state.panel,state.transform,state.active,state.panelScratch,pw,panelRows,pipe);
                DataCopy(state.factors[((uint64_t)state.batch*FactorCols+base)*N],state.transform,N*B);
                for(uint32_t j=0;j<B;++j)state.pivot.template ReinterpretCast<uint32_t>().SetValue(j,j<pw?panelRows[j]:0);
                PipeBarrier<PIPE_ALL>();DataCopy(state.ids[(uint64_t)state.batch*FactorCols+base],state.pivot,B);
            }
            if constexpr(SplitResidual)if(state.lane>0) {
                const uint32_t panels=(state.t.n+B-1)/B,step=(state.teams-1)*panels,first=(base/B)*(state.teams-1)+state.lane-1;
                for(uint32_t r=first;r<P;r+=step) {
                    Duplicate(state.mat,0.0f,P);PipeBarrier<PIPE_ALL>();
                    if(r<state.t.n)DataCopyPad(state.mat,state.ga[((uint64_t)state.batch*state.t.n+r)*state.t.n],DataCopyExtParams{1,state.t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                    PipeBarrier<PIPE_ALL>();SplitRow(state.ad,state.expA,r,P);
                }
            }
            SyncAll<true>();
            if(state.lane!=0){DataCopy(state.pivot,state.ids[(uint64_t)state.batch*FactorCols+base],B);PipeBarrier<PIPE_ALL>();for(uint32_t j=0;j<B;++j)panelRows[j]=state.pivot.template ReinterpretCast<uint32_t>().GetValue(j);}

            for(uint32_t j=0;j<pw;++j){state.labels[panelRows[j]]=base+j;state.order[base+j]=panelRows[j];}
            for(uint32_t j=state.lane;j<B;j+=state.teams) {
                if(j<pw)DataCopy(state.selected[(j/state.teams)*state.L],state.matrix[(uint64_t)state.batch*N*state.L+panelRows[j]*state.L],state.L);
                else Duplicate(state.selected[(j/state.teams)*state.L],0.0f,state.L);
            }
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=state.lane;j<B;j+=state.teams) {
                DataCopy(state.selectedGM[((uint64_t)state.batch*B+j)*state.L],state.selected[(j/state.teams)*state.L],state.L);
                if(j<pw)DataCopy(state.matrix[(uint64_t)state.batch*N*state.L+panelRows[j]*state.L],state.matrixZeros,state.L);
            }
            SyncAll<false>();SyncAll<false>();
    }
    __aicore__ inline void ReorderDirectRhs(DirectWork &state) {
        // Reorder the directly transformed RHS into variable state.order.
        for(uint32_t k=state.lane;k<state.t.k;k+=state.teams) {
            for(uint32_t r=0;r<N;++r)rhsIx.SetValue(r,r*8*4);PipeBarrier<PIPE_ALL>();
            DataCopyPad(a,state.matrix[(uint64_t)state.batch*N*state.L+P+k],DataCopyExtParams{N,4,(state.L-1)*4,0,0},DataCopyPadExtParams<float>{true,0,7,0});PipeBarrier<PIPE_ALL>();
            Gather(x,a,rhsIx,0,N);PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<N;++r)rhsIx.SetValue(r,state.order[r]*4);PipeBarrier<PIPE_ALL>();
            Gather(a,x,rhsIx,0,N);if constexpr(P>N)Duplicate(a[N],0.0f,P-N);PipeBarrier<PIPE_ALL>();DataCopy(state.initial[((uint64_t)state.batch*state.t.k+k)*P],a,P);PipeBarrier<PIPE_ALL>();
            if constexpr(SplitResidual)SplitRow(state.xd,state.expX,k,state.t.k);
        }
        SyncAll<true>();
        for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
            auto ix0=ixBase[4*IT*(4-half)],ix1=ix0[IT*half];for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j){ix0.SetValue(row*half+j,(row*stride+j)*4);ix1.SetValue(row*half+j,(row*stride+half+j)*4);}
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline bool AssembleDirectResidual(DirectWork &state) {
        bool assembled=false;
        if constexpr(SplitResidual){SyncAll<false>();SyncAll<false>();assembled=Base::template SplitAssembleRows<(N<P?5:4)>(state.products,state.expA,state.expX,state.gb,state.residual,state.t,state.batch,state.lane,state.teams);}
        return assembled;
    }
    __aicore__ inline void ComputeDirectResidual(DirectWork &state) {
        for(uint32_t row=state.begin;row<state.end;row+=T) {
            const uint32_t rows=state.end-row<T?state.end-row:T;
            LoadInput(state.ga,(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n,rows,row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0,state.t.n,raw);Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
            for(uint32_t k=0;k<state.t.k;++k) {
                LoadColumn(state.initial,((uint64_t)state.batch*state.t.k+k)*P,rows);AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                const uint32_t rb=k/8*8,kw=state.t.k-rb<8?state.t.k-rb:8,validRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
                Duplicate(a,0.0f,T*8);PipeBarrier<PIPE_ALL>();if(validRows)DataCopyPad(a,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+row*state.t.k+rb],DataCopyExtParams{(uint16_t)validRows,kw*4,(state.t.k-kw)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-kw),0});PipeBarrier<PIPE_ALL>();
                for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k-rb)*4);PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_V>();D(h,x,h,rows);D(h,h,l,rows);PipeBarrier<PIPE_ALL>();DataCopyPad(state.residual[((uint64_t)state.batch*state.t.k+k)*P+row],h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ inline void StoreDirectSolution(DirectWork &state) {
        for(uint32_t row=state.begin;row<state.end;row+=T) {
            const uint32_t rows=state.end-row<T?state.end-row:T;
            for(uint32_t rb=0;rb<state.t.k;rb+=8) {
                const uint32_t kw=state.t.k-rb<8?state.t.k-rb:8;
                for(uint32_t k=rb;k<rb+kw;++k){DataCopyPad(x,state.initial[((uint64_t)state.batch*state.t.k+k)*P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();for(uint32_t i=0;i<rows;++i)state.outputTile.SetValue(i*8+k-rb,x.GetValue(i));}
                PipeBarrier<PIPE_ALL>();const uint32_t validRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;if(validRows)DataCopyPad(state.gx[(uint64_t)state.batch*state.t.n*state.t.k+row*state.t.k+rb],state.outputTile,DataCopyExtParams{(uint16_t)validRows,kw*4,0,(state.t.k-kw)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        DirectWork state;state.t=t;
        InitializeDirectWork(state,aa,bb,xx,workspace);
        LoadDirectMatrix(state);PrepareDirectPanel(state);
        for(uint32_t base=0;base<t.n;base+=B)FactorDirectPanel(state,base);
        ReorderDirectRhs(state);
        if(!AssembleDirectResidual(state))ComputeDirectResidual(state);
        SyncAll<true>();ApplyFactors(state.factors,state.ids,state.initial,state.residual,t,state.batch,state.lane,state.teams,state.order);SyncAll<true>();
        StoreDirectSolution(state);
    }

};

} // namespace SolveDetail
} // namespace AscendC
