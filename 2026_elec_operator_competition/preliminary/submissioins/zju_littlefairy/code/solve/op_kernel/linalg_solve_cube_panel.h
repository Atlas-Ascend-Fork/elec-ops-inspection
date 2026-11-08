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
#include "linalg_solve_panel_math.h"

namespace AscendC {
namespace SolveDetail {
template<uint32_t N,bool MatrixRhs=false,bool SplitResidual=false> class CubePanel : public CubePanelMath<N> {
    using Math=CubePanelMath<N>;
public:
    struct CubeServiceWork {
        GM_ADDR workspace;
        GM_ADDR rhs;
        uint32_t batches,order,rhsCount,broadcastB,tm,tn,mr,nr,per,rblocks,cblocks,tiles;
        uint64_t coeffStart,selectedStart,invStart,initialStart,residualStart;
        GlobalTensor<float> matrix,coefficient,selected,gb,inverse,initial,residual;
    };
    template<class MM> static __aicore__ inline void ServiceFactorPanels(CubeServiceWork &work,MM &mm) {
        for(uint32_t base=0;base<work.order;base+=Math::B) {
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<work.batches*work.per;task+=GetBlockNum()) {
                const uint32_t batch=task/work.per,tile=task%work.per,row=tile/work.nr*work.tm,column=tile%work.nr*work.tn;
                // G is stored as B x N; the logical tile starts at column row.
                if constexpr(MatrixRhs) {mm.SetOrgShape(N,2*N,Math::B);mm.SetSingleShape(work.tm,work.tn,Math::B);}
                mm.SetTensorA(work.coefficient[(uint64_t)batch*N*Math::B+row],true);
                mm.SetTensorB(work.selected[(uint64_t)batch*Math::B*2*N+column]);
                if constexpr(MatrixRhs)mm.SetTail(N-row<work.tm?N-row:work.tm,2*N-column<work.tn?2*N-column:work.tn,Math::B);
                else mm.SetTail(N-row<work.tm?N-row:work.tm,2*N-column<work.tn?2*N-column:work.tn);
                mm.IterateAll(work.matrix[(uint64_t)batch*N*2*N+row*2*N+column],1);mm.End();
            }
            SyncAll<false>();
        }
    }
    template<class MM> static __aicore__ inline void ServiceInitialRhs(CubeServiceWork &work,MM &mm) {
            work.invStart=work.selectedStart+(uint64_t)work.batches*Math::B*2*N+work.batches*Math::B;
            work.initialStart=work.invStart+(uint64_t)work.batches*N*N+(uint64_t)work.batches*work.rhsCount*Math::P;
            work.residualStart=work.initialStart+(uint64_t)work.batches*work.rhsCount*Math::P;

            work.gb.SetGlobalBuffer((__gm__ float*)work.rhs);work.inverse.SetGlobalBuffer((__gm__ float*)work.workspace+work.invStart);
            work.initial.SetGlobalBuffer((__gm__ float*)work.workspace+work.initialStart);work.residual.SetGlobalBuffer((__gm__ float*)work.workspace+work.residualStart);
            work.rblocks=(work.rhsCount+work.tm-1)/work.tm;work.cblocks=(N+work.tn-1)/work.tn;work.tiles=work.rblocks*work.cblocks;
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<work.batches*work.tiles;task+=GetBlockNum()) {
                const uint32_t batch=task/work.tiles,tile=task%work.tiles,row=tile/work.cblocks*work.tm,column=tile%work.cblocks*work.tn;
                mm.SetOrgShape(work.rhsCount,N,work.order,N,N);mm.SetSingleShape(work.tm,work.tn,work.order);
                mm.SetTensorA(work.gb[(uint64_t)(work.broadcastB?0:batch)*work.order*work.rhsCount+row],true);
                mm.SetTensorB(work.inverse[(uint64_t)batch*N*N+column*N],true);
                mm.SetTail(work.rhsCount-row<work.tm?work.rhsCount-row:work.tm,N-column<work.tn?N-column:work.tn,work.order);
                mm.IterateAll(work.initial[((uint64_t)batch*work.rhsCount+row)*Math::P+column],0);mm.End();
            }
            SyncAll<false>();
    }
    template<class MM> static __aicore__ inline void ServiceCorrectRhs(CubeServiceWork &work,MM &mm) {
            SyncAll<false>();
            for(uint32_t task=GetBlockIdx();task<work.batches*work.tiles;task+=GetBlockNum()) {
                const uint32_t batch=task/work.tiles,tile=task%work.tiles,row=tile/work.cblocks*work.tm,column=tile%work.cblocks*work.tn;
                mm.SetOrgShape(work.rhsCount,N,N);mm.SetSingleShape(work.tm,work.tn,N);
                mm.SetTensorA(work.residual[((uint64_t)batch*work.rhsCount+row)*Math::P],false);
                mm.SetTensorB(work.inverse[(uint64_t)batch*N*N+column*N],true);
                mm.SetTail(work.rhsCount-row<work.tm?work.rhsCount-row:work.tm,N-column<work.tn?N-column:work.tn,N);
                mm.IterateAll(work.initial[((uint64_t)batch*work.rhsCount+row)*Math::P+column],1);mm.End();
            }
            SyncAll<false>();
    }
    template<class MM> static __aicore__ inline void ServiceSplitProducts(CubeServiceWork &work,MM &mm,const tiling::TCubeTiling &ht) {
            if constexpr(SplitResidual) {
                const uint64_t splitStart=work.residualStart+(uint64_t)work.batches*work.rhsCount*N;
                const uint64_t adWords=(uint64_t)work.batches*2*N*N,xdWords=(uint64_t)work.batches*2*work.rhsCount*N;
                GlobalTensor<half> ad,xd;GlobalTensor<float> products;
                ad.SetGlobalBuffer((__gm__ half*)((__gm__ float*)work.workspace+splitStart));
                xd.SetGlobalBuffer((__gm__ half*)((__gm__ float*)work.workspace+splitStart+adWords));
                products.SetGlobalBuffer((__gm__ float*)work.workspace+splitStart+adWords+xdWords);
                const uint32_t hmM=ht.singleCoreM,hmN=ht.singleCoreN,hmR=(work.rhsCount+hmM-1)/hmM,hmC=(N+hmN-1)/hmN,hmTiles=hmR*hmC;
                SyncAll<false>();
                for(uint32_t task=GetBlockIdx();task<work.batches*10*hmTiles;task+=GetBlockNum()) {
                    const uint32_t bt=task/(10*hmTiles),tile=task%hmTiles,row=tile/hmC*hmM,column=tile%hmC*hmN;
                    const uint32_t pair=task/hmTiles%10;
                    uint32_t i=0,j=pair;
                    while(j>=4-i) {j-=4-i;++i;}
                    mm.SetOrgShape(work.rhsCount,N,N);mm.SetSingleShape(hmM,hmN,N);
                    mm.SetTensorA(xd[((uint64_t)bt*4+j)*work.rhsCount*N+row*N],false);
                    mm.SetTensorB(ad[((uint64_t)bt*4+i)*N*N+column*N],true);
                    mm.SetTail(work.rhsCount-row<hmM?work.rhsCount-row:hmM,N-column<hmN?N-column:hmN,N);
                    mm.IterateAll(products[(((uint64_t)bt*16+i*4+j)*work.rhsCount+row)*N+column],0);mm.End();
                }
                SyncAll<false>();
            }
    }
public:
    static __aicore__ inline void Service(GM_ADDR workspace,GM_ADDR system,const tiling::TCubeTiling &tc,const tiling::TCubeTiling &ht,GM_ADDR rhs,uint32_t batches,uint32_t order,uint32_t rhsCount,uint32_t broadcastB) {
        using F=MatmulType<TPosition::GM,CubeFormat::ND,float>;
        using H=MatmulType<TPosition::GM,CubeFormat::ND,half>;
        TPipe cubePipe;Matmul<F,F,F> mm;Matmul<H,H,F> hm;
        if constexpr(SplitResidual) {REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc,hm,&ht);}
        else {REGIST_MATMUL_OBJ(&cubePipe,system,mm,&tc);}
        mm.SetHF32(false);
        CubeServiceWork work;
        work.workspace=workspace;work.rhs=rhs;work.batches=batches;work.order=order;work.rhsCount=rhsCount;work.broadcastB=broadcastB;
        work.coeffStart=(uint64_t)work.batches*N*2*N;work.selectedStart=work.coeffStart+(uint64_t)work.batches*N*Math::B;

        work.matrix.SetGlobalBuffer((__gm__ float*)work.workspace);
        work.coefficient.SetGlobalBuffer((__gm__ float*)work.workspace+work.coeffStart);
        work.selected.SetGlobalBuffer((__gm__ float*)work.workspace+work.selectedStart);
        work.tm=tc.singleCoreM;work.tn=tc.singleCoreN;work.mr=(N+work.tm-1)/work.tm;work.nr=(2*N+work.tn-1)/work.tn;work.per=work.mr*work.nr;
        ServiceFactorPanels(work,mm);
        if constexpr(MatrixRhs) {
            ServiceInitialRhs(work,mm);
            if constexpr(SplitResidual)ServiceSplitProducts(work,hm,ht);
            ServiceCorrectRhs(work,mm);
        }
    }
    struct CubeWork {
        LinalgSolveTilingData t;
        uint32_t cores,id,batch,lane,teams,chunk,begin,end,L,cw,firstColumn,width;
        uint64_t coeffStart,selectedStart,idsStart,invStart,inputStart,initialStart,residualStart;
        __gm__ float *gm;
        uint32_t labels[N];
        LocalTensor<float> mat,pivot,col,active,outputTile,matrixZeros,panel,selected,transform,panelScratch;
        LocalTensor<uint32_t> indices;
        LocalTensor<int32_t> inputIndex;
        GlobalTensor<float> ga,gb,gx,matrix,coefficient,selectedGM,ids,inverse,bcolumns,initial,residual;
    };
    __aicore__ inline void InitializeCubeWork(CubeWork &state,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace) {
        state.cores=GetBlockNum()*2;state.id=GetBlockIdx();state.batch=state.id%state.t.batches;state.lane=state.id/state.t.batches;
        state.teams=(state.cores+state.t.batches-1-state.batch)/state.t.batches;state.chunk=(N+state.teams-1)/state.teams;state.begin=state.lane*state.chunk;state.end=state.begin+state.chunk<N?state.begin+state.chunk:N;
        state.L=2*N;state.cw=(state.L+state.teams*8-1)/(state.teams*8)*8;state.firstColumn=state.lane*state.cw;state.width=state.firstColumn<state.L?(state.L-state.firstColumn<state.cw?state.L-state.firstColumn:state.cw):0;
        this->pipe.InitBuffer(this->buf,(Math::CAP+128+2*Math::CP+28*Math::IT)*4);
        state.mat=this->buf.template Get<float>();state.pivot=state.mat[Math::CAP];state.col=state.pivot[128];state.active=state.col[Math::CP];
        state.indices=state.active[Math::CP].template ReinterpretCast<uint32_t>();
        this->ixBase=state.indices;this->rhsIx=state.col.template ReinterpretCast<uint32_t>();
        this->a=state.mat;this->x=state.mat[Math::SZ];this->h=state.mat[2*Math::SZ];this->l=state.mat[3*Math::SZ];this->ah=state.mat[4*Math::SZ];this->al=state.mat[5*Math::SZ];this->xh=state.mat[6*Math::SZ];this->xl=state.mat[7*Math::SZ];
        for(uint32_t i=0;i<4;++i)this->s[i]=state.mat[(4+i)*Math::SZ];
        this->raw=state.mat[8*Math::SZ];this->ahi=state.mat[9*Math::SZ];this->alo=state.mat[10*Math::SZ];state.outputTile=state.mat[Math::ARRAYS*Math::SZ];

        state.ga.SetGlobalBuffer((__gm__ float*)aa);state.gb.SetGlobalBuffer((__gm__ float*)bb);
        state.gx.SetGlobalBuffer((__gm__ float*)xx);
        state.coeffStart=(uint64_t)state.t.batches*N*state.L;state.selectedStart=state.coeffStart+(uint64_t)state.t.batches*N*Math::B;
        state.idsStart=state.selectedStart+(uint64_t)state.t.batches*Math::B*state.L;state.invStart=state.idsStart+state.t.batches*Math::B;
        state.inputStart=state.invStart+(uint64_t)state.t.batches*N*N;state.initialStart=state.inputStart+(uint64_t)state.t.batches*state.t.k*Math::P;state.residualStart=state.initialStart+(uint64_t)state.t.batches*state.t.k*Math::P;
        state.gm=(__gm__ float*)workspace;
        state.matrix.SetGlobalBuffer(state.gm);state.coefficient.SetGlobalBuffer(state.gm+state.coeffStart);state.selectedGM.SetGlobalBuffer(state.gm+state.selectedStart);state.ids.SetGlobalBuffer(state.gm+state.idsStart);
        state.inverse.SetGlobalBuffer(state.gm+state.invStart);state.bcolumns.SetGlobalBuffer(state.gm+state.inputStart);state.initial.SetGlobalBuffer(state.gm+state.initialStart);state.residual.SetGlobalBuffer(state.gm+state.residualStart);
        for(uint32_t i=0;i<N;++i)state.labels[i]=i;
    }
    __aicore__ inline void LoadAugmentedMatrix(CubeWork &state) {
        for(uint32_t row=state.begin;row<state.end;row+=Math::T) {
            const uint32_t rows=state.end-row<Math::T?state.end-row:Math::T;
            Duplicate(state.mat,0.0f,rows*state.L);PipeBarrier<PIPE_ALL>();
            const uint32_t realRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
            if(realRows)DataCopyPad(state.mat,state.ga[(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n],DataCopyExtParams{(uint16_t)realRows,state.t.n*4,0,(state.L-(state.t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-state.t.n%8)%8),0});
            PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<rows;++r) {
                state.mat.SetValue(r*state.L+N+row+r,1.0f);
                if(row+r>=state.t.n)state.mat.SetValue(r*state.L+row+r,1.0f);
            }
            PipeBarrier<PIPE_ALL>();DataCopy(state.matrix[(uint64_t)state.batch*N*state.L+row*state.L],state.mat,rows*state.L);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PreparePanelWork(CubeWork &state) {
        state.matrixZeros=state.mat[Math::CAP-state.L];
        static_assert(N*Math::PL+Math::B*((2*N)/4)<=Math::CAP-(2*N));
        Duplicate(state.matrixZeros,0.0f,state.L);Duplicate(state.active,0.0f,Math::CP);PipeBarrier<PIPE_V>();Duplicate(state.active,1.0f,state.t.n);Duplicate(state.col,0.0f,Math::CP);
        PipeBarrier<PIPE_ALL>();SyncAll<true>();
        state.panel=state.mat[N*Math::B];state.selected=state.mat[N*Math::PL];
        state.transform=state.mat[N*Math::B+2*Math::PP*Math::B];
        state.inputIndex=state.mat[Math::INPUT_INDEX].template ReinterpretCast<int32_t>();
        state.panelScratch=state.mat[Math::PANEL_SCRATCH];
        // Selected-row storage is reused after the state.panel/state.transform is published.
        // Persistent state.indices and helper scratch stay above both phase layouts.
        static_assert(2*N*Math::B+2*Math::PP*Math::B<=Math::INPUT_INDEX);
        static_assert(N*Math::PL+Math::B*((2*N)/4)<=Math::INPUT_INDEX);
        static_assert(Math::INPUT_INDEX+Math::PP*Math::B<=Math::PANEL_SCRATCH);
        static_assert(Math::PANEL_SCRATCH+ColumnPanel<N,Math::PP,Math::B>::SCRATCH<=Math::CAP-(2*N));
        if(state.lane==0) {
            auto rowIndex=state.panelScratch.template ReinterpretCast<int32_t>();
            CreateVecIndex(rowIndex,(int32_t)0,N);PipeBarrier<PIPE_V>();
            Muls(rowIndex,rowIndex,(int32_t)(Math::B*4),N);PipeBarrier<PIPE_V>();
            Duplicate(state.inputIndex,(int32_t)0,Math::PP*Math::B);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<Math::B;++j)Adds(state.inputIndex[j*Math::PP],rowIndex,(int32_t)(j*4),N);
        }
    }
    __aicore__ inline void FactorCubePanel(CubeWork &state,uint32_t base) {
            uint32_t panelRows[Math::B]={};
            const uint32_t panelWidth=state.t.n-base<Math::B?state.t.n-base:Math::B;
            if(state.lane==0) {
                DataCopyPad(state.mat,state.matrix[(uint64_t)state.batch*N*state.L+base],DataCopyExtParams{(uint16_t)N,Math::B*4,(state.L-Math::B)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                Gather(state.panel,state.mat,state.inputIndex.template ReinterpretCast<uint32_t>(),0,Math::PP*Math::B);
                Duplicate(state.panel[Math::PP*Math::B],0.0f,Math::PP*Math::B);PipeBarrier<PIPE_V>();
                ColumnPanel<N,Math::PP,Math::B>::Run(state.panel,state.transform,state.active,state.panelScratch,state.t.n-base<Math::B?state.t.n-base:Math::B,panelRows,this->pipe);
                DataCopy(state.coefficient[(uint64_t)state.batch*N*Math::B],state.transform,N*Math::B);
                for(uint32_t j=0;j<Math::B;++j)state.pivot.template ReinterpretCast<uint32_t>().SetValue(j,j<panelWidth?panelRows[j]:0);
                PipeBarrier<PIPE_ALL>();DataCopy(state.ids[state.batch*Math::B],state.pivot,Math::B);
            }
            SyncAll<true>();
            if(state.lane!=0) {
                DataCopy(state.pivot,state.ids[state.batch*Math::B],Math::B);PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<Math::B;++j)panelRows[j]=state.pivot.template ReinterpretCast<uint32_t>().GetValue(j);
            }
            for(uint32_t j=0;j<panelWidth;++j)state.labels[panelRows[j]]=base+j;
            for(uint32_t j=state.lane;j<Math::B;j+=state.teams) {
                if(j<panelWidth)DataCopy(state.selected[(j/state.teams)*state.L],state.matrix[(uint64_t)state.batch*N*state.L+panelRows[j]*state.L],state.L);
                else Duplicate(state.selected[(j/state.teams)*state.L],0.0f,state.L);
            }
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=state.lane;j<Math::B;j+=state.teams) {
                DataCopy(state.selectedGM[((uint64_t)state.batch*Math::B+j)*state.L],state.selected[(j/state.teams)*state.L],state.L);
                if(j<panelWidth)DataCopy(state.matrix[(uint64_t)state.batch*N*state.L+panelRows[j]*state.L],state.matrixZeros,state.L);
            }
            SyncAll<false>();SyncAll<false>();
    }
    __aicore__ inline void StoreInverseAndRhs(CubeWork &state) {
        for(uint32_t row=state.begin;row<state.end;row+=Math::T) {
            const uint32_t rows=state.end-row<Math::T?state.end-row:Math::T;
            DataCopyPad(this->a,state.matrix[(uint64_t)state.batch*N*state.L+row*state.L+N],DataCopyExtParams{(uint16_t)rows,N*4,(state.L-N)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<rows;++r)DataCopy(state.inverse[(uint64_t)state.batch*N*N+state.labels[row+r]*N],this->a[r*N],N);
            PipeBarrier<PIPE_ALL>();
            const uint32_t realRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
            if constexpr(!MatrixRhs)for(uint32_t rb=0;rb<state.t.k;rb+=8) {
                state.width=state.t.k-rb<8?state.t.k-rb:8;
                Duplicate(this->a,0.0f,Math::T*8);PipeBarrier<PIPE_ALL>();
                if(realRows)DataCopyPad(this->a,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+row*state.t.k+rb],DataCopyExtParams{(uint16_t)realRows,state.width*4,(state.t.k-state.width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-state.width),0});
                PipeBarrier<PIPE_ALL>();
                for(uint32_t k=rb;k<rb+state.width;++k) {
                    for(uint32_t r=0;r<rows;++r)this->rhsIx.SetValue(r,(r*8+k-rb)*4);
                    PipeBarrier<PIPE_ALL>();Gather(this->x,this->a,this->rhsIx,0,rows);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(state.bcolumns[((uint64_t)state.batch*state.t.k+k)*Math::P+row],this->x,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
        }
        SyncAll<true>();
    }
    __aicore__ inline void PrepareCubeReduction(CubeWork &state) {
        for(uint32_t half=4,stride=Math::P;half;stride=half,half/=2) {
            auto ix0=this->ixBase[4*Math::IT*(4-half)],ix1=ix0[Math::IT*half];
            for(uint32_t row=0;row<Math::T;++row)for(uint32_t j=0;j<half;++j) {
                ix0.SetValue(row*half+j,(row*stride+j)*4);ix1.SetValue(row*half+j,(row*stride+half+j)*4);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void InitialCubeSolution(CubeWork &state) {
        if constexpr(MatrixRhs) {SyncAll<false>();SyncAll<false>();}
        else {
        for(uint32_t row=state.begin;row<state.end;row+=Math::T) {
            const uint32_t rows=state.end-row<Math::T?state.end-row:Math::T;
            this->LoadTile(state.inverse,(uint64_t)state.batch*N*N+row*N,rows,this->a);
            for(uint32_t k=0;k<state.t.k;++k) {
                this->LoadColumn(state.bcolumns,((uint64_t)state.batch*state.t.k+k)*Math::P,rows);this->FastDot(rows);PipeBarrier<PIPE_ALL>();
                DataCopyPad(state.initial[((uint64_t)state.batch*state.t.k+k)*Math::P+row],this->s[0],DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
        }
        SyncAll<true>();
    }
    __aicore__ inline bool AssembleSplitResidual(CubeWork &state) {
        bool assembled=false;
        if constexpr(SplitResidual) {
            const uint64_t splitStart=state.residualStart+(uint64_t)state.t.batches*state.t.k*N;
            const uint64_t adWords=(uint64_t)state.t.batches*2*N*N,xdWords=(uint64_t)state.t.batches*2*state.t.k*N,pairWords=(uint64_t)state.t.batches*16*state.t.k*N;
            GlobalTensor<half> ad,xd;GlobalTensor<float> products,expA,expX;
            ad.SetGlobalBuffer((__gm__ half*)(state.gm+splitStart)+(uint64_t)state.batch*4*N*N);
            xd.SetGlobalBuffer((__gm__ half*)(state.gm+splitStart+adWords)+(uint64_t)state.batch*4*state.t.k*N);
            products.SetGlobalBuffer(state.gm+splitStart+adWords+xdWords+(uint64_t)state.batch*16*state.t.k*N);
            expA.SetGlobalBuffer(state.gm+splitStart+adWords+xdWords+pairWords+(uint64_t)state.batch*2*N);
            expX.SetGlobalBuffer(state.gm+splitStart+adWords+xdWords+pairWords+(uint64_t)state.batch*2*N+N);
            for(uint32_t r=state.lane;r<N;r+=state.teams) {
                Duplicate(state.mat,0.0f,N);PipeBarrier<PIPE_ALL>();
                if(r<state.t.n)DataCopyPad(state.mat,state.ga[((uint64_t)state.batch*state.t.n+r)*state.t.n],DataCopyExtParams{1,state.t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();this->SplitRow(ad,expA,r,N);
                if(r<state.t.k) {
                    DataCopy(state.mat,state.initial[((uint64_t)state.batch*state.t.k+r)*N],N);PipeBarrier<PIPE_ALL>();this->SplitRow(xd,expX,r,state.t.k);
                } else {
                    state.mat.SetValue(0,0.0f);PipeBarrier<PIPE_ALL>();DataCopyPad(expX[r],state.mat,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
            }
            SyncAll<false>();SyncAll<false>();
            assembled=this->SplitAssemble(products,expA,expX,state.gb,state.residual,state.t,state.batch,state.lane,state.teams);
        }
        return assembled;
    }
    __aicore__ inline void ComputeScalarResidual(CubeWork &state) {
        for(uint32_t row=state.begin;row<state.end;row+=Math::T) {
            const uint32_t rows=state.end-row<Math::T?state.end-row:Math::T;
            this->LoadInput(state.ga,(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n,rows,row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0,state.t.n,this->raw);
            this->Scale(this->ahi,this->raw,4097.0f,rows*Math::P);this->D(this->alo,this->ahi,this->raw,rows*Math::P);this->D(this->ahi,this->ahi,this->alo,rows*Math::P);this->D(this->alo,this->raw,this->ahi,rows*Math::P);
            for(uint32_t k=0;k<state.t.k;++k) {
                this->LoadColumn(state.initial,((uint64_t)state.batch*state.t.k+k)*Math::P,rows);this->AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                const uint32_t rb=k/8*8;state.width=state.t.k-rb<8?state.t.k-rb:8;
                Duplicate(this->a,0.0f,Math::T*8);PipeBarrier<PIPE_ALL>();
                const uint32_t validRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
                if(validRows)DataCopyPad(this->a,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+row*state.t.k+rb],DataCopyExtParams{(uint16_t)validRows,state.width*4,(state.t.k-state.width)*4,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-state.width),0});PipeBarrier<PIPE_ALL>();
                for(uint32_t i=0;i<rows;++i)this->rhsIx.SetValue(i,(i*8+k-rb)*4);
                PipeBarrier<PIPE_ALL>();Gather(this->x,this->a,this->rhsIx,0,rows);PipeBarrier<PIPE_V>();this->D(this->h,this->x,this->h,rows);this->D(this->h,this->h,this->l,rows);PipeBarrier<PIPE_ALL>();
                DataCopyPad(state.residual[((uint64_t)state.batch*state.t.k+k)*Math::P+row],this->h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ inline void StoreMatrixSolution(CubeWork &state) {

            SyncAll<false>();SyncAll<false>();
            constexpr uint32_t OW=32;
            auto packed=state.mat[OW*Math::IT];auto transposeIx=state.mat[2*OW*Math::IT].template ReinterpretCast<int32_t>();
            CreateVecIndex(transposeIx,(int32_t)0,OW);PipeBarrier<PIPE_V>();
            Muls(transposeIx,transposeIx,(int32_t)(Math::IT*4),OW);PipeBarrier<PIPE_V>();
            for(uint32_t r=1;r<Math::T;++r)Adds(transposeIx[r*OW],transposeIx,(int32_t)(r*4),OW);
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=state.begin;row<state.end;row+=Math::T) {
                const uint32_t rows=state.end-row<Math::T?state.end-row:Math::T;
                const uint32_t validRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
                if(!validRows)continue;
                for(uint32_t rb=0;rb<state.t.k;rb+=OW) {
                    state.width=state.t.k-rb<OW?state.t.k-rb:OW;
                    Duplicate(this->a,0.0f,OW*Math::IT);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(this->a,state.initial[((uint64_t)state.batch*state.t.k+rb)*Math::P+row],DataCopyExtParams{(uint16_t)state.width,rows*4,(Math::P-rows)*4,(Math::IT-(rows+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-rows%8)%8),0});
                    PipeBarrier<PIPE_ALL>();Gather(packed,this->a,transposeIx.template ReinterpretCast<uint32_t>(),0,rows*OW);PipeBarrier<PIPE_ALL>();
                    DataCopyPad(state.gx[(uint64_t)state.batch*state.t.n*state.t.k+row*state.t.k+rb],packed,DataCopyExtParams{(uint16_t)validRows,state.width*4,(OW-(state.width+7)/8*8)/8,(state.t.k-state.width)*4,0});PipeBarrier<PIPE_ALL>();
                }
            }
    }
    __aicore__ inline void StoreVectorSolution(CubeWork &state) {

        for(uint32_t row=state.begin;row<state.end;row+=Math::T) {
            const uint32_t rows=state.end-row<Math::T?state.end-row:Math::T;
            this->LoadTile(state.inverse,(uint64_t)state.batch*N*N+row*N,rows,this->a);
            for(uint32_t rb=0;rb<state.t.k;rb+=8) {
                state.width=state.t.k-rb<8?state.t.k-rb:8;
                for(uint32_t k=rb;k<rb+state.width;++k) {
                this->LoadColumn(state.residual,((uint64_t)state.batch*state.t.k+k)*Math::P,rows);this->FastDot(rows);PipeBarrier<PIPE_ALL>();
                DataCopyPad(this->x,state.initial[((uint64_t)state.batch*state.t.k+k)*Math::P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
                this->A(this->x,this->x,this->s[0],rows);PipeBarrier<PIPE_ALL>();
                for(uint32_t i=0;i<rows;++i)state.outputTile.SetValue(i*8+k-rb,this->x.GetValue(i));
            }
                PipeBarrier<PIPE_ALL>();
                const uint32_t validRows=row<state.t.n?(state.t.n-row<rows?state.t.n-row:rows):0;
                if(validRows)DataCopyPad(state.gx[(uint64_t)state.batch*state.t.n*state.t.k+row*state.t.k+rb],state.outputTile,DataCopyExtParams{(uint16_t)validRows,state.width*4,0,(state.t.k-state.width)*4,0});PipeBarrier<PIPE_ALL>();
            }
        }
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        CubeWork state;state.t=t;
        InitializeCubeWork(state,aa,bb,xx,workspace);
        LoadAugmentedMatrix(state);
        PreparePanelWork(state);
        for(uint32_t base=0;base<t.n;base+=Math::B)FactorCubePanel(state,base);
        StoreInverseAndRhs(state);
        PrepareCubeReduction(state);
        InitialCubeSolution(state);
        if(!AssembleSplitResidual(state))ComputeScalarResidual(state);
        SyncAll<true>();
        if constexpr(MatrixRhs)StoreMatrixSolution(state);
        else StoreVectorSolution(state);
    }

};

} // namespace SolveDetail
} // namespace AscendC
