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
#include "linalg_solve_common.h"

namespace AscendC {
namespace SolveDetail {
// Deferred row normalization avoids scattered pivot-row stores in column layout.
template<uint32_t N,uint32_t PITCH=N,uint32_t WIDTH=16> struct ColumnPanel {
    static constexpr uint32_t B=WIDTH,SCRATCH=3*N+112+8*B;
    static __aicore__ inline void Run(LocalTensor<float> panel,LocalTensor<float> result,
        LocalTensor<float> active,LocalTensor<float> scratch,uint32_t width,uint32_t *rows,TPipe &pipe) {
        auto pivot=scratch,candidates=scratch[32+N],maximum=scratch[32+2*N],scales=scratch[48+2*N];
        auto offsets=scratch[48+3*N].template ReinterpretCast<int32_t>(),address=scratch[80+3*N].template ReinterpretCast<int32_t>();
        auto coefficients=scratch[112+3*N];
        Duplicate(scales,1.0f,N);CreateVecIndex(offsets,(int32_t)0,32);PipeBarrier<PIPE_V>();Muls(offsets,offsets,(int32_t)(PITCH*4),32);PipeBarrier<PIPE_V>();
        const auto vs=pipe.FetchEventID(HardEvent::V_S),sv=pipe.FetchEventID(HardEvent::S_V);
        for(uint32_t j=0;j<width;++j) {
            // The consumed input column is dead after this pivot; reuse it in place.
            auto col=panel[j*PITCH];Abs(candidates,col,N);PipeBarrier<PIPE_V>();Mul(candidates,candidates,active,N);PipeBarrier<PIPE_V>();
            if constexpr(N%64==0)WholeReduceMax(maximum,candidates,64,N/64,1,1,8);
            else {SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,N);WholeReduceMax<float,false>(maximum,candidates,MASK_PLACEHOLDER,1,1,1,8);SetMaskNorm();ResetMask();}
            SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);uint32_t bits,row;GetReduceRepeatMaxMinSpr<float>(bits,row);
            const float inverse=1.0f/col.GetValue(row);rows[j]=row;col.SetValue(row,0.0f);active.SetValue(row,0.0f);scales.SetValue(row,inverse);panel.SetValue((B+j)*PITCH+row,1.0f);
            SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);
            // Only the remaining input columns and the first j+1 transform columns
            // are live. Their contiguous window is always exactly B columns.
            Adds(address,offsets,(int32_t)(((j+1)*PITCH+row)*4),B);PipeBarrier<PIPE_V>();Gather(pivot,panel,address.template ReinterpretCast<uint32_t>(),0,B);PipeBarrier<PIPE_V>();
            Muls(pivot,pivot,-inverse,B);PipeBarrier<PIPE_V>();Brcb(coefficients,pivot,B/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            const BinaryRepeatParams update{1,1,0,PITCH/8,0,1};
            for(uint32_t off=0;off<N;off+=64)MulAddDst(panel[(j+1)*PITCH+off],col[off],coefficients,(uint64_t)(N-off<64?N-off:64),(uint8_t)B,update);
            PipeBarrier<PIPE_V>();
        }
        // Selected rows stay unnormalized during elimination; apply their saved
        // inverse pivot once when publishing the transform.
        for(uint32_t j=0;j<B;++j)Mul(result[j*N],panel[(B+j)*PITCH],scales,N);
        PipeBarrier<PIPE_ALL>();
    }
};
// Shared compensated row arithmetic. Both elimination families keep the
// same tensor views, reduction ordering and event ownership in this base.
template<uint32_t P,uint32_t T> class CompensatedRows {
protected:
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> a,x,ah,al,xh,xl,h,l,s[6],raw,ahi,alo;
    LocalTensor<uint32_t> ixBase,rhsIx;
    template<HardEvent event> __aicore__ inline void Sync() {
        const auto id=pipe.FetchEventID(event);SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void A(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Add(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void D(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Sub(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void M(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t z) {Mul(o,u,v,z);PipeBarrier<PIPE_V>();}
    __aicore__ inline void Scale(LocalTensor<float> o,LocalTensor<float> u,float v,uint32_t z) {Muls(o,u,v,z);PipeBarrier<PIPE_V>();}
    template<bool sub=false> __aicore__ inline void Rows(LocalTensor<float> o,LocalTensor<float> u,LocalTensor<float> v,uint32_t rows,uint32_t width) {
        const BinaryRepeatParams step{1,1,1,P/8,P/8,P/8};
        for(uint32_t off=0;off<width;off+=64) {
            uint64_t mask=width-off<64?width-off:64;
            if constexpr(sub)Sub(o[off],u[off],v[off],mask,(uint8_t)rows,step);
            else Add(o[off],u[off],v[off],mask,(uint8_t)rows,step);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DensePairSum(uint32_t count) {
        A(s[0],a,x,count);D(s[1],s[0],a,count);D(s[2],s[0],s[1],count);
        D(s[2],a,s[2],count);D(s[3],x,s[1],count);A(s[2],s[2],s[3],count);
        A(s[2],s[2],a[64],count);A(l,s[2],x[64],count);
        Scale(h,s[0],1.0f,count);
    }
    __aicore__ inline void AlignedRowSum(LocalTensor<float> out,LocalTensor<float> input,uint32_t rows) {
        for(uint32_t half=P==384?256:P/2;half>=8;half/=2)Rows(input,input,input[half],rows,half<P-half?half:P-half);
        WholeReduceSum(a,input,8,rows,1,1,P/8);PipeBarrier<PIPE_V>();
        Scale(out,a,1.0f,rows);
    }
    __aicore__ inline void FormProducts(uint32_t rows) {
        const uint32_t count=rows*P;
        M(h,raw,x,count);
        Scale(xh,x,4097.0f,count);D(xl,xh,x,count);D(xh,xh,xl,count);D(xl,x,xh,count);
        M(l,ahi,xh,count);D(l,l,h,count);
        M(a,ahi,xl,count);A(l,l,a,count);
        M(a,alo,xh,count);A(l,l,a,count);
        M(a,alo,xl,count);A(l,l,a,count);
    }
    __aicore__ inline bool QuantizedReduction(uint32_t rows) {
        const uint32_t count=rows*P;
        Abs(a,h,count);PipeBarrier<PIPE_V>();
        WholeReduceMax(s[0],a,64,count/64,1,1,8);Sync<HardEvent::V_S>();
        uint32_t maximumBits,maximumIndex;GetReduceRepeatMaxMinSpr<float>(maximumBits,maximumIndex);
        const uint32_t exponent=(maximumBits>>23)&255;
        if(exponent>10 && exponent<240) {
            constexpr uint32_t logP=P>256?9:(P==256?8:7);
            union ScaleBits {uint32_t bits;float value;};
            ScaleBits q0,q1;q0.bits=((exponent+logP+1)<<23)|0x400000;
            q1.bits=((exponent+2*logP-22)<<23)|0x400000;
            Adds(s[0],h,q0.value,count);PipeBarrier<PIPE_V>();Adds(s[0],s[0],-q0.value,count);PipeBarrier<PIPE_V>();
            D(s[1],h,s[0],count);
            Adds(s[2],s[1],q1.value,count);PipeBarrier<PIPE_V>();Adds(s[2],s[2],-q1.value,count);PipeBarrier<PIPE_V>();
            D(s[3],s[1],s[2],count);
            AlignedRowSum(s[0],s[0],rows);AlignedRowSum(s[1],s[2],rows);
            AlignedRowSum(s[2],s[3],rows);AlignedRowSum(s[3],l,rows);
            A(a,s[2],s[3],rows);
            A(x,s[1],a,rows);D(h,x,s[1],rows);D(l,x,h,rows);D(l,s[1],l,rows);D(h,a,h,rows);A(l,l,h,rows);
            A(h,s[0],x,rows);D(s[1],h,s[0],rows);D(s[2],h,s[1],rows);D(s[2],s[0],s[2],rows);D(s[1],x,s[1],rows);A(a,s[2],s[1],rows);A(l,a,l,rows);
            return true;
        }
        return false;
    }
    __aicore__ inline void AccurateDot(uint32_t rows) {
        FormProducts(rows);
        if(QuantizedReduction(rows))return;
        for(uint32_t half=P==384?256:P/2;half>=8;half/=2) {
            const uint32_t width=half<P-half?half:P-half;
            Rows(s[0],h,h[half],rows,width);Rows<true>(s[1],s[0],h,rows,width);
            Rows<true>(s[2],s[0],s[1],rows,width);Rows<true>(s[2],h,s[2],rows,width);
            Rows<true>(s[3],h[half],s[1],rows,width);Rows(s[2],s[2],s[3],rows,width);
            Rows(s[2],s[2],l,rows,width);Rows(l,s[2],l[half],rows,width);
            for(uint32_t off=0;off<width;off+=64)
                Muls(h[off],s[0][off],1.0f,(uint64_t)(width-off<64?width-off:64),(uint8_t)rows,UnaryRepeatParams{1,1,P/8,P/8});
            PipeBarrier<PIPE_V>();
        }
        for(uint32_t half=4;half;half/=2) {
            auto ix0=ixBase[4*T*(4-half)],ix1=ix0[T*half];
            Gather(a,h,ix0,0,rows*half);Gather(x,h,ix1,0,rows*half);
            Gather(a[64],l,ix0,0,rows*half);Gather(x[64],l,ix1,0,rows*half);
            PipeBarrier<PIPE_V>();
            DensePairSum(rows*half);
        }
    }
    __aicore__ inline void FastDot(uint32_t rows) {
        M(h,a,x,rows*P);
        for(uint32_t half=P==384?256:P/2;half>=8;half/=2)Rows(h,h,h[half],rows,half<P-half?half:P-half);
        WholeReduceSum(s[0],h,8,rows,1,1,P/8);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadColumn(GlobalTensor<float> column,uint64_t offset,uint32_t rows) {
        DataCopy(x,column[offset],P);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=1;r<rows;++r)DataCopy(x[r*P],x,P);
        PipeBarrier<PIPE_V>();
    }

};

template<uint32_t N,uint32_t C=32,uint32_t Steps=1,bool SharedRhs=false> class RefinedSquare : public CompensatedRows<256,16> {
    static constexpr uint32_t P=256,T=16,SZ=P*T,CP=(N+7)/8*8;
    static constexpr uint32_t B=16,PL=2*B+8;
    static constexpr uint32_t ELIM_CAP=N*(2*C+8+(SharedRhs?8:0))+N*(PL+B)+B*(2*C+(SharedRhs?8:0));
    static constexpr uint32_t ARRAYS=11;
    static constexpr uint32_t CAP0=ELIM_CAP>ARRAYS*SZ+T*8?ELIM_CAP:ARRAYS*SZ+T*8;
    static constexpr uint32_t RHS_CAP=SharedRhs?N*(2*C+16+64):2*N*64;
    static constexpr uint32_t CAP=CAP0>RHS_CAP?CAP0:RHS_CAP;
    __aicore__ inline void LoadTile(GlobalTensor<float> matrix,uint64_t offset,uint32_t rows,LocalTensor<float> tile,uint32_t width=N) {
        Duplicate(tile,0.0f,T*P);PipeBarrier<PIPE_ALL>();
        DataCopyPad(tile,matrix[offset],DataCopyExtParams{(uint16_t)rows,width*4,0,(P-((width+7)/8*8))/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-width%8)%8),0});
        PipeBarrier<PIPE_ALL>();
    }
public:
    struct RefinedWork {
        LinalgSolveTilingData t;
        uint32_t lane,group,groups,W,L,CTL,batch,base,owner,panelWidth,chunk,begin,end;
        uint64_t at;
        bool rhs,live;
        uint32_t labels[N],panelRows[B];
        LocalTensor<float> mat,ordered,pivot,col,scalar,active,candidates,weights,tmp,outputTile,panel,transform,selected;
        LocalTensor<uint32_t> indices;
        GlobalTensor<float> ga,gb,gx,ctl,inverse,initial,residual;
    };
    __aicore__ inline void InitializeRefined(RefinedWork &state,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace) {
        state.lane=GetBlockIdx()%state.t.teams;state.group=GetBlockIdx()/state.t.teams;state.groups=state.t.blocks/state.t.teams;
        state.rhs=state.lane==state.t.teams-1;
        state.W=SharedRhs?2*C+(state.rhs?8:0):(state.rhs?64:2*C);state.L=SharedRhs?state.W+8:(state.rhs?64:2*C+8);state.CTL=N*B+B;
        pipe.InitBuffer(buf,(CAP+128+14*CP+64)*4);
        state.mat=buf.Get<float>();state.ordered=state.mat[SharedRhs?N*(2*C+16):N*64];state.pivot=state.mat[CAP];state.col=state.pivot[128];state.scalar=state.col[CP];state.active=state.col[CP+8];state.candidates=state.active[CP];state.weights=state.candidates[CP];
        state.tmp=state.weights[CP*8];state.indices=state.tmp[CP].template ReinterpretCast<uint32_t>();
        ixBase=state.indices;rhsIx=state.col.template ReinterpretCast<uint32_t>();
        a=state.mat;x=state.mat[SZ];
        h=state.mat[2*SZ];l=state.mat[3*SZ];ah=state.mat[4*SZ];al=state.mat[5*SZ];xh=state.mat[6*SZ];xl=state.mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=state.mat[(4+i)*SZ];
        raw=state.mat[8*SZ];ahi=state.mat[9*SZ];alo=state.mat[10*SZ];
        state.outputTile=state.mat[ARRAYS*SZ];

        state.ga.SetGlobalBuffer((__gm__ float*)aa);state.gb.SetGlobalBuffer((__gm__ float*)bb);
        state.gx.SetGlobalBuffer((__gm__ float*)xx);state.ctl.SetGlobalBuffer((__gm__ float*)workspace);
        const uint64_t invStart=(uint64_t)state.groups*2*state.CTL,initialStart=invStart+(uint64_t)state.groups*N*N,residualStart=initialStart+(uint64_t)state.groups*state.t.k*P;
        state.inverse.SetGlobalBuffer((__gm__ float*)workspace+invStart);
        state.initial.SetGlobalBuffer((__gm__ float*)workspace+initialStart);
        state.residual.SetGlobalBuffer((__gm__ float*)workspace+residualStart);
    }
    __aicore__ inline void LoadRefinedRound(RefinedWork &state,uint32_t round) {
            state.batch=round*state.groups+state.group;state.live=state.batch<state.t.batches;

            for(uint32_t i=0;i<N;++i)state.labels[i]=i;
            for(uint32_t i=0;i<N;++i)state.indices.SetValue(i,i*PL*4);
            Duplicate(state.mat,0.0f,CAP);Duplicate(state.active,0.0f,CP);PipeBarrier<PIPE_V>();Duplicate(state.active,1.0f,state.t.n);Duplicate(state.col,0.0f,CP);PipeBarrier<PIPE_ALL>();
            if(state.live) {
                if(!state.rhs || SharedRhs) {
                    const uint32_t width=N-state.lane*C<C?N-state.lane*C:C;
                    const uint32_t realWidth=state.lane*C<state.t.n?(state.t.n-state.lane*C<C?state.t.n-state.lane*C:C):0;
                    if(realWidth)DataCopyPad(state.mat,state.ga[(uint64_t)state.batch*state.t.n*state.t.n+state.lane*C],DataCopyExtParams{(uint16_t)state.t.n,realWidth*4,(state.t.n-realWidth)*4,(state.L-((realWidth+7)/8*8))/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-realWidth%8)%8),0});
                    PipeBarrier<PIPE_ALL>();
                    for(uint32_t i=0;i<width;++i) {
                        state.mat.SetValue((state.lane*C+i)*state.L+C+i,1.0f);
                        if(state.lane*C+i>=state.t.n)state.mat.SetValue((state.lane*C+i)*state.L+i,1.0f);
                    }
                }
                if(state.rhs) {
                    const uint32_t qp=(state.t.k+7)/8*8,rhsOffset=SharedRhs?2*C:0;
                    DataCopyPad(state.mat[rhsOffset],state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k],DataCopyExtParams{(uint16_t)state.t.n,state.t.k*4,0,(state.L-qp)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(qp-state.t.k),0});
                }
                PipeBarrier<PIPE_ALL>();
            }
    }
    __aicore__ inline void PublishPanel(RefinedWork &state) {
                    if(state.live && state.lane==state.owner) {

                            auto panelIndex=state.selected[B*state.W].template ReinterpretCast<int32_t>();
                            auto temporary=state.selected[B*state.W+N*B];
                            static_assert(ELIM_CAP+N*B+ColumnPanel<N>::SCRATCH<=CAP);
                            auto rowBytes=temporary.template ReinterpretCast<int32_t>();
                            CreateVecIndex(rowBytes,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(rowBytes,rowBytes,(int32_t)(state.L*4),N);PipeBarrier<PIPE_V>();
                            for(uint32_t j=0;j<B;++j)Adds(panelIndex[j*N],rowBytes,(int32_t)(j*4),N);
                            PipeBarrier<PIPE_V>();Gather(state.panel,state.mat[state.base%C],panelIndex.template ReinterpretCast<uint32_t>(),0,N*B);Duplicate(state.panel[N*B],0.0f,N*B);PipeBarrier<PIPE_V>();
                            ColumnPanel<N>::Run(state.panel,state.transform,state.active,temporary,state.panelWidth,state.panelRows,pipe);

                        PipeBarrier<PIPE_ALL>();DataCopy(state.ctl[state.at],state.transform,N*B);
                        for(uint32_t j=0;j<B;++j)state.pivot.template ReinterpretCast<uint32_t>().SetValue(j,j<state.panelWidth?state.panelRows[j]:0);
                        PipeBarrier<PIPE_ALL>();DataCopy(state.ctl[state.at+N*B],state.pivot,B);PipeBarrier<PIPE_ALL>();
                    }
    }
    __aicore__ inline void ApplyPanel(RefinedWork &state) {
                    if(state.live) {
                        if(state.lane!=state.owner) {
                            DataCopy(state.transform,state.ctl[state.at],N*B);
                            DataCopy(state.pivot,state.ctl[state.at+N*B],B);PipeBarrier<PIPE_ALL>();
                            for(uint32_t j=0;j<state.panelWidth;++j)state.panelRows[j]=state.pivot.template ReinterpretCast<uint32_t>().GetValue(j);
                        }
                        for(uint32_t j=0;j<state.panelWidth;++j) {
                            const uint32_t row=state.panelRows[j];state.labels[row]=state.base+j;state.active.SetValue(row,0);
                            DataCopy(state.selected[j*state.W],state.mat[row*state.L],state.W);
                        }
                        PipeBarrier<PIPE_ALL>();
                        for(uint32_t j=0;j<state.panelWidth;++j)Duplicate(state.mat[state.panelRows[j]*state.L],0.0f,state.W);
                        PipeBarrier<PIPE_V>();
                        // Columns already pivoted are never consumed again; retain all state.inverse/RHS columns.
                        const uint32_t done=state.base+state.panelWidth;
                        const uint32_t first=(!state.rhs || SharedRhs)?(done<=state.lane*C?0:(done-state.lane*C<C?done-state.lane*C:C)):0;
                        const BinaryRepeatParams update{1,1,0,(uint8_t)(state.L/8),0,1};
                        for(uint32_t j=0;j<state.panelWidth;++j) {
                            Brcb(state.weights,state.transform[j*N],CP/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                            for(uint32_t off=first;off<state.W;off+=64) {
                                MulAddDst(state.mat[off],state.selected[j*state.W+off],state.weights,(uint64_t)(state.W-off<64?state.W-off:64),(uint8_t)128,update);
                                if constexpr(N>128)MulAddDst(state.mat[128*state.L+off],state.selected[j*state.W+off],state.weights[128*8],(uint64_t)(state.W-off<64?state.W-off:64),(uint8_t)(N-128),update);
                            }
                            PipeBarrier<PIPE_V>();
                        }
                        PipeBarrier<PIPE_ALL>();
                    }    }
    __aicore__ inline void FactorRefinedPanels(RefinedWork &state) {
        state.panel=state.mat[N*state.L];state.transform=state.panel[N*PL];state.selected=state.transform[N*B];
        for(state.base=0;state.base<state.t.n;state.base+=B) {
            state.owner=state.base/C;state.panelWidth=state.t.n-state.base<B?state.t.n-state.base:B;
            state.at=(uint64_t)state.group*2*state.CTL+((state.base/B)&1)*state.CTL;
            PublishPanel(state);
            SyncAll<true>();
            ApplyPanel(state);
        }
        for(uint32_t i=0;i<N;++i)state.indices.SetValue(i,i*(state.rhs?64:state.L)*4);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void StoreInitialRefined(RefinedWork &state) {
            if(state.live) {
                if(!state.rhs || SharedRhs) {
                    for(uint32_t r=0;r<N;++r)DataCopy(state.mat[state.labels[r]*state.L],state.mat[r*state.L+C],C);
                    PipeBarrier<PIPE_ALL>();
                    const uint32_t width=N-state.lane*C<C?N-state.lane*C:C;
                    DataCopyPad(state.inverse[(uint64_t)state.group*N*N+state.lane*C],state.mat,DataCopyExtParams{(uint16_t)N,width*4,(state.L-width)/8,(N-width)*4,0});
                    PipeBarrier<PIPE_ALL>();
                }
                if(state.rhs) {
                    for(uint32_t r=0;r<N;++r)DataCopy(state.ordered[state.labels[r]*64],state.mat[r*state.L+(SharedRhs?2*C:0)],8);
                    PipeBarrier<PIPE_ALL>();
                    for(uint32_t k=0;k<state.t.k;++k) {
                        Duplicate(state.mat,0.0f,P);PipeBarrier<PIPE_V>();Gather(state.mat,state.ordered[k],state.indices,0,N);PipeBarrier<PIPE_ALL>();
                        DataCopy(state.initial[((uint64_t)state.group*state.t.k+k)*P],state.mat,P);PipeBarrier<PIPE_ALL>();
                    }
                    Duplicate(state.mat,0.0f,P);PipeBarrier<PIPE_ALL>();
                    for(uint32_t k=0;k<state.t.k;++k)DataCopy(state.residual[((uint64_t)state.group*state.t.k+k)*P],state.mat,P);
                }
                PipeBarrier<PIPE_ALL>();
            }
            SyncAll<true>();
    }
    __aicore__ inline void PrepareRefinedReduction(RefinedWork &state) {
            state.chunk=(N+state.t.teams-1)/state.t.teams;state.begin=state.lane*state.chunk;state.end=state.begin+state.chunk<state.t.n?state.begin+state.chunk:state.t.n;
            for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
                auto ix0=ixBase[4*T*(4-half)],ix1=ix0[T*half];
                for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j) {
                    ix0.SetValue(row*half+j,(row*stride+j)*4);
                    ix1.SetValue(row*half+j,(row*stride+half+j)*4);
                }
            }
            PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void RefinedResidualRows(RefinedWork &state) {
                if(state.live)for(uint32_t row=state.begin;row<state.end;row+=T) {
                    const uint32_t rows=state.end-row<T?state.end-row:T;
                    LoadTile(state.ga,(uint64_t)state.batch*state.t.n*state.t.n+row*state.t.n,rows,raw,state.t.n);
                    Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
                    for(uint32_t k=0;k<state.t.k;++k) {
                        LoadColumn(state.initial,((uint64_t)state.group*state.t.k+k)*P,rows);
                        AccurateDot(rows);PipeBarrier<PIPE_ALL>();
                        DataCopyPad(a,state.gb[(uint64_t)(state.t.broadcastB?0:state.batch)*state.t.n*state.t.k+row*state.t.k],DataCopyExtParams{(uint16_t)rows,state.t.k*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(8-state.t.k),0});
                        PipeBarrier<PIPE_ALL>();
                        for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k)*4);
                        PipeBarrier<PIPE_ALL>();Gather(x,a,rhsIx,0,rows);PipeBarrier<PIPE_V>();
                        D(h,x,h,rows);D(h,h,l,rows);PipeBarrier<PIPE_ALL>();
                        DataCopyPad(state.residual[((uint64_t)state.group*state.t.k+k)*P+row],h,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                    }
                }
    }
    __aicore__ inline void RefinedCorrectionRows(RefinedWork &state,uint32_t iteration) {
                if(state.live)for(uint32_t row=state.begin;row<state.end;row+=T) {
                    const uint32_t rows=state.end-row<T?state.end-row:T;
                    LoadTile(state.inverse,(uint64_t)state.group*N*N+row*N,rows,a);
                    for(uint32_t k=0;k<state.t.k;++k) {
                        LoadColumn(state.residual,((uint64_t)state.group*state.t.k+k)*P,rows);FastDot(rows);PipeBarrier<PIPE_ALL>();
                        DataCopyPad(x,state.initial[((uint64_t)state.group*state.t.k+k)*P+row],DataCopyExtParams{1,rows*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
                        A(x,x,s[0],rows);PipeBarrier<PIPE_ALL>();
                        for(uint32_t i=0;i<rows;++i)state.outputTile.SetValue(i*8+k,x.GetValue(i));
                        if constexpr(Steps>1)if(iteration+1<Steps) {
                            PipeBarrier<PIPE_ALL>();DataCopyPad(state.initial[((uint64_t)state.group*state.t.k+k)*P+row],x,DataCopyExtParams{1,rows*4,0,0,0});PipeBarrier<PIPE_ALL>();
                        }
                    }
                    PipeBarrier<PIPE_ALL>();
                    DataCopyPad(state.gx[(uint64_t)state.batch*state.t.n*state.t.k+row*state.t.k],state.outputTile,DataCopyExtParams{(uint16_t)rows,state.t.k*4,0,0,0});PipeBarrier<PIPE_ALL>();
                }
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        RefinedWork state;state.t=t;
        InitializeRefined(state,aa,bb,xx,workspace);
        for(uint32_t round=0;round<(t.batches+state.groups-1)/state.groups;++round) {
            LoadRefinedRound(state,round);
            FactorRefinedPanels(state);
            StoreInitialRefined(state);
            PrepareRefinedReduction(state);
            for(uint32_t iteration=0;iteration<Steps;++iteration) {
                RefinedResidualRows(state);SyncAll<true>();
                RefinedCorrectionRows(state,iteration);SyncAll<true>();
            }
            SyncAll<true>();
        }
    }

};

} // namespace SolveDetail
} // namespace AscendC
