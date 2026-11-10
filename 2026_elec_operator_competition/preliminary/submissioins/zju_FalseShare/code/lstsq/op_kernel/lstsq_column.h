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

#include "lstsq_resident.h"

// Deferred row normalization avoids scattered pivot-row writes in column layout.
template<uint32_t N,uint32_t PITCH=N,uint32_t WIDTH=16> struct ColumnPanel {
    static constexpr uint32_t B=WIDTH,SCRATCH=3*N+112+8*B;
    static __aicore__ inline void Run(AscendC::LocalTensor<float> panel,AscendC::LocalTensor<float> result,
        AscendC::LocalTensor<float> active,AscendC::LocalTensor<float> scratch,uint32_t width,uint32_t *rows,AscendC::TPipe &pipe) {
        auto pivot=scratch,candidates=scratch[32+N],maximum=scratch[32+2*N],scales=scratch[48+2*N];
        auto offsets=scratch[48+3*N].template ReinterpretCast<int32_t>(),address=scratch[80+3*N].template ReinterpretCast<int32_t>();
        auto coefficients=scratch[112+3*N];
        AscendC::Duplicate(scales,1.0f,N);AscendC::CreateVecIndex(offsets,(int32_t)0,32);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(offsets,offsets,(int32_t)(PITCH*4),32);AscendC::PipeBarrier<PIPE_V>();
        const auto vs=pipe.FetchEventID(AscendC::HardEvent::V_S),sv=pipe.FetchEventID(AscendC::HardEvent::S_V);
        for(uint32_t j=0;j<width;++j) {
            // The consumed input column is dead after this pivot; reuse it in place.
            auto col=panel[j*PITCH];AscendC::Abs(candidates,col,N);AscendC::PipeBarrier<PIPE_V>();AscendC::Mul(candidates,candidates,active,N);AscendC::PipeBarrier<PIPE_V>();
            if constexpr(N%64==0)AscendC::WholeReduceMax(maximum,candidates,64,N/64,1,1,8);
            else {AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(0,N);AscendC::WholeReduceMax<float,false>(maximum,candidates,AscendC::MASK_PLACEHOLDER,1,1,1,8);AscendC::SetMaskNorm();AscendC::ResetMask();}
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vs);AscendC::WaitFlag<AscendC::HardEvent::V_S>(vs);uint32_t bits,row;AscendC::GetReduceRepeatMaxMinSpr<float>(bits,row);
            const float inverse=1.0f/col.GetValue(row);rows[j]=row;col.SetValue(row,0.0f);active.SetValue(row,0.0f);scales.SetValue(row,inverse);panel.SetValue((B+j)*PITCH+row,1.0f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(sv);AscendC::WaitFlag<AscendC::HardEvent::S_V>(sv);
            // Only the remaining input columns and the first j+1 transform columns
            // are live. Their contiguous window is always exactly B columns.
            AscendC::Adds(address,offsets,(int32_t)(((j+1)*PITCH+row)*4),B);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(pivot,panel,address.template ReinterpretCast<uint32_t>(),0,B);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(pivot,pivot,-inverse,B);AscendC::PipeBarrier<PIPE_V>();AscendC::Brcb(coefficients,pivot,B/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
            const AscendC::BinaryRepeatParams update{1,1,0,PITCH/8,0,1};
            for(uint32_t off=0;off<N;off+=64)AscendC::MulAddDst(panel[(j+1)*PITCH+off],col[off],coefficients,(uint64_t)(N-off<64?N-off:64),(uint8_t)B,update);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // Selected rows stay unnormalized during elimination; apply their saved
        // inverse pivot once when publishing the transform.
        for(uint32_t j=0;j<B;++j)AscendC::Mul(result[j*N],panel[(B+j)*PITCH],scales,N);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
template<uint32_t N,uint32_t C=32,uint32_t Steps=1,bool SharedRhs=false> class RefinedSquare : public VectorArithmetic<256> {
    using Ops=VectorArithmetic<256>;
    static constexpr uint32_t P=256,T=16,SZ=P*T,CP=(N+7)/8*8;
    static constexpr uint32_t B=16,PL=2*B+8;
    static constexpr uint32_t ELIM_CAP=N*(2*C+8+(SharedRhs?8:0))+N*(PL+B)+B*(2*C+(SharedRhs?8:0));
    static constexpr uint32_t ARRAYS=11;
    static constexpr uint32_t CAP0=ELIM_CAP>ARRAYS*SZ+T*8?ELIM_CAP:ARRAYS*SZ+T*8;
    static constexpr uint32_t RHS_CAP=SharedRhs?N*(2*C+16+64):2*N*64;
    static constexpr uint32_t CAP=CAP0>RHS_CAP?CAP0:RHS_CAP;
    AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    AscendC::LocalTensor<float> a,x,ah,al,xh,xl,h,l,s[6],raw,ahi,alo;
    AscendC::LocalTensor<uint32_t> ixBase,rhsIx;
    template<AscendC::HardEvent event> __aicore__ inline void Sync() {
        const auto id=pipe.FetchEventID(event);AscendC::SetFlag<event>(id);AscendC::WaitFlag<event>(id);
    }
    using Ops::A;using Ops::D;using Ops::M;using Ops::Scale;using Ops::Rows;
    __aicore__ inline void DensePairSum(uint32_t count) {
        A(s[0],a,x,count);D(s[1],s[0],a,count);D(s[2],s[0],s[1],count);
        D(s[2],a,s[2],count);D(s[3],x,s[1],count);A(s[2],s[2],s[3],count);
        A(s[2],s[2],a[64],count);A(l,s[2],x[64],count);
        Scale(h,s[0],1.0f,count);
    }
    __aicore__ inline void AlignedRowSum(AscendC::LocalTensor<float> out,AscendC::LocalTensor<float> input,uint32_t rows) {
        for(uint32_t half=P/2;half>=8;half/=2)Rows(input,input,input[half],rows,half);
        AscendC::WholeReduceSum(a,input,8,rows,1,1,P/8);AscendC::PipeBarrier<PIPE_V>();
        Scale(out,a,1.0f,rows);
    }
    __aicore__ inline void BindDotView(AccurateRows<P,T> &view) {
        view.a=a;view.x=x;view.h=h;view.l=l;view.ixBase=ixBase;
        view.raw=raw;view.ahi=ahi;view.alo=alo;view.xh=xh;view.xl=xl;
        for(uint32_t i=0;i<4;++i)view.s[i]=s[i];
    }
    __aicore__ inline void AccurateDot(uint32_t rows) {
        const uint32_t count=rows*P;
        AccurateRows<P,T> view;BindDotView(view);
        view.AccurateProduct(count);
        AscendC::Abs(a,h,count);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(s[0],a,64,count/64,1,1,8);Sync<AscendC::HardEvent::V_S>();
        uint32_t maximumBits,maximumIndex;AscendC::GetReduceRepeatMaxMinSpr<float>(maximumBits,maximumIndex);
        const uint32_t exponent=(maximumBits>>23)&255;
        if(exponent>10 && exponent<240) {view.LatticeRowSum(rows,exponent);return;}
        view.ReducePairs(rows);
    }
    __aicore__ inline void FastDot(uint32_t rows) {
        M(h,a,x,rows*P);
        for(uint32_t half=P/2;half>=8;half/=2)Rows(h,h,h[half],rows,half);
        AscendC::WholeReduceSum(s[0],h,8,rows,1,1,P/8);AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadTile(AscendC::GlobalTensor<float> matrix,uint64_t offset,uint32_t rows,AscendC::LocalTensor<float> tile,uint32_t width=N) {
        AscendC::Duplicate(tile,0.0f,T*P);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(tile,matrix[offset],AscendC::DataCopyExtParams{(uint16_t)rows,width*4,0,(P-((width+7)/8*8))/8,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)((8-width%8)%8),0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadColumn(AscendC::GlobalTensor<float> column,uint64_t offset,uint32_t rows) {
        AscendC::DataCopy(x,column[offset],P);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=1;r<rows;++r)AscendC::DataCopy(x[r*P],x,P);
        AscendC::PipeBarrier<PIPE_V>();
    }
    LinalgLstsqTilingData t;
    static constexpr uint32_t CTL=N*B+B;
    uint32_t lane,group,groups,W,L;
    uint32_t labels[N];
    bool rhs;
    AscendC::LocalTensor<float> mat,ordered,pivot,col,scalar,active,candidates,weights,tmp,outputTile;
    AscendC::LocalTensor<uint32_t> indices;
    AscendC::GlobalTensor<float> ga,gb,gx,ctl,inverse,initial,residual;
    __aicore__ inline void InitViews(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace) {
        lane=AscendC::GetBlockIdx()%t.teams;group=AscendC::GetBlockIdx()/t.teams;groups=t.blocks/t.teams;
        rhs=lane==t.teams-1;
        W=SharedRhs?2*C+(rhs?8:0):(rhs?64:2*C);L=SharedRhs?W+8:(rhs?64:2*C+8);
        pipe.InitBuffer(buf,(CAP+128+14*CP+64)*4);
        mat=buf.Get<float>();ordered=mat[SharedRhs?N*(2*C+16):N*64];pivot=mat[CAP];col=pivot[128];scalar=col[CP];active=col[CP+8];candidates=active[CP];weights=candidates[CP];
        tmp=weights[CP*8];indices=tmp[CP].template ReinterpretCast<uint32_t>();
        ixBase=indices;rhsIx=col.template ReinterpretCast<uint32_t>();
        a=mat;x=mat[SZ];
        h=mat[2*SZ];l=mat[3*SZ];ah=mat[4*SZ];al=mat[5*SZ];xh=mat[6*SZ];xl=mat[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=mat[(4+i)*SZ];
        raw=mat[8*SZ];ahi=mat[9*SZ];alo=mat[10*SZ];
        outputTile=mat[ARRAYS*SZ];

        ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);
        gx.SetGlobalBuffer((__gm__ float*)xx);ctl.SetGlobalBuffer((__gm__ float*)workspace);
        const uint64_t invStart=(uint64_t)groups*2*CTL,initialStart=invStart+(uint64_t)groups*N*N,residualStart=initialStart+(uint64_t)groups*t.k*P;
        inverse.SetGlobalBuffer((__gm__ float*)workspace+invStart);
        initial.SetGlobalBuffer((__gm__ float*)workspace+initialStart);
        residual.SetGlobalBuffer((__gm__ float*)workspace+residualStart);

    }
    __aicore__ inline void LoadRound(uint32_t batch,bool live) {
            for(uint32_t i=0;i<N;++i)labels[i]=i;
            for(uint32_t i=0;i<N;++i)indices.SetValue(i,i*PL*4);
            AscendC::Duplicate(mat,0.0f,CAP);AscendC::Duplicate(active,0.0f,CP);AscendC::PipeBarrier<PIPE_V>();AscendC::Duplicate(active,1.0f,t.n);AscendC::Duplicate(col,0.0f,CP);AscendC::PipeBarrier<PIPE_ALL>();
            if(live) {
                if(!rhs || SharedRhs) {
                    const uint32_t width=N-lane*C<C?N-lane*C:C;
                    const uint32_t realWidth=lane*C<t.n?(t.n-lane*C<C?t.n-lane*C:C):0;
                    if(realWidth)AscendC::DataCopyPad(mat,ga[(uint64_t)batch*t.n*t.n+lane*C],AscendC::DataCopyExtParams{(uint16_t)t.n,realWidth*4,(t.n-realWidth)*4,(L-((realWidth+7)/8*8))/8,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)((8-realWidth%8)%8),0});
                    AscendC::PipeBarrier<PIPE_ALL>();
                    for(uint32_t i=0;i<width;++i) {
                        mat.SetValue((lane*C+i)*L+C+i,1.0f);
                        if(lane*C+i>=t.n)mat.SetValue((lane*C+i)*L+i,1.0f);
                    }
                }
                if(rhs) {
                    const uint32_t qp=(t.k+7)/8*8,rhsOffset=SharedRhs?2*C:0;
                    AscendC::DataCopyPad(mat[rhsOffset],gb[(uint64_t)batch*t.n*t.k],AscendC::DataCopyExtParams{(uint16_t)t.n,t.k*4,0,(L-qp)/8,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)(qp-t.k),0});
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }

    }
    __aicore__ inline void FactorPanel(AscendC::LocalTensor<float> panel,AscendC::LocalTensor<float> transform,AscendC::LocalTensor<float> selected,uint32_t base,uint32_t panelWidth,uint32_t *panelRows) {
                            auto panelIndex=selected[B*W].template ReinterpretCast<int32_t>();
                            auto temporary=selected[B*W+N*B];
                            static_assert(ELIM_CAP+N*B+ColumnPanel<N>::SCRATCH<=CAP);
                            auto rowBytes=temporary.template ReinterpretCast<int32_t>();
                            AscendC::CreateVecIndex(rowBytes,(int32_t)0,N);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(rowBytes,rowBytes,(int32_t)(L*4),N);AscendC::PipeBarrier<PIPE_V>();
                            for(uint32_t j=0;j<B;++j)AscendC::Adds(panelIndex[j*N],rowBytes,(int32_t)(j*4),N);
                            AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(panel,mat[base%C],panelIndex.template ReinterpretCast<uint32_t>(),0,N*B);AscendC::Duplicate(panel[N*B],0.0f,N*B);AscendC::PipeBarrier<PIPE_V>();
                            ColumnPanel<N>::Run(panel,transform,active,temporary,panelWidth,panelRows,pipe);

    }
    __aicore__ inline void PublishPanel(AscendC::LocalTensor<float> transform,uint64_t at,uint32_t panelWidth,const uint32_t *panelRows) {
                        AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(ctl[at],transform,N*B);
                        for(uint32_t j=0;j<B;++j)pivot.template ReinterpretCast<uint32_t>().SetValue(j,j<panelWidth?panelRows[j]:0);
                        AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(ctl[at+N*B],pivot,B);AscendC::PipeBarrier<PIPE_ALL>();

    }
    __aicore__ inline void ApplyPanelTransform(AscendC::LocalTensor<float> transform,AscendC::LocalTensor<float> selected,uint32_t base,uint32_t panelWidth) {
                        // Columns already pivoted are never consumed again; retain all inverse/RHS columns.
                        const uint32_t done=base+panelWidth;
                        const uint32_t first=((!rhs || SharedRhs)?(done<=lane*C?0:(done-lane*C<C?done-lane*C:C)):0)/8*8;
                        const AscendC::BinaryRepeatParams update{1,1,0,(uint8_t)(L/8),0,1};
                        for(uint32_t j=0;j<panelWidth;++j) {
                            AscendC::Brcb(weights,transform[j*N],CP/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                            for(uint32_t off=first;off<W;off+=64) {
                                AscendC::MulAddDst(mat[off],selected[j*W+off],weights,(uint64_t)(W-off<64?W-off:64),(uint8_t)128,update);
                                if constexpr(N>128)AscendC::MulAddDst(mat[128*L+off],selected[j*W+off],weights[128*8],(uint64_t)(W-off<64?W-off:64),(uint8_t)(N-128),update);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();

    }
    __aicore__ inline void UpdateColumns(AscendC::LocalTensor<float> transform,AscendC::LocalTensor<float> selected,uint32_t base,uint32_t owner,uint32_t panelWidth,uint64_t at,uint32_t *panelRows) {
                        if(lane!=owner) {
                            AscendC::DataCopy(transform,ctl[at],N*B);
                            AscendC::DataCopy(pivot,ctl[at+N*B],B);AscendC::PipeBarrier<PIPE_ALL>();
                            for(uint32_t j=0;j<panelWidth;++j)panelRows[j]=pivot.template ReinterpretCast<uint32_t>().GetValue(j);
                        }
                        for(uint32_t j=0;j<panelWidth;++j) {
                            const uint32_t row=panelRows[j];labels[row]=base+j;active.SetValue(row,0);
                            AscendC::DataCopy(selected[j*W],mat[row*L],W);
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for(uint32_t j=0;j<panelWidth;++j)AscendC::Duplicate(mat[panelRows[j]*L],0.0f,W);
                        AscendC::PipeBarrier<PIPE_V>();
                        ApplyPanelTransform(transform,selected,base,panelWidth);

    }
    __aicore__ inline void EliminateRound(bool live) {
        auto panel=mat[N*L],transform=panel[N*PL],selected=transform[N*B];
        for(uint32_t base=0;base<t.n;base+=B) {
            const uint32_t owner=base/C,panelWidth=t.n-base<B?t.n-base:B;
            const uint64_t at=(uint64_t)group*2*CTL+((base/B)&1)*CTL;
            uint32_t panelRows[B];
            if(live && lane==owner) {
                FactorPanel(panel,transform,selected,base,panelWidth,panelRows);
                PublishPanel(transform,at,panelWidth,panelRows);
            }
            AscendC::SyncAll<true>();
            if(live)UpdateColumns(transform,selected,base,owner,panelWidth,at,panelRows);
        }
        for(uint32_t i=0;i<N;++i)indices.SetValue(i,i*(rhs?64:L)*4);AscendC::PipeBarrier<PIPE_ALL>();

    }
    __aicore__ inline void PublishInitial() {
                if(!rhs || SharedRhs) {
                    for(uint32_t r=0;r<N;++r)AscendC::DataCopy(mat[labels[r]*L],mat[r*L+C],C);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    const uint32_t width=N-lane*C<C?N-lane*C:C;
                    AscendC::DataCopyPad(inverse[(uint64_t)group*N*N+lane*C],mat,AscendC::DataCopyExtParams{(uint16_t)N,width*4,(L-width)/8,(N-width)*4,0});
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(rhs) {
                    for(uint32_t r=0;r<N;++r)AscendC::DataCopy(ordered[labels[r]*64],mat[r*L+(SharedRhs?2*C:0)],8);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    for(uint32_t k=0;k<t.k;++k) {
                        AscendC::Duplicate(mat,0.0f,P);AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(mat,ordered[k],indices,0,N);AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopy(initial[((uint64_t)group*t.k+k)*P],mat,P);AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    AscendC::Duplicate(mat,0.0f,P);AscendC::PipeBarrier<PIPE_ALL>();
                    for(uint32_t k=0;k<t.k;++k)AscendC::DataCopy(residual[((uint64_t)group*t.k+k)*P],mat,P);
                }
                AscendC::PipeBarrier<PIPE_ALL>();

    }
    __aicore__ inline void PrepareReductionIndices() {
            for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
                auto ix0=ixBase[4*T*(4-half)],ix1=ix0[T*half];
                for(uint32_t row=0;row<T;++row)for(uint32_t j=0;j<half;++j) {
                    ix0.SetValue(row*half+j,(row*stride+j)*4);
                    ix1.SetValue(row*half+j,(row*stride+half+j)*4);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

    }
    __aicore__ inline void BuildResidual(uint32_t batch,uint32_t begin,uint32_t end) {
                for(uint32_t row=begin;row<end;row+=T) {
                    const uint32_t rows=end-row<T?end-row:T;
                    LoadTile(ga,(uint64_t)batch*t.n*t.n+row*t.n,rows,raw,t.n);
                    Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
                    for(uint32_t k=0;k<t.k;++k) {
                        LoadColumn(initial,((uint64_t)group*t.k+k)*P,rows);
                        AccurateDot(rows);AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopyPad(a,gb[(uint64_t)batch*t.n*t.k+row*t.k],AscendC::DataCopyExtParams{(uint16_t)rows,t.k*4,0,0,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)(8-t.k),0});
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for(uint32_t i=0;i<rows;++i)rhsIx.SetValue(i,(i*8+k)*4);
                        AscendC::PipeBarrier<PIPE_ALL>();AscendC::Gather(x,a,rhsIx,0,rows);AscendC::PipeBarrier<PIPE_V>();
                        D(h,x,h,rows);D(h,h,l,rows);AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopyPad(residual[((uint64_t)group*t.k+k)*P+row],h,AscendC::DataCopyExtParams{1,rows*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }

    }
    __aicore__ inline void CorrectRows(uint32_t batch,uint32_t begin,uint32_t end,uint32_t iteration) {
                for(uint32_t row=begin;row<end;row+=T) {
                    const uint32_t rows=end-row<T?end-row:T;
                    LoadTile(inverse,(uint64_t)group*N*N+row*N,rows,a);
                    for(uint32_t k=0;k<t.k;++k) {
                        LoadColumn(residual,((uint64_t)group*t.k+k)*P,rows);FastDot(rows);AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopyPad(x,initial[((uint64_t)group*t.k+k)*P+row],AscendC::DataCopyExtParams{1,rows*4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
                        A(x,x,s[0],rows);AscendC::PipeBarrier<PIPE_ALL>();
                        for(uint32_t i=0;i<rows;++i)outputTile.SetValue(i*8+k,x.GetValue(i));
                        if constexpr(Steps>1)if(iteration+1<Steps) {
                            AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(initial[((uint64_t)group*t.k+k)*P+row],x,AscendC::DataCopyExtParams{1,rows*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
                        }
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyPad(gx[(uint64_t)batch*t.n*t.k+row*t.k],outputTile,AscendC::DataCopyExtParams{(uint16_t)rows,t.k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
                }

    }
    __aicore__ inline void WriteZeroResidual(uint32_t batch,bool live,GM_ADDR rr) {
            if(live && lane==0 && rr && t.residualElements>=(uint64_t)t.batches*t.k) {
                AscendC::GlobalTensor<float> gr;gr.SetGlobalBuffer((__gm__ float*)rr);
                AscendC::Duplicate(mat,0.0f,8);AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyPad(gr[(uint64_t)batch*t.k],mat,AscendC::DataCopyExtParams{1,t.k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
            }

    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR rr,GM_ADDR workspace,const LinalgLstsqTilingData &data) {
        t=data;InitViews(aa,bb,xx,workspace);
        for(uint32_t round=0;round<(t.batches+groups-1)/groups;++round) {
            const uint32_t batch=round*groups+group;const bool live=batch<t.batches;
            LoadRound(batch,live);EliminateRound(live);
            if(live)PublishInitial();
            AscendC::SyncAll<true>();
            const uint32_t chunk=(N+t.teams-1)/t.teams,begin=lane*chunk,end=begin+chunk<t.n?begin+chunk:t.n;
            PrepareReductionIndices();
            for(uint32_t iteration=0;iteration<Steps;++iteration) {
                if(live)BuildResidual(batch,begin,end);
                AscendC::SyncAll<true>();
                if(live)CorrectRows(batch,begin,end,iteration);
                AscendC::SyncAll<true>();
            }
            WriteZeroResidual(batch,live,rr);AscendC::SyncAll<true>();
        }
    }

};
