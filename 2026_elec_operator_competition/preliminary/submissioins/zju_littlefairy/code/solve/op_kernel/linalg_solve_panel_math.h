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
#include "linalg_solve_refined.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

namespace AscendC {
namespace SolveDetail {
template<uint32_t N> class CubePanelMath : public CompensatedRows<N,(N<=256?16:8)> {
public:
    // Internal panel interface shared by the direct and normal solvers.
    static constexpr uint32_t P=N,T=N<=256?16:8,SZ=P*T,CP=(N+7)/8*8,IT=(T+7)/8*8;
    static constexpr uint32_t B=N<=256?32:16,PL=2*B+8,PP=N+8,ARRAYS=11;
    static constexpr uint32_t INPUT_INDEX=2*N*B+2*PP*B;
    static constexpr uint32_t PANEL_SCRATCH=INPUT_INDEX+PP*B;
    static constexpr uint32_t PANEL_CAP=PANEL_SCRATCH+ColumnPanel<N,PP,B>::SCRATCH+2*N;
    static constexpr uint32_t DOT_CAP=ARRAYS*SZ+T*8;
    static constexpr uint32_t CAP=PANEL_CAP>DOT_CAP?PANEL_CAP:DOT_CAP;
    static_assert(N*PL+((B+3)/4)*(2*P)<=CAP-2*P);
    using DotBase=CompensatedRows<N,(N<=256?16:8)>;
    using DotBase::pipe;
    using DotBase::buf;
    using DotBase::a;
    using DotBase::x;
    using DotBase::ah;
    using DotBase::al;
    using DotBase::xh;
    using DotBase::xl;
    using DotBase::h;
    using DotBase::l;
    using DotBase::s;
    using DotBase::raw;
    using DotBase::ahi;
    using DotBase::alo;
    using DotBase::ixBase;
    using DotBase::rhsIx;
    using DotBase::A;
    using DotBase::D;
    using DotBase::M;
    using DotBase::Scale;
    using DotBase::Rows;
    using DotBase::DensePairSum;
    using DotBase::AlignedRowSum;
    using DotBase::AccurateDot;
    using DotBase::FastDot;
    using DotBase::LoadColumn;
    using DotBase::Sync;
    __aicore__ inline void LoadTile(GlobalTensor<float> matrix,uint64_t offset,uint32_t rows,LocalTensor<float> tile) {
        Duplicate(tile,0.0f,T*P);PipeBarrier<PIPE_ALL>();
        DataCopyPad(tile,matrix[offset],DataCopyExtParams{(uint16_t)rows,N*4,0,(P-N)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadInput(GlobalTensor<float> matrix,uint64_t offset,uint32_t rows,uint32_t realRows,uint32_t width,LocalTensor<float> tile) {
        Duplicate(tile,0.0f,T*P);PipeBarrier<PIPE_ALL>();
        if(realRows)DataCopyPad(tile,matrix[offset],DataCopyExtParams{(uint16_t)realRows,width*4,0,(P-(width+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-width%8)%8),0});
        PipeBarrier<PIPE_ALL>();
    }
    // Four base-256 signed digits. Each half-input integer dot is exact:
    // 512 terms of magnitude at most 128*128 sum to at most 2^23.
    __aicore__ inline void SplitRow(GlobalTensor<half> digits,GlobalTensor<float> exponents,uint32_t row,uint32_t rowCount) {
        auto w=buf.template Get<float>(),v=w,q=w[N],ab=w[2*N],mx=w[3*N];
        auto hd=w[4*N].template ReinterpretCast<half>();
        Abs(ab,v,N);PipeBarrier<PIPE_V>();WholeReduceMax(mx,ab,64,N/64,1,1,8);this->template Sync<HardEvent::V_S>();
        uint32_t mb,mi;GetReduceRepeatMaxMinSpr<float>(mb,mi);
        const int32_t e=(int32_t)((mb>>23)&255)-127;
        int32_t exponent=e-6;
        if((mb&0x7fffffff)==0) {exponent=0;Duplicate(v,0.0f,N);}
        else if(e<-100 || e>100) {exponent=1000;Duplicate(v,0.0f,N);}
        else {
            union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;
            Scale(v,v,scale.value,N);
        }
        for(uint32_t d=0;d<4;++d) {
            Adds(q,v,12582912.0f,N);PipeBarrier<PIPE_V>();Adds(q,q,-12582912.0f,N);PipeBarrier<PIPE_V>();
            Cast(hd[d*N],q,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();
            D(v,v,q,N);Scale(v,v,256.0f,N);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(digits[(uint64_t)row*N],hd,DataCopyExtParams{4,N*2,0,(rowCount-1)*N*2,0});
        mx.SetValue(0,(float)exponent);PipeBarrier<PIPE_ALL>();
        DataCopyPad(exponents[row],mx,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
    }
    template<bool Minimum> __aicore__ inline int32_t ExponentExtreme(LocalTensor<float> src,LocalTensor<float> scratch,uint32_t count) {
        int32_t result=Minimum?10000:-10000;
        uint32_t index;union {uint32_t bits;float value;} value;
        if(count>=64) {
            if constexpr(Minimum)WholeReduceMin(scratch,src,64,(uint8_t)(count/64),1,1,8);
            else WholeReduceMax(scratch,src,64,(uint8_t)(count/64),1,1,8);
            this->template Sync<HardEvent::V_S>();GetReduceRepeatMaxMinSpr<float>(value.bits,index);result=(int32_t)value.value;
        }
        if(count%64) {
            if constexpr(Minimum)WholeReduceMin(scratch,src[count/64*64],(uint64_t)(count%64),1,1,1,8);
            else WholeReduceMax(scratch,src[count/64*64],(uint64_t)(count%64),1,1,1,8);
            this->template Sync<HardEvent::V_S>();GetReduceRepeatMaxMinSpr<float>(value.bits,index);
            const int32_t tail=(int32_t)value.value;
            result=Minimum?(tail<result?tail:result):(tail>result?tail:result);
        }
        return result;
    }
    template<bool VectorBounds> __aicore__ inline bool ValidExponentRange(LocalTensor<float> ae,LocalTensor<float> xe,LocalTensor<float> high,uint32_t rhsCount) {
        // Every lane reads all exponents, so the fallback decision is uniform.
        int32_t amin,amax,xmin,xmax;
        if constexpr(VectorBounds) {
            amin=ExponentExtreme<true>(ae,high,N);amax=ExponentExtreme<false>(ae,high,N);
            xmin=ExponentExtreme<true>(xe,high,rhsCount);xmax=ExponentExtreme<false>(xe,high,rhsCount);
        } else {
        amin=10000;amax=-10000;xmin=10000;xmax=-10000;
        for(uint32_t r=0;r<N;++r) {
            const int32_t av=(int32_t)ae.GetValue(r),xv=(int32_t)xe.GetValue(r);
            amin=av<amin?av:amin;amax=av>amax?av:amax;
            if(r<rhsCount) {xmin=xv<xmin?xv:xmin;xmax=xv>xmax?xv:xmax;}
        }
        }
        if(amax>=1000 || xmax>=1000 || amin+xmin<-50 || amax+xmax>100)return false;
        return true;
    }
    __aicore__ inline bool SplitAssemble(GlobalTensor<float> pairs,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> gb,GlobalTensor<float> residual,const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams) {
        auto w=buf.template Get<float>(),ae=w,xe=w[N],high=w[2*N],low=w[3*N],gs=w[4*N],tmp=w[5*N],term=w[6*N],u=w[7*N],v=w[8*N],factor=w[9*N],br=w[10*N];
        auto ix=w[18*N].template ReinterpretCast<int32_t>();
        auto integerSum=w[11*N].template ReinterpretCast<int32_t>(),rounded=w[12*N].template ReinterpretCast<int32_t>(),remainder=w[13*N].template ReinterpretCast<int32_t>();
        auto groupLow=w[14*N];
        DataCopy(ae,expA,N);DataCopy(xe,expX,N);PipeBarrier<PIPE_ALL>();
        if(!ValidExponentRange<false>(ae,xe,high,t.k))return false;
        CreateVecIndex(ix,(int32_t)0,N);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)32,N);PipeBarrier<PIPE_V>();
        for(uint32_t k=lane;k<t.k;k+=teams) {
            Duplicate(br,0.0f,N*8);Duplicate(low,0.0f,N);PipeBarrier<PIPE_ALL>();
            DataCopyPad(br,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+k],DataCopyExtParams{(uint16_t)t.n,4,(t.k-1)*4,0,0},DataCopyPadExtParams<float>{true,0,7,0});
            PipeBarrier<PIPE_ALL>();Gather(high,br,ix.template ReinterpretCast<uint32_t>(),0,N);PipeBarrier<PIPE_V>();
            const int32_t xq=(int32_t)xe.GetValue(k);
            for(uint32_t d=0;d<4;++d) {
                if(d<2)Duplicate(gs,0.0f,N);else Duplicate(integerSum,(int32_t)0,N);PipeBarrier<PIPE_V>();
                const uint32_t count=d+1;
                // Same-weight pairs are three matrix planes apart. Fetch them
                // together, after the previous vector reads of tmp complete.
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(tmp,pairs[((uint64_t)d*t.k+k)*N],DataCopyExtParams{(uint16_t)count,N*4,(3*t.k-1)*N*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                if(d<2)for(uint32_t i=0;i<count;++i)A(gs,gs,tmp[i*N],N);
                else {
                    // A weight group may exceed 2^24. Sum in int32, then
                    // retain the float conversion's remainder as a low part.
                    for(uint32_t i=0;i<count;++i) {
                        Cast(rounded,tmp[i*N],RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();Add(integerSum,integerSum,rounded,N);PipeBarrier<PIPE_V>();
                    }
                    Cast(gs,integerSum,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();Cast(rounded,gs,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();
                    Sub(remainder,integerSum,rounded,N);PipeBarrier<PIPE_V>();Cast(groupLow,remainder,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();
                }
                Adds(factor,ae,(float)(xq-8*(int32_t)d+127),N);PipeBarrier<PIPE_V>();
                Cast(factor.template ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();
                ShiftLeft(factor.template ReinterpretCast<int32_t>(),factor.template ReinterpretCast<int32_t>(),(int32_t)23,N);PipeBarrier<PIPE_V>();
                M(term,gs,factor,N);Scale(term,term,-1.0f,N);
                A(tmp,high,term,N);D(u,tmp,high,N);D(v,tmp,u,N);D(v,high,v,N);D(u,term,u,N);A(u,u,v,N);A(low,low,u,N);Scale(high,tmp,1.0f,N);
                if(d>=2) {M(term,groupLow,factor,N);Scale(term,term,-1.0f,N);A(low,low,term,N);}
            }
            A(high,high,low,N);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N],high,N);PipeBarrier<PIPE_ALL>();
        }
        return true;
    }
    template<uint32_t Weights=4,bool VectorBounds=false> __aicore__ inline bool SplitAssembleRows(GlobalTensor<float> pairs,GlobalTensor<float> expA,GlobalTensor<float> expX,GlobalTensor<float> gb,GlobalTensor<float> residual,const LinalgSolveTilingData &t,uint32_t batch,uint32_t lane,uint32_t teams) {
        static_assert(N==256);constexpr uint32_t C=128,R=N/C;
        auto w=buf.template Get<float>(),ae=w,xe=w[N],high=w[2*N],low=w[3*N],gs=w[4*N],tmp=w[5*N],term=w[6*N],u=w[7*N],v=w[8*N],factor=w[9*N],br=w[10*N];
        auto ix=w[18*N].template ReinterpretCast<int32_t>();
        auto integerSum=w[11*N].template ReinterpretCast<int32_t>(),rounded=w[12*N].template ReinterpretCast<int32_t>(),remainder=w[13*N].template ReinterpretCast<int32_t>();
        auto groupLow=w[14*N];
        DataCopy(ae,expA,N);DataCopy(xe,expX,N);PipeBarrier<PIPE_ALL>();
        if(!ValidExponentRange<VectorBounds>(ae,xe,high,t.k))return false;
        CreateVecIndex(ix,(int32_t)0,C);PipeBarrier<PIPE_V>();Muls(ix,ix,(int32_t)(t.k*4),C);PipeBarrier<PIPE_V>();
        for(uint32_t task=lane;task<t.k*R;task+=teams) {
            const uint32_t k=task/R,row=task%R*C,valid=row<t.n?(t.n-row<C?t.n-row:C):0;
            Duplicate(br,0.0f,C*t.k);Duplicate(low,0.0f,C);PipeBarrier<PIPE_ALL>();
            if(valid)DataCopyPad(br,gb[(uint64_t)(t.broadcastB?0:batch)*t.n*t.k+row*t.k],DataCopyExtParams{1,valid*t.k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();Gather(high,br,ix.template ReinterpretCast<uint32_t>(),k*4,C);PipeBarrier<PIPE_V>();
            const int32_t xq=(int32_t)xe.GetValue(k);
            for(uint32_t d=0;d<Weights;++d) {
                if(d<2)Duplicate(gs,0.0f,C);else Duplicate(integerSum,(int32_t)0,C);PipeBarrier<PIPE_V>();
                const uint32_t first=d>3?d-3:0,count=(d<3?d:3)-first+1;
                // Same-weight pairs are three matrix planes apart. Fetch them
                // together, after the previous vector reads of tmp complete.
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(tmp,pairs[((uint64_t)(d+3*first)*t.k+k)*N+row],DataCopyExtParams{(uint16_t)count,C*4,(3*t.k*N-C)*4,(N-C)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                if(d<2)for(uint32_t i=0;i<count;++i)A(gs,gs,tmp[i*N],C);
                else {
                    // A weight group may exceed 2^24. Sum in int32, then
                    // retain the float conversion's remainder as a low part.
                    for(uint32_t i=0;i<count;++i) {
                        Cast(rounded,tmp[i*N],RoundMode::CAST_RINT,C);PipeBarrier<PIPE_V>();Add(integerSum,integerSum,rounded,C);PipeBarrier<PIPE_V>();
                    }
                    Cast(gs,integerSum,RoundMode::CAST_NONE,C);PipeBarrier<PIPE_V>();Cast(rounded,gs,RoundMode::CAST_RINT,C);PipeBarrier<PIPE_V>();
                    Sub(remainder,integerSum,rounded,C);PipeBarrier<PIPE_V>();Cast(groupLow,remainder,RoundMode::CAST_NONE,C);PipeBarrier<PIPE_V>();
                }
                Adds(factor,ae[row],(float)(xq-8*(int32_t)d+127),C);PipeBarrier<PIPE_V>();
                Cast(factor.template ReinterpretCast<int32_t>(),factor,RoundMode::CAST_RINT,C);PipeBarrier<PIPE_V>();
                ShiftLeft(factor.template ReinterpretCast<int32_t>(),factor.template ReinterpretCast<int32_t>(),(int32_t)23,C);PipeBarrier<PIPE_V>();
                M(term,gs,factor,C);Scale(term,term,-1.0f,C);
                A(tmp,high,term,C);D(u,tmp,high,C);D(v,tmp,u,C);D(v,high,v,C);D(u,term,u,C);A(u,u,v,C);A(low,low,u,C);Scale(high,tmp,1.0f,C);
                if(d>=2) {M(term,groupLow,factor,C);Scale(term,term,-1.0f,C);A(low,low,term,C);}
            }
            A(high,high,low,C);PipeBarrier<PIPE_ALL>();DataCopy(residual[((uint64_t)batch*t.k+k)*N+row],high,C);PipeBarrier<PIPE_ALL>();
        }
        return true;
    }

};

} // namespace SolveDetail
} // namespace AscendC
