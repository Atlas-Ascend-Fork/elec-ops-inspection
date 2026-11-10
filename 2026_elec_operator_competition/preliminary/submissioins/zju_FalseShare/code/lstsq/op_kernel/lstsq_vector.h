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

#include <type_traits>
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "linalg_lstsq_tiling.h"
#ifndef LSTSQ_DIRECT_BENCH
#include "tiling_key_linalg_lstsq.h"
#endif
// Keep IEEE signed infinity for a zero divisor: the numerical guards can
// still reject a non-finite candidate instead of silently clamping its value.
__aicore__ inline float CheckedReciprocal(float value) {
    if (value != 0.0f) return 1.0f / value;
    union { float value; uint32_t bits; } result;
    result.value = value;
    result.bits = (result.bits & 0x80000000U) | 0x7f800000U;
    return result.value;
}

// A number is hi+lo. QR and back substitution both run on the device.
struct LstsqPair {float hi,lo;};
class Lstsq {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dataBuf,packBuf;
    AscendC::LocalTensor<float> qh,ql,ah,al,dh,dl,ph,pl,xh,xl,s[8],packed;
    AscendC::GlobalTensor<float> inputA,inputB,outputX,outputR,high,low,columnSolutions;
    LinalgLstsqTilingData t;
    uint32_t rhsRows;
    bool hasResiduals;
    template<AscendC::HardEvent event> __aicore__ inline void Sync() {
        const auto id=pipe.FetchEventID(event);AscendC::SetFlag<event>(id);AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void WriteRows(uint32_t batch) {
        Sync<AscendC::HardEvent::MTE3_S>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(columnSolutions);
        for(uint32_t row=0;row<t.n;++row) {
            for(uint32_t rhs=0;rhs<t.k;++rhs)ph.SetValue(rhs,columnSolutions.GetValue((uint64_t)rhs*t.n+row));
            Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(outputX[(uint64_t)batch*t.n*t.k+row*t.k],ph,AscendC::DataCopyExtParams{1,t.k*4,0,0,0});
            Sync<AscendC::HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline LstsqPair AddPair(LstsqPair a,LstsqPair b) {
        // Vector instructions preserve each required FP32 rounding step.
        AscendC::PipeBarrier<PIPE_ALL>();
        s[6].SetValue(16,a.hi);s[6].SetValue(24,a.lo);
        s[6].SetValue(32,b.hi);s[6].SetValue(40,b.lo);AscendC::PipeBarrier<PIPE_ALL>();
        AddVector(s[6],s[6][8],s[6][16],s[6][24],s[6][32],s[6][40],1);
        AscendC::PipeBarrier<PIPE_ALL>();return {s[6].GetValue(0),s[6].GetValue(8)};
    }
    static __aicore__ inline LstsqPair Neg(LstsqPair a) {return {-a.hi,-a.lo};}
    __aicore__ inline LstsqPair Product(float a,float b) {
        AscendC::PipeBarrier<PIPE_ALL>();
        s[6].SetValue(16,a);s[6].SetValue(24,0);
        s[6].SetValue(32,b);s[6].SetValue(40,0);AscendC::PipeBarrier<PIPE_ALL>();
        MulVector(s[6],s[6][8],s[6][16],s[6][24],s[6][32],s[6][40],1);
        AscendC::PipeBarrier<PIPE_ALL>();return {s[6].GetValue(0),s[6].GetValue(8)};
    }
    __aicore__ inline LstsqPair MulPair(LstsqPair a,LstsqPair b) {
        LstsqPair p=Product(a.hi,b.hi);
        return AddPair(p,{a.hi*b.lo+a.lo*b.hi,a.lo*b.lo});
    }
    __aicore__ inline LstsqPair DivPair(LstsqPair a,LstsqPair b) {
        float h=a.hi/b.hi;
        LstsqPair remainder=AddPair(a,Neg(MulPair(b,{h,0})));
        return AddPair({h,0},{(remainder.hi+remainder.lo)/b.hi,0});
    }
    __aicore__ inline LstsqPair Root(LstsqPair a) {
        s[7].SetValue(0,a.hi);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sqrt(s[7],s[7],1);AscendC::PipeBarrier<PIPE_ALL>();
        const float h=s[7].GetValue(0);
        if(h==0)return {0,0};
        const LstsqPair r=AddPair(a,Neg(Product(h,h)));
        return AddPair({h,0},{(r.hi+r.lo)/(2*h),0});
    }
    __aicore__ inline void VAdd(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b,uint32_t n) {AscendC::Add(o,a,b,n);AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void VSub(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b,uint32_t n) {AscendC::Sub(o,a,b,n);AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void VMul(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b,uint32_t n) {AscendC::Mul(o,a,b,n);AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void VScale(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> a,float b,uint32_t n) {AscendC::Muls(o,a,b,n);AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void Fill(AscendC::LocalTensor<float> o,float v,uint32_t n) {AscendC::Duplicate(o,v,n);AscendC::PipeBarrier<PIPE_ALL>();}
    // In-place output on a is safe; scratch never overlaps an input or output.
    __aicore__ inline void AddVector(AscendC::LocalTensor<float> oh,AscendC::LocalTensor<float> ol,
            AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> ae,AscendC::LocalTensor<float> b,AscendC::LocalTensor<float> be,uint32_t n) {
        VAdd(s[0],a,b,n);VSub(s[1],s[0],a,n);VSub(s[2],s[0],s[1],n);
        VSub(s[2],a,s[2],n);VSub(s[3],b,s[1],n);VAdd(s[2],s[2],s[3],n);
        VAdd(s[2],s[2],ae,n);VAdd(s[2],s[2],be,n);
        VAdd(oh,s[0],s[2],n);VSub(s[3],oh,s[0],n);VSub(s[4],oh,s[3],n);
        VSub(s[4],s[0],s[4],n);VSub(s[5],s[2],s[3],n);VAdd(ol,s[4],s[5],n);
    }
    __aicore__ inline void MulVector(AscendC::LocalTensor<float> oh,AscendC::LocalTensor<float> ol,
            AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> ae,AscendC::LocalTensor<float> b,AscendC::LocalTensor<float> be,uint32_t n) {
        VMul(oh,a,b,n);
        VScale(s[0],a,4097.0f,n);VSub(s[1],s[0],a,n);VSub(s[0],s[0],s[1],n);VSub(s[1],a,s[0],n);
        VScale(s[2],b,4097.0f,n);VSub(s[3],s[2],b,n);VSub(s[2],s[2],s[3],n);VSub(s[3],b,s[2],n);
        VMul(s[4],s[0],s[2],n);VSub(s[4],s[4],oh,n);
        VMul(s[5],s[0],s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],s[1],s[2],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],s[1],s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a,be,n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],ae,b,n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],ae,be,n);VAdd(ol,s[4],s[5],n);
    }
    __aicore__ inline LstsqPair Dot(AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> ae,
            AscendC::LocalTensor<float> b,AscendC::LocalTensor<float> be,uint32_t n) {
        MulVector(ph,pl,a,ae,b,be,n);
        while(n>8) {n/=2;AddVector(ph,pl,ph,pl,ph[n],pl[n],n);}
        AscendC::PipeBarrier<PIPE_ALL>();
        LstsqPair total{0,0};
        for(uint32_t i=0;i<n;++i)total=AddPair(total,{ph.GetValue(i),pl.GetValue(i)});
        return total;
    }
    __aicore__ inline void Load(uint32_t col,AscendC::LocalTensor<float> h,AscendC::LocalTensor<float> l) {
        AscendC::DataCopy(h,high[(uint64_t)col*t.rows],t.rows);AscendC::DataCopy(l,low[(uint64_t)col*t.rows],t.rows);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store(uint32_t col,AscendC::LocalTensor<float> h,AscendC::LocalTensor<float> l) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(high[(uint64_t)col*t.rows],h,t.rows);AscendC::DataCopy(low[(uint64_t)col*t.rows],l,t.rows);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void InputColumn(AscendC::GlobalTensor<float> input,uint64_t base,uint32_t width,uint32_t col) {
        Fill(ah,0,t.rows);Fill(al,0,t.rows);
        for(uint32_t row=0;row<t.m;++row)ah.SetValue(row,input.GetValue(base+(uint64_t)row*width+col));
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Transform(uint32_t col,LstsqPair tau) {
        Load(col,ah,al);const LstsqPair factor=MulPair(tau,Dot(qh,ql,ah,al,t.rows));
        Fill(ph,-factor.hi,t.rows);Fill(pl,-factor.lo,t.rows);
        MulVector(dh,dl,qh,ql,ph,pl,t.rows);
        AddVector(ph,pl,ah,al,dh,dl,t.rows);Store(col,ph,pl);
    }
    __aicore__ inline void Solve(uint32_t batch,uint32_t rhs,float scale) {
        Load(t.n+rhs,ah,al);
        if(hasResiduals) {
            for(uint32_t i=0;i<t.n;++i){ah.SetValue(i,0);al.SetValue(i,0);}
            AscendC::PipeBarrier<PIPE_ALL>();
            LstsqPair norm=Dot(ah,al,ah,al,t.rows);
            norm=MulPair(MulPair(norm,{CheckedReciprocal(scale),0}),{CheckedReciprocal(scale),0});
            s[7].SetValue(8,norm.hi+norm.lo);Sync<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(outputR[(uint64_t)batch*t.k+rhs],s[7][8],AscendC::DataCopyExtParams{1,4,0,0,0});
            Sync<AscendC::HardEvent::MTE3_S>();
        }
        Load(t.n+rhs,qh,ql);
        Fill(xh,0,rhsRows);Fill(xl,0,rhsRows);
        for(int32_t row=t.n-1;row>=0;--row) {
            AscendC::DataCopyPad(packed,high[row],AscendC::DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
            AscendC::PipeBarrier<PIPE_ALL>();Fill(ah,0,rhsRows);
            for(uint32_t j=0;j<t.n;++j)ah.SetValue(j,packed.GetValue(j*8));
            AscendC::DataCopyPad(packed,low[row],AscendC::DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
            AscendC::PipeBarrier<PIPE_ALL>();Fill(al,0,rhsRows);
            for(uint32_t j=0;j<t.n;++j)al.SetValue(j,packed.GetValue(j*8));
            AscendC::PipeBarrier<PIPE_ALL>();
            const LstsqPair diag{ah.GetValue(row),al.GetValue(row)};
            const LstsqPair value{qh.GetValue(row),ql.GetValue(row)};
            const LstsqPair result=DivPair(AddPair(value,Neg(Dot(ah,al,xh,xl,rhsRows))),diag);
            xh.SetValue(row,result.hi);xl.SetValue(row,result.lo);AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::Add(ph,xh,xl,t.n);AscendC::PipeBarrier<PIPE_V>();Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(columnSolutions[(uint64_t)rhs*t.n],ph,AscendC::DataCopyExtParams{1,t.n*4,0,0,0});
        Sync<AscendC::HardEvent::MTE3_S>();
    }
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR r,GM_ADDR workspace,const LinalgLstsqTilingData &data) {
        t=data;hasResiduals=r!=nullptr && t.residualElements>=(uint64_t)t.batches*t.k;rhsRows=8;while(rhsRows<t.n)rhsRows*=2;
        pipe.InitBuffer(dataBuf,18*1024*4);pipe.InitBuffer(packBuf,8*1024*4);
        auto all=dataBuf.Get<float>();
        qh=all;ql=all[1024];ah=all[2048];al=all[3072];dh=all[4096];dl=all[5120];ph=all[6144];pl=all[7168];
        xh=all[8192];xl=all[9216];for(uint32_t i=0;i<8;++i)s[i]=all[(10+i)*1024];
        packed=packBuf.Get<float>();
        inputA.SetGlobalBuffer((__gm__ float*)a);inputB.SetGlobalBuffer((__gm__ float*)b);
        outputX.SetGlobalBuffer((__gm__ float*)x);outputR.SetGlobalBuffer((__gm__ float*)r);
        const uint64_t words=(uint64_t)(t.n+t.k)*t.rows;
        high.SetGlobalBuffer((__gm__ float*)workspace+AscendC::GetBlockIdx()*2*words);
        low.SetGlobalBuffer((__gm__ float*)workspace+(AscendC::GetBlockIdx()*2+1)*words);
        columnSolutions.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.blocks*2*words+(uint64_t)AscendC::GetBlockIdx()*((t.n*t.k+7)/8*8));
        for(uint32_t batch=AscendC::GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            float largest=0;
            for(uint32_t col=0;col<t.n;++col) {
                InputColumn(inputA,(uint64_t)batch*t.m*t.n,t.n,col);
                for(uint32_t row=0;row<t.m;++row){float v=ah.GetValue(row);v=v<0?-v:v;if(v>largest)largest=v;}
                Store(col,ah,al);
            }
            union {float value;uint32_t bits;} exponent;exponent.value=largest;
            uint32_t power=254-((exponent.bits>>23)&255);if(power>253)power=253;if(power<1)power=1;
            exponent.bits=power<<23;const float scale=exponent.value;
            for(uint32_t col=0;col<t.n;++col){Load(col,ah,al);VScale(ah,ah,scale,t.rows);Store(col,ah,al);}
            for(uint32_t rhs=0;rhs<t.k;++rhs){InputColumn(inputB,(uint64_t)batch*t.m*t.k,t.k,rhs);VScale(ah,ah,scale,t.rows);Store(t.n+rhs,ah,al);}
            for(uint32_t j=0;j<t.n;++j) {
                Load(j,qh,ql);
                for(uint32_t i=0;i<j;++i){qh.SetValue(i,0);ql.SetValue(i,0);}
                AscendC::PipeBarrier<PIPE_ALL>();
                LstsqPair alpha=Root(Dot(qh,ql,qh,ql,t.rows));
                if(qh.GetValue(j)>=0)alpha=Neg(alpha);
                const LstsqPair v=AddPair({qh.GetValue(j),ql.GetValue(j)},Neg(alpha));
                qh.SetValue(j,v.hi);ql.SetValue(j,v.lo);AscendC::PipeBarrier<PIPE_ALL>();
                const LstsqPair tau=DivPair({2,0},Dot(qh,ql,qh,ql,t.rows));
                for(uint32_t col=j+1;col<t.n+t.k;++col)Transform(col,tau);
                Load(j,ah,al);ah.SetValue(j,alpha.hi);al.SetValue(j,alpha.lo);
                for(uint32_t i=j+1;i<t.rows;++i){ah.SetValue(i,0);al.SetValue(i,0);}
                Store(j,ah,al);
            }
            for(uint32_t rhs=0;rhs<t.k;++rhs)Solve(batch,rhs,scale);
            WriteRows(batch);
        }
    }
};

template<uint32_t P> struct VectorArithmetic {
    __aicore__ inline void A(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> u,AscendC::LocalTensor<float> v,uint32_t z) {AscendC::Add(o,u,v,z);AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void D(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> u,AscendC::LocalTensor<float> v,uint32_t z) {AscendC::Sub(o,u,v,z);AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void M(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> u,AscendC::LocalTensor<float> v,uint32_t z) {AscendC::Mul(o,u,v,z);AscendC::PipeBarrier<PIPE_V>();}
    __aicore__ inline void Scale(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> u,float v,uint32_t z) {AscendC::Muls(o,u,v,z);AscendC::PipeBarrier<PIPE_V>();}
    template<bool sub=false> __aicore__ inline void Rows(AscendC::LocalTensor<float> o,AscendC::LocalTensor<float> u,AscendC::LocalTensor<float> v,uint32_t rows,uint32_t width) {
        const AscendC::BinaryRepeatParams step{1,1,1,P/8,P/8,P/8};
        for(uint32_t off=0;off<width;off+=64) {
            uint64_t mask=width-off<64?width-off:64;
            if constexpr(sub)AscendC::Sub(o[off],u[off],v[off],mask,(uint8_t)rows,step);
            else AscendC::Add(o[off],u[off],v[off],mask,(uint8_t)rows,step);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
};

// Gauss-Jordan elimination updates all rows in one vector repeat group.
template<uint32_t P,uint32_t T=8> class AccurateRows : public VectorArithmetic<P> {
    using Ops=VectorArithmetic<P>;
    static constexpr uint32_t SZ=P*T,PAIR=4*T;
public:
    static constexpr uint32_t FLOATS=9*SZ+14*T;
    AscendC::LocalTensor<float> a,x,h,l,ah,al,xh,xl,s[4],raw,ahi,alo;
    AscendC::LocalTensor<uint32_t> ixBase;
    __aicore__ inline void Init(AscendC::LocalTensor<float> mem) {
        a=mem;x=mem[SZ];h=mem[2*SZ];l=mem[3*SZ];ah=mem[4*SZ];al=mem[5*SZ];xh=mem[6*SZ];xl=mem[7*SZ];
        for(uint32_t i=0;i<4;++i)s[i]=mem[(4+i)*SZ];raw=mem[8*SZ];ahi=mem[4*SZ];alo=mem[5*SZ];ixBase=mem[9*SZ].template ReinterpretCast<uint32_t>();
    }
    __aicore__ inline void Split(uint32_t rows) {
        Scale(ahi,raw,4097.0f,rows*P);D(alo,ahi,raw,rows*P);D(ahi,ahi,alo,rows*P);D(alo,raw,ahi,rows*P);
    }
    template<AscendC::HardEvent event> __aicore__ inline void Sync() {
        AscendC::SetFlag<event>(0);AscendC::WaitFlag<event>(0);
    }
    using Ops::A;using Ops::D;using Ops::M;using Ops::Scale;using Ops::Rows;
    __aicore__ inline void DensePairSum(uint32_t count) {
        A(s[0],a,x,count);D(s[1],s[0],a,count);D(s[2],s[0],s[1],count);
        D(s[2],a,s[2],count);D(s[3],x,s[1],count);A(s[2],s[2],s[3],count);
        A(s[2],s[2],a[PAIR],count);A(l,s[2],x[PAIR],count);
        Scale(h,s[0],1.0f,count);
    }
    __aicore__ inline void AlignedRowSum(AscendC::LocalTensor<float> out,AscendC::LocalTensor<float> input,uint32_t rows) {
        for(uint32_t half=P/2;half>=8;half/=2)Rows(input,input,input[half],rows,half);
        AscendC::WholeReduceSum(a,input,8,rows,1,1,P/8);AscendC::PipeBarrier<PIPE_V>();
        Scale(out,a,1.0f,rows);
    }
    __aicore__ inline void AccurateProduct(uint32_t count) {

        M(h,raw,x,count);
        Scale(xh,x,4097.0f,count);D(xl,xh,x,count);D(xh,xh,xl,count);D(xl,x,xh,count);
        M(l,ahi,xh,count);D(l,l,h,count);
        M(a,ahi,xl,count);A(l,l,a,count);
        M(a,alo,xh,count);A(l,l,a,count);
        M(a,alo,xl,count);A(l,l,a,count);

    }
    __aicore__ inline void LatticeRowSum(uint32_t rows,uint32_t exponent) {
            const uint32_t count=rows*P;
        constexpr uint32_t logP=P==512?9:(P==256?8:(P==128?7:(P==64?6:4)));
            union ScaleBits {uint32_t bits;float value;};
            ScaleBits q0,q1;q0.bits=((exponent+logP+1)<<23)|0x400000;
            q1.bits=((exponent+2*logP-22)<<23)|0x400000;
            AscendC::Adds(s[0],h,q0.value,count);AscendC::PipeBarrier<PIPE_V>();AscendC::Adds(s[0],s[0],-q0.value,count);AscendC::PipeBarrier<PIPE_V>();
            D(s[1],h,s[0],count);
            if constexpr(P==64) {
                // The coarse lattice keeps its row sum exact. Sum the remaining
                // product and correction together, then retain the final TwoSum error.
                A(s[1],s[1],l,count);
                AlignedRowSum(s[0],s[0],rows);AlignedRowSum(s[1],s[1],rows);
                A(h,s[0],s[1],rows);D(a,h,s[0],rows);D(x,h,a,rows);
                D(x,s[0],x,rows);D(a,s[1],a,rows);A(l,x,a,rows);
                return;
            }
            AscendC::Adds(s[2],s[1],q1.value,count);AscendC::PipeBarrier<PIPE_V>();AscendC::Adds(s[2],s[2],-q1.value,count);AscendC::PipeBarrier<PIPE_V>();
            D(s[3],s[1],s[2],count);
            AlignedRowSum(s[0],s[0],rows);AlignedRowSum(s[1],s[2],rows);
            AlignedRowSum(s[2],s[3],rows);AlignedRowSum(s[3],l,rows);
            A(a,s[2],s[3],rows);
            A(x,s[1],a,rows);D(h,x,s[1],rows);D(l,x,h,rows);D(l,s[1],l,rows);D(h,a,h,rows);A(l,l,h,rows);
            A(h,s[0],x,rows);D(s[1],h,s[0],rows);D(s[2],h,s[1],rows);D(s[2],s[0],s[2],rows);D(s[1],x,s[1],rows);A(a,s[2],s[1],rows);A(l,a,l,rows);
            return;

    }
    __aicore__ inline void PreparePairIndices() {
        for(uint32_t half=4,stride=P;half;stride=half,half/=2) {
            auto i0=ixBase[4*T*(4-half)],i1=i0[T*half];
            for(uint32_t r=0;r<T;++r)for(uint32_t j=0;j<half;++j){i0.SetValue(r*half+j,(r*stride+j)*4);i1.SetValue(r*half+j,(r*stride+half+j)*4);}
        }
        AscendC::PipeBarrier<PIPE_ALL>();

    }
    __aicore__ inline void ReducePairs(uint32_t rows) {
        for(uint32_t half=P/2;half>=8;half/=2) {
            Rows(s[0],h,h[half],rows,half);this->template Rows<true>(s[1],s[0],h,rows,half);
            this->template Rows<true>(s[2],s[0],s[1],rows,half);this->template Rows<true>(s[2],h,s[2],rows,half);
            this->template Rows<true>(s[3],h[half],s[1],rows,half);Rows(s[2],s[2],s[3],rows,half);
            Rows(s[2],s[2],l,rows,half);Rows(l,s[2],l[half],rows,half);
            for(uint32_t off=0;off<half;off+=64)
                AscendC::Muls(h[off],s[0][off],1.0f,(uint64_t)(half-off<64?half-off:64),(uint8_t)rows,AscendC::UnaryRepeatParams{1,1,P/8,P/8});
            AscendC::PipeBarrier<PIPE_V>();
        }
        for(uint32_t half=4;half;half/=2) {
            auto ix0=ixBase[4*T*(4-half)],ix1=ix0[T*half];
            AscendC::Gather(a,h,ix0,0,rows*half);AscendC::Gather(x,h,ix1,0,rows*half);
            AscendC::Gather(a[PAIR],l,ix0,0,rows*half);AscendC::Gather(x[PAIR],l,ix1,0,rows*half);
            AscendC::PipeBarrier<PIPE_V>();
            DensePairSum(rows*half);
        }

    }
    __aicore__ inline void AccurateDot(uint32_t rows) {
        Split(rows);const uint32_t count=rows*P;
        AccurateProduct(count);
        AscendC::Abs(a,h,count);AscendC::PipeBarrier<PIPE_V>();
        if constexpr(P==64) {
            AscendC::WholeReduceMax(s[0],a,64,rows,1,1,8,AscendC::ReduceOrder::ORDER_ONLY_VALUE);AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceMax(s[0][128],s[0],(uint64_t)rows,1,1,1,8);
        } else if constexpr(P>64)AscendC::WholeReduceMax(s[0],a,64,count/64,1,1,8);
        else {
            AscendC::SetMaskCount();AscendC::SetVectorMask<float,AscendC::MaskMode::COUNTER>(0,count);
            AscendC::WholeReduceMax<float,false>(s[0],a,AscendC::MASK_PLACEHOLDER,1,1,1,8);AscendC::SetMaskNorm();AscendC::ResetMask();
        }
        Sync<AscendC::HardEvent::V_S>();
        uint32_t maximumBits,maximumIndex;AscendC::GetReduceRepeatMaxMinSpr<float>(maximumBits,maximumIndex);
        const uint32_t exponent=(maximumBits>>23)&255;
        if(exponent>10 && exponent<240) {LatticeRowSum(rows,exponent);return;}
        PreparePairIndices();ReducePairs(rows);
    }
};
