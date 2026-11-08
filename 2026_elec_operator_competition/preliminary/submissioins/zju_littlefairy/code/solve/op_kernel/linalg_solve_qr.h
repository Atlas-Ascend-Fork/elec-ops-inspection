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
class SolveKernel {
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuf,packBuf,panelBuf;
    LocalTensor<float> qh,ql,ah,al,dh,dl,ph,pl,q1,q2,xh,xl,s[8],packed;
    // Second lane for the two-column interleaved path. Disjoint buffers, same
    // arithmetic per column, issued alternately so the dependent chains of the
    // two columns can overlap instead of serialising on one instruction stream.
    LocalTensor<float> ah2,al2,dh2,dl2,ph2,pl2,s2[6];
    // Panel of kPanel consecutive columns, loaded and stored with one copy each.
    LocalTensor<float> panelH,panelL;
    GlobalTensor<float> inputA,inputB,outputX,high,low,columnSolutions;
    LinalgSolveTilingData t;
    uint32_t rhsRows;
    template<HardEvent event> __aicore__ inline void Sync() {
        const auto id=pipe.FetchEventID(event);SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void WriteRows(uint32_t batch) {
        Sync<HardEvent::MTE3_S>();
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(columnSolutions);
        for(uint32_t row=0;row<t.n;++row) {
            for(uint32_t rhs=0;rhs<t.k;++rhs)ph.SetValue(rhs,columnSolutions.GetValue((uint64_t)rhs*t.n+row));
            Sync<HardEvent::S_MTE3>();
            DataCopyPad(outputX[(uint64_t)batch*t.n*t.k+row*t.k],ph,DataCopyExtParams{1,t.k*4,0,0,0});
            Sync<HardEvent::MTE3_S>();
        }
    }
    __aicore__ inline SolvePair AddPair(SolvePair a,SolvePair b) {
        // Scalar pipe: same rounding sequence as the vector form on one element.
        return AddScalarPair(a,b);
    }
    // Same rounding sequence as AddVector(...,1), executed on the scalar pipe:
    // no vector instruction is issued, so the reduction tail costs no PIPE_ALL.
    static __aicore__ inline SolvePair AddScalarPair(SolvePair a,SolvePair b) {
        const float s0=a.hi+b.hi;
        const float s1=s0-a.hi;
        const float t0=s0-s1;
        const float e0=a.hi-t0;
        const float t1=b.hi-s1;
        float e=e0+t1;
        e=e+a.lo;
        e=e+b.lo;
        const float hi=s0+e;
        const float u0=hi-s0;
        const float u1=hi-u0;
        const float u2=s0-u1;
        const float u3=e-u0;
        return {hi,u2+u3};
    }
    static __aicore__ inline SolvePair Neg(SolvePair a) {return {-a.hi,-a.lo};}
    // Dekker split of a scalar. volatile keeps each rounding separate so the
    // compiler cannot contract the sequence into an FMA.
    static __aicore__ inline void SplitScalar(float c,float &c1,float &c2) {
        volatile float m=4097.0f*c; const float s=m;
        const float t=s-c;
        volatile float n=s-t; c1=n;
        c2=c-c1;
    }
    // Two-product via Dekker split on the scalar pipe. Every multiplication is
    // forced through volatile so the compiler cannot fuse it into an FMA: the
    // split depends on each product being rounded separately.
    static __aicore__ inline SolvePair Product(float a,float b) {
        volatile float m;
        m=4097.0f*a; const float c1=m;
        const float t1=c1-a;
        const float a1=c1-t1;
        const float a2=a-a1;
        m=4097.0f*b; const float c2=m;
        const float t2=c2-b;
        const float b1=c2-t2;
        const float b2=b-b1;
        m=a*b; const float p=m;
        m=a1*b1; const float p11=m;
        const float d1=p11-p;
        m=a1*b2; const float p12=m;
        const float s1=d1+p12;
        m=a2*b1; const float p21=m;
        const float s2=s1+p21;
        m=a2*b2; const float p22=m;
        const float e=s2+p22;
        return {p,e};
    }
    __aicore__ inline SolvePair MulPair(SolvePair a,SolvePair b) {
        SolvePair p=Product(a.hi,b.hi);
        return AddPair(p,{a.hi*b.lo+a.lo*b.hi,a.lo*b.lo});
    }
    __aicore__ inline SolvePair DivPair(SolvePair a,SolvePair b) {
        float h=a.hi/b.hi;
        SolvePair remainder=AddPair(a,Neg(MulPair(b,{h,0})));
        return AddPair({h,0},{(remainder.hi+remainder.lo)/b.hi,0});
    }
    __aicore__ inline SolvePair Root(SolvePair a) {
        s[7].SetValue(0,a.hi);PipeBarrier<PIPE_ALL>();
        Sqrt(s[7],s[7],1);PipeBarrier<PIPE_ALL>();
        const float h=s[7].GetValue(0);
        if(h==0)return {0,0};
        const SolvePair r=AddPair(a,Neg(Product(h,h)));
        return AddPair({h,0},{(r.hi+r.lo)/(2*h),0});
    }
    __aicore__ inline void VAdd(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Add(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void VSub(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Sub(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void VMul(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Mul(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void VScale(LocalTensor<float> o,LocalTensor<float> a,float b,uint32_t n) {Muls(o,a,b,n);PipeBarrier<PIPE_V>();}
    __aicore__ inline void Fill(LocalTensor<float> o,float v,uint32_t n) {Duplicate(o,v,n);PipeBarrier<PIPE_ALL>();}
    // In-place output on a is safe; scratch never overlaps an input or output.
    __aicore__ inline void AddVector(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> b,LocalTensor<float> be,uint32_t n) {
        // Knuth two-sum in 6 ops, then fold both low parts and renormalise once.
        VAdd(s[0],a,b,n);
        VSub(s[1],s[0],a,n);VSub(s[2],s[0],s[1],n);
        VSub(s[3],a,s[2],n);VSub(s[4],b,s[1],n);VAdd(s[3],s[3],s[4],n);
        VAdd(s[3],s[3],ae,n);VAdd(s[3],s[3],be,n);
        VAdd(oh,s[0],s[3],n);
        VSub(s[4],s[0],oh,n);VAdd(ol,s[4],s[3],n);
    }
    __aicore__ inline void SplitA(LocalTensor<float> a1,LocalTensor<float> a2,
            LocalTensor<float> a,uint32_t n) {
        VScale(a1,a,4097.0f,n);VSub(a2,a1,a,n);VSub(a1,a1,a2,n);VSub(a2,a,a1,n);
    }
    __aicore__ inline void MulVector(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> b,LocalTensor<float> be,uint32_t n) {
        VMul(oh,a,b,n);
        VScale(s[0],a,4097.0f,n);VSub(s[1],s[0],a,n);VSub(s[0],s[0],s[1],n);VSub(s[1],a,s[0],n);
        VScale(s[2],b,4097.0f,n);VSub(s[3],s[2],b,n);VSub(s[2],s[2],s[3],n);VSub(s[3],b,s[2],n);
        VMul(s[4],s[0],s[2],n);VSub(s[4],s[4],oh,n);
        VMul(s[5],s[0],s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],s[1],s[2],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],s[1],s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a,be,n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],ae,b,n);VAdd(ol,s[4],s[5],n);
    }
    // double-float vector times double-float scalar, with `a` split already known.
    // Term order matches MulVectorPre, so the result is bit-identical.
    __aicore__ inline void MulVectorScalar(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> a1,LocalTensor<float> a2,
            float c,float ce,uint32_t n) {
        float c1,c2;SplitScalar(c,c1,c2);
        VScale(oh,a,c,n);
        VScale(s[4],a1,c1,n);VSub(s[4],s[4],oh,n);
        VScale(s[5],a1,c2,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],a2,c1,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],a2,c2,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],a,ce,n);VAdd(s[4],s[4],s[5],n);
        VScale(s[5],ae,c,n);VAdd(ol,s[4],s[5],n);
    }
    __aicore__ inline void MulVectorPre(LocalTensor<float> oh,LocalTensor<float> ol,
            LocalTensor<float> a,LocalTensor<float> ae,LocalTensor<float> b,LocalTensor<float> be,
            LocalTensor<float> a1,LocalTensor<float> a2,uint32_t n) {
        VMul(oh,a,b,n);
        VScale(s[2],b,4097.0f,n);VSub(s[3],s[2],b,n);VSub(s[2],s[2],s[3],n);VSub(s[3],b,s[2],n);
        VMul(s[4],a1,s[2],n);VSub(s[4],s[4],oh,n);
        VMul(s[5],a1,s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a2,s[2],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a2,s[3],n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],a,be,n);VAdd(s[4],s[4],s[5],n);
        VMul(s[5],ae,b,n);VAdd(ol,s[4],s[5],n);
    }
    // ---- two-column interleaved path -------------------------------------
    // Identical arithmetic and identical per-column operation order, but the
    // statements of lane 0 and lane 1 are issued alternately. A PIPE_V barrier
    // inside a lane would put both chains back in series, so the NB variants
    // below omit it; ordering is preserved by construction (each lane only
    // touches its own buffers, qh/ql/q1/q2 are read-only here).
    __aicore__ inline void VAddNB(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Add(o,a,b,n);}
    __aicore__ inline void VSubNB(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Sub(o,a,b,n);}
    __aicore__ inline void VMulNB(LocalTensor<float> o,LocalTensor<float> a,LocalTensor<float> b,uint32_t n) {Mul(o,a,b,n);}
    __aicore__ inline void VScaleNB(LocalTensor<float> o,LocalTensor<float> a,float b,uint32_t n) {Muls(o,a,b,n);}
    // Same 11-op compensated add as AddVector, for both lanes, interleaved.
    __aicore__ inline void AddVec2(LocalTensor<float> o0,LocalTensor<float> l0,
            LocalTensor<float> a0,LocalTensor<float> ae0,LocalTensor<float> b0,LocalTensor<float> be0,
            LocalTensor<float> o1,LocalTensor<float> l1,
            LocalTensor<float> a1,LocalTensor<float> ae1,LocalTensor<float> b1,LocalTensor<float> be1,
            uint32_t n) {
        VAddNB(s[0],a0,b0,n);VAddNB(s2[0],a1,b1,n);
        VSubNB(s[1],s[0],a0,n);VSubNB(s2[1],s2[0],a1,n);
        VSubNB(s[2],s[0],s[1],n);VSubNB(s2[2],s2[0],s2[1],n);
        VSubNB(s[3],a0,s[2],n);VSubNB(s2[3],a1,s2[2],n);
        VSubNB(s[4],b0,s[1],n);VSubNB(s2[4],b1,s2[1],n);
        VAddNB(s[3],s[3],s[4],n);VAddNB(s2[3],s2[3],s2[4],n);
        VAddNB(s[3],s[3],ae0,n);VAddNB(s2[3],s2[3],ae1,n);
        VAddNB(s[3],s[3],be0,n);VAddNB(s2[3],s2[3],be1,n);
        VAddNB(o0,s[0],s[3],n);VAddNB(o1,s2[0],s2[3],n);
        VSubNB(s[4],s[0],o0,n);VSubNB(s2[4],s2[0],o1,n);
        VAddNB(l0,s[4],s[3],n);VAddNB(l1,s2[4],s2[3],n);
    }
    // Same 17-op double-double product as MulVectorPre, for both lanes.
    __aicore__ inline void MulVecPre2(LocalTensor<float> o0,LocalTensor<float> l0,
            LocalTensor<float> b0,LocalTensor<float> be0,
            LocalTensor<float> o1,LocalTensor<float> l1,
            LocalTensor<float> b1,LocalTensor<float> be1,uint32_t n) {
        VMulNB(o0,qh,b0,n);VMulNB(o1,qh,b1,n);
        VScaleNB(s[2],b0,4097.0f,n);VScaleNB(s2[2],b1,4097.0f,n);
        VSubNB(s[3],s[2],b0,n);VSubNB(s2[3],s2[2],b1,n);
        VSubNB(s[2],s[2],s[3],n);VSubNB(s2[2],s2[2],s2[3],n);
        VSubNB(s[3],b0,s[2],n);VSubNB(s2[3],b1,s2[2],n);
        VMulNB(s[4],q1,s[2],n);VMulNB(s2[4],q1,s2[2],n);
        VSubNB(s[4],s[4],o0,n);VSubNB(s2[4],s2[4],o1,n);
        VMulNB(s[5],q1,s[3],n);VMulNB(s2[5],q1,s2[3],n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],q2,s[2],n);VMulNB(s2[5],q2,s2[2],n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],q2,s[3],n);VMulNB(s2[5],q2,s2[3],n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],qh,be0,n);VMulNB(s2[5],qh,be1,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VMulNB(s[5],ql,b0,n);VMulNB(s2[5],ql,b1,n);
        VAddNB(l0,s[4],s[5],n);VAddNB(l1,s2[4],s2[5],n);
    }
    // Same 13-op scalar-multiplied product as MulVectorScalar, per-lane scalar.
    __aicore__ inline void MulVecScalar2(float c,float ce,float cL,float ceL,uint32_t n) {
        float c1,c2,c1L,c2L;
        SplitScalar(c,c1,c2);SplitScalar(cL,c1L,c2L);
        VScaleNB(dh,qh,c,n);VScaleNB(dh2,qh,cL,n);
        VScaleNB(s[4],q1,c1,n);VScaleNB(s2[4],q1,c1L,n);
        VSubNB(s[4],s[4],dh,n);VSubNB(s2[4],s2[4],dh2,n);
        VScaleNB(s[5],q1,c2,n);VScaleNB(s2[5],q1,c2L,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],q2,c1,n);VScaleNB(s2[5],q2,c1L,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],q2,c2,n);VScaleNB(s2[5],q2,c2L,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],qh,ce,n);VScaleNB(s2[5],qh,ceL,n);
        VAddNB(s[4],s[4],s[5],n);VAddNB(s2[4],s2[4],s2[5],n);
        VScaleNB(s[5],ql,c,n);VScaleNB(s2[5],ql,cL,n);
        VAddNB(dl,s[4],s[5],n);VAddNB(dl2,s2[4],s2[5],n);
    }
    // Balanced tail shared by the sequential and interleaved paths.
    static __aicore__ inline SolvePair Tail8(LocalTensor<float> h,LocalTensor<float> l,uint32_t n) {
        if(n!=8) {
            SolvePair total{0,0};
            for(uint32_t i=0;i<n;++i)total=AddScalarPair(total,{h.GetValue(i),l.GetValue(i)});
            return total;
        }
        const SolvePair p0{h.GetValue(0),l.GetValue(0)};
        const SolvePair p1{h.GetValue(1),l.GetValue(1)};
        const SolvePair p2{h.GetValue(2),l.GetValue(2)};
        const SolvePair p3{h.GetValue(3),l.GetValue(3)};
        const SolvePair p4{h.GetValue(4),l.GetValue(4)};
        const SolvePair p5{h.GetValue(5),l.GetValue(5)};
        const SolvePair p6{h.GetValue(6),l.GetValue(6)};
        const SolvePair p7{h.GetValue(7),l.GetValue(7)};
        const SolvePair s0=AddScalarPair(p0,p1);
        const SolvePair s1=AddScalarPair(p2,p3);
        const SolvePair s2v=AddScalarPair(p4,p5);
        const SolvePair s3v=AddScalarPair(p6,p7);
        return AddScalarPair(AddScalarPair(s0,s1),AddScalarPair(s2v,s3v));
    }
    __aicore__ inline void DotPre2(SolvePair &r0,SolvePair &r1) {
        uint32_t n=t.rows;
        MulVecPre2(ph,pl,ah,al,ph2,pl2,ah2,al2,n);
        // Tree stops at 8 elements: smaller offsets would violate the 32-byte
        // alignment that UB vector access requires (n=4 -> 16 B, n=2 -> 8 B).
        while(n>8) {n/=2;AddVec2(ph,pl,ph,pl,ph[n],pl[n],ph2,pl2,ph2,pl2,ph2[n],pl2[n],n);}
        PipeBarrier<PIPE_ALL>();
        r0=Tail8(ph,pl,n);r1=Tail8(ph2,pl2,n);
    }
    __aicore__ inline void Transform2(uint32_t col0,uint32_t col1,SolvePair tau) {
        DataCopy(ah,high[(uint64_t)col0*t.rows],t.rows);DataCopy(al,low[(uint64_t)col0*t.rows],t.rows);
        DataCopy(ah2,high[(uint64_t)col1*t.rows],t.rows);DataCopy(al2,low[(uint64_t)col1*t.rows],t.rows);
        PipeBarrier<PIPE_ALL>();
        SolvePair d0,d1;DotPre2(d0,d1);
        const SolvePair f0=MulPair(tau,d0);
        const SolvePair f1=MulPair(tau,d1);
        MulVecScalar2(-f0.hi,-f0.lo,-f1.hi,-f1.lo,t.rows);
        AddVec2(ph,pl,ah,al,dh,dl,ph2,pl2,ah2,al2,dh2,dl2,t.rows);
        PipeBarrier<PIPE_ALL>();
        DataCopy(high[(uint64_t)col0*t.rows],ph,t.rows);DataCopy(low[(uint64_t)col0*t.rows],pl,t.rows);
        DataCopy(high[(uint64_t)col1*t.rows],ph2,t.rows);DataCopy(low[(uint64_t)col1*t.rows],pl2,t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    // kPanel consecutive columns per copy pair. The workspace stores columns
    // back to back, so the panel is contiguous; only the number of DataCopy and
    // barrier instances changes, never the per-column arithmetic.
    __aicore__ inline void TransformPanel(uint32_t start,SolvePair tau) {
        DataCopy(panelH,high[(uint64_t)start*t.rows],kPanel*t.rows);
        DataCopy(panelL,low[(uint64_t)start*t.rows],kPanel*t.rows);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t p=0;p<kPanel;++p) {
            LocalTensor<float> h=panelH[p*t.rows],l=panelL[p*t.rows];
            const SolvePair d=DotPre(qh,ql,h,l,q1,q2,t.rows);
            const SolvePair f=MulPair(tau,d);
            MulVectorScalar(dh,dl,qh,ql,q1,q2,-f.hi,-f.lo,t.rows);
            AddVector(h,l,h,l,dh,dl,t.rows);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(high[(uint64_t)start*t.rows],panelH,kPanel*t.rows);
        DataCopy(low[(uint64_t)start*t.rows],panelL,kPanel*t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline SolvePair Dot(LocalTensor<float> a,LocalTensor<float> ae,
            LocalTensor<float> b,LocalTensor<float> be,uint32_t n) {
        MulVector(ph,pl,a,ae,b,be,n);
        // Tree stops at 8 elements: smaller offsets would violate the 32-byte
        // alignment that UB vector access requires (n=4 -> 16 B, n=2 -> 8 B).
        static_assert(kRowsPerRegion >= kMaxPaddedRows, "region cannot hold a padded row");
        while(n>8) {n/=2;AddVector(ph,pl,ph,pl,ph[n],pl[n],n);}
        // Measured: one PIPE_ALL here beats V_S + S_V by ~0.6% on large cases, so
        // the full drain stays (see interleaved-lanes record).
        PipeBarrier<PIPE_ALL>();
        return Tail8(ph,pl,n);
    }
    __aicore__ inline SolvePair DotPre(LocalTensor<float> a,LocalTensor<float> ae,
            LocalTensor<float> b,LocalTensor<float> be,
            LocalTensor<float> a1,LocalTensor<float> a2,uint32_t n) {
        MulVectorPre(ph,pl,a,ae,b,be,a1,a2,n);
        // Tree stops at 8 elements: smaller offsets would violate the 32-byte
        // alignment that UB vector access requires (n=4 -> 16 B, n=2 -> 8 B).
        static_assert(kRowsPerRegion >= kMaxPaddedRows, "region cannot hold a padded row");
        while(n>8) {n/=2;AddVector(ph,pl,ph,pl,ph[n],pl[n],n);}
        PipeBarrier<PIPE_ALL>();
        return Tail8(ph,pl,n);
    }
    __aicore__ inline void Load(uint32_t col,LocalTensor<float> h,LocalTensor<float> l) {
        DataCopy(h,high[(uint64_t)col*t.rows],t.rows);DataCopy(l,low[(uint64_t)col*t.rows],t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store(uint32_t col,LocalTensor<float> h,LocalTensor<float> l) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(high[(uint64_t)col*t.rows],h,t.rows);DataCopy(low[(uint64_t)col*t.rows],l,t.rows);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void InputColumn(GlobalTensor<float> input,uint64_t base,uint32_t width,uint32_t col) {
        Fill(ah,0,t.rows);Fill(al,0,t.rows);
        // Measured crossover: at n=256 one strided copy beats n global scalar reads
        // by ~7%, while at n<=200 the extra barrier costs more than it saves. The
        // threshold sits above the largest measured loser, so it can only help.
        if(t.n>=kInputDmaThreshold) {
            DataCopyPad(packed,input[base+col],DataCopyExtParams{(uint16_t)t.n,4,(width-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=0;row<t.n;++row)ah.SetValue(row,packed.GetValue(row*8));
        } else {
            for(uint32_t row=0;row<t.n;++row)ah.SetValue(row,input.GetValue(base+(uint64_t)row*width+col));
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Transform(uint32_t col,SolvePair tau) {
        Load(col,ah,al);const SolvePair factor=MulPair(tau,DotPre(qh,ql,ah,al,q1,q2,t.rows));
        MulVectorScalar(dh,dl,qh,ql,q1,q2,-factor.hi,-factor.lo,t.rows);
        AddVector(ph,pl,ah,al,dh,dl,t.rows);Store(col,ph,pl);
    }
    __aicore__ inline void Solve(uint32_t rhs) {
        Load(t.n+rhs,ah,al);
        Load(t.n+rhs,qh,ql);
        Fill(xh,0,rhsRows);Fill(xl,0,rhsRows);
        // Rows only replace [0,n); the complement never changes, so it is zeroed
        // once per right-hand side instead of on every row.
        Fill(ah,0,rhsRows);Fill(al,0,rhsRows);
        for(int32_t row=t.n-1;row>=0;--row) {
            DataCopyPad(packed,high[row],DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<t.n;++j)ah.SetValue(j,packed.GetValue(j*8));
            DataCopyPad(packed,low[row],DataCopyExtParams{(uint16_t)t.n,4,(t.rows-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<t.n;++j)al.SetValue(j,packed.GetValue(j*8));
            PipeBarrier<PIPE_ALL>();
            const SolvePair diag{ah.GetValue(row),al.GetValue(row)};
            const SolvePair value{qh.GetValue(row),ql.GetValue(row)};
            const SolvePair result=DivPair(AddPair(value,Neg(Dot(ah,al,xh,xl,rhsRows))),diag);
            xh.SetValue(row,result.hi);xl.SetValue(row,result.lo);PipeBarrier<PIPE_ALL>();
        }
        Add(ph,xh,xl,t.n);PipeBarrier<PIPE_V>();Sync<HardEvent::V_MTE3>();
        DataCopyPad(columnSolutions[(uint64_t)rhs*t.n],ph,DataCopyExtParams{1,t.n*4,0,0,0});
        Sync<HardEvent::MTE3_S>();
    }
public:
    __aicore__ inline void Initialize(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR workspace,const LinalgSolveTilingData &data) {
        t=data;rhsRows=8;while(rhsRows<t.n)rhsRows*=2;
        pipe.InitBuffer(dataBuf,kScratchRegions*kRowsPerRegion*sizeof(float));pipe.InitBuffer(packBuf,8*1024*4);
        pipe.InitBuffer(panelBuf,2*kPanel*kMaxPaddedRows*sizeof(float));
        auto all=dataBuf.Get<float>();
        // Offsets follow the region map asserted at the top of this file.
        qh=all[0];ql=all[1*kRowsPerRegion];ah=all[2*kRowsPerRegion];al=all[3*kRowsPerRegion];
        dh=all[4*kRowsPerRegion];dl=all[5*kRowsPerRegion];ph=all[6*kRowsPerRegion];pl=all[7*kRowsPerRegion];
        ah2=all[8*kRowsPerRegion];al2=all[9*kRowsPerRegion];
        dh2=all[10*kRowsPerRegion];dl2=all[11*kRowsPerRegion];
        ph2=all[12*kRowsPerRegion];pl2=all[13*kRowsPerRegion];
        xh=all[14*kRowsPerRegion];xl=all[15*kRowsPerRegion];
        for(uint32_t i=0;i<8;++i)s[i]=all[(16+i)*kRowsPerRegion];
        for(uint32_t i=0;i<6;++i)s2[i]=all[(24+i)*kRowsPerRegion];
        q1=all[30*kRowsPerRegion];q2=all[31*kRowsPerRegion];
        packed=packBuf.Get<float>();
        auto panels=panelBuf.Get<float>();
        panelH=panels;panelL=panels[kPanel*kMaxPaddedRows];
        inputA.SetGlobalBuffer((__gm__ float*)a);inputB.SetGlobalBuffer((__gm__ float*)b);
        outputX.SetGlobalBuffer((__gm__ float*)x);
        const uint64_t words=(uint64_t)(t.n+t.k)*t.rows;
        high.SetGlobalBuffer((__gm__ float*)workspace+GetBlockIdx()*2*words);
        low.SetGlobalBuffer((__gm__ float*)workspace+(GetBlockIdx()*2+1)*words);
        columnSolutions.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)t.blocks*2*words+(uint64_t)GetBlockIdx()*((t.n*t.k+7)/8*8));
    }
    __aicore__ inline void LoadScaledBatch(uint32_t batch) {
            float largest=0;
            for(uint32_t col=0;col<t.n;++col) {
                InputColumn(inputA,(uint64_t)batch*t.n*t.n,t.n,col);
                for(uint32_t row=0;row<t.n;++row){float v=ah.GetValue(row);v=v<0?-v:v;if(v>largest)largest=v;}
                Store(col,ah,al);
            }
            union {float value;uint32_t bits;} exponent;exponent.value=largest;
            uint32_t power=254-((exponent.bits>>23)&255);if(power>253)power=253;if(power<1)power=1;
            exponent.bits=power<<23;const float scale=exponent.value;
            for(uint32_t col=0;col<t.n;++col){Load(col,ah,al);VScale(ah,ah,scale,t.rows);Store(col,ah,al);}
            for(uint32_t rhs=0;rhs<t.k;++rhs){InputColumn(inputB,(uint64_t)(t.broadcastB?0:batch)*t.n*t.k,t.k,rhs);VScale(ah,ah,scale,t.rows);Store(t.n+rhs,ah,al);}
    }
    __aicore__ inline void FactorColumn(uint32_t j) {
                Load(j,qh,ql);
                for(uint32_t i=0;i<j;++i){qh.SetValue(i,0);ql.SetValue(i,0);}
                PipeBarrier<PIPE_ALL>();
                SolvePair alpha=Root(Dot(qh,ql,qh,ql,t.rows));
                if(qh.GetValue(j)>=0)alpha=Neg(alpha);
                const SolvePair v=AddPair({qh.GetValue(j),ql.GetValue(j)},Neg(alpha));
                qh.SetValue(j,v.hi);ql.SetValue(j,v.lo);PipeBarrier<PIPE_ALL>();
                SplitA(q1,q2,qh,t.rows);
                const SolvePair normal=DotPre(qh,ql,qh,ql,q1,q2,t.rows);
                const SolvePair tau=DivPair({2,0},normal);
                // Widest batching first, then pairs, then a single column. No column
                // is skipped because each step advances by exactly the batch width.
                uint32_t col=j+1,end=t.n+t.k;
                for(;col+kPanel<=end;col+=kPanel)TransformPanel(col,tau);
                for(;col+1<end;col+=2)Transform2(col,col+1,tau);
                if(col<end)Transform(col,tau);
                Load(j,ah,al);ah.SetValue(j,alpha.hi);al.SetValue(j,alpha.lo);
                for(uint32_t i=j+1;i<t.rows;++i){ah.SetValue(i,0);al.SetValue(i,0);}
                Store(j,ah,al);
    }
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR workspace,const LinalgSolveTilingData &data) {
        Initialize(a,b,x,workspace,data);
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            LoadScaledBatch(batch);
            for(uint32_t j=0;j<t.n;++j)FactorColumn(j);
            for(uint32_t rhs=0;rhs<t.k;++rhs)Solve(rhs);
            WriteRows(batch);
        }
    }

};

} // namespace SolveDetail
} // namespace AscendC
