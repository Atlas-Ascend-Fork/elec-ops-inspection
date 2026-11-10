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

#include "lstsq_narrow_vector.h"

namespace NarrowOrthogonal {
template<uint32_t N,bool Cube,class Op>
__aicore__ __attribute__((noinline)) void Inverse(Op &op,AscendC::GlobalTensor<float> dst,AscendC::GlobalTensor<float> src,uint32_t batches) {
    if constexpr(N==32) {
        if constexpr(!Cube)op.Invert32(dst,src);AscendC::SyncAll<false>();
    } else {
        constexpr uint32_t H=N/2,LO=H*128,DIAG=LO+H;
        if constexpr(!Cube)op.template NegateBlock<H>(src[LO]);
        Inverse<H,Cube>(op,dst,src,batches);
        if constexpr(Cube) {
            op.template LocalRun<H>(dst,src[H],src[H],batches,STRIDE);
            op.template LocalRun<H,true>(src[LO],src[H],src[DIAG],batches,STRIDE);
            op.template LocalRun<H>(src[LO],dst,src[LO],batches,STRIDE);
        }
        AscendC::SyncAll<false>();if constexpr(!Cube)op.template NegateBlock<H>(src[H]);
        Inverse<H,Cube>(op,dst[DIAG],src[DIAG],batches);
        if constexpr(Cube) {
            op.template LocalRun<H>(src[H],dst[DIAG],dst[H],batches,STRIDE);
            op.template LocalRun<H>(dst[DIAG],src[LO],dst[LO],batches,STRIDE);
            op.template LocalRun<H,true>(src[H],dst[LO],dst,batches,STRIDE);
        }
        AscendC::SyncAll<false>();
    }
}
template<bool Cube,uint32_t Panel> __aicore__ inline void Tail(std::conditional_t<Cube,NativeMM<Panel>,Vector<Panel>> &op,
    AscendC::GlobalTensor<float> mem,AscendC::GlobalTensor<float> A,AscendC::GlobalTensor<int8_t> ad,GM_ADDR workspace,
    uint32_t offset,uint32_t real,uint32_t n,uint32_t k,uint32_t batches) {
    const bool local=real>16 && real<=32;
    if(local) {
        if constexpr(Cube) {
            if constexpr(Panel==112) {
                if(real>24)op.template Run<32,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GI+1024],batches,STRIDE,STRIDE,STRIDE,256,256,32);
                else op.template Run<16,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GI+1024],batches,STRIDE,STRIDE,STRIDE,256,256,32);
            } else op.template Run<16,32,256,false,true,false>(mem[CT+offset*256],mem[CT+offset*256],mem[GI+1024],batches,STRIDE,STRIDE,STRIDE,256,256,32);
        }
        AscendC::SyncAll<false>();
    }
    if constexpr(Cube)op.PrepareFactor(mem[H],mem[D],batches,STRIDE);
    if constexpr(!Cube) {
        if(local) {
            if constexpr(Panel==112) {
                if(real>24)op.template LocalTail32<24,true>(mem[D+offset*256],mem[CT+offset*256],mem[GI+1024],real,(GM_ADDR)mem.GetPhyAddr());
                else op.template LocalTail32<16>(mem[D+offset*256],mem[CT+offset*256],mem[GI+1024],real);
            } else op.template LocalTail32<16>(mem[D+offset*256],mem[CT+offset*256],mem[GI+1024],real);
        }
        else op.TailQR(mem[D+offset*256],mem[CT+offset*256],real);
        if(k<=4)for(uint32_t panel=96;panel<160;panel+=32)op.SplitAChunk(ad,mem[EA],A,panel,n);
    }
    if constexpr(Panel==112)if(local && real>24) {
        AscendC::SyncAll<false>();
        if constexpr(Cube)op.ApplyPackedTail(mem,offset,batches);
    }

}
template<bool Cube,uint32_t Panel=96> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgLstsqTilingData &t,GM_ADDR rr) {
    // Uniform metadata guard: every Cube and Vector exits before any barrier.
    if (t.batches == 0) return;
    const uint32_t batches=t.batches,k=t.k,n=t.n;
    AscendC::GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    static_assert(Panel==96 || Panel==112);
    std::conditional_t<Cube,NativeMM<Panel>,Vector<Panel>> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*n*n];B=B[(uint64_t)batch*n*k];X=X[(uint64_t)batch*n*k];}
    AscendC::GlobalTensor<int8_t> ad,xd;AscendC::GlobalTensor<int32_t> products;
    ad.SetGlobalBuffer((__gm__ int8_t*)mem[AD].GetPhyAddr());xd.SetGlobalBuffer((__gm__ int8_t*)mem[XD].GetPhyAddr());products.SetGlobalBuffer((__gm__ int32_t*)mem[PRODUCTS].GetPhyAddr());
    AscendC::GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    if constexpr(!Cube){op.Normalize(mem[AN],mem[CT],A,B,k,n);op.PackCachedB(mem[BP],mem[DX],k);op.ZeroUnusedFactorRows(mem[D]);}
    AscendC::SyncAll<false>();
    for(uint32_t p=0;p<2*Panel;p+=Panel) {
        if constexpr(Cube)op.template GramPanel<256>(mem[CT],mem[G],batches,STRIDE,p);
        else op.PadGramPanel(mem[G]);
        AscendC::SyncAll<false>();
        AscendC::GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128);
        if constexpr(!Cube){op.ZeroDecision(decisions);op.NewtonPreparePanel(halfMem,mem[G]);}
        AscendC::SyncAll<false>();
        if constexpr(Cube) {
            if(p==0)op.template NewtonPanel<false>(halfMem,mem[GI],mem[NT],batches,STRIDE,Panel==96?8:9);
            else op.template NewtonPanel<true>(halfMem,mem[GI],mem[NT],batches,STRIDE,Panel==112?12:(n<=208?14:10));
            op.CrossGramFuture256(mem[CT],mem[CROSS],batches,STRIDE,p);
        } else {
            if(p==0)op.SplitAChunk(ad,mem[EA],A,0,n);
            else {op.SplitAChunk(ad,mem[EA],A,32,n);op.SplitAChunk(ad,mem[EA],A,64,n);}
        }
        AscendC::SyncAll<false>();
        if constexpr(!Cube)op.NewtonGuardPanel(mem[NT],decisions);
        if constexpr(Cube){op.FactorPanel(mem[GI],mem[CT],mem[D],batches,STRIDE,p);op.HFromCross256(mem[CROSS],mem[H],batches,STRIDE,p);}
        AscendC::SyncAll<false>();AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(decisions);
        if(decisions.GetValue(0)!=0) {
            Inverse<128,Cube>(op,mem[GI],mem[G],batches);
            if constexpr(!Cube)op.template NegateBlock<128>(mem[GI]);AscendC::SyncAll<false>();
            if constexpr(Cube){op.FactorPanel(mem[GI],mem[CT],mem[D],batches,STRIDE,p);op.HFromCross256(mem[CROSS],mem[H],batches,STRIDE,p);}AscendC::SyncAll<false>();
        }
        if constexpr(Cube)op.template UpdatePanel<256>(mem[H],mem[CT],batches,STRIDE,p);
        if constexpr(Cube)SyncCube();
    }
    // Host assigns at most 32 trailing rows to either active panel width.
    constexpr uint32_t p=2*Panel;
    const uint32_t real=n-p;
    if constexpr(Panel==112) {
        // D's final 32 rows are zero; a 256-row projection keeps all tiles regular.
        if constexpr(Cube)op.template Run<256,32,256,false,true>(mem[D],mem[CT+p*256],mem[CROSS],batches,STRIDE,STRIDE,STRIDE);
        AscendC::SyncAll<false>();
        if constexpr(!Cube)op.template ReprojectCoefficients<32>(mem[H],mem[CROSS],p);
        AscendC::SyncAll<false>();
        if constexpr(Cube)op.template Run<32,256,256,true,false,true>(mem[CROSS],mem[CT],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE);
    } else {
        if constexpr(Cube)op.template Run<192,32,256,false,true>(mem[D],mem[CT+p*256],mem[CROSS],batches,STRIDE,STRIDE,STRIDE);
        AscendC::SyncAll<false>();
        if constexpr(!Cube)op.template ReprojectCoefficients<32>(mem[H],mem[CROSS],p);
        AscendC::SyncAll<false>();
        if constexpr(Cube)op.template Run<32,256,192,true,false,true>(mem[CROSS],mem[CT],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE);
    }
    AscendC::SyncAll<false>();
    Tail<Cube,Panel>(op,mem,A,ad,workspace,p,real,n,k,batches);
    // Both local QR and Cube application publish directly to the final factor.
    AscendC::SyncAll<false>();
    if constexpr(Cube) {
        // Tail's existing rendezvous publishes both the prefix work and D2.
        op.CompleteFactor(mem[H],mem[D],batches,STRIDE);SyncCube();
        op.template Run<256,16,256,true>(mem[D],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);
    }
    AscendC::SyncAll<false>();
    if constexpr(!Cube)op.SplitXPacked4(xd,mem[EX],mem[XP],k);
    AscendC::SyncAll<false>();
    if constexpr(Cube)op.Int8Packed4(ad,xd,products,batches,STRIDE);
    AscendC::SyncAll<false>();
    if constexpr(!Cube)op.Assemble6Packed4(mem[DX],products,mem[EA],mem[EX],mem[BP]);
    AscendC::SyncAll<false>();
    if constexpr(Cube){op.template Run<256,16,256,true>(mem[D],mem[DX],mem[R],batches,STRIDE,STRIDE,STRIDE);SyncCube();}
    AscendC::SyncAll<false>();
    if constexpr(!Cube)op.Finish(X,mem[XP],mem[R],k,n,rr,rr && t.residualElements>=(uint64_t)batches*k);
}
}
