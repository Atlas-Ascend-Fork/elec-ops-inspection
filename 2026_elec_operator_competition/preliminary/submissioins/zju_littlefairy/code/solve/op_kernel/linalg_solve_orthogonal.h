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
#include "linalg_solve_orthogonal64.h"
#include "linalg_solve_narrow_vector.h"

namespace AscendC {
namespace SolveDetail {
namespace NarrowOrthogonal {

template<uint32_t N,bool Cube,class Op>
__aicore__ __attribute__((noinline)) void Inverse(Op &op,GlobalTensor<float> dst,GlobalTensor<float> src,uint32_t batches) {
    if constexpr(N==32) {
        if constexpr(!Cube)op.Invert32(dst,src);SyncAll<false>();
    } else {
        constexpr uint32_t H=N/2,LO=H*128,DIAG=LO+H;
        if constexpr(!Cube)op.template NegateBlock<H>(src[LO]);
        Inverse<H,Cube>(op,dst,src,batches);
        if constexpr(Cube) {
            op.template LocalRun<H>(dst,src[H],src[H],batches,STRIDE);
            op.template LocalRun<H,true>(src[LO],src[H],src[DIAG],batches,STRIDE);
            op.template LocalRun<H>(src[LO],dst,src[LO],batches,STRIDE);
        }
        SyncAll<false>();if constexpr(!Cube)op.template NegateBlock<H>(src[H]);
        Inverse<H,Cube>(op,dst[DIAG],src[DIAG],batches);
        if constexpr(Cube) {
            op.template LocalRun<H>(src[H],dst[DIAG],dst[H],batches,STRIDE);
            op.template LocalRun<H>(dst[DIAG],src[LO],dst[LO],batches,STRIDE);
            op.template LocalRun<H,true>(src[H],dst[LO],dst,batches,STRIDE);
        }
        SyncAll<false>();
    }
}
// Keep additional accuracy work outside the normal Newton path.
template<bool Cube,class Op>
__aicore__ __attribute__((noinline)) void InverseFallback96(Op &op,GlobalTensor<float> mem,uint32_t panel,uint32_t batches) {
    auto inverse=mem[GI],gram=mem[panel==96?XD:G],saved=mem[NT];
    if constexpr(!Cube)op.SaveFallbackGram96(saved,gram);
    SyncAll<false>();
    Inverse<128,Cube>(op,inverse,gram,batches);
    if constexpr(!Cube)op.template NegateBlock<128>(inverse);
    SyncAll<false>();
    // Factor consumes inverse*Gram; refine this side to avoid amplifying its defect.
    if constexpr(Cube)op.template LocalRun<96>(inverse,saved,gram,batches,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.FallbackDefect96(gram);
    SyncAll<false>();
    if constexpr(Cube){op.template LocalRun<96,true>(gram,inverse,inverse,batches,STRIDE);SyncCube();}
}
// Every Cube participates in the global producer/consumer join.
// Mode2 exchanges only the data-ready notification with its paired two AIVs.
template<bool Cube> __aicore__ inline void CubeToVector() {
    if constexpr(Cube) {
        SyncCube();
        ffts_cross_core_sync(PIPE_FIX,GetffstMsg(0x2,1));
    } else {
        PipeBarrier<PIPE_ALL>();wait_flag_dev(1);
    }
}
template<bool Cube> __aicore__ inline void VectorToCube() {
    if constexpr(Cube) {
        wait_flag_dev(0);SyncCube();
    } else {
        PipeBarrier<PIPE_ALL>();
        ffts_cross_core_sync(PIPE_MTE3,GetffstMsg(0x2,0));
    }
}
template<bool Cube,bool Packed,class Op> __aicore__ inline void FactorNarrowTail(Op &op,GlobalTensor<float> mem,GlobalTensor<half> halfMem,GlobalTensor<int8_t> ad,GlobalTensor<float> A,GM_ADDR workspace,uint32_t batches,uint32_t n,uint32_t p) {
            // G is dead after both local inverses; retain E here through both RHS applications.
            if constexpr(Cube){if constexpr(Packed)op.TailCoefficients224(mem[D],mem[CT],mem[G],batches,STRIDE);else op.template Run<192,32,256,false,true>(mem[D],mem[CT+p*256],mem[R],batches,STRIDE,STRIDE,STRIDE);}
            if constexpr(Packed) {
                if constexpr(Cube)SyncCube();
            } else {
                SyncAll<false>();
                if constexpr(!Cube)op.ReprojectCoefficients(mem[H],mem[R],p);
                SyncAll<false>();
            }
            if constexpr(Cube){if constexpr(Packed)op.ReprojectTail224(mem[G],mem[CT],batches,STRIDE);else op.template Run<32,256,192,true,false,true>(mem[R],mem[CT],mem[CT+p*256],batches,STRIDE,STRIDE,STRIDE);}
            if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
            if constexpr(!Cube){op.TailQR(mem[D+p*256],mem[CT+p*256],n-p);op.SplitAChunk(ad,mem[EA],A,96,n);op.SplitAChunk(ad,mem[EA],A,128,n);}
            if constexpr(Cube && Packed)op.ComposeForward96(mem[H],mem[G],mem[FORWARD_IDENTITY],mem[FORWARD_MAP],batches,STRIDE);
            // Forward uses H/F/BP only; it can overlap TailQR's stores to D.
            if constexpr(Packed){if constexpr(Cube)SyncCube();}else SyncAll<false>();
}
template<bool Cube,bool Packed,class Op> __aicore__ inline void FallbackNarrowPanel(Op &op,GlobalTensor<float> mem,GlobalTensor<float> gram,uint32_t batches,uint32_t p) {
                if constexpr(Packed) {
                    if(p==96) {
                        // The fallback retains the original explicit-C1 Gram target.
                        if constexpr(Cube)op.template Gram96<208>(mem[CT],gram,batches,STRIDE,p);
                        SyncAll<false>();
                    }
                }
                if constexpr(Packed)InverseFallback96<Cube>(op,mem,p,batches);
                else {
                    // Preserve the Packed call graph; both wide-RHS panels use G.
                    if constexpr(!Cube)op.SaveFallbackGram96(mem[NT],gram);
                    SyncAll<false>();
                    Inverse<128,Cube>(op,mem[GI],gram,batches);
                    if constexpr(!Cube)op.template NegateBlock<128>(mem[GI]);SyncAll<false>();
                    if constexpr(Cube)op.template LocalRun<96>(mem[GI],mem[NT],gram,batches,STRIDE);
                    SyncAll<false>();
                    if constexpr(!Cube)op.FallbackDefect96(gram);
                    SyncAll<false>();
                    if constexpr(Cube){op.template LocalRun<96,true>(gram,mem[GI],mem[GI],batches,STRIDE);SyncCube();}
                }
                if constexpr(Cube){op.Factor96(mem[GI],mem[CT],mem[D],batches,STRIDE,p);if constexpr(Packed)op.HFromCross208(mem[GI],mem[AN],mem[H],batches,STRIDE,p);}SyncAll<false>();
}
template<bool Cube,bool Packed,class Op> __aicore__ inline void FactorNarrowPanel(Op &op,GlobalTensor<float> mem,GlobalTensor<half> halfMem,GlobalTensor<int8_t> ad,GlobalTensor<float> A,GM_ADDR workspace,uint32_t batches,uint32_t n,uint32_t p) {
            // XD is unused until initial X is split; a padded128 Gram fits here.
            auto gram=mem[(Packed && p==96)?XD:G];
            if constexpr(Cube) {
                if constexpr(Packed) {
                    if(p==96)op.SchurSecondGram96(mem[AN],mem[H],gram,batches,STRIDE);
                    else op.template Gram96<208>(mem[CT],gram,batches,STRIDE,p);
                } else op.template Gram96<256>(mem[CT],gram,batches,STRIDE,p);
            } else op.PadGram96(gram);
            SyncAll<false>();
            GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128);
            if constexpr(!Cube){op.ZeroDecision(decisions);op.NewtonPrepare96(halfMem,gram);}
            if constexpr(Cube && Packed) {
                if(p==96)op.template Update96<208>(mem[H],mem[CT],batches,STRIDE,0);
            }
            if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
            if constexpr(Cube) {
                if(p==0)op.template Newton96<false>(halfMem,mem[GI],mem[NT],batches,STRIDE,8);
                else {
                    if constexpr(Packed) {
                        op.SecondNewtonRefreshed96(mem,batches);
                    } else op.template Newton96<true>(halfMem,mem[GI],mem[NT],batches,STRIDE,14);
                }
                if constexpr(Packed) {
                    if(p==0)op.CrossAndSecondGram208(mem[CT],mem[AN],mem[XD],batches,STRIDE);

                }
            } else {
                if(p==0)op.SplitAChunk(ad,mem[EA],A,0,n);
                else {op.SplitAChunk(ad,mem[EA],A,32,n);op.SplitAChunk(ad,mem[EA],A,64,n);}
            }
            if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
            if constexpr(!Cube)op.NewtonGuard96(mem[NT],decisions);
            if constexpr(Cube){op.Factor96(mem[GI],mem[CT],mem[D],batches,STRIDE,p);if constexpr(Packed)op.HFromCross208(mem[GI],mem[AN],mem[H],batches,STRIDE,p);}
            SyncAll<false>();DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decisions);
            if(decisions.GetValue(0)!=0)FallbackNarrowPanel<Cube,Packed>(op,mem,gram,batches,p);
}
template<bool Cube,bool Packed,class Op> __aicore__ inline void UpdateNarrowTrailing(Op &op,GlobalTensor<float> mem,GlobalTensor<half> halfMem,GlobalTensor<int8_t> ad,GlobalTensor<float> A,GM_ADDR workspace,uint32_t batches,uint32_t n,uint32_t p) {
        if(p<192) {
            if constexpr(Cube && !Packed)op.template TrailingH96<256>(mem[D],mem[CT],mem[H],batches,STRIDE,p);
            if constexpr(Cube && !Packed)SyncCube();
            if constexpr(Cube) {
                if constexpr(Packed) {
                    if(p!=0){op.template Update96<208>(mem[H],mem[CT],batches,STRIDE,p);SyncCube();}
                } else {op.template Update96<256>(mem[H],mem[CT],batches,STRIDE,p);SyncCube();}
            }
        }
}
template<bool Cube,bool Packed,class Op> __aicore__ inline void InitialNarrowSolution(Op &op,GlobalTensor<float> mem,GlobalTensor<float> X,GlobalTensor<int8_t> ad,GlobalTensor<int8_t> xd,GlobalTensor<int32_t> products,uint32_t batches,uint32_t k,uint32_t n) {
    if constexpr(Cube){if constexpr(Packed)op.Forward96Parallel(mem[H],mem[FORWARD_MAP],mem[BP],mem[DX],batches,STRIDE);else op.Forward96(mem[H],mem[DX],batches,STRIDE);}
    if constexpr(Cube)SyncCube();
    if constexpr(Packed)VectorToCube<Cube>();
    if constexpr(Cube){if constexpr(Packed)op.ApplyFactor208(mem[D],mem[DX],mem[XP],batches,STRIDE);else op.template Run<256,32,256,true>(mem[D],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);}
    if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
    if constexpr(!Cube){if constexpr(Packed)op.SplitXPacked4(xd,mem[EX],mem[XP],k);else op.SplitX(xd,mem[EX],mem[XP],k);}
    if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
}
template<bool Cube,bool Packed,class Op> __aicore__ inline void RefineNarrowSolution(Op &op,GlobalTensor<float> mem,GlobalTensor<float> X,GlobalTensor<int8_t> ad,GlobalTensor<int8_t> xd,GlobalTensor<int32_t> products,uint32_t batches,uint32_t k,uint32_t n) {
    if constexpr(Cube){if constexpr(Packed)op.Int8Packed4(ad,xd,products,batches,STRIDE);else op.Int8All(ad,xd,products,batches,STRIDE);}
    if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
    if constexpr(!Cube){if constexpr(Packed)op.Assemble6Packed4(mem[R],mem[DX],products,mem[EA],mem[EX],mem[BP]);else op.Assemble6(mem[R],products,mem[EA],mem[EX],mem[BP]);}
    if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
    if constexpr(Cube){if constexpr(Packed)op.Forward96Parallel(mem[H],mem[FORWARD_MAP],mem[R],mem[DX],batches,STRIDE);else op.Forward96(mem[H],mem[R],batches,STRIDE);}
    if constexpr(Cube)SyncCube();
    if constexpr(Cube){if constexpr(Packed)op.ApplyFactor208(mem[D],mem[DX],mem[R],batches,STRIDE);else op.template Run<256,32,256,true>(mem[D],mem[R],mem[DX],batches,STRIDE,STRIDE,STRIDE);}
    if constexpr(Packed)CubeToVector<Cube>();else SyncAll<false>();
    if constexpr(!Cube){if constexpr(Packed)op.Finish16(X,mem[XP],mem[R],k,n);else op.Finish(X,mem[XP],mem[DX],k,n);}}
template<bool Cube,bool Packed=false> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,uint32_t batches,uint32_t k,bool broadcast,uint32_t n) {
    // Every participating core observes this metadata before any rendezvous.
    if(batches==0)return;
    GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    std::conditional_t<Cube,NativeMM,Vector> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*n*n];B=B[(uint64_t)(broadcast?0:batch)*n*k];X=X[(uint64_t)batch*n*k];}
    GlobalTensor<int8_t> ad,xd;GlobalTensor<int32_t> products;
    ad.SetGlobalBuffer((__gm__ int8_t*)mem[AD].GetPhyAddr());xd.SetGlobalBuffer((__gm__ int8_t*)mem[XD].GetPhyAddr());products.SetGlobalBuffer((__gm__ int32_t*)mem[PRODUCTS].GetPhyAddr());
    GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    if constexpr(!Cube){if constexpr(Packed){op.NormalizeSeed(mem[CT],A,B,k,n);op.PackCachedB(mem[BP],mem[DX],k);}else{op.Normalize(mem[CT],A,mem[REDUCE],n);op.PackB(mem[BP],mem[DX],B,k,n);}if constexpr(Packed)op.SaveForwardIdentity(mem[FORWARD_IDENTITY]);op.ZeroUnusedFactorRows(mem[D]);}
    if constexpr(Packed)VectorToCube<Cube>();else SyncAll<false>();
    for(uint32_t p=0;p<n;p+=(p<192?96:32)) {
        if(p==192)FactorNarrowTail<Cube,Packed>(op,mem,halfMem,ad,A,workspace,batches,n,p);
        else FactorNarrowPanel<Cube,Packed>(op,mem,halfMem,ad,A,workspace,batches,n,p);
        UpdateNarrowTrailing<Cube,Packed>(op,mem,halfMem,ad,A,workspace,batches,n,p);
    }
    InitialNarrowSolution<Cube,Packed>(op,mem,X,ad,xd,products,batches,k,n);
    RefineNarrowSolution<Cube,Packed>(op,mem,X,ad,xd,products,batches,k,n);
}
}

} // namespace SolveDetail
} // namespace AscendC
