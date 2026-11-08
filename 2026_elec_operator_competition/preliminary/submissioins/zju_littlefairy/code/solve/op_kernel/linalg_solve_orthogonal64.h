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
#include "linalg_solve_matmul.h"

namespace AscendC {
namespace SolveDetail {
namespace Orthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,Q32=266240,
 BP=270336,XP=278528,R=286720,DX=294912,REDUCE=303104,NT=303360,NH=307456,STRIDE=kOrthogonalWorkspaceFloats;
static_assert(NH+12288==STRIDE);
class Vector : public OrthogonalVectorState {
public:
    __aicore__ inline void Init(uint32_t batches) {
        if(batches==0)return;
        InitializeState(batches);
        if(lane<8) {
            auto identity=w[36864];Duplicate(identity,0.0f,512);PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<8;++j)identity.SetValue(j*64+lane*8+j,1.0f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Residual(GlobalTensor<float> dst,GlobalTensor<float> lhs,GlobalTensor<float> rhs,GlobalTensor<float> mirror) {
        for(uint32_t tile=lane;tile<8;tile+=teams) {
            uint32_t off=tile*1024;DataCopy(w,lhs[off],1024);DataCopy(w[4096],rhs[off],1024);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],1024);PipeBarrier<PIPE_ALL>();DataCopy(dst[off],w,1024);DataCopy(mirror[off],w,1024);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void PackCachedB(GlobalTensor<float> dst,GlobalTensor<float> initial,uint32_t k) {
        // Normalize retained the unscaled complete RHS at w[8448].
        // Offset r*32+c maps to r*k+c; only c<k is gathered.
        auto ix=w[16384].ReinterpretCast<int32_t>(),rows=w[16896].ReinterpretCast<int32_t>();
        CreateVecIndex(ix,(int32_t)0,512);PipeBarrier<PIPE_V>();
        ShiftRight(rows,ix,(int32_t)5,512);PipeBarrier<PIPE_V>();
        Muls(rows,rows,(int32_t)(32-k),512);PipeBarrier<PIPE_V>();
        Sub(ix,ix,rows,512);PipeBarrier<PIPE_V>();ShiftLeft(ix,ix,(int32_t)2,512);PipeBarrier<PIPE_V>();
        uint64_t columns=(1ULL<<k)-1,mask[2]={columns|(columns<<32),0};
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            Duplicate(w,0.0f,512);PipeBarrier<PIPE_V>();
            Gather(w,w[8448],ix.ReinterpretCast<uint32_t>(),row*k*4,mask,8,8);PipeBarrier<PIPE_V>();
            Muls(w,w,-inputScale,512);PipeBarrier<PIPE_ALL>();
            DataCopy(dst[row*32],w,512);DataCopy(initial[row*32],w,512);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Normalize(GlobalTensor<float> normalized,GlobalTensor<float> current,GlobalTensor<float> src,GlobalTensor<float> rhs,uint32_t k) {
        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams){DataCopy(w[count],src[tile*4096],4096);count+=4096;}
        PipeBarrier<PIPE_ALL>();
        // Every lane independently derives one scale from identical A/B data.
        Duplicate(w[8192],0.0f,2304);PipeBarrier<PIPE_ALL>();
        DataCopy(w[8192],src,256);DataCopy(w[8448],rhs,256*k);
        PipeBarrier<PIPE_ALL>();Abs(w[12288],w[8192],2304);PipeBarrier<PIPE_V>();
        WholeReduceMax(w[16384],w[12288],64,4,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t aBits,mi;GetReduceRepeatMaxMinSpr<float>(aBits,mi);
        WholeReduceMax(w[16416],w[12544],64,32,1,1,8);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        uint32_t bBits;GetReduceRepeatMaxMinSpr<float>(bBits,mi);
        inputScale=BoundedInputScale(aBits,bBits);
        if(count){Muls(w,w,inputScale,count);PipeBarrier<PIPE_ALL>();}
        count=0;for(uint32_t tile=lane;tile<16;tile+=teams){DataCopy(normalized[tile*4096],w[count],4096);DataCopy(current[tile*4096],w[count],4096);count+=4096;}PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Finish(GlobalTensor<float> dst,GlobalTensor<float> x,GlobalTensor<float> dx,uint32_t k) {
        auto ix=w[16384].ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<k;++c)ix.SetValue(r*k+c,(r*32+c)*4);
        PipeBarrier<PIPE_ALL>();
        for(uint32_t row=lane*16;row<256;row+=teams*16) {
            DataCopy(w,x[row*32],512);DataCopy(w[4096],dx[row*32],512);PipeBarrier<PIPE_ALL>();
            Add(w,w,w[4096],512);PipeBarrier<PIPE_ALL>();
            Gather(w[8192],w,ix,0,16*k);PipeBarrier<PIPE_ALL>();DataCopy(dst[row*k],w[8192],16*k);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Invert32(GlobalTensor<float> dst,GlobalTensor<float> src,bool negative=false) {
        if(lane)return;
        constexpr uint32_t B=32,R=72;
        auto panel=w,pivot=w[B*R],col=pivot[64],weights=col[B];auto ix=weights[256].ReinterpretCast<int32_t>();
        auto recip=w[4160],minusOne=w[4224];
        Duplicate(minusOne,-1.0f,8);
        Duplicate(panel,0.0f,B*R);PipeBarrier<PIPE_ALL>();
        DataCopyPad(panel,src,DataCopyExtParams{B,128,128,5,0},DataCopyPadExtParams<float>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<B;++r){panel.SetValue(r*R+B+r,1.0f);ix.SetValue(r,r*R*4);}PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<B;++j) {
            Gather(col,panel[j],ix.ReinterpretCast<uint32_t>(),0,(uint64_t)32,1,8);PipeBarrier<PIPE_V>();
            Brcb(weights,col,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Div(recip,minusOne,weights[j*8],(uint64_t)8,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Mul(pivot,panel[j*R],recip,(uint64_t)64,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            MulAddDst(panel,pivot,weights,(uint64_t)64,(uint8_t)32,BinaryRepeatParams{1,1,0,9,0,1});PipeBarrier<PIPE_V>();
            Muls(panel[j*R],pivot,-1.0f,(uint64_t)64,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        if(negative){Muls(panel,panel,-1.0f,B*R);PipeBarrier<PIPE_ALL>();}
        DataCopyPad(dst,panel[B],DataCopyExtParams{B,128,5,128,0});PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void NegateBlock(GlobalTensor<float> src,uint32_t owner=1) {
        if(lane!=owner)return;
        DataCopyPad(w,src,DataCopyExtParams{32,128,128,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
        Muls(w,w,-1.0f,1024);PipeBarrier<PIPE_ALL>();DataCopyPad(src,w,DataCopyExtParams{32,128,0,128,0});PipeBarrier<PIPE_ALL>();
    }


    __aicore__ inline void NewtonPrepare(GlobalTensor<half> packed,GlobalTensor<float> gram) {
        if(lane>=8)return;
        auto value=w,ab=w[4096],sum=w[8192],eye=w[12288];auto h=w[16384].ReinterpretCast<half>();
        DataCopy(value,gram,4096);PipeBarrier<PIPE_ALL>();Abs(ab,value,4096);PipeBarrier<PIPE_V>();
        WholeReduceSum(sum,ab,64,64,1,1,8);PipeBarrier<PIPE_V>();WholeReduceMax(sum[128],sum,64,1,1,1,8);PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_ALL>();float alpha=1.8f/sum.GetValue(128);
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(value,gram[row*64],512);PipeBarrier<PIPE_ALL>();
            Cast(h,value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Cast(ab,h,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();Sub(ab,value,ab,512);PipeBarrier<PIPE_V>();Cast(h[2048],ab,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Muls(eye,w[36864],1.0f,512);PipeBarrier<PIPE_V>();
            Cast(h[1024],eye,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Muls(ab,value,-alpha,512);PipeBarrier<PIPE_V>();Add(ab,eye,ab,512);PipeBarrier<PIPE_V>();Cast(h[1536],ab,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Muls(value,eye,-alpha,512);PipeBarrier<PIPE_V>();Cast(h[512],value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
            Muls(value,eye,2.0f,512);PipeBarrier<PIPE_V>();Cast(h[2560],value,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_ALL>();DataCopy(packed[row*64],h,DataCopyParams{4,32,0,224});DataCopy(packed[16384+row*64],h[2048],512);DataCopy(packed[20480+row*64],h[2560],512);PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ZeroDecision(GlobalTensor<int32_t> decision) {
        if(batch || lane)return;auto flag=w.ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NewtonGuard(GlobalTensor<float> dst,GlobalTensor<int32_t> decision) {
        auto ab=w[4096],sum=w[8192];
        for(uint32_t row=lane*8;row<64;row+=teams*8) {
            DataCopy(w,dst[row*64],512);PipeBarrier<PIPE_ALL>();
            Abs(ab,w,512);PipeBarrier<PIPE_V>();WholeReduceSum(sum,ab,64,8,1,1,8);PipeBarrier<PIPE_V>();
            WholeReduceMax(sum[128],sum,8,1,1,1,8);PipeBarrier<PIPE_V>();WholeReduceSum(sum[192],sum,8,1,1,1,8);PipeBarrier<PIPE_ALL>();
            float bound=sum.GetValue(128),total=sum.GetValue(192);
            if(!(bound<=0.015625f && total>=0.0f && total<=2.0f)) {
                auto flag=sum[128].ReinterpretCast<int32_t>();Duplicate(flag,(int32_t)0,8);PipeBarrier<PIPE_ALL>();flag.SetValue(0,1);PipeBarrier<PIPE_ALL>();
                SetAtomicAdd<int32_t>();DataCopy(decision,flag,8);PipeBarrier<PIPE_ALL>();SetAtomicNone();
            }
        }
    }
    __aicore__ inline void NewtonDefect(GlobalTensor<float> dst) {
        NewtonDefect64(w,dst,lane,teams);
    }

};
template<bool Cube,class Op> __aicore__ inline bool ApproximatePanel64(Op &op,GlobalTensor<float> mem,GlobalTensor<half> halfMem,GM_ADDR workspace,uint32_t batches,uint32_t p) {
        const uint32_t giBase=D+p*64;
        bool useExact=p==192;
        if(p<192) {
            GlobalTensor<int32_t> decisions;decisions.SetGlobalBuffer((__gm__ int32_t*)workspace+REDUCE+128+(p/64)*32);
            if constexpr(!Cube)op.ZeroDecision(decisions);
            if constexpr(!Cube)op.NewtonPrepare(halfMem,mem[G]);
            SyncAll<false>();
            if constexpr(Cube){op.Newton64(halfMem,mem[giBase],mem[NT],batches,STRIDE,p==0?6:(p==64?7:9));op.CrossGramImplicit256(mem[CT],mem[D+16384],batches,STRIDE,p);}
            SyncAll<false>();
            if constexpr(!Cube)op.NewtonGuard(mem[NT],decisions);
            if constexpr(Cube)op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);
            SyncAll<false>();
            DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(decisions);
            useExact=decisions.GetValue(0)!=0;
        }
        return useExact;
}
template<bool Cube,class Op> __aicore__ inline void ExactPanel64(Op &op,GlobalTensor<float> mem,GlobalTensor<half> halfMem,GM_ADDR workspace,uint32_t batches,uint32_t p) {
        const uint32_t giBase=D+p*64;
        if constexpr(!Cube){op.Invert32(mem[giBase],mem[G]);op.NegateBlock(mem[G+2048]);}
        if constexpr(Cube){if(p==192)op.template Run<64,128,64,false,false,true>(mem[H+64],mem[H+64*256+128],mem[H+128],batches,STRIDE,STRIDE,STRIDE,256,256,256);}
        SyncAll<false>();
        if constexpr(Cube)op.Schur32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        SyncAll<false>();
        if constexpr(!Cube){op.Invert32(mem[giBase+2080],mem[G+2080],true);op.NegateBlock(mem[G+32]);op.NegateBlock(mem[giBase],2);}
        if constexpr(Cube){if(p==192)op.template Run<128,64,64,false,false,true>(mem[H+128],mem[H+128*256+192],mem[H+192],batches,STRIDE,STRIDE,STRIDE,256,256,256);}
        SyncAll<false>();
        if constexpr(Cube)op.Assemble32Parallel(mem[giBase],mem[G],mem[Q32],batches,STRIDE);
        if constexpr(Cube)SyncCube();
        if constexpr(Cube){if(p<192){op.HImplicit256(mem[giBase],mem[D+16384],mem[H],batches,STRIDE,p);SyncCube();}}

}
template<bool Cube,class Op> __aicore__ inline void UpdateTrailing64(Op &op,GlobalTensor<float> mem,GlobalTensor<half> halfMem,GM_ADDR workspace,uint32_t batches,uint32_t p) {

        if(p+64<256) {
            if constexpr(Cube) {
                if(p==0)op.template Run<192,256,64,true,false,true>(mem[H+64],mem[CT],mem[CT+64*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
                else if(p==64)op.template Run<128,256,64,true,false,true>(mem[H+p*256+128],mem[CT+p*256],mem[CT+128*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
                else op.template Run<64,256,64,true,false,true>(mem[H+p*256+192],mem[CT+p*256],mem[CT+192*256],batches,STRIDE,STRIDE,STRIDE,256,256,256);
            }
            if constexpr(Cube)SyncCube();
        }
}
template<bool Cube> __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,uint32_t batches,uint32_t k,bool broadcast) {
    // Every participating core observes this metadata before any rendezvous.
    if(batches==0)return;
    GlobalTensor<float> mem,A,B,X;
    mem.SetGlobalBuffer((__gm__ float*)workspace);A.SetGlobalBuffer((__gm__ float*)aa);B.SetGlobalBuffer((__gm__ float*)bb);X.SetGlobalBuffer((__gm__ float*)xx);
    std::conditional_t<Cube,NativeMM,Vector> op;
    uint32_t batch=0;
    if constexpr(Cube)op.Init();else {op.Init(batches);batch=op.batch;mem=mem[(uint64_t)batch*STRIDE];A=A[(uint64_t)batch*65536];B=B[(uint64_t)(broadcast?0:batch)*256*k];X=X[(uint64_t)batch*256*k];}
    GlobalTensor<half> halfMem;halfMem.SetGlobalBuffer((__gm__ half*)workspace+((uint64_t)batch*STRIDE+NH)*2);
    if constexpr(!Cube){op.Normalize(mem[AN],mem[CT],A,B,k);op.PackCachedB(mem[BP],mem[DX],k);}
    SyncAll<false>();
    for(uint32_t p=0;p<256;p+=64) {
        if constexpr(Cube)op.template Run<64,64,256,false,true>(mem[CT+p*256],mem[CT+p*256],mem[G],batches,STRIDE,STRIDE,STRIDE);
        SyncAll<false>();
        if(ApproximatePanel64<Cube>(op,mem,halfMem,workspace,batches,p))ExactPanel64<Cube>(op,mem,halfMem,workspace,batches,p);
        UpdateTrailing64<Cube>(op,mem,halfMem,workspace,batches,p);
    }
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[BP],mem[DX],batches,STRIDE);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[XP],batches,STRIDE,STRIDE,STRIDE);
    if constexpr(Cube)SyncCube();
    if constexpr(Cube)op.template Run<256,32,256>(mem[AN],mem[XP],mem[R],batches,STRIDE,STRIDE,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.Residual(mem[R],mem[BP],mem[R],mem[DX]);
    SyncAll<false>();
    if constexpr(Cube){op.ForwardApplyInverse64(mem[H],mem[D],mem[R],mem[DX],batches,STRIDE);SyncCube();}
    if constexpr(Cube)op.template Run<256,32,256,true>(mem[CT],mem[DX],mem[R],batches,STRIDE,STRIDE,STRIDE);
    SyncAll<false>();
    if constexpr(!Cube)op.Finish(X,mem[XP],mem[R],k);
}
}

} // namespace SolveDetail
} // namespace AscendC
