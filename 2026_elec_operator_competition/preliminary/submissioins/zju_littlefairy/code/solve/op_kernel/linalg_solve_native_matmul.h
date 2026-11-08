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
// FP32 GEMM closure. Load3D handles the float transposes that generic Load2D cannot.
__aicore__ inline void SyncCube() {
    PipeBarrier<PIPE_ALL>();
    ffts_cross_core_sync(PIPE_FIX,GetffstMsg(0x0,SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
}
namespace NarrowOrthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,GI=278528,
 BP=294912,XP=303104,R=311296,DX=319488,REDUCE=327680,AD=327936,XD=426240,PRODUCTS=447744,EA=472320,EX=472576,NT=472832,NH=489216,STRIDE=kNarrowOrthogonalWorkspaceFloats;
static_assert(NH+49152==STRIDE);
constexpr uint32_t FORWARD_MAP=G+6144,FORWARD_IDENTITY=AN+32768;
}

class NativeMatmul {
public:
    TPipe pipe;
    TBuf<TPosition::A1> l1buf;
    TBuf<TPosition::A2> abuf;
    TBuf<TPosition::B2> bbuf;
    TBuf<TPosition::CO1> cbuf;
public:
    __aicore__ inline void Init() {
        pipe.InitBuffer(l1buf,262144);pipe.InitBuffer(abuf,65536);
        pipe.InitBuffer(bbuf,65536);pipe.InitBuffer(cbuf,131072);
        SetHF32Mode(false);SetHF32TransMode(false);
    }
    template<uint32_t BM,uint32_t BN,uint32_t BK,bool TA,bool TB,bool Accumulate>
    __aicore__ inline void Tile(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,
        uint32_t K,uint32_t as,uint32_t bs,uint32_t cs,uint32_t skip=0) {
        auto la=l1buf.Get<float>(),lb=la[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        uint8_t pad[4]={};SetFmatrix(1,BK,pad,FmatrixMode::FMATRIX_LEFT);
        for(uint32_t k=0;k<K;k+=BK) {
            PipeBarrier<PIPE_ALL>();
            Nd2NzParams p{};p.ndNum=1;p.nValue=TA?BK:BM;p.dValue=TA?BM:BK;p.srcDValue=as;p.dstNzC0Stride=TA?BK:BM;p.dstNzNStride=1;
            DataCopy(la,A[TA?k*as:k],p);
            p.nValue=TB?BN:BK;p.dValue=TB?BK:BN;p.srcDValue=bs;p.dstNzC0Stride=TB?BN:BK;
            DataCopy(lb,B[TB?k:k*bs],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            if constexpr(TA) {
                LoadData3DParamsV2Pro q{};q.channelSize=BM;q.extConfig=((uint64_t)BK<<16)|BM;q.enTranspose=true;
                LoadData<float>(a,la,q);
            } else {
                LoadData2DParams q{};q.repeatTimes=BK/8;q.srcStride=BM/16;
                for(uint32_t r=0;r<BM/16;++r)LoadData(a[r*BK*16],la[r*128],q);
            }
            if constexpr(TB) {
                LoadData2DParams q{};q.repeatTimes=BK*BN/128;q.srcStride=1;LoadData(b,lb,q);
            } else {
                LoadData3DParamsV2Pro q{};q.channelSize=BN;q.extConfig=((uint64_t)BK<<16)|BN;
                LoadData<float>(b,lb,q);
            }
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=BM;m.n=BN;m.k=BK;m.cmatrixInitVal=k==0;Mmad(c,a,b,m);
        }
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN-skip;f.srcStride=BM;f.dstStride=cs;
        if constexpr(Accumulate)SetAtomicAdd<float>();
        Fixpipe<float,float,CFG_ROW_MAJOR>(C[skip],c[skip*BM],f);PipeBarrier<PIPE_ALL>();
        if constexpr(Accumulate)SetAtomicNone();
    }
    template<uint32_t M,uint32_t N,uint32_t K,bool TA=false,bool TB=false,bool Accumulate=false>
    __aicore__ inline void Run(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,
        uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,
        uint32_t as=TA?M:K,uint32_t bs=TB?K:N,uint32_t cs=N) {
        constexpr uint32_t BM=(M==128&&N>=128)?64:(M>=128?((M==192 || (N==32 && K>=128))?64:128):(M==64 && N>=128?64:32));
        constexpr uint32_t BN=N>=128?(M<=128?64:128):32;
        constexpr uint32_t BK=K==192 && BM<=64 && BN<=64?192:(K>=256 && BM<=64 && BN<=64?256:(K>=128?128:(K>=64?64:32)));
        constexpr uint32_t nr=M/BM,nc=N/BN;
        for(uint32_t task=GetBlockIdx();task<batches*nr*nc;task+=GetBlockNum()) {
            uint32_t batch=task/(nr*nc),r=task/nc%nr*BM,col=task%nc*BN;
            Tile<BM,BN,BK,TA,TB,Accumulate>(A[batch*astride+(TA?r:r*as)],B[batch*bstride+(TB?col*bs:col)],C[batch*cstride+r*cs+col],K,as,bs,cs);
        }
    }
    template<uint32_t K> __aicore__ inline void TileInt8(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t as,uint32_t bs) {
        constexpr uint32_t BM=K>1024?32:64,BN=16,BK=K;
        auto la=l1buf.Get<int8_t>(),lb=la[65536],a=abuf.Get<int8_t>(),b=bbuf.Get<int8_t>();auto c=cbuf.Get<int32_t>();
        for(uint32_t k=0;k<K;k+=BK) {
            PipeBarrier<PIPE_ALL>();Nd2NzParams p{};p.ndNum=1;p.nValue=BM;p.dValue=BK;p.srcDValue=as;p.dstNzC0Stride=BM;p.dstNzNStride=1;DataCopy(la,A[k],p);
            p.nValue=BN;p.dValue=BK;p.srcDValue=bs;p.dstNzC0Stride=BN;DataCopy(lb,B[k],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=BK/32;q.srcStride=BM/16;
            for(uint32_t r=0;r<BM/16;++r)LoadData(a[r*BK*16],la[r*512],q);
            q.repeatTimes=BK*BN/512;q.srcStride=1;LoadData(b,lb,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=BM;m.n=BN;m.k=BK;m.cmatrixInitVal=k==0;Mmad(c,a,b,m);
        }
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN;f.srcStride=BM;f.dstStride=16;
        Fixpipe<int32_t,int32_t,CFG_ROW_MAJOR>(C,c,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t K> __aicore__ inline void RunInt8(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,uint32_t as,uint32_t bs) {
        constexpr uint32_t BM=K>1024?32:64,tiles=256/BM;
        for(uint32_t task=GetBlockIdx();task<batches*tiles;task+=GetBlockNum()) {uint32_t batch=task/tiles,row=task%tiles*BM;TileInt8<K>(A[batch*astride+row*as],B[batch*bstride],C[batch*cstride+row*16],as,bs);}
    }
    template<uint32_t K> __aicore__ inline void Int8Task(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t task,uint32_t stride) {
        constexpr uint32_t BM=K>1024?32:64,tiles=256/BM;
        uint32_t batch=task/tiles,row=task%tiles*BM;
        TileInt8<K>(A[batch*stride*4+row*1536],B[batch*stride*4],C[batch*stride+row*16],1536,K);
    }
    __aicore__ inline void Int8Packed4(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t batches,uint32_t stride) {
        constexpr uint32_t BM=32,BN=32,K=1536;
        auto la=l1buf.Get<int8_t>(),lb=la[65536],a=abuf.Get<int8_t>(),b=bbuf.Get<int8_t>();auto c=cbuf.Get<int32_t>();
        for(uint32_t task=GetBlockIdx();task<batches*8;task+=GetBlockNum()) {
            uint32_t batch=task/8,row=task%8*32;PipeBarrier<PIPE_ALL>();
            Nd2NzParams p{};p.ndNum=1;p.nValue=32;p.dValue=K;p.srcDValue=K;p.dstNzC0Stride=32;p.dstNzNStride=1;
            DataCopy(la,A[batch*stride*4+row*K],p);DataCopy(lb,B[batch*stride*4],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=K/32;q.srcStride=2;
            for(uint32_t r=0;r<2;++r)LoadData(a[r*K*16],la[r*512],q);
            q.repeatTimes=K*BN/512;q.srcStride=1;LoadData(b,lb,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=BM;m.n=BN;m.k=K;m.cmatrixInitVal=true;Mmad(c,a,b,m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN;f.srcStride=BM;f.dstStride=BN;
            Fixpipe<int32_t,int32_t,CFG_ROW_MAJOR>(C[batch*stride+row*BN],c,f);PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Int8All(GlobalTensor<int8_t> A,GlobalTensor<int8_t> B,GlobalTensor<int32_t> C,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*32;task+=GetBlockNum()) {
            if(task<batches*4)Int8Task<256>(A,B,C,task,stride);
            else if(task<batches*8)Int8Task<512>(A,B[4096],C[4096],task-batches*4,stride);
            else if(task<batches*12)Int8Task<768>(A,B[12288],C[8192],task-batches*8,stride);
            else if(task<batches*16)Int8Task<1024>(A,B[24576],C[12288],task-batches*12,stride);
            else if(task<batches*24)Int8Task<1280>(A,B[40960],C[16384],task-batches*16,stride);
            else Int8Task<1536>(A,B[61440],C[20480],task-batches*24,stride);
        }
    }

    __aicore__ inline void Local64(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum())
            Tile<64,64,64,false,false,false>(A[batch*stride],B[batch*stride],C[batch*stride],64,64,64,64);
    }

    template<uint32_t K,uint32_t AS,uint32_t BS,bool Cached=false,bool Accumulate=false>
    __aicore__ inline void HalfProduct(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>()[Cached?0:8192],b=bbuf.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,BS,pad,FmatrixMode::FMATRIX_LEFT);
        if constexpr(!Cached) {
            LoadData2DParams q{};q.repeatTimes=K/16;q.srcStride=AS/16;
            for(uint32_t r=0;r<4;++r)LoadData(a[r*K*16],la[r*256],q);
        }
        LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)K<<16)|64;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=64;m.n=64;m.k=K;m.cmatrixInitVal=!Accumulate;Mmad(c,a,b,m);
        PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t DS> __aicore__ inline void HalfStore(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=DS;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot,uint32_t AS>
    __aicore__ inline void LoadPair64(LocalTensor<half> la,LocalTensor<half> lb) {
        auto a=abuf.Get<half>()[Slot*4096],b=bbuf.Get<half>()[Slot*4096];
        uint8_t pad[4]={};SetFmatrix(1,64,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=4;q.srcStride=AS/16;
        for(uint32_t row=0;row<4;++row)LoadData(a[row*1024],la[row*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)64<<16)|64;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(Slot);WaitFlag<HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair64(LocalTensor<half> x,LocalTensor<half> r,LocalTensor<float> y,LocalTensor<float> tmp,bool square) {
        LoadPair64<0,192>(x,r);auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        MmadParams m{};m.m=64;m.n=64;m.k=64;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
        if(square){LoadPair64<1,64>(r,r);m.cmatrixInitVal=true;Mmad(tmp,a[4096],b[4096],m);SetFlag<HardEvent::M_FIX>(1);}
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=192;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(x,y,f);
        if(square){WaitFlag<HardEvent::M_FIX>(1);f.dstStride=64;Fixpipe<half,float,CFG_NZ>(r,tmp,f);}
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Newton64(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto weights=l1buf.Get<half>(),values=weights[12288],x=values[1024],r=weights[24576],low=weights[8192];auto y=cbuf.Get<float>(),tmp=y[4096];
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            Nd2NzParams p{};p.ndNum=1;p.nValue=64;p.dValue=64;p.srcDValue=64;p.dstNzC0Stride=64;p.dstNzNStride=1;
            DataCopy(weights,packed[batch*stride*2+20480],p);DataCopy(weights[4096],packed[batch*stride*2],p);DataCopy(low,packed[batch*stride*2+16384],p);DataCopy(r,packed[batch*stride*2+12288],p);
            p.dstNzC0Stride=192;
            DataCopy(values,packed[batch*stride*2+8192],p);DataCopy(x,packed[batch*stride*2+4096],p);PipeBarrier<PIPE_ALL>();
            HalfProduct<64,192,192>(x,values,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair64(x,r,y,tmp,iter+2<steps);
            }
            // Recompute the last Newton correction from G to remove residual-power drift.
            auto cachedA=abuf.Get<half>();LoadData2DParams q{};q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*2048],weights[row*256],q);PipeBarrier<PIPE_ALL>();
            HalfProduct<128,64,192,true>(weights,values,tmp);HalfStore<64>(r,tmp);
            HalfProduct<64,192,64>(x,r,y);HalfStore<192>(x,y);
            // [I,G_hi,G_lo]*[I;Y;Y], then [Y,Y]*[I;E].
            HalfStore<192>(x[1024],y);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*3072],values[row*256],q);
            q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)LoadData(cachedA[row*3072+1024],weights[4096+row*256],q);
            PipeBarrier<PIPE_ALL>();HalfProduct<192,64,192,true>(weights,values,tmp);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row) {
                LoadData(cachedA[row*2048],x[row*256],q);
                LoadData(cachedA[row*2048+1024],x[row*256],q);
            }
            PipeBarrier<PIPE_ALL>();HalfStore<192>(x,tmp);
            HalfProduct<128,64,192,true>(weights,values,y);
            FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=64;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);
            Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }


    __aicore__ inline void TrailingH32(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride,uint32_t cut) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,col=task%4*64;if(col+64<=cut)continue;
            Tile<32,64,256,false,true,false>(A[batch*stride],B[batch*stride+col*256],C[batch*stride+col],256,256,256,256,col<cut?cut-col:0);
        }
    }


    template<bool Add=false>
    __aicore__ inline void Product128(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,128,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=8;q.srcStride=8;
        for(uint32_t r=0;r<8;++r)LoadData(a[r*2048],la[r*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=128;z.extConfig=((uint64_t)128<<16)|128;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=128;m.n=128;m.k=128;m.cmatrixInitVal=!Add;Mmad(c,a,b,m);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store128(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=128;f.nSize=128;f.srcStride=128;f.dstStride=128;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot>
    __aicore__ inline void LoadPair128(LocalTensor<half> la,LocalTensor<half> lb) {
        auto a=abuf.Get<half>()[Slot*16384],b=bbuf.Get<half>()[Slot*16384];
        uint8_t pad[4]={};SetFmatrix(1,128,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=8;q.srcStride=8;
        for(uint32_t row=0;row<8;++row)LoadData(a[row*2048],la[row*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=128;z.extConfig=((uint64_t)128<<16)|128;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(Slot);WaitFlag<HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair128(LocalTensor<half> x,LocalTensor<half> r,LocalTensor<float> y,LocalTensor<float> tmp,bool square) {
        LoadPair128<0>(x,r);auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        MmadParams m{};m.m=128;m.n=128;m.k=128;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
        if(square){LoadPair128<1>(r,r);m.cmatrixInitVal=true;Mmad(tmp,a[16384],b[16384],m);SetFlag<HardEvent::M_FIX>(1);}
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=128;f.nSize=128;f.srcStride=128;f.dstStride=128;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(x,y,f);
        if(square){WaitFlag<HardEvent::M_FIX>(1);Fixpipe<half,float,CFG_NZ>(r,tmp,f);}
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Newton128(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto g=l1buf.Get<half>(),low=g[16384],x=g[32768],eye=g[49152],two=g[65536],r=g[81920];auto y=cbuf.Get<float>(),tmp=y[16384];
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            Nd2NzParams p{};p.ndNum=1;p.nValue=128;p.dValue=128;p.srcDValue=128;p.dstNzC0Stride=128;p.dstNzNStride=1;
            auto src=packed[batch*stride*2];DataCopy(g,src,p);DataCopy(low,src[65536],p);DataCopy(x,src[16384],p);DataCopy(eye,src[32768],p);DataCopy(two,src[49152],p);DataCopy(r,src[81920],p);PipeBarrier<PIPE_ALL>();
            Product128(x,eye,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair128(x,r,y,tmp,iter+2<steps);
            }
            Product128(eye,two,tmp);Product128<true>(g,x,tmp);Store128(r,tmp);
            Product128(x,r,y);Store128(x,y);
            Product128(eye,eye,tmp);Product128<true>(g,x,tmp);Product128<true>(low,x,tmp);Store128(r,tmp);
            Product128(x,eye,y);Product128<true>(x,r,y);
            FixpipeParamsV220 f{};f.mSize=128;f.nSize=128;f.srcStride=128;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t N,bool Add=false>
    __aicore__ inline void LocalRun(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum())Tile<N,N,N,false,false,Add>(A[batch*stride],B[batch*stride],C[batch*stride],N,128,128,128);
    }
    __aicore__ inline void TrailingH128(GlobalTensor<float> A,GlobalTensor<float> B,GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*64,col=128+task%2*64;
            Tile<64,64,256,false,true,false>(A[batch*stride+row*256],B[batch*stride+col*256],C[batch*stride+row*256+col],256,256,256,256);
        }
    }
    __aicore__ inline void BacksolveMixed(GlobalTensor<float> H,GlobalTensor<float> X,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],x=X[batch*stride];
            Tile<32,32,64,false,false,true>(h[160*256+192],x[192*32],x[160*32],64,256,32,32);
            Tile<32,32,128,false,false,true>(h[128*256+128],x[128*32],x[128*32],128,256,32,32);
            Tile<128,32,128,false,false,true>(h[128],x[128*32],x,128,256,32,32);
        }
    }

};

} // namespace SolveDetail
} // namespace AscendC
