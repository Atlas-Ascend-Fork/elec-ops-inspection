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

#include "lstsq_column.h"

__aicore__ inline void SyncCube() {
    AscendC::PipeBarrier<PIPE_ALL>();
    ffts_cross_core_sync(PIPE_FIX,AscendC::GetffstMsg(0x0,AscendC::SYNC_AIC_FLAG));
    wait_flag_dev(AscendC::SYNC_AIC_FLAG);
}
class CubeTileStorage {
protected:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1buf;
    AscendC::TBuf<AscendC::TPosition::A2> abuf;
    AscendC::TBuf<AscendC::TPosition::B2> bbuf;
    AscendC::TBuf<AscendC::TPosition::CO1> cbuf;
public:
    __aicore__ inline void Init() {
        pipe.InitBuffer(l1buf,262144);pipe.InitBuffer(abuf,65536);
        pipe.InitBuffer(bbuf,65536);pipe.InitBuffer(cbuf,131072);
        AscendC::SetHF32Mode(false);AscendC::SetHF32TransMode(false);
    }
    template<uint32_t BM,uint32_t BN,uint32_t BK,bool TA,bool TB,bool Accumulate>
    __aicore__ inline void Tile(AscendC::GlobalTensor<float> A,AscendC::GlobalTensor<float> B,AscendC::GlobalTensor<float> C,
        uint32_t K,uint32_t as,uint32_t bs,uint32_t cs,uint32_t skip=0,uint32_t mirrorOffset=0) {
        static_assert(BM*BK*sizeof(float)<=65536 && BN*BK*sizeof(float)<=65536);
        static_assert(BM*BN*sizeof(float)<=131072);
        auto la=l1buf.Get<float>(),lb=la[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        uint8_t pad[4]={};AscendC::SetFmatrix(1,BK,pad,AscendC::FmatrixMode::FMATRIX_LEFT);
        for(uint32_t k=0;k<K;k+=BK) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=TA?BK:BM;p.dValue=TA?BM:BK;p.srcDValue=as;p.dstNzC0Stride=TA?BK:BM;p.dstNzNStride=1;
            AscendC::DataCopy(la,A[TA?k*as:k],p);
            p.nValue=TB?BN:BK;p.dValue=TB?BK:BN;p.srcDValue=bs;p.dstNzC0Stride=TB?BN:BK;
            AscendC::DataCopy(lb,B[TB?k:k*bs],p);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            if constexpr(TA) {
                AscendC::LoadData3DParamsV2Pro q{};q.channelSize=BM;q.extConfig=((uint64_t)BK<<16)|BM;q.enTranspose=true;
                AscendC::LoadData<float>(a,la,q);
            } else {
                AscendC::LoadData2DParams q{};q.repeatTimes=BK/8;q.srcStride=BM/16;
                for(uint32_t r=0;r<BM/16;++r)AscendC::LoadData(a[r*BK*16],la[r*128],q);
            }
            if constexpr(TB) {
                AscendC::LoadData2DParams q{};q.repeatTimes=BK*BN/128;q.srcStride=1;AscendC::LoadData(b,lb,q);
            } else {
                AscendC::LoadData3DParamsV2Pro q{};q.channelSize=BN;q.extConfig=((uint64_t)BK<<16)|BN;
                AscendC::LoadData<float>(b,lb,q);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::MmadParams m{};m.m=BM;m.n=BN;m.k=BK;m.cmatrixInitVal=k==0;AscendC::Mmad(c,a,b,m);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN-skip;f.srcStride=BM;f.dstStride=cs;
        if constexpr(Accumulate)AscendC::SetAtomicAdd<float>();
        AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(C[skip],c[skip*BM],f);
        if(mirrorOffset)AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(C[mirrorOffset+skip],c[skip*BM],f);
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr(Accumulate)AscendC::SetAtomicNone();
    }
};
class NativeMM : public CubeTileStorage {
public:
    template<uint32_t M,uint32_t N,uint32_t K,bool TA=false,bool TB=false,bool Accumulate=false>
    __aicore__ inline void Run(AscendC::GlobalTensor<float> A,AscendC::GlobalTensor<float> B,AscendC::GlobalTensor<float> C,
        uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,
        uint32_t as=TA?M:K,uint32_t bs=TB?K:N,uint32_t cs=N,uint32_t mirrorOffset=0) {
        constexpr uint32_t BM=(M==128&&N>=128)?64:(M>=128?((M==192 || (N==32 && K>=128))?64:128):(M==64 && N>=128?64:(M>=32?32:16)));
        constexpr uint32_t BN=N>=128?(M<=128?64:128):(N>=32?32:16);
        constexpr uint32_t BK=K==192 && BM<=64 && BN<=64?192:(K>=256 && BM<=64 && BN<=64?256:(K>=128?128:(K>=64?64:(K>=32?32:16))));
        constexpr uint32_t nr=M/BM,nc=N/BN;
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*nr*nc;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/(nr*nc),r=task/nc%nr*BM,col=task%nc*BN;
            Tile<BM,BN,BK,TA,TB,Accumulate>(A[batch*astride+(TA?r:r*as)],B[batch*bstride+(TB?col*bs:col)],C[batch*cstride+r*cs+col],K,as,bs,cs,0,mirrorOffset);
        }
    }
    __aicore__ inline void ReprojectTail(AscendC::GlobalTensor<float> ct,AscendC::GlobalTensor<float> inverse,AscendC::GlobalTensor<float> h,AscendC::GlobalTensor<float> cross,AscendC::GlobalTensor<float> correction,uint32_t batches,uint32_t stride,uint32_t panel) {
        if(panel==128)Run<128,64,256,false,true,false>(ct,ct[panel*256],cross,batches,stride,stride,stride,256,256,64);
        else Run<192,64,256,false,true,false>(ct,ct[panel*256],cross,batches,stride,stride,stride,256,256,64);
        SyncCube();
        const uint32_t blocks=panel/64;
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*blocks;task+=AscendC::GetBlockNum()) {
            const uint32_t batch=task/blocks,block=task%blocks;
            Tile<64,64,64,false,false,false>(inverse[batch*stride+block*4096],cross[batch*stride+block*4096],correction[batch*stride+block*4096],64,64,64,64);
            auto acc=cbuf.Get<float>();AscendC::FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=256;
            AscendC::SetAtomicAdd<float>();AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(h[batch*stride+block*64*256+panel],acc,f);AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetAtomicNone();
        }
        SyncCube();
        if(panel==128)Run<64,256,128,true,false,true>(correction,ct,ct[panel*256],batches,stride,stride,stride,64,256,256);
        else Run<64,256,192,true,false,true>(correction,ct,ct[panel*256],batches,stride,stride,stride,64,256,256);
        SyncCube();
    }
    __aicore__ inline void TailProducts32(AscendC::GlobalTensor<float> ct,AscendC::GlobalTensor<float> gram,AscendC::GlobalTensor<float> cross,uint32_t batches,uint32_t stride) {
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*2;task+=AscendC::GetBlockNum()) {
            const uint32_t batch=task/2;
            if(task%2==0)Tile<32,32,256,false,true,false>(ct[batch*stride],ct[batch*stride],gram[batch*stride],256,256,256,64);
            else Tile<32,32,256,false,true,false>(ct[batch*stride],ct[batch*stride+32*256],cross[batch*stride],256,256,256,32);
        }
    }
    __aicore__ inline void ComposeForward(AscendC::GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        Run<64,128,64,false,false,true>(h[64],h[64*256+128],h[128],batches,stride,stride,stride,256,256,256);SyncCube();
        if(panel==192){Run<128,64,64,false,false,true>(h[128],h[128*256+192],h[192],batches,stride,stride,stride,256,256,256);SyncCube();}
    }
    __aicore__ inline void Int8Packed4(AscendC::GlobalTensor<int8_t> A,AscendC::GlobalTensor<int8_t> B,AscendC::GlobalTensor<int32_t> C,uint32_t batches,uint32_t stride) {
        constexpr uint32_t BM=32,BN=32,K=1536;
        auto la=l1buf.Get<int8_t>(),lb=la[65536],a=abuf.Get<int8_t>(),b=bbuf.Get<int8_t>();auto c=cbuf.Get<int32_t>();
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*8;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/8,row=task%8*32;AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=32;p.dValue=K;p.srcDValue=K;p.dstNzC0Stride=32;p.dstNzNStride=1;
            AscendC::DataCopy(la,A[batch*stride*4+row*K],p);AscendC::DataCopy(lb,B[batch*stride*4],p);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::LoadData2DParams q{};q.repeatTimes=K/32;q.srcStride=2;
            for(uint32_t r=0;r<2;++r)AscendC::LoadData(a[r*K*16],la[r*512],q);
            q.repeatTimes=K*BN/512;q.srcStride=1;AscendC::LoadData(b,lb,q);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::MmadParams m{};m.m=BM;m.n=BN;m.k=K;m.cmatrixInitVal=true;AscendC::Mmad(c,a,b,m);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);AscendC::FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN;f.srcStride=BM;f.dstStride=BN;
            AscendC::Fixpipe<int32_t,int32_t,AscendC::CFG_ROW_MAJOR>(C[batch*stride+row*BN],c,f);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    template<uint32_t K,uint32_t AS,uint32_t BS,bool Cached=false,bool Accumulate=false>
    __aicore__ inline void HalfProduct(AscendC::LocalTensor<half> la,AscendC::LocalTensor<half> lb,AscendC::LocalTensor<float> c) {
        AscendC::PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>()[Cached?0:8192],b=bbuf.Get<half>();
        uint8_t pad[4]={};AscendC::SetFmatrix(1,BS,pad,AscendC::FmatrixMode::FMATRIX_LEFT);
        if constexpr(!Cached) {
            AscendC::LoadData2DParams q{};q.repeatTimes=K/16;q.srcStride=AS/16;
            for(uint32_t r=0;r<4;++r)AscendC::LoadData(a[r*K*16],la[r*256],q);
        }
        AscendC::LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)K<<16)|64;AscendC::LoadData<half>(b,lb,z);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::MmadParams m{};m.m=64;m.n=64;m.k=K;m.cmatrixInitVal=!Accumulate;AscendC::Mmad(c,a,b,m);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t DS> __aicore__ inline void HalfStore(AscendC::LocalTensor<half> dst,AscendC::LocalTensor<float> src) {
        AscendC::FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=DS;f.quantPre=QuantMode_t::F322F16;
        AscendC::Fixpipe<half,float,AscendC::CFG_NZ>(dst,src,f);AscendC::PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot,uint32_t AS>
    __aicore__ inline void LoadPair64(AscendC::LocalTensor<half> la,AscendC::LocalTensor<half> lb) {
        auto a=abuf.Get<half>()[Slot*4096],b=bbuf.Get<half>()[Slot*4096];
        uint8_t pad[4]={};AscendC::SetFmatrix(1,64,pad,AscendC::FmatrixMode::FMATRIX_LEFT);
        AscendC::LoadData2DParams q{};q.repeatTimes=4;q.srcStride=AS/16;
        for(uint32_t row=0;row<4;++row)AscendC::LoadData(a[row*1024],la[row*256],q);
        AscendC::LoadData3DParamsV2Pro z{};z.channelSize=64;z.extConfig=((uint64_t)64<<16)|64;AscendC::LoadData<half>(b,lb,z);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(Slot);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair64(AscendC::LocalTensor<half> x,AscendC::LocalTensor<half> r,AscendC::LocalTensor<float> y,AscendC::LocalTensor<float> tmp,bool square) {
        LoadPair64<0,192>(x,r);auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        AscendC::MmadParams m{};m.m=64;m.n=64;m.k=64;m.cmatrixInitVal=false;AscendC::Mmad(y,a,b,m);AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        if(square){LoadPair64<1,64>(r,r);m.cmatrixInitVal=true;AscendC::Mmad(tmp,a[4096],b[4096],m);AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);}
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=192;f.quantPre=QuantMode_t::F322F16;
        AscendC::Fixpipe<half,float,AscendC::CFG_NZ>(x,y,f);
        if(square){AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);f.dstStride=64;AscendC::Fixpipe<half,float,AscendC::CFG_NZ>(r,tmp,f);}
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Newton64(AscendC::GlobalTensor<half> packed,AscendC::GlobalTensor<float> result,AscendC::GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto weights=l1buf.Get<half>(),values=weights[12288],x=values[1024],r=weights[24576],low=weights[8192];auto y=cbuf.Get<float>(),tmp=y[4096];
        for(uint32_t batch=AscendC::GetBlockIdx();batch<batches;batch+=AscendC::GetBlockNum()) {
            AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=64;p.dValue=64;p.srcDValue=64;p.dstNzC0Stride=64;p.dstNzNStride=1;
            AscendC::DataCopy(weights,packed[batch*stride*2+20480],p);AscendC::DataCopy(weights[4096],packed[batch*stride*2],p);AscendC::DataCopy(low,packed[batch*stride*2+16384],p);AscendC::DataCopy(r,packed[batch*stride*2+12288],p);
            p.dstNzC0Stride=192;
            AscendC::DataCopy(values,packed[batch*stride*2+8192],p);AscendC::DataCopy(x,packed[batch*stride*2+4096],p);AscendC::PipeBarrier<PIPE_ALL>();
            HalfProduct<64,192,192>(x,values,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair64(x,r,y,tmp,iter+2<steps);
            }
            // Recompute the last Newton correction from G to remove residual-power drift.
            auto cachedA=abuf.Get<half>();AscendC::LoadData2DParams q{};q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)AscendC::LoadData(cachedA[row*2048],weights[row*256],q);AscendC::PipeBarrier<PIPE_ALL>();
            HalfProduct<128,64,192,true>(weights,values,tmp);HalfStore<64>(r,tmp);
            HalfProduct<64,192,64>(x,r,y);HalfStore<192>(x,y);
            // [I,G_hi,G_lo]*[I;Y;Y], then [Y,Y]*[I;E].
            HalfStore<192>(x[1024],y);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row)AscendC::LoadData(cachedA[row*3072],values[row*256],q);
            q.repeatTimes=8;q.srcStride=4;
            for(uint32_t row=0;row<4;++row)AscendC::LoadData(cachedA[row*3072+1024],weights[4096+row*256],q);
            AscendC::PipeBarrier<PIPE_ALL>();HalfProduct<192,64,192,true>(weights,values,tmp);
            q.repeatTimes=4;q.srcStride=12;
            for(uint32_t row=0;row<4;++row) {
                AscendC::LoadData(cachedA[row*2048],x[row*256],q);
                AscendC::LoadData(cachedA[row*2048+1024],x[row*256],q);
            }
            AscendC::PipeBarrier<PIPE_ALL>();HalfStore<192>(x,tmp);
            HalfProduct<128,64,192,true>(weights,values,y);
            AscendC::FixpipeParamsV220 f{};f.mSize=64;f.nSize=64;f.srcStride=64;f.dstStride=64;
            AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(result[batch*stride],y,f);
            AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);AscendC::PipeBarrier<PIPE_ALL>();
        }
    }


    __aicore__ inline void CrossGramImplicit256(AscendC::GlobalTensor<float> ct,AscendC::GlobalTensor<float> cross,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t worker=AscendC::GetBlockIdx();if(worker<batches)return;worker-=batches;
        uint32_t workers=AscendC::GetBlockNum()-batches,tiles=(192-panel)/64;
        for(uint32_t task=worker;task<batches*tiles;task+=workers) {
            uint32_t batch=task/tiles,future=panel+64+task%tiles*64;
            Tile<64,64,256,false,true,false>(ct[batch*stride+future*256],ct[batch*stride+panel*256],cross[batch*stride+future*64],256,256,256,64);
        }
    }
    __aicore__ inline void HImplicit256(AscendC::GlobalTensor<float> inverse,AscendC::GlobalTensor<float> cross,AscendC::GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t tiles=(192-panel)/64;
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*tiles;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/tiles,future=panel+64+task%tiles*64;
            Tile<64,64,64,false,true,false>(inverse[batch*stride],cross[batch*stride+future*64],h[batch*stride+panel*256+future],64,64,64,256);
        }
    }
    __aicore__ inline void ForwardApplyInverse64(AscendC::GlobalTensor<float> H,AscendC::GlobalTensor<float> inverses,AscendC::GlobalTensor<float> input,AscendC::GlobalTensor<float> output,uint32_t batches,uint32_t stride,uint32_t activeBlocks) {
        // One owner per RHS block. Forward reads immutable input; the same
        // owner then applies its local inverse after Tile's completion fence.
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/4,block=task%4;if(block>=activeBlocks)continue;
            auto h=H[batch*stride+block*64],x=input[batch*stride],y=output[batch*stride+block*2048];
            if(block==1)Tile<64,32,64,true,false,true>(h,x,y,64,256,32,32);
            else if(block==2)Tile<64,32,128,true,false,true>(h,x,y,128,256,32,32);
            else if(block==3)Tile<64,32,192,true,false,true>(h,x,y,192,256,32,32);
            if(block+1<activeBlocks)Tile<64,32,64,true,false,false>(inverses[batch*stride+block*4096],y,y,64,64,32,32);
        }
    }
    __aicore__ inline void Schur32Parallel(AscendC::GlobalTensor<float> inverse,AscendC::GlobalTensor<float> gram,AscendC::GlobalTensor<float> q,uint32_t batches,uint32_t stride) {
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*2;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/2;auto j=inverse[batch*stride],g=gram[batch*stride],tmp=q[batch*stride];
            if(task%2==0) {
                Tile<32,32,32,false,false,false>(j,g[32],g[32],32,64,64,64);
                Tile<32,32,32,false,false,true>(g[2048],g[32],g[2080],32,64,64,64);
            } else {
                // Preserve g[2048] until the other owner has consumed it.
                Tile<32,32,32,false,false,false>(g[2048],j,tmp,32,64,64,32);
            }
        }
    }
    __aicore__ inline void Assemble32Parallel(AscendC::GlobalTensor<float> inverse,AscendC::GlobalTensor<float> gram,AscendC::GlobalTensor<float> q,uint32_t batches,uint32_t stride) {
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*2;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/2;auto j=inverse[batch*stride],g=gram[batch*stride],tmp=q[batch*stride];
            if(task%2==0) {
                Tile<32,32,32,false,false,false>(j[2080],tmp,j[2048],32,64,32,64);
                Tile<32,32,32,false,false,true>(g[32],j[2048],j,32,64,64,64);
            } else {
                Tile<32,32,32,false,false,false>(g[32],j[2080],j[32],32,64,64,64);
            }
        }
    }
};

__aicore__ inline void NormalizeOrthogonalInput(AscendC::LocalTensor<float> w,uint32_t lane,uint32_t teams,float &inputScale,AscendC::GlobalTensor<float> normalized,AscendC::GlobalTensor<float> current,AscendC::GlobalTensor<float> src,AscendC::GlobalTensor<float> rhs,uint32_t k,uint32_t n) {

        uint32_t count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            AscendC::Duplicate(w[count],0.0f,4096);AscendC::PipeBarrier<PIPE_ALL>();
            const uint32_t row=tile*16,realRows=row<n?(n-row<16?n-row:16):0;
            if(realRows)AscendC::DataCopyPad(w[count],src[row*n],AscendC::DataCopyExtParams{(uint16_t)realRows,n*4,0,(256-(n+7)/8*8)/8,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            count+=4096;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // Every lane independently derives one scale from identical A/B data.
        AscendC::Duplicate(w[8192],0.0f,2304);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(w[8192],src,AscendC::DataCopyExtParams{1,n*4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
        AscendC::DataCopyPad(w[8448],rhs,AscendC::DataCopyExtParams{1,n*k*4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
        AscendC::PipeBarrier<PIPE_ALL>();AscendC::Abs(w[12288],w[8192],2304);AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(w[16384],w[12288],64,4,1,1,8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        uint32_t aBits,mi;AscendC::GetReduceRepeatMaxMinSpr<float>(aBits,mi);
        AscendC::WholeReduceMax(w[16416],w[12544],64,32,1,1,8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        uint32_t bBits;AscendC::GetReduceRepeatMaxMinSpr<float>(bBits,mi);
        int32_t aExponent=(int32_t)((aBits>>23)&255)-127;
        int32_t bExponent=(int32_t)((bBits>>23)&255)-127;
        int32_t scaleExponent=-aExponent;
        const int32_t rhsLimit=126-bExponent;
        scaleExponent=scaleExponent<rhsLimit?scaleExponent:rhsLimit;
        scaleExponent=scaleExponent>126?126:(scaleExponent<-126?-126:scaleExponent);
        union {uint32_t bits;float value;} scale;
        scale.bits=(uint32_t)(127+scaleExponent)<<23;inputScale=scale.value;
        if(count){AscendC::Muls(w,w,inputScale,count);AscendC::PipeBarrier<PIPE_ALL>();}
        count=0;
        for(uint32_t tile=lane;tile<16;tile+=teams) {
            for(uint32_t r=0;r<16;++r)if(tile*16+r>=n)w.SetValue(count+r*256+tile*16+r,1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(normalized[tile*4096],w[count],4096);AscendC::DataCopy(current[tile*4096],w[count],4096);count+=4096;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

    }
template<uint32_t Digits,uint32_t Width>
__aicore__ inline void SplitOrthogonalInput(AscendC::LocalTensor<float> w,uint32_t lane,uint32_t teams,float inputScale,AscendC::GlobalTensor<int8_t> digits,AscendC::GlobalTensor<float> exponents,AscendC::GlobalTensor<float> matrix,uint32_t panel,uint32_t n) {

        if(lane==0)return;
        constexpr uint32_t L=2048;auto v=w,q=w[2048],ab=w[4096],mx=w[6144],e=w[12000];auto hd=w[8192].ReinterpretCast<half>();auto bd=w[10000].ReinterpretCast<int8_t>();
        uint32_t start=((panel/32)*(teams-1)+lane-1)*8;
        if(start<256) {
            uint32_t rows=start<n?(n-start<8?n-start:8):0;
            AscendC::Duplicate(v,0.0f,L);AscendC::PipeBarrier<PIPE_ALL>();
            if(rows)AscendC::DataCopyPad(v,matrix[start*n],AscendC::DataCopyExtParams{(uint16_t)rows,n*4,0,(256-(n+7)/8*8)/8,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            AscendC::PipeBarrier<PIPE_ALL>();AscendC::Muls(v,v,inputScale,L);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<8;++r)if(start+r>=n)v.SetValue(r*256+start+r,1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();AscendC::Abs(ab,v,L);AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<8;++r) {
                AscendC::WholeReduceMax(mx,ab[r*256],64,4,1,1,8);AscendC::SetFlag<AscendC::HardEvent::V_S>(0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                uint32_t mb,mi;AscendC::GetReduceRepeatMaxMinSpr<float>(mb,mi);int32_t exponent=(int32_t)((mb>>23)&255)-132;
                if((mb&0x7fffffff)==0){exponent=0;AscendC::Duplicate(v[r*256],0.0f,256);}
                else {union {uint32_t bits;float value;} scale;scale.bits=(uint32_t)(127-exponent)<<23;AscendC::Muls(v[r*256],v[r*256],scale.value,256);}
                AscendC::PipeBarrier<PIPE_ALL>();e.SetValue(r,(float)exponent);
            }
            AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(exponents[start],e,8);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t d=0;d<Digits;++d) {
                AscendC::Adds(q,v,12582912.0f,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Adds(q,q,-12582912.0f,L);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(hd,q,AscendC::RoundMode::CAST_NONE,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(bd,hd,AscendC::RoundMode::CAST_TRUNC,L);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(v,v,q,L);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(v,v,128.0f,L);AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyPad(digits[start*Width+d*256],bd,AscendC::DataCopyExtParams{8,256,0,Width-256,0});AscendC::PipeBarrier<PIPE_ALL>();
            }
        }

    }
__aicore__ inline void OrthogonalTailQR(AscendC::LocalTensor<float> w,AscendC::LocalTensor<float> panel,uint32_t real,uint32_t projectionPasses) {

        auto triangular=w[16384],product=w[20480],partial=w[36864],sum=w[37376],coeff=w[37440],weights=w[37504];
        auto ix=w[38016].ReinterpretCast<int32_t>();
        AscendC::Duplicate(triangular,0.0f,4096);AscendC::Duplicate(partial,0.0f,512);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<real;++j) {
            AscendC::Mul(product,panel[j*256],panel[j*256],256);AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(sum,product,64,4,1,1,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(sum,sum,4,1,1,1,8);AscendC::PipeBarrier<PIPE_V>();AscendC::Sqrt(sum,sum,1);AscendC::PipeBarrier<PIPE_ALL>();
            float norm=sum.GetValue(0);triangular.SetValue(j*64+63,norm);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(panel[j*256],panel[j*256],CheckedReciprocal(norm),256);AscendC::PipeBarrier<PIPE_V>();
            uint32_t rows=real-j-1;
            if(rows)for(uint32_t pass=0;pass<projectionPasses;++pass) {
                for(uint32_t q=0;q<4;++q)AscendC::Mul(product[q*64],panel[(j+1)*256+q*64],panel[j*256+q*64],(uint64_t)64,(uint8_t)rows,AscendC::BinaryRepeatParams{1,1,1,32,32,0});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(partial,product,64,rows*4,2,1,8);AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(coeff,partial,8,rows,1,1,1);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(triangular[j*64],triangular[j*64],coeff,rows);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(coeff,coeff,-1.0f,rows);AscendC::PipeBarrier<PIPE_V>();AscendC::Brcb(weights,coeff,(uint8_t)((rows+7)/8),AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(panel[(j+1)*256+q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)rows,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        for(int32_t j=(int32_t)real-1;j>=0;--j) {
            float norm=triangular.GetValue(j*64+63);
            AscendC::Muls(panel[j*256],panel[j*256],CheckedReciprocal(norm),256);AscendC::PipeBarrier<PIPE_V>();
            if(j) {
                for(int32_t r=0;r<j;++r)ix.SetValue(r,(r*64+j-r-1)*4);AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Gather(coeff,triangular,ix.ReinterpretCast<uint32_t>(),0,(uint32_t)j);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(coeff,coeff,-1.0f,(uint32_t)j);AscendC::PipeBarrier<PIPE_V>();AscendC::Brcb(weights,coeff,(uint8_t)((j+7)/8),AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                for(uint32_t q=0;q<4;++q)AscendC::MulAddDst(panel[q*64],panel[j*256+q*64],weights,(uint64_t)64,(uint8_t)j,AscendC::BinaryRepeatParams{1,1,0,32,0,1});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(panel,panel,-1.0f,real*256);AscendC::PipeBarrier<PIPE_ALL>();

    }
