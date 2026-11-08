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
#include "linalg_solve_newton96.h"

namespace AscendC {
namespace SolveDetail {
class NativeMM : public NativeNewton96 {
public:
    // Helpers form the actual updated-panel Gram while owners iterate on the seed.
    // Split it in Cube: high=half(G), low=half(G-high*(-(-I))).
    __aicore__ __attribute__((noinline)) void RefreshSecondGram96(GlobalTensor<float> mem,uint32_t batches) {
        using namespace NarrowOrthogonal;
        constexpr uint32_t stride=STRIDE;
        auto ct=mem[CT];GlobalTensor<half> packed,truth;
        packed.SetGlobalBuffer((__gm__ half*)mem[NH].GetPhyAddr());
        truth.SetGlobalBuffer((__gm__ half*)mem[PRODUCTS].GetPhyAddr());
        uint32_t worker=GetBlockIdx()-batches,workers=GetBlockNum()-batches;
        auto la=l1buf.Get<float>(),lb=la[16384];
        auto hi=l1buf.Get<half>()[65536],minus=l1buf.Get<half>()[69632];
        auto a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        for(uint32_t task=worker;task<batches*6;task+=workers) {
            const uint32_t batch=task/6,row=task%6/2*32,col=task%2*48;
            PipeBarrier<PIPE_ALL>();
            uint8_t pad[4]={};SetFmatrix(1,208,pad,FmatrixMode::FMATRIX_LEFT);
            Nd2NzParams p{};p.ndNum=1;p.nValue=32;p.dValue=208;p.srcDValue=256;p.dstNzC0Stride=32;p.dstNzNStride=1;
            DataCopy(la,ct[batch*stride+(96+row)*256],p);
            p.nValue=48;p.dstNzC0Stride=48;
            DataCopy(lb,ct[batch*stride+(96+col)*256],p);
            p.nValue=48;p.dValue=48;p.srcDValue=96;p.dstNzC0Stride=48;
            DataCopy(minus,packed[batch*stride*2+55296],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=26;q.srcStride=2;
            for(uint32_t r=0;r<2;++r)LoadData(a[r*3328],la[r*128],q);
            q.repeatTimes=78;q.srcStride=1;LoadData(b,lb,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=32;m.n=48;m.k=208;m.cmatrixInitVal=true;Mmad(c,a,b,m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
            FixpipeParamsV220 f{};f.mSize=32;f.nSize=48;f.srcStride=32;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
            Fixpipe<half,float,CFG_ROW_MAJOR>(truth[batch*stride*2+row*96+col],c,f);
            f.dstStride=32;Fixpipe<half,float,CFG_NZ>(hi,c,f);
            PipeBarrier<PIPE_ALL>();
            auto ah=abuf.Get<half>(),bh=bbuf.Get<half>();
            SetFmatrix(1,48,pad,FmatrixMode::FMATRIX_LEFT);
            q.repeatTimes=3;q.srcStride=2;
            for(uint32_t r=0;r<2;++r)LoadData(ah[r*768],hi[r*256],q);
            LoadData3DParamsV2Pro z{};z.channelSize=48;z.extConfig=((uint64_t)48<<16)|48;LoadData<half>(bh,minus,z);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            m.k=48;m.cmatrixInitVal=false;Mmad(c,ah,bh,m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
            f.dstStride=96;Fixpipe<half,float,CFG_ROW_MAJOR>(truth[batch*stride*2+9216+row*96+col],c,f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SecondNewtonRefreshed96(GlobalTensor<float> mem,uint32_t batches) {
        using namespace NarrowOrthogonal;
        constexpr uint32_t stride=STRIDE;
        const uint32_t batch=GetBlockIdx();
        if(batch<batches) {
            auto base=l1buf.Get<half>(),x=base[18432],eye=base[27648],two=base[36864],r=base[46080],minus=base[55296];
            auto y=cbuf.Get<float>(),tmp=y[9216];
            GlobalTensor<half> packed;packed.SetGlobalBuffer((__gm__ half*)mem[NH].GetPhyAddr());
            Nd2NzParams p{};p.ndNum=1;p.nValue=96;p.dValue=96;p.srcDValue=96;p.dstNzC0Stride=96;p.dstNzNStride=1;
            auto src=packed[batch*stride*2];
            DataCopy(x,src[9216],p);DataCopy(eye,src[18432],p);DataCopy(two,src[27648],p);DataCopy(r,src[46080],p);DataCopy(minus,src[55296],p);PipeBarrier<PIPE_ALL>();
            Product96(x,eye,y);
            for(uint32_t iter=0;iter<13;++iter)NeumannPair96(x,r,y,tmp,iter<12);
        } else {
            RefreshSecondGram96(mem,batches);
            CrossGramFuture208(mem[CT],mem[AN],batches,stride,96);
        }
        SyncCube();
        if(batch<batches) {
            // L1/L0 contents persist; descriptors need not live through helper calls.
            auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];
            auto y=cbuf.Get<float>(),tmp=y[9216];
            auto result=mem[GI],defect=mem[NT];
            GlobalTensor<half> truth;truth.SetGlobalBuffer((__gm__ half*)mem[PRODUCTS].GetPhyAddr());
            Nd2NzParams p{};p.ndNum=1;p.nValue=96;p.dValue=96;p.srcDValue=96;p.dstNzC0Stride=96;p.dstNzNStride=1;
            DataCopy(g,truth[batch*stride*2],p);DataCopy(low,truth[batch*stride*2+9216],p);PipeBarrier<PIPE_ALL>();
            RefineResident96<true>();
            FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }

    template<uint32_t Active=256>
    __aicore__ inline void Gram96(GlobalTensor<float> ct,GlobalTensor<float> g,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48,col=task%2*48;
            Tile<48,48,Active,false,true,false>(ct[batch*stride+(panel+row)*256],ct[batch*stride+(panel+col)*256],g[batch*stride+row*128+col],Active,256,256,128);
        }
    }
    __aicore__ inline void Factor96(GlobalTensor<float> gi,GlobalTensor<float> ct,GlobalTensor<float> d,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48,col=task%2*128;
            Tile<48,128,96,false,false,false>(gi[batch*stride+row*128],ct[batch*stride+panel*256+col],d[batch*stride+(panel+row)*256+col],96,128,256,256);
        }
    }
    template<uint32_t Active=256>
    __aicore__ inline void TrailingH96(GlobalTensor<float> d,GlobalTensor<float> ct,GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48;
            if constexpr(Active==208) {
                if(panel==0) {
                    if(task%2==0)Tile<48,64,208,false,true,false>(d[batch*stride+row*256],ct[batch*stride+96*256],h[batch*stride+row*256+96],208,256,256,256);
                    else Tile<48,64,208,false,true,false>(d[batch*stride+row*256],ct[batch*stride+160*256],h[batch*stride+row*256+160],208,256,256,256);
                } else if(task%2==0) {
                    Tile<48,32,208,false,true,false>(d[batch*stride+(panel+row)*256],ct[batch*stride+192*256],h[batch*stride+(panel+row)*256+192],208,256,256,256);
                }
            } else {
                if(panel==0) {
                    uint32_t col=96+task%2*80;
                    Tile<48,80,128,false,true,false>(d[batch*stride+row*256],ct[batch*stride+col*256],h[batch*stride+row*256+col],256,256,256,256);
                } else {
                    uint32_t col=192+task%2*32;
                    Tile<48,32,256,false,true,false>(d[batch*stride+(panel+row)*256],ct[batch*stride+col*256],h[batch*stride+(panel+row)*256+col],256,256,256,256);
                }
            }
        }
    }
    template<uint32_t Active=256>
    __aicore__ inline void Update96(GlobalTensor<float> h,GlobalTensor<float> ct,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,col=task%2*128;
            if constexpr(Active==208) {
                if(panel==0) {
                    if(task%4<2)Tile<64,128,96,true,false,true>(h[batch*stride+96],ct[batch*stride+col],ct[batch*stride+96*256+col],96,256,256,256);
                    else Tile<64,128,96,true,false,true>(h[batch*stride+160],ct[batch*stride+col],ct[batch*stride+160*256+col],96,256,256,256);
                } else if(task%4<2) {
                    Tile<32,128,96,true,false,true>(h[batch*stride+panel*256+192],ct[batch*stride+panel*256+col],ct[batch*stride+192*256+col],96,256,256,256);
                }
            } else {
                if(panel==0) {
                    uint32_t row=96+task%4/2*80;
                    Tile<80,128,96,true,false,true>(h[batch*stride+row],ct[batch*stride+col],ct[batch*stride+row*256+col],96,256,256,256);
                } else {
                    uint32_t row=192+task%4/2*32;
                    Tile<32,128,96,true,false,true>(h[batch*stride+panel*256+row],ct[batch*stride+panel*256+col],ct[batch*stride+row*256+col],96,256,256,256);
                }
            }
        }
    }
    // Keep strides and the full 256-column factor; only omit zero products.
    __aicore__ inline void CrossGramFuture208(GlobalTensor<float> ct,GlobalTensor<float> cross,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t worker=GetBlockIdx();if(worker<batches)return;worker-=batches;
        uint32_t workers=GetBlockNum()-batches;
        if(panel==0) {
            for(uint32_t task=worker;task<batches*6;task+=workers) {
                uint32_t batch=task/6,future=96+task%6/3*64,row=task%3*32;
                Tile<64,32,208,false,true,false>(ct[batch*stride+future*256],ct[batch*stride+row*256],cross[batch*stride+future*128+row],208,256,256,128);
            }
        } else {
            for(uint32_t task=worker;task<batches*3;task+=workers) {
                uint32_t batch=task/3,row=task%3*32;
                Tile<32,32,208,false,true,false>(ct[batch*stride+192*256],ct[batch*stride+(panel+row)*256],cross[batch*stride+192*128+row],208,256,256,128);
            }
        }
    }
    __aicore__ inline void CrossAndSecondGram208(GlobalTensor<float> ct,GlobalTensor<float> cross,GlobalTensor<float> gram,uint32_t batches,uint32_t stride) {
        uint32_t worker=GetBlockIdx();if(worker<batches)return;worker-=batches;
        const uint32_t workers=GetBlockNum()-batches;
        auto la=l1buf.Get<float>(),lb0=la[16384],lb1=la[24576];
        auto a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        for(uint32_t task=worker;task<batches*6;task+=workers) {
            const uint32_t batch=task/6,future=96+task%6/3*64,row=task%3*32;
            PipeBarrier<PIPE_ALL>();
            uint8_t pad[4]={};SetFmatrix(1,208,pad,FmatrixMode::FMATRIX_LEFT);
            Nd2NzParams p{};p.ndNum=1;p.nValue=64;p.dValue=208;p.srcDValue=256;p.dstNzC0Stride=64;p.dstNzNStride=1;
            DataCopy(la,ct[batch*stride+future*256],p);
            p.nValue=32;p.dstNzC0Stride=32;
            DataCopy(lb0,ct[batch*stride+row*256],p);
            DataCopy(lb1,ct[batch*stride+(96+row)*256],p);
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData2DParams q{};q.repeatTimes=26;q.srcStride=4;
            for(uint32_t r=0;r<4;++r)LoadData(a[r*3328],la[r*128],q);
            q.repeatTimes=52;q.srcStride=1;
            LoadData(b,lb0,q);LoadData(b[8192],lb1,q);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams m{};m.m=64;m.n=32;m.k=208;m.cmatrixInitVal=true;
            Mmad(c,a,b,m);Mmad(c[2048],a,b[8192],m);
            SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
            FixpipeParamsV220 f{};f.mSize=64;f.nSize=32;f.srcStride=64;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(cross[batch*stride+future*128+row],c,f);
            // Extra bottom32 cache rows are replaced by PadGram96 before use.
            Fixpipe<float,float,CFG_ROW_MAJOR>(gram[batch*stride+(future-96)*128+row],c[2048],f);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void SchurSecondGram96(GlobalTensor<float> cross,GlobalTensor<float> h,GlobalTensor<float> gram,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            const uint32_t batch=task/4,row=task%4/2*48,col=task%2*48;
            Tile<48,48,96,false,false,true>(cross[batch*stride+(96+row)*128],h[batch*stride+96+col],gram[batch*stride+row*128+col],96,128,256,128);
        }
    }
    template<uint32_t N>
    __aicore__ inline void HWithResidentInverse(GlobalTensor<float> cross,GlobalTensor<float> h) {
        auto lb=l1buf.Get<float>()[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        PipeBarrier<PIPE_ALL>();
        Nd2NzParams p{};p.ndNum=1;p.nValue=N;p.dValue=96;p.srcDValue=128;p.dstNzC0Stride=N;p.dstNzNStride=1;
        DataCopy(lb,cross,p);
        SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData2DParams q{};q.repeatTimes=96*N/128;q.srcStride=1;LoadData(b,lb,q);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=48;m.n=N;m.k=96;m.cmatrixInitVal=true;Mmad(c,a,b,m);
        SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=48;f.nSize=N;f.srcStride=48;f.dstStride=256;
        Fixpipe<float,float,CFG_ROW_MAJOR>(h,c,f);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void HFromCross208(GlobalTensor<float> gi,GlobalTensor<float> cross,GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        // key26 uses exactly four Cube tasks per batch. Factor96 just loaded
        // this core's GI row tile into L0A, including on the fallback path.
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4/2*48;
            if(panel==0) {
                uint32_t col=96+task%2*64;
                HWithResidentInverse<64>(cross[batch*stride+col*128],h[batch*stride+row*256+col]);
            } else if(task%2==0) {
                HWithResidentInverse<32>(cross[batch*stride+192*128],h[batch*stride+(panel+row)*256+192]);
            }
        }
    }
    __aicore__ inline void TailCoefficients224(GlobalTensor<float> d,GlobalTensor<float> ct,GlobalTensor<float> e,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*3;task+=GetBlockNum()) {
            uint32_t batch=task/3,row=task%3*64;
            Tile<64,32,208,false,true,false>(d[batch*stride+row*256],ct[batch*stride+192*256],e[batch*stride+row*32],208,256,256,32);
        }
    }
    __aicore__ inline void ReprojectTail224(GlobalTensor<float> e,GlobalTensor<float> ct,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,col=task%4*64;
            Tile<32,64,192,true,false,true>(e[batch*stride],ct[batch*stride+col],ct[batch*stride+192*256+col],192,32,256,256);
        }
    }
    __aicore__ inline void ApplyFactor208(GlobalTensor<float> d,GlobalTensor<float> y,GlobalTensor<float> x,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,row=task%4*64;
            Tile<64,16,208,true,false,false>(d[batch*stride+row],y[batch*stride],x[batch*stride+row*16],208,256,16,16);
        }
    }
    __aicore__ inline void CrossGramImplicit256(GlobalTensor<float> ct,GlobalTensor<float> cross,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t worker=GetBlockIdx();if(worker<batches)return;worker-=batches;
        uint32_t workers=GetBlockNum()-batches,tiles=(192-panel)/64;
        for(uint32_t task=worker;task<batches*tiles;task+=workers) {
            uint32_t batch=task/tiles,future=panel+64+task%tiles*64;
            Tile<64,64,256,false,true,false>(ct[batch*stride+future*256],ct[batch*stride+panel*256],cross[batch*stride+future*64],256,256,256,64);
        }
    }
    __aicore__ inline void HImplicit256(GlobalTensor<float> inverse,GlobalTensor<float> cross,GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        uint32_t tiles=(192-panel)/64;
        for(uint32_t task=GetBlockIdx();task<batches*tiles;task+=GetBlockNum()) {
            uint32_t batch=task/tiles,future=panel+64+task%tiles*64;
            Tile<64,64,64,false,true,false>(inverse[batch*stride],cross[batch*stride+future*64],h[batch*stride+panel*256+future],64,64,64,256);
        }
    }
    __aicore__ inline void ForwardApplyInverse64(GlobalTensor<float> H,GlobalTensor<float> inverses,GlobalTensor<float> input,GlobalTensor<float> output,uint32_t batches,uint32_t stride) {
        // One owner per RHS block. Forward reads immutable input; the same
        // owner then applies its local inverse after Tile's completion fence.
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,block=task%4;
            auto h=H[batch*stride+block*64],x=input[batch*stride],y=output[batch*stride+block*2048];
            if(block==1)Tile<64,32,64,true,false,true>(h,x,y,64,256,32,32);
            else if(block==2)Tile<64,32,128,true,false,true>(h,x,y,128,256,32,32);
            else if(block==3)Tile<64,32,192,true,false,true>(h,x,y,192,256,32,32);
            Tile<64,32,64,true,false,false>(inverses[batch*stride+block*4096],y,y,64,64,32,32);
        }
    }
    __aicore__ inline void Schur32Parallel(GlobalTensor<float> inverse,GlobalTensor<float> gram,GlobalTensor<float> q,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*2;task+=GetBlockNum()) {
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
    __aicore__ inline void Assemble32Parallel(GlobalTensor<float> inverse,GlobalTensor<float> gram,GlobalTensor<float> q,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*2;task+=GetBlockNum()) {
            uint32_t batch=task/2;auto j=inverse[batch*stride],g=gram[batch*stride],tmp=q[batch*stride];
            if(task%2==0) {
                Tile<32,32,32,false,false,false>(j[2080],tmp,j[2048],32,64,32,64);
                Tile<32,32,32,false,false,true>(g[32],j[2048],j,32,64,64,64);
            } else {
                Tile<32,32,32,false,false,false>(g[32],j[2080],j[32],32,64,64,64);
            }
        }
    }
    __aicore__ inline void Forward96(GlobalTensor<float> H,GlobalTensor<float> Y,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],y=Y[batch*stride];
            Tile<160,32,96,true,false,true>(h[96],y,y[96*32],96,256,32,32);
            Tile<64,32,96,true,false,true>(h[96*256+192],y[96*32],y[192*32],96,256,32,32);
        }
    }
    __aicore__ inline void ComposeForward96(GlobalTensor<float> H,GlobalTensor<float> E,GlobalTensor<float> identity,GlobalTensor<float> F,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*2;task+=GetBlockNum()) {
            uint32_t batch=task/2;auto h=H[batch*stride],e=E[batch*stride],i=identity[batch*stride],f=F[batch*stride];
            if(task%2==0) {
                Tile<96,32,96,false,false,false>(h[96],h[96*256+192],f,96,256,256,32);
                Tile<96,32,96,false,false,true>(h[96],e[96*32],f,96,256,32,32);
                Tile<96,32,96,false,false,true>(i,h[192],f,96,96,256,32);
                Tile<96,32,96,false,false,true>(i,e,f,96,96,32,32);
            } else {
                Tile<96,32,96,false,false,false>(i,h[96*256+192],f[96*32],96,96,256,32);
                Tile<96,32,96,false,false,true>(i,e[96*32],f[96*32],96,96,32,32);
            }
        }
    }
    __aicore__ inline void Forward96Parallel(GlobalTensor<float> H,GlobalTensor<float> F,GlobalTensor<float> input,GlobalTensor<float> output,uint32_t batches,uint32_t stride) {
        for(uint32_t task=GetBlockIdx();task<batches*4;task+=GetBlockNum()) {
            uint32_t batch=task/4,block=task%4;auto x=input[batch*stride],y=output[batch*stride];
            if(block<3)Tile<32,16,96,true,false,true>(H[batch*stride+96+block*32],x,y[(96+block*32)*16],96,256,16,16);
            else Tile<32,16,192,true,false,true>(F[batch*stride],x,y[192*16],192,32,16,16);
        }
    }
    __aicore__ inline void Forward96Tail224(GlobalTensor<float> H,GlobalTensor<float> E,GlobalTensor<float> Y,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],y=Y[batch*stride];
            Tile<128,32,96,true,false,true>(h[96],y,y[96*32],96,256,32,32);
            Tile<32,32,96,true,false,true>(h[96*256+192],y[96*32],y[192*32],96,256,32,32);
            Tile<32,32,192,true,false,true>(E[batch*stride],y,y[192*32],192,32,32,32);
        }
    }
    __aicore__ inline void Backsolve96(GlobalTensor<float> H,GlobalTensor<float> X,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            auto h=H[batch*stride],x=X[batch*stride];
            Tile<96,32,64,false,false,true>(h[96*256+192],x[192*32],x[96*32],64,256,32,32);
            Tile<96,32,160,false,false,true>(h[96],x[96*32],x,160,256,32,32);
        }
    }

};

} // namespace SolveDetail
} // namespace AscendC
