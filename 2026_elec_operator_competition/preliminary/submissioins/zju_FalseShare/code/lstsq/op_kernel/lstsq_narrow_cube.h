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

#include "lstsq_orthogonal.h"

namespace NarrowOrthogonal {
constexpr uint32_t CT=0,D=65536,H=131072,AN=196608,G=262144,GI=278528,
 BP=294912,XP=303104,R=311296,DX=319488,REDUCE=327680,AD=327936,XD=426240,PRODUCTS=447744,EA=472320,EX=472576,NT=472832,NH=489216,CROSS=538368,STRIDE=kNarrowOrthogonalWorkspaceFloats;
static_assert(NH+49152==CROSS && CROSS+256*128<=STRIDE);
// Five operand digits still feed all six retained convolution coefficients.
constexpr uint32_t kOperandDigits=5,kResidualWidth=256*kOperandDigits;
template<uint32_t Panel> class NativeMM : public CubeTileStorage {
public:
    template<uint32_t M,uint32_t N,uint32_t K,bool TA=false,bool TB=false,bool Accumulate=false>
    __aicore__ inline void Run(AscendC::GlobalTensor<float> A,AscendC::GlobalTensor<float> B,AscendC::GlobalTensor<float> C,
        uint32_t batches,uint32_t astride,uint32_t bstride,uint32_t cstride,
        uint32_t as=TA?M:K,uint32_t bs=TB?K:N,uint32_t cs=N,uint32_t mirrorOffset=0) {
        constexpr uint32_t BM=(M==128&&N>=128)?64:(M>=128?((M==192 || (N<=32 && K>=128))?64:128):(M==64 && N>=128?64:(M>=32?32:16)));
        constexpr uint32_t BN=N>=128?(M<=128?64:128):(N>=32?32:16);
        constexpr uint32_t BK=K==192 && BM<=64 && BN<=64?192:(K>=256 && BM<=64 && BN<=64?256:(K>=128?128:(K>=64?64:(K>=32?32:16))));
        constexpr uint32_t nr=M/BM,nc=N/BN;
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*nr*nc;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/(nr*nc),r=task/nc%nr*BM,col=task%nc*BN;
            Tile<BM,BN,BK,TA,TB,Accumulate>(A[batch*astride+(TA?r:r*as)],B[batch*bstride+(TB?col*bs:col)],C[batch*cstride+r*cs+col],K,as,bs,cs,0,mirrorOffset);
        }
    }
    __aicore__ inline void Int8Packed4(AscendC::GlobalTensor<int8_t> A,AscendC::GlobalTensor<int8_t> B,AscendC::GlobalTensor<int32_t> C,uint32_t batches,uint32_t stride) {
        constexpr uint32_t Rows=64,First=48,BN=48,K=kResidualWidth;
        static_assert(First*K<=65536 && BN*K<=65536);
        static_assert(Rows*K<=131072 && 131072+BN*K<=262144);
        auto la=l1buf.Get<int8_t>(),lb=la[131072],a=abuf.Get<int8_t>(),b=bbuf.Get<int8_t>();auto c=cbuf.Get<int32_t>();
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
            uint32_t batch=task/4,row=task%4*Rows;AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=Rows;p.dValue=K;p.srcDValue=K;p.dstNzC0Stride=Rows;p.dstNzNStride=1;
            AscendC::DataCopy(la,A[batch*stride*4+row*K],p);
            p.nValue=BN;p.dstNzC0Stride=BN;AscendC::DataCopy(lb,B[batch*stride*4],p);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::LoadData2DParams right{};right.repeatTimes=K*BN/512;right.srcStride=1;AscendC::LoadData(b,lb,right);
            // Both row groups reuse the complete right operand in L0B.
            for(uint32_t part=0;part<2;++part) {
                const uint32_t first=part?First:0,rows=part?Rows-First:First;
                AscendC::LoadData2DParams q{};q.repeatTimes=K/32;q.srcStride=Rows/16;
                for(uint32_t r=0;r<rows/16;++r)AscendC::LoadData(a[r*K*16],la[(first/16+r)*512],q);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::MmadParams m{};m.m=rows;m.n=BN;m.k=K;m.cmatrixInitVal=true;AscendC::Mmad(c,a,b,m);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);AscendC::FixpipeParamsV220 f{};f.mSize=rows;f.nSize=BN;f.srcStride=rows;f.dstStride=BN;
                AscendC::Fixpipe<int32_t,int32_t,AscendC::CFG_ROW_MAJOR>(C[batch*stride+(row+first)*BN],c,f);AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
    template<uint32_t N,bool Accumulate=false>
    __aicore__ inline void LocalRun(AscendC::GlobalTensor<float> A,AscendC::GlobalTensor<float> B,AscendC::GlobalTensor<float> C,uint32_t batches,uint32_t stride) {
        for(uint32_t batch=AscendC::GetBlockIdx();batch<batches;batch+=AscendC::GetBlockNum())Tile<N,N,N,false,false,Accumulate>(A[batch*stride],B[batch*stride],C[batch*stride],N,128,128,128);
    }
    template<bool Accumulate=false>
    __aicore__ inline void ProductPanel(AscendC::LocalTensor<half> la,AscendC::LocalTensor<half> lb,AscendC::LocalTensor<float> c) {
        AscendC::PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        uint8_t pad[4]={};AscendC::SetFmatrix(1,Panel,pad,AscendC::FmatrixMode::FMATRIX_LEFT);
        AscendC::LoadData2DParams q{};q.repeatTimes=Panel/16;q.srcStride=Panel/16;
        for(uint32_t r=0;r<Panel/16;++r)AscendC::LoadData(a[r*(16*Panel)],la[r*256],q);
        AscendC::LoadData3DParamsV2Pro z{};z.channelSize=Panel;z.extConfig=((uint64_t)Panel<<16)|Panel;AscendC::LoadData<half>(b,lb,z);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::MmadParams m{};m.m=Panel;m.n=Panel;m.k=Panel;m.cmatrixInitVal=!Accumulate;AscendC::Mmad(c,a,b,m);AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void StorePanel(AscendC::LocalTensor<half> dst,AscendC::LocalTensor<float> src) {
        AscendC::FixpipeParamsV220 f{};f.mSize=Panel;f.nSize=Panel;f.srcStride=Panel;f.dstStride=Panel;f.quantPre=QuantMode_t::F322F16;
        AscendC::Fixpipe<half,float,AscendC::CFG_NZ>(dst,src,f);AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ __attribute__((noinline)) void ReplicatedNeumannPrefix(AscendC::GlobalTensor<half> packed,
        AscendC::GlobalTensor<half> output,uint32_t stride,uint32_t steps) {
        static_assert(Panel==96 || Panel==112);
        const uint32_t batch=AscendC::GetBlockIdx()/4,lane=AscendC::GetBlockIdx()%4,row=lane*32;
        if(row<Panel) {
            const uint32_t rows=Panel-row<32?Panel-row:32;
            auto g=l1buf.Get<half>(),x=g[2*Panel*Panel],eye=g[3*Panel*Panel],r=g[5*Panel*Panel];
            auto a=abuf.Get<half>(),b=bbuf.Get<half>();auto y=cbuf.Get<float>(),tmp=y[Panel*Panel];
            auto src=packed[batch*stride*2];
            AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=Panel;p.dValue=Panel;p.srcDValue=Panel;p.dstNzC0Stride=Panel;p.dstNzNStride=1;
            AscendC::DataCopy(eye,src[2*Panel*Panel],p);AscendC::DataCopy(r,src[5*Panel*Panel],p);
            p.nValue=rows;p.dstNzC0Stride=rows;AscendC::DataCopy(x,src[Panel*Panel+row*Panel],p);AscendC::PipeBarrier<PIPE_ALL>();
            uint8_t pad[4]={};AscendC::SetFmatrix(1,Panel,pad,AscendC::FmatrixMode::FMATRIX_LEFT);
            AscendC::LoadData2DParams left{};left.repeatTimes=Panel/16;left.srcStride=rows/16;
            for(uint32_t q=0;q<rows/16;++q)AscendC::LoadData(a[q*Panel*16],x[q*256],left);
            AscendC::LoadData3DParamsV2Pro right{};right.channelSize=Panel;right.extConfig=((uint64_t)Panel<<16)|Panel;AscendC::LoadData<half>(b,eye,right);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::MmadParams m{};m.m=rows;m.n=Panel;m.k=Panel;m.cmatrixInitVal=true;AscendC::Mmad(y,a,b,m);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t iter=0;iter+1<steps;++iter) {
                left.srcStride=rows/16;
                for(uint32_t q=0;q<rows/16;++q)AscendC::LoadData(a[q*Panel*16],x[q*256],left);
                AscendC::LoadData<half>(b,r,right);AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
                m.m=rows;m.cmatrixInitVal=false;AscendC::Mmad(y,a,b,m);AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
                const bool square=iter+2<steps;
                if(square) {
                    left.srcStride=Panel/16;
                    for(uint32_t q=0;q<Panel/16;++q)AscendC::LoadData(a[Panel*Panel+q*Panel*16],r[q*256],left);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
                    m.m=Panel;m.cmatrixInitVal=true;AscendC::Mmad(tmp,a[Panel*Panel],b,m);AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
                }
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
                AscendC::FixpipeParamsV220 local{};local.mSize=rows;local.nSize=Panel;local.srcStride=rows;local.dstStride=rows;local.quantPre=QuantMode_t::F322F16;
                AscendC::Fixpipe<half,float,AscendC::CFG_NZ>(x,y,local);
                if(square) {
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
                    AscendC::FixpipeParamsV220 retain{};retain.mSize=Panel;retain.nSize=Panel;retain.srcStride=Panel;retain.dstStride=Panel;retain.quantPre=QuantMode_t::F322F16;
                    AscendC::Fixpipe<half,float,AscendC::CFG_NZ>(r,tmp,retain);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            // Publish only the owned X rows once. NT is not a defect buffer yet.
            AscendC::FixpipeParamsV220 f{};f.mSize=rows;f.nSize=Panel;f.srcStride=rows;f.dstStride=Panel;f.quantPre=QuantMode_t::F322F16;
            AscendC::Fixpipe<half,float,AscendC::CFG_ROW_MAJOR>(output[batch*stride*2+row*Panel],y,f);AscendC::PipeBarrier<PIPE_ALL>();
        }
        // For Panel=96, lane3 owns no rows but joins the same rendezvous.
        SyncCube();
    }
    template<bool Second=false>
    __aicore__ inline void NewtonPanel(AscendC::GlobalTensor<half> packed,AscendC::GlobalTensor<float> result,AscendC::GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        AscendC::GlobalTensor<half> prefix;prefix.SetGlobalBuffer((__gm__ half*)defect.GetPhyAddr());
        ReplicatedNeumannPrefix(packed,prefix,stride,steps);
        if(AscendC::GetBlockIdx()%4)return;
        const uint32_t batch=AscendC::GetBlockIdx()/4;
        auto g=l1buf.Get<half>(),low=g[Panel*Panel],x=g[2*Panel*Panel],eye=g[3*Panel*Panel],two=g[4*Panel*Panel],r=g[5*Panel*Panel],minus=g[6*Panel*Panel];
        auto y=cbuf.Get<float>(),tmp=y[Panel*Panel];
        AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=Panel;p.dValue=Panel;p.srcDValue=Panel;p.dstNzC0Stride=Panel;p.dstNzNStride=1;
        auto src=packed[batch*stride*2];
        AscendC::DataCopy(g,src,p);AscendC::DataCopy(low,src[4*Panel*Panel],p);AscendC::DataCopy(x,prefix[batch*stride*2],p);AscendC::DataCopy(two,src[3*Panel*Panel],p);
        if constexpr(Second)AscendC::DataCopy(minus,src[6*Panel*Panel],p);AscendC::PipeBarrier<PIPE_ALL>();
        ProductPanel(eye,two,tmp);ProductPanel<true>(g,x,tmp);StorePanel(r,tmp);
        ProductPanel(x,r,y);StorePanel(x,y);
        ProductPanel(eye,eye,tmp);ProductPanel<true>(g,x,tmp);ProductPanel<true>(low,x,tmp);StorePanel(r,tmp);
        ProductPanel(x,eye,y);ProductPanel<true>(x,r,y);
        if constexpr(Second) {
            StorePanel(x,y);ProductPanel<true>(x,minus,y);StorePanel(two,y);
            ProductPanel(eye,eye,tmp);ProductPanel<true>(g,x,tmp);ProductPanel<true>(low,x,tmp);ProductPanel<true>(g,two,tmp);ProductPanel<true>(low,two,tmp);StorePanel(r,tmp);
            ProductPanel(x,eye,y);ProductPanel<true>(two,eye,y);ProductPanel<true>(x,r,y);ProductPanel<true>(two,r,y);
        }
        AscendC::FixpipeParamsV220 f{};f.mSize=Panel;f.nSize=Panel;f.srcStride=Panel;f.dstStride=128;
        AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(result[batch*stride],y,f);AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);AscendC::PipeBarrier<PIPE_ALL>();
    }

    template<uint32_t Active=256>
    __aicore__ inline void GramPanel(AscendC::GlobalTensor<float> ct,AscendC::GlobalTensor<float> g,uint32_t batches,uint32_t stride,uint32_t panel) {
        if constexpr(Panel==96) {

            for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
                uint32_t batch=task/4,row=task%4/2*48,col=task%2*48;
                Tile<48,48,Active,false,true,false>(ct[batch*stride+(panel+row)*256],ct[batch*stride+(panel+col)*256],g[batch*stride+row*128+col],Active,256,256,128);
            }

        } else {

            static_assert(Active==256);
            for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
                uint32_t batch=task/4,row=task%4<2?0:64,col=task%2?64:0;
                auto A=ct[batch*stride+(panel+row)*256],B=ct[batch*stride+(panel+col)*256],C=g[batch*stride+row*128+col];
                if(row==0) {if(col==0)Tile<64,64,256,false,true,false>(A,B,C,256,256,256,128);else Tile<64,48,256,false,true,false>(A,B,C,256,256,256,128);}
                else {if(col==0)Tile<48,64,256,false,true,false>(A,B,C,256,256,256,128);else Tile<48,48,256,false,true,false>(A,B,C,256,256,256,128);}
            }
        }
    }
    __aicore__ inline void FactorPanel(AscendC::GlobalTensor<float> gi,AscendC::GlobalTensor<float> ct,AscendC::GlobalTensor<float> d,uint32_t batches,uint32_t stride,uint32_t panel) {
        if constexpr(Panel==96) {

            for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
                uint32_t batch=task/4,row=task%4/2*48,col=task%2*128;
                Tile<48,128,96,false,false,false>(gi[batch*stride+row*128],ct[batch*stride+panel*256+col],d[batch*stride+(panel+row)*256+col],96,128,256,256);
            }

        } else {

            for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
                uint32_t batch=task/4,row=task%4<2?0:64,col=task%2*128;
                auto A=gi[batch*stride+row*128],B=ct[batch*stride+panel*256+col],C=d[batch*stride+(panel+row)*256+col];
                if(row==0)Tile<64,128,112,false,false,false>(A,B,C,112,128,256,256);
                else Tile<48,128,112,false,false,false>(A,B,C,112,128,256,256);
            }
        }
    }
    template<uint32_t Active> __aicore__ inline void UpdatePanel96Task(AscendC::GlobalTensor<float> h,AscendC::GlobalTensor<float> ct,uint32_t stride,uint32_t panel,uint32_t task) {
        const uint32_t batch=task/4,col=task%2*128;
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
    __aicore__ inline void UpdatePanel112Task(AscendC::GlobalTensor<float> h,AscendC::GlobalTensor<float> ct,uint32_t stride,uint32_t panel,uint32_t task) {
        const uint32_t batch=task/4,col=task%2*128;
                if(panel==0) {
                    if(task%4<2)Tile<80,128,112,true,false,true>(h[batch*stride+112],ct[batch*stride+col],ct[batch*stride+112*256+col],112,256,256,256);
                    else Tile<64,128,112,true,false,true>(h[batch*stride+192],ct[batch*stride+col],ct[batch*stride+192*256+col],112,256,256,256);
                } else {
                    uint32_t row=224+task%4/2*16;
                    Tile<16,128,112,true,false,true>(h[batch*stride+panel*256+row],ct[batch*stride+panel*256+col],ct[batch*stride+row*256+col],112,256,256,256);
                }
    }
    template<uint32_t Active=256>
    __aicore__ inline void UpdatePanel(AscendC::GlobalTensor<float> h,AscendC::GlobalTensor<float> ct,uint32_t batches,uint32_t stride,uint32_t panel) {
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
            if constexpr(Panel==96)UpdatePanel96Task<Active>(h,ct,stride,panel,task);
            else {static_assert(Active==256);UpdatePanel112Task(h,ct,stride,panel,task);}
        }
    }
    __aicore__ __attribute__((noinline)) void ComposeFactorTileImpl(AscendC::GlobalTensor<float> left,AscendC::GlobalTensor<float> right,
        AscendC::GlobalTensor<float> dst,uint32_t BM,uint32_t K,uint32_t BN) {
        auto la=l1buf.Get<float>(),lb=la[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        for(uint32_t at=0;at<K;at+=128) {
            const uint32_t chunk=K-at<128?K-at:128;
            AscendC::PipeBarrier<PIPE_ALL>();uint8_t pad[4]={};AscendC::SetFmatrix(1,chunk,pad,AscendC::FmatrixMode::FMATRIX_LEFT);
            AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=BM;p.dValue=chunk;p.srcDValue=256;p.dstNzC0Stride=BM;p.dstNzNStride=1;
            AscendC::DataCopy(la,left[at],p);
            p.nValue=chunk;p.dValue=BN;p.dstNzC0Stride=chunk;AscendC::DataCopy(lb,right[at*256],p);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::LoadData2DParams q{};q.repeatTimes=chunk/8;q.srcStride=BM/16;
            for(uint32_t row=0;row<BM/16;++row)AscendC::LoadData(a[row*chunk*16],la[row*128],q);
            AscendC::LoadData3DParamsV2Pro z{};z.channelSize=BN;z.extConfig=((uint64_t)chunk<<16)|BN;AscendC::LoadData<float>(b,lb,z);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::MmadParams m{};m.m=BM;m.n=BN;m.k=chunk;m.cmatrixInitVal=at==0;AscendC::Mmad(c,a,b,m);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::FixpipeParamsV220 f{};f.mSize=BM;f.nSize=BN;f.srcStride=BM;f.dstStride=256;
        AscendC::SetAtomicAdd<float>();AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(dst,c,f);AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetAtomicNone();
    }
    template<uint32_t BM,uint32_t K,uint32_t BN=128>
    __aicore__ inline void ComposeFactorTile(AscendC::GlobalTensor<float> left,AscendC::GlobalTensor<float> right,AscendC::GlobalTensor<float> dst) {
        constexpr uint32_t MaxK=K<128?K:128;
        static_assert(BM*MaxK*sizeof(float)<=65536 && BN*MaxK*sizeof(float)<=65536);
        static_assert(BM*BN*sizeof(float)<=131072 && K%16==0);
        ComposeFactorTileImpl(left,right,dst,BM,K,BN);
    }
    template<uint32_t BM>
    __aicore__ inline void PrepareFactorTile(AscendC::GlobalTensor<float> H,AscendC::GlobalTensor<float> factor,uint32_t row,uint32_t col) {
        ComposeFactorTile<BM,Panel>(H[row*256+Panel],factor[Panel*256+col],factor[row*256+col]);
        if(col==0)ComposeFactorTile<BM,Panel,256-2*Panel>(H[row*256+Panel],H[Panel*256+2*Panel],H[row*256+2*Panel]);
    }
    __aicore__ inline void PrepareFactor(AscendC::GlobalTensor<float> H,AscendC::GlobalTensor<float> factor,uint32_t batches,uint32_t stride) {
        // D0 += H01*D1 and H02 += H01*H12 do not depend on the tail factor.
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
            const uint32_t batch=task/4,col=task%2*128,row=(task%4/2)*(Panel==96?48:64);
            auto h=H[batch*stride],d=factor[batch*stride];
            if constexpr(Panel==96)PrepareFactorTile<48>(h,d,row,col);
            else {if(row==0)PrepareFactorTile<64>(h,d,row,col);else PrepareFactorTile<48>(h,d,row,col);}
        }
    }
    __aicore__ inline void CompleteFactor(AscendC::GlobalTensor<float> H,AscendC::GlobalTensor<float> factor,uint32_t batches,uint32_t stride) {
        // The two row blocks now depend only on the completed tail, not each other.
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
            const uint32_t batch=task/4,col=task%2*128,localRow=(task%4/2)*(Panel==96?48:64);
            for(uint32_t block=0;block<2;++block) {
                const uint32_t row=block*Panel+localRow;
                auto left=H[batch*stride+row*256+2*Panel],right=factor[batch*stride+2*Panel*256+col],dst=factor[batch*stride+row*256+col];
                if constexpr(Panel==96)ComposeFactorTile<48,256-2*Panel>(left,right,dst);
                else {if(localRow==0)ComposeFactorTile<64,256-2*Panel>(left,right,dst);else ComposeFactorTile<48,256-2*Panel>(left,right,dst);}
            }
        }
    }
    __aicore__ inline void CrossGramFuture256(AscendC::GlobalTensor<float> ct,AscendC::GlobalTensor<float> cross,uint32_t batches,uint32_t stride,uint32_t panel) {
        const uint32_t id=AscendC::GetBlockIdx();if(id%4==0)return;
        const uint32_t worker=(id/4)*3+id%4-1,workers=AscendC::GetBlockNum()-batches;
        if constexpr(Panel==96) {

            if(panel==0) {
                for(uint32_t task=worker;task<batches*6;task+=workers) {
                    uint32_t batch=task/6,future=96+task%6/3*80,row=task%3*32;
                    Tile<80,32,128,false,true,false>(ct[batch*stride+future*256],ct[batch*stride+row*256],cross[batch*stride+future*128+row],256,256,256,128);
                }
            } else {
                for(uint32_t task=worker;task<batches*3;task+=workers) {
                    uint32_t batch=task/3,row=task%3*32;
                    Tile<64,32,256,false,true,false>(ct[batch*stride+192*256],ct[batch*stride+(panel+row)*256],cross[batch*stride+192*128+row],256,256,256,128);
                }
            }

        } else {

            if(panel==0) {
                for(uint32_t task=worker;task<batches*8;task+=workers) {
                    uint32_t batch=task/8,part=task%8/4,row=task%4*32,future=112+part*80;
                    auto A=ct[batch*stride+future*256],B=ct[batch*stride+row*256],C=cross[batch*stride+future*128+row];
                    if(part==0) {if(row<96)Tile<80,32,128,false,true,false>(A,B,C,256,256,256,128);else Tile<80,16,128,false,true,false>(A,B,C,256,256,256,128);}
                    else {if(row<96)Tile<64,32,256,false,true,false>(A,B,C,256,256,256,128);else Tile<64,16,256,false,true,false>(A,B,C,256,256,256,128);}
                }
            } else {
                for(uint32_t task=worker;task<batches*4;task+=workers) {
                    uint32_t batch=task/4,row=task%4*32;
                    auto A=ct[batch*stride+224*256],B=ct[batch*stride+(panel+row)*256],C=cross[batch*stride+224*128+row];
                    if(row<96)Tile<32,32,256,false,true,false>(A,B,C,256,256,256,128);else Tile<32,16,256,false,true,false>(A,B,C,256,256,256,128);
                }
            }
        }
    }
    template<uint32_t N,uint32_t Rows=48>
    __aicore__ inline void HWithResidentInverse(AscendC::GlobalTensor<float> cross,AscendC::GlobalTensor<float> h) {
        // FactorPanel leaves this core's inverse row tile in L0A.
        auto lb=l1buf.Get<float>()[16384],a=abuf.Get<float>(),b=bbuf.Get<float>(),c=cbuf.Get<float>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Nd2NzParams p{};p.ndNum=1;p.nValue=N;p.dValue=Panel;p.srcDValue=128;p.dstNzC0Stride=N;p.dstNzNStride=1;
        AscendC::DataCopy(lb,cross,p);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::LoadData2DParams q{};q.repeatTimes=Panel*N/128;q.srcStride=1;AscendC::LoadData(b,lb,q);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::MmadParams m{};m.m=Rows;m.n=N;m.k=Panel;m.cmatrixInitVal=true;AscendC::Mmad(c,a,b,m);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::FixpipeParamsV220 f{};f.mSize=Rows;f.nSize=N;f.srcStride=Rows;f.dstStride=256;
        AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(h,c,f);AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void HFromCross256(AscendC::GlobalTensor<float> cross,AscendC::GlobalTensor<float> h,uint32_t batches,uint32_t stride,uint32_t panel) {
        if constexpr(Panel==96) {

            // Identical task ownership to FactorPanel; no other MMA may intervene.
            for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
                uint32_t batch=task/4,row=task%4/2*48;
                if(panel==0) {
                    uint32_t col=96+task%2*80;
                    HWithResidentInverse<80>(cross[batch*stride+col*128],h[batch*stride+row*256+col]);
                } else {
                    uint32_t col=192+task%2*32;
                    HWithResidentInverse<32>(cross[batch*stride+col*128],h[batch*stride+(panel+row)*256+col]);
                }
            }

        } else {

            for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
                uint32_t batch=task/4,row=task%4<2?0:64,col=panel==0?(112+task%2*80):(224+task%2*16);
                auto A=cross[batch*stride+col*128],C=h[batch*stride+(panel+row)*256+col];
                if(panel==0) {
                    if(row==0) {if(task%2==0)HWithResidentInverse<80,64>(A,C);else HWithResidentInverse<64,64>(A,C);}
                    else {if(task%2==0)HWithResidentInverse<80,48>(A,C);else HWithResidentInverse<64,48>(A,C);}
                } else {if(row==0)HWithResidentInverse<16,64>(A,C);else HWithResidentInverse<16,48>(A,C);}
            }
        }
    }
    __aicore__ inline void ApplyPackedTail(AscendC::GlobalTensor<float> mem,uint32_t offset,uint32_t batches) {
        AscendC::GlobalTensor<int32_t> ready;ready.SetGlobalBuffer((__gm__ int32_t*)mem[REDUCE+96].GetPhyAddr());
        for(uint32_t task=AscendC::GetBlockIdx();task<batches*4;task+=AscendC::GetBlockNum()) {
            const uint32_t batch=task/4,col=task%4*64;
            auto decision=ready[batch*STRIDE];
            AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(decision);
            if(decision.GetValue(0)==0)continue;
            // Each task publishes a disjoint set of final factor columns.
            Tile<32,64,32,false,false,false>(mem[batch*STRIDE+GI],mem[batch*STRIDE+CT+offset*256+col],mem[batch*STRIDE+D+offset*256+col],32,32,256,256);
        }
    }

};
}
