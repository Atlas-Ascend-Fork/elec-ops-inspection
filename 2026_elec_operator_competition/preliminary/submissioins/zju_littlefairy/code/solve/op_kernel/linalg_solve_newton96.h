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
#include "linalg_solve_native_matmul.h"

namespace AscendC {
namespace SolveDetail {
class NativeNewton96 : public NativeMatmul {
public:
    template<bool Add=false>
    __aicore__ inline void Product96(LocalTensor<half> la,LocalTensor<half> lb,LocalTensor<float> c) {
        PipeBarrier<PIPE_ALL>();auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        uint8_t pad[4]={};SetFmatrix(1,96,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=6;q.srcStride=6;
        for(uint32_t r=0;r<6;++r)LoadData(a[r*1536],la[r*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=96;z.extConfig=((uint64_t)96<<16)|96;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams m{};m.m=96;m.n=96;m.k=96;m.cmatrixInitVal=!Add;Mmad(c,a,b,m);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store96(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot>
    __aicore__ inline void LoadPair96(LocalTensor<half> la,LocalTensor<half> lb) {
        auto a=abuf.Get<half>()[Slot*9216],b=bbuf.Get<half>()[Slot*9216];
        uint8_t pad[4]={};SetFmatrix(1,96,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData2DParams q{};q.repeatTimes=6;q.srcStride=6;
        for(uint32_t row=0;row<6;++row)LoadData(a[row*1536],la[row*256],q);
        LoadData3DParamsV2Pro z{};z.channelSize=96;z.extConfig=((uint64_t)96<<16)|96;LoadData<half>(b,lb,z);
        SetFlag<HardEvent::MTE1_M>(Slot);WaitFlag<HardEvent::MTE1_M>(Slot);
    }
    __aicore__ inline void NeumannPair96(LocalTensor<half> x,LocalTensor<half> r,LocalTensor<float> y,LocalTensor<float> tmp,bool square) {
        LoadPair96<0>(x,r);auto a=abuf.Get<half>(),b=bbuf.Get<half>();
        MmadParams m{};m.m=96;m.n=96;m.k=96;m.cmatrixInitVal=false;Mmad(y,a,b,m);SetFlag<HardEvent::M_FIX>(0);
        if(square){LoadPair96<1>(r,r);m.cmatrixInitVal=true;Mmad(tmp,a[9216],b[9216],m);SetFlag<HardEvent::M_FIX>(1);}
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(x,y,f);
        if(square){WaitFlag<HardEvent::M_FIX>(1);Fixpipe<half,float,CFG_NZ>(r,tmp,f);}
        PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot>
    __aicore__ inline void RefineLoadA96(LocalTensor<half> src) {
        static_assert(Slot<3);
        auto dst=abuf.Get<half>()[Slot*9216];
        LoadData2DParams q{};q.repeatTimes=6;q.srcStride=6;
        for(uint32_t row=0;row<6;++row)LoadData(dst[row*1536],src[row*256],q);
    }
    template<uint32_t Slot>
    __aicore__ inline void RefineLoadB96(LocalTensor<half> src) {
        static_assert(Slot<3);
        auto dst=bbuf.Get<half>()[Slot*9216];uint8_t pad[4]={};
        SetFmatrix(1,96,pad,FmatrixMode::FMATRIX_LEFT);
        LoadData3DParamsV2Pro z{};z.channelSize=96;z.extConfig=((uint64_t)96<<16)|96;
        LoadData<half>(dst,src,z);
    }
    template<uint32_t AS,uint32_t BS,bool Add>
    __aicore__ inline void RefineProduct96(LocalTensor<float> dst) {
        MmadParams m{};m.m=96;m.n=96;m.k=96;m.cmatrixInitVal=!Add;
        Mmad(dst,abuf.Get<half>()[AS*9216],bbuf.Get<half>()[BS*9216],m);
    }
    __aicore__ inline void RefineStore96(LocalTensor<half> dst,LocalTensor<float> src) {
        FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=96;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float,CFG_NZ>(dst,src,f);
    }
    __aicore__ inline void RefineInitial96() {
        auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];
        auto y=cbuf.Get<float>(),tmp=y[9216];
        RefineLoadA96<1>(eye);
        RefineLoadB96<2>(two);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,2,false>(tmp);
        RefineLoadA96<0>(g);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,false>(y);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(x,y);
        RefineLoadB96<0>(eye);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,0,false>(tmp);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        RefineLoadA96<1>(low);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,1,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,0,false>(y);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,true>(y);
    }
    __aicore__ inline void RefineSecondSeed96() {
        auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];
        auto y=cbuf.Get<float>(),tmp=y[9216];
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(x,y);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadA96<2>(x);
        RefineLoadB96<2>(minus);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,true>(y);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(two,y);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
    }
    __aicore__ inline void RefineSecondCorrection96() {
        auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];
        auto y=cbuf.Get<float>(),tmp=y[9216];
        RefineLoadA96<2>(eye);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,0,false>(tmp);
        RefineLoadB96<1>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,1,true>(tmp);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,1,true>(tmp);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(two);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<0,2,true>(tmp);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,2,true>(tmp);
        SetFlag<HardEvent::M_FIX>(3);WaitFlag<HardEvent::M_FIX>(3);
        RefineStore96(r,tmp);
        SetFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(3);
        RefineLoadA96<2>(x);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,0,false>(y);
        RefineLoadA96<1>(two);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,0,true>(y);
        SetFlag<HardEvent::FIX_MTE1>(3);WaitFlag<HardEvent::FIX_MTE1>(3);
        RefineLoadB96<2>(r);
        SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
        PipeBarrier<PIPE_M>();
        RefineProduct96<2,2,true>(y);
        PipeBarrier<PIPE_M>();
        RefineProduct96<1,2,true>(y);
    }
    template<bool Second>
    __aicore__ __attribute__((noinline)) void RefineResident96() {
        RefineInitial96();
        if constexpr(Second) {
            RefineSecondSeed96();
            RefineSecondCorrection96();
        }
        PipeBarrier<PIPE_ALL>();
    }
    template<bool Second=false>
    __aicore__ inline void Newton96(GlobalTensor<half> packed,GlobalTensor<float> result,GlobalTensor<float> defect,uint32_t batches,uint32_t stride,uint32_t steps) {
        auto g=l1buf.Get<half>(),low=g[9216],x=g[18432],eye=g[27648],two=g[36864],r=g[46080],minus=g[55296];auto y=cbuf.Get<float>(),tmp=y[9216];
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=GetBlockNum()) {
            Nd2NzParams p{};p.ndNum=1;p.nValue=96;p.dValue=96;p.srcDValue=96;p.dstNzC0Stride=96;p.dstNzNStride=1;
            auto src=packed[batch*stride*2];DataCopy(g,src,p);DataCopy(low,src[36864],p);DataCopy(x,src[9216],p);DataCopy(eye,src[18432],p);DataCopy(two,src[27648],p);DataCopy(r,src[46080],p);if constexpr(Second)DataCopy(minus,src[55296],p);PipeBarrier<PIPE_ALL>();
            Product96(x,eye,y);
            for(uint32_t iter=0;iter+1<steps;++iter) {
                NeumannPair96(x,r,y,tmp,iter+2<steps);
            }
            RefineResident96<Second>();
            FixpipeParamsV220 f{};f.mSize=96;f.nSize=96;f.srcStride=96;f.dstStride=128;
            Fixpipe<float,float,CFG_ROW_MAJOR>(result[batch*stride],y,f);Fixpipe<float,float,CFG_ROW_MAJOR>(defect[batch*stride],tmp,f);PipeBarrier<PIPE_ALL>();
        }
    }


};

} // namespace SolveDetail
} // namespace AscendC
