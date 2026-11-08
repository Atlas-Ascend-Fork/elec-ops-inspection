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
// Single-RHS 16-row capacity: contiguous input, column elimination, deferred scaling.
class TinyColumns {
    template<uint32_t Count>
    __aicore__ inline void StaticIndex(LocalTensor<int32_t> dst) {
        static_assert(Count==24 || Count==256);
        // Match dav_c220 CreateVecIndexCalc for these two aligned counts.
        // Its initial eight scalar stores need S->V, but no TPipe lookup.
        for(int32_t i=0;i<8;++i)dst.SetValue(i,static_cast<float>(i));
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        UnaryRepeatParams params;
        constexpr uint32_t first=Count<64?Count:64;
        for(uint32_t i=0;i<first/8-1;++i){
            Adds(dst[(i+1)*8],dst[i*8],(int32_t)8,(uint64_t)8,1,params);PipeBarrier<PIPE_V>();
        }
        if constexpr(Count>64)for(uint32_t i=0;i<Count/64-1;++i){
            Adds(dst[(i+1)*64],dst[i*64],(int32_t)64,(uint64_t)64,1,params);PipeBarrier<PIPE_V>();
        }
    }
public:
    static constexpr uint32_t P=16,W=24;
    struct ColumnWork {
        LinalgSolveTilingData t;
        LocalTensor<float> mat,input,pivot,candidates,maximum,scales,weights,output;
        LocalTensor<int32_t> offsets,transposeIx,tempIx;
        LocalTensor<uint32_t> outputIx;
        LocalTensor<uint64_t> transposeSrc,transposeDst;
        GlobalTensor<float> ga,gb,gx;
    };
    __aicore__ inline void Initialize(ColumnWork &state,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx) {
        constexpr uint32_t vs=0;
        state.mat=LocalTensor<float>(TPosition::VECCALC,0,2048);
        state.input=state.mat[P*W];state.pivot=state.input[P*P];state.candidates=state.pivot[W];state.maximum=state.candidates[P];
        state.scales=state.maximum[64];state.weights=state.scales[P];state.offsets=state.weights[8*W].ReinterpretCast<int32_t>();
        state.transposeIx=state.offsets[W];state.tempIx=state.transposeIx[P*P];state.output=state.tempIx[P*P].ReinterpretCast<float>();state.outputIx=state.output[P].ReinterpretCast<uint32_t>();
        state.ga.SetGlobalBuffer((__gm__ float*)aa);state.gb.SetGlobalBuffer((__gm__ float*)bb);state.gx.SetGlobalBuffer((__gm__ float*)xx);
        state.transposeSrc=state.transposeIx.ReinterpretCast<uint64_t>();
        state.transposeDst=state.transposeIx[32].ReinterpretCast<uint64_t>();
        if(state.t.n==16) {
            constexpr uint64_t broadcast4=0x0001000100010001ULL;
            const uint64_t srcBase=((uint64_t)state.input.GetPhyAddr()>>5)&0x1fff;
            const uint64_t dstBase=((uint64_t)state.mat.GetPhyAddr()>>5)&0x1fff;
            for(uint32_t q=0;q<4;++q) {
                state.transposeSrc.SetValue(q,(srcBase+q*8)*broadcast4+(2ULL<<16)+(4ULL<<32)+(6ULL<<48));
                state.transposeDst.SetValue(q,(dstBase+q*4)*broadcast4+(1ULL<<16)+(2ULL<<32)+(3ULL<<48));
            }
            SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        } else {
        StaticIndex<P*P>(state.transposeIx);PipeBarrier<PIPE_V>();ShiftRight(state.tempIx,state.transposeIx,(int32_t)4,P*P);PipeBarrier<PIPE_V>();
        Muls(state.transposeIx,state.transposeIx,(int32_t)state.t.n,P*P);Muls(state.tempIx,state.tempIx,(int32_t)(state.t.n*P-1),P*P);PipeBarrier<PIPE_V>();Sub(state.transposeIx,state.transposeIx,state.tempIx,P*P);PipeBarrier<PIPE_V>();Muls(state.transposeIx,state.transposeIx,(int32_t)4,P*P);PipeBarrier<PIPE_V>();
        }
        StaticIndex<W>(state.offsets);PipeBarrier<PIPE_V>();Muls(state.offsets,state.offsets,(int32_t)(P*4),W);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadBatch(ColumnWork &state,uint32_t batch) {
        constexpr uint32_t vs=0;
            Duplicate(state.input,0.0f,P*P);PipeBarrier<PIPE_ALL>();
            DataCopyPad(state.input,state.ga[(uint64_t)batch*state.t.n*state.t.n],DataCopyExtParams{1,state.t.n*state.t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            DataCopyPad(state.mat[state.t.n*P],state.gb[(uint64_t)(state.t.broadcastB?0:batch)*state.t.n],DataCopyExtParams{1,state.t.n*4,0,0,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-state.t.n%8)%8),0});PipeBarrier<PIPE_ALL>();
            if(state.t.n==16) {
                TransDataTo5HDParams params;params.dstHighHalf=false;params.srcHighHalf=false;
                params.repeatTimes=2;params.dstRepStride=P;params.srcRepStride=1;
                TransDataTo5HDVldVaRegImpl<float>((__ubuf__ uint64_t*)state.transposeDst.GetPhyAddr(),
                    (__ubuf__ uint64_t*)state.transposeSrc.GetPhyAddr(),params);PipeBarrier<PIPE_V>();
            } else {
                Gather(state.mat,state.input,state.transposeIx.ReinterpretCast<uint32_t>(),0,state.t.n*P);PipeBarrier<PIPE_V>();
            }
    }
    __aicore__ inline void Eliminate(ColumnWork &state,uint64_t (&liveMask)[2]) {
        constexpr uint32_t vs=0;
            for(uint32_t j=0;j+3<state.t.n;++j) {
                auto col=state.mat[j*P];Abs(state.candidates,col,P);PipeBarrier<PIPE_V>();
                WholeReduceMax(state.maximum,state.candidates,liveMask,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t bits,row;GetReduceRepeatMaxMinSpr<float>(bits,row);
                const float inverse=1.0f/col.GetValue(row);liveMask[0]&=~(1ULL<<row);
                state.outputIx.SetValue(j,(state.t.n*P+row)*4);state.scales.SetValue(j,inverse);
                uint64_t pivotMask[2]={1ULL<<(row%8),0};Duplicate(col[(row/8)*8],0.0f,pivotMask,1,1,8);
                const uint32_t width=state.t.n-j;
                Gather(state.pivot,state.mat,state.offsets.ReinterpretCast<uint32_t>(),((j+1)*P+row)*4,width);PipeBarrier<PIPE_V>();
                Muls(state.pivot,state.pivot,-inverse,width);PipeBarrier<PIPE_V>();Brcb(state.weights,state.pivot,(width+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                MulAddDst(state.mat[(j+1)*P],col,state.weights,(uint64_t)P,(uint8_t)width,BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
            }
    }
    __aicore__ inline void SolveTail(ColumnWork &state,uint64_t (&liveMask)[2]) {
        constexpr uint32_t vs=0;
                const uint32_t j=state.t.n-3;
                auto col0=state.mat[j*P],col1=state.mat[(j+1)*P],col2=state.mat[(j+2)*P],rhs=state.mat[state.t.n*P];
                SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint64_t tailMask=liveMask[0];
                uint32_t row0=63-static_cast<uint32_t>(CountLeadingZero(tailMask));tailMask^=1ULL<<row0;
                uint32_t row1=63-static_cast<uint32_t>(CountLeadingZero(tailMask));tailMask^=1ULL<<row1;
                uint32_t row2=63-static_cast<uint32_t>(CountLeadingZero(tailMask));
                float a0=col0.GetValue(row0),a1=col0.GetValue(row1),a2=col0.GetValue(row2);
                float mag0=a0<0?-a0:a0,mag1=a1<0?-a1:a1,mag2=a2<0?-a2:a2;
                if(mag1>mag0 || (mag1==mag0 && row1<row0)) {
                    const uint32_t r=row0;row0=row1;row1=r;
                    const float v=a0;a0=a1;a1=v;mag0=mag1;
                }
                if(mag2>mag0 || (mag2==mag0 && row2<row0)) {
                    const uint32_t r=row0;row0=row2;row2=r;
                    const float v=a0;a0=a2;a2=v;
                }
                const float b0=col1.GetValue(row0),c0=col2.GetValue(row0),y0=rhs.GetValue(row0);
                float b1=col1.GetValue(row1),c1=col2.GetValue(row1),y1=rhs.GetValue(row1);
                float b2=col1.GetValue(row2),c2=col2.GetValue(row2),y2=rhs.GetValue(row2);
                const float inverse0=1.0f/a0,upperB=b0*inverse0,upperC=c0*inverse0,first=y0*inverse0;
                b1-=a1*upperB;c1-=a1*upperC;y1-=a1*first;
                b2-=a2*upperB;c2-=a2*upperC;y2-=a2*first;
                const float magB1=b1<0?-b1:b1,magB2=b2<0?-b2:b2;
                if(magB2>magB1 || (magB2==magB1 && row2<row1)) {
                    const float b=b1,c=c1,y=y1;b1=b2;c1=c2;y1=y2;b2=b;c2=c;y2=y;
                }
                const float inverse1=1.0f/b1,upper=c1*inverse1,middle=y1*inverse1;
                const float diagonal=c2-b2*upper,second=y2-b2*middle;
                const float x2=second/diagonal,x1=middle-upper*x2,x0=first-upperB*x1-upperC*x2;
                Axpy(rhs,col0,-x0,P);PipeBarrier<PIPE_V>();
                Axpy(rhs,col1,-x1,P);PipeBarrier<PIPE_V>();
                Axpy(rhs,col2,-x2,P);PipeBarrier<PIPE_V>();
                constexpr uint32_t unitOffset=P*W+P*P+W+P+64+P;
                state.weights.SetValue(0,1.0f);
                state.outputIx.SetValue(j,unitOffset*4);state.outputIx.SetValue(j+1,unitOffset*4);state.outputIx.SetValue(j+2,unitOffset*4);
                state.scales.SetValue(j,x0);state.scales.SetValue(j+1,x1);state.scales.SetValue(j+2,x2);
                SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
    }
    __aicore__ inline void StoreBatch(ColumnWork &state,uint32_t batch) {
        constexpr uint32_t vs=0;
            PipeBarrier<PIPE_ALL>();Gather(state.output,state.mat,state.outputIx,0,state.t.n);PipeBarrier<PIPE_V>();Mul(state.output,state.output,state.scales,state.t.n);PipeBarrier<PIPE_ALL>();
            DataCopyPad(state.gx[(uint64_t)batch*state.t.n],state.output,DataCopyExtParams{1,state.t.n*4,0,0,0});PipeBarrier<PIPE_ALL>();
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        ColumnWork state;state.t=t;
        Initialize(state,aa,bb,xx);
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            LoadBatch(state,batch);
            uint64_t liveMask[2]={(1ULL<<t.n)-1,0};
            Eliminate(state,liveMask);
            SolveTail(state,liveMask);
            StoreBatch(state,batch);
        }
        PipeBarrier<PIPE_ALL>();
    }

};
class SmallColumns {
    template<uint32_t Count>
    __aicore__ inline void StaticIndex(LocalTensor<int32_t> dst) {
        static_assert(Count==64 || Count==72);
        for(int32_t i=0;i<8;++i)dst.SetValue(i,i);
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        UnaryRepeatParams params;
        for(uint32_t i=1;i<8;++i){
            Adds(dst[i*8],dst[(i-1)*8],(int32_t)8,(uint64_t)8,1,params);PipeBarrier<PIPE_V>();
        }
        if constexpr(Count>64){
            Adds(dst[64],dst,(int32_t)64,(uint64_t)8,1,params);PipeBarrier<PIPE_V>();
        }
    }
public:
    static constexpr uint32_t P=64,W=72,L=P+8;
    struct ColumnWork {
        LinalgSolveTilingData t;
        LocalTensor<float> mat,input,rhs,pivot,candidates,maximum,scales,weights,output,rawInput;
        LocalTensor<int32_t> colIx,offsets,address,transposeIx,tempIx;
        LocalTensor<uint32_t> outputIx;
        LocalTensor<uint64_t> transposeSrc,transposeDst;
        GlobalTensor<float> ga,gb,gx;
        TransDataTo5HDParams transposeParams;
    };
    __aicore__ inline void Initialize(ColumnWork &state,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx) {
        constexpr uint32_t vs=0;
        state.mat=LocalTensor<float>(TPosition::VECCALC,0,24000);
        state.input=state.mat[(P+8)*W];state.rhs=state.input[P*P];state.pivot=state.rhs[P*8];state.colIx=state.pivot[W].ReinterpretCast<int32_t>();
        state.candidates=state.colIx[P].ReinterpretCast<float>();state.maximum=state.candidates[P];state.scales=state.maximum[64];state.weights=state.scales[P];
        state.offsets=state.weights[8*W].ReinterpretCast<int32_t>();state.address=state.offsets[W];
        state.transposeIx=state.address[W];state.tempIx=state.transposeIx[P*P];state.output=state.tempIx[P*P].ReinterpretCast<float>();state.outputIx=state.output[P*8].ReinterpretCast<uint32_t>();
        state.ga.SetGlobalBuffer((__gm__ float*)aa);state.gb.SetGlobalBuffer((__gm__ float*)bb);state.gx.SetGlobalBuffer((__gm__ float*)xx);
        // Four 16-bit block addresses per word, as required by CANN 9.0 dav_c220.
        // This region holds packed A only on the mutually exclusive unaligned path.
        state.transposeSrc=state.transposeIx.ReinterpretCast<uint64_t>();
        state.transposeDst=state.transposeIx[32].ReinterpretCast<uint64_t>();
        if(state.t.n%8==0) {
            constexpr uint64_t broadcast4=0x0001000100010001ULL;
            const uint64_t srcBase=((uint64_t)state.input.GetPhyAddr()>>5)&0x1fff;
            const uint64_t dstBase=((uint64_t)state.mat.GetPhyAddr()>>5)&0x1fff;
            for(uint32_t group=0;group<4;++group) {
                for(uint32_t q=0;q<4;++q) {
                    state.transposeSrc.SetValue(group*4+q,(srcBase+group*128+q*32)*broadcast4+
                        (8ULL<<16)+(16ULL<<32)+(24ULL<<48));
                    state.transposeDst.SetValue(group*4+q,(dstBase+group*2+q*18)*broadcast4+
                        (1ULL<<16)+(9ULL<<32)+(10ULL<<48));
                }
            }
            constexpr uint32_t transposeSv=0;
            SetFlag<HardEvent::S_V>(transposeSv);WaitFlag<HardEvent::S_V>(transposeSv);
        }

    }
    __aicore__ inline void PrepareIndices(ColumnWork &state) {
        constexpr uint32_t vs=0;
        state.transposeParams.dstHighHalf=false;state.transposeParams.srcHighHalf=false;
        state.transposeParams.repeatTimes=(state.t.n+7)/8;
        state.transposeParams.dstRepStride=L;state.transposeParams.srcRepStride=1;
        StaticIndex<W>(state.offsets);DataCopy(state.colIx,state.offsets,P);PipeBarrier<PIPE_V>();
        // The old transpose-index scratch is free with the native transpose.
        state.rawInput=state.transposeIx.ReinterpretCast<float>();
        if(state.t.n%8) {
            auto rows=state.input.ReinterpretCast<int32_t>();
            auto columns=state.weights[8*P].ReinterpretCast<int32_t>();
            Muls(rows,state.colIx,(int32_t)(state.t.n*4),P);
            Muls(columns,state.colIx,(int32_t)4,P);PipeBarrier<PIPE_V>();
            Brcb(state.weights.ReinterpretCast<int32_t>(),columns,P/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Add(state.tempIx,rows,state.weights.ReinterpretCast<int32_t>(),(uint64_t)P,(uint8_t)state.t.n,BinaryRepeatParams{1,1,0,P/8,0,1});PipeBarrier<PIPE_V>();
        }
        Muls(state.offsets,state.offsets,(int32_t)(L*4),W);Muls(state.colIx,state.colIx,(int32_t)(state.t.k*4),P);PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadBatch(ColumnWork &state,uint32_t batch) {
        constexpr uint32_t vs=0;
            Duplicate(state.mat,0.0f,(P+8)*W);if(state.t.n%8==0)Duplicate(state.input,0.0f,P*P);Duplicate(state.rhs,0.0f,P*8);Duplicate(state.scales,1.0f,P);PipeBarrier<PIPE_ALL>();
            if(state.t.n%8) {
                DataCopyPad(state.rawInput,state.ga[(uint64_t)batch*state.t.n*state.t.n],DataCopyExtParams{1,state.t.n*state.t.n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
            } else {
            DataCopyPad(state.input,state.ga[(uint64_t)batch*state.t.n*state.t.n],DataCopyExtParams{(uint16_t)state.t.n,state.t.n*4,0,(P-(state.t.n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-state.t.n%8)%8),0});
            }
            DataCopyPad(state.rhs,state.gb[(uint64_t)(state.t.broadcastB?0:batch)*state.t.n*state.t.k],DataCopyExtParams{1,state.t.n*state.t.k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
            if(state.t.n%8) {
                Gather(state.mat,state.rawInput,state.tempIx.ReinterpretCast<uint32_t>(),0,(uint64_t)state.t.n,(uint8_t)state.t.n,(uint16_t)(L/8));PipeBarrier<PIPE_V>();
            } else {
                for(uint32_t group=0;group*16<state.t.n;++group) {
                    TransDataTo5HDVldVaRegImpl<float>(
                        (__ubuf__ uint64_t*)state.transposeDst[group*4].GetPhyAddr(),
                        (__ubuf__ uint64_t*)state.transposeSrc[group*4].GetPhyAddr(),state.transposeParams);
                    PipeBarrier<PIPE_V>();
                }
            }
            for(uint32_t k=0;k<state.t.k;++k)Gather(state.mat[(state.t.n+k)*L],state.rhs,state.colIx.ReinterpretCast<uint32_t>(),k*4,P);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Eliminate(ColumnWork &state,uint64_t (&liveMask)[2],uint32_t tail) {
        constexpr uint32_t vs=0;
            for(uint32_t j=0;j+tail<state.t.n;++j) {
                auto col=state.mat[j*L];Abs(state.candidates,col,P);PipeBarrier<PIPE_V>();
                WholeReduceMax(state.maximum,state.candidates,liveMask,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t bits,row;GetReduceRepeatMaxMinSpr<float>(bits,row);
                const float inverse=1.0f/col.GetValue(row);liveMask[0]&=~(1ULL<<row);
                uint64_t pivotMask[2]={1ULL<<(row%8),0};
                Duplicate(col[(row/8)*8],0.0f,pivotMask,1,1,8);
                Duplicate(state.scales[(row/8)*8],inverse,pivotMask,1,1,8);
                for(uint32_t k=0;k<state.t.k;++k)state.outputIx.SetValue(j*state.t.k+k,((state.t.n+k)*L+row)*4);
                PipeBarrier<PIPE_V>();
                const uint32_t width=state.t.n+state.t.k-j-1;
                const uint32_t firstColumn=(j+1)/8*8,gatheredWidth=state.t.n+state.t.k-firstColumn;
                Gather(state.pivot,state.mat,state.offsets.ReinterpretCast<uint32_t>(),(firstColumn*L+row)*4,gatheredWidth);PipeBarrier<PIPE_V>();
                Muls(state.pivot,state.pivot,-inverse,gatheredWidth);PipeBarrier<PIPE_V>();
                // Aligned eight-column groups stay within the72 reserved columns.
                // Consumers use only the original live columns and rows0..63.
                Brcb(state.mat[firstColumn*L+P],state.pivot,(gatheredWidth+7)/8,BrcbRepeatParams{L/8,L});PipeBarrier<PIPE_V>();
                MulAddDst(state.mat[(j+1)*L],col,state.mat[(j+1)*L+P],(uint64_t)P,(uint8_t)width,BinaryRepeatParams{1,1,0,L/8,0,L/8});PipeBarrier<PIPE_V>();
            }
    }
    __aicore__ inline void SolveTail(ColumnWork &state,uint64_t (&liveMask)[2]) {
        constexpr uint32_t vs=0;
                const uint32_t j=state.t.n-2;
                auto col0=state.mat[j*L],col1=state.mat[(j+1)*L];
                SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t row0=63-static_cast<uint32_t>(CountLeadingZero(liveMask[0]));
                uint32_t row1=63-static_cast<uint32_t>(CountLeadingZero(liveMask[0]^(1ULL<<row0)));
                float a0=col0.GetValue(row0),a1=col0.GetValue(row1);
                const float mag0=a0<0?-a0:a0,mag1=a1<0?-a1:a1;
                if(mag1>mag0 || (mag1==mag0 && row1<row0)) {
                    const uint32_t r=row0;row0=row1;row1=r;
                    const float v=a0;a0=a1;a1=v;
                }
                const float b0=col1.GetValue(row0),b1=col1.GetValue(row1);
                const float inverse0=1.0f/a0,upper=b0*inverse0;
                const float inverse1=1.0f/(b1-a1*upper);
                // Reuse the now-dead broadcast region for already-solved values.
                // Final Gather reads these directly, outside the scaled RHS area.
                constexpr uint32_t valueOffset=(P+8)*W+P*P+P*8+W+P+P+64+P;
                for(uint32_t k=0;k<state.t.k;++k) {
                    auto rhsCol=state.mat[(state.t.n+k)*L];
                    const float first=rhsCol.GetValue(row0)*inverse0;
                    const float second=rhsCol.GetValue(row1)-a1*first;
                    const float x1=second*inverse1,x0=first-upper*x1;
                    Axpy(rhsCol,col0,-x0,P);PipeBarrier<PIPE_V>();
                    Axpy(rhsCol,col1,-x1,P);PipeBarrier<PIPE_V>();
                    state.weights.SetValue(k,x0);state.weights.SetValue(8+k,x1);
                    state.outputIx.SetValue(j*state.t.k+k,(valueOffset+k)*4);
                    state.outputIx.SetValue((j+1)*state.t.k+k,(valueOffset+8+k)*4);
                }
                constexpr uint32_t sv=0;
                SetFlag<HardEvent::S_V>(sv);WaitFlag<HardEvent::S_V>(sv);
    }
    __aicore__ inline void StoreBatch(ColumnWork &state,uint32_t batch) {
        constexpr uint32_t vs=0;
            for(uint32_t k=0;k<state.t.k;++k)Mul(state.mat[(state.t.n+k)*L],state.mat[(state.t.n+k)*L],state.scales,P);PipeBarrier<PIPE_ALL>();
            Gather(state.output,state.mat,state.outputIx,0,state.t.n*state.t.k);PipeBarrier<PIPE_ALL>();DataCopyPad(state.gx[(uint64_t)batch*state.t.n*state.t.k],state.output,DataCopyExtParams{1,state.t.n*state.t.k*4,0,0,0});PipeBarrier<PIPE_ALL>();
    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        ColumnWork state;state.t=t;
        Initialize(state,aa,bb,xx);
        PrepareIndices(state);
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            LoadBatch(state,batch);
            uint64_t liveMask[2]={t.n==64?~0ULL:((1ULL<<t.n)-1),0};
            const uint32_t tail=t.k<=2?2:0;
            Eliminate(state,liveMask,tail);
            if(tail)SolveTail(state,liveMask);
            StoreBatch(state,batch);
        }
        PipeBarrier<PIPE_ALL>();
    }

};

template<uint32_t P> class SmallSquare {
    TPipe pipe;TBuf<TPosition::VECCALC> buf;
    __aicore__ inline void LoadRhs(LocalTensor<float> mat,LocalTensor<float> rhsInput,LocalTensor<float> rhsExpanded,
            LocalTensor<int32_t> rhsIndices,LocalTensor<int32_t> rhsRows,GlobalTensor<float> gb,uint32_t batch,const LinalgSolveTilingData &t) {
        const uint32_t n=t.n,k=t.k,p=P,q=(k+7)/8*8;
            if(k<=8) {
                Duplicate(rhsInput,0.0f,8*n+8);PipeBarrier<PIPE_ALL>();
                DataCopyPad(rhsInput,gb[(uint64_t)(t.broadcastB?0:batch)*n*k],DataCopyExtParams{1,n*k*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                if(k==1) {
                    PipeBarrier<PIPE_ALL>();Brcb(rhsExpanded,rhsInput,(n+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                } else {
                CreateVecIndex(rhsIndices,(int32_t)0,8*n);PipeBarrier<PIPE_V>();
                ShiftRight(rhsRows,rhsIndices,(int32_t)3,8*n);PipeBarrier<PIPE_V>();
                Muls(rhsRows,rhsRows,(int32_t)(4*(8-k)),8*n);Muls(rhsIndices,rhsIndices,(int32_t)4,8*n);PipeBarrier<PIPE_V>();
                Sub(rhsIndices,rhsIndices,rhsRows,8*n);PipeBarrier<PIPE_ALL>();
                Gather(rhsExpanded,rhsInput,rhsIndices.ReinterpretCast<uint32_t>(),0,8*n);PipeBarrier<PIPE_V>();
                }
                DataCopy(mat[p],rhsExpanded,DataCopyParams{(uint16_t)n,1,0,(uint16_t)(p/8)});
            } else {
            DataCopyPad(mat[p],gb[(uint64_t)(t.broadcastB?0:batch)*n*k],DataCopyExtParams{(uint16_t)n,k*4,0,p/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)(q-k),0});
            }
    }
public:
    __aicore__ inline void Run(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR workspace,const LinalgSolveTilingData &t) {
        const uint32_t n=t.n,k=t.k,p=P,q=(k+7)/8*8,L=P+q;
        // Keep the measured scratch spacing and initialization; compacting it
        // regressed the small-matrix path in the paired experiments.
        const uint32_t oldCap=n*L+L+22*p+64+n*q;pipe.InitBuffer(buf,(oldCap+24*n+8+8*((n+7)/8*8))*4);
        auto mat=buf.Get<float>(),pivot=mat[n*L],col=pivot[L],scratchGap=col[p],candidates=scratchGap[p],weights=candidates[p];
        auto tmp=weights[p*8],scalar=tmp[2*p];auto indices=tmp[p].ReinterpretCast<uint32_t>();
        auto rhsInput=mat[oldCap],rhsExpanded=rhsInput[24*n+8];auto rhsIndices=rhsInput[8*n+8].ReinterpretCast<int32_t>(),rhsRows=rhsIndices[8*n];
        auto result=scalar[64];auto outputIndices=result[n*q].ReinterpretCast<uint32_t>();
        const auto vs=pipe.FetchEventID(HardEvent::V_S);
        GlobalTensor<float> ga,gb,gx;
        ga.SetGlobalBuffer((__gm__ float*)a);gb.SetGlobalBuffer((__gm__ float*)b);
        gx.SetGlobalBuffer((__gm__ float*)x);
        for(uint32_t i=0;i<n;++i)indices.SetValue(i,i*L*4);PipeBarrier<PIPE_ALL>();
        for(uint32_t batch=GetBlockIdx();batch<t.batches;batch+=t.blocks) {
            uint32_t labels[64];uint64_t liveMask[2]={n==64?~0ULL:((1ULL<<n)-1),0};
            Duplicate(mat,0.0f,n*L);Duplicate(scratchGap,1.0f,p);Duplicate(col,0.0f,p);PipeBarrier<PIPE_ALL>();
            DataCopyPad(mat,ga[(uint64_t)batch*n*n],DataCopyExtParams{(uint16_t)n,n*4,0,(L-(n+7)/8*8)/8,0},DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
            LoadRhs(mat,rhsInput,rhsExpanded,rhsIndices,rhsRows,gb,batch,t);
            PipeBarrier<PIPE_ALL>();
            for(uint32_t j=0;j<n;++j) {
                Gather(col,mat[j],indices,0,n);PipeBarrier<PIPE_V>();
                Abs(candidates,col,n);PipeBarrier<PIPE_V>();
                WholeReduceMax(tmp,candidates,liveMask,1,1,1,8);SetFlag<HardEvent::V_S>(vs);WaitFlag<HardEvent::V_S>(vs);
                uint32_t maximumBits,row;GetReduceRepeatMaxMinSpr<float>(maximumBits,row);
                if(k>8)labels[row]=j;float inv=1.0f/col.GetValue(row);
                liveMask[0]&=~(1ULL<<row);
                const uint32_t first=j/8*8;
                // Include the pivot row in the update, then restore it. No scalar
                // UB store or S_V handoff is needed inside the pivot loop.
                Muls(pivot,mat[row*L+first],-inv,L-first);
                Brcb(weights,col,p/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                BinaryRepeatParams update{1,1,0,(uint8_t)(L/8),0,1};
                for(uint32_t off=first;off<L;off+=64)
                    MulAddDst(mat[off],pivot[off-first],weights,(uint64_t)(L-off<64?L-off:64),(uint8_t)n,update);
                PipeBarrier<PIPE_V>();
                Muls(mat[row*L+first],pivot,-1.0f,L-first);PipeBarrier<PIPE_V>();
                if(k<=8)for(uint32_t rhs=0;rhs<k;++rhs)outputIndices.SetValue(j*k+rhs,(row*L+p+rhs)*4);
            }
            PipeBarrier<PIPE_ALL>();
            if(k<=8) {
                Gather(result,mat,outputIndices,0,n*k);PipeBarrier<PIPE_ALL>();
                DataCopyPad(gx[(uint64_t)batch*n*k],result,DataCopyExtParams{1,n*k*4,0,0,0});
            } else {
                for(uint32_t row=0;row<n;++row)
                    DataCopyPad(gx[(uint64_t)batch*n*k+labels[row]*k],mat[row*L+p],DataCopyExtParams{1,k*4,0,0,0});
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
};

} // namespace SolveDetail
} // namespace AscendC
