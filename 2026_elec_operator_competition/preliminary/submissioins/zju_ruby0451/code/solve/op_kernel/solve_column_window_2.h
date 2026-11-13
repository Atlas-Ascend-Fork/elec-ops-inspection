/*
 * Copyright (c) 2026 ruby0451 contributors.
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

#ifndef RUBY0451_SOLVE_COLUMN_WINDOW_2_H
#define RUBY0451_SOLVE_COLUMN_WINDOW_2_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<uint32_t PREFIX>
    __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::FactorCollapsedTail(){
        constexpr uint32_t rows=256-PREFIX,pitch=rows+8;
        // PREFIX192: (64+8)*72*4=20736B. Existing scratch and metadata remain disjoint.
        auto matrix=LocalTensor<float>(TPosition::VECCALC,0,(rows+8)*pitch);
        auto absolute=indices[2*na].ReinterpretCast<float>();
        auto reduced=indices[3*na].ReinterpretCast<float>();
        auto pivot=indices[4*na].ReinterpretCast<float>();
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(matrix,w[PREFIX*na+PREFIX],
            DataCopyExtParams{uint16_t(rows),rows*4,PREFIX*4,1,0},
            DataCopyPadExtParams<float>{false,0,0,0});
        DataCopyPad(matrix[rows*pitch],w[(n+8)*na+PREFIX],
            DataCopyExtParams{uint16_t(k),rows*4,PREFIX*4,1,0},
            DataCopyPadExtParams<float>{false,0,0,0});
        Fence<HardEvent::MTE2_V>();
        CreateVecIndex(baseIndex,0,rows+8);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(pitch*4),rows+8);PipeBarrier<PIPE_V>();
        uint64_t active[2]={0,0};
        for(uint32_t p=PREFIX;p<n;++p) {
            uint32_t rowBase=p/64*64,firstRow=rowBase-PREFIX;
            if(p%64==0)active[0]=~uint64_t(0);
            auto f=matrix[(p-PREFIX)*pitch];
            if constexpr(rows==64) {
                if(p==PREFIX)Fence<HardEvent::V_S>();
                else WaitFlag<HardEvent::V_S>(static_cast<event_t>(1));
            } else Fence<HardEvent::V_S>();
            uint64_t firstBit=active[0]&(~active[0]+1);
            uint32_t row=rowBase+63-static_cast<uint32_t>(CountLeadingZero(firstBit));
            float diagonal=reinterpret_cast<__ubuf__ float*>(f.GetPhyAddr())[row-PREFIX];
    ChooseCollapsedPivot<PREFIX>(absolute,reduced,f,active,rowBase,firstRow,row,diagonal);
            // This column's first update is complete;its zero can become visible
            // while Gather and scalar reciprocal/metadata are issued.
            f.SetValue(row-PREFIX,0.0f);
            SetFlag<HardEvent::S_V>(static_cast<event_t>(1));
            // Pivot column is ready. Join the prior later-column updates on V
            // before Gather reads them and Brcb overwrites their coefficient scratch.
            if constexpr(rows==64)PipeBarrier<PIPE_V>();
            uint32_t cols=n+k-p-1;
            auto trailing=matrix[(p+1-PREFIX)*pitch];
            Gather(pivot,trailing,indices,(row-PREFIX)*4,cols);PipeBarrier<PIPE_V>();
            Brcb(transposed,pivot,(cols+7)/8,BrcbRepeatParams{1,8});
            float inv=1.0f/diagonal;
            inverse.SetValue(p,inv);order.SetValue(p,row);
            active[0]&=~(uint64_t(1)<<(row-rowBase));
            WaitFlag<HardEvent::S_V>(static_cast<event_t>(1));
    UpdateCollapsedColumns<rows,pitch>(p,cols,firstRow,inv,f,trailing);
        }
    StoreCollapsedFactors<PREFIX,rows,pitch>(matrix);
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::UpdateColumnsWindowReady(LocalTensor<float> tile,uint32_t cols,
                                                     uint32_t row,uint32_t readyBase){
        auto pivot=indices[4*na].ReinterpretCast<float>();
        Gather(pivot,tile,indices,row*4,cols);PipeBarrier<PIPE_V>();
        Brcb(transposed,pivot,(cols+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        // Only the next pivot window is split; the remaining bands keep their full fanout.
        MulAddDst(tile[readyBase],factor[readyBase],transposed,64,1,
                  BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        SetFlag<HardEvent::V_S>(static_cast<event_t>(3));
        MulAddDst(tile[cp+readyBase],factor[readyBase],transposed[8],64,cols-1,
                  BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        SetFlag<HardEvent::V_S>(static_cast<event_t>(4));
        for(uint32_t base=row/64*64;base<na;base+=64) {
            if(base==readyBase)continue;
            MulAddDst(tile[base],factor[base],transposed,64,cols,
                      BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        }
    }

#endif
