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

#ifndef RUBY0451_SOLVE_COLUMN_FACTOR_H
#define RUBY0451_SOLVE_COLUMN_FACTOR_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::StoreUnpivotedFactors(uint32_t begin,uint32_t end,uint32_t localCols,uint32_t lane,uint32_t rhsLane){
        if constexpr(persistent) {
            // Every nonempty receive prefix is 8-column aligned, using both slots.
            if(begin>0) {
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(2));
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(3));
            }
        }
        // Canonical pair composition overwrites the last published source columns.
        if constexpr(persistent)Fence<HardEvent::MTE3_V>();
        factor=baseFactor;
        if constexpr(FAMILY==1) {
            if(k<=workers) {
                // Compose retired factors only; initial RHS and factorization stay sequential.
                Fence<HardEvent::V_S>();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na);
                for(uint32_t p=begin;p+1<end;p+=2) {
                    auto left=mat[(p-begin)*cp];
                    float cross=reinterpret_cast<__ubuf__ float*>(left.GetPhyAddr())[p+1];
                    Axpy<float,float,false>(left,mat[(p+1-begin)*cp],cross,uint64_t(0),1,
                                            UnaryRepeatParams{1,1,8,8});
                }
                ResetMask();SetMaskNorm();
            }
        }
        {Fence<HardEvent::V_MTE3>();DataCopyPad(w[begin*na],mat,DataCopyExtParams{static_cast<uint16_t>(localCols),na*4,(cp-na)/8,0,0});}
        if(lane==rhsLane) {
            if constexpr(FAMILY==1) {
                // All prior owners' final flags were observed. Join this owner's
                // own inverse store before overwriting its UB source with the
                // complete vector. The first solution join precedes residual
                // reuse of the same GM metadata region.
                Fence<HardEvent::MTE3_MTE2>();
                DataCopy(inverse,w[(n+8)*na],na);
                Fence<HardEvent::MTE2_V>();
            }
            Assemble(mat[localCols*cp],output,false,cp);
        }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ReceiveUnpivotedPanel(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t end,uint32_t lane,uint32_t rhsLane,uint32_t localCols,uint32_t packet,uint32_t steps,bool owner){
            if(!owner){
                if constexpr(persistent) {
                    // Receives are the contiguous prefix [0,begin). Owners
                    // then retire, and the RHS owner is the final owner.
                    if(panel>=8)
                        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(2+((panel/4)&1u)));
                    else if(panel==0)Fence<HardEvent::V_MTE2>();
                }
                DataCopy(input[(persistent?panel%8:0)*cp],w[packet],steps*cp);Fence<HardEvent::MTE2_V>();
            }
            if(!persistent && lane==rhsLane) {
                uint32_t metadataBegin=persistent?panel/8*8:panel;
                auto packetView=owner?mat[(metadataBegin-begin)*cp]:input;
                Gather(inverse[metadataBegin],packetView,indices,na*4,panelEnd-metadataBegin);
                PipeBarrier<PIPE_V>();
            }
            for(uint32_t p=panel;p<panelEnd;++p) {
                factor=owner?mat[(p-begin)*cp]:input[((persistent?panel%8:0)+p-panel)*cp];
                uint32_t row=p;float inv=0.0f;
                uint32_t first=panelEnd>begin?panelEnd:begin;
                uint32_t rhsCols=lane==rhsLane?k:0;
                if(first<end)UpdateColumns(mat[(first-begin)*cp],end-first+rhsCols,row,inv);
                else if(rhsCols)UpdateColumns(mat[localCols*cp],rhsCols,row,inv);
            }
            if constexpr(persistent) {
                if(!owner)SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(2+((panel/4)&1u)));
            }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::NormalizeUnpivotedOwner(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t packet,uint32_t steps){
                Fence<HardEvent::S_V>();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na);
                for(uint32_t p=panel;p<panelEnd;++p) {
                    factor=mat[(p-begin)*cp];
                    uint32_t row=p;
                    Fence<HardEvent::V_S>();
                    float diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                    if constexpr(persistent) {
                        factor.SetValue(row,0.0f);
                        SetFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    }
                    float inv=1.0f/diagonal;
                    auto packetView=factor;
                    packetView.SetValue(na,inv);
                    if constexpr(persistent)WaitFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    else {factor.SetValue(row,0.0f);Fence<HardEvent::S_V>();}
                    NormalizeOwnedPanel(mat[(p+1-begin)*cp],panelEnd-p-1,row,inv);
                }
                ResetMask();SetMaskNorm();
                // Deferred inverse stores feed the owner's Gather and packet DMA.
                if constexpr(persistent)Fence<HardEvent::S_V>();
                Fence<HardEvent::V_MTE3>();DataCopy(w[packet],mat[(panel-begin)*cp],steps*cp);

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::FactorPaired(){
        if constexpr(FAMILY==2 || FAMILY==3)FactorWindow();
        else if constexpr(FAMILY==1)FactorUnpivoted();
        else {if(n>=224)FactorWindow();else FactorUnpivoted();}
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::FactorUnpivoted(){
        constexpr bool persistent=FAMILY==1;
        if constexpr(persistent)InitializePublication();
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers,split=(n+31)/32*8;
        uint32_t rhsLane=FAMILY==1?workers-1:0;
        uint32_t blocks=(n+7)/8;
        uint32_t begin=workers==4?lane*split:(blocks*lane/workers)*8;
        uint32_t end=workers==4?begin+split:(blocks*(lane+1)/workers)*8;if(end>n)end=n;
        uint32_t localCols=end-begin;
        {LoadRange(a,uint64_t(batch)*n*n+begin,n,localCols,0);
        if(lane==rhsLane)LoadRange(b,broadcast?0:uint64_t(batch)*n*k,k,k,localCols);}
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(baseIndex,int32_t(0),72);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(cp*4),72);
        CreateVecIndex(order.ReinterpretCast<int32_t>(),int32_t(0),na);
        PipeBarrier<PIPE_V>();
        constexpr uint32_t panelWidth=FAMILY==1?4:8;
        for(uint32_t panel=0;panel<n;panel+=panelWidth) {
            factor=baseFactor;
            uint32_t panelEnd=n-panel<panelWidth?n:panel+panelWidth,steps=panelEnd-panel;
            bool owner=panel>=begin && panel<end;
            uint32_t packet=(n+16)*na+(persistent?panel:((panel/8)%2)*8)*cp;
            if(owner) {
    NormalizeUnpivotedOwner<persistent>(panel,panelEnd,begin,packet,steps);
            }
            if constexpr(persistent) {
                // Each packet has private GM storage through the final join;
                // immutable readiness slots avoid all-matrix rendezvous.
                if(owner) {
    PublishUnpivotedOwner(panel,panelEnd,end,begin,localCols);
                }
                if(end<=panel && lane!=rhsLane)continue;
                if(!owner)WaitPersistentPanel(panel);
            } else SyncAll();
            if constexpr(FAMILY==1) {
                if(end<=panel && lane!=rhsLane)continue;
            }
    ReceiveUnpivotedPanel<persistent>(panel,panelEnd,begin,end,lane,rhsLane,localCols,packet,steps,owner);
        }
    StoreUnpivotedFactors<persistent>(begin,end,localCols,lane,rhsLane);
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::NormalizeOwnedPanelEarly(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv){
        uint32_t start=row/64*64;
        Muls<float,false>(factor[start],factor[start],-inv,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        if(cols) {
            float first=reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr())[row];
            Axpy<float,float,false>(tile[start],factor[start],first,uint64_t(0),1,
                                    UnaryRepeatParams{1,1,8,8});
            SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
            for(uint32_t col=1;col<cols;++col) {
                float coefficient=reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr())[col*cp+row];
                Axpy<float,float,false>(tile[col*cp+start],factor[start],coefficient,uint64_t(0),1,
                                        UnaryRepeatParams{1,1,8,8});
            }
            SetFlag<HardEvent::V_S>(static_cast<event_t>(2));
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::NormalizeOwnedPanel(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv){
        // The pivot-selection V_S already completed all preceding tile writes.
        // Factor normalization touches only the retired column; tile is disjoint.
        // The owner panel retains count mode, restoring it after pivot reductions.
        uint32_t start=FAMILY==3?row/64*64:0;
        Muls<float,false>(factor[start],factor[start],-inv,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
        PipeBarrier<PIPE_V>();
        for(uint32_t col=0;col<cols;++col) {
            float coefficient=reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr())[col*cp+row];
            Axpy<float,float,false>(tile[col*cp+start],factor[start],coefficient,uint64_t(0),1,
                                    UnaryRepeatParams{1,1,8,8});
        }
        // FAMILY1 joins all updates at the next pivot V_S or final packet V_MTE3.
        if constexpr(FAMILY!=1)PipeBarrier<PIPE_V>();
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::UpdateColumns(LocalTensor<float> tile,uint32_t cols,uint32_t row,float inv){
        auto pivot=indices[4*na].ReinterpretCast<float>();
        Gather(pivot,tile,indices,row*4,cols);PipeBarrier<PIPE_V>();
        if(!parallel){Muls(pivot,pivot,-inv,cols);PipeBarrier<PIPE_V>();}
        Brcb(transposed,pivot,(cols+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        if constexpr(FAMILY==1) {
            for(uint32_t col=0;col<cols;col+=8) {
                uint32_t count=cols-col<8?cols-col:8;
                MulAddDst(tile[col*cp],factor,transposed[col*8],count*8,na/8,
                          BinaryRepeatParams{static_cast<uint8_t>(cp/8),0,1,1,1,0});
            }
            PipeBarrier<PIPE_V>();return;
        }
        for(uint32_t base=FAMILY==3?row/64*64:0;base<na;base+=64) {
            uint32_t count=na-base<64?na-base:64;
            MulAddDst(tile[base],factor[base],transposed,count,cols,
                      BinaryRepeatParams{1,1,0,static_cast<uint8_t>(cp/8),0,1});
        }
        PipeBarrier<PIPE_V>();
    }

#endif
