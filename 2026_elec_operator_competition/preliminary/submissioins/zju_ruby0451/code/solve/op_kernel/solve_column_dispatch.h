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

#ifndef RUBY0451_SOLVE_COLUMN_DISPATCH_H
#define RUBY0451_SOLVE_COLUMN_DISPATCH_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::~KernelColumnSolve(){PipeBarrier<PIPE_ALL>();}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::Process(){
        if(parallel){ProcessCooperative();return;}
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=cores) {
            firstResident=initialResident;
            LoadMatrix(a,uint64_t(batch)*n*n,n,0);
            LoadMatrix(b,broadcast?0:uint64_t(batch)*n*k,k,n);
            auto active=indices[na].ReinterpretCast<float>();
            auto absolute=indices[2*na].ReinterpretCast<float>();
            auto reduced=indices[3*na].ReinterpretCast<float>();
            auto baseIndex=indices.ReinterpretCast<int32_t>();
            CreateVecIndex(baseIndex,int32_t(0),64);PipeBarrier<PIPE_V>();
            Muls(baseIndex,baseIndex,int32_t(na*4),64);
            Duplicate(active,0.0f,na);PipeBarrier<PIPE_V>();Duplicate(active,1.0f,n);
            for(uint32_t p=0;p<n;++p) {
                auto current=ReadColumn(p);
                Fence<HardEvent::S_V>();Adds(factor,current,0.0f,na);PipeBarrier<PIPE_V>();
                Abs(absolute,factor,na);PipeBarrier<PIPE_V>();
                Mul(absolute,absolute,active,na);PipeBarrier<PIPE_V>();
                WholeReduceMax(reduced,absolute,64,na/64,4,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                if(na%64) WholeReduceMax(reduced[(na/64)*8],absolute[(na/64)*64],na%64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                Fence<HardEvent::V_S>();
                float largest=-1.0f;uint32_t pivotRow=0;
                for(uint32_t base=0;base<na;base+=64) {
                    float value=reduced.GetValue(base/8);
                    if(value>largest){largest=value;pivotRow=base+reduced.ReinterpretCast<uint32_t>().GetValue(base/8+1);}
                }
                order.SetValue(p,pivotRow);active.SetValue(pivotRow,0.0f);
                float inv=1.0f/factor.GetValue(pivotRow);inverse.SetValue(p,inv);
                factor.SetValue(pivotRow,0.0f);Fence<HardEvent::S_V>();
                for(uint32_t j=p+1;j<firstResident;j+=spillGroup) {
                    uint32_t cols=firstResident-j<spillGroup?firstResident-j:spillGroup;
                    Fence<HardEvent::V_MTE2>();DataCopy(input,w[j*na],cols*na);Fence<HardEvent::MTE2_V>();
                    UpdateColumns(input,cols,pivotRow,inv);
                    Fence<HardEvent::V_MTE3>();DataCopy(w[j*na],input,cols*na);
                    Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
                }
                uint32_t first=p+1>firstResident?p+1:firstResident;
                for(uint32_t j=first;j<n+k;j+=64) {
                    uint32_t cols=n+k-j<64?n+k-j:64;
                    UpdateColumns(mat[(j-firstResident)*na],cols,pivotRow,inv);
                }
            }
            Fence<HardEvent::V_S>();
            Assemble(mat[(n-firstResident)*na],output,false);
            Refine(batch);
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams cp{static_cast<uint16_t>(n),k*4,0,0,0};
            DataCopyPad(x[uint64_t(batch)*n*k],output,cp);Fence<HardEvent::MTE3_S>();
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ProcessCooperative(){
        if constexpr(FAMILY==3){ProcessBlocked();return;}
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers;
        uint32_t rhsLane=FAMILY==1?workers-1:0;
        FactorPaired();
        if(lane==rhsLane){Fence<HardEvent::V_MTE3>();DataCopy(w[n*na],output,n*8);}
        SyncAll();
        if(lane!=rhsLane){DataCopy(output,w[n*na],n*8);Fence<HardEvent::MTE2_V>();}
        firstResident=n+k;
        auto xt=input;auto rhs=transposed;
        Duplicate(xt,0.0f,k*na);
        CreateVecIndex(indices.ReinterpretCast<int32_t>(),int32_t(0),na);PipeBarrier<PIPE_V>();
        Muls(indices.ReinterpretCast<int32_t>(),indices.ReinterpretCast<int32_t>(),int32_t(32),na);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<k;++q)Gather(xt[q*na],output,indices,q*4,n);
        PipeBarrier<PIPE_V>();Duplicate(rhs,0.0f,k*na);PipeBarrier<PIPE_V>();
        PrepareResidualIndices();
        uint32_t tileRows=n>=224?32:16;
        for(uint32_t base=lane*tileRows;base<n;base+=workers*tileRows) {
            uint32_t rows=n-base<tileRows?n-base:tileRows;
            ResidualTile(batch,base,rows,tileRows);Fence<HardEvent::V_MTE3>();
            for(uint32_t q=0;q<k;++q)
                DataCopyPad(w[(n+8+q)*na+base],rhs[q*na+base],DataCopyExtParams{1,rows*4,0,0,0});
            Fence<HardEvent::MTE3_V>();
        }
        SyncAll();
        if(k<=workers) {
            if(lane<k) {
                DataCopy(rhs,w[(n+8+lane)*na],na);Fence<HardEvent::MTE2_V>();
                CorrectSingleRhs();Fence<HardEvent::V_MTE3>();
                DataCopy(w[(n+8+lane)*na],rhs,na);
            }
            SyncAll();
        }
        if(lane==rhsLane) {
            DataCopy(rhs,w[(n+8)*na],k*na);Fence<HardEvent::MTE2_V>();
            if(k<=workers)Assemble(rhs,mat,true);else ApplyCorrection();
            StoreCooperativeOutput(batch);
            Fence<HardEvent::MTE3_S>();
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ProcessBlocked(){
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers;
        constexpr bool localJoins=FAMILY==3 && STATIC_N==256 && (STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3;
        FactorPaired();
        if constexpr(localJoins) {
            if(lane==0)Fence<HardEvent::MTE3_MTE2>();
            else if(lane<k)WaitPersistentPanel(192);
        } else SyncAll();
        if constexpr(FAMILY==3 && STATIC_N==256) {
            if constexpr((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) {
                // Prefix metadata was reloaded before tail generation; the tail
                // filled the remaining64 entries. Order scalar stores before V.
                if(lane==0)Fence<HardEvent::S_V>();
            }
            if(lane<k && (!((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) || lane!=0)) {
                DataCopy(inverse,w[(n+16)*na],na);
                DataCopy(order.ReinterpretCast<float>(),w[(n+17)*na],na);
                Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
            }
        }
        if(lane<k) {
            DataCopy(transposed,w[(n+8+lane)*na],na);Fence<HardEvent::MTE2_V>();
            BacksolveBlocked();Fence<HardEvent::V_MTE3>();
            DataCopy(w[(n+8+lane)*na],transposed,na);
        }
        SyncAll();
        DataCopy(input,w[(n+8)*na],k*na);Fence<HardEvent::MTE2_V>();
        if(lane==0) {
            if constexpr(localJoins) {DataCopy(output,input,k*na);PipeBarrier<PIPE_V>();}
            else Assemble(input,output,false,na,false,false);
        }
    ComputeBlockedResidual(batch,lane);
        SyncAll();
    StoreBlockedCorrection<localJoins>(batch,lane);
    }

#endif
