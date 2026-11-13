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

#ifndef RUBY0451_SOLVE_COLUMN_CORRECTION_H
#define RUBY0451_SOLVE_COLUMN_CORRECTION_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::CorrectRhsBlock(LocalTensor<float> rhs,uint32_t block,uint32_t stop,uint32_t slot){
            for(uint32_t p=block;p<stop;) {
                if constexpr(FAMILY==1) {
                    Fence<HardEvent::V_S>();float first=rhs.GetValue(p);
                    float second=p+1<stop?rhs.GetValue(p+1):0.0f;
                    Axpy<float,float,false>(rhs,mat[(slot*64+p-block)*na],first,uint64_t(0),1,
                                            UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    if(p+1<stop) {
                        Axpy<float,float,false>(rhs,mat[(slot*64+p+1-block)*na],second,uint64_t(0),1,
                                                UnaryRepeatParams{1,1,8,8});
                        PipeBarrier<PIPE_V>();
                    }
                    p+=2;
                } else {
                uint32_t row;
                if constexpr(FAMILY==2)row=order.GetValue(p);
                else row=n<224?p:order.GetValue(p);
                Fence<HardEvent::V_S>();float value=rhs.GetValue(row);
                Axpy<float,float,false>(rhs,mat[(slot*64+p-block)*na],value,uint64_t(0),1,
                                        UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                ++p;
                }
            }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool localJoins>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::StoreBlockedCorrection(uint32_t batch,uint32_t lane){
        if(lane<k) {
            DataCopy(transposed,w[(n+8+lane)*na],na);Fence<HardEvent::MTE2_V>();
            ForwardBlocked();BacksolveBlocked(true);
            if constexpr(localJoins) {
                // Residual scratch overwrote the original seed. Restore it after
                // correction; the existing V_MTE3 orders both result and seed.
                if(lane)Duplicate(mat[128*na].ReinterpretCast<uint32_t>(),uint32_t(1),8);
            }
            Fence<HardEvent::V_MTE3>();
            DataCopy(w[(n+8+lane)*na],transposed,na);
            if constexpr(localJoins) {
                if(lane)PublishPersistentPanel(196+lane*8);
            }
        }
        if constexpr(localJoins) {
            if(lane==0) {
                Fence<HardEvent::MTE3_MTE2>();
                WaitPersistentPanel(204);WaitPersistentPanel(212);
            }
        } else SyncAll();
        if(lane==0) {
            DataCopy(transposed,w[(n+8)*na],k*na);Fence<HardEvent::MTE2_V>();
            if constexpr(localJoins)StoreCompactThreeOutput(batch);
            else {
                Assemble(transposed,mat,true,na,false,false);
                StoreCooperativeOutput(batch);
            }
            Fence<HardEvent::MTE3_S>();
        }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ForwardBlocked(){
        auto rhs=transposed;
        bool cached=n<=200;
        Fence<HardEvent::V_MTE2>();DataCopy(mat,w,(cached?n:(n<64?n:64))*na);
        for(uint32_t block=0;block<n;block+=64) {
            uint32_t stop=block+64<n?block+64:n,slot=(block/64)&1u;
            Fence<HardEvent::MTE2_V>();
            if(!cached && stop<n) {
                uint32_t count=n-stop<64?n-stop:64;
                Fence<HardEvent::V_MTE2>();DataCopy(mat[(slot^1u)*64*na],w[stop*na],count*na);
            }
            if(FAMILY==3 && STATIC_N==256 && block<192) {
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na-block);
                for(uint32_t p=block;p<stop;p+=4) {
                    uint32_t r0=order.GetValue(p),r1=order.GetValue(p+1);
                    uint32_t r2=order.GetValue(p+2),r3=order.GetValue(p+3);
                    Fence<HardEvent::V_S>();
                    float v0=rhs.GetValue(r0),v1=rhs.GetValue(r1);
                    float v2=rhs.GetValue(r2),v3=rhs.GetValue(r3);
                    auto f=mat[(slot*64+p-block)*na+block];
                    Axpy<float,float,false>(rhs[block],f,v0,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    Axpy<float,float,false>(rhs[block],f[na],v1,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    Axpy<float,float,false>(rhs[block],f[2*na],v2,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                    Axpy<float,float,false>(rhs[block],f[3*na],v3,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                }
            } else for(uint32_t p=block;p<stop;p+=2) {
                uint32_t start=p/64*64;
                if(p%64==0){SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na-start);}
                uint32_t row=order.GetValue(p);
                uint32_t row1=p+1<stop?order.GetValue(p+1):row;
                Fence<HardEvent::V_S>();float value=rhs.GetValue(row);
                float value1=p+1<stop?rhs.GetValue(row1):0.0f;
                Axpy<float,float,false>(rhs[start],mat[(cached?p:slot*64+p-block)*na+start],value,uint64_t(0),1,
                                        UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                if(p+1<stop) {
                    Axpy<float,float,false>(rhs[start],mat[(cached?p+1:slot*64+p+1-block)*na+start],value1,uint64_t(0),1,
                                            UnaryRepeatParams{1,1,8,8});
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        ResetMask();SetMaskNorm();
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::BacksolveBlocked(bool coefficientsReady){
        auto rhs=transposed;
        bool cached=n<=200;
        if(cached && !coefficientsReady) {
            Fence<HardEvent::V_MTE2>();DataCopy(mat,w,n*na);Fence<HardEvent::MTE2_V>();
        }
        uint32_t last=(n-1)/64*64;
        if(!cached && !coefficientsReady) {
            uint32_t count=n-last,slot=(last/64)&1u;
            Fence<HardEvent::V_MTE2>();DataCopy(mat[slot*64*na],w[last*na],count*na);
        }
        for(int32_t block=int32_t(last);block>=0;block-=64) {
            uint32_t steps=n-uint32_t(block)<64?n-uint32_t(block):64;
            uint32_t slot=(uint32_t(block)/64)&1u;
            if(!cached) {
                if(block && !(coefficientsReady && uint32_t(block)+64>=last))
                    Fence<HardEvent::MTE2_V>();
                // Forward replay leaves its final two coefficient blocks in these slots.
                if(block>64 && !(coefficientsReady && uint32_t(block)==last)) {
                    Fence<HardEvent::V_MTE2>();
                    DataCopy(mat[(slot^1u)*64*na],w[(uint32_t(block)-64)*na],64*na);
                }
            }
            Muls(indices.ReinterpretCast<int32_t>(),order[uint32_t(block)].ReinterpretCast<int32_t>(),int32_t(4),steps);
            PipeBarrier<PIPE_V>();Gather(column,rhs,indices,0,steps);PipeBarrier<PIPE_V>();
            Mul(column,column,inverse[uint32_t(block)],steps);PipeBarrier<PIPE_V>();
            Adds(baseFactor[uint32_t(block)],column,0.0f,steps);PipeBarrier<PIPE_V>();
            if(block) {
                // Retired upper columns are dead after this block's back-update.
                auto coefficients=mat[cached?uint32_t(block)*na:slot*64*na];
                Brcb(input,column,(steps+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                BinaryRepeatParams productParams{1,1,0,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),1};
                BinaryRepeatParams sumParams{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
                for(uint32_t base=0;base<uint32_t(block);base+=64) {
                    uint32_t count=uint32_t(block)-base<64?uint32_t(block)-base:64;
                    Mul(coefficients[base],coefficients[base],input,count,steps,productParams);
                }
                PipeBarrier<PIPE_V>();
                for(uint32_t remain=steps;remain>1;) {
                    uint32_t half=(remain+1)/2,countCols=remain-half;
                    for(uint32_t base=0;base<uint32_t(block);base+=64) {
                        uint32_t count=uint32_t(block)-base<64?uint32_t(block)-base:64;
                        Add(coefficients[base],coefficients[base],coefficients[half*na+base],count,countCols,sumParams);
                    }
                    PipeBarrier<PIPE_V>();remain=half;
                }
                Sub(rhs,rhs,coefficients,uint32_t(block));PipeBarrier<PIPE_V>();
            }
        }
        Adds(rhs,baseFactor,0.0f,n);PipeBarrier<PIPE_V>();factor=baseFactor;
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::AddWide(float term,float& sum,float& low){
        volatile float total=sum+term;
        volatile float z=total-sum;
        volatile float kept=total-z;
        volatile float left=sum-kept;
        volatile float right=term-z;
        volatile float error=left+right;
        low=low+error;sum=total;
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::Assemble(LocalTensor<float> rhs,LocalTensor<float> target,bool add,uint32_t rhsPitch,bool scale,bool useOrder){
        auto offsets=indices.ReinterpretCast<int32_t>();
        auto pattern=factor.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<8;++q)pattern.SetValue(q,(q<k?q:0)*(rhsPitch?rhsPitch:na)*4);
        Fence<HardEvent::S_V>();
        if(useOrder)Brcb(offsets.ReinterpretCast<float>(),order.ReinterpretCast<float>(),na/8,BrcbRepeatParams{1,8});
        else {
            CreateVecIndex(column.ReinterpretCast<int32_t>(),int32_t(0),na);PipeBarrier<PIPE_V>();
            Brcb(offsets.ReinterpretCast<float>(),column,na/8,BrcbRepeatParams{1,8});
        }
        PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(4),n*8);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,pattern,64,na/8,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
        Gather(target,rhs,indices,0,n*8);PipeBarrier<PIPE_V>();
        if(scale) {
            Brcb(input,inverse,na/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Mul(target,target,input,n*8);PipeBarrier<PIPE_V>();
        }
        if(add){Add(output,output,target,n*8);PipeBarrier<PIPE_V>();}
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::CorrectSingleRhs(){
        auto rhs=transposed;
        Fence<HardEvent::V_MTE2>();DataCopy(mat,w,(n<64?n:64)*na);
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,na);
        for(uint32_t block=0;block<n;block+=64) {
            uint32_t stop=block+64<n?block+64:n;
            uint32_t slot=(block/64)&1u;
            Fence<HardEvent::MTE2_V>();
            if(stop<n) {
                uint32_t count=n-stop<64?n-stop:64;
                Fence<HardEvent::V_MTE2>();
                DataCopy(mat[(slot^1u)*64*na],w[stop*na],count*na);
            }
    CorrectRhsBlock(rhs,block,stop,slot);
        }
        ResetMask();SetMaskNorm();
        factor=baseFactor;
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ApplyCorrection(){
        auto rhs=transposed;
        // Reuse the known inverse and update all RHS columns with one outer product.
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(baseIndex,int32_t(0),8);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(na*4),8);Fence<HardEvent::V_S>();
        auto pivot=indices[na].ReinterpretCast<float>();
        auto spread=indices[2*na].ReinterpretCast<float>();
        for(uint32_t block=0;block<n;block+=64) {
            uint32_t stop=block+64<n?block+64:n;
            if(parallel) {
                Fence<HardEvent::V_MTE2>();DataCopy(mat,w[block*na],(stop-block)*na);
                Fence<HardEvent::MTE2_V>();
            }
            for(uint32_t p=block;p<stop;++p) {
            uint32_t pivotRow=order.GetValue(p);float inv=inverse.GetValue(p);
            if(parallel) factor=mat[(p-block)*na];
            else {
                Fence<HardEvent::V_MTE2>();DataCopy(factor,w[p*na],na);
                Fence<HardEvent::MTE2_S>();
                factor.SetValue(pivotRow,0.0f);Fence<HardEvent::S_V>();
            }
            Gather(pivot,rhs,indices,pivotRow*4,k);PipeBarrier<PIPE_V>();
            if(!parallel){Muls(pivot,pivot,-inv,k);PipeBarrier<PIPE_V>();}
            Brcb(spread,pivot,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t base=0;base<na;base+=64) {
                uint32_t count=na-base<64?na-base:64;
                MulAddDst(rhs[base],factor[base],spread,count,k,
                          BinaryRepeatParams{1,1,0,static_cast<uint8_t>(na/8),0,1});
            }
            PipeBarrier<PIPE_V>();
            }
        }
        factor=baseFactor;
        Assemble(rhs,mat,true);
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::Refine(uint32_t batch){
        // Spill retained coefficients once, reusing the resident matrix UB for batched residuals.
        Fence<HardEvent::S_V>();
        Fence<HardEvent::V_MTE3>();DataCopy(w[firstResident*na],mat,(n+k-firstResident)*na);
        Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        uint32_t tileRows=n+k-firstResident>=144?16:8;
        firstResident=n+k;
        auto xt=input;auto rhs=transposed;
        Duplicate(xt,0.0f,k*na);
        CreateVecIndex(indices.ReinterpretCast<int32_t>(),int32_t(0),na);PipeBarrier<PIPE_V>();
        Muls(indices.ReinterpretCast<int32_t>(),indices.ReinterpretCast<int32_t>(),int32_t(32),na);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<k;++q) Gather(xt[q*na],output,indices,q*4,n);
        PipeBarrier<PIPE_V>();Duplicate(rhs,0.0f,k*na);PipeBarrier<PIPE_V>();
        for(uint32_t base=0;base<n;base+=tileRows)
            ResidualTile(batch,base,n-base<tileRows?n-base:tileRows,tileRows);
        ApplyCorrection();

    }

#endif
