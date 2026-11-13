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

#ifndef RUBY0451_SOLVE_COLUMN_MEMORY_H
#define RUBY0451_SOLVE_COLUMN_MEMORY_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::TransposeColumnGroup(uint32_t group,uint32_t dstColumn,uint32_t col){
            for(uint32_t part=0;part<group;part+=8) {
                uint64_t sourceAddresses[16],targetAddresses[16];
                #pragma unroll
                for(uint32_t i=0;i<16;++i) {
                    sourceAddresses[i]=reinterpret_cast<uint64_t>(input[i*group+part].GetPhyAddr());
                    targetAddresses[i]=reinterpret_cast<uint64_t>(mat[(dstColumn+col+part+i/2)*cp+(i%2)*8].GetPhyAddr());
                }
                TransDataTo5HD<float>(targetAddresses,sourceAddresses,
                                      TransDataTo5HDParams{false,false,uint8_t(na/16),2,uint8_t(2*group)});
            }
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<HardEvent event> __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::Fence(){
        const event_t id=static_cast<event_t>(0);
        SetFlag<event>(id);WaitFlag<event>(id);
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::CreateVecIndex(LocalTensor<int32_t> target,int32_t first,uint32_t count){
        uint32_t seed=count<8?count:8;
        for(uint32_t i=0;i<seed;++i)target.SetValue(i,first+int32_t(i));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<count;filled*=2) {
            uint32_t len=count-filled<filled?count-filled:filled;
            Adds(target[filled],target,int32_t(filled),len);PipeBarrier<PIPE_V>();
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR work,const LinalgSolveTilingData& t){
        n=STATIC_N?STATIC_N:t.n;k=STATIC_K?STATIC_K:t.k;na=(n+7)/8*8;batches=t.batches;cores=t.cores;
        logicalN=PARTIAL_CAPACITY?t.n:n;logicalK=PARTIAL_CAPACITY?t.k:k;
        parallel=STATIC_WORKERS?true:t.resident==3;workers=STATIC_WORKERS?STATIC_WORKERS:parallel?t.cores/t.batches:1;
        cp=parallel?na+8+(FAMILY==1 && (na/8)%2==1?8:0):na;
        firstResident=t.grouped;initialResident=firstResident;broadcast=t.broadcastB;
        a.SetGlobalBuffer((__gm__ float*)A);b.SetGlobalBuffer((__gm__ float*)B);x.SetGlobalBuffer((__gm__ float*)X);
        w.SetGlobalBuffer((__gm__ float*)work+(parallel?uint64_t(GetBlockIdx()/workers)*na*((FAMILY==1 || (FAMILY==3 && STATIC_N==256))?2*n+(FAMILY==1?48:((STATIC_WORKERS==10 && STATIC_K==3)?41:40)):n+40):uint64_t(GetBlockIdx())*na*(n+k)));
        if constexpr(FAMILY==1 || (FAMILY==3 && STATIC_N==256)) {
            publication.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
                const_cast<__gm__ float*>(w.GetPhyAddr()))+(2*n+32)*na);
        }
        uint32_t matrixCols=parallel?(n<=200?192:144):n+k-firstResident;if(matrixCols<(parallel?144u:72u))matrixCols=parallel?144:72;
        if constexpr(FAMILY==3){if(parallel && n<=200)matrixCols=n<144?144:n;}
        spillGroup=n>=224?16:8;
        SetMaskNorm();SetAtomicNone();
        uint32_t offset=0;
        mat=LocalTensor<float>(TPosition::VECCALC,offset,na*matrixCols);offset+=na*matrixCols*4;
        uint32_t inputRows=parallel?(FAMILY==3?12:16):spillGroup;
        uint32_t inputSize=inputRows*na;
        if constexpr(FAMILY==3 && STATIC_N>0 && STATIC_N%16==0)inputSize=16*cp;
        input=LocalTensor<float>(TPosition::VECCALC,offset,inputSize);offset+=inputSize*4;
        transposed=LocalTensor<float>(TPosition::VECCALC,offset,8*na);offset+=8*na*4;
        indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,8*cp);offset+=8*cp*4;
        baseFactor=LocalTensor<float>(TPosition::VECCALC,offset,na+8);offset+=(na+8)*4;
        factor=baseFactor;
        column=LocalTensor<float>(TPosition::VECCALC,offset,na);offset+=na*4;
        order=LocalTensor<uint32_t>(TPosition::VECCALC,offset,na);offset+=na*4;
        output=LocalTensor<float>(TPosition::VECCALC,offset,n*8);offset+=n*8*4;
        inverse=LocalTensor<float>(TPosition::VECCALC,offset,na);
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::LoadMatrix(GlobalTensor<float> source,uint64_t offset,uint32_t cols,uint32_t firstColumn){
        const uint32_t stride=(cols+7)/8*8;
        auto sequence=input.ReinterpretCast<int32_t>();
        auto columnIndex=transposed.ReinterpretCast<int32_t>();
        auto offsets=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(sequence,int32_t(0),cols*8);PipeBarrier<PIPE_V>();
        ShiftRight(columnIndex,sequence,int32_t(3),cols*8);PipeBarrier<PIPE_V>();
        Muls(offsets,columnIndex,int32_t(8),cols*8);PipeBarrier<PIPE_V>();
        Sub(offsets,sequence,offsets,cols*8);PipeBarrier<PIPE_V>();
        Muls(offsets,offsets,int32_t(stride*4),cols*8);
        Muls(columnIndex,columnIndex,int32_t(4),cols*8);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,columnIndex,cols*8);PipeBarrier<PIPE_V>();
        for(uint32_t base=0;base<n;base+=8) {
            uint32_t rows=n-base<8?n-base:8;
            Duplicate(input,0.0f,8*stride);Fence<HardEvent::V_MTE2>();
            DataCopyExtParams cp{static_cast<uint16_t>(rows),cols*4,0,0,0};
            DataCopyPadExtParams<float> pad{false,0,0,0};
            DataCopyPad(input,source[offset+uint64_t(base)*cols],cp,pad);
            Fence<HardEvent::MTE2_V>();
            Gather(transposed,input,indices,0,cols*8);
            uint32_t spilled=firstResident>firstColumn?firstResident-firstColumn:0;
            if(spilled>cols) spilled=cols;
            if(spilled<cols) {
                DataCopyParams tp{static_cast<uint16_t>(cols-spilled),1,0,static_cast<uint16_t>((na-8)/8)};
                PipeBarrier<PIPE_V>();
                DataCopy(mat[(firstColumn+spilled-firstResident)*na+base],transposed[spilled*8],tp);
                PipeBarrier<PIPE_V>();
            }
            if(spilled) {
                DataCopyParams tp{static_cast<uint16_t>(spilled),1,0,static_cast<uint16_t>((na-8)/8)};
                Fence<HardEvent::V_MTE3>();DataCopy(w[firstColumn*na+base],transposed,tp);
                Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
            }
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline LocalTensor<float> KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ReadColumn(uint32_t j){
        if(j>=firstResident) return mat[(j-firstResident)*na];
        Fence<HardEvent::V_MTE2>();DataCopy(column,w[j*na],na);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        return column;
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::WriteColumn(uint32_t j,LocalTensor<float> value){
        if(j<firstResident) {
            Fence<HardEvent::V_MTE3>();DataCopy(w[j*na],value,na);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::LoadRange(GlobalTensor<float> source,uint64_t offset,uint32_t stride,uint32_t count,uint32_t dstColumn){
        if constexpr(FAMILY==3 && STATIC_N>0 && STATIC_N%16==0) {
            LoadAlignedWideRange(source,offset,stride,count,dstColumn);return;
        }
        for(uint32_t col=0;col<count;col+=8) {
            uint32_t cols=count-col<8?count-col:8;
            Duplicate(input,0.0f,8*cp);Fence<HardEvent::V_MTE2>();
            DataCopyPad(input,source[offset+col],
                        DataCopyExtParams{static_cast<uint16_t>(n),cols*4,(stride-cols)*4,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});
            Fence<HardEvent::MTE2_V>();
            uint64_t sourceAddresses[16],targetAddresses[16];
            #pragma unroll
            for(uint32_t i=0;i<16;++i) {
                sourceAddresses[i]=reinterpret_cast<uint64_t>(input[i*8].GetPhyAddr());
                targetAddresses[i]=reinterpret_cast<uint64_t>(mat[(dstColumn+col+i/2)*cp+(i%2)*8].GetPhyAddr());
            }
            TransDataTo5HD<float>(targetAddresses,sourceAddresses,
                                  TransDataTo5HDParams{false,false,uint8_t((na+15)/16),2,16});
            PipeBarrier<PIPE_V>();
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::LoadAlignedWideRange(GlobalTensor<float> source,uint64_t offset,
                                               uint32_t stride,uint32_t count,uint32_t dstColumn){
        for(uint32_t col=0;col<count;) {
            uint32_t group=count-col>=16?16:8;
            uint32_t cols=count-col<group?count-col:group;
            if(cols<group)Duplicate(input,0.0f,group*cp);
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(input,source[offset+col],
                        DataCopyExtParams{static_cast<uint16_t>(n),cols*4,(stride-cols)*4,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});
            Fence<HardEvent::MTE2_V>();
    TransposeColumnGroup(group,dstColumn,col);
            PipeBarrier<PIPE_V>();col+=cols;
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::LoadCapacityRange(GlobalTensor<float> source,uint64_t offset,
                                            uint32_t stride,uint32_t count,uint32_t dstColumn,
                                            uint32_t validCols,uint32_t firstDiagonal){
        if(logicalN==n && validCols==count) {
            LoadAlignedWideRange(source,offset,stride,count,dstColumn);return;
        }
        for(uint32_t col=0;col<count;) {
            uint32_t group=count-col>=16?16:8;
            uint32_t cols=col<validCols?(validCols-col<group?validCols-col:group):0;
            Duplicate(input,0.0f,group*cp);Fence<HardEvent::V_MTE2>();
            if(cols) {
                DataCopyPad(input,source[offset+col],
                    DataCopyExtParams{uint16_t(logicalN),cols*4,(stride-cols)*4,
                                      (group-(cols+7)/8*8)/8,0},
                    DataCopyPadExtParams<float>{true,0,uint8_t((8-cols%8)%8),0});
                Fence<HardEvent::MTE2_V>();
            } else PipeBarrier<PIPE_V>();
    TransposeColumnGroup(group,dstColumn,col);
            PipeBarrier<PIPE_V>();col+=group;
        }
        if(firstDiagonal<n && validCols<count) {
            Fence<HardEvent::V_S>();
            for(uint32_t col=validCols;col<count;++col)
                mat.SetValue((dstColumn+col)*cp+firstDiagonal+col,4.0f);
            Fence<HardEvent::S_V>();
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::StoreCooperativeOutput(uint32_t batch){
        if(n%8==0) {
            Duplicate(indices,uint32_t(((1u<<k)-1)*0x01010101u),8);PipeBarrier<PIPE_V>();
            GatherMask(mat,output,indices.ReinterpretCast<float>(),uint8_t(0),
                       GatherMaskParams{1,static_cast<uint16_t>(n/8),8,0});
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(x[uint64_t(batch)*n*k],mat,DataCopyExtParams{1,n*k*4,0,0,0});
        } else {
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(x[uint64_t(batch)*n*k],output,DataCopyExtParams{static_cast<uint16_t>(n),k*4,0,0,0});
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::PrepareCompactThreeIndices(){
        // Initial solution uses output[0:3*na]; immutable compact gather indices
        // occupy output[3*na:6*na], within the existing8*n allocation.
        auto offsets=output[3*na].ReinterpretCast<int32_t>();
        #pragma unroll
        for(uint32_t i=0;i<24;++i)offsets.SetValue(i,int32_t(((i%3)*na+i/3)*4));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=24;filled<3*na;filled*=2) {
            Adds(offsets[filled],offsets,int32_t((filled/3)*4),filled);
            PipeBarrier<PIPE_V>();
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::StoreCompactThreeOutput(uint32_t batch){
        Add(transposed,output,transposed,3*na);PipeBarrier<PIPE_V>();
        if(logicalK<k) {
            // Reuse the dead initial-solution buffer for dense output indices.
            auto offsets=output.ReinterpretCast<uint32_t>();
            Fence<HardEvent::V_S>();
            for(uint32_t row=0;row<logicalN;++row)
                for(uint32_t q=0;q<logicalK;++q)offsets.SetValue(row*logicalK+q,(q*na+row)*4);
            Fence<HardEvent::S_V>();
            Gather(mat,transposed,offsets,0,logicalN*logicalK);
        } else Gather(mat,transposed,output[3*na].ReinterpretCast<uint32_t>(),0,3*logicalN);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(x[uint64_t(batch)*logicalN*logicalK],mat,DataCopyExtParams{1,logicalN*logicalK*4,0,0,0});
    }

#endif
