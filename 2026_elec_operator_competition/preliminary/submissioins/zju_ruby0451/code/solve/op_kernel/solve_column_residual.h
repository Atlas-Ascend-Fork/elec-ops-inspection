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

#ifndef RUBY0451_SOLVE_COLUMN_RESIDUAL_H
#define RUBY0451_SOLVE_COLUMN_RESIDUAL_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool SPLIT>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualTileCompensated(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows){
        if(!parallel){ResidualTileOriginal(batch,base,rows,tileRows);return;}
        const uint32_t size=tileRows*na;
        auto sourceA=mat;auto sourceHigh=mat[size];auto sourceLow=mat[2*size];
        auto product=mat[3*size];auto low=mat[4*size];auto temp=mat[5*size];auto ah=mat[6*size];
        auto original=mat[7*size];auto al=mat[7*size+64];auto z=mat[7*size+128];
        auto total=mat[7*size+192];auto other=mat[7*size+256];
        auto xh=factor;auto xl=column;
        auto leftIndex=indices;auto rightIndex=indices[64];auto integerTemp=indices[128].ReinterpretCast<int32_t>();
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        auto savedB=input[8*na];
    LoadResidualRhs(savedB,boff,rows);
        {
            Duplicate(sourceA,0.0f,rows*na);Fence<HardEvent::V_MTE2>();
            DataCopyPad(sourceA,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            ShiftRight(sourceHigh.ReinterpretCast<int32_t>(),sourceA.ReinterpretCast<int32_t>(),int32_t(SPLIT?20:12),rows*na);PipeBarrier<PIPE_V>();
            ShiftLeft(sourceHigh.ReinterpretCast<int32_t>(),sourceHigh.ReinterpretCast<int32_t>(),int32_t(SPLIT?20:12),rows*na);PipeBarrier<PIPE_V>();
            Sub(sourceLow,sourceA,sourceHigh,rows*na);PipeBarrier<PIPE_V>();
        }
        // Reuse immutable gather tables and the complete RHS tile across its columns.
        for(uint32_t q=0;q<k;++q) {
            leftIndex=indices;rightIndex=indices[64];
    ComputeCompensatedProducts<SPLIT>(sourceA,sourceHigh,sourceLow,product,low,temp,ah,xh,xl,q,rows);
    ReduceResidualRows(product,low,original,ah,rows);
    StoreResidualRows(other,savedB,original,ah,al,z,total,temp,leftIndex,boff,rows,q,base);
        }
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool SPLIT>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ComputeCompensatedProducts(LocalTensor<float> sourceA,LocalTensor<float> sourceHigh,LocalTensor<float> sourceLow,LocalTensor<float> product,LocalTensor<float> low,LocalTensor<float> temp,LocalTensor<float> ah,LocalTensor<float> xh,LocalTensor<float> xl,uint32_t q,uint32_t rows){
if constexpr(SPLIT){
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(20),na);PipeBarrier<PIPE_V>();
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(20),na);PipeBarrier<PIPE_V>();
            Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            // HH uses four significant bits per operand. Keep both remaining
            // contributions as Al*Xh + A*Xl, including the low*low term.
            RowMul(product,sourceHigh,xh,rows);
            RowMul(low,sourceLow,xh,rows);
            RowMul(temp,sourceA,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);PipeBarrier<PIPE_V>();
}else{
            RowMul(product,sourceA,input[q*na],rows);
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            RowMul(low,sourceHigh,xh,rows);PipeBarrier<PIPE_V>();
            Sub(low,low,product,rows*na);RowMul(temp,sourceHigh,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,sourceLow,xh,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,sourceLow,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);PipeBarrier<PIPE_V>();
            // Split the rounded product before reductions; retain its low part and product error.
            ShiftRight(ah.ReinterpretCast<int32_t>(),product.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            ShiftLeft(ah.ReinterpretCast<int32_t>(),ah.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            Sub(temp,product,ah,rows*na);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);Adds(product,ah,0.0f,rows*na);PipeBarrier<PIPE_V>();
}
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ComputeJointProducts(LocalTensor<float> xh,LocalTensor<float> xl,LocalTensor<float> product,LocalTensor<float> sourceHigh,LocalTensor<float> sourceLow,LocalTensor<float> sourceA,LocalTensor<float> low,LocalTensor<float> temp,uint32_t rows,uint32_t size){
        for(uint32_t q=0;q<k;++q) {
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(20),na);
            PipeBarrier<PIPE_V>();
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(20),na);
            PipeBarrier<PIPE_V>();Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            RowMul(product[q*size],sourceHigh,xh,rows);
            RowMul(low[q*size],sourceLow,xh,rows);
            RowMul(temp,sourceA,xl,rows);PipeBarrier<PIPE_V>();
            Add(low[q*size],low[q*size],temp,size);PipeBarrier<PIPE_V>();
        }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ComputeOriginalProducts(LocalTensor<float> product,LocalTensor<float> original,LocalTensor<float> ah,LocalTensor<float> al,LocalTensor<float> xh,LocalTensor<float> xl,LocalTensor<float> low,LocalTensor<float> temp,uint32_t q,uint32_t rows){
            RowMul(product,original,input[q*na],rows);
            ShiftRight(ah.ReinterpretCast<int32_t>(),original.ReinterpretCast<int32_t>(),int32_t(12),rows*na);
            ShiftRight(xh.ReinterpretCast<int32_t>(),input[q*na].ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            ShiftLeft(ah.ReinterpretCast<int32_t>(),ah.ReinterpretCast<int32_t>(),int32_t(12),rows*na);
            ShiftLeft(xh.ReinterpretCast<int32_t>(),xh.ReinterpretCast<int32_t>(),int32_t(12),na);PipeBarrier<PIPE_V>();
            Sub(al,original,ah,rows*na);Sub(xl,input[q*na],xh,na);PipeBarrier<PIPE_V>();
            RowMul(low,ah,xh,rows);PipeBarrier<PIPE_V>();
            Sub(low,low,product,rows*na);RowMul(temp,ah,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,al,xh,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);RowMul(temp,al,xl,rows);PipeBarrier<PIPE_V>();
            Add(low,low,temp,rows*na);PipeBarrier<PIPE_V>();

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ReduceCompensatedRows(LocalTensor<float> product,LocalTensor<float> low,LocalTensor<float> total,LocalTensor<float> z,LocalTensor<float> temp,LocalTensor<float> other,uint32_t rows){
            BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            UnaryRepeatParams ur{1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            for(uint32_t remain=na;remain>8;) {
                uint32_t half=(remain+15)/16*8,count=remain-half;
                for(uint32_t off=0;off<count;off+=64) {
                    uint32_t len=count-off<64?count-off:64;
                    Add(total[off],product[off],product[off+half],len,rows,rr);PipeBarrier<PIPE_V>();
                    Sub(z[off],total[off],product[off],len,rows,rr);PipeBarrier<PIPE_V>();
                    Sub(temp[off],total[off],z[off],len,rows,rr);PipeBarrier<PIPE_V>();
                    Sub(temp[off],product[off],temp[off],len,rows,rr);
                    Sub(other[off],product[off+half],z[off],len,rows,rr);PipeBarrier<PIPE_V>();
                    Add(temp[off],temp[off],other[off],len,rows,rr);
                    Add(low[off],low[off],low[off+half],len,rows,rr);PipeBarrier<PIPE_V>();
                    Add(low[off],low[off],temp[off],len,rows,rr);
                    Adds(product[off],total[off],0.0f,len,rows,ur);PipeBarrier<PIPE_V>();
                }
                remain=half;
            }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::StoreResidualRows(LocalTensor<float> other,LocalTensor<float> savedB,LocalTensor<float> original,LocalTensor<float> ah,LocalTensor<float> al,LocalTensor<float> z,LocalTensor<float> total,LocalTensor<float> temp,LocalTensor<uint32_t>& leftIndex,uint64_t boff,uint32_t rows,uint32_t q,uint32_t base){
            auto originalB=other;
            if(parallel) {leftIndex=indices[352];originalB=savedB;}
            else {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(other,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
            CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),rows);PipeBarrier<PIPE_V>();
            Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(32),rows);PipeBarrier<PIPE_V>();
            }
            Gather(al,originalB,leftIndex,q*4,rows);Muls(original,original,-1.0f,rows);PipeBarrier<PIPE_V>();
            Add(total,al,original,rows);PipeBarrier<PIPE_V>();Sub(z,total,al,rows);PipeBarrier<PIPE_V>();
            Sub(temp,total,z,rows);PipeBarrier<PIPE_V>();Sub(temp,al,temp,rows);Sub(other,original,z,rows);PipeBarrier<PIPE_V>();
            Add(temp,temp,other,rows);PipeBarrier<PIPE_V>();Sub(temp,temp,ah,rows);PipeBarrier<PIPE_V>();
            Add(transposed[q*na+base],total,temp,rows);PipeBarrier<PIPE_V>();
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ReduceResidualRows(LocalTensor<float> product,LocalTensor<float> low,LocalTensor<float> original,LocalTensor<float> ah,uint32_t rows){
            BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
            for(uint32_t remain=na;remain>8;) {
                uint32_t half=(remain+15)/16*8,count=remain-half;
                for(uint32_t off=0;off<count;off+=64) {
                    uint32_t len=count-off<64?count-off:64;
                    Add(product[off],product[off],product[off+half],len,rows,rr);
                    Add(low[off],low[off],low[off+half],len,rows,rr);
                }
                PipeBarrier<PIPE_V>();remain=half;
            }
            // The remaining eight terms already occupy one block per row.
            // Reduce them in place in the row layout, avoiding compaction tables.
            WholeReduceSum(original,product,8,rows,1,1,na/8);
            WholeReduceSum(ah,low,8,rows,1,1,na/8);PipeBarrier<PIPE_V>();
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::LoadResidualRhs(LocalTensor<float> savedB,uint64_t boff,uint32_t rows){
        if(parallel) {
            Fence<HardEvent::V_MTE2>();
            DataCopyPad(savedB,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        }
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ComputeBlockedResidual(uint32_t batch,uint32_t lane){
        Duplicate(transposed,0.0f,k*na);PipeBarrier<PIPE_V>();
        PrepareResidualIndices();
        uint32_t tileRows=n>=224?32:16;
        for(uint32_t base=lane*tileRows;base<n;base+=workers*tileRows) {
            uint32_t rows=n-base<tileRows?n-base:tileRows;
            if(n>=224)ResidualTileRounded(batch,base,rows,tileRows);
            else ResidualTileAccurate(batch,base,rows,tileRows);
            Fence<HardEvent::V_MTE3>();
            if constexpr(FAMILY==3 && STATIC_N==256) {
                // 32-row aligned segments; equal 256-element source/destination pitch.
                DataCopy(w[(n+8)*na+base],transposed[base],
                    DataCopyParams{uint16_t(k),uint16_t(rows/8),uint16_t((na-rows)/8),uint16_t((na-rows)/8)});
            } else {
                for(uint32_t q=0;q<k;++q)
                    DataCopyPad(w[(n+8+q)*na+base],transposed[q*na+base],DataCopyExtParams{1,rows*4,0,0,0});
            }
            Fence<HardEvent::MTE3_V>();
        }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::RowMul(LocalTensor<float> dst,LocalTensor<float> a0,LocalTensor<float> b0,uint32_t rows){
        for(uint32_t j=0;j<na;j+=64) {
            uint32_t count=na-j<64?na-j:64;
            Mul(dst[j],a0[j],b0[j],count,rows,
                BinaryRepeatParams{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),0});
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::PrepareResidualIndices(){
        CreateVecIndex(indices[352].ReinterpretCast<int32_t>(),int32_t(0),32);PipeBarrier<PIPE_V>();
        Muls(indices[352].ReinterpretCast<int32_t>(),indices[352].ReinterpretCast<int32_t>(),int32_t(32),32);PipeBarrier<PIPE_V>();
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualTileOriginal(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows){
        const uint32_t size=tileRows*na;
        auto original=mat;auto ah=mat[size];auto al=mat[2*size];
        auto product=mat[3*size];auto low=mat[4*size];auto temp=mat[5*size];
        auto z=mat[6*size];auto total=mat[7*size];auto other=mat[8*size];
        auto xh=factor;auto xl=column;
        auto leftIndex=indices;auto rightIndex=indices[64];auto integerTemp=indices[128].ReinterpretCast<int32_t>();
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        auto savedB=input[8*na];
    LoadResidualRhs(savedB,boff,rows);
        // Reuse immutable gather tables and the complete RHS tile across its columns.
        for(uint32_t q=0;q<k;++q) {
            leftIndex=indices;rightIndex=indices[64];
            Duplicate(original,0.0f,rows*na);Fence<HardEvent::V_MTE2>();
            DataCopyPad(original,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                        DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
    ComputeOriginalProducts(product,original,ah,al,xh,xl,low,temp,q,rows);
    ReduceCompensatedRows(product,low,total,z,temp,other,rows);
            // Gather the final8 terms per row, then perform three packed TwoSum levels.
            if(!parallel) {
            CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),rows*8);PipeBarrier<PIPE_V>();
            ShiftRight(integerTemp,leftIndex.ReinterpretCast<int32_t>(),int32_t(3),rows*8);PipeBarrier<PIPE_V>();
            Muls(integerTemp,integerTemp,int32_t(na-8),rows*8);PipeBarrier<PIPE_V>();
            Add(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),integerTemp,rows*8);PipeBarrier<PIPE_V>();
            Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(4),rows*8);PipeBarrier<PIPE_V>();
            }
            Gather(original,product,leftIndex,0,rows*8);Gather(ah,low,leftIndex,0,rows*8);PipeBarrier<PIPE_V>();
            for(int32_t shift=2;shift>=0;--shift) {
                uint32_t half=uint32_t(1)<<shift,count=rows*half;
                if(parallel) {
                    uint32_t offset=shift==2?128:(shift==1?256:320);
                    leftIndex=indices[offset];rightIndex=indices[offset+16*half];
                } else {
                CreateVecIndex(leftIndex.ReinterpretCast<int32_t>(),int32_t(0),count);PipeBarrier<PIPE_V>();
                ShiftRight(integerTemp,leftIndex.ReinterpretCast<int32_t>(),shift,count);PipeBarrier<PIPE_V>();
                Muls(integerTemp,integerTemp,int32_t(half),count);PipeBarrier<PIPE_V>();
                Add(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),integerTemp,count);PipeBarrier<PIPE_V>();
                Muls(leftIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(4),count);PipeBarrier<PIPE_V>();
                Adds(rightIndex.ReinterpretCast<int32_t>(),leftIndex.ReinterpretCast<int32_t>(),int32_t(half*4),count);PipeBarrier<PIPE_V>();
                }
                Gather(al,original,leftIndex,0,count);Gather(product,original,rightIndex,0,count);PipeBarrier<PIPE_V>();
                Add(total,al,product,count);PipeBarrier<PIPE_V>();Sub(z,total,al,count);PipeBarrier<PIPE_V>();
                Sub(temp,total,z,count);PipeBarrier<PIPE_V>();Sub(temp,al,temp,count);Sub(other,product,z,count);PipeBarrier<PIPE_V>();
                Add(temp,temp,other,count);
                Gather(al,ah,leftIndex,0,count);Gather(product,ah,rightIndex,0,count);PipeBarrier<PIPE_V>();
                Add(ah,al,product,count);PipeBarrier<PIPE_V>();Add(ah,ah,temp,count);Adds(original,total,0.0f,count);PipeBarrier<PIPE_V>();
            }
    StoreResidualRows(other,savedB,original,ah,al,z,total,temp,leftIndex,boff,rows,q,base);
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualTile(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows){
        if constexpr(FAMILY==2)ResidualTileRounded(batch,base,rows,tileRows);
        else if constexpr(FAMILY==1)ResidualTileSplitProduct(batch,base,rows,tileRows);
        else if constexpr(FAMILY==3)ResidualTileAccurate(batch,base,rows,tileRows);
        else {
            if(parallel && n>=224)ResidualTileRounded(batch,base,rows,tileRows);
            else ResidualTileAccurate(batch,base,rows,tileRows);
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualPlainThreeRhs32(uint32_t batch,uint32_t base){
        constexpr uint32_t tile=32*256,rows=96;
        auto sourceA=mat,products=mat[tile];
        auto sums=mat[4*tile],originalB=mat[4*tile+128];
        auto savedB=input[8*na];
        uint64_t aoff=uint64_t(batch)*logicalN*logicalN+uint64_t(base)*logicalN;
        uint64_t boff=(broadcast?0:uint64_t(batch)*logicalN*logicalK)+uint64_t(base)*logicalK;
        uint32_t validRows=logicalN-base<32?logicalN-base:32;
        if(logicalN!=n || logicalK!=k) {
            Duplicate(savedB,0.0f,32*8);PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE2>();
        if constexpr(PARTIAL_CAPACITY)
            DataCopyPad(savedB,b[boff],DataCopyExtParams{uint16_t(validRows),logicalK*4,0,0,0},
                        DataCopyPadExtParams<float>{true,0,uint8_t(8-logicalK),0});
        else DataCopyPad(savedB,b[boff],DataCopyExtParams{32,12,0,0,0},
                         DataCopyPadExtParams<float>{false,0,0,0});
        Duplicate(sourceA,0.0f,tile);Fence<HardEvent::V_MTE2>();
        if(logicalN==n)DataCopy(sourceA,a[aoff],tile);
        else DataCopyPad(sourceA,a[aoff],
            DataCopyExtParams{uint16_t(validRows),logicalN*4,0,(na-(logicalN+7)/8*8)/8,0},
            DataCopyPadExtParams<float>{true,0,uint8_t((8-logicalN%8)%8),0});
        Fence<HardEvent::MTE2_V>();
        for(uint32_t q=0;q<3;++q)RowMul(products[q*tile],sourceA,input[q*na],32);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams rr{1,1,1,32,32,32};
        for(uint32_t remain=256;remain>8;) {
            uint32_t half=(remain+15)/16*8,count=remain-half;
            for(uint32_t off=0;off<count;off+=64) {
                uint32_t len=count-off<64?count-off:64;
                Add(products[off],products[off],products[off+half],len,rows,rr);
            }
            PipeBarrier<PIPE_V>();remain=half;
        }
        WholeReduceSum(sums,products,8,rows,1,1,32);
        for(uint32_t q=0;q<3;++q)Gather(originalB[q*32],savedB,indices[352],q*4,32);
        PipeBarrier<PIPE_V>();
        Sub(sums,originalB,sums,rows);PipeBarrier<PIPE_V>();
        DataCopy(transposed[base],sums,DataCopyParams{3,4,0,28});PipeBarrier<PIPE_V>();
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualTileRounded(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows){
        if(!parallel){ResidualTileOriginal(batch,base,rows,tileRows);return;}
        if constexpr(FAMILY==3 && STATIC_N==256 && (STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) {
            ResidualPlainThreeRhs32(batch,base);return;
        }
        const uint32_t size=tileRows*na;
        auto sourceA=mat;auto product=mat[size];auto low=mat[2*size];auto ah=mat[3*size];
        auto original=mat[4*size];auto al=mat[4*size+64];auto z=mat[4*size+128];
        auto total=mat[4*size+192];auto other=mat[4*size+256];auto temp=mat[4*size+320];
        auto xh=factor;auto xl=column;
        auto leftIndex=indices;auto rightIndex=indices[64];auto integerTemp=indices[128].ReinterpretCast<int32_t>();
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        auto savedB=input[8*na];
    LoadResidualRhs(savedB,boff,rows);
        Duplicate(sourceA,0.0f,rows*na);Fence<HardEvent::V_MTE2>();
        DataCopyPad(sourceA,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        // Reuse immutable gather tables and the complete RHS tile across its columns.
        for(uint32_t q=0;q<k;++q) {
            leftIndex=indices;rightIndex=indices[64];
            RowMul(product,sourceA,input[q*na],rows);
            PipeBarrier<PIPE_V>();
            // Full tiles stay live through reduction; row outputs reuse a compact tail.
            ShiftRight(ah.ReinterpretCast<int32_t>(),product.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            ShiftLeft(ah.ReinterpretCast<int32_t>(),ah.ReinterpretCast<int32_t>(),int32_t(15),rows*na);PipeBarrier<PIPE_V>();
            Sub(low,product,ah,rows*na);PipeBarrier<PIPE_V>();
            Adds(product,ah,0.0f,rows*na);PipeBarrier<PIPE_V>();
    ReduceResidualRows(product,low,original,ah,rows);
    StoreResidualRows(other,savedB,original,ah,al,z,total,temp,leftIndex,boff,rows,q,base);
        }
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualTileAccurate(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows){
    ResidualTileCompensated<false>(batch,base,rows,tileRows);
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualJointFour(uint32_t batch,uint32_t base,uint32_t rows){
        const uint32_t size=rows*na,allRows=rows*k;
        auto sourceA=mat;auto sourceHigh=mat[size];auto sourceLow=mat[2*size];
        auto product=mat[3*size];auto low=mat[(3+k)*size];auto temp=mat[(3+2*k)*size];
        auto xh=factor;auto xl=column;auto savedB=input[8*na];
        uint64_t aoff=uint64_t(batch)*n*n+uint64_t(base)*n;
        uint64_t boff=(broadcast?0:uint64_t(batch)*n*k)+uint64_t(base)*k;
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(savedB,b[boff],DataCopyExtParams{static_cast<uint16_t>(rows),k*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        Duplicate(sourceA,0.0f,size);Fence<HardEvent::V_MTE2>();
        DataCopyPad(sourceA,a[aoff],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});Fence<HardEvent::MTE2_V>();
        ShiftRight(sourceHigh.ReinterpretCast<int32_t>(),sourceA.ReinterpretCast<int32_t>(),int32_t(20),size);
        PipeBarrier<PIPE_V>();
        ShiftLeft(sourceHigh.ReinterpretCast<int32_t>(),sourceHigh.ReinterpretCast<int32_t>(),int32_t(20),size);
        PipeBarrier<PIPE_V>();Sub(sourceLow,sourceA,sourceHigh,size);PipeBarrier<PIPE_V>();
    ComputeJointProducts(xh,xl,product,sourceHigh,sourceLow,sourceA,low,temp,rows,size);
        BinaryRepeatParams rr{1,1,1,static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8),static_cast<uint8_t>(na/8)};
        for(uint32_t remain=na;remain>8;) {
            uint32_t half=(remain+15)/16*8,count=remain-half;
            for(uint32_t off=0;off<count;off+=64) {
                uint32_t len=count-off<64?count-off:64;
                Add(product[off],product[off],product[off+half],len,allRows,rr);
                Add(low[off],low[off],low[off+half],len,allRows,rr);
            }
            PipeBarrier<PIPE_V>();remain=half;
        }
        // Input A has no remaining readers; reuse its storage for the short finish.
        auto highSum=mat;auto lowSum=mat[64];auto originalB=mat[128];
        auto total=mat[192];auto z=mat[256];auto other=mat[320];
        WholeReduceSum(highSum,product,8,allRows,1,1,na/8);
        WholeReduceSum(lowSum,low,8,allRows,1,1,na/8);
        auto offsets=indices.ReinterpretCast<int32_t>();
        for(uint32_t q=0;q<k;++q)Adds(offsets[q*rows],indices[352].ReinterpretCast<int32_t>(),int32_t(q*4),rows);
        PipeBarrier<PIPE_V>();
        Gather(originalB,savedB,indices,0,allRows);
        Muls(highSum,highSum,-1.0f,allRows);PipeBarrier<PIPE_V>();
        Add(total,originalB,highSum,allRows);PipeBarrier<PIPE_V>();
        Sub(z,total,originalB,allRows);PipeBarrier<PIPE_V>();
        Sub(temp,total,z,allRows);PipeBarrier<PIPE_V>();
        Sub(temp,originalB,temp,allRows);Sub(other,highSum,z,allRows);PipeBarrier<PIPE_V>();
        Add(temp,temp,other,allRows);PipeBarrier<PIPE_V>();
        Sub(temp,temp,lowSum,allRows);PipeBarrier<PIPE_V>();
        Add(z,total,temp,allRows);PipeBarrier<PIPE_V>();
        for(uint32_t q=0;q<k;++q)DataCopy(transposed[q*na+base],z[q*rows],rows);
        PipeBarrier<PIPE_V>();
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ResidualTileSplitProduct(uint32_t batch,uint32_t base,uint32_t rows,uint32_t tileRows){
        if(parallel && n<=200 && n%8==0 && k<=4){ResidualJointFour(batch,base,rows);return;}
    ResidualTileCompensated<true>(batch,base,rows,tileRows);
    }

#endif
