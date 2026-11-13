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

#ifndef RUBY0451_SOLVE_SOLVE_SMALL_H
#define RUBY0451_SOLVE_SOLVE_SMALL_H
#include "solve_common.h"
__aicore__ inline void TransposeSmallColumns(LocalTensor<float> dst,LocalTensor<float> src,
                                            uint32_t rows,uint32_t sourcePitch,uint32_t targetPitch,
                                            uint32_t columns) {
    for(uint32_t block=0;block<rows;block+=16) {
        uint64_t source[16],target[16];
        #pragma unroll
        for(uint32_t i=0;i<16;++i) {
            source[i]=reinterpret_cast<uint64_t>(src[(block+i)*sourcePitch].GetPhyAddr());
            target[i]=reinterpret_cast<uint64_t>(dst[(i/2)*targetPitch+block+(i%2)*8].GetPhyAddr());
        }
        TransDataTo5HD<float>(target,source,TransDataTo5HDParams{false,false,
            static_cast<uint8_t>(columns/8),static_cast<uint16_t>(targetPitch),1});
    }
}
template<bool BLOCK_TRANSPOSE,bool FULL_WIDTH=false> class KernelSmallOuterT {
public:

 __aicore__ inline void StoreOuterResult(uint32_t batch,uint32_t owners) {
            Fence<HardEvent::V_S>();
            auto offsets=transposeIndex.ReinterpretCast<int32_t>();
            auto pattern=baseIndex;
            for(uint32_t q=0;q<8;++q) reinterpret_cast<__ubuf__ int32_t*>(pattern.GetPhyAddr())[q]=(n+(q<k?q:0))*stride*4;
            Fence<HardEvent::S_V>();
            Brcb(offsets.ReinterpretCast<float>(),order.ReinterpretCast<float>(),(n+7)/8,BrcbRepeatParams{1,8});
            PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(4),pitch*8);PipeBarrier<PIPE_V>();
            Add(offsets,offsets,pattern,64,(n+7)/8,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            Gather(output,mat,offsets.ReinterpretCast<uint32_t>(),0,n*8);PipeBarrier<PIPE_V>();
            Brcb(input,inverse,(n+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
            Mul(output,output,input,n*8);PipeBarrier<PIPE_V>();
            // Keep the first k floats of each padded output row, then one contiguous DMA.
            Duplicate(transposeIndex,uint32_t(((1u<<k)-1)*0x01010101u),8);PipeBarrier<PIPE_V>();
            SetMaskNorm();
            GatherMask(input,output,transposeIndex.ReinterpretCast<float>(),uint8_t(0),
                       GatherMaskParams{1,static_cast<uint16_t>((n+7)/8),8,0});
            Fence<HardEvent::V_MTE3>();
            if(GetBlockIdx()<owners){DataCopyPad(x[uint64_t(batch)*n*k],input,DataCopyExtParams{1,n*k*4,0,0,0});}
            if(batch+owners<batches)Fence<HardEvent::MTE3_S>();

}


 __aicore__ inline void UpdateOuterColumns(uint32_t p,uint32_t row,uint32_t remaining,float inv,uint64_t* activeRows) {
                if constexpr(BLOCK_TRANSPOSE) {
                    if constexpr(FULL_WIDTH)Muls<float>(factor,factor,-inv,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
                    else Muls(factor,factor,-inv,pitch);
                    if constexpr(!FULL_WIDTH)
                        Gather(pivot,mat[(p+1)*stride],baseIndex.ReinterpretCast<uint32_t>(),row*4,remaining);
                    PipeBarrier<PIPE_V>();
                    Brcb(spread,pivot,(remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                    if constexpr(FULL_WIDTH) {
                        if(p+1<n) {
                            auto next=mat[(p+1)*stride];
                            MulAddDst<float,float,false>(next,factor,spread,uint64_t(64),1,BinaryRepeatParams{1,1,0,9,0,1});
                            PipeBarrier<PIPE_V>();
                            WholeReduceMax(reduced,next,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                            SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
                            // Reduction changes the mask. Restore all64 rows
                            // for the disjoint remaining-column updates.
                            MulAddDst<float,float,true>(next[stride],factor,spread[8],uint64_t(64),remaining-1,BinaryRepeatParams{1,1,0,9,0,1});
                        } else MulAddDst<float,float,false>(mat[(p+1)*stride],factor,spread,uint64_t(64),remaining,BinaryRepeatParams{1,1,0,9,0,1});
                    }
                    else MulAddDst(mat[(p+1)*stride],factor,spread,pitch,remaining,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(stride/8),0,1});
                } else {
                if constexpr(!FULL_WIDTH)
                    Gather(pivot,mat,baseIndex.ReinterpretCast<uint32_t>(),row*4,cols);
                PipeBarrier<PIPE_V>();
                if constexpr(FULL_WIDTH) {
                    if(remaining<=64)Muls<float,false>(pivot,pivot,-inv,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                    else Muls(pivot,pivot,-inv,remaining);
                } else Muls(pivot,pivot,-inv,cols);
                PipeBarrier<PIPE_V>();
                Brcb(spread,pivot,FULL_WIDTH?(remaining+7)/8:cols/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                if constexpr(FULL_WIDTH)MulAddDst<float,float,false>(mat[(p+1)*stride],factor,spread,uint64_t(64),remaining,BinaryRepeatParams{1,1,0,9,0,1});
                else MulAddDst(mat[(p+1)*stride],factor,spread[(p+1)*8],pitch,remaining,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(stride/8),0,1});
                }
                if constexpr(!(FULL_WIDTH && BLOCK_TRANSPOSE))PipeBarrier<PIPE_V>();

}


 __aicore__ inline void ChooseOuterPivot(uint32_t p,uint64_t* activeRows,uint32_t& row,float& diagonal) {
                if constexpr(FULL_WIDTH && BLOCK_TRANSPOSE) {
                    if(p==0) {
                        WholeReduceMax(reduced,factor,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                        SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
                    }
                } else if constexpr(FULL_WIDTH)
                    WholeReduceMax(reduced,factor,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                else {
                    Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
                    WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                }
                if constexpr(FULL_WIDTH && BLOCK_TRANSPOSE)
                    WaitFlag<HardEvent::V_S>(static_cast<event_t>(1));
                else Fence<HardEvent::V_S>();
                if constexpr(FULL_WIDTH) {
                    // ORDER_VALUE_INDEX already contains the signed pivot value.
                    uint64_t packet=reinterpret_cast<__ubuf__ uint64_t*>(reduced.GetPhyAddr())[0];
                    union {uint32_t bits;float value;} word;
                    word.bits=static_cast<uint32_t>(packet);
                    row=static_cast<uint32_t>(packet>>32);diagonal=word.value;
                } else {
                    row=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                    diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                }

}

    __aicore__ inline ~KernelSmallOuterT() {PipeBarrier<PIPE_ALL>();}
    template<HardEvent event> __aicore__ inline void Fence() {
        const event_t id=static_cast<event_t>(0);
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,const LinalgSolveTilingData& t) {
        n=t.n;k=t.k;pitch=FULL_WIDTH?64:8;if constexpr(!FULL_WIDTH){while(pitch<n)pitch*=2;}stride=pitch+8;cols=(n+k+7)/8*8;batches=t.batches;cores=t.cores;broadcast=t.broadcastB;
        a.SetGlobalBuffer((__gm__ float*)A);b.SetGlobalBuffer((__gm__ float*)B);x.SetGlobalBuffer((__gm__ float*)X);
        SetMaskNorm();SetAtomicNone();
        mat=LocalTensor<float>(TPosition::VECCALC,0,5184);
        input=LocalTensor<float>(TPosition::VECCALC,20736,4096);
        transposed=LocalTensor<float>(TPosition::VECCALC,37120,4096);
        transposeIndex=LocalTensor<uint32_t>(TPosition::VECCALC,53504,4096);
        pivot=LocalTensor<float>(TPosition::VECCALC,69888,72);
        spread=LocalTensor<float>(TPosition::VECCALC,70176,576);
        baseIndex=LocalTensor<int32_t>(TPosition::VECCALC,72480,72);
        absolute=LocalTensor<float>(TPosition::VECCALC,72768,64);
        reduced=LocalTensor<float>(TPosition::VECCALC,73024,8);
        order=LocalTensor<uint32_t>(TPosition::VECCALC,73056,64);
        inverse=LocalTensor<float>(TPosition::VECCALC,73312,64);
        output=LocalTensor<float>(TPosition::VECCALC,73568,512);
    }
    __aicore__ inline void MakeIndex(LocalTensor<int32_t> target,uint32_t count) {
        for(uint32_t i=0;i<8;++i)target.SetValue(i,int32_t(i));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<count;filled*=2) {
            uint32_t len=count-filled<filled?count-filled:filled;
            Adds(target[filled],target,int32_t(filled),len);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadMatrix(GlobalTensor<float> source,uint64_t offset,uint32_t count,uint32_t first) {
        if constexpr(BLOCK_TRANSPOSE) {
            if(first==0 && count==n) {
                uint32_t sourcePitch=(n+7)/8*8;
                DataCopyPad(input,source[offset],DataCopyExtParams{static_cast<uint16_t>(n),n*4,0,0,0},
                            DataCopyPadExtParams<float>{false,0,0,0});
                Fence<HardEvent::MTE2_V>();
                TransposeSmallColumns(mat,input,pitch,sourcePitch,stride,sourcePitch);
                PipeBarrier<PIPE_V>();return;
            }
        }
        uint32_t size=count*pitch;
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(input,source[offset],DataCopyExtParams{1,n*count*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        auto offsets=transposeIndex.ReinterpretCast<int32_t>();
        Fence<HardEvent::V_S>();
        for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*count*4);
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<pitch;filled*=2) {
            Adds(offsets[filled],offsets,int32_t(filled*count*4),filled);PipeBarrier<PIPE_V>();
        }
        for(uint32_t filled=pitch;filled<size;filled*=2) {
            uint32_t len=size-filled<filled?size-filled:filled;
            Adds(offsets[filled],offsets,int32_t(filled/pitch*4),len);PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::MTE2_V>();Gather(transposed,input,transposeIndex,0,size);PipeBarrier<PIPE_V>();
        Adds(mat[first*stride],transposed,0.0f,pitch,count,
             UnaryRepeatParams{1,1,static_cast<uint8_t>(stride/8),static_cast<uint8_t>(pitch/8)});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        uint32_t owners=cores<batches?cores:batches;
        for(uint32_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners) {
            MakeIndex(baseIndex,cols);PipeBarrier<PIPE_V>();
            Muls(baseIndex,baseIndex,static_cast<int32_t>(stride*4),cols);PipeBarrier<PIPE_V>();
            LoadMatrix(a,uint64_t(batch)*n*n,n,0);
            LoadMatrix(b,broadcast?0:uint64_t(batch)*n*k,k,n);
            uint64_t activeRows[2]={n==64?~uint64_t(0):(uint64_t(1)<<n)-1,0};
            for(uint32_t p=0;p<n;++p) {
                factor=mat[p*stride];
    uint32_t row;float diagonal;
    ChooseOuterPivot(p,activeRows,row,diagonal);
                uint32_t remaining=n+k-p-1;
                // The reduced next column is ready independently of later FMA
                // destinations. Join those updates before Gather reads them.
                if constexpr(FULL_WIDTH && BLOCK_TRANSPOSE)PipeBarrier<PIPE_V>();
                if constexpr(FULL_WIDTH)
                    Gather(pivot,mat[(p+1)*stride],baseIndex.ReinterpretCast<uint32_t>(),row*4,remaining);
                if constexpr(FULL_WIDTH) {
                    if(diagonal==0.0f) {
                        Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
                        WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                        Fence<HardEvent::V_S>();row=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                        diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                        Gather(pivot,mat[(p+1)*stride],baseIndex.ReinterpretCast<uint32_t>(),row*4,remaining);
                    }
                }
                reinterpret_cast<__ubuf__ uint32_t*>(order.GetPhyAddr())[p]=row;activeRows[0]&=~(uint64_t(1)<<row);
                float inv=1.0f/diagonal;reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr())[p]=inv;reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row]=0.0f;
                Fence<HardEvent::S_V>();
    UpdateOuterColumns(p,row,remaining,inv,activeRows);
            }
    StoreOuterResult(batch,owners);
        }
    }
private:
    LocalTensor<float> mat,input,transposed,factor,pivot,spread,inverse,output,absolute,reduced;
    LocalTensor<int32_t> baseIndex;LocalTensor<uint32_t> transposeIndex,order;
    GlobalTensor<float> a,b,x;
    uint32_t n,k,pitch,stride,cols,batches,cores;bool broadcast;
};
using KernelSmallOuter=KernelSmallOuterT<false>;

template<uint32_t SN,uint32_t SK,bool PARTIAL_CAPACITY=false> class KernelSmallFixed {
public:

 __aicore__ inline void ChoosePivot(uint32_t p,uint64_t* activeRows,uint32_t& pairFirst,float& pivotThreshold,uint32_t& row,float& diagonal) {
                if constexpr(n==16) {
                    if((p&1)==0) {
                        Fence<HardEvent::V_S>();
                        uint64_t packet=reinterpret_cast<__ubuf__ uint64_t*>(factor.GetPhyAddr())[p/2];
                        uint32_t left=static_cast<uint32_t>(packet),right=static_cast<uint32_t>(packet>>32);
                        bool chooseRight=(left&0x7fffffffu)<(right&0x7fffffffu);
                        row=p+uint32_t(chooseRight);pairFirst=row;
                        union {uint32_t bits;float value;} word;
                        word.bits=chooseRight?right:left;diagonal=word.value;
                    } else {
                        Fence<HardEvent::V_S>();row=pairFirst^1u;
                        diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                    }
                } else if constexpr(n==15) {
    ChooseTailPivot(p,activeRows,pivotThreshold,row,diagonal);
                    activeRows[0]&=~(uint64_t(1)<<row);
                } else {
                    Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
                    WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                    Fence<HardEvent::V_S>();
                    row=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                    activeRows[0]&=~(uint64_t(1)<<row);
                    diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                }

}


 __aicore__ inline void ChooseTailPivot(uint32_t p,uint64_t* activeRows,float& pivotThreshold,uint32_t& row,float& diagonal) {
                    if(p+2>=n) {
                        uint64_t first=activeRows[0]&(~activeRows[0]+1);
                        uint32_t left=63-static_cast<uint32_t>(CountLeadingZero(first));
                        Fence<HardEvent::V_S>();
                        row=left;diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[left];
                        if(p+2==n) {
                            uint32_t right=63-static_cast<uint32_t>(CountLeadingZero(activeRows[0]^first));
                            float other=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[right];
                            union {uint32_t bits;float value;} aWord,bWord;
                            aWord.value=diagonal;bWord.value=other;
                            if((aWord.bits&0x7fffffffu)<(bWord.bits&0x7fffffffu)) {
                                row=right;diagonal=other;
                            }
                        }
                    } else {
                        bool search=p==0;
                        if(p) {
                            uint64_t first=activeRows[0]&(~activeRows[0]+1);
                            row=63-static_cast<uint32_t>(CountLeadingZero(first));
                            Fence<HardEvent::V_S>();diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                            float magnitude=diagonal<0.0f?-diagonal:diagonal;
                            search=magnitude<pivotThreshold;
                        }
                        if(search) {
                            auto candidates=absolute;
                            if(p==0) {
                                Muls(candidates,factor,-1.0f,pitch);
                                Adds(candidates[pitch],factor,0.0f,pitch);
                            } else {
                                // Retired tiny GJ columns are dead; the current column follows this copy.
                                candidates=mat[(p-1)*pitch];
                                Muls(candidates,factor,-1.0f,pitch);
                            }
                            PipeBarrier<PIPE_V>();
                            uint64_t signedRows[2]={activeRows[0]|(activeRows[0]<<pitch),0};
                            WholeReduceMax(reduced,candidates,signedRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                            Fence<HardEvent::V_S>();
                            uint64_t packet=reinterpret_cast<__ubuf__ uint64_t*>(reduced.GetPhyAddr())[0];
                            uint32_t index=static_cast<uint32_t>(packet>>32);
                            union {uint32_t bits;float value;} word;
                            word.bits=static_cast<uint32_t>(packet)|(uint32_t(index<pitch)<<31);
                            diagonal=word.value;row=index&(pitch-1);
                        }
                        if(p==0)pivotThreshold=(diagonal<0.0f?-diagonal:diagonal)*0.5f;
                    }

}


 __aicore__ inline void LoadBatch(uint32_t batch) {
            if constexpr(PARTIAL_CAPACITY)LoadPartial(batch);
            else if constexpr(n>8 && n<=16 && k==1) {
                Fence<HardEvent::V_MTE2>();
                DataCopyPad(input,a[uint64_t(batch)*n*n],DataCopyExtParams{1,n*n*4,0,0,0},
                    DataCopyPadExtParams<float>{true,0,(8-n*n%8)%8,0});
                DataCopyPad(mat[n*pitch],b[broadcast?0:uint64_t(batch)*n],DataCopyExtParams{1,n*4,0,0,0},
                    DataCopyPadExtParams<float>{true,0,pitch-n,0});
                auto offsets=transposeIndex.ReinterpretCast<int32_t>();
                #pragma unroll
                for(uint32_t i=0;i<8;++i){reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*n*4);reinterpret_cast<__ubuf__ int32_t*>(baseIndex.GetPhyAddr())[i]=int32_t(i*pitch*4);}
                Fence<HardEvent::S_V>();
                Adds(offsets[8],offsets,int32_t(8*n*4),8);
                Adds(baseIndex[8],baseIndex,int32_t(8*pitch*4),8);PipeBarrier<PIPE_V>();
                if constexpr(cols>16){Adds(baseIndex[16],baseIndex,int32_t(16*pitch*4),cols-16);}
                for(uint32_t filled=pitch;filled<n*pitch;filled*=2) {
                    uint32_t len=n*pitch-filled<filled?n*pitch-filled:filled;
                    Adds(offsets[filled],offsets,int32_t(filled/pitch*4),len);PipeBarrier<PIPE_V>();
                }
                Fence<HardEvent::MTE2_V>();
                Gather(mat,input,transposeIndex,0,n*pitch);PipeBarrier<PIPE_V>();
            } else {
                MakeIndex(baseIndex,cols);PipeBarrier<PIPE_V>();
                Muls(baseIndex,baseIndex,static_cast<int32_t>(pitch*4),cols);PipeBarrier<PIPE_V>();
                Duplicate(mat,0.0f,cols*pitch);PipeBarrier<PIPE_V>();
                LoadMatrix(a,uint64_t(batch)*n*n,n,0);
                LoadMatrix(b,broadcast?0:uint64_t(batch)*n*k,k,n);
            }

}

    __aicore__ inline ~KernelSmallFixed() {PipeBarrier<PIPE_ALL>();}
    template<HardEvent event> __aicore__ inline void Fence() {
        const event_t id=static_cast<event_t>(0);
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,const LinalgSolveTilingData& t) {
        batches=t.batches;cores=t.cores;broadcast=t.broadcastB;logicalN=PARTIAL_CAPACITY?t.n:n;
        a.SetGlobalBuffer((__gm__ float*)A);b.SetGlobalBuffer((__gm__ float*)B);x.SetGlobalBuffer((__gm__ float*)X);
        SetMaskNorm();SetAtomicNone();
        constexpr uint32_t offset0=0;
        mat=LocalTensor<float>(TPosition::VECCALC,offset0,cols*pitch);
        constexpr uint32_t offset1=offset0+(cols*pitch)*4;
        input=LocalTensor<float>(TPosition::VECCALC,offset1,pitch*pitch);
        constexpr uint32_t offset2=offset1+(pitch*pitch)*4;
        transposed=LocalTensor<float>(TPosition::VECCALC,offset2,pitch*pitch);
        constexpr uint32_t offset3=offset2+(pitch*pitch)*4;
        transposeIndex=LocalTensor<uint32_t>(TPosition::VECCALC,offset3,pitch*pitch);
        constexpr uint32_t offset4=offset3+(pitch*pitch)*4;
        pivot=LocalTensor<float>(TPosition::VECCALC,offset4,cols);
        constexpr uint32_t offset5=offset4+(cols)*4;
        spread=LocalTensor<float>(TPosition::VECCALC,offset5,cols*8);
        constexpr uint32_t offset6=offset5+(cols*8)*4;
        baseIndex=LocalTensor<int32_t>(TPosition::VECCALC,offset6,cols);
        constexpr uint32_t offset7=offset6+(cols)*4;
        absolute=LocalTensor<float>(TPosition::VECCALC,offset7,64);
        constexpr uint32_t offset8=offset7+(64)*4;
        reduced=LocalTensor<float>(TPosition::VECCALC,offset8,8);
        constexpr uint32_t offset9=offset8+(8)*4;
        order=LocalTensor<uint32_t>(TPosition::VECCALC,offset9,64);
        constexpr uint32_t offset10=offset9+(64)*4;
        inverse=LocalTensor<float>(TPosition::VECCALC,offset10,64);
        constexpr uint32_t offset11=offset10+(64)*4;
        output=LocalTensor<float>(TPosition::VECCALC,offset11,n*8);
    }
    __aicore__ inline void MakeIndex(LocalTensor<int32_t> target,uint32_t count) {
        uint32_t seed=count<8?count:8;
        for(uint32_t i=0;i<seed;++i)target.SetValue(i,int32_t(i));
        Fence<HardEvent::S_V>();
        for(uint32_t filled=8;filled<count;filled*=2) {
            uint32_t len=count-filled<filled?count-filled:filled;
            Adds(target[filled],target,int32_t(filled),len);PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void LoadMatrix(GlobalTensor<float> source,uint64_t offset,uint32_t count,uint32_t first) {
        uint32_t shift=3;while((uint32_t(1)<<shift)<pitch)++shift;
        uint32_t size=count*pitch;
        auto sequence=input.ReinterpretCast<int32_t>();
        auto columnIndex=transposed.ReinterpretCast<int32_t>();
        auto offsets=transposeIndex.ReinterpretCast<int32_t>();
        MakeIndex(sequence,size);PipeBarrier<PIPE_V>();
        ShiftRight(columnIndex,sequence,int32_t(shift),size);PipeBarrier<PIPE_V>();
        Muls(offsets,columnIndex,static_cast<int32_t>(pitch),size);PipeBarrier<PIPE_V>();
        Sub(offsets,sequence,offsets,size);PipeBarrier<PIPE_V>();
        Muls(offsets,offsets,static_cast<int32_t>(count*4),size);
        Muls(columnIndex,columnIndex,int32_t(4),size);PipeBarrier<PIPE_V>();
        Add(offsets,offsets,columnIndex,size);PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(input,source[offset],DataCopyExtParams{1,n*count*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        Fence<HardEvent::MTE2_V>();Gather(mat[first*pitch],input,transposeIndex,0,size);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadPartial(uint32_t batch) {
        // n15 + one RHS fill16 private columns; rows use a16-lane capacity.
        // Only external dimensions vary. Extend A by an independent4I block.
        Duplicate(input,0.0f,pitch*pitch);Fence<HardEvent::V_MTE2>();
        DataCopyPad(input,a[uint64_t(batch)*logicalN*logicalN],
            DataCopyExtParams{uint16_t(logicalN),logicalN*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t(pitch-logicalN),0});
        DataCopyPad(mat[n*pitch],b[broadcast?0:uint64_t(batch)*logicalN],
            DataCopyExtParams{1,logicalN*4,0,0,0},
            DataCopyPadExtParams<float>{true,0,uint8_t(pitch-logicalN),0});
        Fence<HardEvent::MTE2_S>();
        for(uint32_t i=logicalN;i<n;++i)input.SetValue(i*pitch+i,4.0f);
        auto offsets=transposeIndex.ReinterpretCast<int32_t>();
        for(uint32_t i=0;i<8;++i) {
            offsets.SetValue(i,int32_t(i*pitch*4));
            baseIndex.SetValue(i,int32_t(i*pitch*4));
        }
        Fence<HardEvent::S_V>();
        Adds(offsets[8],offsets,int32_t(8*pitch*4),8);
        Adds(baseIndex[8],baseIndex,int32_t(8*pitch*4),8);PipeBarrier<PIPE_V>();
        for(uint32_t filled=pitch;filled<n*pitch;filled*=2) {
            uint32_t len=n*pitch-filled<filled?n*pitch-filled:filled;
            Adds(offsets[filled],offsets,int32_t(filled/pitch*4),len);PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::MTE2_V>();
        Gather(mat,input,transposeIndex,0,n*pitch);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process() {
        uint32_t owners=cores<batches?cores:batches;
        for(uint32_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners) {
    LoadBatch(batch);
            uint64_t activeRows[2]={n==64?~uint64_t(0):(uint64_t(1)<<n)-1,0};
            uint32_t pairFirst=0;
            float pivotThreshold=0.0f;
            #pragma unroll
            for(uint32_t p=0;p<n;++p) {
                factor=mat[p*pitch];
                uint32_t row;float diagonal;
    ChoosePivot(p,activeRows,pairFirst,pivotThreshold,row,diagonal);
                reinterpret_cast<__ubuf__ uint32_t*>(order.GetPhyAddr())[p]=row;
                float inv=1.0f/diagonal;reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr())[p]=inv;reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row]=0.0f;
                Fence<HardEvent::S_V>();
                Gather(pivot,mat,baseIndex.ReinterpretCast<uint32_t>(),row*4,cols);PipeBarrier<PIPE_V>();
                Muls(pivot,pivot,-inv,cols);PipeBarrier<PIPE_V>();
                Brcb(spread,pivot,cols/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
                uint32_t remaining=n+k-p-1;
                MulAddDst(mat[(p+1)*pitch],factor,spread[(p+1)*8],pitch,remaining,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
                PipeBarrier<PIPE_V>();
            }
            auto offsets=transposeIndex.ReinterpretCast<int32_t>();
            Muls(offsets,order.ReinterpretCast<int32_t>(),int32_t(4),n);PipeBarrier<PIPE_V>();
            Gather(output,mat[n*pitch],transposeIndex,0,n);PipeBarrier<PIPE_V>();
            Mul(output,output,inverse,n);Fence<HardEvent::V_MTE3>();
            if(GetBlockIdx()<owners){DataCopyPad(x[uint64_t(batch)*logicalN],output,DataCopyExtParams{1,logicalN*4,0,0,0});if(batch+owners<batches)Fence<HardEvent::MTE3_S>();}
        }
    }
private:
    LocalTensor<float> mat,input,transposed,factor,pivot,spread,inverse,output,absolute,reduced;
    LocalTensor<int32_t> baseIndex;LocalTensor<uint32_t> transposeIndex,order;
    GlobalTensor<float> a,b,x;
    static constexpr uint32_t n=SN,k=SK,pitch=SN<=8?8:SN<=16?16:SN<=32?32:64,cols=(SN+SK+7)/8*8;
    uint32_t batches,cores,logicalN;bool broadcast;
};

#endif
