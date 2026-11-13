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

#ifndef RUBY0451_SOLVE_COLUMN_WINDOW_1_H
#define RUBY0451_SOLVE_COLUMN_WINDOW_1_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent,bool retiredMetadata>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::GatherWindowMetadata(uint32_t lane,LocalTensor<float> packetView,uint32_t panel,uint32_t steps){
            if((!retiredMetadata && lane==0) || (!persistent && FAMILY==3 && lane<k))
                Gather(inverse[panel],packetView,indices,na*4,steps);
            if(!retiredMetadata && (persistent?lane==0:lane<k))
                Gather(order[panel].ReinterpretCast<float>(),packetView,indices,(na+1)*4,steps);
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool retiredMetadata>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::JoinCollapsedOwner(){
                    if constexpr(retiredMetadata)PrepareCompactThreeIndices();
                    Fence<HardEvent::MTE3_MTE2>();
                    if constexpr(retiredMetadata) {
                        for(uint32_t producer=1;producer<STATIC_WORKERS;++producer)
                            WaitPersistentPanel(192+producer*8);
                        DataCopy(inverse,w[(n+12)*na],192);
                        DataCopy(order.ReinterpretCast<float>(),w[(n+13)*na],192);
                        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
                    } else {WaitPersistentPanel(240);WaitPersistentPanel(248);}
                    FactorCollapsedTail<192>();
                    // Every prefix producer was already joined above. Slot48
                    // now covers the complete tail/RHS/inverse/order stores.
                    if constexpr(retiredMetadata)PublishPersistentPanel(192);
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<uint32_t PREFIX>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ChooseCollapsedPivot(LocalTensor<float> absolute,LocalTensor<float> reduced,LocalTensor<float> f,uint64_t* active,uint32_t rowBase,uint32_t firstRow,uint32_t& row,float& diagonal){
            union {float value;uint32_t bits;} word;word.value=diagonal;
            if((word.bits&0x7fffffffu)<0x40000000u) {
                Abs(absolute,f[firstRow],64);PipeBarrier<PIPE_V>();
                WholeReduceMax(reduced,absolute,active,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                Fence<HardEvent::V_S>();
                row=rowBase+reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                diagonal=reinterpret_cast<__ubuf__ float*>(f.GetPhyAddr())[row-PREFIX];
            }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<uint32_t PREFIX,uint32_t rows,uint32_t pitch>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::StoreCollapsedFactors(LocalTensor<float> matrix){
        Fence<HardEvent::V_S>();
        SetMaskCount();uint32_t currentStart=rows;
        for(uint32_t p=PREFIX;p+1<n;p+=2) {
            uint32_t start=p/64*64-PREFIX;
            if(start!=currentStart) {
                SetVectorMask<float,MaskMode::COUNTER>(0,rows-start);currentStart=start;
            }
            auto left=matrix[(p-PREFIX)*pitch],right=matrix[(p+1-PREFIX)*pitch];
            uint32_t row1=order.GetValue(p+1)-PREFIX;
            float cross=reinterpret_cast<__ubuf__ float*>(left.GetPhyAddr())[row1];
            Axpy<float,float,false>(left[start],right[start],cross,uint64_t(0),1,
                UnaryRepeatParams{1,1,8,8});
        }
        ResetMask();SetMaskNorm();
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(w[PREFIX*na+PREFIX],matrix,
            DataCopyExtParams{uint16_t(rows),rows*4,1,PREFIX*4,0});
        DataCopyPad(w[(n+8)*na+PREFIX],matrix[rows*pitch],
            DataCopyExtParams{uint16_t(k),rows*4,1,PREFIX*4,0});
        // Retired packet space is disjoint from RHS n+8..n+15.
        Fence<HardEvent::S_MTE3>();
        DataCopy(w[(n+16)*na],inverse,na);
        DataCopy(w[(n+17)*na],order.ReinterpretCast<float>(),na);

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<uint32_t rows,uint32_t pitch>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::UpdateCollapsedColumns(uint32_t p,uint32_t cols,uint32_t firstRow,float inv,LocalTensor<float> f,LocalTensor<float> trailing){
            if constexpr(rows==64) {
                // Explicit normal-mode 64-lane normalization leaves the same mask
                // for the independent-column updates; no count-mode transition.
                Muls<float>(f,f,-inv,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                if(p+1<n) {
                    MulAddDst<float,float,false>(trailing,f,transposed,uint64_t(64),1,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                    SetFlag<HardEvent::V_S>(static_cast<event_t>(1));
                    MulAddDst<float,float,false>(trailing[pitch],f,transposed[8],uint64_t(64),cols-1,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                } else {
                    MulAddDst<float,float,false>(trailing,f,transposed,uint64_t(64),cols,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                }
            } else {
                Muls(f[firstRow],f[firstRow],-inv,rows-firstRow);PipeBarrier<PIPE_V>();
                for(uint32_t base=firstRow;base<rows;base+=64) {
                    MulAddDst(trailing[base],f[base],transposed,64,cols,
                        BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
                }
            }
            if constexpr(rows!=64)PipeBarrier<PIPE_V>();

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::LoadWindowColumns(uint32_t batch,uint32_t lane,uint32_t rhsLane,uint32_t begin,uint32_t localCols){
        if constexpr(FAMILY==3 && STATIC_N==256 && STATIC_K==3) {
            uint32_t validCols=begin<logicalN?(logicalN-begin<localCols?logicalN-begin:localCols):0;
            LoadCapacityRange(a,uint64_t(batch)*logicalN*logicalN+begin,logicalN,localCols,0,validCols,begin);
            if(lane==rhsLane)
                LoadCapacityRange(b,broadcast?0:uint64_t(batch)*logicalN*logicalK,logicalK,k,localCols,logicalK,n);
        } else {
            LoadRange(a,uint64_t(batch)*n*n+begin,n,localCols,0);
            if(lane==rhsLane)LoadRange(b,broadcast?0:uint64_t(batch)*n*k,k,k,localCols);
        }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent,bool retiredMetadata>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::JoinWindowTail(uint32_t lane){
        if constexpr(persistent) {
            if constexpr((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3) {
                if constexpr(retiredMetadata) {
                    // One ordered flag joins each nonzero lane's canonical data,
                    // compact metadata and (on lane7) the complete RHS stream.
                    if(lane>0)PublishPersistentPanel(192+lane*8);
                } else if(lane>=(STATIC_WORKERS==10?8:6))PublishPersistentPanel(192+lane*8);
                if(lane==0) {
    JoinCollapsedOwner<retiredMetadata>();
                }
            } else {
                SyncAll();
                if(lane==0)FactorCollapsedTail<192>();
            }
        }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool retiredMetadata>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::StoreWindowFactors(uint32_t begin,uint32_t localCols,uint32_t lane,uint32_t rhsLane){
        if constexpr(retiredMetadata) {
            if(lane<(STATIC_WORKERS==10?8:6)) {
                // Retired local columns still contain their inverse/order words.
                // Staging n+12/n+13 is disjoint from live prefix packets at n+16.
                Gather(inverse[begin],mat,indices,na*4,localCols);
                Gather(order[begin].ReinterpretCast<float>(),mat,indices,(na+1)*4,localCols);
                Fence<HardEvent::V_MTE3>();
                DataCopy(w[(n+12)*na+begin],inverse[begin],localCols);
                DataCopy(w[(n+13)*na+begin],order[begin].ReinterpretCast<float>(),localCols);
            }
        }
        {Fence<HardEvent::V_MTE3>();DataCopyPad(w[begin*na],mat,DataCopyExtParams{static_cast<uint16_t>(localCols),na*4,1,0,0});}
        if(lane==rhsLane) {
            if constexpr(FAMILY==3) {
                Fence<HardEvent::V_MTE3>();
                DataCopyPad(w[(n+8)*na],mat[localCols*cp],
                            DataCopyExtParams{static_cast<uint16_t>(k),na*4,1,0,0});
            } else Assemble(mat[localCols*cp],output,false,cp);
        }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ComposeWindowPairs(uint32_t begin,uint32_t end){
            Fence<HardEvent::V_S>();
            SetMaskCount();
            uint32_t currentStart=na;
            for(uint32_t p=begin;p+1<end && (!persistent || p+1<192);p+=2) {
                uint32_t start=p/64*64;
                if(start!=currentStart) {
                    SetVectorMask<float,MaskMode::COUNTER>(0,na-start);
                    currentStart=start;
                }
                auto left=mat[(p-begin)*cp],right=mat[(p+1-begin)*cp];
                uint32_t row1=reinterpret_cast<__ubuf__ uint32_t*>(right.GetPhyAddr())[na+1];
                float cross=reinterpret_cast<__ubuf__ float*>(left.GetPhyAddr())[row1];
                Axpy<float,float,false>(left[start],right[start],cross,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
            }
            ResetMask();SetMaskNorm();

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ComposeWindowQuartets(uint32_t begin,uint32_t end){
            Fence<HardEvent::V_S>();SetMaskCount();
            uint32_t currentStart=na;
            for(uint32_t p=begin;p<end && p<192;p+=4) {
                uint32_t start=p/64*64;
                if(start!=currentStart) {
                    SetVectorMask<float,MaskMode::COUNTER>(0,na-start);
                    currentStart=start;
                }
                auto f0=mat[(p-begin)*cp],f1=mat[(p+1-begin)*cp];
                auto f2=mat[(p+2-begin)*cp],f3=mat[(p+3-begin)*cp];
                uint32_t r1=reinterpret_cast<__ubuf__ uint32_t*>(f1.GetPhyAddr())[na+1];
                uint32_t r2=reinterpret_cast<__ubuf__ uint32_t*>(f2.GetPhyAddr())[na+1];
                uint32_t r3=reinterpret_cast<__ubuf__ uint32_t*>(f3.GetPhyAddr())[na+1];
                float c01=reinterpret_cast<__ubuf__ float*>(f0.GetPhyAddr())[r1];
                float c02=reinterpret_cast<__ubuf__ float*>(f0.GetPhyAddr())[r2];
                float c03=reinterpret_cast<__ubuf__ float*>(f0.GetPhyAddr())[r3];
                float c12=reinterpret_cast<__ubuf__ float*>(f1.GetPhyAddr())[r2];
                float c13=reinterpret_cast<__ubuf__ float*>(f1.GetPhyAddr())[r3];
                float c23=reinterpret_cast<__ubuf__ float*>(f2.GetPhyAddr())[r3];
                Axpy<float,float,false>(f2[start],f3[start],c23,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f1[start],f2[start],c12,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f1[start],f3[start],c13,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f0[start],f1[start],c01,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f0[start],f2[start],c02,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(f0[start],f3[start],c03,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
                PipeBarrier<PIPE_V>();
            }
            ResetMask();SetMaskNorm();

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent,bool retiredMetadata,bool boundaryReady>
__aicore__ inline bool KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ProcessWindowPanel(uint32_t panel,uint32_t panelWidth,uint32_t window,uint32_t begin,uint32_t end,uint32_t lane,uint32_t rhsLane,uint32_t localCols,uint64_t* activeRows,bool& pendingBoundary,LocalTensor<float> absolute,LocalTensor<float> reduced){
            uint32_t rowBase=panel/window*window;
            if(panel%window==0)activeRows[0]=n-rowBase>=window?((~uint64_t(0))>>(64-window)):(uint64_t(1)<<(n-rowBase))-1;
            factor=baseFactor;
            uint32_t panelEnd=n-panel<panelWidth?n:panel+panelWidth,steps=panelEnd-panel;
            bool owner=panel>=begin && panel<end;
            uint32_t packet=(n+16)*na+(persistent?panel:((panel/8)%2)*8)*(na+8);
            if(owner) {
    PublishWindowOwner<persistent,boundaryReady>(panel,panelEnd,begin,rowBase,window,packet,steps,activeRows,pendingBoundary,absolute,reduced);
            }
            if constexpr(!persistent)SyncAll();
            if constexpr(FAMILY==3) {
                // The collapsed-tail owner publishes the complete inverse/order copy.
                // Other lanes reload it before backsolve and need no retired packets.
                if constexpr(retiredMetadata) {if(end<=panel)return false;}
                else if constexpr(persistent) {if(end<=panel && lane!=0)return true;}
                else {if(end<=panel && lane>=k)return true;}
            }
            if(!owner){
                if constexpr(persistent) {
                    WaitPersistentPanel(panel);
                    // The single receiver slot must outlive its preceding V reads.
                    Fence<HardEvent::V_MTE2>();
                }
                DataCopy(input,w[packet],steps*(na+8));Fence<HardEvent::MTE2_V>();
            }
            auto packetView=owner?mat[(panel-begin)*cp]:input;
    GatherWindowMetadata<persistent,retiredMetadata>(lane,packetView,panel,steps);
            Fence<HardEvent::V_S>();
    UpdateWindowColumns<boundaryReady>(panel,panelEnd,begin,end,lane,rhsLane,localCols,rowBase,owner,activeRows,pendingBoundary);
return true;
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool boundaryReady>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::UpdateWindowColumns(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t end,uint32_t lane,uint32_t rhsLane,uint32_t localCols,uint32_t rowBase,bool owner,uint64_t* activeRows,bool& pendingBoundary){
            for(uint32_t p=panel;p<panelEnd;++p) {
                factor=owner?mat[(p-begin)*cp]:input[(p-panel)*cp];
                uint32_t row=reinterpret_cast<__ubuf__ uint32_t*>(factor.GetPhyAddr())[na+1];
                activeRows[0]&=~(uint64_t(1)<<(row-rowBase));
                uint32_t first=panelEnd>begin?panelEnd:begin;
                uint32_t rhsCols=lane==rhsLane?k:0;
                if constexpr(boundaryReady) {
                    if(p+1==panelEnd && panelEnd>=begin && panelEnd<end && panelEnd<192) {
                        UpdateColumnsWindowReady(mat[(first-begin)*cp],end-first+rhsCols,row,panelEnd/64*64);
                        pendingBoundary=true;
                    } else if(first<end)UpdateColumns(mat[(first-begin)*cp],end-first+rhsCols,row,0.0f);
                    else if(rhsCols)UpdateColumns(mat[localCols*cp],rhsCols,row,0.0f);
                } else {
                    if(first<end)UpdateColumns(mat[(first-begin)*cp],end-first+rhsCols,row,0.0f);
                    else if(rhsCols)UpdateColumns(mat[localCols*cp],rhsCols,row,0.0f);
                }
            }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent,bool boundaryReady>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::PublishWindowOwner(uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t rowBase,uint32_t window,uint32_t packet,uint32_t steps,uint64_t* activeRows,bool& pendingBoundary,LocalTensor<float> absolute,LocalTensor<float> reduced){
                Fence<HardEvent::S_V>();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,FAMILY==3?na-rowBase:na);
                for(uint32_t p=panel;p<panelEnd;++p) {
    NormalizeWindowPivot<persistent,boundaryReady>(p,panel,panelEnd,begin,rowBase,window,activeRows,pendingBoundary,absolute,reduced);
                }
                ResetMask();SetMaskNorm();
                // Metadata feeds both this owner's inverse/order Gather and packet DMA.
                // S_V orders Gather;the following V_MTE3 also orders its DMA read.
                if constexpr(persistent)Fence<HardEvent::S_V>();
                Fence<HardEvent::V_MTE3>();DataCopy(w[packet],mat[(panel-begin)*cp],steps*cp);
                if constexpr(persistent)PublishPersistentPanel(panel);

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent,bool boundaryReady>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::NormalizeWindowPivot(uint32_t p,uint32_t panel,uint32_t panelEnd,uint32_t begin,uint32_t rowBase,uint32_t window,uint64_t* activeRows,bool& pendingBoundary,LocalTensor<float> absolute,LocalTensor<float> reduced){
                    factor=mat[(p-begin)*cp];
                    uint32_t rowCount=n-rowBase<window?n-rowBase:window;
    WaitWindowPivot<persistent,boundaryReady>(p,panel,pendingBoundary);
                    uint64_t firstBit=activeRows[0]&(~activeRows[0]+1);
                    uint32_t row=rowBase+63-static_cast<uint32_t>(CountLeadingZero(firstBit));
                    float diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
    ChooseWindowPivot<persistent>(rowBase,rowCount,activeRows,absolute,reduced,row,diagonal);
                    if constexpr(persistent) {
                        factor.SetValue(row,0.0f);
                        SetFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    }
                    float inv=1.0f/diagonal;
                    auto packetView=factor;
                    packetView.SetValue(na,inv);packetView.ReinterpretCast<uint32_t>().SetValue(na+1,row);
                    activeRows[0]&=~(uint64_t(1)<<(row-rowBase));
                    if constexpr(!persistent)factor.SetValue(row,0.0f);
                    // Other columns may still be completing the preceding pivot.
                    // Join them before reading their new scalar coefficients.
    JoinWindowCoefficients<persistent,boundaryReady>(p,panel,pendingBoundary);
                    if constexpr(persistent)WaitFlag<HardEvent::S_V>(static_cast<event_t>(1));
                    else Fence<HardEvent::S_V>();
                    if constexpr(persistent)
                        NormalizeOwnedPanelEarly(mat[(p+1-begin)*cp],panelEnd-p-1,row,inv);
                    else NormalizeOwnedPanel(mat[(p+1-begin)*cp],panelEnd-p-1,row,inv);

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent,bool boundaryReady>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::JoinWindowCoefficients(uint32_t p,uint32_t panel,bool& pendingBoundary){
                    if constexpr(persistent) {
                        if(p!=panel)WaitFlag<HardEvent::V_S>(static_cast<event_t>(2));
                        else if constexpr(boundaryReady) {
                            if(pendingBoundary) {
                                // Scalar coefficients read only the completed pivot window.
                                WaitFlag<HardEvent::V_S>(static_cast<event_t>(4));
                                // Normalization reads the entire factor suffix on V.
                                PipeBarrier<PIPE_V>();
                                pendingBoundary=false;
                            }
                        }
                    }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent,bool boundaryReady>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::WaitWindowPivot(uint32_t p,uint32_t panel,bool pendingBoundary){
                    if constexpr(persistent) {
                        if(p==panel) {
                            if constexpr(boundaryReady) {
                                if(pendingBoundary)WaitFlag<HardEvent::V_S>(static_cast<event_t>(3));
                                else Fence<HardEvent::V_S>();
                            } else Fence<HardEvent::V_S>();
                        }
                        else WaitFlag<HardEvent::V_S>(static_cast<event_t>(1));
                    } else Fence<HardEvent::V_S>();

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
template<bool persistent>
 __aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::ChooseWindowPivot(uint32_t rowBase,uint32_t rowCount,uint64_t* activeRows,LocalTensor<float> absolute,LocalTensor<float> reduced,uint32_t& row,float& diagonal){
                    bool search;
                    if constexpr(persistent) {
                        union {float value;uint32_t bits;} word;word.value=diagonal;
                        search=(word.bits&0x7fffffffu)<0x40000000u;
                    } else {
                        float magnitude=diagonal<0.0f?-diagonal:diagonal;
                        search=magnitude<2.0f;
                    }
                    if(search) {
                        Abs(absolute,factor[rowBase],rowCount);PipeBarrier<PIPE_V>();
                        WholeReduceMax(reduced,absolute,activeRows,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
                        Fence<HardEvent::V_S>();
                        row=rowBase+reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];
                        diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[row];
                        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,FAMILY==3?na-rowBase:na);
                    }

}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::FactorWindow(){
        constexpr bool persistent=FAMILY==3 && STATIC_N==256;
        constexpr bool retiredMetadata=persistent && (STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3;
        constexpr uint32_t rhsLane=retiredMetadata?STATIC_WORKERS-1:0;
        constexpr uint32_t panelWidth=retiredMetadata?4:8;
        if constexpr(persistent)InitializePublication();
        constexpr uint32_t window=64;
        uint32_t batch=GetBlockIdx()/workers,lane=GetBlockIdx()%workers,split=(n+31)/32*8;
        uint32_t blocks=(n+7)/8;
        uint32_t begin=workers==4?lane*split:(blocks*lane/workers)*8;
        uint32_t end=workers==4?begin+split:(blocks*(lane+1)/workers)*8;if(end>n)end=n;
        if constexpr(STATIC_WORKERS==10 && STATIC_K==3) {
            // Eight owners cover the192-column prefix; the two tail owners
            // retain32 columns each, keeping their trailing-update load fixed.
            begin=lane<8?lane*24:192+(lane-8)*32;
            end=begin+(lane<8?24:32);
        }
        uint32_t localCols=end-begin;
    LoadWindowColumns(batch,lane,rhsLane,begin,localCols);
        auto absolute=indices[2*na].ReinterpretCast<float>();
        auto reduced=indices[3*na].ReinterpretCast<float>();
        auto baseIndex=indices.ReinterpretCast<int32_t>();
        CreateVecIndex(baseIndex,int32_t(0),72);PipeBarrier<PIPE_V>();
        Muls(baseIndex,baseIndex,int32_t(cp*4),72);
        PipeBarrier<PIPE_V>();
        constexpr bool boundaryReady=retiredMetadata;
        bool pendingBoundary=false;
        uint64_t activeRows[2]={0,0};
        for(uint32_t panel=0;panel<(persistent?192:n);panel+=panelWidth) {
if(!ProcessWindowPanel<persistent,retiredMetadata,boundaryReady>(panel,panelWidth,window,begin,end,lane,rhsLane,localCols,activeRows,pendingBoundary,absolute,reduced))break;
        }
        if constexpr(persistent)Fence<HardEvent::MTE3_V>();
        factor=baseFactor;
        if constexpr(FAMILY==3 && STATIC_N==256) {
            // Publication DMA is joined above; only retired canonical factors change.
    ComposeWindowQuartets(begin,end);
        } else if constexpr(FAMILY==3) {
    ComposeWindowPairs<persistent>(begin,end);
        }
    StoreWindowFactors<retiredMetadata>(begin,localCols,lane,rhsLane);
    JoinWindowTail<persistent,retiredMetadata>(lane);
    }

#endif
