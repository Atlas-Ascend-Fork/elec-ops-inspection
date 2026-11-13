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

#ifndef RUBY0451_SOLVE_COLUMN_PUBLICATION_H
#define RUBY0451_SOLVE_COLUMN_PUBLICATION_H
#include "solve_column.h"

template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::PublishUnpivotedOwner(uint32_t panel,uint32_t panelEnd,uint32_t end,uint32_t begin,uint32_t localCols){
                    if(panelEnd==end) {
                        // This owner has written every local inverse. Its last
                        // immutable packet flag also publishes this disjoint GM
                        // metadata slice before the RHS owner can assemble X.
                        Gather(inverse,mat,indices,na*4,localCols);
                        Fence<HardEvent::V_MTE3>();
                        DataCopyPad(w[(n+8)*na+begin],inverse,
                                    DataCopyExtParams{1,localCols*4,0,0,0});
                    }
                    PublishPersistentPanel(panel);
}
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::InitializePublication(){
        auto ready=mat[128*na].ReinterpretCast<uint32_t>();
        if(GetBlockIdx()%workers==0) {
            constexpr uint32_t flagWords=FAMILY==1?2048:STATIC_WORKERS==10 && STATIC_K==3?2304:STATIC_WORKERS==8 && STATIC_K==3?2048:1024;
            Duplicate(ready,uint32_t(0),flagWords);
            Fence<HardEvent::V_MTE3>();DataCopy(publication,ready,flagWords);
        }
        // Global initialization joins the zero DMA before its source is reused.
        SyncAll();
        // Discard prior-invocation external ready lines after the zero DMA is joined.
        DataCacheCleanAndInvalid<uint32_t,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(publication);
        Duplicate(ready,uint32_t(1),8);Fence<HardEvent::V_MTE3>();
        // This 32B payload remains immutable through the final publication join.
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::PublishPersistentPanel(uint32_t panel){
        uint32_t index=(panel/(FAMILY==1?4:((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3?4:8)))*32;
        auto ready=mat[128*na].ReinterpretCast<uint32_t>();
        // Previous packet bytes finish before its flag; immutable UB source.
        PipeBarrier<PIPE_MTE3>();DataCopy(publication[index],ready,8);
    }
template<uint32_t FAMILY,uint32_t STATIC_N,uint32_t STATIC_WORKERS,uint32_t STATIC_K,bool PARTIAL_CAPACITY>
__aicore__ inline void KernelColumnSolve<FAMILY,STATIC_N,STATIC_WORKERS,STATIC_K,PARTIAL_CAPACITY>::WaitPersistentPanel(uint32_t panel){
        auto flag=publication[(panel/(FAMILY==1?4:((STATIC_WORKERS==8 || STATIC_WORKERS==10) && STATIC_K==3?4:8)))*32];
        auto ready=reinterpret_cast<volatile const __gm__ uint32_t*>(flag.GetPhyAddr());
        // Each slot only advances 0->1 here; a cached 1 is already current and final.
        while(*ready==0) {
            DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE>(flag);
        }
    }

#endif
