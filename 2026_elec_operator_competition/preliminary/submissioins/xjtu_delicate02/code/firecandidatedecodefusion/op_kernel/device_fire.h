/*
 * Copyright (c) 2026 Delicate02. All rights reserved.
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
#include "fire_small_decode.h"
#include "fire_dense_fallback.h"
#include "fire_mask_decode.h"
#include "fire_flat_decode.h"
#include "blocked_transpose.h"

template<class T,uint32_t K,bool Early>
__aicore__ __attribute__((noinline)) void RunDenseShortN(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile) {
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile};
    MaskDecode<T,K,Early> op;op.Init(r,p,o,t);op.Process();
}

template<class T,uint32_t K,bool Early>
__aicore__ __attribute__((noinline)) void RunDenseManyTiles(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile,
    uint32_t unitsPerWorker,uint32_t extraUnits,uint32_t totalRows) {
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile,
        unitsPerWorker,extraUnits,0,totalRows};
    FlatMaskDecode<T,K,Early> op;op.Init(r,p,o,t);op.Process();
}

template<class T,uint32_t K,bool Early>
__aicore__ inline void RunDenseDecode(GM_ADDR r,GM_ADDR p,GM_ADDR o,
    FireCandidateDecodeFusionTilingData t) {
    if constexpr(K<=1024) {
        if(__builtin_expect(t.denseOne!=0,true)) {
            FlatMaskDecode<T,K,Early> op;op.template Init<true>(r,p,o,t);op.ProcessOne();
        } else if(t.n>=K) {
            RunDenseManyTiles<T,K,Early>(r,p,o,t.batch,t.n,t.blocks,t.tile,
                t.unitsPerWorker,t.extraUnits,t.totalRows);
        } else {
            RunDenseShortN<T,K,Early>(r,p,o,t.batch,t.n,t.blocks,t.tile);
        }
    } else {BlockedTransposeDecode<T,K> op;op.Init(r,p,o,t);op.Process();}
}
