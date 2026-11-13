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

#ifndef RUBY0451_DET_BLOCK_PAIR_H
#define RUBY0451_DET_BLOCK_PAIR_H
#include "det_common.h"
template<uint32_t ColumnBegin,uint32_t ActiveWidth,bool ODD>
__aicore__ inline void DetTailPairPivot(LocalTensor<float> mat,LocalTensor<float> paired,LocalTensor<float> pairSpread,LocalTensor<int32_t> bases,uint32_t n,uint32_t p,float& product,float& cachedDiagonal,float& cachedInverse){
constexpr uint32_t pitch=72;
    auto factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(!ODD){
     // The event covers prior matrix producers; Gather and scalar reads only read them.
     event_t priorMatrixReady=static_cast<event_t>(0);
     SetFlag<HardEvent::V_S>(priorMatrixReady);
     if(remaining){
      // Matrix plus the initialized input copy are owned by this core.
      // Extra gathered words never enter a live coefficient or matrix update.
      // Gather only the two live pivot coefficients for each trailing column.
      auto* dst=reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr());
      auto* src=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(p+1)*pitch;
      auto* offsets=reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr());
      uint64_t firstMask[2]={remaining>=32?~uint64_t(0):(uint64_t(1)<<(2*remaining))-1,0};
      AscendC::GatherImpl<float>(dst,src,offsets,4608,p*4,firstMask,uint8_t(1),uint16_t(8));
      if(remaining>32){
       uint64_t secondMask[2]={(uint64_t(1)<<(2*remaining-64))-1,0};
       AscendC::GatherImpl<float>(dst+64,src,offsets+64,4608,p*4,secondMask,uint8_t(1),uint16_t(8));
      }
     }
     WaitFlag<HardEvent::V_S>(priorMatrixReady);
     diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[p];
     if(remaining){
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[p+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }
    }else diagonal=cachedDiagonal;
    // No per-pivot zero guard in this bound performance path.
    DetUpdatePairColumns<ColumnBegin,ActiveWidth,ODD>(mat,factor,paired,pairSpread,p,remaining,diagonal,below,cross,future,product,cachedDiagonal,cachedInverse);
}
template<uint32_t ColumnBegin,uint32_t ActiveWidth>
 __aicore__ inline void DetTailPairEliminate(LocalTensor<float> mat,LocalTensor<float> paired,LocalTensor<float> pairSpread,LocalTensor<int32_t> bases,uint32_t n,uint32_t first,uint32_t last,float& product){
  constexpr uint32_t pitch=72;
  float cachedDiagonal=0.0f,cachedInverse=0.0f;
   for(uint32_t pairBase=first;pairBase<last;pairBase+=2){
    {const uint32_t p=pairBase;
    DetTailPairPivot<ColumnBegin,ActiveWidth,false>(mat,paired,pairSpread,bases,n,p,product,cachedDiagonal,cachedInverse);
    }
    {const uint32_t p=pairBase+1;
    DetTailPairPivot<ColumnBegin,ActiveWidth,true>(mat,paired,pairSpread,bases,n,p,product,cachedDiagonal,cachedInverse);
    }
   }
 }


#endif
