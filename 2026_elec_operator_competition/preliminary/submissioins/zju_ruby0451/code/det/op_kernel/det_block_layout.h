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

#ifndef RUBY0451_DET_BLOCK_LAYOUT_H
#define RUBY0451_DET_BLOCK_LAYOUT_H
#include "det_common.h"
#include "det_resident.h"
#include "det_tiny.h"
class KernelLinalgDet {
public:
 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,GM_ADDR W,const LinalgDetTilingData& t){a=A;y=Y;w=W;data=t;}
 __aicore__ inline void Process(){
  if(data.n==16){KernelTinyExplicit16 op;op.Init(a,y,data);op.Process();}
 else if(data.n<=16){if(data.resident){KernelTinyFullwidthDet op;op.Init(a,y,data);op.Process();}else{KernelTinyDet op;op.Init(a,y,data);op.Process();}}
  else if(data.n>=33&&data.n<=64){KernelMediumDet op;op.Init(a,y,data);op.Process();}
  else if(data.n<=128){KernelColumnDet op;op.Init(a,y,data);op.Process();}
  else {KernelCooperativeDet op;op.Init(a,y,w,data);op.Process();}
 }
private: GM_ADDR a;GM_ADDR y;GM_ADDR w;LinalgDetTilingData data;
};

#define DET_FAST_VECTOR_FINISH 1
// Experimental complete FP32 blocked path. A singular diagonal block produces NaN;
// there is no precision admission or fallback in this speed/error experiment.
// TPipe's three L0 TBuf allocations seed M_MTE1 events 0,1,2.
// Manual reuse acknowledgements must not consume those initial tokens.
template<HardEvent E> __aicore__ inline void FastFence(){
 constexpr event_t id=E==HardEvent::M_MTE1?EVENT_ID4:EVENT_ID0;
 SetFlag<E>(id);WaitFlag<E>(id);
}
__aicore__ inline uint32_t FastParity(uint64_t x){x^=x>>32;x^=x>>16;x^=x>>8;x^=x>>4;return(0x6996u>>(uint32_t(x)&15))&1;}
__aicore__ inline void FastLoadLocal(LocalTensor<float>l1,LocalTensor<float>l0,uint32_t rows,uint32_t cols){
 LoadData3DParamsV2<float>p;p.l1H=1;p.l1W=rows;p.channelSize=cols;p.kExtension=cols;p.mExtension=rows;
 p.strideW=1;p.strideH=1;p.filterW=1;p.filterH=1;p.dilationFilterW=1;p.dilationFilterH=1;p.enTranspose=false;LoadData(l0,l1,p);
}
// Block LU in A = L * U form, with block-unit diagonal U. Reconstruct only
// the current cross from original A and all previously published block factors.
struct DetLeftJob { uint32_t batch,row,col,m,n; };
__aicore__ inline uint32_t DetLeftColumnWidth(uint32_t size,uint32_t start,uint32_t slots,uint32_t cores){
 // Host scheduling supplies a positive core count on supported 910B devices.
 if(cores==0)return 64;
 uint32_t remain=size-start;if(remain<=64)return 64;
 uint32_t rows=(remain+127)/128,oldJobs=(rows+(remain-32+63)/64)*slots,newJobs=(rows+(remain-32+127)/128)*slots;
 return (newJobs+cores-1)/cores<(oldJobs+cores-1)/cores?128:64;
}
__aicore__ inline uint32_t DetLeftJobs(uint32_t size,uint32_t start,uint32_t slots,uint32_t cores){
 if(size-start<=64)return 1;
 uint32_t width=DetLeftColumnWidth(size,start,slots,cores);
 // The width selector returns 64 or 128; retain a local divisor check.
 if(width==0)return 0;
 return (size-start+127)/128+(size-start-32+width-1)/width;
}
__aicore__ inline DetLeftJob DetLeftTile(uint32_t size,uint32_t start,uint32_t job,uint32_t slots,uint32_t cores){
 uint32_t count=DetLeftJobs(size,start,slots,cores);
 if(count==0)return DetLeftJob{};
 uint32_t index=job%count;
 DetLeftJob t;t.batch=job/count;
 if(size-start<=64){t.row=start;t.col=start;t.m=(size-start+31)/32*32;t.n=t.m;return t;}
 uint32_t rowJobs=(size-start+127)/128,width=DetLeftColumnWidth(size,start,slots,cores);
 if(index<rowJobs){
  t.row=start;t.col=start+index*128;t.m=32;
  uint32_t live=size-t.col;if(live>128)live=128;t.n=(live+31)/32*32;
 }else{
  t.row=start+32+(index-rowJobs)*width;t.col=start;t.n=32;
  uint32_t live=size-t.row;if(live>width)live=width;t.m=(live+31)/32*32;
 }
 return t;
}


__aicore__ inline uint32_t DetOwnerRhsChunk(uint32_t n,uint32_t start){
 uint32_t chunk=(n-start-32+63)/64*8;
 return n<=320&&chunk<32?chunk:32;
}



#endif
