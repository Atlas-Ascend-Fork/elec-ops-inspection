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

#include "det_resident.h"
#include "det_tiny.h"
#include "det_block.h"
#include "det_tiny_bound.h"
#include "det_medium_bound.h"

template<uint32_t TASK_MODE,bool tiny,bool fullVector,bool compact,bool alignedLarge>
__aicore__ inline LinalgDetTilingData ReadDetTiling(GM_ADDR tiling){
 LinalgDetTilingData t;
 if constexpr(tiny){
  t.n=TASK_MODE-1;t.cores=8;t.batches=1;t.resident=t.n<16?1:0;
 }else if constexpr(fullVector){
  if constexpr(TASK_MODE>=160){t.n=58+2*((TASK_MODE-160)/8);t.batches=(TASK_MODE-160)%8+1;}
  else {t.n=64;t.batches=TASK_MODE-32+1;}
  t.cores=8;t.resident=0;
 }else if constexpr(compact){
  t.n=50+2*((TASK_MODE-48)/8);t.cores=8;t.batches=(TASK_MODE-48)%8+1;t.resident=0;
 }else if constexpr(alignedLarge){
  t.n=(17+(TASK_MODE-96)/4)*8;t.cores=16;t.batches=(TASK_MODE-96)%4+1;t.resident=0;
 }else{
  REGISTER_TILING_DEFAULT(LinalgDetTilingData);GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData,data,tiling);t=data;
 }
 return t;
}
template<typename DT_A,uint32_t TASK_MODE>
__global__ __aicore__ void linalg_det(GM_ADDR A,GM_ADDR det,GM_ADDR workspace,GM_ADDR tiling){
 constexpr bool tiny=TASK_MODE>=2&&TASK_MODE<=17;
 constexpr bool fullVector=(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=160&&TASK_MODE<=183);
 constexpr bool compact=TASK_MODE>=48&&TASK_MODE<=79;
 constexpr bool alignedLarge=TASK_MODE>=96&&TASK_MODE<=159;
 auto t=ReadDetTiling<TASK_MODE,tiny,fullVector,compact,alignedLarge>(tiling);
 uint64_t owner=0;
 if constexpr(fullVector||compact){owner=uint64_t(GetBlockIdx());if(owner>=t.batches)return;}
 else if(t.n<=64&&uint64_t(GetBlockIdx())>=t.batches)return;
 if constexpr(TASK_MODE==1||alignedLarge){FastBlockProcess(A,det,GetUserWorkspace(workspace),t.n,t.batches,t.cores/4);}
 else if constexpr(tiny){
  if constexpr(TASK_MODE==17){KernelTinyBoundPrefetch16 op;op.Init(A,det,t);op.Process();}
  else if constexpr(TASK_MODE==16){KernelTinyBoundTerminal15 op;op.Init(A,det,t);op.Process();}
  else if constexpr(TASK_MODE>=10){KernelTinyBoundPrefetchPadded16 op;op.Init(A,det,t);op.Process();}
  else {KernelTinyFullwidthDet op;op.Init(A,det,t);op.Process();}
 }else if constexpr(fullVector){KernelMediumBoundStaticSuffix64 op;op.Init(A,det,t);op.Process(owner);}
 else if constexpr(compact){KernelMediumBoundCompact56 op;op.Init(A,det,t);op.Process(owner);}
 else if(t.n==16){KernelTinyExplicit16 op;op.Init(A,det,t);op.Process();}
 else if(t.n<=16){if(t.resident){KernelTinyFullwidthDet op;op.Init(A,det,t);op.Process();}else{KernelTinyDet op;op.Init(A,det,t);op.Process();}}
 else if(t.n>=33&&t.n<=64){KernelMediumDet op;op.Init(A,det,t);op.Process();}
 else {KernelColumnDet op;op.Init(A,det,t);op.Process();}
}

// The bound singleton kernels use only UB and explicit input/output pointers.
// Keep system workspace registration for all other modes and debug facilities.
// CANN 9 appends its generated wrapper after this source and defines
// TEMPLATE_PARAMS before invoking SetSysWorkspaceForce in that wrapper.
namespace AscendC {
template<typename DT_A, uint32_t TASK_MODE>
__aicore__ inline void DetSetSysWorkspaceForMode(GM_ADDR workspace) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
 SetSysWorkspaceForce(workspace);
#else
 if constexpr(!((TASK_MODE>=2&&TASK_MODE<=17)||(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=48&&TASK_MODE<=79)||(TASK_MODE>=160&&TASK_MODE<=183))) SetSysWorkspaceForce(workspace);
#endif
}
}
#define SetSysWorkspaceForce(workspace) DetSetSysWorkspaceForMode<TEMPLATE_PARAMS>(workspace)

// The same singleton modes perform no cross-core synchronization. These
// adapters affect only the generated calls after the user source is included.
template<typename DT_A, uint32_t TASK_MODE>
__aicore__ inline void DetSetFftsForMode(uint64_t address) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
 set_ffts_base_addr(address);
#else
 if constexpr(!((TASK_MODE>=2&&TASK_MODE<=17)||(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=48&&TASK_MODE<=79)||(TASK_MODE>=160&&TASK_MODE<=183))) set_ffts_base_addr(address);
#endif
}
template<typename DT_A, uint32_t TASK_MODE>
__aicore__ inline void DetGeneratedPreloadForMode(uint64_t mode) {
 (void)mode; // The audited CANN9 generated call always passes1.
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
 icache_preload(1);
#else
 if constexpr(!((TASK_MODE>=2&&TASK_MODE<=17)||(TASK_MODE>=32&&TASK_MODE<=39)||(TASK_MODE>=48&&TASK_MODE<=79)||(TASK_MODE>=160&&TASK_MODE<=183))) icache_preload(1);
#endif
}
#define set_ffts_base_addr(address) DetSetFftsForMode<TEMPLATE_PARAMS>(address)
#define icache_preload(mode) DetGeneratedPreloadForMode<TEMPLATE_PARAMS>(mode)
