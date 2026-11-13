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

#include "solve_generic.h"
#include "solve_column.h"
#include "solve_small.h"

template<uint32_t N_VALUE>
__aicore__ inline void RunTinySolve(GM_ADDR A,GM_ADDR B,GM_ADDR X,const LinalgSolveTilingData& tiling_data){
        if constexpr(N_VALUE==15) {
            // One16-column capacity covers n9..15 and one RHS. Full-capacity
            // inputs retain static extents; smaller inputs use bounded GM IO.
            if(tiling_data.n==N_VALUE) {
                KernelSmallFixed<15,1> op;
                op.Init(A,B,X,tiling_data);op.Process();
            } else {
                KernelSmallFixed<15,1,true> op;
                op.Init(A,B,X,tiling_data);op.Process();
            }
        } else if constexpr(N_VALUE>0 && N_VALUE<=16) {
            KernelSmallFixed<N_VALUE,1> op;
            op.Init(A,B,X,tiling_data);op.Process();
        } else {
            KernelSmallOuter op;op.Init(A,B,X,tiling_data);op.Process();
        }
}
template<uint32_t TASK_MODE>
__aicore__ inline void RunCapacitySolve(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,const LinalgSolveTilingData& tiling_data){
        // One registered capacity template handles both full and partial tiles.
        // Static full-tile extents remove tail bookkeeping from the hot path.
        if(tiling_data.n==256 && tiling_data.k==3) {
            KernelColumnSolve<3,256,TASK_MODE==8?10:8,3,false> op;
            op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);op.Process();
        } else {
            KernelColumnSolve<3,256,TASK_MODE==8?10:8,3,true> op;
            op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);op.Process();
        }
}
template<class DT_A,uint32_t N_VALUE>
__aicore__ inline void RunGeneralSolve(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,const LinalgSolveTilingData& tiling_data){
        if constexpr(sizeof(DT_A)==4) {
            if(tiling_data.n<=256 && tiling_data.k<=8) {
                KernelColumnSolve<> op;
                op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);
                op.Process();return;
            }
        }
        KernelLinalgSolve<DT_A, N_VALUE> op;
        op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);
        op.Process();
}
template<typename DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__global__ __aicore__ void linalg_solve(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,GM_ADDR tiling) {
    if constexpr(TASK_MODE>=64 && TASK_MODE<80) {
        // Full private row/column tiles bind batch metadata for one8-core wave.
        LinalgSolveTilingData data{};
        data.n=N_VALUE;data.k=1;data.batches=(TASK_MODE-64)/2+1;
        data.cores=8;data.broadcastB=(TASK_MODE-64)&1u;
        KernelSmallFixed<N_VALUE,1> op;
        op.Init(A,B,X,data);op.Process();
    } else {
    REGISTER_TILING_DEFAULT(LinalgSolveTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,tiling_data,tiling);
    if constexpr(sizeof(DT_A)==4 && TASK_MODE==3) {
        RunTinySolve<N_VALUE>(A,B,X,tiling_data);
    } else if constexpr(sizeof(DT_A)==4 && TASK_MODE==2) {
        if(tiling_data.n>32) {
            KernelSmallOuterT<N_VALUE==32 || N_VALUE==64,true> op;
            op.Init(A,B,X,tiling_data);op.Process();
        } else {
        KernelSmallOuterT<N_VALUE==32 || N_VALUE==64> op;
        op.Init(A,B,X,tiling_data);op.Process();
        }
    } else if constexpr(sizeof(DT_A)==4 && (TASK_MODE==7 || TASK_MODE==8)) {
        RunCapacitySolve<TASK_MODE>(A,B,X,workspace,tiling_data);
    } else if constexpr(sizeof(DT_A)==4 && (TASK_MODE==1 || TASK_MODE==4 || TASK_MODE==5 || TASK_MODE==6)) {
        KernelColumnSolve<(TASK_MODE==5 || TASK_MODE==6)?3:TASK_MODE==4?2:1,
                          (TASK_MODE==5 || TASK_MODE==6)?N_VALUE:0,TASK_MODE==6?8:0> op;
        op.Init(A,B,X,GetUserWorkspace(workspace),tiling_data);
        op.Process();
    } else {
        RunGeneralSolve<DT_A,N_VALUE>(A,B,X,workspace,tiling_data);
    }
    }
}

// Small/medium AIV entries use only explicit A/B/X pointers and private UB.
// Retain generated system initialization for every other mode and debug build.
// CANN9 appends its wrapper after this source; TEMPLATE_PARAMS is defined there.
namespace AscendC {
template<class DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__aicore__ inline void SolveSetSysWorkspaceForMode(GM_ADDR workspace) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
    SetSysWorkspaceForce(workspace);
#else
    if constexpr(!(sizeof(DT_A)==4 && (TASK_MODE==2 || TASK_MODE==3 || (TASK_MODE>=64 && TASK_MODE<80)))) SetSysWorkspaceForce(workspace);
#endif
}
}
template<class DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__aicore__ inline void SolveSetFftsForMode(uint64_t address) {
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
    set_ffts_base_addr(address);
#else
    if constexpr(!(sizeof(DT_A)==4 && (TASK_MODE==2 || TASK_MODE==3 || (TASK_MODE>=64 && TASK_MODE<80)))) set_ffts_base_addr(address);
#endif
}
template<class DT_A, uint32_t TASK_MODE, uint32_t N_VALUE>
__aicore__ inline void SolveGeneratedPreloadForMode(uint64_t mode) {
    (void)mode;
#if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || defined(ASCENDC_CPU_DEBUG)
    icache_preload(1);
#else
    if constexpr(!(sizeof(DT_A)==4 && (TASK_MODE==2 || TASK_MODE==3 || (TASK_MODE>=64 && TASK_MODE<80)))) icache_preload(1);
#endif
}
#define SetSysWorkspaceForce(workspace) SolveSetSysWorkspaceForMode<TEMPLATE_PARAMS>(workspace)
#define set_ffts_base_addr(address) SolveSetFftsForMode<TEMPLATE_PARAMS>(address)
#define icache_preload(mode) SolveGeneratedPreloadForMode<TEMPLATE_PARAMS>(mode)
