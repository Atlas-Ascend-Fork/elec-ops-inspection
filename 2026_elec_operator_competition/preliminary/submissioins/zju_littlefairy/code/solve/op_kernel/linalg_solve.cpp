/*
 * Copyright (c) 2026 LittleFairy.
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

#include <type_traits>
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "linalg_solve_tiling.h"
#ifndef SOLVE_DIRECT_BENCH
#include "tiling_key_linalg_solve.h"
#endif

#include "linalg_solve_vector.h"
#include "linalg_solve_panel.h"
#include "linalg_solve_matmul.h"
#include "linalg_solve_orthogonal.h"

using namespace AscendC;
using namespace AscendC::SolveDetail;

#ifndef SOLVE_DIRECT_BENCH
struct CubeEntryWork {
    LinalgSolveCubeTilingData cube;
    GM_ADDR user;
    GM_ADDR system;
    AscendC::tiling::TCubeTiling tc{},ht{};
};
template<uint32_t KERNEL_PATH>
__aicore__ inline void PrepareCubeEntry(CubeEntryWork &entry,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
    entry.cube=cube;entry.user=GetUserWorkspace(workspace);entry.system=entry.user-cube.systemWorkspaceBytes;
    if ASCEND_IS_AIC {
        auto words=(uint32_t*)&entry.tc;
        for(uint32_t i=0;i<sizeof(entry.tc)/4;++i)words[i]=cube.cubeWords[i];
    }
    if constexpr(KERNEL_PATH==15 || (KERNEL_PATH>=18 && KERNEL_PATH<=22)) {
        if ASCEND_IS_AIC {
            auto words=(uint32_t*)&entry.ht;
            for(uint32_t i=0;i<sizeof(entry.ht)/4;++i)words[i]=cube.halfWords[i];
        }
    }
}
template<uint32_t KERNEL_PATH>
__aicore__ inline void InvokeNormalEntry(CubeEntryWork &entry,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,const LinalgSolveTilingData &t) {
    constexpr uint32_t N=KERNEL_PATH==21?384:(KERNEL_PATH==18?512:256);
    constexpr bool Grouped=KERNEL_PATH==22 || N>256;
    if ASCEND_IS_AIC {NormalInverse<N,32,2,Grouped>::Service(entry.user,entry.system,entry.tc,t.batches,t.n,t.k,&entry.ht,aa);}
    if ASCEND_IS_AIV {NormalInverse<N,32,2,Grouped> op;op.Run(aa,bb,xx,entry.user,t);}
}
template<uint32_t KERNEL_PATH>
__aicore__ inline void InvokeDirectEntry(CubeEntryWork &entry,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,const LinalgSolveTilingData &t) {
    if constexpr(KERNEL_PATH==19) {
        if ASCEND_IS_AIC {DirectPanel<208,256,true>::Service(entry.user,entry.system,entry.tc,t.batches,t.n,t.k,&entry.ht);}
        if ASCEND_IS_AIV {DirectPanel<208,256,true> op;op.Run(aa,bb,xx,entry.user,t);}
    } else {
        constexpr uint32_t N=KERNEL_PATH==23?128:256;
        if ASCEND_IS_AIC {DirectPanel<N>::Service(entry.user,entry.system,entry.tc,t.batches,t.n,t.k);}
        if ASCEND_IS_AIV {DirectPanel<N> op;op.Run(aa,bb,xx,entry.user,t);}
    }
}
template<uint32_t KERNEL_PATH>
__aicore__ inline void InvokeCubeEntry(CubeEntryWork &entry,GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,const LinalgSolveTilingData &t) {
    if constexpr(KERNEL_PATH==18 || KERNEL_PATH==20 || KERNEL_PATH==21 || KERNEL_PATH==22)
        InvokeNormalEntry<KERNEL_PATH>(entry,aa,bb,xx,t);
    else if constexpr(KERNEL_PATH==19 || KERNEL_PATH==23 || KERNEL_PATH==10)
        InvokeDirectEntry<KERNEL_PATH>(entry,aa,bb,xx,t);
    else {
        constexpr uint32_t N=(KERNEL_PATH==9 || KERNEL_PATH==13)?128:256;
        if ASCEND_IS_AIC {CubePanel<N,(KERNEL_PATH>=11),(KERNEL_PATH==15)>::Service(entry.user,entry.system,entry.tc,entry.ht,bb,t.batches,t.n,t.k,t.broadcastB);}
        if ASCEND_IS_AIV {CubePanel<N,(KERNEL_PATH>=11),(KERNEL_PATH==15)> op;op.Run(aa,bb,xx,entry.user,t);}
    }
}
template<uint32_t KERNEL_PATH>
__aicore__ inline void RunCubeEntry(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,GM_ADDR tiling) {
    CubeEntryWork entry;PrepareCubeEntry<KERNEL_PATH>(entry,workspace,tiling);
    if constexpr(KERNEL_PATH==20) {
        InvokeNormalEntry<KERNEL_PATH>(entry,A,B,X,entry.cube.base);
    } else {
        const uint32_t resident=GetBlockNum()/2;
        for(uint32_t first=0;first<entry.cube.base.batches;first+=resident) {
            auto t=entry.cube.base;t.batches=entry.cube.base.batches-first<resident?entry.cube.base.batches-first:resident;
            GM_ADDR aa=A+(uint64_t)first*t.n*t.n*4;
            GM_ADDR bb=B+(uint64_t)(t.broadcastB?0:first)*t.n*t.k*4;
            GM_ADDR xx=X+(uint64_t)first*t.n*t.k*4;
            InvokeCubeEntry<KERNEL_PATH>(entry,aa,bb,xx,t);
            if(first+resident<entry.cube.base.batches)SyncAll<false>();
        }
    }
}
template<uint32_t KERNEL_PATH>
__aicore__ inline void RunVectorEntry(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,t,tiling);
    if constexpr(KERNEL_PATH==5) {RefinedSquare<208,16,1,true> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==6) {RefinedSquare<256,16,1,true> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==7) {RefinedSquare<160,16,1,true> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==2) {RefinedSquare<128,32> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==3) {RefinedSquare<256,32> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==1) {
        if(t.n<=16) {SmallSquare<16> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
        else if(t.n<=32) {SmallSquare<32> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
        else if(t.n<=56) {SmallSquare<56> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
        else {SmallSquare<64> op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    }
    else {SolveKernel op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
}
template<typename DT_A,uint32_t KERNEL_PATH>
__global__ __aicore__ void linalg_solve(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR workspace,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    InitSocState();REGISTER_TILING_DEFAULT(LinalgSolveCubeTilingData);
    if constexpr(KERNEL_PATH==27 || KERNEL_PATH==28) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,t,tiling);
        if constexpr(KERNEL_PATH==27) {TinyColumns op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
        else {SmallColumns op;op.Run(A,B,X,GetUserWorkspace(workspace),t);}
    } else if constexpr(KERNEL_PATH==25 || KERNEL_PATH==26) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        if ASCEND_IS_AIC {NarrowOrthogonal::Run<true,KERNEL_PATH==26>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB,cube.base.n);}
        if ASCEND_IS_AIV {NarrowOrthogonal::Run<false,KERNEL_PATH==26>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB,cube.base.n);}
    } else if constexpr(KERNEL_PATH==24) {
        GET_TILING_DATA_WITH_STRUCT(LinalgSolveCubeTilingData,cube,tiling);
        if ASCEND_IS_AIC {Orthogonal::Run<true>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB);}
        if ASCEND_IS_AIV {Orthogonal::Run<false>(A,B,X,GetUserWorkspace(workspace),cube.base.batches,cube.base.k,cube.base.broadcastB);}
    } else if constexpr(KERNEL_PATH>=9)RunCubeEntry<KERNEL_PATH>(A,B,X,workspace,tiling);
    else RunVectorEntry<KERNEL_PATH>(A,B,X,workspace,tiling);
}
#endif
