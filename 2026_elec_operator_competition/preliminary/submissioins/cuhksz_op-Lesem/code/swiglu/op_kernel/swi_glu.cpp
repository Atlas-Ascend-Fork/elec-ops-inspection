/*
 * Copyright (c) 2026 Lesem.
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

#include "c_api/asc_simd.h"
#include "kernel_operator.h"
#include "lib/matmul/matmul_intf.h"
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"

#include "swiglu_transfer.h"
#include "swiglu_vector_math.h"
#include "swiglu_resident_vector.h"
#include "swiglu_small_vector.h"
#include "swiglu_wide_cube.h"
#include "swiglu_k256_cube.h"
#include "swiglu_short_cube.h"
#include "swiglu_short_vector.h"
#include "swiglu_dual_k128_cube.h"
#include "swiglu_compact_vector.h"

using namespace AscendC;
using namespace AscendC::SwiGluKernel;

// Static one-owner vector kernels share the launch guard and tail row lookup.
template <uint32_t BatchR, bool DirectProduct, uint32_t ACTIVE_BLOCKS>
__aicore__ inline void RunStaticShortVector(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
    GM_ADDR b2, GM_ADDR y, GM_ADDR tiling) {
    static_assert(ACTIVE_BLOCKS >= 1U && ACTIVE_BLOCKS <= 3U);
    static_assert(offsetof(SwiGluWorkTiling, rows) == 0);
    if (GetBlockIdx() >= ACTIVE_BLOCKS) return;
    InitSocState();
    if (ACTIVE_BLOCKS > 1U && GetBlockIdx() + 1U < ACTIVE_BLOCKS) {
        KernelShortVector<BatchR, true, DirectProduct, true> op;
        op.Init(x, w1, b1, w2, b2, y, 0);
        op.Process();
    } else {
        const uint64_t rows = *reinterpret_cast<__gm__ uint64_t*>(tiling);
        KernelShortVector<BatchR, true, DirectProduct, false> op;
        op.Init(x, w1, b1, w2, b2, y, rows);
        op.Process();
    }
}

template <uint32_t Mode>
__aicore__ inline void RunPaddedCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
    GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
    if (GetBlockIdx() >= data.blocks) return;
    SetSysWorkspaceForce(workspace);
    KernelDirectCube<false, false, true, Mode >= 11U, Mode == 12U> op; op.Init(x,w1,b1,w2,b2,y,workspace,data); op.Process();
}

__aicore__ inline void RunVector(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, data, tiling);
    if (GetBlockIdx() >= data.blocks) return;
    if (data.dim <= 64) {
        KernelResident op; op.Init(x,w1,b1,w2,b2,y,data); op.Process();
    } else {
        KernelSwiGlu op; op.Init(x,w1,b1,w2,b2,y,data); op.Process();
    }
}

__aicore__ inline void RunMatmul(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, data, tiling);
    if (GetBlockIdx() >= data.blocks) return;
    KernelCube op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &data.cube);
    op.Init(x,w1,b1,w2,b2,y,data); op.Process();
}

__aicore__ inline void RunPackedVector(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, data, tiling);
    if (GetBlockIdx() >= data.blocks) return;
    KernelResident<true> op; op.Init(x,w1,b1,w2,b2,y,data); op.Process();
}

template <uint32_t StaticD>
__aicore__ inline constexpr bool UsesCompactRowTag() {
    return StaticD == 2U || StaticD == 3U || StaticD == 4U || StaticD == 64U || StaticD == 5U;
}

template <uint32_t StaticD>
__aicore__ inline constexpr uint32_t GetRowTagRange() {
    switch (StaticD) {
        case 2U: return 0U;
        case 3U: return 1U;
        case 4U: return 2U;
        case 64U: return 3U;
        default: return 4U;
    }
}

template <uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void RunShortVector(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR tiling) {
    if constexpr(STATIC_D>=33U && STATIC_D<=47U) {
        static_assert(ACTIVE_BLOCKS==0U);
        InitSocState();
        if constexpr(STATIC_D<=40U) {
            KernelCompactBlockDotShortVector<STATIC_D> op;
            op.Init(x,w1,b1,w2,b2,y,tiling);op.Process();
        } else {
            KernelCompactOuterShortVector<STATIC_D> op;
            op.Init(x,w1,b1,w2,b2,y,tiling);op.Process();
        }
    } else if constexpr (UsesCompactRowTag<STATIC_D>()) {
        constexpr uint32_t range = GetRowTagRange<STATIC_D>();
        constexpr uint32_t rows=13U+4U*range+ACTIVE_BLOCKS;
        static_assert(rows>=13U && rows<=32U);
        // ACTIVE_BLOCKS encodes row-count bits only on these new selectors.
        if(GetBlockIdx()>=1U) return;
        InitSocState();
        KernelConstRcShortVector<rows> op;op.Init(x,w1,b1,w2,b2,y,0);op.Process();
    } else if constexpr (STATIC_D == 32U && ACTIVE_BLOCKS != 0U) {
        RunStaticShortVector<32U, true, ACTIVE_BLOCKS>(x, w1, b1, w2, b2, y, tiling);
    } else if constexpr (STATIC_D == 128U && ACTIVE_BLOCKS != 0U) {
        RunStaticShortVector<16U, false, ACTIVE_BLOCKS>(x, w1, b1, w2, b2, y, tiling);
    } else {
        static_assert(ACTIVE_BLOCKS==0U);
        GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
        if (GetBlockIdx() >= data.blocks) return;
        InitSocState();
        if constexpr (STATIC_D==128U) {
            KernelShortVector<16U, true> op;op.Init(x,w1,b1,w2,b2,y,data);op.Process();
        } else {
            static_assert(STATIC_D==0U);
            KernelShortVector<16U, false> op;op.Init(x,w1,b1,w2,b2,y,data);op.Process();
        }
    }
}

__aicore__ inline void RunStaticResident(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
    if (GetBlockIdx() >= data.blocks) return;
    InitSocState();
    KernelResident<false, true, true> op; op.Init(x,w1,b1,w2,b2,y,data); op.Process();
}

template <uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void RunStripedVector(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR tiling) {
    if constexpr(STATIC_D==256U) {
        static_assert(ACTIVE_BLOCKS>=1U && ACTIVE_BLOCKS<=3U);
        if(GetBlockIdx()>=ACTIVE_BLOCKS) return;
        InitSocState();
        KernelPairedAligned32 op;op.Init(x,w1,b1,w2,b2,y);op.Process();
    } else if constexpr (STATIC_D!=0U) {
        static_assert((STATIC_D==32U || STATIC_D==64U) && ACTIVE_BLOCKS>=1U && ACTIVE_BLOCKS<=3U);
        // ACTIVE_BLOCKS encodes small-row count in this static template.
        // Both aligned K32/K64 families have two complete-K column owners per row.
        if (GetBlockIdx() >= ACTIVE_BLOCKS*2U) return;
        InitSocState();
        const SwiGluWorkTiling data{ACTIVE_BLOCKS,ACTIVE_BLOCKS,STATIC_D,STATIC_D,32U,ACTIVE_BLOCKS,7U,1U};
        KernelStripedAligned<STATIC_D> op;
        op.Init(x,w1,b1,w2,b2,y,data);
        op.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
        if (GetBlockIdx() >= data.blocks) return;
        InitSocState();
        KernelResident<true, true, true> op;op.Init(x,w1,b1,w2,b2,y,data);op.Process();
    }
}

template <uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void RunTinyGemv(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
    static_assert(ACTIVE_BLOCKS>=1U && ACTIVE_BLOCKS<=3U && STATIC_D>=2U && STATIC_D<=4U);
    if (GetBlockIdx() >= ACTIVE_BLOCKS) return;
    InitSocState();
    if constexpr(ACTIVE_BLOCKS==1U) {
        KernelCachedDotVectorSuffix<STATIC_D> op;op.Init(x,w1,b1,w2,b2,y);op.Process();
    } else {
        KernelTinyGEMVCompact<STATIC_D> op; op.Init(x,w1,b1,w2,b2,y); op.Process();
    }
}

template <uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void RunShortCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr(STATIC_D==128U && ACTIVE_BLOCKS!=0U) {
        static_assert(ACTIVE_BLOCKS>=1U && ACTIVE_BLOCKS<=3U);
        static_assert(offsetof(SwiGluWorkTiling,rows)==0 && offsetof(SwiGluWorkTiling,dim)==16);
        if(GetBlockIdx()>=ACTIVE_BLOCKS) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        const uint32_t dim=*reinterpret_cast<__gm__ uint32_t*>(tiling+16U);
        if(ACTIVE_BLOCKS>1U && GetBlockIdx()+1U<ACTIVE_BLOCKS) {
            KernelShortCube<true, ACTIVE_BLOCKS, true> op;
            op.Init(x,w1,b1,w2,b2,y,workspace,0,dim);op.Process();
        } else {
            const uint64_t rows=*reinterpret_cast<__gm__ uint64_t*>(tiling);
            KernelShortCube<true, ACTIVE_BLOCKS, false> op;
            op.Init(x,w1,b1,w2,b2,y,workspace,rows,dim);op.Process();
        }
    } else {
        static_assert(ACTIVE_BLOCKS==0U);
        GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
        if (GetBlockIdx() >= data.blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        if constexpr (STATIC_D==128U) {
            KernelShortCube<true> op;op.Init(x,w1,b1,w2,b2,y,workspace,data);op.Process();
        } else {
            static_assert(STATIC_D==0U);
            KernelShortCube<false> op;op.Init(x,w1,b1,w2,b2,y,workspace,data);op.Process();
        }
    }
}

template <uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void RunFixedJobsCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr(STATIC_D==6U) {
        static_assert(ACTIVE_BLOCKS==2U);
        const uint32_t blocks=static_cast<uint32_t>(GetBlockNum());
        if(GetBlockIdx()>=blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelTwoJobK256Cube op;op.Init(x,w1,b1,w2,b2,y,workspace,blocks);op.Process();
    } else if constexpr(STATIC_D==64U) {
        static_assert(ACTIVE_BLOCKS==0U);
        // This full-row family has >=8 useful groups: launch equals owners.
        const uint32_t blocks=static_cast<uint32_t>(GetBlockNum());
        const uint32_t worker=GetBlockIdx()/GetSubBlockNum();
        if(worker>=blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelStaticDualFourJobK128Cube op;op.Init(x,w1,b1,w2,b2,y,workspace,blocks,worker);op.Process();
    } else if constexpr(STATIC_D==2U) {
        static_assert(ACTIVE_BLOCKS==0U && offsetof(SwiGluWorkTiling,blocks)==28);
        const uint32_t blocks=*reinterpret_cast<__gm__ uint32_t*>(tiling+28U);
        if(GetBlockIdx()>=blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelStaticTwoJobK128FullCube op;
        op.Init(x,w1,b1,w2,b2,y,workspace,blocks);op.Process();
    } else if constexpr(STATIC_D==4U) {
        static_assert(ACTIVE_BLOCKS==2U && offsetof(SwiGluWorkTiling,blocks)==28);
        const uint32_t blocks=*reinterpret_cast<__gm__ uint32_t*>(tiling+28U);
        if(GetBlockIdx()>=blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelTwoJobK256Cube op;op.Init(x,w1,b1,w2,b2,y,workspace,blocks);op.Process();
    } else {
        static_assert(STATIC_D==0U && ACTIVE_BLOCKS==0U);
        GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
        if (GetBlockIdx() >= data.blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelStaticShortCube op;op.Init(x,w1,b1,w2,b2,y,workspace,data);op.Process();
    }
}

template <uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void RunWideCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr(STATIC_D==5U || STATIC_D==512U) {
        static_assert(ACTIVE_BLOCKS==0U && offsetof(SwiGluWorkTiling,rows)==0 && offsetof(SwiGluWorkTiling,blocks)==28);
        const uint64_t rows=*reinterpret_cast<__gm__ uint64_t*>(tiling);
        const uint32_t blocks=*reinterpret_cast<__gm__ uint32_t*>(tiling+28U);
        const uint32_t worker=GetBlockIdx()/GetSubBlockNum();
        if(worker>=blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelStaticFullRowK256Cube<STATIC_D==512U> op;
        if (op.Init(x,w1,b1,w2,b2,y,workspace,rows,blocks,worker)) {
            op.Process();
        }
    } else if constexpr(STATIC_D==4U) {
        static_assert(ACTIVE_BLOCKS==0U && offsetof(SwiGluWorkTiling,rows)==0 && offsetof(SwiGluWorkTiling,blocks)==28);
        const uint64_t rows=*reinterpret_cast<__gm__ uint64_t*>(tiling);
        const uint32_t blocks=*reinterpret_cast<__gm__ uint32_t*>(tiling+28U);
        if(GetBlockIdx()>=blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelStaticK256PairCube<true> op;op.Init(x,w1,b1,w2,b2,y,workspace,rows,blocks);op.Process();
    } else if constexpr(STATIC_D==3U) {
        static_assert(ACTIVE_BLOCKS==0U && offsetof(SwiGluWorkTiling,rows)==0 && offsetof(SwiGluWorkTiling,blocks)==28);
        const uint64_t rows=*reinterpret_cast<__gm__ uint64_t*>(tiling);
        const uint32_t blocks=*reinterpret_cast<__gm__ uint32_t*>(tiling+28U);
        if(GetBlockIdx()>=blocks) return;
        SetSysWorkspaceForce(workspace);InitSocState();
        KernelStaticK256PairCube<false> op;op.Init(x,w1,b1,w2,b2,y,workspace,rows,blocks);op.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
        if (GetBlockIdx() >= data.blocks) return;
        SetSysWorkspaceForce(workspace);
        if constexpr (STATIC_D==128U) {
            KernelDirectCube<true, true, false, false, false, true> op;op.Init(x,w1,b1,w2,b2,y,workspace,data);op.Process();
        } else {
            static_assert(STATIC_D==0U);
            KernelDirectCube<true,true> op; op.Init(x,w1,b1,w2,b2,y,workspace,data); op.Process();
        }
    }
}

template <uint32_t ACTIVE_BLOCKS>
__aicore__ inline void RunD1(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y) {
    // Existing legal D1 family has1..3 useful workers; launch remains8.
    static_assert(ACTIVE_BLOCKS>=1U && ACTIVE_BLOCKS<=3U);
    if (GetBlockIdx() >= ACTIVE_BLOCKS) return;
    InitSocState();
    KernelTinyD1 op; op.Init(x,w1,b1,w2,b2,y); op.Process();
}

__aicore__ inline void RunDoubleBufferedCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
    if (GetBlockIdx() >= data.blocks) return;
    SetSysWorkspaceForce(workspace);
    KernelDirectCube<true> op; op.Init(x,w1,b1,w2,b2,y,workspace,data); op.Process();
}

__aicore__ inline void RunFullKCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluWorkTiling, data, tiling);
    if (GetBlockIdx() >= data.blocks) return;
    SetSysWorkspaceForce(workspace);
    KernelDirectCube<false> op; op.Init(x,w1,b1,w2,b2,y,workspace,data); op.Process();
}

template <uint32_t CUBE_MODE, uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void DispatchSmallMode(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (CUBE_MODE == 0U) {
        RunVector(x, w1, b1, w2, b2, y, tiling);
    } else if constexpr (CUBE_MODE == 1U) {
        RunMatmul(x, w1, b1, w2, b2, y, tiling);
    } else if constexpr (CUBE_MODE == 3U) {
        RunPackedVector(x, w1, b1, w2, b2, y, tiling);
    } else if constexpr (CUBE_MODE == 4U) {
        RunD1<ACTIVE_BLOCKS>(x, w1, b1, w2, b2, y);
    } else if constexpr (CUBE_MODE == 5U) {
        RunStaticResident(x, w1, b1, w2, b2, y, tiling);
    } else if constexpr (CUBE_MODE == 6U) {
        RunDoubleBufferedCube(x, w1, b1, w2, b2, y, workspace, tiling);
    } else if constexpr (CUBE_MODE == 7U) {
        RunStripedVector<ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y, tiling);
    } else {
        RunFullKCube(x, w1, b1, w2, b2, y, workspace, tiling);
    }
}

template <uint32_t CUBE_MODE, uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__aicore__ inline void DispatchWideMode(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (CUBE_MODE == 8U) {
        RunWideCube<ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y, workspace, tiling);
    } else if constexpr (CUBE_MODE == 9U) {
        RunTinyGemv<ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y);
    } else if constexpr (CUBE_MODE >= 10U && CUBE_MODE <= 12U) {
        RunPaddedCube<CUBE_MODE>(x, w1, b1, w2, b2, y, workspace, tiling);
    } else if constexpr (CUBE_MODE == 13U) {
        RunShortCube<ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y, workspace, tiling);
    } else if constexpr (CUBE_MODE == 14U) {
        RunShortVector<ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y, tiling);
    } else if constexpr (CUBE_MODE == 15U) {
        RunFixedJobsCube<ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y, workspace, tiling);
    }
}

template <typename DT_INPUT_X, uint32_t CUBE_MODE, uint32_t ACTIVE_BLOCKS, uint32_t STATIC_D>
__global__ __aicore__ void swi_glu(GM_ADDR x, GM_ADDR w1, GM_ADDR b1,
    GM_ADDR w2, GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    if constexpr (CUBE_MODE < 8U) {
        DispatchSmallMode<CUBE_MODE, ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y, workspace, tiling);
    } else {
        DispatchWideMode<CUBE_MODE, ACTIVE_BLOCKS, STATIC_D>(x, w1, b1, w2, b2, y, workspace, tiling);
    }
}

// CANN 9 wrapper initialization follows the selected computation family.
// Library Matmul and all other families keep the original workspace protocol.
namespace AscendC {
template<typename InputType,uint32_t Mode,uint32_t Active,uint32_t StaticD>
__aicore__ inline void InitSwiGluKernelWorkspace(__gm__ uint8_t* workspace) {
    if constexpr(Mode==13U || (Mode==15U && (StaticD==64U || StaticD==6U)) || (Mode==8U && StaticD==512U)) {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    AscendC::SetAtomicNone();
    if ASCEND_IS_AIC {
        AscendC::SetMaskNorm();
        AscendC::SetLoadDataBoundary((uint64_t)0);
        AscendC::SetLoadDataPaddingValue((uint64_t)0);
        // This algorithm uses MMAD/Fixpipe and hardware flags; no KFC queue is used.
        AscendC::NotifyEvent<PIPE_MTE3>(WORKSPACE_SYNC_ID);
    } else {
        SetVectorMask<uint64_t, MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
        AscendC::SetMaskNorm();
    }
#endif

    } else {
        AscendC::clearWorkspace(workspace);
    }
}
}
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#define clearWorkspace(workspace) InitSwiGluKernelWorkspace<TEMPLATE_PARAMS>(workspace)
#endif
