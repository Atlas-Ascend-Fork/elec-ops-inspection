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

// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/swi_glu_tiling.h"
#include "../op_kernel/tiling_key_swi_glu.h"

namespace optiling {
using Platform = platform_ascendc::PlatformAscendC;

static bool ReadInputShape(gert::TilingContext *context, uint64_t &rows, int64_t &d) {
    const auto *xs = context->GetInputShape(0);
    if (xs == nullptr) return false;
    const auto &shape = xs->GetOriginShape();
    const size_t rank = shape.GetDimNum();
    if (rank < 1) return false;
    d = shape.GetDim(rank - 1);
    if (d < 1 || d > 8192) return false;
    rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        const int64_t v = shape.GetDim(i);
        if (v <= 0 || rows > 1048576ULL / static_cast<uint64_t>(v)) return false;
        rows *= static_cast<uint64_t>(v);
    }
    return true;
}

static bool CheckWeightsAndBiases(gert::TilingContext *context, int64_t d) {
    for (size_t i = 1; i < 5; ++i) {
        const auto *ss = context->GetInputShape(i);
        if (ss == nullptr) return false;
        const auto &s = ss->GetOriginShape();
        const bool weight = i == 1 || i == 3;
        if (s.GetDimNum() != (weight ? 2U : 1U) || s.GetDim(0) != d ||
            (weight && s.GetDim(1) != d)) return false;
    }
    return true;
}

struct CubeMemory {
    uint64_t l0a = 0;
    uint64_t l0b = 0;
    uint64_t l0c = 0;
    uint64_t l1 = 0;
    uint64_t ub = 0;
};

static CubeMemory ReadCubeMemory(Platform &platform) {
    CubeMemory memory;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, memory.l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, memory.l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, memory.l0c);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, memory.l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, memory.ub);
    return memory;
}

static uint32_t ChooseInitialTile(SwiGluTilingData *t, Platform &platform, uint64_t rows, int64_t d, uint32_t aic, uint64_t work) {
    t->useCube = d > 64 || (d == 64 && rows >= 8 && work >= 1048576ULL);
    uint32_t alignedRows = rows < 64 ? static_cast<uint32_t>((rows+15)/16*16) : 64;
    uint64_t tasks128=((rows+127)/128)*((t->dim+63)/64);
    if(rows>=128 && tasks128>=2ULL*aic) alignedRows=128;
    t->tileM = t->useCube ? alignedRows : 1;
    t->tileN = t->useCube ? 64 : 32;
    uint64_t groups128=(t->dim+127U)/128U;
    uint64_t tasks96x128=((rows+95U)/96U)*groups128;
    uint64_t tileUb=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,tileUb);
    if(t->useCube && rows>=96 && 3ULL*groups128*128ULL<=4ULL*t->dim &&
       tasks96x128>=2ULL*aic && 10ULL*96U*128U+4ULL*128U+4096ULL<tileUb) {
        t->tileM=96; t->tileN=128;
    }
    return alignedRows;
}

static void TryLargerFullKTile(SwiGluTilingData *t, uint64_t rows, uint32_t aic, const CubeMemory &memory) {
    if (aic == 0U) return;
    if(rows>=128 && t->dim%128U==0U) {
        const uint64_t oldTasks=((rows+t->tileM-1U)/t->tileM)*(t->dim/t->tileN);
        const uint64_t newTasks=((rows+127U)/128U)*(t->dim/128U);
        const uint64_t oldChain=((oldTasks+aic-1U)/aic)*t->tileM;
        const uint64_t newChain=((newTasks+aic-1U)/aic)*128U;
        const uint64_t candidateArea=128ULL*128U;
        const uint64_t candidateA=2ULL*128U*t->dim;
        const uint64_t candidateB=2ULL*128U*t->dim;
        if(newTasks>=aic && 8ULL*newChain*128U<=9ULL*oldChain*t->tileN &&
           candidateA<=memory.l0a && candidateB<=memory.l0b && 4ULL*candidateArea<=memory.l0c &&
           candidateA+2ULL*candidateB<=memory.l1 && 10ULL*candidateArea+4ULL*128U+4096ULL<=memory.ub) {
            t->tileM=128; t->tileN=128;
        }
    }
}

static void SelectFullKCube(SwiGluTilingData *t, Platform &platform, uint64_t rows, uint32_t aic) {
    // Full-K sharing is gated by aligned geometry and queried on-chip resources.
    if(t->useCube && rows>=16 && t->dim%t->tileN==0U && t->dim<=4095U && t->dim/16U<=255U) {
        const auto memory = ReadCubeMemory(platform);
        TryLargerFullKTile(t, rows, aic, memory);
        uint64_t area=static_cast<uint64_t>(t->tileM)*t->tileN;
        uint64_t abytes=2ULL*t->tileM*t->dim,bbytes=2ULL*t->tileN*t->dim;
        if(abytes<=memory.l0a && bbytes<=memory.l0b && 4ULL*area<=memory.l0c &&
           abytes+2ULL*bbytes<=memory.l1 && 10ULL*area+4ULL*t->tileN+4096ULL<=memory.ub)
            t->useCube=2;
    }
}

static void TrySmallCubeM48(SwiGluTilingData *t, uint64_t rows, uint32_t aic, const CubeMemory &memory) {
    // Source-isolated small-Cube M48: fewer q1 workers, no added M padding.
    const uint64_t q1Groups=(t->dim+15U)/16U;
    const uint64_t q1OldTasks=((rows+31U)/32U)*q1Groups;
    const uint64_t q1NewTasks=((rows+47U)/48U)*q1Groups;
    if(q1OldTasks<=aic && q1NewTasks<q1OldTasks &&
       ((rows+47U)/48U)*48U<=((rows+31U)/32U)*32U &&
       2ULL*48U*t->alignedDim<=memory.l0a &&
       8ULL*48U*16U<=memory.l0c && 4ULL*16U*t->alignedDim<=memory.l0b &&
       2ULL*48U*t->alignedDim+4ULL*16U*t->alignedDim<=memory.l1 &&
       10ULL*48U*16U+4ULL*16U+4096ULL<=memory.ub) t->tileM=48U;
}

static void SelectSmallCube(SwiGluTilingData *t, Platform &platform, uint64_t rows, int64_t d, uint32_t aic, uint64_t work) {
    // Small matrices use a padded-K 16x16 Cube tile only when real work
    // can amortize the handoff and all queried resources hold the live set.
    const uint64_t smallTasks=((rows+15U)/16U)*((t->dim+15U)/16U);
    if(!t->useCube && rows>=16 && d>=8 && d<=64 && work>=2048ULL &&
       smallTasks>=(static_cast<uint64_t>(aic)+1U)/2U) {
        const auto memory = ReadCubeMemory(platform);
        const uint64_t ab=2ULL*16U*t->alignedDim;
        const uint64_t aa=2ULL*32U*t->alignedDim;
        if(aa<=memory.l0a && ab<=memory.l0b && 4ULL*32U*16U<=memory.l0c &&
           aa+2ULL*ab<=memory.l1 && 10ULL*32U*16U+4ULL*16U+4096ULL<=memory.ub) {
            t->useCube=2; t->tileM=32; t->tileN=16;
            TrySmallCubeM48(t, rows, aic, memory);
        }
    }
}

static bool ReadMatmulTiling(matmul_tiling::MatmulApiTiling &mm, SwiGluTilingData *t,
                             uint64_t ubSize) {
    const uint64_t fusedUb = 10ULL * t->tileM * t->tileN + 4ULL * t->tileN;
    mm.SetBufferSpace(-1, -1, ubSize - fusedUb - 4096);
    return mm.GetTiling(t->cube) >= 0 && t->cube.baseM == t->tileM && t->cube.baseN == t->tileN;
}

static bool RetryMatmulTile(matmul_tiling::MatmulApiTiling &mm, SwiGluTilingData *t,
                            uint32_t tileM, int64_t d, uint64_t ubSize) {
    t->tileM = tileM;
    t->tileN = 64U;
    mm.SetShape(t->tileM, t->tileN, d);
    mm.SetFixSplit(t->tileM, t->tileN, 64);
    return ReadMatmulTiling(mm, t, ubSize);
}

static void SelectLibraryTiling(SwiGluTilingData *t, Platform &platform, uint64_t rows, int64_t d, uint32_t alignedRows) {
    if (t->useCube != 1U) return;
    matmul_tiling::MatmulApiTiling mm(platform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    mm.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetShape(t->tileM, t->tileN, d);
    mm.SetOrgShape(rows, d, d);
    mm.EnableBias(false);
    mm.SetFixSplit(t->tileM, t->tileN, 64);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t fusedUb = 10ULL * t->tileM * t->tileN + 4ULL * t->tileN;
    if (ubSize <= fusedUb + 4096) {
        t->useCube = 0;
        return;
    }
    bool valid = ReadMatmulTiling(mm, t, ubSize);
    if (!valid && t->tileN != 64U) {
        valid = RetryMatmulTile(mm, t, alignedRows, d, ubSize);
    }
    if (!valid && t->tileM == 128U) {
        valid = RetryMatmulTile(mm, t, 64U, d, ubSize);
    }
    if (!valid || (rows > 1U && t->dim % t->tileN != 0U)) {
        t->useCube = 0;
    }
}

static void SelectVectorTiles(SwiGluTilingData *t, Platform &platform, uint64_t rows, int64_t d, uint64_t work, uint32_t swigluAivBudget) {
    // A library tiling rejection remains a valid-shape Vector execution.
    if (!t->useCube) { t->tileM=1; t->tileN=32; }
    t->tasks = ((rows+t->tileM-1)/t->tileM)*((t->dim+t->tileN-1)/t->tileN);
    // Short K16 Vector q1: bulk product planes and grouped broadcasts.
    if(!t->useCube && t->alignedDim==16U && t->dim>=8U && rows>=8U && work>=2048ULL) {
        uint64_t ub=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
        const uint64_t tasks=(rows+15U)/16U;
        const uint64_t live=3136ULL+4192ULL*16U;
        if(tasks<=swigluAivBudget && live+4096ULL<=ub) {
            t->useCube=14U;t->tileM=16U;t->tileN=16U;t->tasks=tasks;
        }
    }
    if (!t->useCube && d<=64) {
        t->tileM = rows>=4 ? 4U : 1U;
        uint64_t residentUb=0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,residentUb);
        uint64_t r8bytes=16ULL*t->dim*t->alignedDim+(22ULL*8U+8U)*t->alignedDim;
        if(rows>=8 && t->dim>=16 && r8bytes+4096ULL<=residentUb) t->tileM=8;
        t->tasks=(rows+t->tileM-1)/t->tileM;
    }
}

static uint32_t DistributeTiles(SwiGluTilingData *t, Platform &platform, uint32_t swigluAicBudget, uint32_t swigluAivBudget) {
    uint32_t cores=static_cast<uint32_t>(t->useCube && t->useCube!=14U ? swigluAicBudget : swigluAivBudget);
    if (cores==0) cores=1;
    t->blocks=t->tasks<cores ? static_cast<uint32_t>(t->tasks) : cores;
    if(t->useCube==2 && t->tasks>t->blocks) {
        const uint64_t rounds=(t->tasks+t->blocks-1U)/t->blocks;
        const uint64_t needed=(t->tasks+rounds-1U)/rounds;
        const uint64_t ng=(t->dim+t->tileN-1U)/t->tileN;
        const uint64_t alignedBlocks=(needed+ng-1U)/ng*ng;
        if(alignedBlocks<=t->blocks) t->blocks=static_cast<uint32_t>(alignedBlocks);
    }
    if(t->useCube==2 && t->tasks/t->blocks>=2U &&
       t->blocks%((t->dim+t->tileN-1U)/t->tileN)==0U) {
        uint64_t l1=0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
        if(4ULL*t->tileM*t->alignedDim+4ULL*t->tileN*t->alignedDim<=l1) t->useCube=6;
    }
    return cores;
}

static void SelectCompactGemv(SwiGluTilingData *t, Platform &platform, uint64_t rows) {
    if(t->useCube==5U && t->dim>=2U && t->dim<=4U && rows<4U && t->tasks==t->blocks) {
        uint64_t compactUb=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,compactUb);
        if(1696ULL+4096ULL<=compactUb) t->useCube=9U;
    }
}

static void SelectTinyVectorPath(SwiGluTilingData *t, Platform &platform, uint64_t rows) {
    if(!t->useCube && t->tileM==1U && t->dim>=8U && t->dim<=64U && rows<4U) t->useCube=3;
    if(!t->useCube && t->dim==1U && rows<4U && t->tasks==t->blocks) {
        uint64_t tinyUb=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,tinyUb);
        if(544ULL+4096ULL<=tinyUb) t->useCube=4;
    }
    if ((t->useCube==0U || t->useCube==3U) && t->dim<=64U && t->tileM<=2U*t->dim)
        t->useCube=t->useCube==3U ? 7U : 5U;
    SelectCompactGemv(t, platform, rows);
}

static void SelectCubeChain(SwiGluTilingData *t, Platform &platform) {
    if(t->useCube==6U && t->tasks/t->blocks>=3U) {
        uint64_t shiftedUb=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,shiftedUb);
        if(10ULL*t->tileM*t->tileN+4ULL*t->tileN+256ULL+4096ULL<=shiftedUb) t->useCube=8U;
    }
    if(t->useCube==2U && t->tasks==t->blocks) t->useCube=10U;
    if(t->useCube==10U) {
        uint64_t q1L0B=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,q1L0B);
        if(4ULL*t->tileN*t->alignedDim<=q1L0B) t->useCube=11U;
    }
    if(t->useCube==11U) {
        uint64_t q1L0C=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,q1L0C);
        if(8ULL*t->tileM*t->tileN<=q1L0C) t->useCube=12U;
    }
    if(t->useCube==12U) t->useCube=13U;
}

static void SelectShortCubeChain(SwiGluTilingData *t, Platform &platform, uint64_t rows) {
    // Short two/three-job complete-16 chains with fixed N per worker.
    // Dual B0 remains immutable; A1 and GM slots have explicit reuse credits.
    if(t->useCube==6U && t->tasks>=2ULL*t->blocks && t->tasks<=3ULL*t->blocks &&
       rows%16U==0U && t->dim%t->tileN==0U && t->blocks%(t->dim/t->tileN)==0U) {
        uint64_t rb=0,rc=0,rl1=0,rub=0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,rb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,rc);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,rl1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,rub);
        if(4ULL*t->tileN*t->alignedDim<=rb && 8ULL*t->tileM*t->tileN<=rc &&
           4ULL*t->tileM*t->alignedDim+4ULL*t->tileN*t->alignedDim<=rl1 &&
           10ULL*t->tileM*t->tileN+4ULL*t->tileN+4096ULL<=rub) t->useCube=15U;
    }
}

static bool SelectPackedFullShort(SwiGluTilingData *t, Platform &platform, uint64_t rows, uint32_t swigluAicBudget) {
    // Full Kpad48/Npad48 removes the three disjoint N-output stripes.
    bool packedFullShort=false;
    if(t->useCube==13U && t->tileM==48U && t->tileN==16U &&
       t->dim>=33U && t->dim<48U && t->alignedDim==48U && t->tasks==t->blocks) {
        const auto memory = ReadCubeMemory(platform);
        const uint64_t ft=(rows+47U)/48U;
        if(ft && ft<=t->blocks && ft<=swigluAicBudget &&
           2ULL*48U*48U<=memory.l0a && 4ULL*48U*48U<=memory.l0b && 8ULL*48U*48U<=memory.l0c &&
           6ULL*48U*48U<=memory.l1 && 10ULL*48U*48U+4ULL*48U+4096ULL<=memory.ub) {
            t->tileN=48U;t->tasks=ft;t->blocks=static_cast<uint32_t>(ft);
            packedFullShort=true;
        }
    }
    return packedFullShort;
}

static bool CanRunStaticTwoJob(const SwiGluTilingData *t, uint64_t rows, uint64_t twoJobUb) {
    return t->useCube==15U && t->tileM==64U && t->tileN==64U &&
        t->dim==256U && t->alignedDim==256U && rows%64U==0U &&
        t->blocks%4U==0U && t->tasks==2ULL*t->blocks &&
        12ULL*64U*64U+4ULL*64U+4096ULL<=twoJobUb;
}

static bool CanRunStaticK128TwoJob(const SwiGluTilingData *t, uint64_t rows) {
    return t->useCube==15U && t->tileM==128U &&
        t->tileN==128U && t->dim==128U && t->alignedDim==128U &&
        rows==256ULL*t->blocks && t->tasks==2ULL*t->blocks;
}

static bool SelectDualFourJob(SwiGluTilingData *t, uint64_t twoJobUb, uint32_t swigluAicBudget, uint32_t swigluAivBudget, bool staticK128TwoJob) {
    bool staticDualFourJob=false;
    if(staticK128TwoJob && t->blocks>=16U && t->blocks%2U==0U &&
       12ULL*(128U/2U)*128U+4ULL*128U+256ULL+4096ULL<=twoJobUb) {
        const uint32_t producers=t->blocks/2U;
        if(producers>=8U && producers<=swigluAicBudget && 2U*producers<=swigluAivBudget) {
            t->blocks=producers;staticDualFourJob=true;
        }
    }
    return staticDualFourJob;
}

static bool CanPrefetchStages(const SwiGluTilingData *t, uint64_t rows) {
    return t->useCube==8U && t->tileM==128U && t->tileN==128U &&
        t->dim==256U && t->alignedDim==t->dim && rows%128U==0U && t->blocks%2U==0U;
}

static bool CanRunStagePair(const SwiGluTilingData *t, uint64_t twoJobUb, bool stagePrefetch) {
    return stagePrefetch && t->blocks!=0U &&
        t->tasks>=6ULL*t->blocks && t->tasks<=7ULL*t->blocks &&
        10ULL*128U*128U+4ULL*128U+256ULL+4096ULL<=twoJobUb;
}

static bool SelectFullRowCube(SwiGluTilingData *t, Platform &platform, uint64_t rows, uint64_t twoJobUb, uint32_t swigluAivBudget, bool staticStagePair) {
    // Whole-output owner, three full M128 jobs plus a true M16-aligned tail.
    bool staticFullRow=false;
    if(staticStagePair && t->blocks>=8U && 2U*t->blocks<=swigluAivBudget) {
        uint64_t a=0,b=0,c=0,l=0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,a);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,b);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,c);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,l);
        if(65536ULL<=a && 65536ULL<=b && 131072ULL<=c && 393216ULL<=l &&
           165120ULL+4096ULL<=twoJobUb) {
            staticFullRow=true;t->tileN=256U;t->tasks=rows/128ULL;
        }
    }
    // Reduce replicated full-N owners while retaining at most four M128 epochs.
    if(staticFullRow) {
        const uint32_t producers=static_cast<uint32_t>((t->tasks+3ULL)/4ULL);
        if(producers>=8U && producers<t->blocks && 2U*producers<=swigluAivBudget) {
            const uint64_t units=rows/16ULL;
            const uint64_t lower=units/producers,upper=(units+producers-1ULL)/producers;
            if(lower>=24ULL && upper<=32ULL) t->blocks=producers;
        }
    }
    return staticFullRow;
}

static bool SelectDirectR32(SwiGluTilingData *t, uint64_t rows, uint64_t twoJobUb) {
    // Combine one-owner R32 with direct products, retaining the R16 fallback.
    const uint64_t directR32Tasks=(rows+31ULL)/32ULL;
    const bool directR32=t->useCube==14U && t->dim==13U && t->alignedDim==16U &&
        t->tileM==16U && t->tasks==t->blocks && directR32Tasks>=1ULL &&
        directR32Tasks<=3ULL && 137280ULL+4096ULL<=twoJobUb;
    if(directR32) {
        t->tileM=32U;t->tasks=directR32Tasks;t->blocks=static_cast<uint32_t>(directR32Tasks);
    }
    return directR32;
}

static bool SelectOuterShort(SwiGluTilingData *t, Platform &platform, uint64_t rows, uint32_t swigluAivBudget, bool packedFullShort) {
    // Compact weights, separated k-major panels, independent eight-row owners.
    uint64_t outerUb=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,outerUb);
    const uint64_t outerTasks=(rows+7ULL)/8ULL;
    const bool outerShort=packedFullShort && t->dim<=40U && rows<=96ULL && outerTasks>=8ULL &&
        outerTasks<=swigluAivBudget && 188416ULL+4096ULL<=outerUb;
    if(outerShort) {
        t->useCube=14U;t->tileM=8U;t->tileN=64U;
        t->tasks=outerTasks;t->blocks=static_cast<uint32_t>(outerTasks);
    }
    return outerShort;
}

static bool IsStaticResident(const SwiGluTilingData *t, Platform &platform, uint64_t rows, uint32_t cores) {
    bool staticResident=t->useCube==7U && t->dim==t->alignedDim &&
        (t->dim==32U || t->dim==64U) && rows<4U && t->tileM==1U &&
        t->blocks==rows && t->tasks==rows;
    if(staticResident && t->dim==32U) {
        uint64_t ub=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
        if(18880ULL+4096ULL>ub) staticResident=false;
    }
    // The static K64 family owns G aligned column stripes per logical row.
    // All other modes retain their original useful-worker/task interpretation.
    if(staticResident && 2ULL*t->blocks>cores) staticResident=false;
    return staticResident;
}

struct StaticPaths {
    bool packedFullShort;
    bool packedRows;
    bool staticResident;
    bool staticTwoJob;
    bool staticK128TwoJob;
    bool staticDualFourJob;
    bool stagePrefetch;
    bool staticStagePair;
    bool staticFullRow;
    bool directR32;
    bool outerShort;
};

struct KernelSelector {
    uint32_t active;
    uint32_t dimTag;
};

static bool UsesWorkerTag(uint32_t mode, bool staticResident, bool smallPacked) {
    return mode == 4U || mode == 9U || staticResident || smallPacked;
}

static KernelSelector SelectCubeKey(const SwiGluTilingData *t, const StaticPaths &paths,
    KernelSelector selector) {
    if (paths.staticTwoJob) selector = {2U, t->blocks >= 8U ? 6U : 4U};
    if (paths.staticK128TwoJob) selector.dimTag = 2U;
    if (paths.staticDualFourJob) selector = {0U, 64U};
    if (paths.staticStagePair) selector = {0U, 4U};
    if (paths.staticFullRow) selector = {0U, 5U};
    return selector;
}

static KernelSelector SelectBaseKey(const SwiGluTilingData *t, const StaticPaths &paths) {
    const bool smallPacked = (paths.packedRows || paths.packedFullShort) &&
        t->tasks == t->blocks && t->blocks <= 3U;
    const uint32_t active = UsesWorkerTag(t->useCube, paths.staticResident, smallPacked) ? t->blocks : 0U;
    const bool packed = paths.packedFullShort || paths.stagePrefetch || paths.packedRows;
    const bool resident = t->useCube == 9U || paths.staticResident;
    KernelSelector selector{active, packed ? 128U : (resident ? t->dim : 0U)};
    selector = SelectCubeKey(t, paths, selector);
    if (paths.directR32) selector = {t->blocks, 32U};
    return selector;
}

static void SpecializeVectorKey(const SwiGluTilingData *t, uint64_t rows, uint64_t twoJobUb, const StaticPaths &paths,
                                KernelSelector &selector) {
    // Specialize the complete one-owner R32 family without reading tensor data.
    if(paths.directR32 && rows>=13ULL && rows<=32ULL && t->blocks==1U &&
       153888ULL<=twoJobUb) {
        constexpr uint32_t rowTags[5]={2U,3U,4U,64U,5U};
        const uint32_t code=static_cast<uint32_t>(rows-13ULL);
        selector.active=code%4U;selector.dimTag=rowTags[code/4U];
    }
    if(paths.outerShort) { selector.active=0U;selector.dimTag=t->dim; }
    if(paths.staticResident && t->dim==32U && 139328ULL<=twoJobUb) { selector.dimTag=256U;selector.active=static_cast<uint32_t>(rows); }
}

static bool UsesCubeWorkspace(uint32_t mode) {
    switch (mode) {
        case 2U:
        case 6U:
        case 8U:
        case 10U:
        case 11U:
        case 12U:
        case 13U:
        case 15U:
            return true;
        default:
            return false;
    }
}

static void SetWorkspace(gert::TilingContext *context, Platform &platform, const SwiGluTilingData *t,
                         bool staticDualFourJob, bool sharedSecondWeight) {
    const bool usesCubeWorkspace = UsesCubeWorkspace(t->useCube);
    auto &bytes = context->GetWorkspaceSizes(1)[0];
    bytes = (t->useCube == 1U || usesCubeWorkspace) ? platform.GetLibApiWorkSpaceSize() : 0;
    if (usesCubeWorkspace) bytes += 8ULL*t->blocks*t->tileM*t->tileN;
    if (staticDualFourJob) bytes += 512ULL*t->blocks;
    if (sharedSecondWeight) bytes += 2ULL*256ULL*256ULL;
}

static ge::graphStatus SetKernelKey(gert::TilingContext *context, Platform &platform,
                                    SwiGluTilingData *t, uint32_t cores, uint64_t twoJobUb,
                                    const StaticPaths &paths) {
    const uint32_t actualWorkers = paths.staticResident ? 2U*t->blocks : t->blocks;
    const uint32_t launchBlocks = actualWorkers < 8U ? (cores < 8U ? cores : 8U) : actualWorkers;
    context->SetBlockDim(launchBlocks);
    auto selector = SelectBaseKey(t, paths);
    SpecializeVectorKey(t, t->rows, twoJobUb, paths, selector);
    const bool sharedSecondWeight = paths.staticFullRow && t->blocks >= 8U && t->blocks <= 16U &&
        256U % t->blocks == 0U && (256U / t->blocks) % 16U == 0U;
    if (sharedSecondWeight) {
        selector.dimTag = 512U;
        if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    }
    uint32_t DT_INPUT_X = static_cast<uint32_t>(ge::DT_FLOAT16);
    uint32_t CUBE_MODE = t->useCube;
    uint32_t ACTIVE_BLOCKS = selector.active;
    uint32_t STATIC_D = selector.dimTag;
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, CUBE_MODE, ACTIVE_BLOCKS, STATIC_D);
    SetWorkspace(context, platform, t, paths.staticDualFourJob, sharedSecondWeight);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    uint64_t rows;
    int64_t d;
    if (!ReadInputShape(context, rows, d) || !CheckWeightsAndBiases(context, d)) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // Local B4 core-count control. Other B2 hardware properties are unchanged.
    const uint32_t swigluAvailableAic = platform.GetCoreNumAic();
    const uint32_t swigluAvailableAiv = platform.GetCoreNumAiv();
    const uint32_t swigluAicBudget = swigluAvailableAic > 20U ? 20U : swigluAvailableAic;
    const uint32_t swigluAivBudget = swigluAvailableAiv > 40U ? 40U : swigluAvailableAiv;
    auto *t = context->GetTilingData<SwiGluTilingData>();
    t->cube={};
    t->rows = rows; t->dim = static_cast<uint32_t>(d);
    t->alignedDim = (t->dim + 15U) / 16U * 16U;
    // Existing bulk domain plus aligned GEMV-like shapes with enough N/K work.
    // d<=64 keeps the resident Vector implementation for low-work shapes.
    const uint64_t work = rows * static_cast<uint64_t>(d) * d;
    const uint32_t aic = swigluAicBudget == 0U ? 1U : swigluAicBudget;
    const uint32_t alignedRows = ChooseInitialTile(t, platform, rows, d, aic, work);
    SelectFullKCube(t, platform, rows, aic);
    SelectSmallCube(t, platform, rows, d, aic, work);
    SelectLibraryTiling(t, platform, rows, d, alignedRows);
    SelectVectorTiles(t, platform, rows, d, work, swigluAivBudget);
    const uint32_t cores = DistributeTiles(t, platform, swigluAicBudget, swigluAivBudget);
    SelectTinyVectorPath(t, platform, rows);
    SelectCubeChain(t, platform);
    SelectShortCubeChain(t, platform, rows);

    StaticPaths paths{};
    paths.packedFullShort = SelectPackedFullShort(t, platform, rows, swigluAicBudget);
    uint64_t twoJobUb = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, twoJobUb);
    paths.staticTwoJob = CanRunStaticTwoJob(t, rows, twoJobUb);
    paths.staticK128TwoJob = CanRunStaticK128TwoJob(t, rows);
    paths.staticDualFourJob = SelectDualFourJob(t, twoJobUb, swigluAicBudget, swigluAivBudget, paths.staticK128TwoJob);
    paths.stagePrefetch = CanPrefetchStages(t, rows);
    paths.staticStagePair = CanRunStagePair(t, twoJobUb, paths.stagePrefetch);
    paths.staticFullRow = SelectFullRowCube(t, platform, rows, twoJobUb, swigluAivBudget, paths.staticStagePair);
    paths.directR32 = SelectDirectR32(t, rows, twoJobUb);
    paths.packedRows = t->useCube == 14U && t->dim == 13U && t->alignedDim == 16U && t->tileM == 16U;
    paths.outerShort = SelectOuterShort(t, platform, rows, swigluAivBudget, paths.packedFullShort);
    paths.staticResident = IsStaticResident(t, platform, rows, cores);
    return SetKernelKey(context, platform, t, cores, twoJobUb, paths);
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *x = context->GetInputShape(0);
    auto *y = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr) return GRAPH_FAILED;
    *y = *x;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}
}

namespace ops {
    class SwiGlu : public OpDef {
    public:
        explicit SwiGlu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("weight_1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("bias_1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("weight_2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("bias_2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SwiGlu);
}  // namespace ops
