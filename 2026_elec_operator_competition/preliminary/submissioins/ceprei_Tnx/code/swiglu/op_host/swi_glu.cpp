/*
 * Copyright (c) 2026 ceprei_Tnx.
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
// v105: key 2 always uses AIC-local MatmulImpl + AIV epilogue.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/swi_glu_tiling.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace optiling {
namespace {
constexpr int32_t PREFERRED_BASE_N = 512;
constexpr int32_t ALTERNATE_BASE_N = 256;
constexpr int32_t FALLBACK_BASE_N = 64;
// Submission v171: lean engine hoist + PIPE_ALL sync (no MTE3_V token).
// Scheduling-only A/B. Set to 1 to restore v46 launch dimensions.
// Logical ownership remains based on usedCoreNum, not this launch floor.
constexpr uint32_t MIN_VECTOR_LAUNCH_CORES = 8U;
constexpr uint64_t UB_RESERVED_BYTES = 32U * 1024U;
constexpr uint32_t VECTOR_ALIGN_ELEMENTS = 16;
constexpr int64_t MAX_FEATURE_DIM = 8192;
constexpr int64_t MAX_PREFIX_ELEMENTS = 1048576;
// v-tight: two FP32 arrays plus two 2-deep half queues; the FP16 output
// stages inside the second array. Bias resident is accounted separately.
constexpr uint32_t VEC_BYTES_PER_ELEMENT = 2U * sizeof(float) + 2U * 2U * sizeof(uint16_t);
constexpr uint64_t PLANE_STRIDE_ALIGN_ELEMENTS = 1024U / sizeof(uint16_t);
constexpr uint64_t PLANE_GUARD_ELEMENTS = 1024U / sizeof(uint16_t);

// 只用于本地 A/B 实验的环境变量开关。判分环境不会设置它们。
static int EnvI(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return v != nullptr ? std::atoi(v) : fallback;
}

static thread_local uint32_t cubeShapeCoreCap = 0U;

class CubeCoreCapScope {
  public:
    explicit CubeCoreCapScope(uint32_t cap) : previous_(cubeShapeCoreCap) {
        cubeShapeCoreCap = cap;
    }
    ~CubeCoreCapScope() {
        cubeShapeCoreCap = previous_;
    }

  private:
    uint32_t previous_;
};

static uint32_t PlatAic(platform_ascendc::PlatformAscendC &platform) {
    const int ov = EnvI("SWGLU_AIC", 0);
    if (ov > 0) {
        return static_cast<uint32_t>(ov);
    }
    const uint32_t physical = static_cast<uint32_t>(platform.GetCoreNumAic());
    return cubeShapeCoreCap != 0U && cubeShapeCoreCap < physical ? cubeShapeCoreCap : physical;
}

static uint32_t PlatAiv(platform_ascendc::PlatformAscendC &platform) {
    const int ov = EnvI("SWGLU_AIV", 0);
    return ov > 0 ? static_cast<uint32_t>(ov) : static_cast<uint32_t>(platform.GetCoreNumAiv());
}

// sN 强制对齐到 baseN 的整数倍。SDK 为凑满核数常给出 sN=171/86 这类非 16
// 对齐值（sN%16=11），会让 L0B 装载与 FixPipe 走非对齐路径。历史上本机
// 实测 1024x1024 收益约 -11.3%（sM 256->384、sN 171->128、bK 64->128）。
constexpr bool kEnableAlignedTiling = true;
// ============================================================================
static bool GenerateCubeTiling(platform_ascendc::PlatformAscendC &platform, uint32_t m, uint32_t n, uint32_t k,
                               int32_t fixedBaseM, int32_t fixedBaseN, AscendC::tiling::TCubeTiling &result,
                               uint32_t logicalWorkers = 0U, int32_t singleM = -1, int32_t singleN = -1) {
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(platform);
    const uint32_t aicNum = PlatAic(platform);
    if (aicNum == 0U) {
        return false;
    }
    const uint32_t dim = logicalWorkers == 0U ? aicNum : logicalWorkers;
    if (cubeTiling.SetDim(static_cast<int32_t>(dim)) == -1) {
        return false;
    }
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16, false);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    // 与 kernel SetHF32(false, 0) 配套：保持 fp32 累加（NORMAL）
    cubeTiling.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetShape(m, n, k);
    cubeTiling.SetOrgShape(m, n, k);
    if ((singleM > 0 || singleN > 0) && cubeTiling.SetSingleShape(singleM, singleN, -1) == -1) {
        return false;
    }
    if (fixedBaseM > 0 || fixedBaseN > 0) {
        cubeTiling.SetFixSplit(fixedBaseM, fixedBaseN, -1);
    }
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    return cubeTiling.GetTiling(result) != -1;
}

// Refine only the application task grid, never SDK baseK/L1/L0 parameters.
// Two rounds per AIC permit CV overlap without reducing launch parallelism.
// Prefer M splitting to preserve the AIV bias-column cache. Bound refinement
// and task area to amortize Matmul calls and cross-core notification overhead.
static void ChooseCvTaskShape(const AscendC::tiling::TCubeTiling &cube, uint32_t m, uint32_t n, uint32_t aic,
                              uint32_t &taskM, uint32_t &taskN) {
    taskM = static_cast<uint32_t>(cube.singleCoreM);
    taskN = static_cast<uint32_t>(cube.singleCoreN);
    if (cube.baseM <= 0 || cube.baseN <= 0 || aic == 0U) {
        return;
    }
    constexpr uint64_t minTaskElements = 16384ULL;
    for (uint32_t refinement = 0U; refinement < 2U; ++refinement) {
        const uint64_t tiles =
            (static_cast<uint64_t>(m) + taskM - 1U) / taskM * ((static_cast<uint64_t>(n) + taskN - 1U) / taskN);
        if (tiles >= 2ULL * aic) {
            break;
        }
        const uint32_t baseM = static_cast<uint32_t>(cube.baseM);
        const uint32_t baseN = static_cast<uint32_t>(cube.baseN);
        const uint32_t halfM = taskM / 2U / baseM * baseM;
        const uint32_t halfN = taskN / 2U / baseN * baseN;
        const uint32_t validM = taskM < m ? taskM : m;
        const uint32_t validN = taskN < n ? taskN : n;
        if (halfM >= baseM && halfM < taskM && halfM < m && halfM % 16U == 0U &&
            static_cast<uint64_t>(halfM) * validN >= minTaskElements) {
            taskM = halfM;
        } else if (halfN >= baseN && halfN < taskN && halfN < n && halfN % 16U == 0U &&
                   static_cast<uint64_t>(validM) * halfN >= minTaskElements) {
            taskN = halfN;
        } else {
            break;
        }
    }
}

static bool ConfigureVectorBuffers(const AscendC::tiling::TCubeTiling &cube, uint64_t ubSize, uint32_t n,
                                   uint32_t &vecChunkElements) {
    const uint64_t elements = static_cast<uint64_t>(cube.baseM) * cube.baseN;
    const uint64_t biasBytes = (static_cast<uint64_t>(n) + 7U) / 8U * 8U * 2U * sizeof(float);
    if (elements == 0 || elements > UINT32_MAX || ubSize <= UB_RESERVED_BYTES + biasBytes) {
        return false;
    }
    // v-stage: narrow-N tiles take a tight budget -- two FP32 arrays plus
    // the pair queues (16B/element) with a fixed 7168-element FP16 out
    // stage -- so the chunk covers a whole 86x128 tile in one batch. Wide-N
    // shapes keep the original full-out-stage budget and the 8192 cap.
    uint64_t available;
    uint64_t perElement = VEC_BYTES_PER_ELEMENT;
    if (n <= 128U) {
        const uint64_t stageBytes = 7168ULL * sizeof(uint16_t);
        const uint64_t reserve = 4U * 1024U;
        if (ubSize <= reserve + stageBytes + biasBytes) {
            return false;
        }
        available = ubSize - reserve - stageBytes - biasBytes;
        perElement = 16U;
    } else {
        available = ubSize - UB_RESERVED_BYTES - biasBytes;
        perElement = 18U;
    }
    uint64_t chunk = available / perElement;
    // v136: the old 4096 cap doubled the per-plane DMA round count on the
    // AIV epilogue for small/medium tiles (128x256: mte2 4.7us + scalar
    // 3.6us of a 12.7us AIV wall with only 0.7us of vector math). 8192
    // halves the rounds; UB still fits 3x32KB + queues + bias + 32KB reserve.
    if (n > 128U && chunk > 8192U) {
        chunk = 8192U;
    }
    chunk = chunk / VECTOR_ALIGN_ELEMENTS * VECTOR_ALIGN_ELEMENTS;
    if (chunk == 0) {
        return false;
    }
    // Local A/B only; the judge never sets this.
    const int chunkOverride = EnvI("SWGLU_CHUNK", 0);
    if (chunkOverride > 0) {
        chunk = static_cast<uint64_t>(chunkOverride) / VECTOR_ALIGN_ELEMENTS * VECTOR_ALIGN_ELEMENTS;
    }
    vecChunkElements = static_cast<uint32_t>(chunk);
    return true;
}

static uint64_t CeilDiv(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1U) / divisor;
}

static uint64_t PeakTileElements(uint32_t m, uint32_t n, const AscendC::tiling::TCubeTiling &cube, uint32_t aic) {
    const uint64_t tiles = CeilDiv(m, cube.singleCoreM) * CeilDiv(n, cube.singleCoreN);
    const uint64_t cores = tiles < aic ? tiles : aic;
    const uint64_t rows = m < cube.singleCoreM ? m : cube.singleCoreM;
    const uint64_t cols = n < cube.singleCoreN ? n : cube.singleCoreN;
    return CeilDiv(tiles, cores) * CeilDiv(rows, 16U) * 16U * CeilDiv(cols, 16U) * 16U;
}

static uint64_t InputTrafficElements(uint32_t m, uint32_t n, uint32_t k, const AscendC::tiling::TCubeTiling &cube) {
    // Model both matrix products without assuming inter-tile cache reuse.
    return 2ULL * m * k * CeilDiv(n, cube.singleCoreN) + 2ULL * n * k * CeilDiv(m, cube.singleCoreM);
}

static bool GenerateNarrowCubeTiling(platform_ascendc::PlatformAscendC &platform, uint32_t m, uint32_t n, uint32_t k,
                                     const AscendC::tiling::TCubeTiling &baseline, uint32_t targetN,
                                     AscendC::tiling::TCubeTiling &result) {
    matmul_tiling::MultiCoreMatmulTiling cube(platform);
    if (cube.SetDim(static_cast<int32_t>(PlatAic(platform))) == -1) {
        return false;
    }
    cube.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                  false);
    cube.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                  true);
    cube.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cube.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
    cube.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cube.SetBias(false);
    cube.SetShape(m, n, k);
    cube.SetOrgShape(m, n, k);
    if (cube.SetSingleShape(static_cast<int32_t>(baseline.singleCoreM), static_cast<int32_t>(targetN), -1) == -1 ||
        cube.SetFixSplit(static_cast<int32_t>(baseline.baseM), static_cast<int32_t>(targetN), -1) == -1) {
        return false;
    }
    cube.SetBufferSpace(-1, -1, -1);
    if (cube.GetTiling(result) == -1) {
        return false;
    }
    // SetFixSplit cannot fix K: accept only an unchanged SDK-generated K block.
    return result.singleCoreM == baseline.singleCoreM && result.singleCoreN == targetN &&
           result.baseM == baseline.baseM && result.baseN == targetN && result.baseK == baseline.baseK;
}

static void TryNarrowCubeTiling(platform_ascendc::PlatformAscendC &platform, uint32_t m, uint32_t n, uint32_t k,
                                uint64_t ubSize, AscendC::tiling::TCubeTiling &selected, uint32_t &chunk) {
    const uint32_t aic = PlatAic(platform);
    const AscendC::tiling::TCubeTiling baseline = selected;
    if (aic == 0U || baseline.singleCoreM <= 0 || baseline.singleCoreN <= 64 || baseline.baseM <= 0 ||
        baseline.baseK <= 0) {
        return;
    }
    const uint64_t oldTiles = CeilDiv(m, baseline.singleCoreM) * CeilDiv(n, baseline.singleCoreN);
    if (oldTiles == 0U || oldTiles >= aic) {
        return; // Fully occupied baseline: do not increase tile count or input traffic.
    }
    const uint64_t oldPeak = PeakTileElements(m, n, baseline, aic);
    const uint64_t oldTraffic = InputTrafficElements(m, n, k, baseline);
    uint64_t bestPeak = oldPeak;
    const uint32_t widths[2] = {64U, 32U};
    for (uint32_t i = 0U; i < 2U; ++i) {
        const uint32_t width = widths[i];
        if (baseline.singleCoreN <= width || (width == 32U && n > 256U)) {
            continue;
        }
        AscendC::tiling::TCubeTiling candidate = {};
        uint32_t candidateChunk = 0U;
        if (!GenerateNarrowCubeTiling(platform, m, n, k, baseline, width, candidate) ||
            !ConfigureVectorBuffers(candidate, ubSize, n, candidateChunk)) {
            continue;
        }
        const uint64_t tiles = CeilDiv(m, candidate.singleCoreM) * CeilDiv(n, candidate.singleCoreN);
        const uint64_t cores = tiles < aic ? tiles : aic;
        const uint64_t peak = PeakTileElements(m, n, candidate, aic);
        const uint64_t traffic = InputTrafficElements(m, n, k, candidate);
        // >=50% more active cores; >=25% less modeled peak tile work;
        // <=50% more modeled input traffic. These are heuristics, not profiling.
        if (cores * 2U < oldTiles * 3U || peak * 4U > oldPeak * 3U || traffic * 2U > oldTraffic * 3U ||
            peak >= bestPeak) {
            continue;
        }
        selected = candidate;
        chunk = candidateChunk;
        bestPeak = peak;
    }
}

static void TryLargeCubeTiling(platform_ascendc::PlatformAscendC &platform, uint32_t m, uint32_t n, uint32_t k,
                               uint64_t ubSize, AscendC::tiling::TCubeTiling &selected, uint32_t &chunk) {
    if (m < 4096U || static_cast<uint64_t>(m) * n < 1048576ULL) {
        return;
    }
    const auto baseline = selected;
    const uint32_t aic = PlatAic(platform);
    if (aic == 0U || baseline.singleCoreM <= 0 || baseline.singleCoreN <= 0) {
        return;
    }
    const uint64_t oldTiles = CeilDiv(m, baseline.singleCoreM) * CeilDiv(n, baseline.singleCoreN);
    // Coarsening is safe for occupancy only if at least two waves exist.
    if (oldTiles >= 2ULL * aic && baseline.singleCoreM <= m / 2U) {
        const uint32_t targetM = static_cast<uint32_t>(baseline.singleCoreM) * 2U;
        matmul_tiling::MultiCoreMatmulTiling cube(platform);
        const bool dimReady = cube.SetDim(static_cast<int32_t>(aic)) != -1;
        cube.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                      false);
        cube.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                      true);
        cube.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        cube.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16);
        cube.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
        cube.SetBias(false);
        cube.SetShape(m, n, k);
        cube.SetOrgShape(m, n, k);
        cube.SetBufferSpace(-1, -1, -1);
        AscendC::tiling::TCubeTiling candidate = {};
        uint32_t candidateChunk = 0U;
        if (dimReady && cube.SetSingleShape(static_cast<int32_t>(targetM), baseline.singleCoreN, -1) != -1 &&
            cube.SetFixSplit(baseline.baseM, baseline.baseN, -1) != -1 && cube.GetTiling(candidate) != -1 &&
            candidate.singleCoreM == targetM && candidate.singleCoreN == baseline.singleCoreN &&
            candidate.singleCoreK == baseline.singleCoreK && candidate.baseM == baseline.baseM &&
            candidate.baseN == baseline.baseN && candidate.baseK == baseline.baseK &&
            ConfigureVectorBuffers(candidate, ubSize, n, candidateChunk)) {
            const uint64_t newTiles = CeilDiv(m, targetM) * CeilDiv(n, candidate.singleCoreN);
            if (newTiles >= aic && newTiles < oldTiles &&
                PeakTileElements(m, n, candidate, aic) <= PeakTileElements(m, n, baseline, aic) &&
                InputTrafficElements(m, n, k, candidate) <= InputTrafficElements(m, n, k, baseline)) {
                selected = candidate;
                chunk = candidateChunk;
            }
        }
    }
    // Same buffers and same 32KB library reserve; only enlarge the DMA batch.
    // Kernel subdivides SiLU calls to <=4096, preserving baseline scratch demand.
    const uint64_t biasBytes = (static_cast<uint64_t>(n) + 7U) / 8U * 8U * 2U * sizeof(float);
    if (ubSize > UB_RESERVED_BYTES + biasBytes) {
        uint64_t candidateChunk = (ubSize - UB_RESERVED_BYTES - biasBytes) / VEC_BYTES_PER_ELEMENT;
        if (candidateChunk > 8192U) {
            candidateChunk = 8192U;
        }
        candidateChunk = candidateChunk / VECTOR_ALIGN_ELEMENTS * VECTOR_ALIGN_ELEMENTS;
        if (candidateChunk > chunk) {
            chunk = static_cast<uint32_t>(candidateChunk);
        }
    }
}

// v95: 把 singleCoreN 对齐到 baseN 的整数倍。
// 触发条件很窄：只有当 SDK 给出的 singleCoreN 不是 16 的倍数时（sN=171/86 等）
// 才介入；对齐后重新搜一个 targetM，接受条件同时约束
//   1) 每核 tile 数不超过原来的 2 倍
//   2) 建模输入搬运量不超过原来的 1.2 倍
// 两条都过才替换，否则原样返回（零副作用）。
static void TryAlignedTiling(platform_ascendc::PlatformAscendC &platform, uint32_t m, uint32_t n, uint32_t k,
                             uint64_t ubSize, AscendC::tiling::TCubeTiling &selected, uint32_t &chunk) {
    const uint32_t aic = PlatAic(platform);
    const auto baseline = selected;
    if (aic == 0U || baseline.baseN <= 0 || baseline.baseK <= 0 || n % 16U != 0U || m % 16U != 0U) {
        return;
    }
    const uint32_t baseN = static_cast<uint32_t>(baseline.baseN);
    if (n % baseN != 0U) {
        return;
    }
    if (static_cast<uint32_t>(baseline.singleCoreN) % 16U == 0U) {
        return;
    }
    const uint64_t oldTiles = CeilDiv(m, baseline.singleCoreM) * CeilDiv(n, baseline.singleCoreN);
    const uint64_t oldPerCore = CeilDiv(oldTiles, aic);
    AscendC::tiling::TCubeTiling best = {};
    uint32_t bestChunk = 0U;
    uint64_t bestPerCore = UINT64_MAX;
    const int32_t baseMCandidates[] = {128, 64, 48, 32};
    const uint32_t sN = baseN;
    const uint32_t nTilesN = n / sN;
    for (auto tryBaseM : baseMCandidates) {
        for (uint32_t factor = 2U; factor <= 16U; ++factor) {
            const uint32_t targetM = static_cast<uint32_t>(tryBaseM) * factor;
            if (targetM > m)
                break;
            const uint64_t tiles = CeilDiv(m, targetM) * nTilesN;
            if (tiles < aic)
                continue;
            const uint64_t perCore = CeilDiv(tiles, aic);
            if (perCore > oldPerCore * 2U)
                continue;
            if (perCore >= bestPerCore)
                continue;
            AscendC::tiling::TCubeTiling candidate = {};
            uint32_t candidateChunk = 0U;
            if (!GenerateCubeTiling(platform, m, n, k, tryBaseM, baseline.baseN, candidate, 0U,
                                    static_cast<int32_t>(targetM), static_cast<int32_t>(sN)) ||
                candidate.singleCoreM != static_cast<int32_t>(targetM) ||
                candidate.singleCoreN != static_cast<int32_t>(sN) || candidate.baseM != tryBaseM ||
                candidate.baseN != baseline.baseN || !ConfigureVectorBuffers(candidate, ubSize, n, candidateChunk)) {
                continue;
            }
            // v97: 只有「同时减少了 K 迭代次数」的候选才接受。
            // 依据本机实测（同一 v96 构建，两个形状的切分是唯一变量）：
            //   1024x1024  bK 64 -> 128（K 迭代 16->8），净赚 11.2%
            //   512x1024   bK 128 -> 128（未改善），只是 bM 128->64，
            //              结果 mte1 6.54->9.15(+40%)、整体 +5.4%
            // 即「单纯把 sN 对齐」本身没有机制收益，必须附带 bK 改善才划算；
            // 这也让改动面收敛到"确实有机制依据"的形状上，降低回归风险。
            if (candidate.baseK <= baseline.baseK) {
                continue;
            }
            bestPerCore = perCore;
            best = candidate;
            bestChunk = candidateChunk;
        }
    }
    if (bestPerCore < UINT64_MAX && bestPerCore <= oldPerCore) {
        if (InputTrafficElements(m, n, k, best) <= InputTrafficElements(m, n, k, baseline) * 6ULL / 5ULL) {
            selected = best;
            chunk = bestChunk;
        }
    }
}

} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    const gert::Tensor *w1 = context->GetRequiredInputTensor(1);
    const gert::Tensor *b1 = context->GetRequiredInputTensor(2);
    const gert::Tensor *w2 = context->GetRequiredInputTensor(3);
    const gert::Tensor *b2 = context->GetRequiredInputTensor(4);
    if (x == nullptr || w1 == nullptr || b1 == nullptr || w2 == nullptr || b2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = x->GetOriginShape();
    const int64_t rank = xShape.GetDimNum();
    if (rank < 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t k64 = xShape.GetDim(rank - 1);
    const int64_t total64 = xShape.GetShapeSize();
    if (k64 <= 0 || total64 <= 0 || total64 % k64 != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t m64 = total64 / k64;
    if (m64 > MAX_PREFIX_ELEMENTS || k64 > MAX_FEATURE_DIM) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t m = static_cast<uint32_t>(m64);
    const uint32_t n = static_cast<uint32_t>(k64);
    const uint32_t k = static_cast<uint32_t>(k64);

    const auto w1Shape = w1->GetOriginShape();
    const auto w2Shape = w2->GetOriginShape();
    const auto b1Shape = b1->GetOriginShape();
    const auto b2Shape = b2->GetOriginShape();
    if (w1Shape.GetDimNum() != 2 || w1Shape.GetDim(0) != n || w1Shape.GetDim(1) != k || w2Shape.GetDimNum() != 2 ||
        w2Shape.GetDim(0) != n || w2Shape.GetDim(1) != k || b1Shape.GetDimNum() != 1 || b1Shape.GetDim(0) != n ||
        b2Shape.GetDimNum() != 1 || b2Shape.GetDim(0) != n || x->GetDataType() != ge::DT_FLOAT16 ||
        w1->GetDataType() != ge::DT_FLOAT16 || b1->GetDataType() != ge::DT_FLOAT16 ||
        w2->GetDataType() != ge::DT_FLOAT16 || b2->GetDataType() != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    SwiGluTilingData *tiling = context->GetTilingData<SwiGluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    tiling->m = m;
    tiling->n = n;
    tiling->k = k;
    tiling->reserved = 0;
    tiling->totalOutputs = static_cast<uint64_t>(m) * n;
    tiling->blocksPerCore = 0;
    tiling->extraBlocks = 0;
    tiling->usedCoreNum = 0;
    tiling->cvTaskM = 0U;
    tiling->cvTaskN = 0U;
    // Small-M matrix-vector family. Each worker caches 32 output weight rows.
    if (k >= 65U && k <= 256U && tiling->totalOutputs <= 512ULL && ubSize >= 128U * 1024U &&
        !(k == 128U && tiling->totalOutputs <= 256ULL) && !(k == 256U && m == 1U)) {
        const uint32_t av = PlatAiv(platform), nb = static_cast<uint32_t>(CeilDiv(n, k % 16U == 0U ? 32U : 16U));
        const uint32_t tasks = m * nb;
        uint32_t used = av < 32U ? av : 32U;
        if (used > tasks)
            used = tasks;
        if (used >= nb)
            used = used / nb * nb;
        if (used != 0U) {
            tiling->usedCoreNum = used;
            tiling->blockDim = used < 8U ? (av < 8U ? av : 8U) : used;
            context->SetBlockDim(tiling->blockDim);
            context->SetTilingKey(32U);
            size_t *workspace = context->GetWorkspaceSizes(1);
            if (workspace == nullptr)
                return ge::GRAPH_FAILED;
            workspace[0] = 0U;
            return ge::GRAPH_SUCCESS;
        }
    }
    // Combined candidate: specialized d=1; wider small-output Vector coverage.
    const bool scalarFeature = n == 1U;
    const bool smallVector =
        n <= 256U && ubSize >= 128U * 1024U &&
        (tiling->totalOutputs <= 512U ||
         (k <= 16U && (tiling->totalOutputs <= 2048U || (n <= 2U && tiling->totalOutputs <= 8192U))));
    if (scalarFeature || smallVector) {
        const uint32_t aivNum = PlatAiv(platform);
        // v113: small d=1 batches use a 32-element chunk so the M outputs are
        // spread over many AIVs and each core runs a shorter scalar repair loop.
        // Large batches keep the 256-element chunk to bound the loop count.
        uint32_t outputsPerBlock = scalarFeature ? (tiling->totalOutputs <= 4096ULL ? 256U : 512U) : 16U;
        if (!scalarFeature) {
            // key4 tiny-cube and k>32 vector paths keep the proven 16.
            // key7 (k<=32) size tiers, measured with the 32-wide inner block:
            outputsPerBlock = 16U;
            if (k <= 32U) {
                const uint64_t total = tiling->totalOutputs;
                if (n == 2U) {
                    outputsPerBlock = total <= 2048ULL ? 32U : (total <= 4096ULL ? 64U : 16U);
                } else if (total <= 512ULL) {
                    // 16-output blocks double the active AIVs for the tiny tier
                    // while keeping blockDim at its measured sweet spot.
                    outputsPerBlock = 16U;
                } else if (total <= 1024ULL) {
                    outputsPerBlock = 64U;
                }
            }
            const int vblk = EnvI("SWGLU_VBLK", 0);
            if (vblk >= 1 && vblk <= 128) {
                outputsPerBlock = static_cast<uint32_t>(vblk);
            }
        }
        const uint64_t outputBlocks = (tiling->totalOutputs + outputsPerBlock - 1U) / outputsPerBlock;
        if (aivNum == 0U || ubSize < 64U * 1024U) {
            return ge::GRAPH_FAILED;
        }
        // Large d=1 batches gain from the wider AIV set: at 2^20 outputs
        // 40 AIVs measured -7.4% versus 32, while 48 regressed (launch and
        // bandwidth saturation). Keep the original 32 below this size tier.
        const uint32_t d1CoreCap = tiling->totalOutputs >= 524288ULL ? 40U : 32U;
        uint32_t coreCap = scalarFeature ? d1CoreCap : 16U;
        if (!scalarFeature) {
            const int vcap = EnvI("SWGLU_VCAP", 16);
            if (vcap >= 1 && vcap <= 48) {
                coreCap = static_cast<uint32_t>(vcap);
            }
        }
        const uint32_t available = aivNum < coreCap ? aivNum : coreCap;
        tiling->usedCoreNum = static_cast<uint32_t>(outputBlocks < available ? outputBlocks : available);
        const uint32_t launchFloor =
            static_cast<uint32_t>(EnvI("SWGLU_LNCH", static_cast<int>(MIN_VECTOR_LAUNCH_CORES)));
        const uint32_t launchTarget = tiling->usedCoreNum < launchFloor ? launchFloor : tiling->usedCoreNum;
        tiling->blockDim = launchTarget < aivNum ? launchTarget : aivNum;
        tiling->blocksPerCore = outputBlocks / tiling->usedCoreNum;
        tiling->extraBlocks = static_cast<uint32_t>(outputBlocks % tiling->usedCoreNum);
        tiling->vecChunkElements = outputsPerBlock;
        tiling->cubeTilingData = {};
        const bool leanEligible = !scalarFeature && k <= 32U;
        // K=64 aligned small-output shapes take the TPipe-free TinyLean64
        // path: full W cache, no queues, no gather/DirectTiny/multi-batch.
        const bool lean64 = !scalarFeature && k >= 33U && k <= 64U && n == k && tiling->totalOutputs <= 512ULL;
        for (uint32_t leanSlot = 0U; leanSlot < 8U; ++leanSlot) {
            tiling->leanSlots[leanSlot].slot.totalOutputs = tiling->totalOutputs;
            tiling->leanSlots[leanSlot].slot.blocksPerCore = tiling->blocksPerCore;
            tiling->leanSlots[leanSlot].slot.m = m;
            tiling->leanSlots[leanSlot].slot.n = n;
            tiling->leanSlots[leanSlot].slot.k = k;
            tiling->leanSlots[leanSlot].slot.extraBlocks = tiling->extraBlocks;
            tiling->leanSlots[leanSlot].slot.usedCoreNum = tiling->usedCoreNum;
            tiling->leanSlots[leanSlot].slot.vecChunkElements = tiling->vecChunkElements;
        }
        uint32_t vectorKey = leanEligible ? 7U : (lean64 ? 8U : (scalarFeature ? 3U : 1U));
        // Only a single logical work block: idle blocks exit before reading anything,
        // and block 0 needs just the low 32 bits. All other shapes keep key 7/8.
        if (leanEligible && n == k && k >= 2U && k <= 64U && tiling->totalOutputs >= 1ULL &&
            tiling->totalOutputs <= 8192ULL && outputsPerBlock >= 1U && outputsPerBlock <= 128U &&
            tiling->usedCoreNum == 1U && tiling->blocksPerCore == 1ULL && tiling->extraBlocks == 0U) {
            const uint64_t entry = (tiling->totalOutputs - 1ULL) | (static_cast<uint64_t>(k - 1U) << 13U) |
                                   (static_cast<uint64_t>(outputsPerBlock - 1U) << 19U);
            for (uint32_t slot = 0U; slot < 8U; ++slot) {
                tiling->leanSlots[slot].slot.totalOutputs = entry;
            }
            vectorKey = 12U;
        } else if (leanEligible && n == k && k >= 2U && k <= 64U && tiling->totalOutputs >= 1ULL &&
                   tiling->totalOutputs <= 8192ULL && outputsPerBlock >= 1U && outputsPerBlock <= 128U &&
                   (tiling->usedCoreNum == 2U || tiling->usedCoreNum == 4U) && tiling->blocksPerCore == 1ULL &&
                   tiling->extraBlocks == 0U) {
            // One work block per worker; the key carries the worker count, so
            // idle blocks exit without reading any metadata.
            const uint64_t entry = (tiling->totalOutputs - 1ULL) | (static_cast<uint64_t>(k - 1U) << 13U) |
                                   (static_cast<uint64_t>(outputsPerBlock - 1U) << 19U);
            for (uint32_t slot = 0U; slot < 8U; ++slot) {
                tiling->leanSlots[slot].slot.totalOutputs = entry;
            }
            vectorKey = 11U + tiling->usedCoreNum;
        }
        // Multi-block row DMA benefits from slicing; flat copies keep the
        // original engine with no per-worker slice arithmetic.
        if (k > 16U && k % 8U != 0U) {
            if (vectorKey == 7U) {
                vectorKey = 16U;
            } else if (vectorKey == 8U) {
                vectorKey = 17U;
            } else if (vectorKey == 13U) {
                vectorKey = 18U;
            } else if (vectorKey == 15U) {
                vectorKey = 19U;
            }
        }
        if (scalarFeature && tiling->totalOutputs == 1ULL)
            vectorKey = 30U;
        if (m == 1U && k >= 2U && k <= 4U)
            vectorKey = 4000U + k;
        if (m == 1U && (k == 32U || k == 64U))
            vectorKey = 5000U + k;
        if (m >= 2U && tiling->totalOutputs <= 256ULL && (k == 32U || k == 64U))
            vectorKey = 50000U + k * 16U + m;
        if (tiling->totalOutputs > 256ULL && tiling->totalOutputs <= 512ULL && (k == 32U || k == 64U))
            vectorKey = 60000U + k * 16U + m;
        if (tiling->totalOutputs <= 256ULL && (k == 128U || k == 256U))
            vectorKey = 70000U + k * 4U + m;
        context->SetTilingKey(vectorKey);
        context->SetBlockDim(tiling->blockDim);
        size_t *workspace = context->GetWorkspaceSizes(1);
        if (workspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspace[0] = 0; // Pure Vector has no Cube or intermediate GM planes.
        return ge::GRAPH_SUCCESS;
    }

    // Try the old fast version's 128-column split first, then legal library fallbacks.
    // CAP16: short-K small/mid grids (m<=512, n<=256) measured faster on 16
    // workers than on 24 (e.g. 512x256: 18.6 -> 12.9 local); the same grid on
    // 24 workers loses to descriptor/latency overhead. Larger m (1024+),
    // larger K, and the 2M+ key5 grids keep the original 24-worker policy.
    uint32_t selectedCap = 0U;
    const uint64_t capMac = static_cast<uint64_t>(m) * n * k;
    const uint64_t capOutputs = static_cast<uint64_t>(m) * n;
    if (k >= 64U && k <= 256U && m <= 512U && n <= 256U && capMac >= 2097152ULL && capMac < 134217728ULL &&
        capOutputs < 2097152ULL) {
        selectedCap = 16U;
    }
    CubeCoreCapScope coreCapScope(selectedCap);
    bool tilingReady = GenerateCubeTiling(platform, m, n, k, -1, PREFERRED_BASE_N, tiling->cubeTilingData) &&
                       ConfigureVectorBuffers(tiling->cubeTilingData, ubSize, n, tiling->vecChunkElements);
    if (!tilingReady) {
        tilingReady = GenerateCubeTiling(platform, m, n, k, -1, ALTERNATE_BASE_N, tiling->cubeTilingData) &&
                      ConfigureVectorBuffers(tiling->cubeTilingData, ubSize, n, tiling->vecChunkElements);
    }
    if (!tilingReady) {
        tilingReady =
            GenerateCubeTiling(platform, m, n, k, PREFERRED_BASE_N, PREFERRED_BASE_N, tiling->cubeTilingData) &&
            ConfigureVectorBuffers(tiling->cubeTilingData, ubSize, n, tiling->vecChunkElements);
    }
    if (!tilingReady) {
        tilingReady = GenerateCubeTiling(platform, m, n, k, -1, -1, tiling->cubeTilingData) &&
                      ConfigureVectorBuffers(tiling->cubeTilingData, ubSize, n, tiling->vecChunkElements);
    }
    if (!tilingReady) {
        tilingReady = GenerateCubeTiling(platform, m, n, k, FALLBACK_BASE_N, FALLBACK_BASE_N, tiling->cubeTilingData) &&
                      ConfigureVectorBuffers(tiling->cubeTilingData, ubSize, n, tiling->vecChunkElements);
    }
    if (!tilingReady) {
        return ge::GRAPH_FAILED;
    }
    tiling->m = m;
    tiling->n = n;
    tiling->k = k;
    tiling->reserved = 0;

    TryNarrowCubeTiling(platform, m, n, k, ubSize, tiling->cubeTilingData, tiling->vecChunkElements);
    if (kEnableAlignedTiling) {
        TryAlignedTiling(platform, m, n, k, ubSize, tiling->cubeTilingData, tiling->vecChunkElements);
    }
    TryLargeCubeTiling(platform, m, n, k, ubSize, tiling->cubeTilingData, tiling->vecChunkElements);

    const uint32_t aicNum = PlatAic(platform);
    if (aicNum == 0) {
        return ge::GRAPH_FAILED;
    }
    // 用 tiling 里的 singleCoreM/singleCoreN 自己算 tile 总数，不再依赖 cube.usedCoreNum。
    // 旧版直接拿 cube.usedCoreNum 当"实际处理的 tile 数"，一旦它小于 tile 总数，
    // 尾部 tile 就既不算 cube 也不做后处理 → 输出残留脏数据（且与精度/分核方式无关）。
    uint32_t sM = tiling->cubeTilingData.singleCoreM;
    uint32_t sN = tiling->cubeTilingData.singleCoreN;
    if (sM == 0U || sN == 0U) {
        return ge::GRAPH_FAILED;
    }
    ChooseCvTaskShape(tiling->cubeTilingData, m, n, aicNum, sM, sN);
    tiling->cvTaskM = sM;
    tiling->cvTaskN = sN;
    const uint64_t mBlocks = (static_cast<uint64_t>(m) + sM - 1U) / sM;
    const uint64_t nBlocks = (static_cast<uint64_t>(n) + sN - 1U) / sN;
    const uint64_t totalTiles = mBlocks * nBlocks;
    if (totalTiles == 0ULL) {
        return ge::GRAPH_FAILED;
    }
    // 启动核数：取 aic 与 tile 总数的较小者。kernel 侧按 t = blockIdx + k*blockDim 遍历全部
    // tile，所以无论这里给多少核，覆盖都是完备的。
    const uint32_t usedCoreNum =
        static_cast<uint32_t>(totalTiles < static_cast<uint64_t>(aicNum) ? totalTiles : static_cast<uint64_t>(aicNum));
    // blockDim in the payload is the LOGICAL AIV stride; launch uses physical AICs.
    tiling->blockDim = usedCoreNum;
    tiling->usedCoreNum = tiling->blockDim;
    context->SetBlockDim(tiling->blockDim);

    // The auto-picked cube route is always the basic-API L1-resident engine
    // (key 4): it beats the MatmulImpl path on every measured shape class.
    context->SetTilingKey(4);

    // v127: on the mid-size key-4 shapes the SDK's "tiles == cores" balance
    // leaves exactly one task per AIC, where the AIC and its paired AIV can
    // only run one after the other (no cross-tile ring overlap), and the wall
    // becomes cube-work + epilogue-work. Split N (then M) down to 16-aligned
    // halves until every core holds at least two rounds.
    if (context->GetTilingKey() == 4U && static_cast<uint64_t>(m) * n * k >= 200000000ULL) {
        uint32_t tM = tiling->cvTaskM;
        uint32_t tN = tiling->cvTaskN;
        const uint64_t oldTiles = CeilDiv(m, tM) * CeilDiv(n, tN);
        const uint32_t oldUsed =
            static_cast<uint32_t>(oldTiles < static_cast<uint64_t>(aicNum) ? oldTiles : static_cast<uint64_t>(aicNum));
        for (uint32_t guard = 0U; guard < 6U; ++guard) {
            if (CeilDiv(m, tM) * CeilDiv(n, tN) >= 2ULL * aicNum) {
                break;
            }
            bool split = false;
            // 16-aligned candidates only; a task tail below 16 columns/rows
            // traps in Fixpipe/Nd2Nz (507015), so keep exact or >=16 tails.
            for (uint32_t cand = tN / 2U / 16U * 16U; cand >= 16U; cand -= 16U) {
                const uint32_t tail = n % cand;
                if ((tail == 0U || tail >= 16U) && CeilDiv(n, cand) > CeilDiv(n, tN)) {
                    tN = cand;
                    split = true;
                    break;
                }
            }
            if (split) {
                continue;
            }
            for (uint32_t cand = tM / 2U / 16U * 16U; cand >= 16U; cand -= 16U) {
                const uint32_t tail = m % cand;
                if ((tail == 0U || tail >= 16U) && CeilDiv(m, cand) > CeilDiv(m, tM)) {
                    tM = cand;
                    split = true;
                    break;
                }
            }
            if (!split) {
                break;
            }
        }
        // v129/v131: the kernel's L1-resident bulk path is layout-proven for
        // task extents <= 128 and k <= 512 (k > 256 is consumed in half-K
        // units). Clamp the task shape to 128 for those k; the final
        // acceptance below filters it out when the tile count would drop
        // below two rounds per core.
        if (k <= 512U) {
            if (tM > 128U && (m % 128U == 0U || m % 128U >= 16U)) {
                tM = 128U;
            }
            if (tN > 128U && (n % 128U == 0U || n % 128U >= 16U)) {
                tN = 128U;
            }
        }
        const uint64_t newTiles = CeilDiv(m, tM) * CeilDiv(n, tN);
        if ((tM != tiling->cvTaskM || tN != tiling->cvTaskN) && newTiles >= 2ULL * aicNum) {
            const uint32_t used = static_cast<uint32_t>(
                newTiles < static_cast<uint64_t>(aicNum) ? newTiles : static_cast<uint64_t>(aicNum));
            if (used >= oldUsed) {
                tiling->cvTaskM = tM;
                tiling->cvTaskN = tN;
                tiling->blockDim = used;
                tiling->usedCoreNum = used;
                context->SetBlockDim(used);
            }
        }
    }

    // v143: small-shape parallelism. On tiny MAC sizes the library's traffic
    // guard rejects narrow tilings (e.g. 256^3 keeps a 2-4 tile grid), so the
    // whole wall rides on the per-task prologue (W panel fetch, launch) of a
    // handful of AICs. For k <= 256 the L1-resident bulk path is layout-proven
    // for task extents <= 128, so clamp an under-occupied task grid to 64
    // (same tail rule as v127: no task below 16 elements) and spread the work
    // over more cores. Shapes that already hold >= 8 tasks are untouched.
    if (context->GetTilingKey() == 4U && k <= 256U) {
        uint32_t tM = tiling->cvTaskM;
        uint32_t tN = tiling->cvTaskN;
        if (tM != 0U && tN != 0U) {
            const uint64_t tilesNow = CeilDiv(m, tM) * CeilDiv(n, tN);
            if (tilesNow < 8ULL) {
                if (tN > 64U && (n % 64U == 0U || n % 64U >= 16U) && CeilDiv(n, 64U) > CeilDiv(n, tN)) {
                    tN = 64U;
                }
                if (tM > 64U && (m % 64U == 0U || m % 64U >= 16U) && CeilDiv(m, 64U) > CeilDiv(m, tM)) {
                    tM = 64U;
                }
                // v148: if the 64-clamped grid still leaves AICs idle
                // (128x256 ends at 8 tasks), halve N once more so up to 16
                // AICs share the wall; n=32 stays inside the proven <=128
                // task layouts.
                if (CeilDiv(m, tM) * CeilDiv(n, tN) < 16ULL && tN > 32U && (n % 32U == 0U || n % 32U >= 16U) &&
                    CeilDiv(n, 32U) > CeilDiv(n, tN)) {
                    tN = 32U;
                }
                const uint64_t newTiles = CeilDiv(m, tM) * CeilDiv(n, tN);
                if (newTiles > tilesNow) {
                    const uint32_t used = static_cast<uint32_t>(
                        newTiles < static_cast<uint64_t>(aicNum) ? newTiles : static_cast<uint64_t>(aicNum));
                    tiling->cvTaskM = tM;
                    tiling->cvTaskN = tN;
                    tiling->blockDim = used;
                    tiling->usedCoreNum = used;
                    context->SetBlockDim(used);
                }
            }
        }
    }

    // v-tail: ragged shapes (e.g. 65x33, 72x36) keep a single huge library
    // task because the 64/32 clamps above cannot pass their tail rule
    // (a task tail below 16 rows/columns traps Fixpipe/Nd2Nz). Only when the
    // library left exactly one task, search a wider set of tail-safe widths
    // so the work still spreads over workers; grids that already hold two or
    // more tasks are left alone (measured: further splitting them regresses).
    if (context->GetTilingKey() == 4U && k <= 256U) {
        uint32_t tM = tiling->cvTaskM;
        uint32_t tN = tiling->cvTaskN;
        if (tM != 0U && tN != 0U) {
            const uint64_t tilesNow = CeilDiv(m, tM) * CeilDiv(n, tN);
            if (tilesNow <= 2ULL) {
                uint32_t bestM = tM;
                uint32_t bestN = tN;
                uint64_t best = tilesNow;
                for (uint32_t cand = 16U; cand < bestN; ++cand) {
                    const uint32_t tail = n % cand;
                    const uint64_t tiles = CeilDiv(m, bestM) * CeilDiv(n, cand);
                    if ((tail == 0U || tail >= 16U) && tiles > best) {
                        bestN = cand;
                        best = tiles;
                        break;
                    }
                }
                for (uint32_t cand = 16U; cand < bestM; ++cand) {
                    const uint32_t tail = m % cand;
                    const uint64_t tiles = CeilDiv(m, cand) * CeilDiv(n, bestN);
                    if ((tail == 0U || tail >= 16U) && tiles > best) {
                        bestM = cand;
                        best = tiles;
                        break;
                    }
                }
                if (best > tilesNow && best <= 12ULL) {
                    const uint32_t used = static_cast<uint32_t>(
                        best < static_cast<uint64_t>(aicNum) ? best : static_cast<uint64_t>(aicNum));
                    tiling->cvTaskM = bestM;
                    tiling->cvTaskN = bestN;
                    tiling->blockDim = used;
                    tiling->usedCoreNum = used;
                    context->SetBlockDim(used);
                }
            }
        }
    }

    // v154: 16-task grids run two tasks per AIC (8 cores) with the
    // N-fastest order in the kernel, so the second task's whole fetch hides
    // behind the first and W panels stay resident.
    if (context->GetTilingKey() == 4U && k <= 256U) {
        const uint64_t tMv154 = tiling->cvTaskM;
        const uint64_t tNv154 = tiling->cvTaskN;
        if (tMv154 != 0U && tNv154 != 0U) {
            const uint64_t tiles = CeilDiv(m, tMv154) * CeilDiv(n, tNv154);
            if (tiles == 16ULL && aicNum >= 8U) {
                tiling->blockDim = 8U;
                tiling->usedCoreNum = 8U;
                context->SetBlockDim(8U);
            }
        }
    }

    // Default Cube shapes enter direct key 2, without a key-11 shape gate.

    // key 9 disabled: ProcessRoutePipeline has a known hang bug for certain
    // tile configurations (507046 stream sync timeout). These shapes fall back
    // to the safe key 2 serial path instead.

    // v-c5-2: two tiles per AIC for the narrow short-K slab family. With 48
    // balanced tiles every AIC holds exactly two, so the key4 cross-tile
    // handshake pipelines cube(i+1) with the epilogue of tile i. Measured on
    // the 3585..4096x128 family: -0.7..-1.0us versus the library task grid
    // (tm=64 / tm=128 grids and one-tile-per-core lose; m=2048 regresses, so
    // the rule starts at 3585). Arithmetic and rounding are unchanged (each
    // output keeps its exact K order), outputs stay bit-equal.
    if (n <= 128U && k <= 128U && m >= 3585U && m <= 8192U && context->GetTilingKey() == 4U) {
        uint32_t tm = (m + 47U) / 48U;
        // Balance to exactly three rounds over the real AIC count when the
        // resulting task still fits the single-batch / plane-split capacity.
        const uint32_t rounds3 = 3U * aicNum;
        const uint32_t tm3 = rounds3 != 0U ? (m + rounds3 - 1U) / rounds3 : 0U;
        const uint32_t tail3 = tm3 != 0U ? (m % tm3) : 0U;
        if (tm3 >= 16U && tm3 <= 128U && (tail3 == 0U || tail3 >= 16U) &&
            static_cast<uint64_t>(tm3) * CeilDiv(tiling->cvTaskN, 64U) * 64U <= tiling->vecChunkElements) {
            tm = tm3;
        }
        const uint32_t tail = (tm != 0U) ? (m % tm) : 0U;
        if (tm >= 16U && (tail == 0U || tail >= 16U)) {
            const uint64_t tiles = CeilDiv(m, tm);
            const uint32_t used =
                static_cast<uint32_t>(tiles < static_cast<uint64_t>(aicNum) ? tiles : static_cast<uint64_t>(aicNum));
            tiling->cvTaskM = tm;
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            context->SetBlockDim(used);
        }
    }
    // v173: mid-M x n=128 tiling (single N block, single K slab <=128).
    // The single-batch epilogue needs tm*n <= vecChunk, so tm is capped by the
    // device capacity; within that cap about 60% of the AIC count cuts weight
    // re-fetches versus the auto 20-tile grid without adding rounds. Derived
    // from runtime resources only, and any failed guard keeps auto tiling.
    if (context->GetTilingKey() == 4U && n == 128U && k <= 128U && m >= 640U && m <= 1280U) {
        const uint32_t capTm = tiling->vecChunkElements / 128U;
        uint32_t goal = aicNum >= 4U ? (aicNum * 3U) / 5U : aicNum;
        if (goal == 0U) {
            goal = 1U;
        }
        uint32_t tm = (m + goal - 1U) / goal;
        if (tm > capTm) {
            tm = capTm;
        }
        if (tm > 128U) {
            tm = 128U;
        }
        if (tm < 16U) {
            tm = 16U;
        }
        uint32_t tail = m % tm;
        while (tm > 16U && tail != 0U && tail < 16U) {
            --tm;
            tail = m % tm;
        }
        const uint64_t tiles = CeilDiv(m, tm);
        if (tm >= 16U && static_cast<uint64_t>(tm) * 128U <= tiling->vecChunkElements &&
            tiles <= static_cast<uint64_t>(aicNum)) {
            tiling->cvTaskM = tm;
            tiling->cvTaskN = 128U;
            tiling->blockDim = static_cast<uint32_t>(tiles);
            tiling->usedCoreNum = static_cast<uint32_t>(tiles);
            context->SetBlockDim(static_cast<uint32_t>(tiles));
        }
    }
    // v86: the 256-column mid-M grid lands at one tile per AIC (1024x256:
    // tm=94, 22 tiles), so the paired AIV cannot overlap the next tile.
    // Halve the M task to 64 there, keeping a tail-safe grid with a second round.
    if (context->GetTilingKey() == 4U && n == 256U && k == 256U && m >= 768U && m <= 1536U && tiling->cvTaskM > 64U) {
        const uint32_t tail = m % 64U;
        const uint64_t tiles = CeilDiv(m, 64U) * CeilDiv(n, tiling->cvTaskN);
        if ((tail == 0U || tail >= 16U) && tiles > static_cast<uint64_t>(aicNum) &&
            static_cast<uint64_t>(64U) * CeilDiv(tiling->cvTaskN, 16U) * 16U <= tiling->vecChunkElements) {
            const uint32_t used =
                static_cast<uint32_t>(tiles < static_cast<uint64_t>(aicNum) ? tiles : static_cast<uint64_t>(aicNum));
            tiling->cvTaskM = 64U;
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            context->SetBlockDim(used);
        }
    }
    // v78: the L1-resident bulk engine requires a task extent <=128 in M.
    // The older ragged-tail rule only searched when the tail was 1..15, so
    // shapes like 2048x256 kept tm=171 and fell into the slab path
    // (historical 16384x256: bulk 31.1 vs slab 44.4 us). Pull every oversized
    // M task back to the largest tail-safe width <=128 while the grid still
    // covers at least one full wave.
    if (context->GetTilingKey() == 4U && k <= 512U && m >= 256U && tiling->cvTaskM > 128U) {
        uint32_t cand = 128U;
        uint32_t tail = m % cand;
        while (cand > 16U && tail != 0U && tail < 16U) {
            --cand;
            tail = m % cand;
        }
        const uint64_t tiles = CeilDiv(m, cand) * CeilDiv(n, tiling->cvTaskN);
        const uint32_t used =
            static_cast<uint32_t>(tiles < static_cast<uint64_t>(aicNum) ? tiles : static_cast<uint64_t>(aicNum));
        if (cand < tiling->cvTaskM && tiles >= static_cast<uint64_t>(aicNum)) {
            tiling->cvTaskM = cand;
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            context->SetBlockDim(used);
        }
    }
    // v-c8t3: tiny-cube family. A 6-12 tile grid pays the full per-tile
    // prologue (W fetch, A Nd2Nz, L0 chain, handshakes) on every core for
    // a few hundred outputs; three one-chain tiles measured fastest
    // (65x33 -0.23, 88x33 -1.06, 96x44 -1.58 us vs the fine grid).
    if (context->GetTilingKey() == 4U && k <= 64U && m >= 32U && m <= 128U && n >= 17U && n <= 64U) {
        const uint32_t tm3 = (m + 2U) / 3U;
        const uint32_t tail3 = tm3 != 0U ? (m % tm3) : 0U;
        if (tm3 >= 16U && (tail3 == 0U || tail3 >= 16U)) {
            tiling->cvTaskM = tm3;
            tiling->cvTaskN = n;
            tiling->blockDim = 3U;
            tiling->usedCoreNum = 3U;
            context->SetBlockDim(3U);
        }
    }
    // v-rag: ragged M tails (m % tm in 1..15) either waste a prologue-only
    // sliver tile or, when they pushed tm above 128, drop the shape out of
    // the WP bulk engine entirely (measured 16384x256 bulk 31.1 vs
    // 16385x256 slab 44.4 us at k=256). Search downward for the first
    // tail-safe width, preferring to stay at or below 128.
    if ((context->GetTilingKey() == 4U || context->GetTilingKey() == 5U) && k <= 256U && tiling->cvTaskM != 0U) {
        const uint32_t tailNow = m % tiling->cvTaskM;
        if (tailNow != 0U && tailNow < 16U) {
            const uint32_t startCand =
                tiling->cvTaskM > 128U ? 128U : (tiling->cvTaskM > 16U ? tiling->cvTaskM - 1U : 16U);
            for (uint32_t cand = startCand; cand >= 16U; --cand) {
                const uint32_t t2 = m % cand;
                if (t2 == 0U || t2 >= 16U) {
                    const uint64_t tiles = CeilDiv(m, cand) * CeilDiv(n, tiling->cvTaskN);
                    const uint32_t used = static_cast<uint32_t>(
                        tiles < static_cast<uint64_t>(aicNum) ? tiles : static_cast<uint64_t>(aicNum));
                    if (used != 0U) {
                        tiling->cvTaskM = cand;
                        tiling->blockDim = used;
                        tiling->usedCoreNum = used;
                        context->SetBlockDim(used);
                    }
                    break;
                }
            }
        }
    }
    // v-rag2: same guard on the N tile: a tn above 128 also loses the WP
    // bulk engine (nUse <= 128 gate); a sub-16 tail wastes a sliver task.
    if ((context->GetTilingKey() == 4U || context->GetTilingKey() == 5U) && k <= 256U && tiling->cvTaskN != 0U) {
        const uint32_t tailN = n % tiling->cvTaskN;
        if (tiling->cvTaskN > 128U || (tailN != 0U && tailN < 16U)) {
            const uint32_t startC =
                tiling->cvTaskN > 128U ? 128U : (tiling->cvTaskN > 16U ? tiling->cvTaskN - 1U : 16U);
            for (uint32_t cand = startC; cand >= 16U; --cand) {
                const uint32_t t2 = n % cand;
                if (t2 == 0U || t2 >= 16U) {
                    const uint64_t tiles = CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, cand);
                    const uint32_t used = static_cast<uint32_t>(
                        tiles < static_cast<uint64_t>(aicNum) ? tiles : static_cast<uint64_t>(aicNum));
                    if (used != 0U) {
                        tiling->cvTaskN = cand;
                        tiling->blockDim = used;
                        tiling->usedCoreNum = used;
                        context->SetBlockDim(used);
                    }
                    break;
                }
            }
        }
    }
    // v-kfix3: defect-zone guard for the stock K-unit addressing bug.
    // Shapes whose K is not a multiple of 64 but splits into an even number
    // of 64-wide K units produce corrupt results (or 507015) when a task
    // covers part of N. Reproduced on (6,96)/(6,112) and also on
    // (96,96)/(65,96) whose task M extent is only 16/24 rows: the earlier
    // mBlocks<=2 restriction missed them. Fetching the N column in one piece
    // (cvTaskN = min(n,128)) is bit-identical correct for every failing cell,
    // so apply it to the whole defect class. K % 64 == 0 and K <= 64 never
    // enter this branch; official-shape configs are unaffected.
    if (context->GetTilingKey() == 4U && k > 64U && (k % 64U) != 0U && ((k + 63U) / 64U) % 2U == 0U &&
        n > tiling->cvTaskN) {
        const uint32_t tnFix = n <= 128U ? static_cast<uint32_t>(n) : 128U;
        if (tnFix != tiling->cvTaskN) {
            const uint64_t tiles = CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tnFix);
            const uint32_t used =
                static_cast<uint32_t>(tiles < static_cast<uint64_t>(aicNum) ? tiles : static_cast<uint64_t>(aicNum));
            tiling->cvTaskN = tnFix;
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            context->SetBlockDim(used);
        }
    }
    // r103: aligned N split for the 16-task key-4 family. The half split
    // rounded UP to a 16-aligned width keeps the wide column block aligned
    // for Fixpipe/vector bursts; measured (512x193: 112+81 -0.51us and
    // 449x208: 112+96 -0.88us versus the unaligned balanced split; 512x240
    // 128+112 remains best). Gated to n>=176 where this was measured; the
    // smaller half keeps the r101 balanced split. One task per AIC.
    if (context->GetTilingKey() == 4U && k > 128U && k <= 256U && n > 128U && n <= 256U && tiling->cvTaskM <= 128U &&
        tiling->cvTaskM != 0U && tiling->cvTaskN != 0U &&
        CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tiling->cvTaskN) == 16ULL && CeilDiv(n, tiling->cvTaskN) == 2ULL) {
        const uint32_t half = static_cast<uint32_t>(CeilDiv(n, 2U));
        uint32_t tn = half;
        if (n >= 176U) {
            tn = ((half + 15U) / 16U) * 16U;
            if (tn > 128U) {
                tn = 128U;
            }
        }
        if (tn >= 16U && (n - tn) >= 16U && tn != tiling->cvTaskN) {
            const uint64_t tiles = CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tn);
            if (tiles == 16ULL) {
                const uint32_t used = static_cast<uint32_t>(tiles < aicNum ? tiles : aicNum);
                tiling->cvTaskN = tn;
                tiling->blockDim = used;
                tiling->usedCoreNum = used;
                context->SetBlockDim(used);
            }
        }
    }
    // r99: same one-task-per-AIC schedule as the r97 family, extended to
    // its aligned twin (16 tiles with taskN already 128). R96 measured
    // 512x256-class grids ~0.7..1.0us faster with 16 cores than with the
    // v154 two-tasks-per-AIC mapping.
    if (context->GetTilingKey() == 4U && k > 128U && k <= 256U && n > 128U && n <= 256U && tiling->cvTaskM != 0U &&
        tiling->cvTaskM <= 128U && tiling->cvTaskN == 128U && CeilDiv(m, tiling->cvTaskM) * 2ULL == 16ULL &&
        tiling->blockDim != 16U) {
        const uint32_t used = aicNum < 16U ? aicNum : 16U;
        tiling->blockDim = used;
        tiling->usedCoreNum = used;
        context->SetBlockDim(used);
    }
    // Generic short-K task merge (no test-case dimensions): when the auto
    // tiler has folded exactly two <=32x32 bulk tasks onto each AIC, each
    // task still pays a fixed prologue and two Fixpipe calls. Merge to one
    // task per AIC by widening N to 64 first, then M to 64 while the task
    // stays below 4096 elements. Bounded by the proven <=128 extents, the
    // UB chunk capacity, 16-aligned tails and >=4 active AICs.
    if (context->GetTilingKey() == 4U && k <= 256U && tiling->cvTaskM >= 16U && tiling->cvTaskM <= 32U &&
        tiling->cvTaskN >= 16U && tiling->cvTaskN <= 32U && tiling->blockDim != 0U) {
        const uint64_t tilesNow = CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tiling->cvTaskN);
        if (tilesNow == 2ULL * static_cast<uint64_t>(tiling->blockDim)) {
            uint32_t tm = tiling->cvTaskM;
            uint32_t tn = tiling->cvTaskN;
            const bool nTail = (n % 64U == 0U) || (n % 64U >= 16U);
            const bool mTail = (m % 64U == 0U) || (m % 64U >= 16U);
            if (n > tn && nTail) {
                tn = 64U;
            }
            if (static_cast<uint64_t>(tm) * tn < 4096ULL && m > tm && mTail) {
                tm = 64U;
            }
            const uint64_t tiles = CeilDiv(m, tm) * CeilDiv(n, tn);
            const uint32_t used =
                static_cast<uint32_t>(tiles < static_cast<uint64_t>(aicNum) ? tiles : static_cast<uint64_t>(aicNum));
            const uint32_t pitch = (tn + 15U) / 16U * 16U;
            if (tm <= 128U && tn <= 128U && used >= 4U && tiles == used &&
                static_cast<uint64_t>(tm) * pitch <= tiling->vecChunkElements && (m % tm == 0U || m % tm >= 16U) &&
                (n % tn == 0U || n % tn >= 16U)) {
                tiling->cvTaskM = tm;
                tiling->cvTaskN = tn;
                tiling->blockDim = used;
                tiling->usedCoreNum = used;
                context->SetBlockDim(used);
            }
        }
    }
    // v56: one 16-aligned row pitch for every key. Fixpipe writes nTake
    // columns at this stride, so the stride equals the write width and each
    // plane row is a contiguous 32B-multiple burst.
    // Mid-M wide-K family: favor N parallelism while retaining 128-row L0 tiles.
    // B3 sweep covered M=128,256,384 and K=768,1024; tail and UB guards retain fallback.
    if (context->GetTilingKey() == 4U && k >= 768U && k <= 1024U && m >= 128U && m <= 384U &&
        (m % 128U == 0U || m % 128U >= 16U) && (n % 64U == 0U || n % 64U >= 16U) &&
        tiling->vecChunkElements >= 128U * 64U) {
        const uint64_t tiles = CeilDiv(m, 128U) * CeilDiv(n, 64U);
        const uint32_t used = static_cast<uint32_t>(tiles < aicNum ? tiles : aicNum);
        tiling->cvTaskM = 128U;
        tiling->cvTaskN = 64U;
        tiling->blockDim = used;
        tiling->usedCoreNum = used;
        context->SetBlockDim(used);
    }
    // Profiling-backed 1:2 route: large short-K output grids benefit from
    // splitting Vector work; smaller grids retain the original 1:1 route.
    const uint64_t outputElements = static_cast<uint64_t>(m) * n;
    if (((outputElements >= 2097152ULL && k <= 256U) || (outputElements >= 1048576ULL && k == 1024U)) &&
        static_cast<uint64_t>(tiling->cvTaskM) * tiling->cvTaskN >= 8192ULL) {
        context->SetTilingKey(5);
    }
    // Full-K resident weights: 2*64*K half weights plus 128*K half input.
    // Use a multiple of N blocks so consecutive tasks retain their W column.
    if (k >= 512U && k <= 1024U && (k <= 512U || k % 256U != 0U) && k % 64U == 0U && n % 64U == 0U && m >= 512U &&
        (m % 128U == 0U || m % 128U >= 16U) && tiling->vecChunkElements >= 8192U) {
        uint64_t l1Bytes = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Bytes);
        const uint32_t nb = n / 64U;
        const uint32_t used = nb != 0U ? (aicNum / nb) * nb : 0U;
        if (used >= nb && 512ULL * k <= l1Bytes) {
            tiling->cvTaskM = 128U;
            tiling->cvTaskN = 64U;
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            tiling->reserved |= 512U;
            context->SetBlockDim(used);
            context->SetTilingKey(4U);
        }
    }
    // Tiny key9: three one-tile cores for short-K small rectangular outputs.
    if (context->GetTilingKey() == 4U && k <= 64U && m >= 32U && m <= 128U && n >= 17U && n <= 64U &&
        tiling->blockDim == 3U && tiling->usedCoreNum == 3U && tiling->cvTaskM >= 16U && tiling->cvTaskM <= 64U &&
        tiling->cvTaskN == n && CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tiling->cvTaskN) == tiling->blockDim &&
        ubSize >= SWIGLU_TINY_RAW_UB_BYTES + UB_RESERVED_BYTES) {
        context->SetTilingKey(9);
    }
    // Each core owns one <=64x64 tile, with complete-K branch accumulation.
    if (context->GetTilingKey() == 4U && k >= 65U && k <= 256U && m >= 17U && m <= 128U) {
        const uint32_t tn = 64U;
        uint32_t tm = m < 64U ? m : 64U;
        while (tm > 17U && m % tm != 0U && m % tm < 17U)
            --tm;
        const uint32_t cores = static_cast<uint32_t>(CeilDiv(m, tm) * CeilDiv(n, tn));
        uint64_t l1 = 0U, l0a = 0U, l0b = 0U, l0c = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        const uint64_t ma = CeilDiv(tm, 16U) * 16U;
        if (tm >= 17U && (m % tm == 0U || m % tm >= 17U) && cores <= aicNum && l1 >= 393216ULL &&
            l0a >= ma * 128ULL * 4ULL && l0b >= 64ULL * 128ULL * 4ULL && l0c >= 131072ULL &&
            ubSize >= 16ULL * 4096ULL + 3072ULL + 8192ULL) {
            tiling->cvTaskM = tm;
            tiling->cvTaskN = tn;
            tiling->blockDim = cores;
            tiling->usedCoreNum = cores;
            context->SetBlockDim(cores);
            context->SetTilingKey(31U);
        }
    }
    // Branch-overlap consumes W1/W2 sequentially, so only two L0B slots
    // are live. Select a wider K block only when both double buffers fit.
    if (context->GetTilingKey() == 4U && k > 64U && k <= 256U) {
        uint64_t l0aBytes = 0U;
        uint64_t l0bBytes = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aBytes);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bBytes);
        const uint32_t maxRows = m < tiling->cvTaskM ? m : tiling->cvTaskM;
        const uint32_t maxCols = n < tiling->cvTaskN ? n : tiling->cvTaskN;
        const uint64_t alignedRows = CeilDiv(maxRows, 16U) * 16U;
        const uint64_t alignedCols = CeilDiv(maxCols, 16U) * 16U;
        constexpr uint32_t branchKBlock = 128U;
        const uint64_t aBytes = 2ULL * alignedRows * branchKBlock * sizeof(uint16_t);
        const uint64_t bBytes = 2ULL * alignedCols * branchKBlock * sizeof(uint16_t);
        if (maxRows <= 128U && maxCols <= 128U && aBytes <= l0aBytes && bBytes <= l0bBytes) {
            tiling->reserved = branchKBlock;
            // WP keeps both weight branches double-buffered: four B slots.
            if (2ULL * bBytes <= l0bBytes) {
                tiling->reserved |= 256U;
            }
        }
    }
    // Two rounds for large-M short-K tasks when a full single-batch epilogue fits.
    // Actual allocation: 2 FP32 arrays + 2 one-slot FP16 inputs + 1 FP16 output
    // = 14 bytes/element, plus two aligned FP32 bias vectors and 4KB reserve.
    if (context->GetTilingKey() == 4U && k == 64U && n == 64U && m >= 3585U && m <= 8192U && aicNum > 0U) {
        uint32_t tm = static_cast<uint32_t>(CeilDiv(m, 2ULL * aicNum));
        while (tm < 128U && m % tm != 0U && m % tm < 16U)
            ++tm;
        const uint32_t pitch = static_cast<uint32_t>(CeilDiv(n, 16U) * 16U);
        const uint64_t chunk = static_cast<uint64_t>(tm) * pitch;
        const uint64_t tasks = CeilDiv(m, tm);
        const uint64_t biasBytes = CeilDiv(n, 8U) * 8ULL * 8ULL;
        if (tm >= 16U && tm <= 112U && (m % tm == 0U || m % tm >= 16U) && tasks <= 2ULL * aicNum &&
            chunk * 14ULL + biasBytes + 4096ULL <= ubSize) {
            const uint32_t used = static_cast<uint32_t>(tasks < aicNum ? tasks : aicNum);
            tiling->cvTaskM = tm;
            tiling->cvTaskN = n;
            tiling->vecChunkElements = static_cast<uint32_t>(chunk);
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            context->SetBlockDim(used);
        }
    }
    if ((context->GetTilingKey() == 4U || context->GetTilingKey() == 5U) && k >= 65U && k <= 1024U && m <= 16U) {
        const uint32_t tn = 64U, used = static_cast<uint32_t>(CeilDiv(n, tn));
        const uint64_t ma = CeilDiv(m, 16U) * 16ULL, kal = CeilDiv(k, 16U) * 16ULL;
        uint64_t l1 = 0U, la = 0U, lb = 0U, lc = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, la);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, lb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, lc);
        if (used <= static_cast<uint32_t>(platform.GetCoreNumAic()) && (128ULL + ma) * kal * 2ULL <= l1 &&
            ma * 128ULL * 4ULL <= la && 64ULL * 128ULL * 8ULL <= lb && 2ULL * ma * 64ULL * 4ULL <= lc &&
            ubSize >= 16ULL * 4096ULL + 3072ULL + 8192ULL) {
            tiling->cvTaskM = m;
            tiling->cvTaskN = tn;
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            context->SetBlockDim(used);
            context->SetTilingKey(35U);
        }
    }
    if (context->GetTilingKey() == 4U && m >= 1024U && k >= 65U &&
        ((k <= 256U && static_cast<uint64_t>(m) * n >= 1048576ULL) ||
         (k > 256U && static_cast<uint64_t>(m) * n >= 524288ULL)) &&
        tiling->cvTaskM >= 32U && tiling->cvTaskM <= 128U && tiling->cvTaskN <= 128U &&
        static_cast<uint64_t>(tiling->cvTaskM) * tiling->cvTaskN >= 4096ULL) {
        context->SetTilingKey(40U);
    }
    if (context->GetTilingKey() == 4U && k >= 65U && k <= 256U && m >= 129U) {
        const uint32_t tn = 128U;
        uint32_t tm = 64U;
        while (tm > 17U && m % tm != 0U && m % tm < 17U)
            --tm;
        const uint64_t tasks = CeilDiv(m, tm) * CeilDiv(n, tn);
        const uint32_t av = static_cast<uint32_t>(platform.GetCoreNumAic());
        uint64_t l1 = 0U, la = 0U, lb = 0U, lc = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, la);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, lb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, lc);
        const uint64_t ma = CeilDiv(tm, 16U) * 16ULL;
        if (tasks <= av && (m % tm == 0U || m % tm >= 17U) && l1 >= 393216ULL && la >= ma * 128ULL * 4ULL &&
            lb >= 65536ULL && lc >= 131072ULL && ubSize >= 16ULL * 8192ULL + 3072ULL + 8192ULL) {
            tiling->cvTaskM = tm;
            tiling->cvTaskN = tn;
            tiling->blockDim = static_cast<uint32_t>(tasks);
            tiling->usedCoreNum = tiling->blockDim;
            context->SetBlockDim(tiling->blockDim);
            context->SetTilingKey(43U);
        }
    }
    if ((context->GetTilingKey() == 4U || context->GetTilingKey() == 5U || context->GetTilingKey() == 40U) &&
        k >= 65U && k <= 128U && k % 16U == 0U && m >= 1024U) {
        const uint32_t tn = n;
        const uint32_t av = static_cast<uint32_t>(platform.GetCoreNumAic());
        uint32_t tm = av > 0U ? static_cast<uint32_t>(CeilDiv(CeilDiv(m, av), 32U) * 32U) : 32U;
        uint64_t l1 = 0U, la = 0U, lb = 0U, lc = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, la);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, lb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, lc);
        uint64_t maxRows = la / (2ULL * k);
        if (maxRows > lc / (4ULL * tn))
            maxRows = lc / (4ULL * tn);
        const uint64_t vectorRows = 2ULL * ((16384ULL / tn) / 16ULL) * 16ULL;
        if (maxRows > vectorRows)
            maxRows = vectorRows;
        maxRows = maxRows / 32ULL * 32ULL;
        while (tm <= maxRows && m % tm != 0U && m % tm < 17U)
            tm += 32U;
        const uint64_t tasks = CeilDiv(m, tm);
        if (av > 0U && tm >= 32U && tm <= maxRows && tasks <= av && (m % tm == 0U || m % tm >= 17U) &&
            l1 >= 2ULL * (tm + 2ULL * tn) * k && la >= 2ULL * tm * k && lb >= 4ULL * tn * k && lc >= 4ULL * tm * tn &&
            ubSize >= 10ULL * 16384ULL + 3072ULL + 8192ULL) {
            tiling->cvTaskM = tm;
            tiling->cvTaskN = tn;
            tiling->blockDim = static_cast<uint32_t>(tasks);
            tiling->usedCoreNum = tiling->blockDim;
            context->SetBlockDim(tiling->blockDim);
            context->SetTilingKey(52U);
        }
    }
    // Medium/large output grids: full-K combined weights with two input slots.
    if ((context->GetTilingKey() == 4U || context->GetTilingKey() == 5U || context->GetTilingKey() == 40U) &&
        m >= 512U && k >= 512U && k <= 640U && k % 16U == 0U && tiling->cvTaskM <= 128U && tiling->cvTaskN <= 128U) {
        uint64_t l1 = 0U, la = 0U, lb = 0U, lc = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, la);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, lb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, lc);
        const uint64_t ma = CeilDiv(m < tiling->cvTaskM ? m : tiling->cvTaskM, 16U) * 16U;
        const uint64_t na = CeilDiv(n < tiling->cvTaskN ? n : tiling->cvTaskN, 16U) * 16U;
        const uint64_t bk = na <= 64ULL ? 128ULL : 64ULL;
        if (ma * na >= 2048ULL && (2ULL * na + 2ULL * ma) * k * 2ULL <= l1 && 2ULL * ma * bk * 2ULL <= la &&
            4ULL * na * bk * 2ULL <= lb && 2ULL * ma * na * 4ULL <= lc) {
            tiling->reserved |= 1024U;
            if (bk == 128ULL)
                tiling->reserved |= 2048U;
        }
    }
    if ((context->GetTilingKey() == 4U || context->GetTilingKey() == 5U || context->GetTilingKey() == 40U ||
         context->GetTilingKey() == 43U) &&
        m >= 512U && k >= 129U && k <= 256U && k % 16U != 0U) {
        const uint32_t tn = 64U, nb = static_cast<uint32_t>(CeilDiv(n, tn));
        const uint32_t available = static_cast<uint32_t>(platform.GetCoreNumAic());
        const uint32_t groups = nb ? available / nb : 0U;
        if (groups > 0U) {
            const uint32_t tm = static_cast<uint32_t>(CeilDiv(m, groups));
            const uint64_t ma = CeilDiv(tm, 16U) * 16ULL, ka = CeilDiv(k, 16U) * 16ULL;
            const uint64_t tasks = CeilDiv(m, tm) * nb;
            uint64_t l1 = 0U, la = 0U, lb = 0U, lc = 0U;
            platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
            platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, la);
            platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, lb);
            platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, lc);
            if (tm >= 17U && tm <= 128U && (m % tm == 0U || m % tm >= 17U) && tasks <= available && l1 >= 393216ULL &&
                ma * ka * 2ULL <= la && 4ULL * tn * ka <= lb && lc >= 131072ULL &&
                static_cast<uint64_t>(tm) * tn <= 8192ULL && ubSize >= 16ULL * 8192ULL + 3072ULL + 8192ULL) {
                tiling->cvTaskM = tm;
                tiling->cvTaskN = tn;
                tiling->blockDim = static_cast<uint32_t>(tasks);
                tiling->usedCoreNum = tiling->blockDim;
                context->SetBlockDim(tiling->blockDim);
                context->SetTilingKey(65U);
            }
        }
    }
    // Only replace an underfilled second/last 128-column N subgroup.
    // Complete 128-column groups retain the original route and task grid.
    const uint32_t oldNTail = tiling->cvTaskN % 128U;
    if ((context->GetTilingKey() == 4U || context->GetTilingKey() == 5U || context->GetTilingKey() == 40U) &&
        m >= 1024U && k >= 768U && k <= 1536U && k % 128U == 0U && tiling->cvTaskN > 128U && oldNTail > 0U &&
        oldNTail <= 64U && (tiling->reserved & (512U | 1024U)) == 0U) {
        uint64_t l1 = 0U, la = 0U, lb = 0U, lc = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, la);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, lb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, lc);
        const uint64_t tiles = CeilDiv(m, 64U) * CeilDiv(n, 128U);
        const uint32_t available = static_cast<uint32_t>(platform.GetCoreNumAic());
        const uint32_t used = tiles < available ? static_cast<uint32_t>(tiles) : available;
        if (used > 0U && tiles >= 3ULL * used && 256ULL * k + 131072ULL <= l1 && la >= 65536ULL && lb >= 65536ULL &&
            lc >= 65536ULL) {
            const uint64_t oldKey = context->GetTilingKey();
            tiling->cvTaskM = 64U;
            tiling->cvTaskN = 128U;
            tiling->blockDim = used;
            tiling->usedCoreNum = used;
            tiling->reserved |= 16777216U;
            context->SetBlockDim(used);
            context->SetTilingKey(oldKey == 4U ? 66U : (oldKey == 5U ? 67U : 68U));
        }
    }
    // Generic long-K: eliminate internal N128+small-tail subgroups.
    // Pick an M-grid making N128 task count an exact number of AIC waves.
    // Admit only when both worst-core padded area and MMAD command bound fall.
    if (context->GetTilingKey() == 4U && k >= 512U && tiling->reserved == 0U && m > 128U && tiling->cvTaskM > 128U &&
        tiling->cvTaskN > 128U && tiling->cvTaskN % 128U != 0U) {
        const uint32_t available = static_cast<uint32_t>(platform.GetCoreNumAic());
        const uint64_t nb = CeilDiv(n, 128U);
        uint64_t ga = available, gb = nb;
        while (gb != 0U) {
            const uint64_t r = ga % gb;
            ga = gb;
            gb = r;
        }
        if (available > 0U && ga > 0U) {
            const uint64_t bands = available / ga;
            const uint64_t tm = CeilDiv(CeilDiv(m, bands), 16U) * 16U;
            const uint64_t actualBands = CeilDiv(m, tm), tasks = actualBands * nb;
            const uint64_t oldTasks = CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tiling->cvTaskN);
            const uint64_t oldRounds = CeilDiv(oldTasks, tiling->blockDim);
            const uint64_t oldM = m < tiling->cvTaskM ? m : tiling->cvTaskM;
            const uint64_t oldN = n < tiling->cvTaskN ? n : tiling->cvTaskN;
            const uint64_t newM = m < tm ? m : tm, newN = n < 128U ? n : 128U;
            const uint64_t rounds = CeilDiv(tasks, available);
            const uint64_t oldCommands = oldRounds * CeilDiv(oldM, 128U) * CeilDiv(oldN, 128U);
            const uint64_t commands = rounds * CeilDiv(newM, 128U);
            const uint64_t oldArea = oldRounds * CeilDiv(oldM, 16U) * 16U * CeilDiv(oldN, 16U) * 16U;
            const uint64_t area = rounds * CeilDiv(newM, 16U) * 16U * CeilDiv(newN, 16U) * 16U;
            if (actualBands == bands && tasks % available == 0U && tm >= 128U && tm <= 0xFFFFFFFFULL &&
                commands < oldCommands && area <= oldArea) {
                tiling->cvTaskM = static_cast<uint32_t>(tm);
                tiling->cvTaskN = 128U;
                tiling->blockDim = available;
                tiling->usedCoreNum = available;
                context->SetBlockDim(available);
                context->SetTilingKey(185U);
            }
        }
    }
    // Extend the existing full-K one-wave primitives beyond K256 only.
    // Select a balanced resource-feasible grid, not a sampled shape gate.
    const uint64_t oneWaveSourceKey = context->GetTilingKey();
    if ((oneWaveSourceKey == 4U || oneWaveSourceKey == 5U || oneWaveSourceKey == 40U) && k > 256U &&
        tiling->reserved == 0U) {
        const uint64_t available = platform.GetCoreNumAic(), ka = CeilDiv(k, 16U) * 16U;
        uint64_t la = 0U, lb = 0U, lc = 0U, l1 = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, la);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, lb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, lc);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        la = la < 65536U ? la : 65536U;
        lb = lb < 65536U ? lb : 65536U;
        const uint64_t oldM = m < tiling->cvTaskM ? m : tiling->cvTaskM;
        const uint64_t oldN = n < tiling->cvTaskN ? n : tiling->cvTaskN;
        const uint64_t oldMa = CeilDiv(oldM, 16U) * 16U, oldNa = CeilDiv(oldN, 16U) * 16U;
        const uint64_t oldTasks = CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tiling->cvTaskN);
        const uint64_t oldRounds = CeilDiv(oldTasks, tiling->blockDim);
        const uint64_t oldArea = oldRounds * oldMa * oldNa;
        const uint64_t oldReads = oldRounds * (oldMa * CeilDiv(oldN, 128U) + 2ULL * CeilDiv(oldM, 128U) * oldNa) * ka;
        const uint64_t oldMmas = oldRounds * CeilDiv(oldM, 128U) * CeilDiv(oldN, 128U) * 2ULL * CeilDiv(k, 64U);
        uint64_t bestM = 0U, bestN = 0U, bestTasks = 0U, bestMode = 0U, bestCost[6] = {0U};
        if (available > 0U && ka / 16U <= 255U && static_cast<uint64_t>(m) * n <= available * 8192U) {
            for (uint32_t mode = 0U; mode < 2U; ++mode) {
                const uint64_t e = mode == 0U ? 4096U : 8192U;
                if (ubSize < 16U * e + 3072U + UB_RESERVED_BYTES)
                    continue;
                if (mode == 1U && (l1 < 393216U || lc < 131072U))
                    continue;
                for (uint64_t tn = 16U; tn <= 128U; tn += 16U) {
                    if (mode == 0U && tn > 64U)
                        continue;
                    const uint64_t nb = CeilDiv(n, tn);
                    if (nb > available)
                        continue;
                    const uint64_t rowGroups = available / nb;
                    const uint64_t tm = CeilDiv(CeilDiv(m, rowGroups), 16U) * 16U;
                    const uint64_t tasks = CeilDiv(m, tm) * nb;
                    if (tm > 128U || tasks > available || tm * tn > e || tm * ka * 2ULL > la)
                        continue;
                    if ((mode == 0U ? 4ULL : 2ULL) * tn * ka > lb)
                        continue;
                    if (mode == 0U && ((tm + 2ULL * tn) * ka * 2ULL > l1 || 8ULL * tm * tn > lc))
                        continue;
                    const uint64_t area = tm * tn, reads = (tm + 2ULL * tn) * ka, mmas = mode == 0U ? 1U : 2U;
                    if (area > oldArea || reads > oldReads || mmas >= oldMmas)
                        continue;
                    const uint64_t mr = m < tm ? m : tm;
                    const uint64_t stores = mr * CeilDiv(2U * tn + 30U, 32U);
                    const uint64_t biasRepeats = mr * CeilDiv(tn, 64U);
                    const uint64_t totalReads = static_cast<uint64_t>(m) * k * nb + 2ULL * n * k * CeilDiv(m, tm);
                    const uint64_t cost[6] = {area, reads, stores, biasRepeats, totalReads, mmas};
                    bool better = bestM == 0U;
                    if (!better) {
                        for (uint32_t j = 0U; j < 6U; ++j) {
                            if (cost[j] != bestCost[j]) {
                                better = cost[j] < bestCost[j];
                                break;
                            }
                        }
                    }
                    if (better) {
                        bestM = tm;
                        bestN = tn;
                        bestTasks = tasks;
                        bestMode = mode;
                        for (uint32_t j = 0U; j < 6U; ++j)
                            bestCost[j] = cost[j];
                    }
                }
            }
        }
        if (bestM != 0U) {
            tiling->cvTaskM = static_cast<uint32_t>(bestM);
            tiling->cvTaskN = static_cast<uint32_t>(bestN);
            tiling->blockDim = static_cast<uint32_t>(bestTasks);
            tiling->usedCoreNum = tiling->blockDim;
            tiling->vecChunkElements = bestMode == 0U ? 4096U : 8192U;
            tiling->reserved = 0U;
            context->SetBlockDim(tiling->blockDim);
            context->SetTilingKey(bestMode == 0U ? 190U : 189U);
        }
    }
    const uint64_t workspacePitch = CeilDiv(tiling->cvTaskN, 16U) * 16U;
    const uint64_t matrixElements =
        CeilDiv(m, tiling->cvTaskM) * CeilDiv(n, tiling->cvTaskN) * tiling->cvTaskM * workspacePitch;
    // 与 kernel 侧的 planeElements_ 公式必须逐字一致
    const uint64_t alignedPlane = (matrixElements + PLANE_STRIDE_ALIGN_ELEMENTS - 1ULL) / PLANE_STRIDE_ALIGN_ELEMENTS *
                                  PLANE_STRIDE_ALIGN_ELEMENTS;
    const uint64_t planeElements = alignedPlane + PLANE_GUARD_ELEMENTS;
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t libWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
    // Two unbiased half planes enforce dot rounding BEFORE FP32 bias addition.
    // v96: 只剩两个结果平面（key 10 的 streamScratch_ 已随该类删除）。
    const uint64_t userWorkspaceBytes = planeElements * 2ULL * sizeof(uint16_t);
    if (libWorkspaceBytes > SIZE_MAX - userWorkspaceBytes) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = static_cast<size_t>(libWorkspaceBytes + userWorkspaceBytes);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (xShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SwiGlu : public OpDef {
  public:
    explicit SwiGlu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight_1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias_1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SwiGlu);
} // namespace ops
