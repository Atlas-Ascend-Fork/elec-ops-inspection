/*
 * Copyright (c) 2026 dihidao.
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
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"
#include <algorithm>
#include <iterator>
#include <cstddef>
#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"
namespace optiling {
constexpr uint32_t LDET_SMALL_N_MAX = 208;
constexpr uint32_t LDET_MAX_N = 512;
constexpr uint32_t LDET_ALIGN = 8;
static void FillBlockFourMaps(LinalgDetTilingDataFull *td) {
    std::fill(std::begin(td->block4Offsets), std::end(td->block4Offsets), 0);
    std::fill(std::begin(td->block4Signs), std::end(td->block4Signs), 0);
    constexpr uint32_t pivot[4] = {0U, 4U, 164U, 160U};
    for (uint32_t lane = 0U; lane < 32U; ++lane)
        td->block4Offsets[lane] = pivot[lane / 8U];
    for (uint32_t phase = 0U; phase < 6U; ++phase)
        for (uint32_t lane = 0U; lane < 32U; ++lane) {
            uint32_t live = 14U - phase * 2U, r = lane % 16U, k = lane / 16U;
            if (r >= live)
                r = 0U;
            td->block4Offsets[32U + phase * 32U + lane] = ((r + 2U) * 40U + k) * 4U;
        }
    constexpr uint32_t ii[6] = {0U, 0U, 0U, 1U, 1U, 2U};
    constexpr uint32_t jj[6] = {1U, 2U, 3U, 2U, 3U, 3U};
    for (uint32_t j = 0U; j < 6U; ++j) {
        uint32_t a = ii[j], b = jj[j];
        if (j == 1U || j == 4U) {
            uint32_t t = a;
            a = b;
            b = t;
        }
        const uint32_t c = ii[5U - j], d = jj[5U - j];
        const uint32_t av[4] = {a, b, 80U + c, 80U + d};
        const uint32_t bv[4] = {40U + b, 40U + a, 120U + d, 120U + c};
        for (uint32_t group = 0U; group < 4U; ++group) {
            td->block4Offsets[224U + group * 8U + j] = av[group] * 4U;
            td->block4Offsets[256U + group * 8U + j] = bv[group] * 4U;
        }
    }
}

template <class PhaseArg0, class PhaseArg1, class PhaseArg2, class PhaseArg3, class PhaseArg4, class PhaseArg5>
static inline uint32_t SelectKernelRouteStatePhase1(PhaseArg0 &aligned, PhaseArg1 &group, PhaseArg2 &isLarge,
                                                    PhaseArg3 &n, PhaseArg4 &raggedMedium, PhaseArg5 &raggedSmall) {

    if (n == 2U || n == 3U)
        return 7U;
    if (raggedSmall)
        return 6U;
    if (raggedMedium)
        return 5U;
    if (n >= 4U && n <= 8U)
        return 4U;
    if (n >= 16U && n <= 64U && aligned)
        return 3U;
    if (isLarge == 2U && n <= 256U && group >= 4U)
        return 16U;
    return n >= 16U && n <= 208U && aligned ? 1U : 0U;
}
template <class PhaseArg0, class PhaseArg1, class PhaseArg2, class PhaseArg3, class PhaseArg4, class PhaseArg5,
          class PhaseArg6, class PhaseArg7>
static inline uint32_t SelectKernelRouteStatePhase2(PhaseArg0 &aligned, PhaseArg1 &batch, PhaseArg2 &group,
                                                    PhaseArg3 &isLarge, PhaseArg4 &matPerCore, PhaseArg5 &n,
                                                    PhaseArg6 &raggedMedium, PhaseArg7 &raggedSmall) {

    if (n == 128U)
        return 17U;
    if (n >= 152U && n <= 208U && aligned)
        return 15U;
    if (raggedSmall && matPerCore > 1U)
        return 14U;
    if (n == 8U && (batch == 1U || batch >= 128U))
        return 13U;
    if (n == 7U)
        return 12U;
    if (n == 64U)
        return 11U;
    if (n == 32U)
        return 10U;
    if (n == 16U)
        return 9U;
    if (n == 8U && batch > 1U)
        return 8U;
    return SelectKernelRouteStatePhase1(aligned, group, isLarge, n, raggedMedium, raggedSmall);
}
template <class PhaseArg0, class PhaseArg1, class PhaseArg2, class PhaseArg3, class PhaseArg4, class PhaseArg5,
          class PhaseArg6>
static inline uint32_t SelectKernelRouteStatePhase3(PhaseArg0 &batch, PhaseArg1 &cube32, PhaseArg2 &cubeRange,
                                                    PhaseArg3 &group, PhaseArg4 &isLarge, PhaseArg5 &matPerCore,
                                                    PhaseArg6 &n) {

    if (cubeRange) {
        const bool keepOriginalTail = n == 231U || n == 242U || n == 243U || n == 244U || n == 250U;
        return keepOriginalTail ? 21U : 22U;
    }
    if (cube32)
        return 20U;
    const bool aligned = n % LDET_ALIGN == 0;
    const bool raggedSmall = n >= 9U && n <= 64U && !aligned;
    const bool raggedMedium = n >= 65U && n <= 208U && !aligned;
    if (raggedSmall && matPerCore >= 3U)
        return 19U;
    if (raggedMedium && ((n + 7U) & ~7U) == 128U)
        return 18U;
    return SelectKernelRouteStatePhase2(aligned, batch, group, isLarge, matPerCore, n, raggedMedium, raggedSmall);
}
static uint32_t SelectKernelRoute(uint32_t n, uint32_t batch, uint32_t matPerCore, uint32_t isLarge, uint32_t group,
                                  bool cube32, bool cubeRange, bool ub216) {
    if (n == 8U && (batch == 128U || batch == 129U))
        return 25U;
    if (ub216)
        return 23U;
    return SelectKernelRouteStatePhase3(batch, cube32, cubeRange, group, isLarge, matPerCore, n);
}
static bool ColdCoverageFamily(uint32_t n, uint32_t b) {
    return (n == 8U && b == 1U) || (n == 32U && b == 1U) || (n == 208U && b == 1U);
}
static uint32_t ColdCoverageAlias(uint32_t route) {
    if (route == 13U)
        return 64U;
    if (route == 33U)
        return 65U;
    if (route == 58U)
        return 85U;
    return route;
}

static ge::graphStatus SelectColdCoverage(gert::TilingContext *context, uint32_t dtype, uint32_t selectedRoute,
                                          platform_ascendc::PlatformAscendC &platform, uint32_t n, uint32_t batch) {
    const uint32_t blocks = context->GetBlockDim();

    if (ColdCoverageFamily(n, batch) && blocks > 0U && blocks <= static_cast<uint32_t>(platform.GetCoreNumAic())) {
        const uint32_t clone = ColdCoverageAlias(selectedRoute);
        if (clone != selectedRoute) {
            auto *ws = context->GetWorkspaceSizes(1);
            if (ws == nullptr)
                return ge::GRAPH_FAILED;
            ws[0] =
                std::max<size_t>(ws[0], std::max<size_t>(platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) + 32U);
            if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS)
                return ge::GRAPH_FAILED;
            selectedRoute = clone;
        }
    }
    ASCENDC_TPL_SEL_PARAM(context, dtype, selectedRoute);
    return ge::GRAPH_SUCCESS;
}
struct DetTilingInputs {
    gert::TilingContext *context;
    platform_ascendc::PlatformAscendC platform;
    int32_t numCoresAiv;
    const gert::Tensor *tensorA;
    int64_t n;
    uint32_t batch;
    explicit DetTilingInputs(gert::TilingContext *ctx) : context(ctx), platform(ctx->GetPlatformInfo()) {}
};
struct DetFullPlan {
    LinalgDetTilingDataFull *tiling = nullptr;
    size_t *workspace = nullptr;
};
static bool PrepareFullTiling(DetTilingInputs &s, DetFullPlan &plan, uint32_t n, uint32_t stride) {
    plan.tiling = s.context->template GetTilingData<LinalgDetTilingDataFull>();
    plan.workspace = s.context->GetWorkspaceSizes(1);
    if (plan.tiling == nullptr || plan.workspace == nullptr)
        return false;
    *plan.tiling = LinalgDetTilingDataFull{};
    plan.tiling->batchCount = s.batch;
    plan.tiling->n = n;
    plan.tiling->stride = stride;
    return true;
}

static bool ConfigureCyclicGroup(DetTilingInputs &s, LinalgDetTilingDataFull &td, uint32_t group, uint32_t blockDim) {
    td.matPerCore = 1U;
    td.isLarge = 2U;
    td.groupSize = group;
    s.context->SetBlockDim(blockDim);
    return s.context->SetScheduleMode(1U) == ge::GRAPH_SUCCESS;
}

static bool PrepareCyclicPlan47(DetTilingInputs &s, DetFullPlan &plan, uint32_t n, uint32_t stride, uint32_t group,
                                uint32_t blockDim) {
    if (!PrepareFullTiling(s, plan, n, stride))
        return false;
    return ConfigureCyclicGroup(s, *plan.tiling, group, blockDim);
}
struct DetDefaultPlan {
    uint32_t stride;
    uint32_t isLarge;
    uint32_t matPerCore;
    uint32_t blockDim;
    uint32_t group;
    bool wide512;
    bool cube32;
    uint32_t cubeGroups;
    uint32_t cubePerGroup;
    bool cubeRange;
    bool ub216;
    LinalgDetTilingDataFull *tiling;
    static constexpr uint32_t kSysWsReserve = 16U * 1024U * 1024U;
    uint32_t libApiWs;
    size_t sysWs;
    size_t *currentWorkspace;
    size_t wsPerMat;
    size_t userWs;
    uint32_t DT_A;
    uint32_t LD_BATCH;
};
static bool ReadTilingInputs(DetTilingInputs &s) {

    s.numCoresAiv = std::min<int32_t>(s.platform.GetCoreNumAiv(), 40);
    if (s.numCoresAiv <= 0) {
        return false;
    }
    s.tensorA = s.context->GetRequiredInputTensor(0);
    if (s.tensorA == nullptr) {
        return false;
    }
    int64_t totalElems = s.tensorA->GetShapeSize();
    if (totalElems <= 0) {
        return false;
    }
    const gert::StorageShape *storageShape = s.context->GetInputShape(0);
    if (storageShape == nullptr) {
        return false;
    }
    const gert::Shape &shapeA = storageShape->GetStorageShape();
    size_t rank = shapeA.GetDimNum();
    if (rank < 2) {
        return false;
    }
    s.n = shapeA.GetDim(rank - 1);
    if (s.n < 1 || s.n > static_cast<int64_t>(LDET_MAX_N) || shapeA.GetDim(rank - 2) != s.n ||
        totalElems % (s.n * s.n) != 0) {
        return false;
    }
    s.batch = static_cast<uint32_t>(totalElems / (s.n * s.n));
    if (s.batch == 0U)
        return false;

    return true;
}
static bool TryRagged208(DetTilingInputs &s, ge::graphStatus &result) {
    if (s.n >= 193 && s.n <= 207 && s.n != 200 && s.batch <= 12U && s.tensorA->GetDataType() == ge::DT_FLOAT) {
        const uint32_t available = std::min<uint32_t>(static_cast<uint32_t>(s.numCoresAiv), 40U);
        if (available >= 3U * s.batch) {
            const uint32_t g = available >= 7U * s.batch ? 7U : (available >= 4U * s.batch ? 4U : 3U);
            DetFullPlan plan;
            if (!PrepareCyclicPlan47(s, plan, static_cast<uint32_t>(s.n), 208U, g, g * s.batch)) {
                result = ge::GRAPH_FAILED;
                return true;
            }
            auto *td = plan.tiling;
            size_t *ws = plan.workspace;
            ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) +
                    static_cast<size_t>(s.batch) * (32U * 208U + 72U) * 8U;
            uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType()), LD_BATCH = 56U;
            {
                result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
                return true;
            }
        }
    }
    return false;
}

static bool Try208SmallBatch(DetTilingInputs &s, ge::graphStatus &result) {
    if (s.n == 208 && s.batch <= 12U && s.tensorA->GetDataType() == ge::DT_FLOAT &&
        static_cast<uint32_t>(s.numCoresAiv) >= 3U * s.batch) {
        DetFullPlan plan;
        if (!PrepareCyclicPlan47(s, plan, 208U, 208U, 3U, 3U * s.batch)) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) +
                static_cast<size_t>(s.batch) * (32U * 208U + 72U) * 8U;
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType()), LD_BATCH = 58U;
        {
            result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
            return true;
        }
    }
    return false;
}

static bool Try208MediumBatch(DetTilingInputs &s, ge::graphStatus &result) {
    if (s.n == 208 && s.batch <= 16U && s.tensorA->GetDataType() == ge::DT_FLOAT &&
        static_cast<uint32_t>(s.numCoresAiv) >= 3U * s.batch) {
        const uint32_t g =
            static_cast<uint32_t>(s.numCoresAiv) >= 7U * s.batch
                ? 7U
                : ((s.batch == 7U || s.batch == 8U) && static_cast<uint32_t>(s.numCoresAiv) >= 6U * s.batch
                       ? 6U
                       : (static_cast<uint32_t>(s.numCoresAiv) >= 4U * s.batch ? 4U : 3U));
        DetFullPlan plan;
        if (!PrepareCyclicPlan47(s, plan, 208U, 208U, g, g * s.batch)) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        const size_t sysWs = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U);
        ws[0] = sysWs + static_cast<size_t>(s.batch) * (32U * 208U + 72U) * 8U;
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType());
        uint32_t LD_BATCH = g == 7U ? (s.batch == 4U ? 39U : 38U) : (g == 6U ? 38U : (g == 4U ? 35U : 41U));
        {
            result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
            return true;
        }
    }
    return false;
}

static bool Try200WideGroup(DetTilingInputs &s, ge::graphStatus &result) {
    if (s.n == 200 && s.batch <= 20U && s.tensorA->GetDataType() == ge::DT_FLOAT &&
        static_cast<uint32_t>(s.numCoresAiv) >= 3U * s.batch) {
        DetFullPlan plan;
        if (!PrepareCyclicPlan47(s, plan, 200U, 200U, 3U, 3U * s.batch)) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) +
                static_cast<size_t>(s.batch) * (32U * 200U + 72U) * 8U;
        const bool splitOwners = s.batch == 3U && static_cast<uint32_t>(s.numCoresAiv) >= 18U;
        if (splitOwners) {
            td->groupSize = 6U;
            s.context->SetBlockDim(18U);
            ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) + 1048576U;
        }
        const bool tracePipeline =
            splitOwners && static_cast<uint32_t>(s.numCoresAiv) >= 24U && s.platform.GetCoreNumAic() >= 12U;
        if (tracePipeline) {
            s.context->SetBlockDim(12U);
            ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) + 4194304U;
        }
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType()),
                 LD_BATCH = (tracePipeline ? 108U : (splitOwners ? 63U : 60U));
        {
            result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
            return true;
        }
    }
    return false;
}

static bool Try200PairGroup(DetTilingInputs &s, ge::graphStatus &result) {
    if (s.n == 200 && s.batch >= 13U && s.batch <= 20U && s.tensorA->GetDataType() == ge::DT_FLOAT &&
        static_cast<uint32_t>(s.numCoresAiv) >= 2U * s.batch) {
        const uint32_t g = static_cast<uint32_t>(s.numCoresAiv) >= 3U * s.batch ? 3U : 2U;
        if (g == 3U || s.platform.GetCoreNumAic() >= s.batch) {
            DetFullPlan plan;
            if (!PrepareCyclicPlan47(s, plan, 200U, 200U, g, g == 2U ? s.batch : g * s.batch)) {
                result = ge::GRAPH_FAILED;
                return true;
            }
            auto *td = plan.tiling;
            size_t *ws = plan.workspace;
            ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) +
                    static_cast<size_t>(s.batch) * (32U * 200U + 72U) * 8U;
            uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType()), LD_BATCH = g == 2U ? 59U : 58U;
            {
                result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
                return true;
            }
        }
    }
    return false;
}

static bool Try192Or200Triple(DetTilingInputs &s, ge::graphStatus &result) {
    if (((s.n == 192 && (s.batch == 13U || (s.batch == 8U && static_cast<uint32_t>(s.numCoresAiv) < 6U * s.batch))) ||
         (s.n == 200 && s.batch >= 1U && s.batch <= 12U)) &&
        s.tensorA->GetDataType() == ge::DT_FLOAT && static_cast<uint32_t>(s.numCoresAiv) >= 3U * s.batch) {
        DetFullPlan plan;
        if (!PrepareCyclicPlan47(s, plan, static_cast<uint32_t>(s.n), static_cast<uint32_t>(s.n), 3U, 3U * s.batch)) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) +
                static_cast<size_t>(s.batch) * (32U * static_cast<uint32_t>(s.n) + 72U) * 8U;
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType()), LD_BATCH = 58U;
        {
            result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
            return true;
        }
    }
    return false;
}

static bool Is192Or200PairCandidate(DetTilingInputs &s) {
    return (((s.n == 192 && s.batch >= 14U && s.batch <= 20U && static_cast<uint32_t>(s.numCoresAiv) < 3U * s.batch) ||
             (s.n == 200 && s.batch >= 9U && s.batch <= 12U && static_cast<uint32_t>(s.numCoresAiv) < 4U * s.batch)) &&
            s.tensorA->GetDataType() == ge::DT_FLOAT && static_cast<uint32_t>(s.numCoresAiv) >= 2U * s.batch);
}
static bool Try192Or200Pair(DetTilingInputs &s, ge::graphStatus &result) {
    if (Is192Or200PairCandidate(s)) {
        DetFullPlan plan;
        if (!PrepareFullTiling(s, plan, static_cast<uint32_t>(s.n), static_cast<uint32_t>(s.n))) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        const uint32_t g = (s.n == 200 && static_cast<uint32_t>(s.numCoresAiv) >= 3U * s.batch) ? 3U : 2U;
        td->matPerCore = 1U;
        td->isLarge = 2U;
        td->groupSize = g;
        const bool pairLocal = s.n == 192 && g == 2U && s.platform.GetCoreNumAic() >= s.batch;
        s.context->SetBlockDim(pairLocal ? s.batch : g * s.batch);
        if (s.context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        ws[0] = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) +
                static_cast<size_t>(s.batch) * (32U * static_cast<uint32_t>(s.n) + 72U) * 8U;
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType()), LD_BATCH = pairLocal ? 59U : 57U;
        {
            result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
            return true;
        }
    }
    return false;
}

static bool Try192Cyclic(DetTilingInputs &s, ge::graphStatus &result) {
    if (s.n == 192 && s.batch <= 16U && s.tensorA->GetDataType() == ge::DT_FLOAT &&
        static_cast<uint32_t>(s.numCoresAiv) >= 3U * s.batch) {
        const uint32_t g = static_cast<uint32_t>(s.numCoresAiv) >= 6U * s.batch
                               ? 6U
                               : (static_cast<uint32_t>(s.numCoresAiv) >= 4U * s.batch ? 4U : 3U);
        DetFullPlan plan;
        if (!PrepareCyclicPlan47(s, plan, 192U, 192U, g, g * s.batch)) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        const size_t sysWs = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U);
        ws[0] = sysWs + static_cast<size_t>(s.batch) * (32U * 192U + 72U) * 8U;
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType());
        uint32_t LD_BATCH = g == 6U ? 38U : (g == 3U ? 40U : (s.batch == 12U ? 36U : 35U));
        {
            result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
            return true;
        }
    }
    return false;
}

static bool IsCommonColumnGroupCandidate(DetTilingInputs &s) {
    return (((s.n == 192 && s.batch <= 12U) || (s.n == 200 && s.batch <= 12U) || (s.n == 208 && s.batch <= 12U)) &&
            s.tensorA->GetDataType() == ge::DT_FLOAT && s.numCoresAiv >= 4 &&
            s.batch <= static_cast<uint32_t>(s.numCoresAiv) / 4U);
}
static uint32_t SelectColumnGroup(DetTilingInputs &s) {
    return s.n == 192 ? 6U : ((s.n == 200 && (s.batch == 7U || s.batch == 8U)) ? 6U : 7U);
}
static bool UseColumnGroups(DetTilingInputs &s, uint32_t columnGroup) {
    if (columnGroup == 0U)
        return false;

    return (s.n == 192 || s.n == 200 || s.n == 208) &&
           (s.n == 200 ? s.batch <= 8U : (s.batch == 1U || s.batch == 4U)) &&
           s.batch <= static_cast<uint32_t>(s.numCoresAiv) / columnGroup;
}
static bool UseIndexSeeds(DetTilingInputs &s) {
    return (s.n == 192 && (s.batch == 1U || s.batch == 4U || s.batch == 8U)) || (s.n == 200) ||
           (s.n == 208 && (s.batch == 1U || s.batch == 4U || s.batch == 8U || s.batch == 12U));
}
static bool TryCommonColumnGroups(DetTilingInputs &s, ge::graphStatus &result) {
    if (IsCommonColumnGroupCandidate(s)) {
        DetFullPlan plan;
        if (!PrepareFullTiling(s, plan, static_cast<uint32_t>(s.n), static_cast<uint32_t>(s.n))) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        const uint32_t columnGroup = SelectColumnGroup(s);
        const bool useColumnGroups = UseColumnGroups(s, columnGroup);
        td->matPerCore = 1U;
        td->isLarge = 2U;
        td->groupSize = useColumnGroups ? columnGroup : 4U;
        s.context->SetBlockDim((useColumnGroups ? columnGroup : 4U) * s.batch);
        if (s.context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        const size_t sysWs = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U);
        ws[0] = sysWs + static_cast<size_t>(s.batch) * (32U * static_cast<uint32_t>(s.n) + 72U) * 8U;
        const bool useIndexSeeds = UseIndexSeeds(s);
        const bool useScopedCombo = (s.n == 192 && s.batch == 12U);
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType()),
                 LD_BATCH = useColumnGroups ? ((s.n == 208 && s.batch == 4U) ? 39U : 38U)
                                            : (useScopedCombo ? 36U : (useIndexSeeds ? 35U : 26U));
        {
            result = SelectColdCoverage(s.context, DT_A, LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
            return true;
        }
    }
    return false;
}

template <class PhaseArg0, class PhaseArg1, class PhaseArg2, class PhaseArg3, class PhaseArg4>
static inline void TryN256maxPhase1(PhaseArg0 &groups, PhaseArg1 &pad11, PhaseArg2 &s, PhaseArg3 &selectedGroup,
                                    PhaseArg4 &ws) {

    const size_t sysWs = std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U);
    ws[0] = sysWs + static_cast<size_t>(s.batch) * 66112U;
    if (pad11)
        ws[0] += 66112U;
    if (selectedGroup == 3U && s.batch == 12U)
        ws[0] += static_cast<size_t>(s.batch) * 33056U;
    if ((selectedGroup == 3U || selectedGroup == 4U || selectedGroup == 5U) && s.batch <= 12U)
        ws[0] += static_cast<size_t>(s.batch) * 224U * (selectedGroup == 3U ? 64U : 32U) * sizeof(float);
    if (groups == s.batch && s.batch <= 12U && (selectedGroup == 7U || selectedGroup == 6U || selectedGroup == 4U))
        ws[0] += static_cast<size_t>(groups) * 6U * 64U;
    if (pad11)
        ws[0] = sysWs + 12U * (3U * 8264U + 224U * 64U) * sizeof(float) + 32U;
}
static uint32_t SelectN256Group(DetTilingInputs &s) {
    return (s.batch <= 5U && 7U * s.batch <= static_cast<uint32_t>(s.numCoresAiv))                        ? 7U
           : (s.batch <= 6U && 6U * s.batch <= static_cast<uint32_t>(s.numCoresAiv))                      ? 6U
           : (s.batch <= 8U && 5U * s.batch <= static_cast<uint32_t>(s.numCoresAiv))                      ? 5U
           : ((s.batch == 11U || s.batch == 12U) && 3U * s.batch <= static_cast<uint32_t>(s.numCoresAiv)) ? 3U
           : 4U * s.batch <= static_cast<uint32_t>(s.numCoresAiv)                                         ? 4U
                                                                                                          : 2U;
}
static uint32_t SelectN256Route(uint32_t batch, uint32_t selectedGroup, bool pad11) {
    switch (selectedGroup) {
    case 7U:
        if (batch == 4U)
            return 104U;
        if (batch <= 5U)
            return 37U;
        break;
    case 6U:
        if (batch <= 6U)
            return 27U;
        break;
    case 5U:
        if (batch <= 12U)
            return 51U;
        break;
    case 3U:
        if (batch <= 12U)
            return batch == 12U ? 54U : (pad11 ? 53U : 52U);
        break;
    case 4U:
        if (batch <= 12U)
            return (batch == 9U || batch == 10U) ? 55U : 30U;
        break;
    case 8U:
        if (batch == 1U)
            return 27U;
        break;
    default:
        break;
    }
    return batch == 4U && selectedGroup >= 4U ? 26U : 31U;
}
static bool TryN256(DetTilingInputs &s, ge::graphStatus &result) {
    if (s.n == 256 && s.tensorA->GetDataType() == ge::DT_FLOAT && s.numCoresAiv >= 4) {
        s.numCoresAiv = std::min<int32_t>(s.numCoresAiv, 40);
        DetFullPlan plan;
        if (!PrepareFullTiling(s, plan, 256U, 256U)) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        auto *td = plan.tiling;
        size_t *ws = plan.workspace;
        const uint32_t selectedGroup = SelectN256Group(s);
        if (selectedGroup == 0U) {
            result = ge::GRAPH_FAILED;
            return true;
        }

        const uint32_t groups = std::min(s.batch, static_cast<uint32_t>(s.numCoresAiv) / selectedGroup);
        if (groups == 0U) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        td->matPerCore = (s.batch + groups - 1U) / groups;
        td->isLarge = 2U;
        td->groupSize = selectedGroup;
        const bool pad11 = selectedGroup == 3U && s.batch == 11U && 36U <= static_cast<uint32_t>(s.numCoresAiv);
        s.context->SetBlockDim(selectedGroup * (pad11 ? 12U : groups));
        if (s.context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
            result = ge::GRAPH_FAILED;
            return true;
        }
        TryN256maxPhase1(groups, pad11, s, selectedGroup, ws);
        uint32_t DT_A = static_cast<uint32_t>(s.tensorA->GetDataType());
        uint32_t LD_BATCH = 31U;
        if (groups == s.batch)
            LD_BATCH = SelectN256Route(s.batch, selectedGroup, pad11);
        if (LD_BATCH == 51U && groups >= 8U)
            LD_BATCH = 103U;
        ASCENDC_TPL_SEL_PARAM(s.context, DT_A, LD_BATCH);
        {
            result = ge::GRAPH_SUCCESS;
            return true;
        }
    }
    return false;
}
template <class PhaseArg0, class PhaseArg1>
static inline ge::graphStatus BuildDefaultPlanSetBlockDimPhase1(PhaseArg0 &p, PhaseArg1 &s) {

    if (s.n == 64 && s.batch == 4U && p.matPerCore == 1U) {
        if (s.numCoresAiv < 8)
            return ge::GRAPH_FAILED;
        s.context->SetBlockDim(8U);
        p.LD_BATCH = 50U;
    }
    if (p.LD_BATCH == 47U && s.n >= 49 && s.n < 64 && s.n % 8 != 0 && s.batch <= 8U) {
        if (s.batch == 1U && s.platform.GetCoreNumAic() >= 1) {
            auto *mixedWs = s.context->GetWorkspaceSizes(1);
            if (mixedWs == nullptr)
                return ge::GRAPH_FAILED;
            mixedWs[0] = std::max<size_t>(
                mixedWs[0], std::max<size_t>(s.platform.GetLibApiWorkSpaceSize(), 16U * 1024U * 1024U) + 32U);
            if (s.context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS)
                return ge::GRAPH_FAILED;
            p.LD_BATCH = s.n >= 57 ? 106U : 105U;
        } else {
            p.LD_BATCH = 107U;
        }
        ASCENDC_TPL_SEL_PARAM(s.context, p.DT_A, p.LD_BATCH);
        return ge::GRAPH_SUCCESS;
    }
    return SelectColdCoverage(s.context, p.DT_A, p.LD_BATCH, s.platform, static_cast<uint32_t>(s.n), s.batch);
}
template <class PhaseArg0, class PhaseArg1> static inline void BuildDefaultPlanbatchPhase2(PhaseArg0 &p, PhaseArg1 &s) {

    if (s.n >= 65 && s.n <= 208 && s.n % LDET_ALIGN != 0 && (((s.n + 7U) & ~7U) == 128U))
        p.stride += 8U;
    p.isLarge = (s.n > static_cast<int64_t>(LDET_SMALL_N_MAX)) ? 1U : 0U;
    p.matPerCore = 1;
    p.blockDim = s.batch;
    if (p.isLarge == 0U && s.batch > static_cast<uint32_t>(s.numCoresAiv)) {
        p.matPerCore = (s.batch + static_cast<uint32_t>(s.numCoresAiv) - 1U) / static_cast<uint32_t>(s.numCoresAiv);
        p.blockDim = (s.batch + p.matPerCore - 1U) / p.matPerCore;
    }
    p.group = 1;
    if (p.isLarge != 0U && s.batch < static_cast<uint32_t>(s.numCoresAiv)) {
        uint32_t g = static_cast<uint32_t>(s.numCoresAiv) / s.batch;
        uint32_t cap = s.n <= 256 ? 12U : (static_cast<uint32_t>(s.n) + 31U) / 32U;
        if (g > cap) {
            g = cap;
        }
        if (g >= 2U) {
            p.group = g;
            p.isLarge = 2U;
            p.blockDim = s.batch * p.group;
        }
    }
}
template <class PhaseArg0, class PhaseArg1>
static inline void BuildDefaultPlanFillBlockFourMapsPhase3(PhaseArg0 &p, PhaseArg1 &s) {

    p.tiling->batchCount = s.batch;
    p.tiling->n = static_cast<uint32_t>(s.n);
    if (s.n == 16 && s.batch == 1U)
        FillBlockFourMaps(p.tiling);
    if (s.n == 15 && s.batch == 1U) {
        for (uint32_t phase = 0U; phase < 6U; ++phase)
            for (uint32_t lane = 0U; lane < 32U; ++lane) {
                uint32_t r = lane % 16U;
                if (r >= 13U - phase * 2U)
                    r = 0U;
                p.tiling->block4Offsets[phase * 32U + lane] = ((r + 2U) * 40U + lane / 16U) * 4U;
            }
        constexpr uint32_t tail3[40] = {41U, 42U, 40U, 0U,  0U,  0U, 0U, 0U, 42U, 40U, 41U, 0U,  0U,  0U,
                                        0U,  0U,  82U, 80U, 81U, 0U, 0U, 0U, 0U,  0U,  81U, 82U, 80U, 0U,
                                        0U,  0U,  0U,  0U,  0U,  1U, 2U, 0U, 0U,  0U,  0U,  0U};
        for (uint32_t lane = 0U; lane < 40U; ++lane)
            p.tiling->block4Offsets[192U + lane] = tail3[lane] * 4U;
    }
    p.tiling->stride = p.stride;
    p.tiling->matPerCore = p.matPerCore;
    p.tiling->isLarge = p.isLarge;
    p.tiling->groupSize = p.group;

    p.libApiWs = s.platform.GetLibApiWorkSpaceSize();
    p.sysWs = static_cast<size_t>(p.libApiWs >= p.kSysWsReserve ? p.libApiWs : p.kSysWsReserve);
    s.context->SetBlockDim(p.blockDim);
    p.currentWorkspace = s.context->GetWorkspaceSizes(1);
}
template <class PhaseArg0, class PhaseArg1>
static inline void BuildDefaultPlanGetDataTypePhase4(PhaseArg0 &p, PhaseArg1 &s) {

    p.currentWorkspace[0] = (p.userWs > 0) ? (p.sysWs + p.userWs) : 0;
    p.DT_A = static_cast<uint32_t>(s.tensorA->GetDataType());
    p.LD_BATCH = SelectKernelRoute(static_cast<uint32_t>(s.n), s.batch, p.matPerCore, p.isLarge, p.group, p.cube32,
                                   p.cubeRange, p.ub216);
    if (p.ub216 && s.batch > 1U)
        p.LD_BATCH = 32U;
    if (p.wide512)
        p.LD_BATCH = 24U;
    if (s.batch == 1U && s.n == 32)
        p.LD_BATCH = 33U;
    if (s.batch == 1U && s.n == 16)
        p.LD_BATCH = 34U;
    if (s.n >= 9 && s.n <= 32 && s.batch <= 8U && p.matPerCore == 1U && !(s.n == 32 && s.batch == 1U))
        p.LD_BATCH = 42U;
}
template <class PhaseArg0, class PhaseArg1> static inline void BuildDefaultPlanStatePhase5(PhaseArg0 &p, PhaseArg1 &s) {

    if (s.n == 18 && s.batch == 8U && p.matPerCore == 1U)
        p.LD_BATCH = 45U;
    if (s.n >= 49 && s.n <= 63 && (s.n % 8) != 0 && s.batch <= 8U && p.matPerCore == 1U)
        p.LD_BATCH = 46U;
    if (s.n >= 49 && s.n <= 63 && (s.n % 8) != 0 && s.batch <= 8U && p.matPerCore == 1U)
        p.LD_BATCH = 47U;
    if (s.batch == 1U && s.n == 16)
        p.LD_BATCH = 34U;
}
template <class PhaseArg0, class PhaseArg1>
static inline void BuildDefaultPlanSetBlockDimPhase6(PhaseArg0 &p, PhaseArg1 &s) {

    if (s.batch == 1U && (s.n == 9 || s.n == 10 || s.n == 20) && s.numCoresAiv >= 2)
        s.context->SetBlockDim(2U);
    if (s.batch == 1U && s.n == 32 && s.numCoresAiv >= 2)
        s.context->SetBlockDim(2U);
    if (s.n == 19 && p.matPerCore == 1U && (s.batch == 6U || s.batch == 8U))
        p.LD_BATCH = s.batch == 6U ? 43U : 44U;
    BuildDefaultPlanStatePhase5(p, s);
    if (s.n == 15 && s.batch == 1U)
        p.LD_BATCH = 48U;
}
template <class PhaseArg0, class PhaseArg1> static inline void BuildDefaultPlanStatePhase7(PhaseArg0 &p, PhaseArg1 &s) {

    if (s.n == 16U && s.batch >= 8U && s.batch <= 40U && p.matPerCore == 1U)
        p.LD_BATCH = 62U;
    if (s.n == 31 && s.batch >= 9U && s.batch <= 48U && p.matPerCore == 1U)
        p.LD_BATCH = 42U;
    if ((s.n == 40 || s.n == 48 || s.n == 56 || (s.n == 63 && s.batch >= 9U)) && s.batch <= 48U && p.matPerCore == 1U)
        p.LD_BATCH = 47U;
    BuildDefaultPlanSetBlockDimPhase6(p, s);
}
template <class PhaseArg0, class PhaseArg1> static inline void BuildDefaultPlanminPhase8(PhaseArg0 &p, PhaseArg1 &s) {

    p.cubePerGroup = p.cube32 ? std::min(4U, std::min(static_cast<uint32_t>(s.numCoresAiv) / (2U * p.cubeGroups),
                                                      static_cast<uint32_t>(s.platform.GetCoreNumAic()) / p.cubeGroups))
                              : 4U;
    p.cubeRange = (s.n >= 225 && s.n < 251 && s.batch == 1U && s.numCoresAiv >= 8 && s.platform.GetCoreNumAic() >= 4);
    if (p.cube32 || p.cubeRange) {
        p.group = 2U * p.cubePerGroup;
        p.blockDim = p.cubePerGroup * p.cubeGroups;
        p.isLarge = 2U;
    }
    if (p.wide512)
        p.stride = 512U;
    p.ub216 = (s.n >= 209 && s.n <= 216);
}
template <class PhaseArg0, class PhaseArg1>
static inline ge::graphStatus BuildDefaultPlanStatePhase9(PhaseArg0 &p, PhaseArg1 &s) {

    BuildDefaultPlanGetDataTypePhase4(p, s);
    if (s.n == 32 && s.batch >= 9U && s.batch <= 48U && p.matPerCore == 1U)
        p.LD_BATCH = 42U;
    if (s.n == 16 && s.batch >= 9U && s.batch <= 48U && p.matPerCore == 1U)
        p.LD_BATCH = 42U;
    BuildDefaultPlanStatePhase7(p, s);
    if (s.n == 64 && p.matPerCore > 1U)
        p.LD_BATCH = 49U;
    return BuildDefaultPlanSetBlockDimPhase1(p, s);
}
template <class PhaseArg0, class PhaseArg1, class PhaseArg2>
static inline void BuildDefaultPlanfillPhase10(PhaseArg0 &p, PhaseArg1 &s, PhaseArg2 &tc) {

    std::fill(std::begin(p.tiling->pad), std::end(p.tiling->pad), 0);
    std::fill(std::begin(p.tiling->cubeTiling), std::end(p.tiling->cubeTiling), 0);
    std::copy_n(reinterpret_cast<const unsigned char *>(&tc), sizeof(tc),
                reinterpret_cast<unsigned char *>(p.tiling->cubeTiling));
    if (p.cube32) {
        p.tiling->pad[0] = p.cubeGroups;
        p.tiling->pad[1] = p.cubePerGroup;
        if (p.wide512)
            p.userWs = static_cast<size_t>(p.cubeGroups) *
                       ((2ULL * s.n * 512U + 512U * 32U + 48U + 256U) * 4U + 65536U + 65536U + 27U * 64U);
        else
            p.userWs = static_cast<size_t>(p.cubeGroups) *
                       ((2U * 256U * 256U + 512U * 24U + 32U + 256U) * sizeof(float) + 32768U + 65536U + 27U * 64U);
    } else
        p.userWs += 32768U + 65536U + 18U * 64U;
}
template <class PhaseArg0, class PhaseArg1>
static inline void BuildDefaultPlanSetShapePhase11(PhaseArg0 &mt, PhaseArg1 &p) {

    mt.SetShape(64, p.wide512 ? 480 : 224, 32);
    mt.SetOrgShape(p.wide512 ? 512 : 256, 512, 32, 32);
    mt.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mt.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mt.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mt.SetBias(false);
    mt.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
    mt.SetBufferSpace(-1, -1, -1);
    mt.SetMatmulConfigParams(1);
    mt.SetDoubleBuffer(true, true, true, false, true, true);
    mt.SetFixSplit(64, 64, 16);
}
template <class PhaseArg0, class PhaseArg1>
static inline ge::graphStatus BuildDefaultPlanstatic_assertPhase12(PhaseArg0 &p, PhaseArg1 &s) {

    if (p.cube32 || p.cubeRange) {
        static_assert(sizeof(AscendC::tiling::TCubeTiling) <= sizeof(p.tiling->cubeTiling),
                      "cube tiling tail too small");
        static_assert(offsetof(LinalgDetTilingDataFull, cubeTiling) == 64, "cube tiling offset changed");
        matmul_tiling::MatmulApiTiling mt(s.platform);
        BuildDefaultPlanSetShapePhase11(mt, p);
        AscendC::tiling::TCubeTiling tc{};
        if (mt.GetTiling(tc) != 0)
            return ge::GRAPH_FAILED;
        tc.stepM = tc.stepN = tc.stepKa = tc.stepKb = 1;
        tc.depthA1 = tc.depthB1 = tc.dbL0A = tc.dbL0B = tc.dbL0C = 2;
        tc.shareL1Size = std::max(tc.shareL1Size, 20480);
        tc.shareL0CSize = std::max(tc.shareL0CSize, 32768);
        uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0;
        s.platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        s.platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
        s.platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
        s.platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        if (tc.baseM != 64 || tc.baseN != 64 || tc.baseK != 32 || tc.singleCoreM != 64 ||
            tc.singleCoreN != (p.wide512 ? 480 : 224) || tc.singleCoreK != 32 || tc.usedCoreNum != 1 ||
            tc.shareL1Size > l1 || tc.shareL0CSize > l0c || 16384 > l0a || 16384 > l0b)
            return ge::GRAPH_FAILED;
        BuildDefaultPlanfillPhase10(p, s, tc);
    }
    return BuildDefaultPlanStatePhase9(p, s);
}
template <class PhaseArg0, class PhaseArg1>
static inline ge::graphStatus BuildDefaultPlanGetCoreNumAicPhase13(PhaseArg0 &p, PhaseArg1 &s) {

    p.wide512 = (s.n > 256 && s.numCoresAiv >= 2 && s.platform.GetCoreNumAic() >= 1);
    p.cube32 = (s.n >= 251 && s.n <= 256) || p.wide512;
    p.cubeGroups = p.cube32 ? std::min(s.batch, std::min(static_cast<uint32_t>(s.numCoresAiv) / 2U,
                                                         static_cast<uint32_t>(s.platform.GetCoreNumAic())))
                            : 1U;
    if (p.cubeGroups == 0U)
        return ge::GRAPH_FAILED;
    BuildDefaultPlanminPhase8(p, s);
    if (p.ub216) {
        p.stride = static_cast<uint32_t>(s.n);
        p.group = 1U;
        p.blockDim = std::min(s.batch, static_cast<uint32_t>(s.numCoresAiv));
        if (p.blockDim == 0U)
            return ge::GRAPH_FAILED;
        p.matPerCore = (s.batch + p.blockDim - 1U) / p.blockDim;
        p.isLarge = 0U;
    }
    p.tiling = s.context->template GetTilingData<LinalgDetTilingDataFull>();
    if (p.tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    BuildDefaultPlanFillBlockFourMapsPhase3(p, s);
    if (p.currentWorkspace == nullptr)
        return ge::GRAPH_FAILED;
    p.wsPerMat = static_cast<size_t>(p.stride) * static_cast<uint32_t>(s.n);
    if (p.isLarge == 2U) {
        p.wsPerMat = 2U * p.wsPerMat + 512U * 24U + 32U + 256U;
    }
    p.userWs = (p.isLarge != 0U) ? static_cast<size_t>(s.batch) * p.wsPerMat * sizeof(float) : 0;
    return BuildDefaultPlanstatic_assertPhase12(p, s);
}
static ge::graphStatus BuildDefaultPlan(DetTilingInputs &s) {
    DetDefaultPlan p;
    p.stride = (static_cast<uint32_t>(s.n) + LDET_ALIGN - 1) / LDET_ALIGN * LDET_ALIGN;
    if (s.n == 16 || s.n == 32 || s.n == 64 || s.n == 128)
        p.stride += 8U;
    BuildDefaultPlanbatchPhase2(p, s);
    if (s.n >= 251 && s.n <= 256 && (s.numCoresAiv < 2 || s.platform.GetCoreNumAic() < 1))
        return ge::GRAPH_FAILED;
    return BuildDefaultPlanGetCoreNumAicPhase13(p, s);
}
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    DetTilingInputs s(context);
    if (!ReadTilingInputs(s))
        return ge::GRAPH_FAILED;
    ge::graphStatus result;
    if (TryRagged208(s, result))
        return result;
    if (s.n == 208)
        s.numCoresAiv = std::min<int32_t>(s.numCoresAiv, 40);
    if (Try208SmallBatch(s, result))
        return result;
    if (Try208MediumBatch(s, result))
        return result;
    if (Try200WideGroup(s, result))
        return result;
    if (Try200PairGroup(s, result))
        return result;
    if (Try192Or200Triple(s, result))
        return result;
    if (Try192Or200Pair(s, result))
        return result;
    if (Try192Cyclic(s, result))
        return result;
    if (TryCommonColumnGroups(s, result))
        return result;
    if (TryN256(s, result))
        return result;
    return BuildDefaultPlan(s);
}

} // namespace optiling
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inShape = context->GetInputShape(0);
    if (inShape == nullptr || inShape->GetDimNum() < 2) {
        return GRAPH_FAILED;
    }
    gert::Shape *outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return GRAPH_FAILED;
    }
    size_t rank = inShape->GetDimNum();
    outShape->SetDimNum(rank - 2);
    for (size_t i = 0; i + 2 < rank; ++i) {
        outShape->SetDim(i, inShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge
namespace ops {
class LinalgDet : public OpDef {
  public:
    explicit LinalgDet(const char *name) : OpDef(name) {
        this->Input("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("det")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgDet);
} // namespace ops
