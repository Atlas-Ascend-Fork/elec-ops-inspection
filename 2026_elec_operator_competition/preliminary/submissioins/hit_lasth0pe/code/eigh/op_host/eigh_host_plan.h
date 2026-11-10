/*
 * Copyright (c) 2026 lasth0pe.
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
#include "../op_kernel/eigh_case_traits.h"
#include "../op_kernel/linalg_eigh_tiling.h"

constexpr int32_t CalcStedcBlocks(uint32_t n, uint32_t solver)
{
    return (solver == 3U) ? ((n > 128U) ? 4 : 2) : 0;
}

constexpr int32_t CalcBlockBound(uint32_t n, int32_t m, int32_t b)
{
    return (m > 0 && b <= m) ? (((b * static_cast<int32_t>(n / 8U)) / m) * 8) : 0;
}

constexpr int32_t CalcBlockSpan(uint32_t n, int32_t m, int32_t b)
{
    return CalcBlockBound(n, m, b + 1) - CalcBlockBound(n, m, b);
}

constexpr int32_t CalcLeafOffsetInBlock(int32_t span, int32_t rem)
{
    return (rem == 0) ? 0 : (
           (rem == 1) ? (((span > 48) ? 32 : 24) - 16) : (
           (rem == 2) ? ((span > 48) ? 32 : 24) : (
                        (span > 56) ? 48 : (((span > 48) ? 32 : 24) + 8))));
}

constexpr int32_t CalcLeafBound(uint32_t n, int32_t m, int32_t lf)
{
    return (m <= 0 || lf > 4 * m) ? 0 : (
           (lf == 4 * m) ? static_cast<int32_t>(n) : (
           CalcBlockBound(n, m, lf / 4) + CalcLeafOffsetInBlock(CalcBlockSpan(n, m, lf / 4), lf % 4)));
}

constexpr uint32_t CalcMaxBlockSpan(uint32_t n, int32_t m, int32_t b = 0)
{
    return (m <= 0 || b >= m) ? 0U : (
           (static_cast<uint32_t>(CalcBlockSpan(n, m, b)) > CalcMaxBlockSpan(n, m, b + 1))
               ? static_cast<uint32_t>(CalcBlockSpan(n, m, b))
               : CalcMaxBlockSpan(n, m, b + 1));
}

constexpr uint32_t FindCoreBlockIdx(uint32_t p, uint32_t pCount, uint32_t m, uint32_t b = 0U)
{
    return (m == 0U || b >= m) ? 0U : (
           (p >= (b * pCount) / m && p < ((b + 1U) * pCount) / m)
               ? b
               : FindCoreBlockIdx(p, pCount, m, b + 1U));
}

constexpr EighCorePlan MakeActiveCorePlan(uint32_t n, uint32_t pCount, uint32_t m, uint32_t c, uint32_t blk)
{
    return EighCorePlan{
        (pCount > 0U) ? ((c / pCount) * n * n) : 0U,
        (pCount > 0U) ? ((c / pCount) * n) : 0U,
        (pCount > 0U) ? (c % pCount) : 0U,
        (pCount > 0U) ? (((c % pCount) * n) / pCount) : 0U,
        (pCount > 0U) ? ((((c % pCount) + 1U) * n) / pCount) : 0U,
        (pCount > 0U) ? ((((c % pCount) + 1U) * n) / pCount - ((c % pCount) * n) / pCount) : 0U,
        blk,
        (m > 0U) ? ((blk * pCount) / m) : 0U,
        (m > 0U && pCount > 0U) ? ((c % pCount) - (blk * pCount) / m) : 0U,
        (m > 0U) ? (((blk + 1U) * pCount) / m - (blk * pCount) / m) : pCount
    };
}

constexpr EighCorePlan MakeCorePlanEntry(uint32_t n, uint32_t usedCores, uint32_t pCount, uint32_t m, uint32_t c)
{
    return (c < usedCores && pCount > 0U)
        ? MakeActiveCorePlan(n, pCount, m, c, FindCoreBlockIdx(c % pCount, pCount, m))
        : EighCorePlan{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
}

constexpr uint32_t CalcPstride(uint32_t n, uint32_t pCount)
{
    return (pCount > 0U && ((n + pCount - 1U) / pCount) <= 16U) ? 16U : 32U;
}

constexpr LinalgEighTilingData BuildEighTilingPlanImpl(
    uint32_t n, uint32_t totalBatch, uint32_t pCount, uint32_t solver, uint32_t uplo,
    int32_t m, uint32_t mu, uint32_t uCores, uint32_t ps)
{
    return LinalgEighTilingData{
        n, totalBatch, uplo, (uCores > 0U) ? ((uCores <= 32U) ? uCores : 32U) : 1U,
        pCount, (solver == 0U || solver == 4U) ? 4096U : 48640U,
        (n + 63U) & ~63U, (pCount > 0U) ? ((n + pCount - 1U) / pCount) : n, ps, pCount * ps, pCount * ps + 512U, 2U * (pCount * ps + 512U),
        m, 4 * m, CalcMaxBlockSpan(n, m), CalcMaxBlockSpan(n, m) * 64U,
        (solver == 3U) ? (n * n - static_cast<uint32_t>(4 * m) * 320U) : 0U,
        {CalcBlockBound(n, m, 0), CalcBlockBound(n, m, 1), CalcBlockBound(n, m, 2), CalcBlockBound(n, m, 3), CalcBlockBound(n, m, 4), CalcBlockBound(n, m, 5), CalcBlockBound(n, m, 6), CalcBlockBound(n, m, 7), CalcBlockBound(n, m, 8)},
        {CalcLeafBound(n, m, 0), CalcLeafBound(n, m, 1), CalcLeafBound(n, m, 2), CalcLeafBound(n, m, 3), CalcLeafBound(n, m, 4), CalcLeafBound(n, m, 5), CalcLeafBound(n, m, 6), CalcLeafBound(n, m, 7), CalcLeafBound(n, m, 8), CalcLeafBound(n, m, 9), CalcLeafBound(n, m, 10),
         CalcLeafBound(n, m, 11), CalcLeafBound(n, m, 12), CalcLeafBound(n, m, 13), CalcLeafBound(n, m, 14), CalcLeafBound(n, m, 15), CalcLeafBound(n, m, 16), CalcLeafBound(n, m, 17), CalcLeafBound(n, m, 18), CalcLeafBound(n, m, 19), CalcLeafBound(n, m, 20), CalcLeafBound(n, m, 21),
         CalcLeafBound(n, m, 22), CalcLeafBound(n, m, 23), CalcLeafBound(n, m, 24), CalcLeafBound(n, m, 25), CalcLeafBound(n, m, 26), CalcLeafBound(n, m, 27), CalcLeafBound(n, m, 28), CalcLeafBound(n, m, 29), CalcLeafBound(n, m, 30), CalcLeafBound(n, m, 31), CalcLeafBound(n, m, 32)},
        {MakeCorePlanEntry(n, uCores, pCount, mu, 0U), MakeCorePlanEntry(n, uCores, pCount, mu, 1U), MakeCorePlanEntry(n, uCores, pCount, mu, 2U), MakeCorePlanEntry(n, uCores, pCount, mu, 3U), MakeCorePlanEntry(n, uCores, pCount, mu, 4U), MakeCorePlanEntry(n, uCores, pCount, mu, 5U), MakeCorePlanEntry(n, uCores, pCount, mu, 6U), MakeCorePlanEntry(n, uCores, pCount, mu, 7U),
         MakeCorePlanEntry(n, uCores, pCount, mu, 8U), MakeCorePlanEntry(n, uCores, pCount, mu, 9U), MakeCorePlanEntry(n, uCores, pCount, mu, 10U), MakeCorePlanEntry(n, uCores, pCount, mu, 11U), MakeCorePlanEntry(n, uCores, pCount, mu, 12U), MakeCorePlanEntry(n, uCores, pCount, mu, 13U), MakeCorePlanEntry(n, uCores, pCount, mu, 14U), MakeCorePlanEntry(n, uCores, pCount, mu, 15U),
         MakeCorePlanEntry(n, uCores, pCount, mu, 16U), MakeCorePlanEntry(n, uCores, pCount, mu, 17U), MakeCorePlanEntry(n, uCores, pCount, mu, 18U), MakeCorePlanEntry(n, uCores, pCount, mu, 19U), MakeCorePlanEntry(n, uCores, pCount, mu, 20U), MakeCorePlanEntry(n, uCores, pCount, mu, 21U), MakeCorePlanEntry(n, uCores, pCount, mu, 22U), MakeCorePlanEntry(n, uCores, pCount, mu, 23U),
         MakeCorePlanEntry(n, uCores, pCount, mu, 24U), MakeCorePlanEntry(n, uCores, pCount, mu, 25U), MakeCorePlanEntry(n, uCores, pCount, mu, 26U), MakeCorePlanEntry(n, uCores, pCount, mu, 27U), MakeCorePlanEntry(n, uCores, pCount, mu, 28U), MakeCorePlanEntry(n, uCores, pCount, mu, 29U), MakeCorePlanEntry(n, uCores, pCount, mu, 30U), MakeCorePlanEntry(n, uCores, pCount, mu, 31U)}
    };
}

constexpr LinalgEighTilingData BuildEighTilingPlan(
    uint32_t n, uint32_t totalBatch, uint32_t pCount, uint32_t solver, uint32_t uplo = 0U)
{
    return BuildEighTilingPlanImpl(
        n, totalBatch, pCount, solver, uplo,
        CalcStedcBlocks(n, solver), static_cast<uint32_t>(CalcStedcBlocks(n, solver)),
        totalBatch * pCount, CalcPstride(n, pCount));
}

template <uint32_t CASE_KEY>
struct EighHostPlan {
    static constexpr LinalgEighTilingData value = BuildEighTilingPlan(
        EighCase<CASE_KEY>::N, EighCase<CASE_KEY>::B,
        EighCase<CASE_KEY>::P, EighCase<CASE_KEY>::Solver, EighCase<CASE_KEY>::U);
};

template <uint32_t CASE_KEY>
constexpr LinalgEighTilingData EighHostPlan<CASE_KEY>::value;

template <uint32_t CASE_KEY>
inline LinalgEighTilingData BuildCaseTilingPlan(uint32_t uplo = EighCase<CASE_KEY>::U)
{
    LinalgEighTilingData plan = EighHostPlan<CASE_KEY>::value;
    plan.uplo = uplo;
    return plan;
}

static_assert(sizeof(LinalgEighTilingData) == 1516U, "Tiling ABI changed");
