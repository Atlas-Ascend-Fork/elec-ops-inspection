/*
 * Copyright (c) 2026 aodebiao.
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
#include <cstdint>
#include "exe_graph/runtime/shape.h"
#include "../op_kernel/linalg_solve_tiling.h"

namespace optiling {
struct SolveShapePlan {
    LinalgSolveTilingData data{};
    bool vectorRhs = false;
};

static bool CheckMatrixOrder(const gert::Shape &matrix, uint32_t &order)
{
    const size_t rank = matrix.GetDimNum();
    if (rank < 2U) {
        return false;
    }
    const int64_t size = matrix.GetDim(rank - 1U);
    if (size < 0 || size > UINT32_MAX - 7U || matrix.GetDim(rank - 2U) != size) {
        return false;
    }
    order = static_cast<uint32_t>(size);
    return true;
}

static bool IsRhsVector(const gert::Shape &matrix, const gert::Shape &rhs)
{
    if (rhs.GetDimNum() == 1U) {
        return true;
    }
    if (rhs.GetDimNum() + 1U != matrix.GetDimNum()) {
        return false;
    }
    for (size_t i = 0U; i < rhs.GetDimNum(); ++i) {
        if (rhs.GetDim(i) != matrix.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool ResolveRhs(const gert::Shape &matrix, const gert::Shape &rhs, bool left, SolveShapePlan &plan)
{
    const size_t rank = rhs.GetDimNum();
    if (rank == 0U) {
        return false;
    }
    plan.vectorRhs = IsRhsVector(matrix, rhs);
    if (plan.vectorRhs) {
        plan.data.k = 1U;
        return rhs.GetDim(rank - 1U) == plan.data.n;
    }
    const int64_t n = rhs.GetDim(rank - (left ? 2U : 1U));
    const int64_t k = rhs.GetDim(rank - (left ? 1U : 2U));
    if (n != plan.data.n || k < 0 || k > UINT32_MAX - 7U) {
        return false;
    }
    plan.data.k = static_cast<uint32_t>(k);
    return true;
}

static bool BroadcastBatchShape(const gert::Shape &matrix, const gert::Shape &rhs, SolveShapePlan &plan)
{
    const uint32_t matrixRank = matrix.GetDimNum() - 2U;
    const uint32_t rhsRank = rhs.GetDimNum() - (plan.vectorRhs ? 1U : 2U);
    auto &data = plan.data;
    data.batchRank = matrixRank > rhsRank ? matrixRank : rhsRank;
    if (data.batchRank > MAX_SOLVE_BATCH_DIMS ||
        data.batchRank + (plan.vectorRhs ? 1U : 2U) > gert::Shape::kMaxDimNum) {
        return false;
    }
    for (uint32_t i = 0U; i < data.batchRank; ++i) {
        const int64_t a = i + matrixRank < data.batchRank ? 1 : matrix.GetDim(i + matrixRank - data.batchRank);
        const int64_t b = i + rhsRank < data.batchRank ? 1 : rhs.GetDim(i + rhsRank - data.batchRank);
        if (a < 0 || b < 0 || a > UINT32_MAX || b > UINT32_MAX) {
            return false;
        }
        if (a != b && a != 1 && b != 1) {
            return false;
        }
        data.batchDims[i] = static_cast<uint32_t>(a == 1 ? b : a);
    }
    return true;
}

static bool CountOutputBatch(LinalgSolveTilingData &data)
{
    for (uint32_t i = 0U; i < data.batchRank; ++i) {
        if (data.batchDims[i] == 0U) {
            data.batch = 0U;
            return true;
        }
    }
    uint64_t batch = 1U;
    for (uint32_t i = 0U; i < data.batchRank; ++i) {
        batch *= data.batchDims[i];
        if (batch > UINT32_MAX) {
            return false;
        }
    }
    data.batch = static_cast<uint32_t>(batch);
    return true;
}

static bool FillInputStrides(const gert::Shape &shape, uint32_t inputRank, uint32_t outputRank,
                            uint64_t stride, uint64_t *strides)
{
    for (int32_t i = static_cast<int32_t>(inputRank) - 1; i >= 0; --i) {
        const uint64_t dimension = static_cast<uint64_t>(shape.GetDim(i));
        strides[outputRank - inputRank + i] = dimension == 1U ? 0U : stride;
        if (dimension != 0U && stride > UINT64_MAX / dimension) {
            return false;
        }
        stride *= dimension;
    }
    return true;
}

static bool MakeSolvePlan(const gert::Shape *matrix, const gert::Shape *rhs, bool left, SolveShapePlan &plan)
{
    if (matrix == nullptr || rhs == nullptr || !CheckMatrixOrder(*matrix, plan.data.n)) {
        return false;
    }
    if (!ResolveRhs(*matrix, *rhs, left, plan) || !BroadcastBatchShape(*matrix, *rhs, plan) ||
        !CountOutputBatch(plan.data)) {
        return false;
    }
    auto &data = plan.data;
    data.rightSide = left ? 0U : 1U;
    data.p = (data.n + 7U) / 8U * 8U;
    data.q = (data.k + 7U) / 8U * 8U;
    if (data.batch == 0U) {
        return true;
    }
    const uint32_t matrixRank = matrix->GetDimNum() - 2U;
    const uint32_t rhsRank = rhs->GetDimNum() - (plan.vectorRhs ? 1U : 2U);
    return FillInputStrides(*matrix, matrixRank, data.batchRank,
                           static_cast<uint64_t>(data.n) * data.n, data.matrixBatchStrides) &&
           FillInputStrides(*rhs, rhsRank, data.batchRank,
                           static_cast<uint64_t>(data.n) * data.k, data.rhsBatchStrides);
}
} // namespace optiling
