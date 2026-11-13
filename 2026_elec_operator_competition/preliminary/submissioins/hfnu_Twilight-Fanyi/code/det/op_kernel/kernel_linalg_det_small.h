/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Author: Twilight-Fanyi.
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
#ifndef KERNEL_LINALG_DET_SMALL_H
#define KERNEL_LINALG_DET_SMALL_H

#include "kernel_linalg_det.h"

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline float KernelLinalgDet<LocalLuBand, Pipelined>::ReadMatrixValue(const float *matrix,
                                                                                 uint32_t index) const
{
    return matrix[index];
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline float KernelLinalgDet<LocalLuBand, Pipelined>::ReadMatrixValue(AscendC::LocalTensor<float> matrix,
                                                                                 uint32_t index) const
{
    return matrix.GetValue(index);
}

template <uint32_t LocalLuBand, bool Pipelined>
template <typename Matrix>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::ProbeTriangular(Matrix matrix, uint32_t n,
                                                                                uint32_t pitch,
                                                                                float &triangularDeterminant) const
{
    bool lowerTriangular = true;
    bool upperTriangular = true;
    triangularDeterminant = 1.0F;
    for (uint32_t row = 0U; row < n; ++row) {
        triangularDeterminant *= ReadMatrixValue(matrix, row * pitch + row);
        if (lowerTriangular) {
            for (uint32_t column = row + 1U; column < n; ++column) {
                if (ReadMatrixValue(matrix, row * pitch + column) != 0.0F) {
                    lowerTriangular = false;
                    break;
                }
            }
        }
        if (upperTriangular) {
            for (uint32_t column = 0U; column < row; ++column) {
                if (ReadMatrixValue(matrix, row * pitch + column) != 0.0F) {
                    upperTriangular = false;
                    break;
                }
            }
        }
        if (!lowerTriangular && !upperTriangular)
            break;
    }
    return lowerTriangular || upperTriangular;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::DetSpecializedSmall(uint32_t batch, uint64_t inputBase)
{
    if constexpr (LocalLuBand == 16U) {
        StoreDet(batch, DetSmall<676U, 26U>(inputBase));
        return true;
    }
    if constexpr (LocalLuBand == 17U) {
        StoreDet(batch, DetSmall<961U, 31U>(inputBase));
        return true;
    }
    if constexpr ((LocalLuBand >= 18U && LocalLuBand <= 53U) || (LocalLuBand >= 59U && LocalLuBand <= 61U)) {
        constexpr uint32_t SmallN = LocalLuBand - 14U;
        StoreDet(batch, DetSmall<SmallN * SmallN, SmallN>(inputBase));
        return true;
    }
    if constexpr (LocalLuBand == 2U) {
        StoreDet(batch, DetSmall<3025U>(inputBase));
        return true;
    }
    if constexpr (LocalLuBand == 3U) {
        StoreDet(batch, DetSmall<3969U>(inputBase));
        return true;
    }
    if constexpr (LocalLuBand == 4U) {
        StoreDet(batch, DetSmall<5041U>(inputBase));
        return true;
    }
    if constexpr (LocalLuBand == 5U) {
        StoreDet(batch, DetSmall<6241U>(inputBase));
        return true;
    }
    if constexpr (LocalLuBand == 6U) {
        StoreDet(batch, DetSmall<7569U>(inputBase));
        return true;
    }

    return false;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::DetSmallRuntime(uint32_t batch, uint64_t inputBase)
{
    if (n_ <= 8U) {
        StoreDet(batch, DetSmall<64U>(inputBase));
        return true;
    }
    if (n_ <= 16U) {
        StoreDet(batch, DetSmall<256U>(inputBase));
        return true;
    }
    if (n_ <= 24U) {
        StoreDet(batch, DetSmall<576U>(inputBase));
        return true;
    }
    if (n_ <= 32U) {
        StoreDet(batch, n_ == 32U ? DetSmall<1024U, 32U>(inputBase) : DetSmall<1024U>(inputBase));
        return true;
    }

    return false;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetTiny(uint32_t batch, uint64_t inputBase)
{
    if (n_ == 1U) {
        StoreDet(batch, aGm_.GetValue(inputBase));
        return;
    }
    if (n_ == 2U) {
        const float a00 = aGm_.GetValue(inputBase);
        const float a01 = aGm_.GetValue(inputBase + 1U);
        const float a10 = aGm_.GetValue(inputBase + 2U);
        const float a11 = aGm_.GetValue(inputBase + 3U);
        StoreDet(batch, a00 * a11 - a01 * a10);
        return;
    }
    if (n_ == 3U) {
        const float a00 = aGm_.GetValue(inputBase);
        const float a01 = aGm_.GetValue(inputBase + 1U);
        const float a02 = aGm_.GetValue(inputBase + 2U);
        const float a10 = aGm_.GetValue(inputBase + 3U);
        const float a11 = aGm_.GetValue(inputBase + 4U);
        const float a12 = aGm_.GetValue(inputBase + 5U);
        const float a20 = aGm_.GetValue(inputBase + 6U);
        const float a21 = aGm_.GetValue(inputBase + 7U);
        const float a22 = aGm_.GetValue(inputBase + 8U);
        StoreDet(batch, a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20));
        return;
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::SwapScalarLocalPivot(float *matrix, uint32_t n,
                                                                                     uint32_t pivotColumn,
                                                                                     uint32_t pivotRow,
                                                                                     float &sign) const
{
    if (pivotRow != pivotColumn) {
        sign = -sign;
        for (uint32_t column = pivotColumn; column < n; ++column) {
            const uint32_t first = pivotColumn * n + column;
            const uint32_t second = pivotRow * n + column;
            const float value = matrix[first];
            matrix[first] = matrix[second];
            matrix[second] = value;
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::FindScalarLocalPivot(const float *matrix, uint32_t n,
                                                                                     uint32_t pivotColumn,
                                                                                     uint32_t &pivotRow,
                                                                                     float &pivotAbs) const
{
    pivotRow = pivotColumn;
    pivotAbs = Abs(matrix[pivotColumn * n + pivotColumn]);
    // FixedN gives this scan a compile-time upper bound; probe whether the
    // compiler can schedule the pivot search better without changing math.
#pragma unroll
    for (uint32_t row = pivotColumn + 1U; row < n; ++row) {
        const float candidate = Abs(matrix[row * n + pivotColumn]);
        if (candidate > pivotAbs) {
            pivotAbs = candidate;
            pivotRow = row;
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
template <uint32_t FixedN>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateScalarLocalRows(float *matrix, uint32_t n,
                                                                                         uint32_t pivotColumn,
                                                                                         float pivot) const
{
    constexpr uint32_t FIXED_N = FixedN;
    if (pivot == 0.0F) {
        return;
    }
    for (uint32_t row = pivotColumn + 1U; row < n; ++row) {
        const uint32_t rowBase = row * n;
        const float factor = matrix[rowBase + pivotColumn] / pivot;
        // This column has left the active panel.  `factor` is consumed only
        // by the trailing-row update below; no later pivot search or update
        // reads matrix[rowBase + pivotColumn].  Avoid a dead local write.
        if constexpr (FixedN != 0U) {
            // Unlike the runtime n<=31 instantiations, this branch has a fixed
            // 32-wide local row.  Let the compiler schedule the dependent
            // trailing FMA chain without changing the pivot/factor order.
#pragma unroll
            for (uint32_t column = pivotColumn + 1U; column < FIXED_N; ++column) {
                matrix[rowBase + column] -= factor * matrix[pivotColumn * FIXED_N + column];
            }
        } else {
            for (uint32_t column = pivotColumn + 1U; column < n; ++column) {
                matrix[rowBase + column] -= factor * matrix[pivotColumn * n + column];
            }
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::ProbeScalarTriangular(
    const float *matrix, uint32_t n, float &triangularDeterminant) const
{
    return ProbeTriangular(matrix, n, n, triangularDeterminant);
}

// Small matrices are dominated by repeated GM/workspace traffic in the
// generic path. Keep each local capacity separate so the 25..32 extension
// cannot increase local-storage pressure in the previously tuned ranges.
template <uint32_t LocalLuBand, bool Pipelined>
template <uint32_t Capacity, uint32_t FixedN>
__aicore__ inline float KernelLinalgDet<LocalLuBand, Pipelined>::DetSmall(uint64_t inputBase) const
{
    // Every element read below is copied from aGm_ before use; avoid clearing
    // the complete capacity (notably 1024 floats for the fixed32 path).
    float matrix[Capacity];
    constexpr uint32_t FIXED_N = FixedN;
    const uint32_t n = FixedN != 0U ? FIXED_N : n_;
    const uint32_t matrixElements = n * n;
    if constexpr (FixedN != 0U) {
        // Keep a runtime row loop to avoid v38's 1024-way code expansion, but
        // expose the fixed 32-element contiguous row transfer to scheduling.
        for (uint32_t row = 0U; row < FIXED_N; ++row) {
            const uint32_t rowBase = row * FIXED_N;
#pragma unroll
            for (uint32_t column = 0U; column < FIXED_N; ++column) {
                matrix[rowBase + column] = aGm_.GetValue(inputBase + rowBase + column);
            }
        }
    } else {
        for (uint32_t index = 0U; index < matrixElements; ++index) {
            matrix[index] = aGm_.GetValue(inputBase + index);
        }
    }

    float triangularDeterminant = 1.0F;
    if (ProbeScalarTriangular(matrix, n, triangularDeterminant)) {
        return triangularDeterminant;
    }
    float sign = 1.0F;
    float determinant = 1.0F;
    for (uint32_t pivotColumn = 0U; pivotColumn < n; ++pivotColumn) {
        uint32_t pivotRow = pivotColumn;
        float pivotAbs = 0.0F;
        FindScalarLocalPivot(matrix, n, pivotColumn, pivotRow, pivotAbs);
        if (pivotAbs == 0.0F) {
            return 0.0F;
        }
        SwapScalarLocalPivot(matrix, n, pivotColumn, pivotRow, sign);
        const float pivot = matrix[pivotColumn * n + pivotColumn];
        if (pivot == 0.0F) {
            return 0.0F;
        }
        determinant *= pivot;
        EliminateScalarLocalRows<FixedN>(matrix, n, pivotColumn, pivot);
    }
    return sign * determinant;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetBatch(uint32_t batch)
{
    const uint64_t matrixElements = static_cast<uint64_t>(n_) * n_;
    const uint64_t inputBase = static_cast<uint64_t>(batch) * matrixElements;
    if (n_ <= 3U) {
        DetTiny(batch, inputBase);
        return;
    }
    if constexpr (LocalLuBand == 1U) {
        StoreDet(batch, n_ == 40U
                            ? DetSmall40UbVector(inputBase)
                            : (n_ >= 41U && n_ <= 44U ? DetPaddedUbVector(inputBase) : DetSmall<2209U>(inputBase)));
        return;
    }
    if (DetSpecializedSmall(batch, inputBase)) {
        return;
    }
    if (DetSmallRuntime(batch, inputBase)) {
        return;
    }
    DetBatchGm(batch, inputBase);
}

#endif
