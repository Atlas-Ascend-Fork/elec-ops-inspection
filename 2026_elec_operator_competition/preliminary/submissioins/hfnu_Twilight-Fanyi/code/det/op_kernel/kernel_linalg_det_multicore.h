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
#ifndef KERNEL_LINALG_DET_MULTICORE_H
#define KERNEL_LINALG_DET_MULTICORE_H

#include "kernel_linalg_det.h"

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::CompleteMultiCoreColumn(uint64_t matrixBase,
                                                                                        uint32_t pivotColumn,
                                                                                        float &determinant)
{
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
    const float pivot = workGm_.GetValue(MatrixIndex(matrixBase, pivotColumn, pivotColumn));
    determinant *= pivot;
    EliminateRowBlocksPacked(matrixBase, pivotColumn, pivot);
    AscendC::SyncAll<true>();
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::StoreMultiCoreResult(uint32_t batch, float sign,
                                                                                     float determinant)
{
    if (AscendC::GetBlockIdx() == 0U) {
        StoreDet(batch, sign * determinant);
    }
    AscendC::SyncAll<true>();
}

template <uint32_t LocalLuBand, bool Pipelined>
template <bool OwnerPivot>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetGrouped(uint32_t batch)
{
    const uint32_t owner = RowOwner();
    const uint64_t inputBase = static_cast<uint64_t>(batch) * n_ * n_;
    const uint64_t matrixBase = static_cast<uint64_t>(batch) * workspaceStride_;
    if (owner == 0U) {
        CopyMatrix(inputBase, matrixBase);
    }
    AscendC::SyncAll<true>();
    bool singular = false;
    float sign = 1.0F;
    float determinant = 1.0F;
    for (uint32_t pivotColumn = 0U; pivotColumn < n_; ++pivotColumn) {
        if (singular) {
            AscendC::SyncAll<true>();
            AscendC::SyncAll<true>();
            continue;
        }
        singular = !GroupedPivotStep<OwnerPivot>(matrixBase, pivotColumn, owner, sign, determinant);
    }
    if (owner == 0U) {
        StoreDet(batch, singular ? 0.0F : sign * determinant);
    }
    AscendC::SyncAll<true>();
}

template <uint32_t LocalLuBand, bool Pipelined>
template <bool OwnerPivot>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::GroupedPivotStep(uint64_t matrixBase,
                                                                                 uint32_t pivotColumn, uint32_t owner,
                                                                                 float &sign, float &determinant)
{
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
    uint32_t pivotRow = pivotColumn;
    float pivotAbs = 0.0F;
    if (!OwnerPivot || owner == 0U) {
        if (n_ - pivotColumn >= 16U) {
            FindPivotVector(matrixBase, pivotColumn, pivotRow, pivotAbs);
        } else {
            FindPivotScalar(matrixBase, pivotColumn, pivotRow, pivotAbs);
        }
    }
    if constexpr (!OwnerPivot) {
        if (pivotAbs == 0.0F) {
            AscendC::SyncAll<true>();
            AscendC::SyncAll<true>();
            return false;
        }
    }
    const bool swapped = pivotAbs != 0.0F && pivotRow != pivotColumn;
    if (owner == 0U && swapped) {
        SwapGmPivot(matrixBase, pivotColumn, pivotRow);
    }
    if (swapped) {
        sign = -sign;
    }
    AscendC::SyncAll<true>();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
    const float pivot = workGm_.GetValue(MatrixIndex(matrixBase, pivotColumn, pivotColumn));
    if (pivot == 0.0F) {
        AscendC::SyncAll<true>();
        return false;
    }
    if (!OwnerPivot || owner == 0U) {
        determinant *= pivot;
    }
    EliminateRowBlocksPacked(matrixBase, pivotColumn, pivot);
    AscendC::SyncAll<true>();
    return true;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::StoreTriangularDet(uint32_t batch, uint64_t inputBase)
{
    if constexpr (LocalLuBand == 13U || LocalLuBand == 14U) {
        float triangularDeterminant = 1.0F;
        const bool triangular = TryLowerTriangularDet(inputBase, triangularDeterminant);
        if (triangular) {
            if (AscendC::GetBlockIdx() == 0U)
                StoreDet(batch, triangularDeterminant);
            // StoreDet waits for the exact four-byte MTE3 output transaction.
            // No AIV consumes that output inside this kernel, and kernel
            // completion already waits for every launched AIV.  A second global
            // barrier here therefore only serialized the successful fast path;
            // the publication barrier inside TryLowerTriangularDet remains.
            return true;
        }
        // Keep every owner at the same phase boundary before the original v97
        // copy/LU fallback starts.
        AscendC::SyncAll<true>();
    }

    return false;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::FindPipelinedPivot(uint64_t matrixBase,
                                                                                   uint32_t pivotColumn,
                                                                                   uint32_t &pivotRow, float &pivotAbs)
{
    // The final scalar-only iterations publish no metadata.  Fall back to
    // the complete proven scan instead of consuming the preceding slot.
    const uint32_t previousTrailing = n_ - pivotColumn;
    const uint32_t previousPrefix = static_cast<uint32_t>((8U - (pivotColumn & 7U)) & 7U);
    const bool consumePublished =
        pivotColumn > 0U && previousPrefix < previousTrailing && ((previousTrailing - previousPrefix) & ~7U) > 0U;
    if (!consumePublished) {
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
        if (n_ - pivotColumn >= 16U)
            FindPivotVector(matrixBase, pivotColumn, pivotRow, pivotAbs);
        else
            FindPivotScalar(matrixBase, pivotColumn, pivotRow, pivotAbs);
    } else if (AscendC::GetBlockIdx() == 0U) {
        ReducePublishedPivot(pivotColumn, pivotAbs, pivotRow);
    }

    return consumePublished;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::SwapGmPivot(uint64_t matrixBase, uint32_t pivotColumn,
                                                                            uint32_t pivotRow)
{
    for (uint32_t column = pivotColumn; column < n_; ++column) {
        const uint64_t first = MatrixIndex(matrixBase, pivotColumn, column);
        const uint64_t second = MatrixIndex(matrixBase, pivotRow, column);
        const float value = workGm_.GetValue(first);
        workGm_.SetValue(first, workGm_.GetValue(second));
        workGm_.SetValue(second, value);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::StorePivotCandidate(uint32_t slot, float value,
                                                                                    uint32_t row)
{
    auto output = outputRowQue_.AllocTensor<float>();
    output.SetValue(0U, value);
    output.ReinterpretCast<uint32_t>().SetValue(1U, row);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3_);
    const AscendC::DataCopyExtParams storeParams{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    AscendC::DataCopyPad(triangularMetaGm_[static_cast<uint64_t>(slot) * 16U], output, storeParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    outputRowQue_.FreeTensor(output);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::LoadPivotCandidate(uint32_t slot, float &value,
                                                                                   uint32_t &row)
{
    const uint64_t base = static_cast<uint64_t>(slot) * 16U;
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(triangularMetaGm_[base]);
    value = triangularMetaGm_.GetValue(base);
    row = pivotMetaBitsGm_.GetValue(base + 1U);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::ReducePublishedPivot(uint32_t pivotColumn,
                                                                                     float &pivotAbs,
                                                                                     uint32_t &pivotRow)
{
    pivotAbs = -1.0F;
    pivotRow = n_;
    for (uint32_t owner = 0U; owner < RowOwners(); ++owner) {
        float candidateAbs = -1.0F;
        uint32_t candidateRow = n_;
        LoadPivotCandidate(owner, candidateAbs, candidateRow);
        if (candidateRow >= pivotColumn && candidateRow < n_ &&
            (candidateAbs > pivotAbs || (candidateAbs == pivotAbs && candidateRow < pivotRow))) {
            pivotAbs = candidateAbs;
            pivotRow = candidateRow;
        }
    }
    if (pivotRow == n_) {
        pivotAbs = 0.0F;
        pivotRow = pivotColumn;
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetBatchMultiCorePipelined(uint32_t batch)
{
    const uint64_t matrixElements = static_cast<uint64_t>(n_) * n_;
    const uint64_t inputBase = static_cast<uint64_t>(batch) * matrixElements;
    const uint64_t matrixBase = static_cast<uint64_t>(batch) * workspaceStride_;
    float triangularDeterminant = 1.0F;
    if (TryLowerTriangularDet(inputBase, triangularDeterminant)) {
        if (AscendC::GetBlockIdx() == 0U)
            StoreDet(batch, triangularDeterminant);
        return;
    }
    AscendC::SyncAll<true>();
    if (AscendC::GetBlockIdx() == 0U)
        CopyMatrix(inputBase, matrixBase);
    AscendC::SyncAll<true>();

    float sign = 1.0F;
    float determinant = 1.0F;
    for (uint32_t pivotColumn = 0U; pivotColumn < n_; ++pivotColumn) {
        uint32_t pivotRow = pivotColumn;
        float pivotAbs = 0.0F;
        const bool consumePublished = FindPipelinedPivot(matrixBase, pivotColumn, pivotRow, pivotAbs);
        if (AscendC::GetBlockIdx() == 0U && pivotAbs != 0.0F && pivotRow != pivotColumn) {
            sign = -sign;
            SwapGmPivot(matrixBase, pivotColumn, pivotRow);
        } else if (!consumePublished && pivotRow != pivotColumn) {
            sign = -sign;
        }
        if (consumePublished && AscendC::GetBlockIdx() == 0U)
            StorePivotCandidate(0U, pivotAbs, pivotRow);
        AscendC::SyncAll<true>();
        if (consumePublished && AscendC::GetBlockIdx() != 0U)
            LoadPivotCandidate(0U, pivotAbs, pivotRow);
        if (pivotAbs == 0.0F) {
            if (AscendC::GetBlockIdx() == 0U)
                StoreDet(batch, 0.0F);
            AscendC::SyncAll<true>();
            return;
        }
        CompleteMultiCoreColumn(matrixBase, pivotColumn, determinant);
    }
    StoreMultiCoreResult(batch, sign, determinant);
}

// Keys 9--12 retain the isolated checkpoints. Key 13 covers aligned large
// matrices and key 14 extends padded row ownership to every batch-one
// matrix in [128, 512].
// Selection remains
// replicated and deterministic; only block zero swaps rows. Every trailing
// 40-row block has one owner.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetBatchMultiCore(uint32_t batch)
{
    const uint64_t matrixElements = static_cast<uint64_t>(n_) * n_;
    const uint64_t inputBase = static_cast<uint64_t>(batch) * matrixElements;
    const uint64_t matrixBase = static_cast<uint64_t>(batch) * workspaceStride_;
    if (StoreTriangularDet(batch, inputBase)) {
        return;
    }
    if (AscendC::GetBlockIdx() == 0U)
        CopyMatrix(inputBase, matrixBase);
    AscendC::SyncAll<true>();

    float sign = 1.0F;
    float determinant = 1.0F;
    for (uint32_t pivotColumn = 0U; pivotColumn < n_; ++pivotColumn) {
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
        uint32_t pivotRow = pivotColumn;
        float pivotAbs = 0.0F;
        if constexpr (LocalLuBand == 13U || LocalLuBand == 14U) {
            if (n_ - pivotColumn >= 16U) {
                FindPivotVector(matrixBase, pivotColumn, pivotRow, pivotAbs);
            } else {
                FindPivotScalar(matrixBase, pivotColumn, pivotRow, pivotAbs);
            }
        } else {
            FindPivotScalar(matrixBase, pivotColumn, pivotRow, pivotAbs);
        }
        if (pivotAbs == 0.0F) {
            AscendC::SyncAll<true>();
            if (AscendC::GetBlockIdx() == 0U)
                StoreDet(batch, 0.0F);
            AscendC::SyncAll<true>();
            return;
        }
        if (AscendC::GetBlockIdx() == 0U && pivotRow != pivotColumn) {
            sign = -sign;
            SwapGmPivot(matrixBase, pivotColumn, pivotRow);
        } else if (pivotRow != pivotColumn) {
            sign = -sign;
        }
        AscendC::SyncAll<true>();
        CompleteMultiCoreColumn(matrixBase, pivotColumn, determinant);
    }
    StoreMultiCoreResult(batch, sign, determinant);
}

// Baseline grouped protocol: all owners choose the pivot before the swap.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetBatchGroupedBaseline(uint32_t batch)
{
    DetGrouped<false>(batch);
}

// Key 62 assigns one fixed-size AIV group to every matrix. Only owner zero
// searches and swaps the pivot. After the existing first barrier every
// owner reads the resulting diagonal value, so neither pivotAbs nor the
// swapped bit needs a GM metadata round trip. Determinant/sign bookkeeping
// is also owner-zero-only. The number and position of AscendC::SyncAll calls remain
// identical to key 15.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetBatchGrouped(uint32_t batch)
{
    DetGrouped<true>(batch);
}

#endif
