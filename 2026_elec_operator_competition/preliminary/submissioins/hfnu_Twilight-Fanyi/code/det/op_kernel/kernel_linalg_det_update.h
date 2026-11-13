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
#ifndef KERNEL_LINALG_DET_UPDATE_H
#define KERNEL_LINALG_DET_UPDATE_H

#include "kernel_linalg_det.h"

template <uint32_t LocalLuBand, bool Pipelined>
template <bool Padded>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateRangeSuffix(uint64_t targetOffset,
                                                                                  uint64_t pivotOffset, uint32_t count,
                                                                                  float factor)
{
    const uint32_t vectorCount = count & ~7U;
    if (vectorCount > 0U) {
        UpdateRangeVector<Padded>(targetOffset, pivotOffset, vectorCount, factor);
        targetOffset += vectorCount;
        pivotOffset += vectorCount;
        count -= vectorCount;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    }
    while (count > 0U) {
        workGm_.SetValue(targetOffset, workGm_.GetValue(targetOffset) - factor * workGm_.GetValue(pivotOffset));
        ++targetOffset;
        ++pivotOffset;
        --count;
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateGmRows(uint64_t matrixBase,
                                                                                uint32_t pivotColumn, float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    for (uint32_t row = pivotColumn + 1U; row < n_; ++row) {
        const uint64_t factorIndex = MatrixIndex(matrixBase, row, pivotColumn);
        const float factor = workGm_.GetValue(factorIndex) / pivot;
        workGm_.SetValue(factorIndex, factor);
        UpdateRange(MatrixIndex(matrixBase, row, pivotColumn + 1U),
                    MatrixIndex(matrixBase, pivotColumn, pivotColumn + 1U), n_ - pivotColumn - 1U, factor);
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
template <bool Padded>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateRangeVector(uint64_t targetOffset,
                                                                                  uint64_t pivotOffset,
                                                                                  uint32_t vectorCount, float factor)
{
    auto target = targetRowQue_.AllocTensor<float>();
    auto pivot = pivotRowQue_.AllocTensor<float>();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(workGm_[targetOffset]);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    if constexpr (Padded) {
        const AscendC::DataCopyExtParams params{1U, vectorCount * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> padding{true, 0U, 0U, 0.0F};
        AscendC::DataCopyPad(target, workGm_[targetOffset], params, padding);
        AscendC::DataCopyPad(pivot, workGm_[pivotOffset], params, padding);
    } else {
        AscendC::DataCopy(target, workGm_[targetOffset], vectorCount);
        AscendC::DataCopy(pivot, workGm_[pivotOffset], vectorCount);
    }
    targetRowQue_.EnQue(target);
    pivotRowQue_.EnQue(pivot);
    target = targetRowQue_.DeQue<float>();
    pivot = pivotRowQue_.DeQue<float>();
    auto output = outputRowQue_.AllocTensor<float>();
    AscendC::Muls(output, pivot, -factor, vectorCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(output, output, target, vectorCount);
    outputRowQue_.EnQue(output);
    output = outputRowQue_.DeQue<float>();
    if constexpr (Padded) {
        const AscendC::DataCopyExtParams params{1U, vectorCount * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        AscendC::DataCopyPad(workGm_[targetOffset], output, params);
    } else {
        AscendC::DataCopy(workGm_[targetOffset], output, vectorCount);
    }
    targetRowQue_.FreeTensor(target);
    pivotRowQue_.FreeTensor(pivot);
    outputRowQue_.FreeTensor(output);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateAlignedBlock(
    AscendC::LocalTensor<float> output, AscendC::LocalTensor<float> target, AscendC::LocalTensor<float> cachedPivot,
    const float *factors, uint32_t rows, uint32_t pivotColumn, uint32_t prefix, uint32_t vectorCount, uint32_t trailing)
{
    for (uint32_t localRow = 0U; localRow < rows; ++localRow) {
        const uint32_t localOffset = localRow * n_ + pivotColumn + 1U + prefix;
        AscendC::Muls(output[localOffset], cachedPivot[pivotColumn + 1U + prefix], -factors[localRow], vectorCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t localRow = 0U; localRow < rows; ++localRow) {
        const uint32_t localOffset = localRow * n_ + pivotColumn + 1U + prefix;
        AscendC::Add(output[localOffset], output[localOffset], target[localOffset], vectorCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t tail = trailing - prefix - vectorCount;
    for (uint32_t localRow = 0U; localRow < rows; ++localRow) {
        const uint32_t localStart = localRow * n_ + pivotColumn + 1U;
        const uint32_t pivotStart = pivotColumn + 1U;
        for (uint32_t index = 0U; index < prefix; ++index)
            output.SetValue(localStart + index, target.GetValue(localStart + index) -
                                                    factors[localRow] * cachedPivot.GetValue(pivotStart + index));
        for (uint32_t index = 0U; index < tail; ++index) {
            const uint32_t offset = prefix + vectorCount + index;
            output.SetValue(localStart + offset, target.GetValue(localStart + offset) -
                                                     factors[localRow] * cachedPivot.GetValue(pivotStart + offset));
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateCachedVector(
    uint64_t targetStart, uint32_t vectorCount, float factor, AscendC::LocalTensor<float> cachedPivot)
{
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(workGm_[targetStart]);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    auto target = targetRowQue_.AllocTensor<float>();
    AscendC::DataCopy(target, workGm_[targetStart], vectorCount);
    targetRowQue_.EnQue(target);
    target = targetRowQue_.DeQue<float>();
    auto output = outputRowQue_.AllocTensor<float>();
    AscendC::Muls(output, cachedPivot, -factor, vectorCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(output, output, target, vectorCount);
    outputRowQue_.EnQue(output);
    output = outputRowQue_.DeQue<float>();
    AscendC::DataCopy(workGm_[targetStart], output, vectorCount);
    targetRowQue_.FreeTensor(target);
    outputRowQue_.FreeTensor(output);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateCachedAlignedRow(
    uint64_t matrixBase, uint32_t pivotColumn, float pivot, uint32_t row, uint32_t count, uint32_t vectorCount,
    uint64_t pivotStart, AscendC::LocalTensor<float> cachedPivot, uint32_t prefix)
{
    if (pivot == 0.0F) {
        return;
    }
    const uint64_t factorIndex = MatrixIndex(matrixBase, row, pivotColumn);
    const float factor = workGm_.GetValue(factorIndex) / pivot;
    workGm_.SetValue(factorIndex, factor);
    uint64_t targetStart = MatrixIndex(matrixBase, row, pivotColumn + 1U);
    for (uint32_t index = 0U; index < prefix; ++index) {
        workGm_.SetValue(targetStart + index,
                         workGm_.GetValue(targetStart + index) - factor * workGm_.GetValue(pivotStart + index));
    }
    targetStart += prefix;
    UpdateCachedVector(targetStart, vectorCount, factor, cachedPivot);
    const uint32_t tail = count - prefix - vectorCount;
    for (uint32_t index = 0U; index < tail; ++index) {
        const uint64_t targetIndex = targetStart + vectorCount + index;
        const uint64_t pivotIndex = pivotStart + prefix + vectorCount + index;
        workGm_.SetValue(targetIndex, workGm_.GetValue(targetIndex) - factor * workGm_.GetValue(pivotIndex));
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateCachedPaddedRow(
    uint64_t matrixBase, uint32_t pivotColumn, float pivot, uint32_t row, uint32_t count, uint32_t vectorCount,
    uint64_t pivotStart, AscendC::LocalTensor<float> cachedPivot)
{
    const AscendC::DataCopyExtParams copyParams{1U, vectorCount * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0.0F};
    if (pivot == 0.0F) {
        return;
    }
    const uint64_t factorIndex = MatrixIndex(matrixBase, row, pivotColumn);
    const float factor = workGm_.GetValue(factorIndex) / pivot;
    workGm_.SetValue(factorIndex, factor);
    const uint64_t targetStart = MatrixIndex(matrixBase, row, pivotColumn + 1U);
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(workGm_[targetStart]);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    auto target = targetRowQue_.AllocTensor<float>();
    AscendC::DataCopyPad(target, workGm_[targetStart], copyParams, padParams);
    targetRowQue_.EnQue(target);
    target = targetRowQue_.DeQue<float>();
    auto output = outputRowQue_.AllocTensor<float>();
    AscendC::Muls(output, cachedPivot, -factor, vectorCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(output, output, target, vectorCount);
    outputRowQue_.EnQue(output);
    output = outputRowQue_.DeQue<float>();
    AscendC::DataCopyPad(workGm_[targetStart], output, copyParams);
    targetRowQue_.FreeTensor(target);
    outputRowQue_.FreeTensor(output);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    for (uint32_t index = vectorCount; index < count; ++index) {
        const uint64_t targetIndex = targetStart + index;
        workGm_.SetValue(targetIndex, workGm_.GetValue(targetIndex) - factor * workGm_.GetValue(pivotStart + index));
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateRange(uint64_t targetOffset, uint64_t pivotOffset,
                                                                            uint32_t count, float factor)
{
    while (count > 0U && ((targetOffset & 7ULL) != 0ULL || (pivotOffset & 7ULL) != 0ULL)) {
        workGm_.SetValue(targetOffset, workGm_.GetValue(targetOffset) - factor * workGm_.GetValue(pivotOffset));
        ++targetOffset;
        ++pivotOffset;
        --count;
    }
    UpdateRangeSuffix<false>(targetOffset, pivotOffset, count, factor);
}

// The generic non-8-aligned path cannot use ordinary AscendC::DataCopy because the
// target and pivot GM addresses never become aligned together. AscendC::DataCopyPad
// is the A2-supported unaligned transport primitive; keep all visibility
// events from UpdateRange and only change the transfer operation.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdateRangePadded(uint64_t targetOffset,
                                                                                  uint64_t pivotOffset, uint32_t count,
                                                                                  float factor)
{
    UpdateRangeSuffix<true>(targetOffset, pivotOffset, count, factor);
}

// The v57 padded route removes scalar tail traffic but still reloads the
// same pivot suffix once for every target row.  For the isolated n>=89
// specialization, retain one padded UB copy for the pivot column and reuse
// it across all row updates.  The scalar tail remains explicitly ordered
// after MTE3 completion just as in UpdateRangePadded.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateRowsWithCachedPaddedPivot(uint64_t matrixBase,
                                                                                                   uint32_t pivotColumn,
                                                                                                   float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    const uint32_t count = n_ - pivotColumn - 1U;
    const uint32_t vectorCount = count & ~7U;
    const uint64_t pivotStart = MatrixIndex(matrixBase, pivotColumn, pivotColumn + 1U);
    if (vectorCount == 0U) {
        for (uint32_t row = pivotColumn + 1U; row < n_; ++row) {
            const uint64_t factorIndex = MatrixIndex(matrixBase, row, pivotColumn);
            const float factor = workGm_.GetValue(factorIndex) / pivot;
            workGm_.SetValue(factorIndex, factor);
            UpdateRangePadded(MatrixIndex(matrixBase, row, pivotColumn + 1U), pivotStart, count, factor);
        }
        return;
    }

    auto cachedPivot = pivotRowQue_.AllocTensor<float>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    const AscendC::DataCopyExtParams copyParams{1U, vectorCount * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0.0F};
    AscendC::DataCopyPad(cachedPivot, workGm_[pivotStart], copyParams, padParams);
    pivotRowQue_.EnQue(cachedPivot);
    cachedPivot = pivotRowQue_.DeQue<float>();

    for (uint32_t row = pivotColumn + 1U; row < n_; ++row) {
        UpdateCachedPaddedRow(matrixBase, pivotColumn, pivot, row, count, vectorCount, pivotStart, cachedPivot);
    }
    pivotRowQue_.FreeTensor(cachedPivot);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateRowsWithCachedPivot(uint64_t matrixBase,
                                                                                             uint32_t pivotColumn,
                                                                                             float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    const uint32_t count = n_ - pivotColumn - 1U;
    const uint64_t pivotStart = MatrixIndex(matrixBase, pivotColumn, pivotColumn + 1U);
    const uint32_t prefix = static_cast<uint32_t>((8ULL - (pivotStart & 7ULL)) & 7ULL);
    if (prefix >= count) {
        for (uint32_t row = pivotColumn + 1U; row < n_; ++row) {
            const uint64_t factorIndex = MatrixIndex(matrixBase, row, pivotColumn);
            const float factor = workGm_.GetValue(factorIndex) / pivot;
            workGm_.SetValue(factorIndex, factor);
            UpdateRange(MatrixIndex(matrixBase, row, pivotColumn + 1U), pivotStart, count, factor);
        }
        return;
    }
    const uint32_t vectorCount = (count - prefix) & ~7U;
    if (vectorCount == 0U) {
        for (uint32_t row = pivotColumn + 1U; row < n_; ++row) {
            const uint64_t factorIndex = MatrixIndex(matrixBase, row, pivotColumn);
            const float factor = workGm_.GetValue(factorIndex) / pivot;
            workGm_.SetValue(factorIndex, factor);
            UpdateRange(MatrixIndex(matrixBase, row, pivotColumn + 1U), pivotStart, count, factor);
        }
        return;
    }

    auto cachedPivot = pivotRowQue_.AllocTensor<float>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::DataCopy(cachedPivot, workGm_[pivotStart + prefix], vectorCount);
    pivotRowQue_.EnQue(cachedPivot);
    cachedPivot = pivotRowQue_.DeQue<float>();

    for (uint32_t row = pivotColumn + 1U; row < n_; ++row) {
        UpdateCachedAlignedRow(matrixBase, pivotColumn, pivot, row, count, vectorCount, pivotStart, cachedPivot,
                               prefix);
    }
    pivotRowQue_.FreeTensor(cachedPivot);
}

// Preserve v59's coalesced full-row path for aligned n<128, where the
// fixed cost of strided 2-D DMA is larger than the saved late-panel bytes.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateRowBlocks(uint64_t matrixBase,
                                                                                   uint32_t pivotColumn, float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    const uint32_t trailing = n_ - pivotColumn - 1U;
    const uint32_t prefix = static_cast<uint32_t>((8U - ((pivotColumn + 1U) & 7U)) & 7U);
    if (prefix >= trailing) {
        EliminateRowsWithCachedPivot(matrixBase, pivotColumn, pivot);
        return;
    }
    const uint32_t vectorCount = (trailing - prefix) & ~7U;
    if (vectorCount == 0U) {
        EliminateRowsWithCachedPivot(matrixBase, pivotColumn, pivot);
        return;
    }

    auto cachedPivot = LoadPivotRow(matrixBase, pivotColumn);

    for (uint32_t firstRow = pivotColumn + 1U; firstRow < n_; firstRow += 40U) {
        const uint32_t rows = n_ - firstRow < 40U ? n_ - firstRow : 40U;
        const uint32_t blockElements = rows * n_;
        const uint64_t rowStart = MatrixIndex(matrixBase, firstRow, 0U);
        float factors[40];
        for (uint32_t localRow = 0U; localRow < rows; ++localRow)
            factors[localRow] = workGm_.GetValue(rowStart + static_cast<uint64_t>(localRow) * n_ + pivotColumn) / pivot;

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
        auto target = targetRowQue_.AllocTensor<float>();
        AscendC::DataCopy(target, workGm_[rowStart], blockElements);
        targetRowQue_.EnQue(target);
        target = targetRowQue_.DeQue<float>();
        auto output = outputRowQue_.AllocTensor<float>();
        UpdateAlignedBlock(output, target, cachedPivot, factors, rows, pivotColumn, prefix, vectorCount, trailing);
        outputRowQue_.EnQue(output);
        output = outputRowQue_.DeQue<float>();
        AscendC::DataCopy(workGm_[rowStart], output, blockElements);
        targetRowQue_.FreeTensor(target);
        outputRowQue_.FreeTensor(output);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }
    pivotRowQue_.FreeTensor(cachedPivot);
}

#endif
