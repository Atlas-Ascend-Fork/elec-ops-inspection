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
#ifndef KERNEL_LINALG_DET_PACKED_H
#define KERNEL_LINALG_DET_PACKED_H

#include "kernel_linalg_det.h"

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline AscendC::LocalTensor<float> KernelLinalgDet<LocalLuBand, Pipelined>::LoadPivotRow(
    uint64_t matrixBase, uint32_t pivotColumn)
{
    const uint64_t pivotRowStart = MatrixIndex(matrixBase, pivotColumn, 0U);
    auto cachedPivot = pivotRowQue_.AllocTensor<float>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    if constexpr (LocalLuBand == 14U) {
        const AscendC::DataCopyExtParams pivotCopyParams{1U, n_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> pivotPad{true, 0U, 0U, 0.0F};
        AscendC::DataCopyPad(cachedPivot, workGm_[pivotRowStart], pivotCopyParams, pivotPad);
    } else {
        AscendC::DataCopy(cachedPivot, workGm_[pivotRowStart], n_);
    }
    pivotRowQue_.EnQue(cachedPivot);
    cachedPivot = pivotRowQue_.DeQue<float>();

    return cachedPivot;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminatePackedBlocks(
    uint64_t matrixBase, uint32_t pivotColumn, float pivot, uint32_t prefix, uint32_t vectorCount, uint32_t trailing,
    AscendC::LocalTensor<float> cachedPivot, float &nextPivotAbs, uint32_t &nextPivotRow)
{
    const uint32_t owner = LocalLuBand >= 9U ? RowOwner() : 0U;
    const uint32_t owners = LocalLuBand >= 9U ? RowOwners() : 1U;
    for (uint32_t firstRow = pivotColumn + 1U + owner * 40U; firstRow < n_; firstRow += owners * 40U) {
        uint32_t rows = 0U;
        uint32_t packedColumn = 0U;
        uint32_t packedPitch = 0U;
        uint64_t rowStart = 0U;
        AscendC::DataCopyExtParams storePackParams{};
        auto target = LoadPackedRows(matrixBase, pivotColumn, firstRow, rows, packedColumn, packedPitch, rowStart,
                                     storePackParams);
        auto output = outputRowQue_.AllocTensor<float>();

        const uint32_t pivotLocal = pivotColumn - packedColumn;
        auto factorStorage = factorTileBuf_.Get<float>();
        auto factorSource = factorStorage;
        auto factorRows = factorStorage[64U];
        BuildPackedFactors(target, factorSource, factorRows, rows, packedPitch, pivotLocal, pivot);
        UpdatePackedVectors(output, target, cachedPivot, factorRows, packedPitch, packedColumn, pivotColumn, pivotLocal,
                            prefix, vectorCount, rows);
        RepairPackedFringes(output, target, cachedPivot, factorSource, rows, packedPitch, pivotLocal, pivotColumn,
                            trailing, prefix, vectorCount);
        ProbeNextPackedPivot(output, pivotColumn, packedColumn, packedPitch, rows, firstRow, nextPivotAbs,
                             nextPivotRow);
        outputRowQue_.EnQue(output);
        output = outputRowQue_.DeQue<float>();
        AscendC::DataCopyPad(workGm_[rowStart], output, storePackParams);
        targetRowQue_.FreeTensor(target);
        outputRowQue_.FreeTensor(output);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline AscendC::LocalTensor<float> KernelLinalgDet<LocalLuBand, Pipelined>::LoadPackedRows(
    uint64_t matrixBase, uint32_t pivotColumn, uint32_t firstRow, uint32_t &rows, uint32_t &packedColumn,
    uint32_t &packedPitch, uint64_t &rowStart, AscendC::DataCopyExtParams &storePackParams)
{
    rows = n_ - firstRow < 40U ? n_ - firstRow : 40U;
    // Pack one aligned active suffix from every row with a single 2-D DMA.
    // Unlike v19's row-by-row suffix transfers this keeps one command per
    // 40-row block, while the transferred bytes shrink after each 8-column
    // boundary.  Columns through pivotColumn are dead after this step.
    packedColumn = pivotColumn & ~7U;
    const uint32_t packedWidth = n_ - packedColumn;
    packedPitch = LocalLuBand == 14U ? ((packedWidth + 7U) & ~7U) : packedWidth;
    rowStart = MatrixIndex(matrixBase, firstRow, packedColumn);
    const uint32_t rowGapBytes = static_cast<uint32_t>(packedColumn * sizeof(float));
    // AscendC::DataCopyExtParams strides use bytes on the GM side but 32-byte
    // DataBlocks on the UB side. One FP32 DataBlock contains eight values.
    const uint32_t localGapBlocks = (packedPitch - packedWidth) / 8U;
    const AscendC::DataCopyExtParams loadPackParams{
        static_cast<uint16_t>(rows), static_cast<uint32_t>(packedWidth * sizeof(float)), rowGapBytes, 0U, 0U};
    storePackParams =
        AscendC::DataCopyExtParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(packedWidth * sizeof(float)),
                                   localGapBlocks, rowGapBytes, 0U};
    const AscendC::DataCopyPadExtParams<float> noPad{true, 0U, 0U, 0.0F};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    auto target = targetRowQue_.AllocTensor<float>();
    AscendC::DataCopyPad(target, workGm_[rowStart], loadPackParams, noPad);
    targetRowQue_.EnQue(target);
    target = targetRowQue_.DeQue<float>();

    return target;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::ProbeNextPackedPivot(
    AscendC::LocalTensor<float> output, uint32_t pivotColumn, uint32_t packedColumn, uint32_t packedPitch,
    uint32_t rows, uint32_t firstRow, float &nextPivotAbs, uint32_t &nextPivotRow)
{
    if constexpr (Pipelined) {
        if (pivotColumn + 1U < n_) {
            auto candidateScratch = factorTileBuf_.Get<float>()[64U];
            auto signedOffsets = candidateScratch.ReinterpretCast<int32_t>();
            auto offsets = signedOffsets.ReinterpretCast<uint32_t>();
            auto values = candidateScratch[64U];
            auto reduceTmp = candidateScratch[128U];
            auto reduceResult = candidateScratch[640U];
            const uint32_t nextLocal = pivotColumn + 1U - packedColumn;
            AscendC::ArithProgression<int32_t>(signedOffsets, 0, static_cast<int32_t>(packedPitch * sizeof(float)),
                                               rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather<float>(values, output[nextLocal], offsets, 0U, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs<float>(values, values, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(reduceResult, values, reduceTmp, static_cast<int32_t>(rows), true);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar_);
            const float candidateAbs = reduceResult.GetValue(0U);
            const uint32_t candidateIndex = reduceResult.ReinterpretCast<uint32_t>().GetValue(1U);
            const uint32_t candidateRow = firstRow + candidateIndex;
            if (candidateAbs > nextPivotAbs || (candidateAbs == nextPivotAbs && candidateRow < nextPivotRow)) {
                nextPivotAbs = candidateAbs;
                nextPivotRow = candidateRow;
            }
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::RepairPackedFringes(
    AscendC::LocalTensor<float> output, AscendC::LocalTensor<float> target, AscendC::LocalTensor<float> cachedPivot,
    AscendC::LocalTensor<float> factorSource, uint32_t rows, uint32_t packedPitch, uint32_t pivotLocal,
    uint32_t pivotColumn, uint32_t trailing, uint32_t prefix, uint32_t vectorCount)
{
    const uint32_t tail = trailing - prefix - vectorCount;
    if constexpr (LocalLuBand == 13U || LocalLuBand == 15U || LocalLuBand == 62U) {
        // keys 13/15 are selected only for aligned n, making tail zero. Keep a
        // defensive tail repair without reintroducing the now-vectorized
        // prefix loop in case the host contract is changed later.
        if (tail > 0U) {
            for (uint32_t localRow = 0U; localRow < rows; ++localRow) {
                const uint32_t localStart = localRow * packedPitch + pivotLocal + 1U;
                const uint32_t pivotStart = pivotColumn + 1U;
                const float negativeFactor = factorSource.GetValue(localRow);
                for (uint32_t index = 0U; index < tail; ++index) {
                    const uint32_t offset = prefix + vectorCount + index;
                    output.SetValue(localStart + offset,
                                    target.GetValue(localStart + offset) +
                                        negativeFactor * cachedPivot.GetValue(pivotStart + offset));
                }
            }
        }
    } else {
        for (uint32_t localRow = 0U; localRow < rows; ++localRow) {
            const uint32_t localStart = localRow * packedPitch + pivotLocal + 1U;
            const uint32_t pivotStart = pivotColumn + 1U;
            const float negativeFactor = factorSource.GetValue(localRow);
            for (uint32_t index = 0U; index < prefix; ++index)
                output.SetValue(localStart + index, target.GetValue(localStart + index) +
                                                        negativeFactor * cachedPivot.GetValue(pivotStart + index));
            for (uint32_t index = 0U; index < tail; ++index) {
                const uint32_t offset = prefix + vectorCount + index;
                output.SetValue(localStart + offset, target.GetValue(localStart + offset) +
                                                         negativeFactor * cachedPivot.GetValue(pivotStart + offset));
            }
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::UpdatePackedVectors(
    AscendC::LocalTensor<float> output, AscendC::LocalTensor<float> target, AscendC::LocalTensor<float> cachedPivot,
    AscendC::LocalTensor<float> factorRows, uint32_t packedPitch, uint32_t packedColumn, uint32_t pivotColumn,
    uint32_t pivotLocal, uint32_t prefix, uint32_t vectorCount, uint32_t rows)
{
    const uint8_t packedStride = static_cast<uint8_t>(packedPitch / 8U);
    const AscendC::BinaryRepeatParams mulRows{1U, 1U, 1U, packedStride, 8U, 0U};
    const AscendC::BinaryRepeatParams addRows{1U, 1U, 1U, packedStride, packedStride, packedStride};
    uint32_t vectorLocalStart = pivotLocal + 1U + prefix;
    uint32_t vectorPivotStart = pivotColumn + 1U + prefix;
    uint32_t vectorWorkCount = vectorCount;
    if constexpr (LocalLuBand == 13U || LocalLuBand == 15U || LocalLuBand == 62U) {
        if (prefix > 0U) {
            // The packed row starts at an aligned column.  Columns through the
            // current pivot are dead after factor extraction, so updating the
            // complete first 32-byte block is semantically harmless and removes
            // up to seven scalar prefix repairs per row.
            vectorLocalStart = 0U;
            vectorPivotStart = packedColumn;
            vectorWorkCount += 8U;
        }
    }
    for (uint32_t tile = 0U; tile < vectorWorkCount; tile += 64U) {
        const uint32_t tileWidth = vectorWorkCount - tile < 64U ? vectorWorkCount - tile : 64U;
        const uint32_t localOffset = vectorLocalStart + tile;
        AscendC::Mul<float>(output[localOffset], factorRows, cachedPivot[vectorPivotStart + tile], tileWidth,
                            static_cast<uint8_t>(rows), mulRows);
    }
    // Column tiles are disjoint. Let every outer-product AscendC::Mul enter the
    // vector pipeline before one dependency boundary, then issue every AscendC::Add
    // before the second boundary. This replaces up to sixteen barriers per
    // row block with two without changing data dependencies.
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t tile = 0U; tile < vectorWorkCount; tile += 64U) {
        const uint32_t tileWidth = vectorWorkCount - tile < 64U ? vectorWorkCount - tile : 64U;
        const uint32_t localOffset = vectorLocalStart + tile;
        AscendC::Add<float>(output[localOffset], output[localOffset], target[localOffset], tileWidth,
                            static_cast<uint8_t>(rows), addRows);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ResetMask();
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::BuildPackedFactors(
    AscendC::LocalTensor<float> target, AscendC::LocalTensor<float> factorSource,
    AscendC::LocalTensor<float> factorRows, uint32_t rows, uint32_t packedPitch, uint32_t pivotLocal, float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    if constexpr (LocalLuBand == 13U || LocalLuBand == 15U || LocalLuBand == 62U) {
        // factorRows is dead until AscendC::Brcb below.  Reuse its first 64 words for
        // byte offsets so this key13/key15 optimization adds no AscendC::TBuf member or
        // UB allocation to the other template instances.
        auto signedFactorOffsets = factorRows.ReinterpretCast<int32_t>();
        auto factorOffsets = signedFactorOffsets.ReinterpretCast<uint32_t>();
        AscendC::ArithProgression<int32_t>(signedFactorOffsets, static_cast<int32_t>(pivotLocal * sizeof(float)),
                                           static_cast<int32_t>(packedPitch * sizeof(float)), rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather<float>(factorSource, target, factorOffsets, 0U, rows);
        AscendC::PipeBarrier<PIPE_V>();
        // Preserve v80's operation order: negate the numerator, then divide
        // directly by the pivot.  A reciprocal-multiply was rejected because
        // the reciprocal can overflow first on near-singular matrices.
        AscendC::Muls<float>(factorSource, factorSource, -1.0F, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<float>(factorRows, pivot, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float>(factorSource, factorSource, factorRows, rows);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        for (uint32_t localRow = 0U; localRow < rows; ++localRow) {
            factorSource.SetValue(localRow, -target.GetValue(localRow * packedPitch + pivotLocal) / pivot);
        }
    }
    // AscendC::Brcb turns eight scalar factors into eight 32-byte blocks. Leave
    // 64 floats between rows, then copy each first block across its row.
    // The resulting [rows, 64] tile lets one advanced-repeat AscendC::Mul service
    // the whole row block instead of issuing one AscendC::Muls command per row.
    const uint32_t paddedRows = (rows + 7U) & ~7U;
    for (uint32_t localRow = rows; localRow < paddedRows; ++localRow) factorSource.SetValue(localRow, 0.0F);
    AscendC::Brcb<float>(factorRows, factorSource, static_cast<uint8_t>(paddedRows / 8U),
                         AscendC::BrcbRepeatParams(8U, 64U));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float>(factorRows, factorRows, 0.0F, 64U, static_cast<uint8_t>(rows),
                         AscendC::UnaryRepeatParams(1U, 0U, 8U, 8U));
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateRowBlocksPacked(uint64_t matrixBase,
                                                                                         uint32_t pivotColumn,
                                                                                         float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    const uint32_t trailing = n_ - pivotColumn - 1U;
    const uint32_t prefix = static_cast<uint32_t>((8U - ((pivotColumn + 1U) & 7U)) & 7U);
    if (prefix >= trailing) {
        if constexpr (LocalLuBand >= 9U) {
            EliminateScalarRowsPartitioned(matrixBase, pivotColumn, pivot);
            return;
        }
        EliminateRowsWithCachedPivot(matrixBase, pivotColumn, pivot);
        return;
    }
    const uint32_t vectorCount = (trailing - prefix) & ~7U;
    if (vectorCount == 0U) {
        if constexpr (LocalLuBand >= 9U) {
            EliminateScalarRowsPartitioned(matrixBase, pivotColumn, pivot);
            return;
        }
        EliminateRowsWithCachedPivot(matrixBase, pivotColumn, pivot);
        return;
    }

    auto cachedPivot = LoadPivotRow(matrixBase, pivotColumn);
    float nextPivotAbs = -1.0F;
    uint32_t nextPivotRow = n_;
    EliminatePackedBlocks(matrixBase, pivotColumn, pivot, prefix, vectorCount, trailing, cachedPivot, nextPivotAbs,
                          nextPivotRow);
    pivotRowQue_.FreeTensor(cachedPivot);
    if constexpr (Pipelined) {
        if (pivotColumn + 1U < n_)
            StorePivotCandidate(AscendC::GetBlockIdx(), nextPivotAbs, nextPivotRow);
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateScalarRowsPartitioned(uint64_t matrixBase,
                                                                                               uint32_t pivotColumn,
                                                                                               float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    const uint32_t blockCount = RowOwners();
    for (uint32_t row = pivotColumn + 1U + RowOwner(); row < n_; row += blockCount) {
        const uint64_t factorIndex = MatrixIndex(matrixBase, row, pivotColumn);
        const float factor = workGm_.GetValue(factorIndex) / pivot;
        const uint64_t targetStart = MatrixIndex(matrixBase, row, pivotColumn + 1U);
        const uint64_t pivotStart = MatrixIndex(matrixBase, pivotColumn, pivotColumn + 1U);
        for (uint32_t index = 0U; index < n_ - pivotColumn - 1U; ++index)
            workGm_.SetValue(targetStart + index,
                             workGm_.GetValue(targetStart + index) - factor * workGm_.GetValue(pivotStart + index));
    }
}

#endif
