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
#ifndef KERNEL_LINALG_DET_LOCAL_H
#define KERNEL_LINALG_DET_LOCAL_H

#include "kernel_linalg_det.h"

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline AscendC::LocalTensor<float> KernelLinalgDet<LocalLuBand, Pipelined>::LoadTriangularRectangle(
    uint64_t inputBase, uint32_t rowStart, uint32_t rowCount, uint32_t rectangleStart, uint32_t rectangleWidth)
{
    auto slab = targetRowQue_.AllocTensor<float>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    const AscendC::DataCopyExtParams loadParams{static_cast<uint16_t>(rowCount),
                                                rectangleWidth * static_cast<uint32_t>(sizeof(float)),
                                                rectangleStart * static_cast<uint32_t>(sizeof(float)), 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0.0F};
    AscendC::DataCopyPad(slab, aGm_[inputBase + static_cast<uint64_t>(rowStart) * n_ + rectangleStart], loadParams,
                         padParams);
    targetRowQue_.EnQue(slab);
    slab = targetRowQue_.DeQue<float>();

    return slab;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::HasNonzeroMaximums(
    AscendC::LocalTensor<float> rowMaximums, uint32_t resultCount)
{
    bool found = false;
    if (resultCount > 0U) {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar_);
        for (uint32_t index = 0U; index < resultCount; ++index) {
            if (rowMaximums.GetValue(index * 8U) != 0.0F) {
                found = true;
            }
        }
    }
    return found;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::SwapUbPivot(AscendC::LocalTensor<float> matrix,
                                                                            uint32_t n, uint32_t pitch,
                                                                            uint32_t pivotColumn, uint32_t pivotRow,
                                                                            float &sign)
{
    if (pivotRow != pivotColumn) {
        sign = -sign;
        for (uint32_t column = pivotColumn; column < n; ++column) {
            const uint32_t first = pivotColumn * pitch + column;
            const uint32_t second = pivotRow * pitch + column;
            const float value = matrix.GetValue(first);
            matrix.SetValue(first, matrix.GetValue(second));
            matrix.SetValue(second, value);
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::FindUbPivot(AscendC::LocalTensor<float> matrix,
                                                                            uint32_t n, uint32_t pitch,
                                                                            uint32_t pivotColumn, uint32_t &pivotRow,
                                                                            float &pivotAbs)
{
    pivotRow = pivotColumn;
    pivotAbs = Abs(matrix.GetValue(pivotColumn * pitch + pivotColumn));
    for (uint32_t row = pivotColumn + 1U; row < n; ++row) {
        const float candidate = Abs(matrix.GetValue(row * pitch + pivotColumn));
        if (candidate > pivotAbs) {
            pivotAbs = candidate;
            pivotRow = row;
        }
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::EliminateUbRows(AscendC::LocalTensor<float> matrix,
                                                                                AscendC::LocalTensor<float> update,
                                                                                uint32_t n, uint32_t pitch,
                                                                                uint32_t pivotColumn, float pivot)
{
    if (pivot == 0.0F) {
        return;
    }
    const uint32_t trailing = n - pivotColumn - 1U;
    uint32_t prefix = (8U - ((pivotColumn + 1U) & 7U)) & 7U;
    if (prefix > trailing)
        prefix = trailing;
    const uint32_t vectorCount = (trailing - prefix) & ~7U;
    for (uint32_t row = pivotColumn + 1U; row < n; ++row) {
        const uint32_t factorIndex = row * pitch + pivotColumn;
        const float factor = matrix.GetValue(factorIndex) / pivot;
        matrix.SetValue(factorIndex, factor);
        const uint32_t targetStart = factorIndex + 1U;
        const uint32_t pivotStart = pivotColumn * pitch + pivotColumn + 1U;
        for (uint32_t index = 0U; index < prefix; ++index)
            matrix.SetValue(targetStart + index,
                            matrix.GetValue(targetStart + index) - factor * matrix.GetValue(pivotStart + index));
        if (vectorCount > 0U) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVector_);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVector_);
            AscendC::Muls(update, matrix[pivotStart + prefix], -factor, vectorCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(matrix[targetStart + prefix], matrix[targetStart + prefix], update, vectorCount);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar_);
        }
        for (uint32_t index = prefix + vectorCount; index < trailing; ++index)
            matrix.SetValue(targetStart + index,
                            matrix.GetValue(targetStart + index) - factor * matrix.GetValue(pivotStart + index));
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::ProbeUbTriangular(AscendC::LocalTensor<float> matrix,
                                                                                  uint32_t n, uint32_t pitch,
                                                                                  float &triangularDeterminant)
{
    return ProbeTriangular(matrix, n, pitch, triangularDeterminant);
}

// Publish through an exact eight-byte MTE3 transaction into a private
// cache-line slot. Scalar GM stores are deliberately avoided because two
// AIVs may otherwise update different words of the same cache line.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::StoreTriangularProbe(uint32_t owner, float value,
                                                                                     float diagonalProduct)
{
    auto output = outputRowQue_.AllocTensor<float>();
    output.SetValue(0U, value);
    output.SetValue(1U, diagonalProduct);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3_);
    const AscendC::DataCopyExtParams storeParams{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    AscendC::DataCopyPad(triangularMetaGm_[static_cast<uint64_t>(owner) * 16U], output, storeParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    outputRowQue_.FreeTensor(output);
}

// Exact v114 fallback for the only case where an aligned rectangle cannot
// begin at or before rowStart. It uses one flat DMA and never depends on the
// per-block destination layout of AscendC::DataCopyPad.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::ScanTriangularFullRowSlab(uint64_t inputBase,
                                                                                          uint32_t rowStart,
                                                                                          uint32_t rowEnd,
                                                                                          float &diagonalProduct)
{
    const uint32_t rowCount = rowEnd - rowStart;
    auto slab = targetRowQue_.AllocTensor<float>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    const AscendC::DataCopyExtParams loadParams{1U, rowCount * n_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0.0F};
    AscendC::DataCopyPad(slab, aGm_[inputBase + static_cast<uint64_t>(rowStart) * n_], loadParams, padParams);
    targetRowQue_.EnQue(slab);
    slab = targetRowQue_.DeQue<float>();

    bool found = false;
    uint32_t resultCount = 0U;
    auto scratch = factorTileBuf_.Get<float>();
    auto values = scratch;
    auto reduceTmp = scratch[512U];
    auto rowMaximums = scratch[1024U];
    for (uint32_t row = rowStart; row < rowEnd && !found; ++row) {
        const uint32_t localBase = (row - rowStart) * n_;
        diagonalProduct *= slab.GetValue(localBase + row);
        uint32_t source = localBase + row + 1U;
        uint32_t upperCount = n_ - row - 1U;
        uint32_t prefix = (8U - (source & 7U)) & 7U;
        if (prefix > upperCount)
            prefix = upperCount;
        for (uint32_t index = 0U; index < prefix; ++index)
            if (slab.GetValue(source + index) != 0.0F)
                found = true;
        source += prefix;
        upperCount -= prefix;
        if (!found && upperCount > 0U) {
            AscendC::Abs<float>(values, slab[source], upperCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(rowMaximums[resultCount * 8U], values, reduceTmp,
                                      static_cast<int32_t>(upperCount), true);
            ++resultCount;
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    found = HasNonzeroMaximums(rowMaximums, resultCount) || found;
    targetRowQue_.FreeTensor(slab);
    return found;
}

// Load one naturally 32-byte-wide rectangle. Later
// rows contain a short diagonal/lower prefix; clear that prefix in UB and
// reduce the packed rectangle. This preserves v114's one DMA per slab while
// avoiding both complete-row traffic and scalar GM fringe reads.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::ScanTriangularRowSlab(uint64_t inputBase,
                                                                                      uint32_t rowStart,
                                                                                      uint32_t rowEnd,
                                                                                      float &diagonalProduct)
{
    if (rowStart >= rowEnd)
        return false;
    const uint32_t rowCount = rowEnd - rowStart;
    const uint32_t nResidue = n_ & 7U;
    const uint32_t alignBack = (rowStart + 8U - nResidue) & 7U;
    if (rowStart < alignBack)
        return ScanTriangularFullRowSlab(inputBase, rowStart, rowEnd, diagonalProduct);
    const uint32_t rectangleStart = rowStart - alignBack;
    const uint32_t rectangleWidth = n_ - rectangleStart;
    const uint32_t rectanglePitch = rectangleWidth;
    auto slab = LoadTriangularRectangle(inputBase, rowStart, rowCount, rectangleStart, rectangleWidth);
    // Preserve each diagonal value, then mask the diagonal/lower prefix. Every
    // UB row begins on a 32-byte boundary and the prefix is at most 40 values.
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        const uint32_t localBase = (row - rowStart) * rectanglePitch;
        const uint32_t diagonalOffset = row - rectangleStart;
        diagonalProduct *= slab.GetValue(localBase + diagonalOffset);
        AscendC::Duplicate(slab[localBase], 0.0F, diagonalOffset + 1U);
    }
    AscendC::PipeBarrier<PIPE_V>();

    bool found = false;
    uint32_t resultCount = 0U;
    auto scratch = factorTileBuf_.Get<float>();
    auto values = scratch;
    auto reduceTmp = scratch[512U];
    auto rowMaximums = scratch[1024U];
    const uint32_t rectangleCount = rowCount * rectanglePitch;
    for (uint32_t offset = 0U; offset < rectangleCount; offset += 512U) {
        const uint32_t count = rectangleCount - offset < 512U ? rectangleCount - offset : 512U;
        AscendC::Abs<float>(values, slab[offset], count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax<float>(rowMaximums[resultCount * 8U], values, reduceTmp, static_cast<int32_t>(count), true);
        ++resultCount;
        AscendC::PipeBarrier<PIPE_V>();
    }
    found = HasNonzeroMaximums(rowMaximums, resultCount) || found;
    targetRowQue_.FreeTensor(slab);
    return found;
}

// Pair low and high rows so every owner receives nearly the same number of
// strict-upper elements.  Each half of the pair range is contiguous, hence
// the complete proof needs only two large row-slab DMAs per owner.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline bool KernelLinalgDet<LocalLuBand, Pipelined>::TryLowerTriangularDet(uint64_t inputBase,
                                                                                      float &determinant)
{
    const uint32_t owner = AscendC::GetBlockIdx();
    const uint32_t owners = AscendC::GetBlockNum();
    const uint32_t proofRows = n_ - 1U;
    const uint32_t pairCount = proofRows / 2U;
    const uint32_t activeOwners = owners < pairCount ? owners : pairCount;
    float localMaximum = 0.0F;
    float localProduct = 1.0F;
    if (activeOwners != 0U && owner < activeOwners) {
        const uint32_t pairStart = owner * pairCount / activeOwners;
        const uint32_t pairEnd = (owner + 1U) * pairCount / activeOwners;
        const uint32_t lowStart = pairStart;
        const uint32_t lowEnd = pairEnd;
        uint32_t highStart = n_ - 1U - pairEnd;
        const uint32_t highEnd = n_ - 1U - pairStart;
        // An odd number of proof rows leaves one center row. Attach it to the
        // last high slab without adding a third DMA.
        if ((proofRows & 1U) != 0U && owner + 1U == activeOwners)
            --highStart;
        if (ScanTriangularRowSlab(inputBase, lowStart, lowEnd, localProduct) ||
            ScanTriangularRowSlab(inputBase, highStart, highEnd, localProduct))
            localMaximum = 1.0F;
    }

    // The final row has no strict-upper entries and is absent from both slabs.
    if (owner == 0U)
        localProduct *= aGm_.GetValue(inputBase + static_cast<uint64_t>(n_ - 1U) * n_ + n_ - 1U);

    // Owners without a row chunk still join both global barriers, but do not
    // pay for a private metadata DMA.
    if (owner < activeOwners)
        StoreTriangularProbe(owner, localMaximum, localProduct);
    AscendC::SyncAll<true>();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(triangularMetaGm_);
    bool isLowerTriangular = true;
    for (uint32_t index = 0U; index < activeOwners; ++index) {
        if (triangularMetaGm_.GetValue(static_cast<uint64_t>(index) * 16U) != 0.0F) {
            isLowerTriangular = false;
            break;
        }
    }
    if (!isLowerTriangular)
        return false;

    // Only block zero reduces the per-owner products.  This reads at most one
    // value per participating AIV instead of all n diagonal values.
    if (owner == 0U) {
        determinant = 1.0F;
        for (uint32_t index = 0U; index < activeOwners; ++index)
            determinant *= triangularMetaGm_.GetValue(static_cast<uint64_t>(index) * 16U + 1U);
    }
    return true;
}

// Isolated n=40 implementation. Keeping this outside DetSmall preserves the
// generated n<=32 kernels while key 1 gets the measured UB/vector crossover.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline float KernelLinalgDet<LocalLuBand, Pipelined>::DetSmall40UbVector(uint64_t inputBase)
{
    constexpr uint32_t N = 40U;
    constexpr uint32_t ELEMENTS = N * N;
    auto matrix = targetRowQue_.AllocTensor<float>();
    const AscendC::DataCopyExtParams copyParams{1U, ELEMENTS * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> noPad{true, 0U, 0U, 0.0F};
    AscendC::DataCopyPad(matrix, aGm_[inputBase], copyParams, noPad);
    targetRowQue_.EnQue(matrix);
    matrix = targetRowQue_.DeQue<float>();

    float triangularDeterminant = 1.0F;
    if (ProbeUbTriangular(matrix, N, N, triangularDeterminant)) {
        targetRowQue_.FreeTensor(matrix);
        return triangularDeterminant;
    }

    auto update = pivotRowQue_.AllocTensor<float>();
    float sign = 1.0F;
    float determinant = 1.0F;
    for (uint32_t pivotColumn = 0U; pivotColumn < N; ++pivotColumn) {
        uint32_t pivotRow = pivotColumn;
        float pivotAbs = 0.0F;
        FindUbPivot(matrix, N, N, pivotColumn, pivotRow, pivotAbs);
        if (pivotAbs == 0.0F) {
            pivotRowQue_.FreeTensor(update);
            targetRowQue_.FreeTensor(matrix);
            return 0.0F;
        }
        SwapUbPivot(matrix, N, N, pivotColumn, pivotRow, sign);
        const float pivot = matrix.GetValue(pivotColumn * N + pivotColumn);
        if (pivot == 0.0F) {
            pivotRowQue_.FreeTensor(update);
            targetRowQue_.FreeTensor(matrix);
            return 0.0F;
        }
        determinant *= pivot;
        EliminateUbRows(matrix, update, N, N, pivotColumn, pivot);
    }
    pivotRowQue_.FreeTensor(update);
    targetRowQue_.FreeTensor(matrix);
    return sign * determinant;
}

// The measured padded-row crossover is positive only for n=41..44. Keep
// this body separate so n=33..39 and n=45..47 retain the proven scalar LU.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline float KernelLinalgDet<LocalLuBand, Pipelined>::DetPaddedUbVector(uint64_t inputBase)
{
    const uint32_t n = n_;
    const uint32_t pitch = (n + 7U) & ~7U;
    auto matrix = targetRowQue_.AllocTensor<float>();
    const AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(n), n * static_cast<uint32_t>(sizeof(float)), 0U,
                                                0U, 0U};
    const AscendC::DataCopyPadExtParams<float> rowPad{true, 0U, static_cast<uint8_t>(pitch - n), 0.0F};
    AscendC::DataCopyPad(matrix, aGm_[inputBase], copyParams, rowPad);
    targetRowQue_.EnQue(matrix);
    matrix = targetRowQue_.DeQue<float>();

    float triangularDeterminant = 1.0F;
    if (ProbeUbTriangular(matrix, n, pitch, triangularDeterminant)) {
        targetRowQue_.FreeTensor(matrix);
        return triangularDeterminant;
    }

    auto update = pivotRowQue_.AllocTensor<float>();
    float sign = 1.0F;
    float determinant = 1.0F;
    for (uint32_t pivotColumn = 0U; pivotColumn < n; ++pivotColumn) {
        uint32_t pivotRow = pivotColumn;
        float pivotAbs = 0.0F;
        FindUbPivot(matrix, n, pitch, pivotColumn, pivotRow, pivotAbs);
        if (pivotAbs == 0.0F) {
            pivotRowQue_.FreeTensor(update);
            targetRowQue_.FreeTensor(matrix);
            return 0.0F;
        }
        SwapUbPivot(matrix, n, pitch, pivotColumn, pivotRow, sign);
        const float pivot = matrix.GetValue(pivotColumn * pitch + pivotColumn);
        if (pivot == 0.0F) {
            pivotRowQue_.FreeTensor(update);
            targetRowQue_.FreeTensor(matrix);
            return 0.0F;
        }
        determinant *= pivot;
        EliminateUbRows(matrix, update, n, pitch, pivotColumn, pivot);
    }
    pivotRowQue_.FreeTensor(update);
    targetRowQue_.FreeTensor(matrix);
    return sign * determinant;
}

#endif
