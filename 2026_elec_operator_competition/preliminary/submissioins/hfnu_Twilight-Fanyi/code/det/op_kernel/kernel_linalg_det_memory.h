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
#ifndef KERNEL_LINALG_DET_MEMORY_H
#define KERNEL_LINALG_DET_MEMORY_H

#include "kernel_linalg_det.h"

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::Init(GM_ADDR a, GM_ADDR det, GM_ADDR workspace,
                                                                     const LinalgDetTilingData *tiling)
{
    batchCount_ = tiling->batchCount;
    n_ = tiling->matrixSize;
    workspaceStride_ = tiling->workspaceStride;
    const uint64_t matrixElements = static_cast<uint64_t>(n_) * n_;
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), batchCount_ * matrixElements);
    detGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(det), batchCount_);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const uint64_t matrixWorkspaceElements = static_cast<uint64_t>(batchCount_) * workspaceStride_;
    workGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace), matrixWorkspaceElements);
    triangularMetaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + matrixWorkspaceElements,
                                      LocalLuBand == 13U || LocalLuBand == 14U || Pipelined
                                          ? static_cast<uint64_t>(AscendC::GetBlockNum()) * 16U
                                          : (LocalLuBand == 62U ? static_cast<uint64_t>(batchCount_) * 16U : 0U));
    if constexpr (Pipelined) {
        pivotMetaBitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(userWorkspace) + matrixWorkspaceElements,
                                         static_cast<uint64_t>(AscendC::GetBlockNum()) * 16U);
    }
    const uint32_t paddedN = (n_ + 7U) & ~7U;
    const bool paddedKey1 = LocalLuBand == 1U && n_ >= 41U && n_ <= 44U;
    const uint32_t rowPitch = paddedKey1 || LocalLuBand == 14U ? paddedN : n_;
    const uint32_t residentRows = paddedKey1 ? 44U : 40U;
    pipe_.InitBuffer(targetRowQue_, 1U, residentRows * rowPitch * sizeof(float));
    pipe_.InitBuffer(pivotRowQue_, 1U, rowPitch * sizeof(float));
    pipe_.InitBuffer(outputRowQue_, 1U, residentRows * rowPitch * sizeof(float));
    if constexpr ((LocalLuBand >= 8U && LocalLuBand <= 15U) || LocalLuBand == 62U) {
        // Keep the source factors separate from their 64-wide broadcast rows.
        // 64 floats cover the padded source group and 40 * 64 cover one factor
        // tile for every row in the elimination block.
        pipe_.InitBuffer(factorTileBuf_, (64U + 40U * 64U) * sizeof(float));
    }
    scalarToMte2_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::S_MTE2));
    scalarToMte3_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::S_MTE3));
    mte3ToMte2_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    mte3ToScalar_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::MTE3_S));
    vectorToScalar_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::V_S));
    if constexpr (LocalLuBand == 1U)
        scalarToVector_ = static_cast<event_t>(pipe_.FetchEventID(AscendC::HardEvent::S_V));
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::Process()
{
    if constexpr (Pipelined) {
        DetBatchMultiCorePipelined(0U);
        return;
    }
    if constexpr (LocalLuBand == 15U) {
        const uint32_t owners = RowOwners();
        if (owners == 0U) {
            return;
        }
        DetBatchGroupedBaseline(AscendC::GetBlockIdx() / owners);
        return;
    }
    if constexpr (LocalLuBand == 62U) {
        const uint32_t owners = RowOwners();
        if (owners == 0U) {
            return;
        }
        DetBatchGrouped(AscendC::GetBlockIdx() / owners);
        return;
    }
    if constexpr (LocalLuBand >= 9U && LocalLuBand <= 14U) {
        DetBatchMultiCore(0U);
        return;
    }
    const uint32_t blockCount = AscendC::GetBlockNum();
    for (uint32_t batch = AscendC::GetBlockIdx(); batch < batchCount_; batch += blockCount) DetBatch(batch);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::CopyUnalignedMatrix(uint64_t inputBase,
                                                                                    uint64_t matrixBase,
                                                                                    uint64_t matrixElements)
{
    if ((n_ & 7U) != 0U) {
        if constexpr (LocalLuBand == 14U) {
            const uint32_t capacity = (40U * n_) & ~7U;
            uint64_t copied = 0U;
            while (matrixElements - copied >= 8U) {
                const uint32_t remaining =
                    static_cast<uint32_t>(matrixElements - copied > capacity ? capacity : matrixElements - copied);
                const uint32_t count = remaining & ~7U;
                auto input = targetRowQue_.AllocTensor<float>();
                AscendC::DataCopy(input, aGm_[inputBase + copied], count);
                targetRowQue_.EnQue(input);
                input = targetRowQue_.DeQue<float>();
                auto output = outputRowQue_.AllocTensor<float>();
                AscendC::Adds(output, input, 0.0F, count);
                outputRowQue_.EnQue(output);
                output = outputRowQue_.DeQue<float>();
                AscendC::DataCopy(workGm_[matrixBase + copied], output, count);
                targetRowQue_.FreeTensor(input);
                outputRowQue_.FreeTensor(output);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
                copied += count;
            }
            while (copied < matrixElements) {
                workGm_.SetValue(matrixBase + copied, aGm_.GetValue(inputBase + copied));
                ++copied;
            }
            return;
        }
        for (uint64_t index = 0U; index < matrixElements; ++index)
            workGm_.SetValue(matrixBase + index, aGm_.GetValue(inputBase + index));
        return;
    }
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::DetBatchGm(uint32_t batch, uint64_t inputBase)
{
    const uint64_t matrixBase = static_cast<uint64_t>(batch) * workspaceStride_;
    CopyMatrix(inputBase, matrixBase);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);

    float sign = 1.0F;
    float determinant = 1.0F;
    for (uint32_t pivotColumn = 0U; pivotColumn < n_; ++pivotColumn) {
        uint32_t pivotRow = pivotColumn;
        float pivotAbs = 0.0F;
        FindPivotScalar(matrixBase, pivotColumn, pivotRow, pivotAbs);
        if (pivotAbs == 0.0F) {
            StoreDet(batch, 0.0F);
            return;
        }
        if (pivotRow != pivotColumn) {
            sign = -sign;
            // Columns left of the active panel are no longer consumed: the
            // determinant product is accumulated as pivots are chosen and all
            // later pivoting/elimination starts at the active column.
            SwapGmPivot(matrixBase, pivotColumn, pivotRow);
        } else if ((n_ & 7U) != 0U) {
            // The non-aligned fallback mixes scalar row updates and later scalar
            // pivot searches, so retain the proven full-cache boundary there.
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        const float pivot = workGm_.GetValue(MatrixIndex(matrixBase, pivotColumn, pivotColumn));
        if (pivot == 0.0F) {
            StoreDet(batch, 0.0F);
            return;
        }
        determinant *= pivot;
        if constexpr (LocalLuBand == 7U) {
            // Key 7 is dedicated to the proven non-aligned cached-pivot path.
            EliminateRowsWithCachedPaddedPivot(matrixBase, pivotColumn, pivot);
        } else if ((n_ & 7U) == 0U) {
            if constexpr (LocalLuBand >= 8U)
                EliminateRowBlocksPacked(matrixBase, pivotColumn, pivot);
            else
                EliminateRowBlocks(matrixBase, pivotColumn, pivot);
        } else {
            EliminateGmRows(matrixBase, pivotColumn, pivot);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(workGm_);
    }
    StoreDet(batch, sign * determinant);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline uint32_t KernelLinalgDet<LocalLuBand, Pipelined>::RowOwners() const
{
    if constexpr (LocalLuBand == 15U || (LocalLuBand == 62U && !Pipelined))
        return batchCount_ == 0U ? 0U : AscendC::GetBlockNum() / batchCount_;
    return AscendC::GetBlockNum();
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline uint32_t KernelLinalgDet<LocalLuBand, Pipelined>::RowOwner() const
{
    if constexpr (LocalLuBand == 15U || (LocalLuBand == 62U && !Pipelined)) {
        const uint32_t owners = RowOwners();
        return owners == 0U ? 0U : AscendC::GetBlockIdx() % owners;
    }
    return AscendC::GetBlockIdx();
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline uint64_t KernelLinalgDet<LocalLuBand, Pipelined>::MatrixIndex(uint64_t base, uint32_t row,
                                                                                uint32_t column) const
{
    return base + static_cast<uint64_t>(row) * n_ + column;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline float KernelLinalgDet<LocalLuBand, Pipelined>::Abs(float value) const
{
    return value < 0.0F ? -value : value;
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::FindPivotScalar(uint64_t matrixBase,
                                                                                uint32_t pivotColumn,
                                                                                uint32_t &pivotRow, float &pivotAbs)
{
    pivotRow = pivotColumn;
    pivotAbs = Abs(workGm_.GetValue(MatrixIndex(matrixBase, pivotColumn, pivotColumn)));
    for (uint32_t row = pivotColumn + 1U; row < n_; ++row) {
        const float candidate = Abs(workGm_.GetValue(MatrixIndex(matrixBase, row, pivotColumn)));
        if (candidate > pivotAbs) {
            pivotAbs = candidate;
            pivotRow = row;
        }
    }
}

// Pack one 32-byte slab per candidate row, then gather the pivot lane in
// UB. AscendC::DataCopyPad permits the unaligned GM addresses used by key 14. The
// 16-row cutoff also guarantees that every eight-value slab stays inside
// the current logical row. Selection remains replicated on every row owner,
// preserving the existing synchronization protocol.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::FindPivotVector(uint64_t matrixBase,
                                                                                uint32_t pivotColumn,
                                                                                uint32_t &pivotRow, float &pivotAbs)
{
    const uint32_t count = n_ - pivotColumn;
    const uint32_t packedColumn = pivotColumn & ~7U;
    const uint32_t pivotLocal = pivotColumn - packedColumn;
    const uint64_t slabStart = MatrixIndex(matrixBase, pivotColumn, packedColumn);
    const AscendC::DataCopyExtParams slabParams{static_cast<uint16_t>(count), 8U * sizeof(float),
                                                (n_ - 8U) * static_cast<uint32_t>(sizeof(float)), 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> noPad{true, 0U, 0U, 0.0F};

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(scalarToMte2_);
    auto slabs = targetRowQue_.AllocTensor<float>();
    AscendC::DataCopyPad(slabs, workGm_[slabStart], slabParams, noPad);
    targetRowQue_.EnQue(slabs);
    slabs = targetRowQue_.DeQue<float>();

    auto scratch = factorTileBuf_.Get<float>();
    auto signedOffsets = scratch.ReinterpretCast<int32_t>();
    auto offsets = signedOffsets.ReinterpretCast<uint32_t>();
    auto values = scratch[512U];
    auto reduceTmp = scratch[1024U];
    auto reduceResult = scratch[1536U];
    AscendC::ArithProgression<int32_t>(signedOffsets, static_cast<int32_t>(pivotLocal * sizeof(float)),
                                       static_cast<int32_t>(8U * sizeof(float)), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather<float>(values, slabs, offsets, 0U, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs<float>(values, values, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax<float>(reduceResult, values, reduceTmp, static_cast<int32_t>(count), true);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar_);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar_);

    pivotAbs = reduceResult.GetValue(0U);
    auto resultBits = reduceResult.ReinterpretCast<uint32_t>();
    pivotRow = pivotColumn + resultBits.GetValue(1U);
    targetRowQue_.FreeTensor(slabs);
}

// Scalar GM stores are cache-line transactions. They cannot safely publish
// adjacent FP32 results from different AIVs. MTE3 writes the exact four-byte
// interval instead, making disjoint batch outputs independent even when
// they share one 64-byte cache line.
template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::StoreDet(uint32_t batch, float value)
{
    auto output = outputRowQue_.AllocTensor<float>();
    output.SetValue(0U, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3_);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3_);
    const AscendC::DataCopyExtParams storeParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    AscendC::DataCopyPad(detGm_[batch], output, storeParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    outputRowQue_.FreeTensor(output);
}

template <uint32_t LocalLuBand, bool Pipelined>
__aicore__ inline void KernelLinalgDet<LocalLuBand, Pipelined>::CopyMatrix(uint64_t inputBase, uint64_t matrixBase)
{
    const uint64_t matrixElements = static_cast<uint64_t>(n_) * n_;
    if ((n_ & 7U) != 0U) {
        CopyUnalignedMatrix(inputBase, matrixBase, matrixElements);
        return;
    }
    // Both row starts are 32-byte aligned when n is a multiple of eight.
    // Move up to eight contiguous rows through UB instead of issuing scalar
    // GM loads and stores. This is shape-generic and preserves the scalar
    // fallback for every non-aligned width.
    for (uint32_t row = 0U; row < n_; row += 8U) {
        const uint32_t rows = n_ - row < 8U ? n_ - row : 8U;
        const uint32_t count = rows * n_;
        auto inputRow = targetRowQue_.AllocTensor<float>();
        AscendC::DataCopy(inputRow, aGm_[inputBase + static_cast<uint64_t>(row) * n_], count);
        targetRowQue_.EnQue(inputRow);
        inputRow = targetRowQue_.DeQue<float>();
        auto outputRow = outputRowQue_.AllocTensor<float>();
        AscendC::Adds(outputRow, inputRow, 0.0F, count);
        outputRowQue_.EnQue(outputRow);
        outputRow = outputRowQue_.DeQue<float>();
        AscendC::DataCopy(workGm_[matrixBase + static_cast<uint64_t>(row) * n_], outputRow, count);
        targetRowQue_.FreeTensor(inputRow);
        outputRowQue_.FreeTensor(outputRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar_);
}

#endif
