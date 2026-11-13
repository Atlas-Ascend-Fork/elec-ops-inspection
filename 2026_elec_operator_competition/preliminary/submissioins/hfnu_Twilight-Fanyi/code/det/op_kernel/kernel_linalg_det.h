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
#ifndef KERNEL_LINALG_DET_H
#define KERNEL_LINALG_DET_H

#include "kernel_operator.h"
#include "linalg_det_tiling.h"

template <uint32_t LocalLuBand, bool Pipelined = false>
class KernelLinalgDet {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR det, GM_ADDR workspace, const LinalgDetTilingData *tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CompleteMultiCoreColumn(uint64_t matrixBase, uint32_t pivotColumn, float &determinant);

    __aicore__ inline void StoreMultiCoreResult(uint32_t batch, float sign, float determinant);

    template <bool Padded>
    __aicore__ inline void UpdateRangeSuffix(uint64_t targetOffset, uint64_t pivotOffset, uint32_t count, float factor);

    __aicore__ inline float ReadMatrixValue(const float *matrix, uint32_t index) const;

    __aicore__ inline float ReadMatrixValue(AscendC::LocalTensor<float> matrix, uint32_t index) const;

    template <typename Matrix>
    __aicore__ inline bool ProbeTriangular(Matrix matrix, uint32_t n, uint32_t pitch,
                                           float &triangularDeterminant) const;

    __aicore__ inline void EliminateGmRows(uint64_t matrixBase, uint32_t pivotColumn, float pivot);

    template <bool OwnerPivot>
    __aicore__ inline void DetGrouped(uint32_t batch);

    template <bool OwnerPivot>
    __aicore__ inline bool GroupedPivotStep(uint64_t matrixBase, uint32_t pivotColumn, uint32_t owner, float &sign,
                                            float &determinant);

    template <bool Padded>
    __aicore__ inline void UpdateRangeVector(uint64_t targetOffset, uint64_t pivotOffset, uint32_t vectorCount,
                                             float factor);

    __aicore__ inline AscendC::LocalTensor<float> LoadPivotRow(uint64_t matrixBase, uint32_t pivotColumn);

    __aicore__ inline bool StoreTriangularDet(uint32_t batch, uint64_t inputBase);

    __aicore__ inline bool FindPipelinedPivot(uint64_t matrixBase, uint32_t pivotColumn, uint32_t &pivotRow,
                                              float &pivotAbs);

    __aicore__ inline void UpdateAlignedBlock(AscendC::LocalTensor<float> output, AscendC::LocalTensor<float> target,
                                              AscendC::LocalTensor<float> cachedPivot, const float *factors,
                                              uint32_t rows, uint32_t pivotColumn, uint32_t prefix,
                                              uint32_t vectorCount, uint32_t trailing);

    __aicore__ inline void CopyUnalignedMatrix(uint64_t inputBase, uint64_t matrixBase, uint64_t matrixElements);

    __aicore__ inline AscendC::LocalTensor<float> LoadTriangularRectangle(uint64_t inputBase, uint32_t rowStart,
                                                                          uint32_t rowCount, uint32_t rectangleStart,
                                                                          uint32_t rectangleWidth);

    __aicore__ inline bool HasNonzeroMaximums(AscendC::LocalTensor<float> rowMaximums, uint32_t resultCount);

    __aicore__ inline bool DetSpecializedSmall(uint32_t batch, uint64_t inputBase);

    __aicore__ inline bool DetSmallRuntime(uint32_t batch, uint64_t inputBase);

    __aicore__ inline void DetTiny(uint32_t batch, uint64_t inputBase);

    __aicore__ inline void DetBatchGm(uint32_t batch, uint64_t inputBase);

    __aicore__ inline void SwapGmPivot(uint64_t matrixBase, uint32_t pivotColumn, uint32_t pivotRow);

    __aicore__ inline void UpdateCachedVector(uint64_t targetStart, uint32_t vectorCount, float factor,
                                              AscendC::LocalTensor<float> cachedPivot);

    __aicore__ inline void UpdateCachedAlignedRow(uint64_t matrixBase, uint32_t pivotColumn, float pivot, uint32_t row,
                                                  uint32_t count, uint32_t vectorCount, uint64_t pivotStart,
                                                  AscendC::LocalTensor<float> cachedPivot, uint32_t prefix);

    __aicore__ inline void UpdateCachedPaddedRow(uint64_t matrixBase, uint32_t pivotColumn, float pivot, uint32_t row,
                                                 uint32_t count, uint32_t vectorCount, uint64_t pivotStart,
                                                 AscendC::LocalTensor<float> cachedPivot);

    __aicore__ inline void EliminatePackedBlocks(uint64_t matrixBase, uint32_t pivotColumn, float pivot,
                                                 uint32_t prefix, uint32_t vectorCount, uint32_t trailing,
                                                 AscendC::LocalTensor<float> cachedPivot, float &nextPivotAbs,
                                                 uint32_t &nextPivotRow);

    __aicore__ inline AscendC::LocalTensor<float> LoadPackedRows(uint64_t matrixBase, uint32_t pivotColumn,
                                                                 uint32_t firstRow, uint32_t &rows,
                                                                 uint32_t &packedColumn, uint32_t &packedPitch,
                                                                 uint64_t &rowStart,
                                                                 AscendC::DataCopyExtParams &storePackParams);

    __aicore__ inline void ProbeNextPackedPivot(AscendC::LocalTensor<float> output, uint32_t pivotColumn,
                                                uint32_t packedColumn, uint32_t packedPitch, uint32_t rows,
                                                uint32_t firstRow, float &nextPivotAbs, uint32_t &nextPivotRow);

    __aicore__ inline void RepairPackedFringes(AscendC::LocalTensor<float> output, AscendC::LocalTensor<float> target,
                                               AscendC::LocalTensor<float> cachedPivot,
                                               AscendC::LocalTensor<float> factorSource, uint32_t rows,
                                               uint32_t packedPitch, uint32_t pivotLocal, uint32_t pivotColumn,
                                               uint32_t trailing, uint32_t prefix, uint32_t vectorCount);

    __aicore__ inline void UpdatePackedVectors(AscendC::LocalTensor<float> output, AscendC::LocalTensor<float> target,
                                               AscendC::LocalTensor<float> cachedPivot,
                                               AscendC::LocalTensor<float> factorRows, uint32_t packedPitch,
                                               uint32_t packedColumn, uint32_t pivotColumn, uint32_t pivotLocal,
                                               uint32_t prefix, uint32_t vectorCount, uint32_t rows);

    __aicore__ inline void BuildPackedFactors(AscendC::LocalTensor<float> target,
                                              AscendC::LocalTensor<float> factorSource,
                                              AscendC::LocalTensor<float> factorRows, uint32_t rows,
                                              uint32_t packedPitch, uint32_t pivotLocal, float pivot);

    __aicore__ inline void SwapScalarLocalPivot(float *matrix, uint32_t n, uint32_t pivotColumn, uint32_t pivotRow,
                                                float &sign) const;

    __aicore__ inline void FindScalarLocalPivot(const float *matrix, uint32_t n, uint32_t pivotColumn,
                                                uint32_t &pivotRow, float &pivotAbs) const;

    template <uint32_t FixedN>
    __aicore__ inline void EliminateScalarLocalRows(float *matrix, uint32_t n, uint32_t pivotColumn, float pivot) const;

    __aicore__ inline bool ProbeScalarTriangular(const float *matrix, uint32_t n, float &triangularDeterminant) const;

    __aicore__ inline void SwapUbPivot(AscendC::LocalTensor<float> matrix, uint32_t n, uint32_t pitch,
                                       uint32_t pivotColumn, uint32_t pivotRow, float &sign);

    __aicore__ inline void FindUbPivot(AscendC::LocalTensor<float> matrix, uint32_t n, uint32_t pitch,
                                       uint32_t pivotColumn, uint32_t &pivotRow, float &pivotAbs);

    __aicore__ inline void EliminateUbRows(AscendC::LocalTensor<float> matrix, AscendC::LocalTensor<float> update,
                                           uint32_t n, uint32_t pitch, uint32_t pivotColumn, float pivot);

    __aicore__ inline bool ProbeUbTriangular(AscendC::LocalTensor<float> matrix, uint32_t n, uint32_t pitch,
                                             float &triangularDeterminant);

    __aicore__ inline void StorePivotCandidate(uint32_t slot, float value, uint32_t row);

    __aicore__ inline void LoadPivotCandidate(uint32_t slot, float &value, uint32_t &row);

    __aicore__ inline void ReducePublishedPivot(uint32_t pivotColumn, float &pivotAbs, uint32_t &pivotRow);

    __aicore__ inline void DetBatchMultiCorePipelined(uint32_t batch);

    __aicore__ inline uint32_t RowOwners() const;

    __aicore__ inline uint32_t RowOwner() const;

    __aicore__ inline uint64_t MatrixIndex(uint64_t base, uint32_t row, uint32_t column) const;
    __aicore__ inline float Abs(float value) const;

    __aicore__ inline void FindPivotScalar(uint64_t matrixBase, uint32_t pivotColumn, uint32_t &pivotRow,
                                           float &pivotAbs);

    __aicore__ inline void FindPivotVector(uint64_t matrixBase, uint32_t pivotColumn, uint32_t &pivotRow,
                                           float &pivotAbs);

    __aicore__ inline void StoreDet(uint32_t batch, float value);

    __aicore__ inline void StoreTriangularProbe(uint32_t owner, float value, float diagonalProduct);

    __aicore__ inline bool ScanTriangularFullRowSlab(uint64_t inputBase, uint32_t rowStart, uint32_t rowEnd,
                                                     float &diagonalProduct);

    __aicore__ inline bool ScanTriangularRowSlab(uint64_t inputBase, uint32_t rowStart, uint32_t rowEnd,
                                                 float &diagonalProduct);

    __aicore__ inline bool TryLowerTriangularDet(uint64_t inputBase, float &determinant);

    __aicore__ inline float DetSmall40UbVector(uint64_t inputBase);

    __aicore__ inline float DetPaddedUbVector(uint64_t inputBase);

    template <uint32_t Capacity, uint32_t FixedN = 0U>
    __aicore__ inline float DetSmall(uint64_t inputBase) const;

    __aicore__ inline void UpdateRange(uint64_t targetOffset, uint64_t pivotOffset, uint32_t count, float factor);

    __aicore__ inline void UpdateRangePadded(uint64_t targetOffset, uint64_t pivotOffset, uint32_t count, float factor);

    __aicore__ inline void EliminateRowsWithCachedPaddedPivot(uint64_t matrixBase, uint32_t pivotColumn, float pivot);

    __aicore__ inline void EliminateRowsWithCachedPivot(uint64_t matrixBase, uint32_t pivotColumn, float pivot);

    __aicore__ inline void EliminateRowBlocksPacked(uint64_t matrixBase, uint32_t pivotColumn, float pivot);

    __aicore__ inline void EliminateScalarRowsPartitioned(uint64_t matrixBase, uint32_t pivotColumn, float pivot);

    __aicore__ inline void EliminateRowBlocks(uint64_t matrixBase, uint32_t pivotColumn, float pivot);

    __aicore__ inline void CopyMatrix(uint64_t inputBase, uint64_t matrixBase);

    __aicore__ inline void DetBatchMultiCore(uint32_t batch);

    __aicore__ inline void DetBatchGroupedBaseline(uint32_t batch);

    __aicore__ inline void DetBatchGrouped(uint32_t batch);

    __aicore__ inline void DetBatch(uint32_t batch);

    AscendC::GlobalTensor<uint32_t> pivotMetaBitsGm_;
    AscendC::GlobalTensor<float> aGm_;
    AscendC::GlobalTensor<float> detGm_;
    AscendC::GlobalTensor<float> workGm_;
    AscendC::GlobalTensor<float> triangularMetaGm_;
    uint32_t batchCount_ = 0U;
    uint32_t n_ = 0U;
    uint32_t workspaceStride_ = 0U;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> targetRowQue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> pivotRowQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputRowQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> factorTileBuf_;
    event_t scalarToMte2_;
    event_t scalarToMte3_;
    event_t mte3ToMte2_;
    event_t mte3ToScalar_;
    event_t vectorToScalar_;
    event_t scalarToVector_;
};

#endif
