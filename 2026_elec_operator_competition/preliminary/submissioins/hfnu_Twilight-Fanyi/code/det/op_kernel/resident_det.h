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
#ifndef RESIDENT_DET_H
#define RESIDENT_DET_H

#include "kernel_operator.h"

// Isolated key63: n64..127 with padded UB pitch. Each AIV owns complete
// matrices in grid-stride order, so UB state and GM outputs never overlap.
// Other contracts stay on the complete baseline. This helper owns its AscendC::TPipe
// and all UB storage.
class ResidentDet {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t batches, uint32_t matrixSize)
    {
        n = matrixSize;
        batchCount = batches;

        pitch = (n + ALIGN_ELEMENTS - 1U) & ~(ALIGN_ELEMENTS - 1U);
        pipe.InitBuffer(matrixQueue, 1U, n * pitch * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(outputQueue, 1U, ALIGN_ELEMENTS * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(scratchBuffer, n * pitch * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(factorsBuffer, pitch * static_cast<uint32_t>(sizeof(float)));
        pipe.InitBuffer(factorRowsBuffer, pitch * 64U * static_cast<uint32_t>(sizeof(float)));
        matrixElements = static_cast<uint64_t>(n) * n;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input),
                               static_cast<uint64_t>(batchCount) * matrixElements);
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), batchCount);
        update = scratchBuffer.Get<float>();
        factors = factorsBuffer.Get<float>();
        factorRows = factorRowsBuffer.Get<float>();
        scalarToVector = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::S_V));
        vectorToScalar = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_S));
        scalarToMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::S_MTE3));
        mte3ToScalar = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_S));
    }

    __aicore__ inline void Process()
    {
        const AscendC::DataCopyExtParams load{static_cast<uint16_t>(n), n * static_cast<uint32_t>(sizeof(float)), 0U,
                                              0U, 0U};
        const AscendC::DataCopyPadExtParams<float> padding{true, 0U, static_cast<uint8_t>(pitch - n), 0.0F};
        const uint32_t blockCount = AscendC::GetBlockNum();
        for (uint32_t batch = AscendC::GetBlockIdx(); batch < batchCount; batch += blockCount) {
            auto matrix = matrixQueue.AllocTensor<float>();
            const uint64_t inputBase = static_cast<uint64_t>(batch) * matrixElements;
            AscendC::DataCopyPad(matrix, source[inputBase], load, padding);
            matrixQueue.EnQue(matrix);
            matrix = matrixQueue.DeQue<float>();
            float determinant = 1.0F;
            float sign = 1.0F;
            for (uint32_t k = 0U; k < n; ++k) {
                uint32_t pivotRow = k;
                float maximum = 0.0F;
                FindPivot(matrix, k, pivotRow, maximum);
                if (maximum == 0.0F) {
                    determinant = 0.0F;
                    break;
                }
                if (pivotRow != k) {
                    for (uint32_t column = k; column < n; ++column) {
                        const float value = matrix.GetValue(k * pitch + column);
                        matrix.SetValue(k * pitch + column, matrix.GetValue(pivotRow * pitch + column));
                        matrix.SetValue(pivotRow * pitch + column, value);
                    }
                    sign = -sign;
                }
                const float pivot = matrix.GetValue(k * pitch + k);
                if (pivot == 0.0F) {
                    determinant = 0.0F;
                    break;
                }
                determinant *= pivot;
                const uint32_t start = k & ~(ALIGN_ELEMENTS - 1U);
                const uint32_t rows = n - k - 1U;
                const bool vectorFactors = rows >= 16U;
                BuildFactors(matrix, k, pivot, rows, vectorFactors);
                UpdateTrailing(matrix, k, start, rows, vectorFactors);
            }
            StoreDet(batch, sign * determinant);
            matrixQueue.FreeTensor(matrix);
        }
    }

private:
    __aicore__ inline void StoreDet(uint32_t batch, float value)
    {
        auto result = outputQueue.AllocTensor<float>();
        result.SetValue(0U, value);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);
        const AscendC::DataCopyExtParams store{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        AscendC::DataCopyPad(destination[batch], result, store);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalar);
        outputQueue.FreeTensor(result);
    }

    __aicore__ inline void UpdateTrailing(AscendC::LocalTensor<float> matrix, uint32_t k, uint32_t start, uint32_t rows,
                                          bool vectorFactors)
    {
        if (k + 1U < n) {
            const uint32_t paddedRows = (rows + 7U) & ~7U;
            if (!vectorFactors) {
                for (uint32_t row = rows; row < paddedRows; ++row) {
                    factors.SetValue(row, 0.0F);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
            // Reuse the complete baseline's proven factor-broadcast layout:
            // each scalar becomes one 32-byte block, then a 64-float row.
            AscendC::Brcb<float>(factorRows, factors, static_cast<uint8_t>(paddedRows / 8U),
                                 AscendC::BrcbRepeatParams(8U, 64U));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float>(factorRows, factorRows, 0.0F, 64U, static_cast<uint8_t>(rows),
                                 AscendC::UnaryRepeatParams(1U, 0U, 8U, 8U));
            AscendC::PipeBarrier<PIPE_V>();
            const uint8_t rowStrideBlocks = static_cast<uint8_t>(pitch / 8U);
            const AscendC::BinaryRepeatParams multiplyRows{1U, 1U, 1U, rowStrideBlocks, 8U, 0U};
            const AscendC::BinaryRepeatParams addRows{1U, 1U, 1U, rowStrideBlocks, rowStrideBlocks, rowStrideBlocks};
            const uint32_t firstTarget = (k + 1U) * pitch + start;
            for (uint32_t tile = 0U; tile < pitch - start; tile += 64U) {
                const uint32_t width = pitch - start - tile < 64U ? pitch - start - tile : 64U;
                AscendC::Mul<float>(update[firstTarget + tile], factorRows, matrix[k * pitch + start + tile], width,
                                    static_cast<uint8_t>(rows), multiplyRows);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t tile = 0U; tile < pitch - start; tile += 64U) {
                const uint32_t width = pitch - start - tile < 64U ? pitch - start - tile : 64U;
                AscendC::Add<float>(matrix[firstTarget + tile], matrix[firstTarget + tile], update[firstTarget + tile],
                                    width, static_cast<uint8_t>(rows), addRows);
            }
            AscendC::ResetMask();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        }
    }

    __aicore__ inline void BuildFactors(AscendC::LocalTensor<float> matrix, uint32_t k, float pivot, uint32_t rows,
                                        bool vectorFactors)
    {
        if (pivot == 0.0F) {
            return;
        }
        if (vectorFactors) {
            const uint32_t paddedRows = (rows + 7U) & ~7U;
            auto signedOffsets = factorRows.ReinterpretCast<int32_t>();
            auto offsets = signedOffsets.ReinterpretCast<uint32_t>();
            auto divisors = factorRows[128U];
            // Scalar row swaps must be visible to the column AscendC::Gather.
            AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
            AscendC::Duplicate<float>(factors, 0.0F, paddedRows);
            AscendC::ArithProgression<int32_t>(signedOffsets,
                                               static_cast<int32_t>(((k + 1U) * pitch + k) * sizeof(float)),
                                               static_cast<int32_t>(pitch * sizeof(float)), rows);
            AscendC::Duplicate<float>(divisors, pivot, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather<float>(factors, matrix, offsets, 0U, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls<float>(factors, factors, -1.0F, rows);
            AscendC::PipeBarrier<PIPE_V>();
            // Direct division, never reciprocal-multiply (v85 inf failure).
            AscendC::Div<float>(factors, factors, divisors, rows);
            AscendC::PipeBarrier<PIPE_V>();
            // offsets/divisors are dead before AscendC::Brcb overwrites factorRows.
        } else {
            for (uint32_t row = k + 1U; row < n; ++row) {
                factors.SetValue(row - k - 1U, -matrix.GetValue(row * pitch + k) / pivot);
            }
        }
    }

    __aicore__ inline void FindPivot(AscendC::LocalTensor<float> matrix, uint32_t k, uint32_t &pivotRow, float &maximum)
    {
        // v196: exact n128 only. Scratch is dead until factor broadcast.
        if (n - k >= 16U) {
            auto signedOffsets = factorRows.ReinterpretCast<int32_t>();
            auto offsets = signedOffsets.ReinterpretCast<uint32_t>();
            auto values = factorRows[128U];
            auto magnitudes = factorRows[256U];
            auto reduceTmp = factorRows[384U];
            auto reduceOut = factorRows[512U];
            AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
            AscendC::ArithProgression<int32_t>(signedOffsets, static_cast<int32_t>((k * pitch + k) * sizeof(float)),
                                               static_cast<int32_t>(pitch * sizeof(float)), n - k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather<float>(values, matrix, offsets, 0U, n - k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs<float>(magnitudes, values, n - k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(reduceOut, magnitudes, reduceTmp, static_cast<int32_t>(n - k), true);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
            maximum = reduceOut.GetValue(0U);
            pivotRow = k + reduceOut.ReinterpretCast<uint32_t>().GetValue(1U);
        } else {
            for (uint32_t row = k; row < n; ++row) {
                const float value = matrix.GetValue(row * pitch + k);
                const float magnitude = value < 0.0F ? -value : value;
                if (magnitude > maximum) {
                    maximum = magnitude;
                    pivotRow = row;
                }
            }
        }
    }

    static constexpr uint32_t ALIGN_ELEMENTS = 8U;
    uint32_t n = 0U;
    uint32_t batchCount = 0U;
    uint32_t pitch = 0U;
    uint64_t matrixElements = 0U;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> matrixQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scratchBuffer;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> factorsBuffer;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> factorRowsBuffer;
    AscendC::GlobalTensor<float> source;
    AscendC::GlobalTensor<float> destination;
    AscendC::LocalTensor<float> update;
    AscendC::LocalTensor<float> factors;
    AscendC::LocalTensor<float> factorRows;
    event_t scalarToVector;
    event_t vectorToScalar;
    event_t scalarToMte3;
    event_t mte3ToScalar;
};

__aicore__ inline void RunResidentDet(GM_ADDR input, GM_ADDR output, uint32_t batchCount, uint32_t n)
{
    ResidentDet op;
    op.Init(input, output, batchCount, n);
    op.Process();
}

#endif
