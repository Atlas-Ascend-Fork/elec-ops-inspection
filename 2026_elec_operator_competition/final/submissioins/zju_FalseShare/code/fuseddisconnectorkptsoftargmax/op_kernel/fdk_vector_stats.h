/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"

namespace fdk {
using namespace AscendC;

struct RowCopy {
    uint32_t source;
    uint32_t destination;
    uint32_t elements;
};

__aicore__ inline RowCopy GetRowCopy(uint32_t first, uint32_t rows, uint32_t height, uint32_t width) {
    const uint32_t sourceRow = first == 0 ? 0 : first - 1;
    const uint32_t destinationRow = first == 0 ? 1 : 0;
    const uint32_t copiedRows = rows + 2 - (first == 0 ? 1 : 0) - (first + rows == height ? 1 : 0);
    return {sourceRow * width, destinationRow * width, copiedRows * width};
}

__aicore__ inline void LoadBiases(const LocalTensor<float>& biases, const GlobalTensor<float>& biasGm) {
    DataCopyPad(biases, biasGm, DataCopyExtParams{1, 16, 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline float RestoreCoordinate(float numerator, float denominator, uint32_t stride) {
    // A valid stable softmax includes exp(0), so its denominator is at least one.
    // Preserve a non-finite result if the reduction is invalid instead of dividing by zero.
    if (denominator > 0.0f) {
        return (numerator / denominator + 0.5f) * static_cast<float>(static_cast<int32_t>(stride));
    }
    return __builtin_nanf("");
}

__aicore__ inline void StoreVectorResult(const LocalTensor<float>& reduced,
                                        const GlobalTensor<float>& keypointsGm,
                                        const GlobalTensor<float>& scoresGm,
                                        uint32_t plane, float x, float y, float score) {
    reduced.SetValue(0, x);
    reduced.SetValue(1, y);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(keypointsGm[plane * 2], reduced, DataCopyExtParams{1, 8, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
    reduced.SetValue(0, score);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(scoresGm[plane], reduced, DataCopyExtParams{1, 4, 0, 0, 0});
}

// The three Vector convolution paths share reduction order and output layout.
class VectorStatistics {
public:
    __aicore__ inline VectorStatistics(const LocalTensor<float>& acc, const LocalTensor<float>& tmp,
                                      const LocalTensor<float>& coord, const LocalTensor<float>& scratch,
                                      const LocalTensor<float>& reduced, uint32_t width, uint32_t rows,
                                      uint32_t first)
        : acc(acc), tmp(tmp), coord(coord), scratch(scratch), reduced(reduced), width(width), rows(rows),
          first(first), count(width * rows) {}

    template <bool SCALE_TEMPERATURE>
    __aicore__ inline void PrepareSoftmax(float tau) {
        ReduceMax(reduced, acc, scratch, count, false);
        PipeBarrier<PIPE_ALL>();
        score = reduced.GetValue(0);
        maxLogit = score;
        if constexpr (SCALE_TEMPERATURE) {
            Muls(acc, acc, 1.0f / tau, count);
            PipeBarrier<PIPE_V>();
            ReduceMax(reduced, acc, scratch, count, false);
            PipeBarrier<PIPE_ALL>();
            maxLogit = reduced.GetValue(0);
        }
        Adds(acc, acc, -maxLogit, count);
        PipeBarrier<PIPE_V>();
        Exp(acc, acc, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduced, acc, scratch, count);
        PipeBarrier<PIPE_ALL>();
        denominator = reduced.GetValue(0);
    }

    template <bool VECTOR_INDEX>
    __aicore__ inline float SumX() {
        if constexpr (VECTOR_INDEX) {
            CreateVecIndex(coord, 0.0f, width);
        } else {
            for (uint32_t x = 0; x < width; ++x) {
                coord.SetValue(x, static_cast<float>(static_cast<int32_t>(x)));
            }
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t y = 1; y < rows; ++y) {
            DataCopy(coord[y * width], coord, width);
        }
        PipeBarrier<PIPE_ALL>();
        return SumCoordinate();
    }

    __aicore__ inline float SumY() {
        for (uint32_t y = 0; y < rows; ++y) {
            Duplicate(coord[y * width], static_cast<float>(static_cast<int32_t>(first + y)), width);
        }
        PipeBarrier<PIPE_V>();
        return SumCoordinate();
    }

    template <bool VECTOR_INDEX>
    __aicore__ inline void StorePartial(const GlobalTensor<float>& statsGm, uint32_t block) {
        const float numeratorX = SumX<VECTOR_INDEX>();
        const float numeratorY = SumY();
        reduced.SetValue(0, score);
        reduced.SetValue(1, maxLogit);
        reduced.SetValue(2, denominator);
        reduced.SetValue(3, numeratorX);
        reduced.SetValue(4, numeratorY);
        PipeBarrier<PIPE_ALL>();
        DataCopy(statsGm[block * 8], reduced, 8);
        SyncAll();
    }

    template <bool SCALE_TEMPERATURE, bool VECTOR_INDEX>
    __aicore__ inline void ProcessPartial(float bias, float tau, const GlobalTensor<float>& statsGm, uint32_t block) {
        Adds(acc, acc, bias, count);
        PipeBarrier<PIPE_ALL>();
        PrepareSoftmax<SCALE_TEMPERATURE>(tau);
        StorePartial<VECTOR_INDEX>(statsGm, block);
    }

    __aicore__ inline void MergeAndStore(const GlobalTensor<float>& statsGm, const LocalTensor<float>& stats,
                                        const LocalTensor<float>& factors, uint32_t parts, uint32_t plane,
                                        uint32_t stride, const GlobalTensor<float>& keypointsGm,
                                        const GlobalTensor<float>& scoresGm) {
        DataCopy(stats, statsGm[plane * parts * 8], parts * 8);
        PipeBarrier<PIPE_ALL>();
        float totalScore = stats.GetValue(0), totalMax = stats.GetValue(1);
        for (uint32_t p = 1; p < parts; ++p) {
            totalScore = (totalScore > stats.GetValue(p * 8) ? totalScore : stats.GetValue(p * 8));
            totalMax = (totalMax > stats.GetValue(p * 8 + 1) ? totalMax : stats.GetValue(p * 8 + 1));
        }
        for (uint32_t p = 0; p < parts; ++p) {
            factors.SetValue(p, stats.GetValue(p * 8 + 1) - totalMax);
        }
        PipeBarrier<PIPE_ALL>();
        Exp(factors, factors, parts > 8 ? 16 : 8);
        PipeBarrier<PIPE_ALL>();
        float den = 0.0f, nx = 0.0f, ny = 0.0f;
        for (uint32_t p = 0; p < parts; ++p) {
            const float factor = factors.GetValue(p);
            den += factor * stats.GetValue(p * 8 + 2);
            nx += factor * stats.GetValue(p * 8 + 3);
            ny += factor * stats.GetValue(p * 8 + 4);
        }
        StoreVectorResult(reduced, keypointsGm, scoresGm, plane,
                          RestoreCoordinate(nx, den, stride), RestoreCoordinate(ny, den, stride), totalScore);
    }

    __aicore__ inline float GetScore() const { return score; }
    __aicore__ inline float GetDenominator() const { return denominator; }

private:
    __aicore__ inline float SumCoordinate() {
        Mul(tmp, acc, coord, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduced, tmp, scratch, count);
        PipeBarrier<PIPE_ALL>();
        return reduced.GetValue(0);
    }

    LocalTensor<float> acc, tmp, coord, scratch, reduced;
    uint32_t width, rows, first, count;
    float score, maxLogit, denominator;
};
}
