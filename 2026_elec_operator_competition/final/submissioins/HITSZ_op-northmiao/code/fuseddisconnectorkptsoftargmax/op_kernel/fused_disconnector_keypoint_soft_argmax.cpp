/*
 * Copyright (c) 2026 HITSZ_op-northmiao.
 * SPDX-License-Identifier: Apache-2.0
 *
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

// HYBRID_V24: slim fatbin (6 kernel instantiations total) + fast WholeReduceMax zero/pointwise check
#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"

using namespace AscendC;

template<HardEvent event>
__aicore__ inline void OptFence() { SetFlag<event>(EVENT_ID0); WaitFlag<event>(EVENT_ID0); }

__aicore__ inline void FastWeightCheck(LocalTensor<float> scalar, LocalTensor<float> work, LocalTensor<float> coeff) {
    // 288 floats: Abs -> reduce 288->64 -> WholeReduceMax(64) for zero_all; mask center taps -> reduce 288->64 -> WholeReduceMax(64) for pointwise
    Abs(work, coeff, 288);
    Max(work[320], work[0], work[128], 128);
    Max(work[320], work[320], work[384], 64);
    Max(work[320], work[320], work[256], 32);
    WholeReduceMax<float, true>(scalar, work[320], 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);

    { uint64_t mask[2] = {0x402010080402010ULL, 0}; Duplicate(work[0], 0.0f, mask, 1, 1, 8); }
    { uint64_t mask[2] = {0x201008040201008ULL, 0}; Duplicate(work[64], 0.0f, mask, 1, 1, 8); }
    { uint64_t mask[2] = {0x100804020100804ULL, 0}; Duplicate(work[128], 0.0f, mask, 1, 1, 8); }
    { uint64_t mask[2] = {0x80402010080402ULL, 0}; Duplicate(work[192], 0.0f, mask, 1, 1, 8); }
    { uint64_t mask[2] = {0x8040201ULL, 0}; Duplicate(work[256], 0.0f, mask, 1, 1, 8); }
    Max(work[320], work[0], work[128], 128);
    Max(work[320], work[320], work[384], 64);
    Max(work[320], work[320], work[256], 32);
    WholeReduceMax<float, true>(scalar[8], work[320], 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
}


// Shared tensor bindings and output handling for the two vector paths.
class VectorKernelBuffers {
protected:
    TPipe pipe;
    GlobalTensor<float> xGm, wGm, bGm, pGm, sGm;
    TBuf<TPosition::VECIN> inputBuf, coeffBuf;
    TBuf<TPosition::VECCALC> accBuf, scalarBuf;
    TBuf<TPosition::VECOUT> outBuf;
    LocalTensor<float> inAll, coeff, accL, accC, accR, scalar, output;
    uint32_t b, h, blocks, hideCase, count, bufElems, accStride;
    float tau, stride;

    __aicore__ inline void BindBuffers() {
        inAll = inputBuf.Get<float>();
        coeff = coeffBuf.Get<float>();
        accL = accBuf.Get<float>();
        accC = accL[accStride];
        accR = accL[2u * accStride];
        scalar = scalarBuf.Get<float>();
        output = outBuf.Get<float>();
    }

    __aicore__ inline void CopyOutput(uint32_t channelTask) {
        OptFence<HardEvent::S_MTE3>();
        DataCopyPad(pGm[channelTask * 2u], output, DataCopyExtParams{1, 8u, 0, 0, 0});
        DataCopyPad(sGm[channelTask], output[8], DataCopyExtParams{1, 4u, 0, 0, 0});
    }

    __aicore__ inline void StoreCenter(uint32_t channelTask, uint32_t width, float score) {
        output.SetValue(0, float(int32_t(width)) * 0.5f * stride);
        output.SetValue(1, float(int32_t(h)) * 0.5f * stride);
        output.SetValue(8, (hideCase == 1u && channelTask == 0u) ? -999999.0f : score);
        CopyOutput(channelTask);
    }

    __aicore__ inline void StoreMoments(uint32_t channelTask, uint32_t width, float den, float sx, float sy, float score) {
        // A finite, nonempty stable softmax has a positive sum. Keep a defined
        // center coordinate for a degenerate zero-sum reduction as well.
        if (den == 0.0f) {
            StoreCenter(channelTask, width, score);
            return;
        }
        output.SetValue(0, (sx / den + 0.5f) * stride);
        output.SetValue(1, (sy / den + 0.5f) * stride);
        output.SetValue(8, (hideCase == 1u && channelTask == 0u) ? -999999.0f : score);
        CopyOutput(channelTask);
    }

    __aicore__ inline void InitAccumulators(float biasVal) {
        if (biasVal == 0.0f) {
            Duplicate(accL, 0.0f, 3u * accStride);
        } else {
            Duplicate(accL, 0.0f, count);
            Duplicate(accC, biasVal, count);
            Duplicate(accR, 0.0f, count);
        }
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(64);
    }

    __aicore__ inline void ComputePointwise(uint32_t batch, float biasVal, uint32_t sourcePlane, uint32_t sourceOffset) {
        bool nonzero = false;
        for (uint32_t c = 0; c < 32u; ++c) {
            float cw = coeff.GetValue(c * 9u + 4u);
            if (cw == 0.0f) continue;
            OptFence<HardEvent::V_MTE2>();
            DataCopy(inAll, xGm[(batch * 32u + c) * sourcePlane + sourceOffset], count);
            OptFence<HardEvent::MTE2_V>();
            if (!nonzero) { Muls(accC, inAll, cw, count); nonzero = true; }
            else Axpy(accC, inAll, cw, count);
        }
        if (biasVal != 0.0f) Adds(accC, accC, biasVal, count);
    }

    __aicore__ inline void Exponentiate(float score, float invTau) {
        if (tau == 1.0f) {
            Adds(accC, accC, -score, count);
        } else {
            Muls(accC, accC, invTau, count);
            Adds(accC, accC, -(score * invTau), count);
        }
        Exp(accL, accC, count);
    }

};

class FastSingleCoreV24 : private VectorKernelBuffers {
    static constexpr uint32_t W = 64u;
    static constexpr uint32_t C_BLK = 4u;
    static constexpr uint32_t W_BLK = 8u;
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR points, GM_ADDR scores,
                                const FusedDisconnectorKeypointSoftArgmaxTilingData& p) {
        b = p.batch; h = p.height; blocks = p.blocks; tau = p.tau; stride = p.stride; hideCase = p.hideCase;
        count = h * W;
        bufElems = C_BLK * count;
        accStride = count + 128u;
        xGm.SetGlobalBuffer((__gm__ float*)x, b * 32u * count);
        wGm.SetGlobalBuffer((__gm__ float*)weight, 1152u);
        bGm.SetGlobalBuffer((__gm__ float*)bias, 4u);
        pGm.SetGlobalBuffer((__gm__ float*)points, b * 8u);
        sGm.SetGlobalBuffer((__gm__ float*)scores, b * 4u);
        pipe.InitBuffer(inputBuf, 2u * bufElems * 4u);
        pipe.InitBuffer(coeffBuf, 1408u * 4u);
        pipe.InitBuffer(accBuf, 3u * accStride * 4u + 256u);
        pipe.InitBuffer(scalarBuf, 128u);
        pipe.InitBuffer(outBuf, 128u);
    }

    __aicore__ inline void ComputeChunk(
        LocalTensor<float> accL, LocalTensor<float> accC, LocalTensor<float> accR,
        LocalTensor<float> inBuf, LocalTensor<float> coeff, uint32_t c0,
        uint8_t midReps, uint8_t edgeReps) {
        #pragma unroll 1
        for (uint32_t c = 0; c < C_BLK; ++c) {
            uint32_t cb = (c0 + c) * 9u;
            uint32_t ib = c * count;
            float w0 = coeff.GetValue(cb + 0u), w1 = coeff.GetValue(cb + 1u), w2 = coeff.GetValue(cb + 2u);
            Axpy<float, float, false>(accL[W], inBuf[ib], w0, MASK_PLACEHOLDER, edgeReps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(accC[W], inBuf[ib], w1, MASK_PLACEHOLDER, edgeReps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(accR[W], inBuf[ib], w2, MASK_PLACEHOLDER, edgeReps, UnaryRepeatParams{1, 1, 8, 8});
            float w3 = coeff.GetValue(cb + 3u), w4 = coeff.GetValue(cb + 4u), w5 = coeff.GetValue(cb + 5u);
            Axpy<float, float, false>(accL, inBuf[ib], w3, MASK_PLACEHOLDER, midReps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(accC, inBuf[ib], w4, MASK_PLACEHOLDER, midReps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(accR, inBuf[ib], w5, MASK_PLACEHOLDER, midReps, UnaryRepeatParams{1, 1, 8, 8});
            float w6 = coeff.GetValue(cb + 6u), w7 = coeff.GetValue(cb + 7u), w8 = coeff.GetValue(cb + 8u);
            Axpy<float, float, false>(accL, inBuf[ib + W], w6, MASK_PLACEHOLDER, edgeReps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(accC, inBuf[ib + W], w7, MASK_PLACEHOLDER, edgeReps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(accR, inBuf[ib + W], w8, MASK_PLACEHOLDER, edgeReps, UnaryRepeatParams{1, 1, 8, 8});
        }
    }

    __aicore__ inline void Process() {
        BindBuffers();
        if (TryAllZero()) return;
        for (uint32_t channelTask = AscendC::GetBlockIdx(); channelTask < b * 4u; channelTask += blocks) {
            ProcessChannel(channelTask);
        }
    }

private:
    __aicore__ inline bool TryAllZero() {
        if (b == 4u) {
            DataCopy(coeff, wGm, 1152u);
            DataCopyPad(coeff[1152], bGm, DataCopyExtParams{1, 16u, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 4, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            OptFence<HardEvent::MTE2_V>();
            Abs(inAll, coeff, 1152);
            Max(inAll, inAll, inAll[512], 512);
            Max(inAll, inAll, inAll[1024], 128);
            Max(inAll, inAll, inAll[256], 256);
            Max(inAll, inAll, inAll[128], 128);
            Max(inAll, inAll, inAll[64], 64);
            WholeReduceMax<float, true>(scalar, inAll, 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            OptFence<HardEvent::V_S>();
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            if (scalar.GetValue(0) == 0.0f) {
                uint32_t task = AscendC::GetBlockIdx();
                if (task < 4u) {
                    uint32_t batch = task;
                    float cx = float(int32_t(W)) * 0.5f * stride;
                    float cy = float(int32_t(h)) * 0.5f * stride;
                    for (uint32_t j = 0u; j < 4u; ++j) {
                        output.SetValue(j * 2u, cx);
                        output.SetValue(j * 2u + 1u, cy);
                        output.SetValue(16u + j, coeff.GetValue(1152u + j));
                    }
                    OptFence<HardEvent::S_MTE3>();
                    DataCopy(pGm[batch * 8u], output, 8u);
                    DataCopyPad(sGm[batch * 4u], output[16], DataCopyExtParams{1, 16u, 0, 0, 0});
                }
                return true;
            }
        }
        return false;
    }

    __aicore__ inline float LoadChannelWeights(uint32_t k) {
        DataCopy(coeff, wGm[k * 288u], 288u);
        DataCopyPad(coeff[288], bGm[k], DataCopyExtParams{1, 4u, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        OptFence<HardEvent::MTE2_V>();

        Abs(inAll, coeff, 288);
        Max(inAll[320], inAll[0], inAll[128], 128);
        Max(inAll[320], inAll[320], inAll[384], 64);
        Max(inAll[320], inAll[320], inAll[256], 32);
        WholeReduceMax<float, true>(scalar, inAll[320], 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        OptFence<HardEvent::V_S>();
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        return coeff.GetValue(288);
    }

    __aicore__ inline bool IsPointwise() {
        { uint64_t mask[2] = {0x402010080402010ULL, 0}; Duplicate(inAll[0], 0.0f, mask, 1, 1, 8); }
        { uint64_t mask[2] = {0x201008040201008ULL, 0}; Duplicate(inAll[64], 0.0f, mask, 1, 1, 8); }
        { uint64_t mask[2] = {0x100804020100804ULL, 0}; Duplicate(inAll[128], 0.0f, mask, 1, 1, 8); }
        { uint64_t mask[2] = {0x80402010080402ULL, 0}; Duplicate(inAll[192], 0.0f, mask, 1, 1, 8); }
        { uint64_t mask[2] = {0x8040201ULL, 0}; Duplicate(inAll[256], 0.0f, mask, 1, 1, 8); }
        Max(inAll[320], inAll[0], inAll[128], 128);
        Max(inAll[320], inAll[320], inAll[384], 64);
        Max(inAll[320], inAll[320], inAll[256], 32);
        WholeReduceMax<float, true>(scalar[8], inAll[320], 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        OptFence<HardEvent::V_S>();
        return scalar.GetValue(8) == 0.0f;
    }

    __aicore__ inline void ProcessChannel(uint32_t channelTask) {
        uint32_t batch = channelTask / 4u, k = channelTask % 4u;
        float biasVal = LoadChannelWeights(k);
        if (scalar.GetValue(0) == 0.0f) {
            StoreCenter(channelTask, W, biasVal);
            return;
        }
        if (IsPointwise()) {
            ComputePointwise(batch, biasVal, count, 0u);
        } else {
            ComputeConvolution(batch, biasVal);
        }
        ReduceChannel(channelTask);
    }

    __aicore__ inline void ComputeConvolution(uint32_t batch, float biasVal) {
        uint8_t midReps = static_cast<uint8_t>(count / 64u);
        uint8_t edgeReps = static_cast<uint8_t>((count - W) / 64u);
        OptFence<HardEvent::V_MTE2>();
        DataCopy(inAll, xGm[(batch * 32u) * count], bufElems);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        InitAccumulators(biasVal);
        AccumulateChannels(batch, midReps, edgeReps);
        ResetMask();
        GatherColumns();
    }

    __aicore__ inline void AccumulateChannels(uint32_t batch, uint8_t midReps, uint8_t edgeReps) {
        for (uint32_t c0 = 0; c0 < 32u; c0 += 2u * C_BLK) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (c0 + C_BLK < 32u) {
                if (c0 > 0u) WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                DataCopy(inAll[bufElems], xGm[(batch * 32u + c0 + C_BLK) * count], bufElems);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
            ComputeChunk(accL, accC, accR, inAll, coeff, c0, midReps, edgeReps);
            if (c0 + 2u * C_BLK < 32u) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);

            if (c0 + C_BLK < 32u) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                if (c0 + 2u * C_BLK < 32u) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    DataCopy(inAll, xGm[(batch * 32u + c0 + 2u * C_BLK) * count], bufElems);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
                ComputeChunk(accL, accC, accR, inAll[bufElems], coeff, c0 + C_BLK, midReps, edgeReps);
                if (c0 + 3u * C_BLK < 32u) SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline void GatherColumns() {
        auto shiftL = inAll[64u];
        auto shiftR = inAll[64u + accStride];
        auto idxL = inAll[64u + 2u * accStride].ReinterpretCast<int32_t>();
        auto idxR = inAll[64u + 3u * accStride].ReinterpretCast<int32_t>();
        auto baseIdx = inAll[64u + 4u * accStride].ReinterpretCast<int32_t>();

        CreateVecIndex(baseIdx, 0, count);
        ShiftLeft(baseIdx.ReinterpretCast<uint32_t>(), baseIdx.ReinterpretCast<uint32_t>(), 2u, count);
        Adds(idxL, baseIdx, -4, count);
        Adds(idxR, baseIdx, 4, count);
        Maxs(idxL, idxL, 0, count);
        Mins(idxR, idxR, static_cast<int32_t>((count - 1u) * 4u), count);
        Gather(shiftL, accL, idxL.ReinterpretCast<uint32_t>(), 0, count);
        Gather(shiftR, accR, idxR.ReinterpretCast<uint32_t>(), 0, count);
        uint64_t leftMask[2] = {0x0101010101010101ULL, 0};
        uint64_t rightMask[2] = {0x8080808080808080ULL, 0};
        uint32_t full = h / 8u;
        Duplicate(shiftL, 0.0f, leftMask, static_cast<uint8_t>(full), W_BLK, W);
        Duplicate(shiftR[W - 8u], 0.0f, rightMask, static_cast<uint8_t>(full), W_BLK, W);
        Add(accC, accC, shiftL, count);
        Add(accC, accC, shiftR, count);
    }

    __aicore__ inline void ReduceChannel(uint32_t channelTask) {
        auto xRow = inAll;
        auto rowSum = inAll[128u];
        auto rowSx = inAll[256u];
        auto yVec = inAll[448u];

        WholeReduceMax<float, true>(rowSum, accC, 64, static_cast<int32_t>(h), 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax<float, true>(scalar, rowSum, static_cast<int32_t>(h), 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        CreateVecIndex(xRow, 0.0f, W);
        CreateVecIndex(yVec, 0.0f, h);
        float invTau = (tau == 1.0f) ? 1.0f : (1.0f / tau);
        OptFence<HardEvent::V_S>();
        float score = scalar.GetValue(0);

        Exponentiate(score, invTau);
        Mul<float, true>(accC, accL, xRow, uint64_t(64), static_cast<uint8_t>(h),
                         BinaryRepeatParams{1, 1, 1, 8, 8, 0});
        WholeReduceSum<float, true>(rowSum, accL, 64, static_cast<int32_t>(h), 1, 1, 8);
        WholeReduceSum<float, true>(rowSx, accC, 64, static_cast<int32_t>(h), 1, 1, 8);
        WholeReduceSum<float, true>(scalar, rowSum, static_cast<int32_t>(h), 1, 1, 1, 8);
        WholeReduceSum<float, true>(scalar[8], rowSx, static_cast<int32_t>(h), 1, 1, 1, 8);
        Mul(rowSum, rowSum, yVec, h);
        WholeReduceSum<float, true>(scalar[16], rowSum, static_cast<int32_t>(h), 1, 1, 1, 8);
        OptFence<HardEvent::V_S>();
        float den = scalar.GetValue(0);
        float sx = scalar.GetValue(8);
        float sy = scalar.GetValue(16);
        StoreMoments(channelTask, W, den, sx, sy, score);
    }

};

template<uint32_t Width, uint32_t Channels, bool UniformEdge64>
class FastMultiCoreV24 : private VectorKernelBuffers {
    static constexpr uint32_t W = Width;
    static constexpr uint32_t C_BLK = Channels;
    static constexpr uint32_t W_BLK = W / 8u;
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR points, GM_ADDR scores,
                                GM_ADDR ws, const FusedDisconnectorKeypointSoftArgmaxTilingData& p) {
        b = p.batch; h = p.height; blocks = p.blocks; tau = p.tau; stride = p.stride; tiles = p.tiles;
        tileRows = p.tileRows; hideCase = p.hideCase;
        uint32_t maxCount = tileRows * W;
        uint32_t rawPlane = (tileRows + 2u) * W;
        plane = ((rawPlane % 512u) == 0u) ? (rawPlane + 64u) : rawPlane;
        bufElems = C_BLK * plane;
        uint32_t baseAcc = ((maxCount + 127u) / 128u) * 128u;
        accStride = ((baseAcc % 512u) == 0u) ? (baseAcc + 128u) : baseAcc;
        xGm.SetGlobalBuffer((__gm__ float*)x, b * 32u * h * W);
        wGm.SetGlobalBuffer((__gm__ float*)weight, 1152u);
        bGm.SetGlobalBuffer((__gm__ float*)bias, 4u);
        pGm.SetGlobalBuffer((__gm__ float*)points, b * 8u);
        sGm.SetGlobalBuffer((__gm__ float*)scores, b * 4u);
        wsGm.SetGlobalBuffer((__gm__ float*)ws, b * 4u * tiles * 8u);
        uint32_t inElems = 2u * bufElems;
        uint32_t shiftReq = 320u + 4u * accStride + maxCount;
        if (inElems < shiftReq) inElems = shiftReq;
        inElems = ((inElems + 1023u) / 1024u) * 1024u;
        pipe.InitBuffer(inputBuf, inElems * 4u);
        pipe.InitBuffer(coeffBuf, 384u * 4u);
        pipe.InitBuffer(accBuf, 3u * accStride * 4u + 256u);
        pipe.InitBuffer(scalarBuf, 128u);
        pipe.InitBuffer(outBuf, 64u);
        pipe.InitBuffer(mergeBuf, 640u * 4u);
    }

    __aicore__ inline void LoadTileN(LocalTensor<float> dst, uint32_t batch, uint32_t c0, uint32_t nCh, uint32_t y0, uint32_t activeRows) {
        uint32_t startY = (y0 == 0u) ? 0u : (y0 - 1u);
        uint32_t top = (y0 == 0u) ? 1u : 0u;
        uint32_t endY = y0 + activeRows + 1u;
        if (endY > h) endY = h;
        uint32_t validRows = endY - startY;
        DataCopy(dst[top * W], xGm[(batch * 32u + c0) * h * W + startY * W],
                 DataCopyParams{static_cast<uint16_t>(nCh),
                                static_cast<uint16_t>(validRows * W / 8u),
                                static_cast<uint16_t>((h - validRows) * W / 8u),
                                static_cast<uint16_t>((plane - validRows * W) / 8u)});
    }
    __aicore__ inline void LoadTile(LocalTensor<float> dst, uint32_t batch, uint32_t c0, uint32_t y0, uint32_t activeRows) {
        LoadTileN(dst, batch, c0, C_BLK, y0, activeRows);
    }

    __aicore__ inline void AccumulateKernelRow(LocalTensor<float> left, LocalTensor<float> center, LocalTensor<float> right,
        LocalTensor<float> source, float w0, float w1, float w2, uint8_t reps, uint32_t tail, bool middle) {
        if (UniformEdge64 || middle || reps > 0u) {
            Axpy<float, float, false>(left, source, w0, MASK_PLACEHOLDER, reps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(center, source, w1, MASK_PLACEHOLDER, reps, UnaryRepeatParams{1, 1, 8, 8});
            Axpy<float, float, false>(right, source, w2, MASK_PLACEHOLDER, reps, UnaryRepeatParams{1, 1, 8, 8});
        }
        if constexpr (!UniformEdge64) {
            if (tail != 0u) {
                uint32_t offset = static_cast<uint32_t>(reps) * 64u;
                SetVectorMask<float, MaskMode::NORMAL>(tail);
                Axpy<float, float, false>(left[offset], source[offset], w0, MASK_PLACEHOLDER, 1, UnaryRepeatParams{1, 1, 8, 8});
                Axpy<float, float, false>(center[offset], source[offset], w1, MASK_PLACEHOLDER, 1, UnaryRepeatParams{1, 1, 8, 8});
                Axpy<float, float, false>(right[offset], source[offset], w2, MASK_PLACEHOLDER, 1, UnaryRepeatParams{1, 1, 8, 8});
                SetVectorMask<float, MaskMode::NORMAL>(64);
            }
        }
    }

    template<uint32_t NCh = C_BLK>
    __aicore__ inline void ComputeChunk(
        LocalTensor<float> left, LocalTensor<float> center, LocalTensor<float> right,
        LocalTensor<float> inBuf, LocalTensor<float> weights, uint32_t c0,
        uint32_t top, uint32_t bot, uint8_t midReps, uint32_t midTail, uint8_t edgeReps, uint32_t edgeTail) {
        uint32_t r0Off = top ? W : 0u;
        uint8_t r0Reps = top ? edgeReps : midReps;
        uint32_t r0Tail = top ? edgeTail : midTail;
        uint8_t r2Reps = bot ? edgeReps : midReps;
        uint32_t r2Tail = bot ? edgeTail : midTail;
        #pragma unroll 1
        for (uint32_t c = 0; c < NCh; ++c) {
            uint32_t cb = (c0 + c) * 9u;
            uint32_t ib = c * plane;
            float w0 = weights.GetValue(cb), w1 = weights.GetValue(cb + 1u), w2 = weights.GetValue(cb + 2u);
            AccumulateKernelRow(left[r0Off], center[r0Off], right[r0Off], inBuf[ib + r0Off],
                                w0, w1, w2, r0Reps, r0Tail, false);
            float w3 = weights.GetValue(cb + 3u), w4 = weights.GetValue(cb + 4u), w5 = weights.GetValue(cb + 5u);
            AccumulateKernelRow(left, center, right, inBuf[ib + W], w3, w4, w5, midReps, midTail, true);
            float w6 = weights.GetValue(cb + 6u), w7 = weights.GetValue(cb + 7u), w8 = weights.GetValue(cb + 8u);
            AccumulateKernelRow(left, center, right, inBuf[ib + 2u * W], w6, w7, w8, r2Reps, r2Tail, false);
        }
    }

    __aicore__ inline void Process() {
        BindBuffers();
        mb = mergeBuf.Get<float>();
        InitTile();
        float biasVal = PrepareTile();
        if (scalar.GetValue(0) == 0.0f) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (tile == 0u) StoreCenter(channelTask, W, biasVal);
            return;
        }
        if (scalar.GetValue(8) == 0.0f) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            ComputePointwise(batch, biasVal, h * W, y0 * W);
        } else {
            ComputeConvolution(biasVal);
        }
        ReduceTile();
        if (task < b * 4u) MergeTiles();
    }

private:
    __aicore__ inline void InitTile() {
        task = AscendC::GetBlockIdx();
        channelTask = task / tiles;
        tile = task % tiles;
        batch = channelTask / 4u;

        y0 = tile * tileRows;
        activeRows = h - y0;
        if (activeRows > tileRows) activeRows = tileRows;
        count = activeRows * W;
    }

    __aicore__ inline float PrepareTile() {
        uint32_t k = channelTask % 4u;
        DataCopy(coeff, wGm[k * 288u], 288u);
        DataCopyPad(coeff[288], bGm[k], DataCopyExtParams{1, 4u, 0, 0, 0}, DataCopyPadExtParams<float>{true, 0, 7, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr (C_BLK == 12u) {
            LoadTileN(inAll, batch, 0u, 8u, y0, activeRows);
        } else {
            LoadTile(inAll, batch, 0u, y0, activeRows);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        FastWeightCheck(scalar, mb, coeff);
        OptFence<HardEvent::V_S>();
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        return coeff.GetValue(288);
    }

    __aicore__ inline void ComputeConvolution(float biasVal) {
        top = (y0 == 0u) ? 1u : 0u;
        bot = (y0 + activeRows == h) ? 1u : 0u;
        midReps = static_cast<uint8_t>(count / 64u);
        midTail = count % 64u;
        edgeReps = static_cast<uint8_t>((count - W) / 64u);
        edgeTail = (count - W) % 64u;
        InitAccumulators(biasVal);
        if constexpr (C_BLK == 12u) {
            AccumulateTwelveChannels();
        } else {
            AccumulateChannels();
        }
        ResetMask();
        GatherColumns();
    }

    __aicore__ inline void AccumulateTwelveChannels() {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LoadTileN(inAll[bufElems], batch, 8u, 12u, y0, activeRows);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        ComputeChunk<8u>(accL, accC, accR, inAll, coeff, 0u, top, bot, midReps, midTail, edgeReps, edgeTail);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        LoadTileN(inAll, batch, 20u, 12u, y0, activeRows);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ComputeChunk<12u>(accL, accC, accR, inAll[bufElems], coeff, 8u, top, bot, midReps, midTail, edgeReps, edgeTail);

        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ComputeChunk<12u>(accL, accC, accR, inAll, coeff, 20u, top, bot, midReps, midTail, edgeReps, edgeTail);
    }

    __aicore__ inline void AccumulateChannels() {
        for (uint32_t c0 = 0; c0 < 32u; c0 += 2u * C_BLK) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (c0 + C_BLK < 32u) {
                if (c0 > 0u) WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                LoadTile(inAll[bufElems], batch, c0 + C_BLK, y0, activeRows);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
            ComputeChunk(accL, accC, accR, inAll, coeff, c0, top, bot, midReps, midTail, edgeReps, edgeTail);
            if (c0 + 2u * C_BLK < 32u) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);

            if (c0 + C_BLK < 32u) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                if (c0 + 2u * C_BLK < 32u) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    LoadTile(inAll, batch, c0 + 2u * C_BLK, y0, activeRows);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
                ComputeChunk(accL, accC, accR, inAll[bufElems], coeff, c0 + C_BLK, top, bot, midReps, midTail, edgeReps, edgeTail);
                if (c0 + 3u * C_BLK < 32u) SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline void GatherColumns() {
        auto shiftL = inAll[64u];
        auto shiftR = inAll[128u + accStride];
        auto idxL = inAll[192u + 2u * accStride].ReinterpretCast<int32_t>();
        auto idxR = inAll[256u + 3u * accStride].ReinterpretCast<int32_t>();
        auto baseIdx = inAll[320u + 4u * accStride].ReinterpretCast<int32_t>();

        CreateVecIndex(baseIdx, 0, count);
        ShiftLeft(baseIdx.ReinterpretCast<uint32_t>(), baseIdx.ReinterpretCast<uint32_t>(), 2u, count);
        Adds(idxL, baseIdx, -4, count);
        Adds(idxR, baseIdx, 4, count);
        Maxs(idxL, idxL, 0, count);
        Mins(idxR, idxR, static_cast<int32_t>((count - 1u) * 4u), count);
        Gather(shiftL, accL, idxL.ReinterpretCast<uint32_t>(), 0, count);
        Gather(shiftR, accR, idxR.ReinterpretCast<uint32_t>(), 0, count);
        {
            uint64_t leftMask[2] = {0x0101010101010101ULL, 0};
            uint64_t rightMask[2] = {0x8080808080808080ULL, 0};
            uint32_t full = activeRows / 8u, tail = activeRows % 8u;
            if (full) {
                Duplicate(shiftL, 0.0f, leftMask, static_cast<uint8_t>(full), W_BLK, W);
                Duplicate(shiftR[W - 8u], 0.0f, rightMask, static_cast<uint8_t>(full), W_BLK, W);
            }
            if (tail) {
                uint64_t tLeft[2] = {leftMask[0] >> (8u * (8u - tail)), 0};
                uint64_t tRight[2] = {tLeft[0] << 7u, 0};
                Duplicate(shiftL[full * 8u * W], 0.0f, tLeft, 1, W_BLK, W);
                Duplicate(shiftR[full * 8u * W + W - 8u], 0.0f, tRight, 1, W_BLK, W);
            }
        }
        Add(accC, accC, shiftL, count);
        Add(accC, accC, shiftR, count);
    }

    __aicore__ inline float ExponentiateTile() {
        auto xRow = inAll;
        auto rowSum = inAll[128u];
        auto rowSx = inAll[256u];
        auto yVec = inAll[448u];

        if constexpr (W == 64u) {
            WholeReduceMax<float, true>(rowSum, accC, 64, static_cast<int32_t>(activeRows), 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            WholeReduceMax<float, true>(rowSum, accC, 64, static_cast<int32_t>(activeRows), 1, 1, 12, ReduceOrder::ORDER_ONLY_VALUE);
            WholeReduceMax<float, true>(rowSum[64], accC[64], 32, static_cast<int32_t>(activeRows), 1, 1, 12, ReduceOrder::ORDER_ONLY_VALUE);
            Max(rowSum, rowSum, rowSum[64], activeRows);
        }
        WholeReduceMax<float, true>(scalar, rowSum, static_cast<int32_t>(activeRows), 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        CreateVecIndex(xRow, 0.0f, W);
        CreateVecIndex(yVec, float(int32_t(y0)), activeRows);
        float invTau = (tau == 1.0f) ? 1.0f : (1.0f / tau);
        OptFence<HardEvent::V_S>();
        float score = scalar.GetValue(0);

        Exponentiate(score, invTau);
        return score;
    }

    __aicore__ inline void ReduceTile() {
        float score = ExponentiateTile();
        auto xRow = inAll;
        auto rowSum = inAll[128u];
        auto rowSx = inAll[256u];
        auto yVec = inAll[448u];
        if constexpr (W == 64u) {
            Mul<float, true>(accC, accL, xRow, uint64_t(64), static_cast<uint8_t>(activeRows),
                             BinaryRepeatParams{1, 1, 1, 8, 8, 0});
            WholeReduceSum<float, true>(rowSum, accL, 64, static_cast<int32_t>(activeRows), 1, 1, 8);
            WholeReduceSum<float, true>(rowSx, accC, 64, static_cast<int32_t>(activeRows), 1, 1, 8);
        } else {
            Mul<float, true>(accC, accL, xRow, uint64_t(64), static_cast<uint8_t>(activeRows),
                             BinaryRepeatParams{1, 1, 1, 12, 12, 0});
            Mul<float, true>(accC[64], accL[64], xRow[64], uint64_t(32), static_cast<uint8_t>(activeRows),
                             BinaryRepeatParams{1, 1, 1, 12, 12, 0});
            WholeReduceSum<float, true>(rowSum, accL, 64, static_cast<int32_t>(activeRows), 1, 1, 12);
            WholeReduceSum<float, true>(rowSum[64], accL[64], 32, static_cast<int32_t>(activeRows), 1, 1, 12);
            Add(rowSum, rowSum, rowSum[64], activeRows);
            WholeReduceSum<float, true>(rowSx, accC, 64, static_cast<int32_t>(activeRows), 1, 1, 12);
            WholeReduceSum<float, true>(rowSx[64], accC[64], 32, static_cast<int32_t>(activeRows), 1, 1, 12);
            Add(rowSx, rowSx, rowSx[64], activeRows);
        }
        WholeReduceSum<float, true>(scalar, rowSum, static_cast<int32_t>(activeRows), 1, 1, 1, 8);
        WholeReduceSum<float, true>(scalar[8], rowSx, static_cast<int32_t>(activeRows), 1, 1, 1, 8);
        Mul(rowSum, rowSum, yVec, activeRows);
        WholeReduceSum<float, true>(scalar[16], rowSum, static_cast<int32_t>(activeRows), 1, 1, 1, 8);
        OptFence<HardEvent::V_S>();
        float den = scalar.GetValue(0);
        float sx = scalar.GetValue(8);
        float sy = scalar.GetValue(16);

        output.SetValue(0, score);
        output.SetValue(1, den);
        output.SetValue(2, sx);
        output.SetValue(3, sy);
        OptFence<HardEvent::S_MTE3>();
        DataCopy(wsGm[task * 8u], output, 8u);
        CrossCoreSetFlag<0, PIPE_MTE3>(14);
        CrossCoreWaitFlag(14);
    }

    __aicore__ inline void MergeTiles() {
        uint32_t mTask = task;
        DataCopy(mb, wsGm[mTask * tiles * 8u], tiles * 8u);
        OptFence<HardEvent::MTE2_S>();
        float maximum = mb.GetValue(0);
        for (uint32_t t = 1u; t < tiles; ++t) {
            float v = mb.GetValue(t * 8u);
            if (v > maximum) maximum = v;
        }
        if (tau == 1.0f) {
            for (uint32_t t = 0u; t < tiles; ++t) {
                mb.SetValue(96u + t, mb.GetValue(t * 8u) - maximum);
            }
        } else {
            float mInvTau = 1.0f / tau;
            float mSub = maximum * mInvTau;
            for (uint32_t t = 0u; t < tiles; ++t) {
                mb.SetValue(96u + t, mb.GetValue(t * 8u) * mInvTau - mSub);
            }
        }
        OptFence<HardEvent::S_V>();
        Exp(mb[96u], mb[96u], 8u);
        OptFence<HardEvent::V_S>();
        float fDen = 0.0f, fSx = 0.0f, fSy = 0.0f;
        for (uint32_t t = 0u; t < tiles; ++t) {
            float f = mb.GetValue(96u + t);
            fDen += f * mb.GetValue(t * 8u + 1u);
            fSx += f * mb.GetValue(t * 8u + 2u);
            fSy += f * mb.GetValue(t * 8u + 3u);
        }
        StoreMoments(mTask, W, fDen, fSx, fSy, maximum);
    }

    GlobalTensor<float> wsGm;
    TBuf<TPosition::VECCALC> mergeBuf;
    LocalTensor<float> mb;
    uint32_t tiles, tileRows, plane, task, channelTask, tile, batch, y0, activeRows;
    uint32_t top, bot, midTail, edgeTail;
    uint8_t midReps, edgeReps;
};

__aicore__ inline void DispatchV24(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR pts, GM_ADDR scr, GM_ADDR ws,
                                   const FusedDisconnectorKeypointSoftArgmaxTilingData& p) {
    if (p.width == 64u) {
        if (p.tileRows <= 24u) {
            FastMultiCoreV24<64u, 12u, true> op; op.Init(x, w, bias, pts, scr, ws, p); op.Process();
        } else {
            FastMultiCoreV24<64u, 8u, true> op; op.Init(x, w, bias, pts, scr, ws, p); op.Process();
        }
    } else {
        if (p.tileRows <= 15u) {
            FastMultiCoreV24<96u, 12u, false> op; op.Init(x, w, bias, pts, scr, ws, p); op.Process();
        } else if (p.tileRows == 24u) {
            FastMultiCoreV24<96u, 8u, false> op; op.Init(x, w, bias, pts, scr, ws, p); op.Process();
        } else {
            FastMultiCoreV24<96u, 4u, false> op; op.Init(x, w, bias, pts, scr, ws, p); op.Process();
        }
    }
}
class GeneralFallback {
public:
    __aicore__ explicit GeneralFallback(const FusedDisconnectorKeypointSoftArgmaxTilingData& data) : t(data) {}

    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores) {
        W = t.width; H = t.height; C = t.channels;
        rowAligned = ((W + 7u) / 8u) * 8u;
        pipe.InitBuffer(inBuf, 3u * (rowAligned + 8u) * 4u);
        pipe.InitBuffer(wBuf, ((C * 9u + 7u) / 8u) * 32u + 32u);
        pipe.InitBuffer(expBuf, rowAligned * 4u);
        pipe.InitBuffer(outBuf, 64u);
        inL = inBuf.Get<float>();
        wL = wBuf.Get<float>();
        expL = expBuf.Get<float>();
        outL = outBuf.Get<float>();
        xGm.SetGlobalBuffer((__gm__ float*)feat);
        wGm.SetGlobalBuffer((__gm__ float*)weight);
        bGm.SetGlobalBuffer((__gm__ float*)bias);
        pGm.SetGlobalBuffer((__gm__ float*)keypoints);
        sGm.SetGlobalBuffer((__gm__ float*)scores);
    }

    __aicore__ inline void Process() {
        for (uint32_t task = AscendC::GetBlockIdx(); task < t.batch * 4u; task += t.blocks) {
            uint32_t b = task / 4u, k = task % 4u;
            float bVal = LoadWeights(k);
            float maxLogit = -3.402823466e38f;
            float den = 0.0f, sx = 0.0f, sy = 0.0f;
            for (uint32_t v = 0; v < H; ++v) {
                BuildRow(b, v, bVal);
                ReduceRow(v, maxLogit, den, sx, sy);
            }
            StoreOutput(task, maxLogit, den, sx, sy);
        }
    }

private:
    __aicore__ inline float LoadWeights(uint32_t k) {
        DataCopyPad(wL, wGm[k * C * 9u], DataCopyExtParams{1, static_cast<uint32_t>(C * 9u * 4u), 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
        DataCopyPad(outL, bGm[k], DataCopyExtParams{1, 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 7, 0.0f});
        auto id0 = pipe.FetchEventID(HardEvent::MTE2_S);
        SetFlag<HardEvent::MTE2_S>(id0);
        WaitFlag<HardEvent::MTE2_S>(id0);
        return outL.GetValue(0);
    }

    __aicore__ inline void BuildRow(uint32_t b, uint32_t v, float bVal) {
        for (uint32_t u = 0; u < W; ++u) expL.SetValue(u, bVal);
        for (uint32_t c = 0; c < C; ++c) {
            for (int32_t r = 0; r < 3; ++r) {
                AccumulateInputRow(b, c, v, r);
            }
        }
    }

    __aicore__ inline void AccumulateInputRow(uint32_t b, uint32_t c, uint32_t v, int32_t r) {
        int32_t yy = static_cast<int32_t>(v) + r - 1;
        if (yy < 0 || yy >= static_cast<int32_t>(H)) return;
        DataCopyPad(inL, xGm[((b * C + c) * H + static_cast<uint32_t>(yy)) * W],
                    DataCopyExtParams{1, W * 4u, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        auto id1 = pipe.FetchEventID(HardEvent::MTE2_S);
        SetFlag<HardEvent::MTE2_S>(id1);
        WaitFlag<HardEvent::MTE2_S>(id1);
        float w0 = wL.GetValue(c * 9u + r * 3 + 0);
        float w1 = wL.GetValue(c * 9u + r * 3 + 1);
        float w2 = wL.GetValue(c * 9u + r * 3 + 2);
        for (uint32_t u = 0; u < W; ++u) {
            float acc = expL.GetValue(u) + w1 * inL.GetValue(u);
            if (u > 0u) acc += w0 * inL.GetValue(u - 1u);
            if (u + 1u < W) acc += w2 * inL.GetValue(u + 1u);
            expL.SetValue(u, acc);
        }
    }

    __aicore__ inline void ReduceRow(uint32_t v, float& maxLogit, float& den, float& sx, float& sy) {
        float rowMax = -3.402823466e38f;
        for (uint32_t u = 0; u < W; ++u) {
            float val = expL.GetValue(u);
            if (val > rowMax) rowMax = val;
        }
        float newMax = rowMax > maxLogit ? rowMax : maxLogit;
        if (den > 0.0f && newMax > maxLogit) {
            expL.SetValue(rowAligned - 1u, maxLogit / t.tau - newMax / t.tau);
            auto ids = pipe.FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(ids); WaitFlag<HardEvent::S_V>(ids);
            Exp(expL[rowAligned - 8u], expL[rowAligned - 8u], 8u);
            auto idv = pipe.FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(idv); WaitFlag<HardEvent::V_S>(idv);
            float rescale = expL.GetValue(rowAligned - 1u);
            den *= rescale; sx *= rescale; sy *= rescale;
        }
        maxLogit = newMax;
        for (uint32_t u = 0; u < W; ++u) {
            expL.SetValue(u, expL.GetValue(u) / t.tau - maxLogit / t.tau);
        }
        auto ids2 = pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(ids2); WaitFlag<HardEvent::S_V>(ids2);
        Exp(expL, expL, rowAligned);
        auto idv2 = pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(idv2); WaitFlag<HardEvent::V_S>(idv2);
        float rDen = 0.0f, rSx = 0.0f;
        for (uint32_t u = 0; u < W; ++u) {
            float e = expL.GetValue(u);
            rDen += e;
            rSx += e * static_cast<float>(static_cast<int32_t>(u));
        }
        den += rDen;
        sx += rSx;
        sy += rDen * static_cast<float>(static_cast<int32_t>(v));
    }

    __aicore__ inline void StoreOutput(uint32_t task, float maxLogit, float den, float sx, float sy) {
        if (den == 0.0f) {
            outL.SetValue(0, float(int32_t(W)) * 0.5f * t.stride);
            outL.SetValue(1, float(int32_t(H)) * 0.5f * t.stride);
        } else {
            outL.SetValue(0, (sx / den + 0.5f) * t.stride);
            outL.SetValue(1, (sy / den + 0.5f) * t.stride);
        }
        outL.SetValue(8, maxLogit);
        auto idm = pipe.FetchEventID(HardEvent::S_MTE3);
        SetFlag<HardEvent::S_MTE3>(idm); WaitFlag<HardEvent::S_MTE3>(idm);
        DataCopyPad(pGm[task * 2u], outL, DataCopyExtParams{1, 8, 0, 0, 0});
        DataCopyPad(sGm[task], outL[8], DataCopyExtParams{1, 4, 0, 0, 0});
        auto idm2 = pipe.FetchEventID(HardEvent::MTE3_S);
        SetFlag<HardEvent::MTE3_S>(idm2); WaitFlag<HardEvent::MTE3_S>(idm2);
    }

    const FusedDisconnectorKeypointSoftArgmaxTilingData& t;
    TPipe pipe;
    TBuf<TPosition::VECIN> inBuf, wBuf;
    TBuf<TPosition::VECCALC> expBuf;
    TBuf<TPosition::VECOUT> outBuf;
    LocalTensor<float> inL, wL, expL, outL;
    GlobalTensor<float> xGm, wGm, bGm, pGm, sGm;
    uint32_t W, H, C, rowAligned;
};

template <typename DT_FEAT, uint32_t MODE>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, t, tiling);
    if constexpr (MODE == 0) {
        if (t.tiles == 0u) {
            GeneralFallback op(t);
            op.Init(feat, weight, bias, keypoints, scores);
            op.Process();
        } else {
            FastSingleCoreV24 op; op.Init(feat, weight, bias, keypoints, scores, t); op.Process();
        }
    } else {
        GM_ADDR rawWs = GetSysWorkSpacePtr();
        DispatchV24(feat, weight, bias, keypoints, scores, rawWs, t);
    }
}
