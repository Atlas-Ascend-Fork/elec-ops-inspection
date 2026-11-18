/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

#pragma once
#include "kernel_operator.h"
#include "case5_steps.h"

namespace FusedKeypointBranches {
class KernelCase5SingleCore {
public:
    static constexpr uint32_t B_TPL = 4;
    static constexpr uint32_t H_TPL = 48;
    static constexpr uint32_t W_TPL = 64;
    static constexpr uint32_t HW_TPL = 3072;
    static constexpr uint32_t X_BUF_ELEMS = 3136;
    static constexpr uint32_t IN_BUF_ELEMS = 3136;
    static constexpr uint32_t H_MINUS_1_W = 3008;
    static constexpr uint8_t W_BLOCKS = 8;
    static constexpr uint32_t TOTAL_FEAT_ELEMS = 393216;
    static constexpr float STRIDE_F = 4.0f;
    static constexpr float INV_TAU = 1.0f;
    static constexpr uint64_t KP_PAIR_BITS = (static_cast<uint64_t>(0x42C00000ULL) << 32) | 0x43000000ULL;
    __aicore__ inline KernelCase5SingleCore() {}
    __aicore__ inline void Init(
        GM_ADDR feat,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR keypoints,
        GM_ADDR scores)
    {
        featGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(feat), TOTAL_FEAT_ELEMS);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), 1152);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), 4);
        keypointsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(keypoints), B_TPL * 8);
        scoresGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores), B_TPL * 4);
        pipe.InitBuffer(inBuf, IN_BUF_ELEMS * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0U) {
            return;
        }
        AscendC::LocalTensor<float> inLocal = inBuf.Get<float>();
        AscendC::LocalTensor<float> wLocal = inLocal[0];
        AscendC::LocalTensor<float> bLocal = inLocal[1152];
        AscendC::LocalTensor<float> outKpLocal = inLocal[1160];
        AscendC::LocalTensor<float> outScLocal = inLocal[1192];
        AscendC::LocalTensor<float> absLocal = inLocal[1216];
        AscendC::LocalTensor<float> reduceTmpLocal = inLocal[2368];
        AscendC::LocalTensor<float> scalarLocal = inLocal[2496];
        AscendC::DataCopy(wLocal, weightGm, 1152);
        AscendC::DataCopyPad(bLocal, biasGm,
            AscendC::DataCopyExtParams{1, 4U * sizeof(float), 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        InitCase5UniformKeypoints<KP_PAIR_BITS>(outKpLocal);
        AscendC::PipeBarrier<PIPE_ALL>();
        ReduceCase5WeightMagnitude(absLocal, wLocal, reduceTmpLocal, scalarLocal);
        InitCase5UniformScores(outScLocal, bLocal);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (scalarLocal.GetValue(0) == 0.0f) {
            AscendC::DataCopy(keypointsGm, outKpLocal, B_TPL * 8U);
            AscendC::DataCopy(scoresGm, outScLocal, B_TPL * 4U);
            return;
        }
        RunDense(inLocal);
    }
private:
    struct DenseTensors {
        AscendC::LocalTensor<float> wFull;
        AscendC::LocalTensor<float> bFull;
        AscendC::LocalTensor<float> xLocal;
        AscendC::LocalTensor<float> xLocal0_base;
        AscendC::LocalTensor<float> xLocal0;
        AscendC::LocalTensor<float> xLocal2;
        AscendC::LocalTensor<float> shiftLocal;
        AscendC::LocalTensor<float> uCoordLocal;
        AscendC::LocalTensor<uint32_t> rowOff0Local;
        AscendC::LocalTensor<uint32_t> rowOff2Local;
        AscendC::LocalTensor<int32_t> off0I32;
        AscendC::LocalTensor<int32_t> off2I32;
        AscendC::LocalTensor<float> maskLeftLocal;
        AscendC::LocalTensor<float> maskRightLocal;
        AscendC::LocalTensor<float> redTmp;
        AscendC::LocalTensor<float> scTmp;
        AscendC::LocalTensor<float> kpOut;
        AscendC::LocalTensor<float> scOut;
    };

    __aicore__ inline void AllocateDenseBuffers()
    {
        pipe.InitBuffer(wBuf, 1152 * sizeof(float));
        pipe.InitBuffer(bBuf, 8 * sizeof(float));
        pipe.InitBuffer(xBuf, X_BUF_ELEMS * sizeof(float));
        pipe.InitBuffer(xBuf0, 3264 * sizeof(float));
        pipe.InitBuffer(xBuf2, 3200 * sizeof(float));
        pipe.InitBuffer(shiftBuf, 3072 * sizeof(float));
        pipe.InitBuffer(uCoordBuf, W_TPL * sizeof(float));
        pipe.InitBuffer(rowOff0Buf, HW_TPL * sizeof(uint32_t));
        pipe.InitBuffer(rowOff2Buf, HW_TPL * sizeof(uint32_t));
        pipe.InitBuffer(maskLeftBuf, 8 * sizeof(float));
        pipe.InitBuffer(maskRightBuf, 8 * sizeof(float));
        pipe.InitBuffer(reduceTmpBuf, 256 * sizeof(float));
        pipe.InitBuffer(scalarBuf, 16 * sizeof(float));
        pipe.InitBuffer(outKpBuf, 32 * sizeof(float));
        pipe.InitBuffer(outScBuf, 16 * sizeof(float));
    }

    __aicore__ inline void BindDenseTensors(DenseTensors& local)
    {
        local.wFull = wBuf.Get<float>();
        local.bFull = bBuf.Get<float>();
        local.xLocal = xBuf.Get<float>();
        local.xLocal0_base = xBuf0.Get<float>();
        local.xLocal0 = local.xLocal0_base[8];
        local.xLocal2 = xBuf2.Get<float>();
        local.shiftLocal = shiftBuf.Get<float>();
        local.uCoordLocal = uCoordBuf.Get<float>();
        local.rowOff0Local = rowOff0Buf.Get<uint32_t>();
        local.rowOff2Local = rowOff2Buf.Get<uint32_t>();
        local.off0I32 = local.rowOff0Local.ReinterpretCast<int32_t>();
        local.off2I32 = local.rowOff2Local.ReinterpretCast<int32_t>();
        local.maskLeftLocal = maskLeftBuf.Get<float>();
        local.maskRightLocal = maskRightBuf.Get<float>();
        local.redTmp = reduceTmpBuf.Get<float>();
        local.scTmp = scalarBuf.Get<float>();
        local.kpOut = outKpBuf.Get<float>();
        local.scOut = outScBuf.Get<float>();
    }

    __aicore__ inline void InitDenseCoordinates(DenseTensors& local)
    {
        AscendC::DataCopy(local.wFull, weightGm, 1152);
        AscendC::DataCopyPad(local.bFull, biasGm,
            AscendC::DataCopyExtParams{1, 4U * sizeof(float), 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        for (uint32_t u = 0; u < 8; ++u) {
            local.uCoordLocal.SetValue(u, static_cast<float>(static_cast<int32_t>(u)));
            local.off2I32.SetValue(u, static_cast<int32_t>((u + 1U) * sizeof(float)));
            local.maskLeftLocal.SetValue(u, (u == 0) ? 0.0f : 1.0f);
            local.maskRightLocal.SetValue(u, (u == 7) ? 0.0f : 1.0f);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t step = 8; step < W_TPL; step <<= 1) {
            AscendC::Adds<float>(local.uCoordLocal[step], local.uCoordLocal[0],
                static_cast<float>(static_cast<int32_t>(step)), static_cast<int32_t>(step));
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t step = 8; step < HW_TPL; step <<= 1) {
            uint32_t cnt = (step <= HW_TPL - step) ? step : (HW_TPL - step);
            AscendC::Adds<int32_t>(local.off2I32[step], local.off2I32[0],
                static_cast<int32_t>(step * sizeof(float)), static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Adds<int32_t>(local.off0I32, local.off2I32, 24, static_cast<int32_t>(HW_TPL));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateDenseChannel(DenseTensors& local,
        AscendC::LocalTensor<float>& wkLocal, uint32_t baseGmOffset, uint32_t c)
    {
        uint32_t gmOffset = baseGmOffset + c * HW_TPL;
        uint32_t wBase = c * 9;
        AscendC::DataCopy(local.shiftLocal, featGm[gmOffset], HW_TPL);
        AscendC::PipeBarrier<PIPE_ALL>();
        float w00 = wkLocal.GetValue(wBase + 0);
        float w01 = wkLocal.GetValue(wBase + 1);
        float w02 = wkLocal.GetValue(wBase + 2);
        float w10 = wkLocal.GetValue(wBase + 3);
        float w11 = wkLocal.GetValue(wBase + 4);
        float w12 = wkLocal.GetValue(wBase + 5);
        float w20 = wkLocal.GetValue(wBase + 6);
        float w21 = wkLocal.GetValue(wBase + 7);
        float w22 = wkLocal.GetValue(wBase + 8);
        AscendC::Axpy<float, float>(local.xLocal0[W_TPL], local.shiftLocal[0], w00, static_cast<int32_t>(H_MINUS_1_W));
        AscendC::Axpy<float, float>(local.xLocal[W_TPL], local.shiftLocal[0], w01, static_cast<int32_t>(H_MINUS_1_W));
        AscendC::Axpy<float, float>(local.xLocal2[W_TPL], local.shiftLocal[0], w02, static_cast<int32_t>(H_MINUS_1_W));
        AscendC::Axpy<float, float>(local.xLocal0[0], local.shiftLocal[0], w10, static_cast<int32_t>(HW_TPL));
        AscendC::Axpy<float, float>(local.xLocal[0], local.shiftLocal[0], w11, static_cast<int32_t>(HW_TPL));
        AscendC::Axpy<float, float>(local.xLocal2[0], local.shiftLocal[0], w12, static_cast<int32_t>(HW_TPL));
        AscendC::Axpy<float, float>(local.xLocal0[0], local.shiftLocal[W_TPL], w20, static_cast<int32_t>(H_MINUS_1_W));
        AscendC::Axpy<float, float>(local.xLocal[0], local.shiftLocal[W_TPL], w21, static_cast<int32_t>(H_MINUS_1_W));
        AscendC::Axpy<float, float>(local.xLocal2[0], local.shiftLocal[W_TPL], w22, static_cast<int32_t>(H_MINUS_1_W));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AccumulateDenseHeatmap(DenseTensors& local, uint32_t bk)
    {
        uint32_t b = bk >> 2;
        uint32_t k = bk & 3U;
        AscendC::LocalTensor<float> wkLocal = local.wFull[k * 288];
        float bVal = local.bFull.GetValue(k);
        AscendC::Duplicate<float>(local.xLocal, bVal, static_cast<int32_t>(HW_TPL));
        AscendC::Duplicate<float>(local.xLocal0_base, 0.0f, static_cast<int32_t>(HW_TPL + 8U));
        AscendC::Duplicate<float>(local.xLocal2, 0.0f, static_cast<int32_t>(HW_TPL + 8U));
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t baseGmOffset = b * 32 * HW_TPL;
        for (uint32_t c = 0; c < 32; ++c) {
            AccumulateDenseChannel(local, wkLocal, baseGmOffset, c);
        }
    }

    __aicore__ inline void MergeDensePlanes(DenseTensors& local,
        AscendC::LocalTensor<float>& inLocal, const AscendC::BinaryRepeatParams& maskRepParams)
    {
        AscendC::Mul<float>(
            local.xLocal0[W_TPL - 8],
            local.xLocal0[W_TPL - 8],
            local.maskRightLocal,
            static_cast<uint64_t>(8),
            static_cast<uint8_t>(H_TPL),
            maskRepParams);
        AscendC::Mul<float>(
            local.xLocal2[0],
            local.xLocal2[0],
            local.maskLeftLocal,
            static_cast<uint64_t>(8),
            static_cast<uint8_t>(H_TPL),
            maskRepParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(inLocal, local.xLocal0_base, local.rowOff0Local, 0U, HW_TPL);
        AscendC::Gather(local.shiftLocal, local.xLocal2, local.rowOff2Local, 0U, HW_TPL);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(local.xLocal, local.xLocal, inLocal, static_cast<int32_t>(HW_TPL));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(local.xLocal, local.xLocal, local.shiftLocal, static_cast<int32_t>(HW_TPL));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeDenseHeatmap(DenseTensors& local, uint32_t bk)
    {
        AscendC::ReduceMax<float>(local.scTmp, local.xLocal, local.shiftLocal, static_cast<int32_t>(HW_TPL), false);
        AscendC::PipeBarrier<PIPE_ALL>();
        float maxX = local.scTmp.GetValue(0);
        local.scOut.SetValue(bk, maxX);
        AscendC::Adds<float>(local.shiftLocal, local.xLocal, -maxX, static_cast<int32_t>(HW_TPL));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(local.shiftLocal, local.shiftLocal, INV_TAU, static_cast<int32_t>(HW_TPL));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float>(local.shiftLocal, local.shiftLocal, static_cast<int32_t>(HW_TPL));
        AscendC::PipeBarrier<PIPE_V>();
        FoldCase5Rows<H_TPL, W_TPL>(local.xLocal, local.shiftLocal);
        AscendC::ReduceSum<float, true>(local.scTmp, local.shiftLocal[0], local.redTmp, static_cast<int32_t>(W_TPL));
        AscendC::PipeBarrier<PIPE_ALL>();
        float denom = local.scTmp.GetValue(0);
        if (denom == 0.0f) {
            denom = 1.0f;
        }
        AscendC::ReduceSum<float, true>(local.scTmp, local.xLocal[0], local.redTmp, static_cast<int32_t>(W_TPL));
        AscendC::PipeBarrier<PIPE_ALL>();
        float sumV = local.scTmp.GetValue(0);
        AscendC::Mul<float>(local.shiftLocal[0], local.shiftLocal[0], local.uCoordLocal, static_cast<int32_t>(W_TPL));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float, true>(local.scTmp, local.shiftLocal[0], local.redTmp, static_cast<int32_t>(W_TPL));
        AscendC::PipeBarrier<PIPE_ALL>();
        float sumU = local.scTmp.GetValue(0);
        float xFeat = sumU / denom;
        float yFeat = sumV / denom;
        local.kpOut.SetValue(bk * 2, (xFeat + 0.5f) * STRIDE_F);
        local.kpOut.SetValue(bk * 2 + 1, (yFeat + 0.5f) * STRIDE_F);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunDense(AscendC::LocalTensor<float>& inLocal)
    {
        AllocateDenseBuffers();
        DenseTensors local;
        BindDenseTensors(local);
        InitDenseCoordinates(local);
        AscendC::BinaryRepeatParams maskRepParams{1, 1, 1, W_BLOCKS, W_BLOCKS, 0};
        for (uint32_t bk = 0; bk < 4U * B_TPL; ++bk) {
            AccumulateDenseHeatmap(local, bk);
            MergeDensePlanes(local, inLocal, maskRepParams);
            DecodeDenseHeatmap(local, bk);
        }

        AscendC::DataCopy(keypointsGm, local.kpOut, B_TPL * 8U);
        AscendC::DataCopy(scoresGm, local.scOut, B_TPL * 4U);
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shiftBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> uCoordBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowOff0Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowOff2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskLeftBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskRightBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outKpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outScBuf;
    AscendC::GlobalTensor<float> featGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> keypointsGm;
    AscendC::GlobalTensor<float> scoresGm;
};
}
