/*
 * Copyright (c) 2026 Qianqiuer.
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

#include "kernel_operator.h"
#include "kernel_common.h"
#include "spline_conv_tiling.h"
#include "tiling_key_spline_conv.h"

using namespace AscendC;

template <bool REUSE_CALLER_MASK>
__aicore__ inline __attribute__((always_inline)) void SplineBrcbUsingNormal64(
    const LocalTensor<float>& dst, const LocalTensor<float>& src,
    uint8_t repeats, const BrcbRepeatParams& params) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 && !defined(ASCENDC_CPU_DEBUG)
    if constexpr (REUSE_CALLER_MASK) {
        if ASCEND_IS_AIV {
            vbrcb((__ubuf__ uint32_t*)dst.GetPhyAddr(),
                  (__ubuf__ uint32_t*)src.GetPhyAddr(), params.dstBlkStride,
                  params.dstRepStride, repeats);
        }
    } else Brcb(dst, src, repeats, params);
#else
    Brcb(dst, src, repeats, params);
#endif
}

__aicore__ inline void PreloadSplineRouting(GM_ADDR edges, GM_ADDR pseudo) {
    GlobalTensor<uint64_t> edgeLine, coordLine;
    edgeLine.SetGlobalBuffer((__gm__ uint64_t*)edges);
    coordLine.SetGlobalBuffer((__gm__ uint64_t*)pseudo);
    DataCachePreload(edgeLine, int16_t{0});
    DataCachePreload(coordLine, int16_t{0});
}

__aicore__ inline void PreloadSplineHalfEdges(GM_ADDR edges) {
    GlobalTensor<uint64_t> line;
    line.SetGlobalBuffer((__gm__ uint64_t*)edges);
    DataCachePreload(line, int16_t{0});
}

__aicore__ inline void PreloadSplineHalfCoordinates(GM_ADDR pseudo) {
    GlobalTensor<uint64_t> line;
    line.SetGlobalBuffer((__gm__ uint64_t*)pseudo);
    DataCachePreload(line, int16_t{0});
}

template <uint32_t ROWS>
__aicore__ inline void PreloadSplineWeights(GM_ADDR weight) {
    constexpr uint32_t count = 4U * ROWS * 16U;
    LocalTensor<float> weights(TPosition::VECCALC, 0, count);
    GlobalTensor<float> source;
    source.SetGlobalBuffer((__gm__ float*)weight);
    DataCopy(weights, source, count);
}

template <uint32_t ROWS, uint16_t BLOCKS>
__aicore__ inline void PreloadSplineWeightsHalfBlocks(GM_ADDR weight) {
    constexpr uint32_t elements = 4U * ROWS * 16U;
    static_assert(elements % (8U * BLOCKS) == 0U);
    LocalTensor<float> weights(TPosition::VECCALC, 0, elements);
    GlobalTensor<float> source;
    source.SetGlobalBuffer((__gm__ float*)weight);
    const DataCopyParams copy{BLOCKS,
        static_cast<uint16_t>(elements / (8U * BLOCKS)), 0, 0};
    DataCopy(weights, source, copy);
}

template <uint32_t ROWS>
__aicore__ inline void PreloadSplineWeightsStreamed(GM_ADDR weight) {
    constexpr uint32_t matrixSize = ROWS * 16U;
    LocalTensor<float> weights(TPosition::VECCALC, 0, 4U * matrixSize);
    GlobalTensor<float> source;
    source.SetGlobalBuffer((__gm__ float*)weight);
    DataCopy(weights, source, matrixSize);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
    DataCopy(weights[matrixSize], source[matrixSize], matrixSize);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
    DataCopy(weights[2U * matrixSize], source[2U * matrixSize], matrixSize);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
    DataCopy(weights[3U * matrixSize], source[3U * matrixSize], matrixSize);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
}

template <uint32_t FIXED_INPUT, bool FIXED_LAYOUT = false>
__aicore__ inline SplineResidentTilingData LoadResidentTiling(GM_ADDR tiling) {
    GlobalTensor<uint32_t> descriptor;
    descriptor.SetGlobalBuffer((__gm__ uint32_t*)tiling, 8);
    SplineResidentTilingData data;
    data.nodes = descriptor.GetValue(0);
    data.edges = descriptor.GetValue(1);
    data.inChannels = FIXED_LAYOUT ? FIXED_INPUT : descriptor.GetValue(2);
    data.outChannels = FIXED_INPUT ? 16U : descriptor.GetValue(3);
    data.mean = descriptor.GetValue(4);
    data.owners = descriptor.GetValue(5);
    data.paddedInput = FIXED_INPUT ? FIXED_INPUT : descriptor.GetValue(6);
    data.ownerShift = descriptor.GetValue(7);
    return data;
}

template <uint32_t FIXED_INPUT, uint32_t FIXED_NODES = 0U, uint32_t MEAN_MODE = 0U>
__aicore__ inline SplineResidentTilingData LoadFullOwnerTiling(GM_ADDR tiling) {
    GlobalTensor<uint32_t> descriptor;
    descriptor.SetGlobalBuffer((__gm__ uint32_t*)tiling, 8);
    SplineResidentTilingData data;
    static_assert(MEAN_MODE <= 2U, "Valid aggregation mode");
    data.nodes = FIXED_NODES ? FIXED_NODES : descriptor.GetValue(0);
    data.edges = 8U;
    data.inChannels = FIXED_INPUT;
    data.outChannels = 16U;
    data.mean = MEAN_MODE ? MEAN_MODE - 1U : descriptor.GetValue(4);
    data.owners = 8U;
    data.paddedInput = FIXED_INPUT;
    data.ownerShift = 3U;
    return data;
}

template <uint32_t FIXED_INPUT, uint32_t FIXED_NODES, uint32_t MEAN_MODE>
__aicore__ inline SplineResidentTilingData LoadHalfOwnerTiling() {
    return SplineResidentTilingData{FIXED_NODES, 4U, FIXED_INPUT, 16U,
        MEAN_MODE - 1U, 4U, FIXED_INPUT, 2U};
}

template <uint32_t FIXED_INPUT, bool STREAM_WEIGHTS = false>
__aicore__ inline __attribute__((always_inline)) LocalTensor<float> SplineReduceTree(
    LocalTensor<float> blend, LocalTensor<float> expanded) {
    constexpr uint32_t channels = 16U;
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    LocalTensor<float> reduced = blend;
    #pragma unroll
    for (uint32_t half = FIXED_INPUT / 2U; half > 1U; half /= 2U) {
        const uint32_t count = half * channels;
        if constexpr (STREAM_WEIGHTS) {
            if (count < 64U) {
                Add<float, false>(expanded[8U], blend, blend[count], 64, 1U, binary);
                reduced = expanded[8U];
            } else Add<float, false>(blend, blend, blend[count], 64, count / 64U, binary);
        } else {
            if (count < 64U) SetVectorMask<float, MaskMode::NORMAL>(count);
            Add<float, false>(blend, blend, blend[count], 64,
                              count < 64U ? 1U : count / 64U, binary);
        }
        PipeBarrier<PIPE_V>();
    }
    return reduced;
}

template <uint32_t CORNER, bool FIXED_MASK = true, bool WAIT_WEIGHT = false, bool BARRIER = true>
__aicore__ inline __attribute__((always_inline)) void SplineAxpyCorner(
    LocalTensor<float> blend, LocalTensor<float> weights, uint32_t matrixSize, float coefficient) {
    if constexpr (WAIT_WEIGHT) {
        if constexpr (CORNER == 1U) WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
        else if constexpr (CORNER == 2U) WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        else WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
    }
    if constexpr (FIXED_MASK) {
        const UnaryRepeatParams unary{1, 1, 8, 8};
        Axpy<float, float, false>(blend, weights[CORNER * matrixSize], coefficient,
                                 64, matrixSize / 64U, unary);
    } else Axpy(blend, weights[CORNER * matrixSize], coefficient, matrixSize);
    if constexpr (BARRIER) PipeBarrier<PIPE_V>();
}

template <uint32_t FIXED_INPUT, uint32_t FIXED_NODES = 0U, bool FULL_INPUT = false,
          bool PADDED_ZERO = false, bool ALIGNED = false>
class KernelSplineSingleImpl {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo,
                              GM_ADDR weight, GM_ADDR out, GM_ADDR tiling) {
        if constexpr (FIXED_NODES == 0U) PreloadSplineRouting(edges, pseudo);
        bool numericalWorker, zeroWorker;
        ResolveWorkers(numericalWorker, zeroWorker);
        if constexpr (disjointZero) {
            if (zeroWorker) ZeroOtherRows(edges, out);
        }
        if (!numericalWorker) return;
        if constexpr (FIXED_NODES != 0U) PreloadSplineRouting(edges, pseudo);
        LoadLayout(tiling);
        InitArena();
        BindInputs(x, edges, pseudo, weight, out);
        LoadInputs();
        if constexpr (fixedTinyTail && !ALIGNED) {
            TailAddresses();
            AscendC::SetVaReg(destinationList, sourceList);
        }
        InitOutput();
        ReadEdge();
        BroadcastInitial();
        StartBlend();
        BroadcastPartial();
        FinishBlend();
        BroadcastLate();
        Multiply();
        if constexpr (fixedTinyTail) ReduceTiny();
        else ReduceGeneral();
        Store();
    }

private:
    static constexpr bool disjointZero = FULL_INPUT && FIXED_INPUT > 0U &&
        FIXED_NODES > 0U && FIXED_NODES <= 32U && FIXED_NODES % 4U != 0U;
    static constexpr bool partialTinyOutput = FULL_INPUT && FIXED_NODES > 0U &&
        FIXED_NODES <= 32U && (FIXED_NODES * 16U) % 64U != 0U;
    static constexpr bool fixedTinyTail = FULL_INPUT && FIXED_INPUT > 0U &&
        FIXED_NODES > 0U && FIXED_NODES <= 32U;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 && !defined(ASCENDC_CPU_DEBUG)
    static constexpr bool reuseAlignedBroadcastMask = fixedTinyTail;
    static constexpr bool reuseBroadcastMask = disjointZero;
#else
    static constexpr bool reuseAlignedBroadcastMask = false;
    static constexpr bool reuseBroadcastMask = false;
#endif
    static constexpr bool splitReady = FIXED_INPUT != 0U &&
        (FIXED_NODES == 0U || partialTinyOutput);
    uint32_t nodes, input, channels, paddedInput, matrixSize, featureCount, zeroCount, source, target;
    bool fullFeatures;
    float u, v;
    LocalTensor<float> weights, features, expanded, blend, output;
    GlobalTensor<float> xGm, coordGm, weightGm, outputGm;
    GlobalTensor<int64_t> edgeGm;
    uint64_t sourceList[16], destinationList[16];

    __aicore__ inline __attribute__((always_inline)) void ResolveWorkers(bool& numericalWorker, bool& zeroWorker) {
        if constexpr (ALIGNED) {
            numericalWorker = GetBlockIdx() == 0U;
            zeroWorker = GetBlockIdx() == 1U;
        } else {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 && !defined(ASCENDC_CPU_DEBUG)
        if constexpr (FULL_INPUT && FIXED_INPUT > 0U && FIXED_NODES > 0U &&
                      FIXED_NODES <= 32U) {
            if ASCEND_IS_AIV {
                const uint32_t block = get_block_idx();
                const uint32_t subblock = get_subblockid();
                numericalWorker = (block | subblock) == 0U;
                zeroWorker = block == 1U && subblock == 0U;
            } else {
                numericalWorker = GetBlockIdx() == 0U;
                zeroWorker = GetBlockIdx() == 1U;
            }
        } else {
            numericalWorker = GetBlockIdx() == 0U;
            zeroWorker = GetBlockIdx() == 1U;
        }
#else
        numericalWorker = GetBlockIdx() == 0U;
        zeroWorker = GetBlockIdx() == 1U;
#endif
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ZeroOtherRows(GM_ADDR edges, GM_ADDR out) {
        GlobalTensor<int64_t> routing;
        routing.SetGlobalBuffer((__gm__ int64_t*)edges, 2U);
        const uint32_t target = static_cast<uint32_t>(routing.GetValue(1U));
        constexpr uint32_t count = (FIXED_NODES * 16U + 63U) & ~63U;
        LocalTensor<float> zeros(TPosition::VECCALC, 0U, count);
        GlobalTensor<float> destination;
        destination.SetGlobalBuffer((__gm__ float*)out);
        SetMaskNorm(); SetVectorMask<float, MaskMode::NORMAL>(64U);
        Duplicate<float, false>(zeros, 0.0F, 64U, count / 64U, 1U, 8U);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (target != 0U) DataCopy(destination, zeros, target * 16U);
        if (target + 1U < FIXED_NODES)
            DataCopy(destination[(target + 1U) * 16U], zeros,
                     (FIXED_NODES - target - 1U) * 16U);
        ResetMask();
    }

    __aicore__ inline __attribute__((always_inline)) void LoadLayout(GM_ADDR tiling) {
        GlobalTensor<uint32_t> descriptor;
        descriptor.SetGlobalBuffer((__gm__ uint32_t*)tiling, 8);
        nodes = FIXED_NODES ? FIXED_NODES : descriptor.GetValue(0);
        input = FULL_INPUT ? FIXED_INPUT : descriptor.GetValue(2);
        channels = FIXED_INPUT ? 16U : descriptor.GetValue(3);
        paddedInput = FIXED_INPUT ? FIXED_INPUT : descriptor.GetValue(6);
        matrixSize = paddedInput * channels;
    }

    __aicore__ inline __attribute__((always_inline)) void InitArena() {
        fullFeatures = FULL_INPUT && FIXED_INPUT && input == paddedInput && nodes * input <= 8192U;
        uint32_t address = 0;
        weights = LocalTensor<float>(TPosition::VECCALC, address, matrixSize * 4U);
        address += matrixSize * 16U;
        featureCount = fullFeatures ? nodes * paddedInput : paddedInput;
        features = LocalTensor<float>(TPosition::VECCALC, address, featureCount);
        address += featureCount * 4U;
        if (fullFeatures) address += ((paddedInput - featureCount) * 4U) & 511U;
        expanded = LocalTensor<float>(TPosition::VECCALC, address, matrixSize);
        address += matrixSize * 4U + (matrixSize >= 512U ? 256U : 0U);
        blend = LocalTensor<float>(TPosition::VECCALC, address, matrixSize);
        address += matrixSize * 4U;
        zeroCount = PADDED_ZERO ? ((nodes * channels + 63U) & ~63U) : nodes * channels;
        output = LocalTensor<float>(TPosition::VECCALC, address, zeroCount);
    }

    __aicore__ inline __attribute__((always_inline)) void BindInputs(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out) {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        coordGm.SetGlobalBuffer((__gm__ float*)pseudo, 2);
        weightGm.SetGlobalBuffer((__gm__ float*)weight);
        outputGm.SetGlobalBuffer((__gm__ float*)out);
        edgeGm.SetGlobalBuffer((__gm__ int64_t*)edges, 2);
    }

    __aicore__ inline __attribute__((always_inline)) void LoadInputs() {
        if (input != paddedInput) {
            Duplicate(weights, 0.0F, matrixSize * 4U);
            Duplicate(features, 0.0F, paddedInput);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        if (input == paddedInput) DataCopy(weights, weightGm, matrixSize * 4U);
        else DataCopy(weights, weightGm, DataCopyParams{4,
                      static_cast<uint16_t>(input * channels / 8U), 0,
                      static_cast<uint16_t>((paddedInput - input) * channels / 8U)});
        if constexpr (splitReady) SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
        if (fullFeatures) DataCopy(features, xGm, featureCount);
    }

    __aicore__ inline __attribute__((always_inline)) void TailAddresses() {
            #pragma unroll
            for (uint32_t i = 0; i < 8U; ++i) {
                const uint32_t reversed = ((i & 1U) << 2U) | (i & 2U) | ((i & 4U) >> 2U);
                sourceList[i] = reinterpret_cast<uint64_t>(blend[reversed * 16U].GetPhyAddr());
                sourceList[8U+i] = reinterpret_cast<uint64_t>(blend[reversed * 16U+8U].GetPhyAddr());
                destinationList[2U*i] = reinterpret_cast<uint64_t>(expanded[i * 8U].GetPhyAddr());
                destinationList[2U*i+1U] = reinterpret_cast<uint64_t>(expanded[(i+8U)*8U].GetPhyAddr());
            }
    }

    __aicore__ inline __attribute__((always_inline)) void InitOutput() {
        if constexpr (PADDED_ZERO && !disjointZero) {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
            Duplicate<float, false>(output, 0.0F, 64, static_cast<uint8_t>(zeroCount / 64U), 1U, 8U);
        } else if constexpr (!disjointZero) {
            Duplicate(output, 0.0F, nodes * channels);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ReadEdge() {
        source = static_cast<uint32_t>(edgeGm.GetValue(0));
        target = static_cast<uint32_t>(edgeGm.GetValue(1));
        if (!fullFeatures) DataCopy(features, xGm[static_cast<uint64_t>(source) * input], input);
        u = coordGm.GetValue(0);
        v = coordGm.GetValue(1);
    }

    __aicore__ inline __attribute__((always_inline)) void BroadcastInitial() {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr (splitReady) WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        else {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (reuseAlignedBroadcastMask) {
                SetMaskNorm();
                SetVectorMask<float, MaskMode::NORMAL>(64U);
            }
        #pragma unroll
        for (uint32_t column = 0; column < channels / 8U; ++column)
            SplineBrcbUsingNormal64<reuseAlignedBroadcastMask>(expanded[column * 8U],
                features[fullFeatures ? source * input : 0U], paddedInput / 8U,
                BrcbRepeatParams(channels / 8U, channels));
        }
    }

    __aicore__ inline __attribute__((always_inline)) void StartBlend() {
        const UnaryRepeatParams unary{1, 1, 8, 8};
        const uint8_t repeats = matrixSize / 64U;
        if constexpr (FIXED_INPUT) {
            if constexpr (disjointZero) SetMaskNorm();
            if constexpr ((!PADDED_ZERO || disjointZero) && !(reuseAlignedBroadcastMask && !splitReady))
                SetVectorMask<float, MaskMode::NORMAL>(64);
            Muls<float, false>(blend, weights, (1.0F - u) * (1.0F - v), 64, repeats, unary);
        } else Muls(blend, weights, (1.0F - u) * (1.0F - v), matrixSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline __attribute__((always_inline)) void BroadcastPartial() {
        if constexpr (partialTinyOutput) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        #pragma unroll
        for (uint32_t column = 0; column < channels / 8U; ++column)
            SplineBrcbUsingNormal64<reuseBroadcastMask>(
                expanded[column * 8U], features[fullFeatures ? source * input : 0U], paddedInput / 8U,
                BrcbRepeatParams(channels / 8U, channels));
            if constexpr (!reuseBroadcastMask) SetVectorMask<float, MaskMode::NORMAL>(64);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void FinishBlend() {
        SplineAxpyCorner<1, (FIXED_INPUT != 0U)>(blend, weights, matrixSize, u * (1.0F - v));
        SplineAxpyCorner<2, (FIXED_INPUT != 0U)>(blend, weights, matrixSize, (1.0F - u) * v);
        SplineAxpyCorner<3, (FIXED_INPUT != 0U)>(blend, weights, matrixSize, u * v);
    }

    __aicore__ inline __attribute__((always_inline)) void BroadcastLate() {
        if constexpr (splitReady && !partialTinyOutput) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        #pragma unroll
        for (uint32_t column = 0; column < channels / 8U; ++column)
            Brcb(expanded[column * 8U], features[fullFeatures ? source * input : 0U], paddedInput / 8U,
                 BrcbRepeatParams(channels / 8U, channels));
            PipeBarrier<PIPE_V>();
            SetVectorMask<float, MaskMode::NORMAL>(64);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void Multiply() {
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const uint8_t repeats = matrixSize / 64U;
        if constexpr (FIXED_INPUT) Mul<float, false>(blend, blend, expanded, 64, repeats, binary);
        else Mul(blend, blend, expanded, matrixSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline __attribute__((always_inline)) void ReduceTiny() {
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        #pragma unroll
        for (uint32_t half = FIXED_INPUT / 2U; half >= 8U; half /= 2U) {
            const uint32_t count = half * 16U;
            Add<float, false>(blend, blend, blend[count], 64U, count / 64U, binary);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (ALIGNED) {
            TailAddresses();
            TransDataTo5HD<float>(destinationList, sourceList, TransDataTo5HDParams(false, false, 1, 0, 0));
        } else {
            AscendC::TransDataTo5HDIntrinsicsImpl<float>(destinationList, sourceList, TransDataTo5HDParams(false, false, 1, 0, 0));
        }
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(output[disjointZero ? 0U : target * channels], expanded,
                                    2U, MASK_PLACEHOLDER, 1U, 1U, 8U);
    }

    __aicore__ inline __attribute__((always_inline)) void ReduceGeneral() {
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        #pragma unroll
        for (uint32_t half = paddedInput / 2U; half > 1U; half /= 2U) {
            if constexpr (FIXED_INPUT) {
                const uint32_t count = half * channels;
                if (count < 64U) SetVectorMask<float, MaskMode::NORMAL>(count);
                Add<float, false>(blend, blend, blend[count], 64,
                                  count < 64U ? 1U : count / 64U, binary);
            } else Add(blend, blend, blend[half * channels], half * channels);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (FIXED_INPUT) {
            SetVectorMask<float, MaskMode::NORMAL>(channels);
            Add<float, false>(output[target * channels], blend, blend[channels], 64, 1, binary);
            ResetMask();
        } else Add(output[target * channels], blend, blend[channels], channels);
    }

    __aicore__ inline __attribute__((always_inline)) void Store() {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if constexpr (disjointZero)
            DataCopy(outputGm[target * channels], output, channels);
        else DataCopy(outputGm, output, nodes * channels);
    }

};

template <uint32_t FIXED_INPUT, uint32_t FIXED_NODES, bool FULL_INPUT, bool PADDED_ZERO>
using KernelSplineSingleGeneral = KernelSplineSingleImpl<FIXED_INPUT, FIXED_NODES, FULL_INPUT, PADDED_ZERO, false>;

template <uint32_t FIXED_INPUT, uint32_t FIXED_NODES, bool FULL_INPUT, bool PADDED_ZERO>
using KernelSplineSingleAligned = KernelSplineSingleImpl<FIXED_INPUT, FIXED_NODES, FULL_INPUT, PADDED_ZERO, true>;

template <uint32_t FIXED_INPUT, uint32_t FIXED_NODES = 0U, bool FULL_INPUT = false, bool PADDED_ZERO = false>
class KernelSplineSingle {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo,
                              GM_ADDR weight, GM_ADDR out, GM_ADDR tiling) {
        if constexpr (FULL_INPUT && FIXED_INPUT > 0U && FIXED_NODES > 0U &&
                      FIXED_NODES <= 32U && FIXED_NODES % 4U == 0U) {
            KernelSplineSingleAligned<FIXED_INPUT, FIXED_NODES, FULL_INPUT, PADDED_ZERO> op;
            op.Run(x, edges, pseudo, weight, out, tiling);
        } else {
            KernelSplineSingleGeneral<FIXED_INPUT, FIXED_NODES, FULL_INPUT, PADDED_ZERO> op;
            op.Run(x, edges, pseudo, weight, out, tiling);
        }
    }
};

struct SplineOwnerBuffers {
    LocalTensor<float> weights, features, expanded, blend, accumulated, zeros;
    GlobalTensor<float> xGm, coord, weightGm, outputGm;
    GlobalTensor<int64_t> edge;
    GlobalTensor<uint32_t> edgeWords;

    __aicore__ inline __attribute__((always_inline)) void InitArena(
        uint32_t matrixSize, uint32_t featureCount, bool wideAccumulator, uint32_t zeroCount) {
        uint32_t address = 0;
        weights = LocalTensor<float>(TPosition::VECCALC, address, 4U * matrixSize);
        address += 16U * matrixSize;
        features = LocalTensor<float>(TPosition::VECCALC, address, featureCount);
        address += featureCount * 4U;
        expanded = LocalTensor<float>(TPosition::VECCALC, address, matrixSize);
        address += matrixSize * 4U + 256U;
        blend = LocalTensor<float>(TPosition::VECCALC, address, matrixSize);
        address += matrixSize * 4U;
        accumulated = LocalTensor<float>(TPosition::VECCALC, address, wideAccumulator ? 64U : 16U);
        address += 64U + (wideAccumulator ? 512U : 0U);
        zeros = LocalTensor<float>(TPosition::VECCALC, address, zeroCount);
    }

    __aicore__ inline __attribute__((always_inline)) void Bind(
        GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out) {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        coord.SetGlobalBuffer((__gm__ float*)pseudo);
        weightGm.SetGlobalBuffer((__gm__ float*)weight);
        outputGm.SetGlobalBuffer((__gm__ float*)out);
        edge.SetGlobalBuffer((__gm__ int64_t*)edges);
    }
};

template <uint32_t FIXED_INPUT, bool EARLY_WEIGHT>
__aicore__ inline __attribute__((always_inline)) void LoadSplineOwnerWeights(
    SplineOwnerBuffers& buffers, const SplineResidentTilingData& t, uint32_t featureCount) {
    constexpr uint32_t channels = 16U, matrixSize = FIXED_INPUT * channels;
    if constexpr (!EARLY_WEIGHT) {
        if (t.inChannels != FIXED_INPUT) {
            Duplicate(buffers.weights, 0.0F, 4U * matrixSize);
            Duplicate(buffers.features, 0.0F, featureCount);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopy(buffers.weights, buffers.weightGm, DataCopyParams{4,
                static_cast<uint16_t>(t.inChannels * channels / 8U), 0,
                static_cast<uint16_t>((FIXED_INPUT - t.inChannels) * channels / 8U)});
        } else DataCopy(buffers.weights, buffers.weightGm, 4U * matrixSize);
    }
}

template <uint32_t FIXED_INPUT>
__aicore__ inline __attribute__((always_inline)) void SplineMultiplyReduce(
    SplineOwnerBuffers& buffers, float u, float v) {
    constexpr uint32_t matrixSize = FIXED_INPUT * 16U;
    const UnaryRepeatParams unary{1, 1, 8, 8};
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    constexpr uint8_t repeats = matrixSize / 64U;
    Muls<float, false>(buffers.blend, buffers.weights, (1.0F - u) * (1.0F - v), 64, repeats, unary);
    PipeBarrier<PIPE_V>();
    SplineAxpyCorner<1, true, false, true>(buffers.blend, buffers.weights, matrixSize, u * (1.0F - v));
    SplineAxpyCorner<2, true, false, true>(buffers.blend, buffers.weights, matrixSize, (1.0F - u) * v);
    SplineAxpyCorner<3, true, false, true>(buffers.blend, buffers.weights, matrixSize, u * v);
    Mul<float, false>(buffers.blend, buffers.blend, buffers.expanded, 64, repeats, binary);
    PipeBarrier<PIPE_V>();
    SplineReduceTree<FIXED_INPUT>(buffers.blend, buffers.expanded);
    SetVectorMask<float, MaskMode::NORMAL>(16U);
}

template <typename Index>
__aicore__ inline __attribute__((always_inline)) void ZeroRuns(
    GlobalTensor<float> out, LocalTensor<float> zeros, Index rowBegin,
    uint64_t occupied, uint32_t base, Index& begin) {
    while (occupied) {
        const Index target = base + static_cast<Index>(GetSFFValue<1>(occupied));
        occupied &= occupied - 1U;
        if (begin < target) DataCopy(out[(rowBegin + begin) * 16U], zeros, (target - begin) * 16U);
        begin = target + 1U;
    }
}

template <bool SINGLE_MASK, typename State, typename Index>
__aicore__ inline __attribute__((always_inline)) void RecordSplineTarget(State& state, Index offset) {
    const uint64_t bit = uint64_t{1} << (offset & 63U);
    if constexpr (SINGLE_MASK) state.occupied0 |= bit;
    else {
        if (offset < 64U) state.occupied0 |= bit;
        else if (offset < 128U) state.occupied1 |= bit;
        else if (offset < 192U) state.occupied2 |= bit;
        else state.occupied3 |= bit;
    }
}

template <bool SINGLE_MASK, typename State>
__aicore__ inline __attribute__((always_inline)) void StoreSplineZeroRanges(State& state) {
    decltype(state.rowBegin) begin = 0;
    ZeroRuns(state.outputGm, state.zeros, state.rowBegin, state.occupied0, 0U, begin);
    if constexpr (!SINGLE_MASK) {
        ZeroRuns(state.outputGm, state.zeros, state.rowBegin, state.occupied1, 64U, begin);
        ZeroRuns(state.outputGm, state.zeros, state.rowBegin, state.occupied2, 128U, begin);
        ZeroRuns(state.outputGm, state.zeros, state.rowBegin, state.occupied3, 192U, begin);
    }
    if (begin < state.rows)
        DataCopy(state.outputGm[(state.rowBegin + begin) * 16U], state.zeros, (state.rows - begin) * 16U);
}

template <bool COMPACT> struct SplineOwnerIndex { using Type = uint32_t; };
template <> struct SplineOwnerIndex<true> { using Type = uint64_t; };

// For a graph with no more edges than launched cores, the first incoming
// edge assigns each target to one core. Zero-only node ranges exclude every
// target, so the compute and zero stores are disjoint without cross-core flags.
template <uint32_t FIXED_INPUT, bool EARLY_WEIGHT = false, bool SINGLE_MASK = false, bool DEFER_DEGREE = false>
class KernelSplineEdgeOwnerLate : public SplineOwnerBuffers {
    static constexpr uint32_t channels = 16U, matrixSize = FIXED_INPUT * channels, cores = 8U;
public:
    uint32_t core, rowBegin, rows, ownTarget, degree;
    uint64_t occupied0, occupied1, occupied2, occupied3, pending;
    bool active, first;
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo,
                              GM_ADDR weight, GM_ADDR out, const SplineResidentTilingData& t) {
        core = GetBlockIdx();
        rowBegin = t.nodes * core / cores;
        rows = t.nodes * (core + 1U) / cores - rowBegin;
        InitArena(matrixSize, t.edges * FIXED_INPUT, false, (rows ? rows : 1U) * channels);
        Bind(x, edges, pseudo, weight, out);
        active = core < t.edges;
        LoadFirst(t);
        if (rows) Duplicate(zeros, 0.0F, rows * channels);
        ownTarget = active ? static_cast<uint32_t>(edge.GetValue(t.edges + core)) : ~0U;
        occupied0 = occupied1 = occupied2 = occupied3 = pending = 0;
        degree = 0;
        ScanRouting(t);
        if (rows) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            StoreSplineZeroRanges<SINGLE_MASK>(*this);
        }
        if constexpr (DEFER_DEGREE)
            degree = t.mean ? static_cast<uint32_t>(GetBitCount<1>(pending)) : 0U;
        if (core < t.edges) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        first = true;
        ProcessPending();
        StoreContribution(t);
    }

private:
    __aicore__ inline __attribute__((always_inline)) void LoadFirst(const SplineResidentTilingData& t) {
        if (active) {
            LoadSplineOwnerWeights<FIXED_INPUT, EARLY_WEIGHT>(*this, t, t.edges * FIXED_INPUT);
            const uint32_t source = static_cast<uint32_t>(edge.GetValue(core));
            DataCopy(features[core * FIXED_INPUT],
                     xGm[static_cast<uint64_t>(source) * t.inChannels], t.inChannels);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ScanRouting(const SplineResidentTilingData& t) {
        for (uint32_t e = 0; e < t.edges; ++e) {
            const uint32_t target = static_cast<uint32_t>(edge.GetValue(t.edges + e));
            const uint32_t offset = target - rowBegin;
            if (offset < rows) {
                RecordSplineTarget<SINGLE_MASK>(*this, offset);
            }
            if (e < core && target == ownTarget) active = false;
            if (active && target == ownTarget) {
                pending |= uint64_t{1} << e;
                if constexpr (!DEFER_DEGREE) ++degree;
                if (e != core) {
                    const uint32_t source = static_cast<uint32_t>(edge.GetValue(e));
                    DataCopy(features[e * FIXED_INPUT],
                             xGm[static_cast<uint64_t>(source) * t.inChannels], t.inChannels);
                }
            }
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ProcessPending() {
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        while (pending) {
            const uint32_t e = static_cast<uint32_t>(GetSFFValue<1>(pending));
            pending &= pending - 1U;
            const float u = coord.GetValue(2U * e), v = coord.GetValue(2U * e + 1U);
            #pragma unroll
            for (uint32_t column = 0; column < channels / 8U; ++column)
                Brcb(expanded[column * 8U], features[e * FIXED_INPUT], FIXED_INPUT / 8U,
                     BrcbRepeatParams(channels / 8U, channels));
            SetVectorMask<float, MaskMode::NORMAL>(64);
            SplineMultiplyReduce<FIXED_INPUT>(*this, u, v);
            if (first) Add<float, false>(accumulated, blend, blend[channels], 64, 1, binary);
            else {
                Add<float, false>(blend, blend, blend[channels], 64, 1, binary);
                PipeBarrier<PIPE_V>();
                Add<float, false>(accumulated, accumulated, blend, 64, 1, binary);
            }
            if constexpr (!DEFER_DEGREE) PipeBarrier<PIPE_V>();
            ResetMask();
            first = false;
            if constexpr (DEFER_DEGREE) {
                if (pending) PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline __attribute__((always_inline)) void StoreContribution(const SplineResidentTilingData& t) {
        if (!first) {
            if (t.mean && degree > 1U) {
                if constexpr (DEFER_DEGREE) PipeBarrier<PIPE_V>();
                Muls(accumulated, accumulated, 1.0F / static_cast<float>(static_cast<int32_t>(degree)), channels);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outputGm[ownTarget * channels], accumulated, channels);
        }
    }

};

template <uint32_t FIXED_INPUT, bool EARLY_WEIGHT = false, bool SINGLE_MASK = false, bool UNLIKELY_DUPLICATES = false>
class KernelSplineNumericResident : public SplineOwnerBuffers {
    static constexpr uint32_t channels = 16U, matrixSize = FIXED_INPUT * channels;
public:
    uint32_t core, firstSource, degree;
    uint64_t ownTarget, pending;
    bool hasContribution;
    float first00, first10, first01, first11;
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo,
                              GM_ADDR weight, GM_ADDR out, const SplineResidentTilingData& t) {
        core = GetBlockIdx();
        InitArena(matrixSize, t.nodes * FIXED_INPUT, false, channels);
        Bind(x, edges, pseudo, weight, out);
        firstSource = 0U;
        LoadFeatures(t);
        ReadRouting();
        ReadCoefficients();
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        InterpolateFirst();
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        hasContribution = pending != 0U;
        ComputeFirst();
        ProcessPending();
        StoreContribution(t);
    }

private:
    __aicore__ inline __attribute__((always_inline)) void LoadFeatures(const SplineResidentTilingData& t) {
            LoadSplineOwnerWeights<FIXED_INPUT, EARLY_WEIGHT>(*this, t, t.edges * FIXED_INPUT);
            DataCopy(features, xGm, t.nodes * FIXED_INPUT);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            firstSource = static_cast<uint32_t>(edge.GetValue(core));
    }

    __aicore__ inline __attribute__((always_inline)) void ReadRouting() {
        SetMaskNorm();
        ResetMask();
        const uint64_t target0 = static_cast<uint64_t>(edge.GetValue(4U));
        const uint64_t target1 = static_cast<uint64_t>(edge.GetValue(5U));
        const uint64_t target2 = static_cast<uint64_t>(edge.GetValue(6U));
        const uint64_t target3 = static_cast<uint64_t>(edge.GetValue(7U));
        ownTarget = core == 0U ? target0 :
            core == 1U ? target1 : core == 2U ? target2 : target3;
        const uint64_t matching = static_cast<uint64_t>(target0 == ownTarget) |
            (static_cast<uint64_t>(target1 == ownTarget) << 1U) |
            (static_cast<uint64_t>(target2 == ownTarget) << 2U) |
            (static_cast<uint64_t>(target3 == ownTarget) << 3U);
        const uint64_t ownBit = uint64_t{1} << core;
        pending = (matching & (ownBit - 1U)) == 0U ? matching : 0U;
        uint64_t additional = pending & ~ownBit;
        degree = 1U;
        while (additional) {
            ++degree;
            additional &= additional - 1U;
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ReadCoefficients() {
        const float firstU = coord.GetValue(2U * core);
        const float firstV = coord.GetValue(2U * core + 1U);
        first00 = (1.0F - firstU) * (1.0F - firstV);
        first10 = firstU * (1.0F - firstV);
        first01 = (1.0F - firstU) * firstV;
        first11 = firstU * firstV;
    }

    __aicore__ inline __attribute__((always_inline)) void InterpolateFirst() {
            const UnaryRepeatParams unary{1, 1, 8, 8};
            constexpr uint8_t repeats = matrixSize / 64U;
            SetVectorMask<float, MaskMode::NORMAL>(64);
            Muls<float, false>(blend, weights, first00, 64, repeats, unary);
            PipeBarrier<PIPE_V>();
            SplineAxpyCorner<1, true, false, true>(blend, weights, matrixSize, first10);
            SplineAxpyCorner<2, true, false, true>(blend, weights, matrixSize, first01);
            SplineAxpyCorner<3, true, false, true>(blend, weights, matrixSize, first11);
    }

    __aicore__ inline __attribute__((always_inline)) void ComputeFirst() {
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        constexpr uint8_t repeats = matrixSize / 64U;
        if (hasContribution) {
            pending &= pending - 1U;
            #pragma unroll
            for (uint32_t column = 0; column < channels / 8U; ++column)
                Brcb(expanded[column * 8U], features[firstSource * FIXED_INPUT], FIXED_INPUT / 8U,
                     BrcbRepeatParams(channels / 8U, channels));
            PipeBarrier<PIPE_V>();
            Mul<float, false>(blend, blend, expanded, 64, repeats, binary);
            PipeBarrier<PIPE_V>();
            SplineReduceTree<FIXED_INPUT>(blend, expanded);
            SetVectorMask<float, MaskMode::NORMAL>(channels);
            Add<float, false>(accumulated, blend, blend[channels], 64, 1, binary);
            ResetMask();
            if (pending) PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ProcessPending() {
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        while (UNLIKELY_DUPLICATES ? __builtin_expect(pending != 0U, 0) : pending != 0U) {
            const uint32_t e = static_cast<uint32_t>(GetSFFValue<1>(pending));
            pending &= pending - 1U;
            const uint32_t source = static_cast<uint32_t>(edge.GetValue(e));
            #pragma unroll
            for (uint32_t column = 0; column < channels / 8U; ++column)
                Brcb(expanded[column * 8U], features[source * FIXED_INPUT], FIXED_INPUT / 8U,
                     BrcbRepeatParams(channels / 8U, channels));
                const float u = coord.GetValue(2U * e), v = coord.GetValue(2U * e + 1U);
            SplineMultiplyReduce<FIXED_INPUT>(*this, u, v);
                Add<float, false>(blend, blend, blend[channels], 64, 1, binary);
                PipeBarrier<PIPE_V>();
                Add<float, false>(accumulated, accumulated, blend, 64, 1, binary);
            ResetMask();
            if (pending) PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline __attribute__((always_inline)) void StoreContribution(const SplineResidentTilingData& t) {
        if (hasContribution) {
            if (UNLIKELY_DUPLICATES ? __builtin_expect(t.mean && degree > 1U, 0) : (t.mean && degree > 1U)) {
                PipeBarrier<PIPE_V>();
                Muls(accumulated, accumulated, 1.0F / static_cast<float>(static_cast<int32_t>(degree)), channels);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outputGm[ownTarget * channels], accumulated, channels);
        }
    }

};

class KernelSplineHalfOriginalSeparateZero {
public:
    __aicore__ inline void Run(GM_ADDR edges, GM_ADDR out, uint32_t nodes) {
        const uint32_t owner = GetBlockIdx() - 4U;
        constexpr uint32_t rowWidth = 16U;
        const uint32_t rowBegin = owner * rowWidth;
        const uint32_t remaining = nodes > rowBegin ? nodes - rowBegin : 0U;
        const uint32_t rows = remaining < rowWidth ? remaining : rowWidth;
        if (rows) {
            LocalTensor<float> zeros(TPosition::VECCALC, 32768U, rowWidth * 16U);
            GlobalTensor<int64_t> edge;
            GlobalTensor<float> output;
            edge.SetGlobalBuffer((__gm__ int64_t*)edges);
            output.SetGlobalBuffer((__gm__ float*)out);
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
            Duplicate<float, false>(zeros, 0.0F, 64, 4U, 1U, 8U);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            uint64_t occupied = 0U;
            for (uint32_t e = 0; e < 4U; ++e) {
                const uint32_t target = static_cast<uint32_t>(edge.GetValue(4U + e));
                const uint32_t offset = target - rowBegin;
                if (offset < rows) occupied |= uint64_t{1} << offset;
            }
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            uint32_t begin = 0U;
            ZeroRuns(output, zeros, rowBegin, occupied, 0U, begin);
            if (begin < rows)
                DataCopy(output[(rowBegin + begin) * 16U], zeros, (rows - begin) * 16U);
        }
    }

};



// For a graph with no more edges than launched cores, the first incoming
// edge assigns each target to one core. Zero-only node ranges exclude every
// target, so the compute and zero stores are disjoint without cross-core flags.
template <uint32_t FIXED_INPUT, bool EARLY_WEIGHT, bool STREAM_WEIGHTS,
          bool SINGLE_MASK, bool FULL_LANE_TERMINAL, bool ALIGNED_ROWS,
          bool COMPACT, bool NATIVE>
class KernelSplineOwnerImpl : public SplineOwnerBuffers {
    static_assert(!NATIVE || (FIXED_INPUT == 128U && STREAM_WEIGHTS && FULL_LANE_TERMINAL),
                  "Native readiness requires full-lane streamed weights");
    using Index = typename SplineOwnerIndex<COMPACT>::Type;
    static constexpr uint32_t channels = 16U, matrixSize = FIXED_INPUT * channels;
    static constexpr Index cores = 8U;
public:
    Index core, edgeCount, rowBegin, rows, ownTarget;
    uint64_t occupied0, occupied1, occupied2, occupied3, pending;
    uint32_t degree, zeroCount;
    bool active;

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo,
                              GM_ADDR weight, GM_ADDR out, const SplineResidentTilingData& t) {
        InitPartition(t);
        InitArena(matrixSize, edgeCount * FIXED_INPUT, FULL_LANE_TERMINAL, zeroCount);
        Bind(x, edges, pseudo, weight, out);
        if constexpr (FULL_LANE_TERMINAL) edgeWords.SetGlobalBuffer((__gm__ uint32_t*)edges);
        active = (STREAM_WEIGHTS || core < edgeCount);
        LoadFirst(t);
        InitZeros();
        ownTarget = ~Index{0};
        if constexpr (!STREAM_WEIGHTS)
            ownTarget = active ? NodeIndex(edge, edgeWords, edgeCount + core) : ~Index{0};
        PrepareFirst();
        if constexpr (!COMPACT && STREAM_WEIGHTS)
            ownTarget = static_cast<uint32_t>(edge.GetValue(edgeCount + core));
        occupied0 = occupied1 = occupied2 = occupied3 = pending = 0;
        degree = 0;
        ScanRouting(t);
        if (rows) {
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            StoreSplineZeroRanges<SINGLE_MASK>(*this);
        }
        if (active) ReduceEdge<true>(blend, expanded, accumulated);
        if (pending) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1); WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        ProcessPending(edges, pseudo, t);
        StoreContribution(t);
    }

private:
    __aicore__ inline __attribute__((always_inline)) void InitPartition(const SplineResidentTilingData& t) {
        core = GetBlockIdx();
        edgeCount = STREAM_WEIGHTS ? cores : t.edges;
        const Index rowWidth = ALIGNED_ROWS ? (static_cast<Index>(t.nodes) + 63U) / 64U * 8U : 0U;
        rowBegin = ALIGNED_ROWS ? rowWidth * core : static_cast<Index>(t.nodes) * core / cores;
        const Index remaining = ALIGNED_ROWS && static_cast<Index>(t.nodes) > rowBegin ?
            static_cast<Index>(t.nodes) - rowBegin : 0U;
        rows = ALIGNED_ROWS ? (remaining < rowWidth ? remaining : rowWidth) :
            static_cast<Index>(t.nodes) * (core + 1U) / cores - rowBegin;
        if constexpr (COMPACT && !NATIVE)
            zeroCount = (static_cast<uint32_t>(rows ? rows : 1U) * channels + 63U) / 64U * 64U;
        else zeroCount = (rows ? rows : 1U) * channels;
    }

    __aicore__ inline __attribute__((always_inline)) void LoadFirst(const SplineResidentTilingData& t) {
        if (active) {
            LoadSplineOwnerWeights<FIXED_INPUT, EARLY_WEIGHT>(*this, t, edgeCount * FIXED_INPUT);
            const Index source = NATIVE ? static_cast<Index>(edge.GetValue(core)) :
                NodeIndex(edge, edgeWords, core);
            DataCopy(features[core * FIXED_INPUT],
                     xGm[static_cast<uint64_t>(source) * t.inChannels], t.inChannels);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void InitZeros() {
        if constexpr (COMPACT && !NATIVE) {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
        }
        if (rows) {
            if constexpr (NATIVE) {
                SetMaskNorm();
                SetVectorMask<float, MaskMode::NORMAL>(64);
                Duplicate<float, false>(zeros, 0.0F, 64,
                    static_cast<uint8_t>(rows * channels / 64U), 1U, 8U);
            } else if constexpr (COMPACT) {
                Duplicate<float, false>(zeros, 0.0F, 64,
                    static_cast<uint8_t>(zeroCount / 64U), 1U, 8U);
            } else Duplicate(zeros, 0.0F, rows * channels);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void PrepareFirst() {
        if (active) {
            const float u = coord.GetValue(2U * core), v = coord.GetValue(2U * core + 1U);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (!STREAM_WEIGHTS) WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            PrepareEdge<true>(weights, features[core * FIXED_INPUT], expanded, blend, u, v, edge, edgeWords, core, edgeCount, ownTarget);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ScanRouting(const SplineResidentTilingData& t) {
        for (Index e = 0; e < edgeCount; ++e) {
            const Index target = NodeIndex(edge, edgeWords, edgeCount + e);
            const Index offset = target - rowBegin;
            if (offset < rows) {
                RecordSplineTarget<SINGLE_MASK>(*this, offset);
            }
            if (e < core && target == ownTarget) active = false;
            if (active && target == ownTarget) {
                ++degree;
                if (e != core) {
                    pending |= uint64_t{1} << e;
                    const Index source = NodeIndex(edge, edgeWords, e);
                    DataCopy(features[e * FIXED_INPUT],
                             xGm[static_cast<uint64_t>(source) * t.inChannels], t.inChannels);
                }
            }
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ProcessPending(GM_ADDR edges, GM_ADDR pseudo, const SplineResidentTilingData& t) {
        if constexpr (NATIVE) {
            if (__builtin_expect(pending != 0U, 0)) {
                PipeBarrier<PIPE_V>();
                AdditionalContributions(edges, pseudo, pending, degree, t.mean);
                PipeBarrier<PIPE_ALL>();
                ResetMask();
            }
        } else {
        while (pending) {
            const Index e = static_cast<Index>(GetSFFValue<1>(pending));
            pending &= pending - 1U;
            const float u = coord.GetValue(2U * e), v = coord.GetValue(2U * e + 1U);
            PrepareEdge<false>(weights, features[e * FIXED_INPUT], expanded, blend, u, v, edge, edgeWords, core, edgeCount, ownTarget);
            ReduceEdge<false>(blend, expanded, accumulated);
        }
        }
    }

    __aicore__ inline __attribute__((always_inline)) void StoreContribution(const SplineResidentTilingData& t) {
        if (active) {
            if (!NATIVE && t.mean && degree > 1U) {
                Muls(accumulated, accumulated, 1.0F / static_cast<float>(static_cast<int32_t>(degree)), channels);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(outputGm[ownTarget * channels], accumulated, channels);
        }
    }

    __aicore__ inline Index NodeIndex(const GlobalTensor<int64_t>& edge,
                                       const GlobalTensor<uint32_t>& words,
                                       Index index) {
        if constexpr (FULL_LANE_TERMINAL) return words.GetValue(2U * index);
        else return static_cast<uint32_t>(edge.GetValue(index));
    }

    template <bool INITIAL>
    __aicore__ inline void PrepareEdge(LocalTensor<float> weights, LocalTensor<float> feature,
                                      LocalTensor<float> expanded, LocalTensor<float> blend,
                                      float u, float v,
                                      const GlobalTensor<int64_t>& edge,
                                      const GlobalTensor<uint32_t>& edgeWords,
                                      Index core, Index edgeCount, Index& ownTarget) {
        constexpr uint32_t channels = 16U, matrixSize = FIXED_INPUT * channels;
        const UnaryRepeatParams unary{1, 1, 8, 8};
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        constexpr uint8_t repeats = matrixSize / 64U;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 && !defined(ASCENDC_CPU_DEBUG)
        constexpr bool reuseStreamMask = NATIVE && STREAM_WEIGHTS;
#else
        constexpr bool reuseStreamMask = false;
#endif
        if constexpr (!STREAM_WEIGHTS) {
        #pragma unroll
        for (uint32_t column = 0; column < channels / 8U; ++column)
            Brcb(expanded[column * 8U], feature, FIXED_INPUT / 8U,
                 BrcbRepeatParams(channels / 8U, channels));
        }
        SetVectorMask<float, MaskMode::NORMAL>(64);
        if constexpr (STREAM_WEIGHTS && INITIAL) WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        Muls<float, false>(blend, weights, (1.0F - u) * (1.0F - v), 64, repeats, unary);
        PipeBarrier<PIPE_V>();
        if constexpr (COMPACT && STREAM_WEIGHTS && INITIAL)
            ownTarget = NodeIndex(edge, edgeWords, edgeCount + core);
        SplineAxpyCorner<1, true, (STREAM_WEIGHTS && INITIAL), true>(blend, weights, matrixSize, u * (1.0F - v));
        SplineAxpyCorner<2, true, (STREAM_WEIGHTS && INITIAL), true>(blend, weights, matrixSize, (1.0F - u) * v);
        SplineAxpyCorner<3, true, (STREAM_WEIGHTS && INITIAL), (!NATIVE || !STREAM_WEIGHTS)>(blend, weights, matrixSize, u * v);
        if constexpr (STREAM_WEIGHTS) {
            if constexpr (INITIAL) WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        #pragma unroll
        for (uint32_t column = 0; column < channels / 8U; ++column)
            SplineBrcbUsingNormal64<reuseStreamMask>(expanded[column * 8U], feature,
                FIXED_INPUT / 8U, BrcbRepeatParams(channels / 8U, channels));
            PipeBarrier<PIPE_V>();
            if constexpr (!reuseStreamMask) SetVectorMask<float, MaskMode::NORMAL>(64);
        }
        Mul<float, false>(blend, blend, expanded, 64, repeats, binary);
        PipeBarrier<PIPE_V>();
    }

    template <bool FIRST>
    __aicore__ inline void ReduceEdge(LocalTensor<float> blend, LocalTensor<float> expanded,
                                     LocalTensor<float> accumulated) {
        constexpr uint32_t channels = 16U;
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        LocalTensor<float> reduced = blend;
        reduced = SplineReduceTree<FIXED_INPUT, STREAM_WEIGHTS>(blend, expanded);
        if constexpr (!FULL_LANE_TERMINAL) SetVectorMask<float, MaskMode::NORMAL>(channels);
        if constexpr (FIRST) Add<float, false>(accumulated, reduced, reduced[channels], 64, 1, binary);
        else {
            Add<float, false>(blend, reduced, reduced[channels], 64, 1, binary);
            PipeBarrier<PIPE_V>();
            Add<float, false>(accumulated, accumulated, blend, 64, 1, binary);
        }
        if constexpr (!NATIVE) PipeBarrier<PIPE_V>();
        if constexpr (!FULL_LANE_TERMINAL) ResetMask();
    }

    __aicore__ __attribute__((noinline)) static void AdditionalContributions(
        GM_ADDR edges, GM_ADDR pseudo, Index pending, uint32_t degree, uint32_t mean) {
        constexpr Index edgeCount = 8U;
        constexpr uint32_t channels = 16U, matrixSize = FIXED_INPUT * channels;
        KernelSplineOwnerImpl op;
        op.InitArena(matrixSize, edgeCount * FIXED_INPUT, FULL_LANE_TERMINAL, channels);
        op.coord.SetGlobalBuffer((__gm__ float*)pseudo);
        op.edge.SetGlobalBuffer((__gm__ int64_t*)edges);
        op.edgeWords.SetGlobalBuffer((__gm__ uint32_t*)edges);
        const Index core = GetBlockIdx();
        Index ownTarget = 0U;
        SetMaskNorm();
        ResetMask();
        while (pending) {
            const Index e = static_cast<Index>(GetSFFValue<1>(pending));
            pending &= pending - 1U;
            const float u = op.coord.GetValue(2U * e), v = op.coord.GetValue(2U * e + 1U);
            PipeBarrier<PIPE_V>();
            op.template PrepareEdge<false>(op.weights, op.features[e * FIXED_INPUT], op.expanded, op.blend, u, v, op.edge, op.edgeWords, core, edgeCount, ownTarget);
            op.template ReduceEdge<false>(op.blend, op.expanded, op.accumulated);
        }
            if (mean && degree > 1U) {
                PipeBarrier<PIPE_V>();
                Muls(op.accumulated, op.accumulated, 1.0F / static_cast<float>(static_cast<int32_t>(degree)), channels);
                PipeBarrier<PIPE_V>();
            }
        PipeBarrier<PIPE_V>();
        ResetMask();
    }
};

template <uint32_t FIXED_INPUT, bool EARLY_WEIGHT = false, bool STREAM_WEIGHTS = false, bool SINGLE_MASK = false>
using KernelSplineEdgeOwner = KernelSplineOwnerImpl<FIXED_INPUT, EARLY_WEIGHT, STREAM_WEIGHTS, SINGLE_MASK, false, false, false, false>;

template <uint32_t FIXED_INPUT, bool EARLY_WEIGHT = false, bool STREAM_WEIGHTS = false, bool SINGLE_MASK = false, bool FULL_LANE_TERMINAL = false, bool ALIGNED_ROWS = false>
using KernelSplineCompactOwner = KernelSplineOwnerImpl<FIXED_INPUT, EARLY_WEIGHT, STREAM_WEIGHTS, SINGLE_MASK, FULL_LANE_TERMINAL, ALIGNED_ROWS, true, false>;

template <uint32_t FIXED_INPUT, bool EARLY_WEIGHT = false, bool STREAM_WEIGHTS = false, bool SINGLE_MASK = false, bool FULL_LANE_TERMINAL = false>
using KernelSplineNativeConsumerReadiness = KernelSplineOwnerImpl<FIXED_INPUT, EARLY_WEIGHT, STREAM_WEIGHTS, SINGLE_MASK, FULL_LANE_TERMINAL, false, true, true>;

template <uint32_t FIXED_INPUT = 0, bool EARLY_WEIGHT = false, bool SCALAR_MEAN = false>
class KernelSplineResident : public SplineOwnerBuffers {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo,
                              GM_ADDR weight, GM_ADDR out, const SplineResidentTilingData& t) {
        owner = GetBlockIdx();
        if (owner >= t.owners || owner >= t.nodes) return;
        InitResidentArena(t);
        Bind(x, edges, pseudo, weight, out);
        LoadWeights(t);
        Duplicate(output, 0.0F, rows * channels);
        needCounts = t.mean && t.edges > 1U;
        if (needCounts) {
            Duplicate(counts, 0, countSize);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        repeatedTarget = false;
        pending = 0;
        ScanRouting(t);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ProcessPending(t);
        AggregateMean(t);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(outputGm[owner * channels], output,
                    {static_cast<uint16_t>(rows), channels * 4U, 0,
                     (t.owners - 1U) * channels * 4U, 0});
    }

private:
    uint32_t owner, rows, channels, paddedInput, matrixSize, countSize, featureSize;
    bool needCounts, repeatedTarget;
    uint64_t pending;
    LocalTensor<float> output, floatCounts;
    LocalTensor<int32_t> counts;
    __aicore__ inline __attribute__((always_inline)) void InitResidentArena(const SplineResidentTilingData& t) {
        rows = (t.nodes - owner + t.owners - 1U) >> t.ownerShift;
        channels = FIXED_INPUT ? 16U : t.outChannels;
        paddedInput = FIXED_INPUT ? FIXED_INPUT : t.paddedInput;
        matrixSize = paddedInput * channels;
        countSize = (rows + 7U) / 8U * 8U;
        // A single, explicitly bounded UB arena removes queue initialization.
        uint32_t address = 0;
        weights = LocalTensor<float>(TPosition::VECCALC, address, 4 * matrixSize);
        address += 4 * matrixSize * 4;
        featureSize = (t.edges ? t.edges : 1U) * paddedInput;
        features = LocalTensor<float>(TPosition::VECCALC, address, featureSize);
        address += featureSize * 4;
        const uint32_t expandedSize = matrixSize > countSize * channels ? matrixSize : countSize * channels;
        expanded = LocalTensor<float>(TPosition::VECCALC, address, expandedSize);
        address += expandedSize * 4;
        if (matrixSize >= 512U) address += 256U;
        blend = LocalTensor<float>(TPosition::VECCALC, address, matrixSize);
        address += matrixSize * 4;
        output = LocalTensor<float>(TPosition::VECCALC, address, rows * channels);
        address += rows * channels * 4;
        counts = LocalTensor<int32_t>(TPosition::VECCALC, address, countSize);
        address += countSize * 4;
        floatCounts = LocalTensor<float>(TPosition::VECCALC, address, countSize);
    }

    __aicore__ inline __attribute__((always_inline)) void LoadWeights(const SplineResidentTilingData& t) {
        if constexpr (!EARLY_WEIGHT) {
            // Full aligned matrices have no padding and need only one descriptor.
            if (t.inChannels != paddedInput) {
                Duplicate(weights, 0.0F, 4 * matrixSize);
                Duplicate(features, 0.0F, featureSize);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            const DataCopyParams matrixCopy{4,
                static_cast<uint16_t>(t.inChannels * channels / 8U), 0,
                static_cast<uint16_t>((paddedInput - t.inChannels) * channels / 8U)};
            if (t.inChannels == paddedInput) DataCopy(weights, weightGm, 4 * matrixSize);
            else DataCopy(weights, weightGm, matrixCopy);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ScanRouting(const SplineResidentTilingData& t) {
        for (uint32_t e = 0; e < t.edges; ++e) {
            const uint32_t target = static_cast<uint32_t>(edge.GetValue(t.edges + e));
            if ((target & (t.owners - 1U)) != owner) continue;
            pending |= uint64_t{1} << e;
            const uint32_t source = static_cast<uint32_t>(edge.GetValue(e));
            DataCopy(features[e * paddedInput],
                     xGm[static_cast<uint64_t>(source) * t.inChannels], t.inChannels);
            if (needCounts) {
                const uint32_t row = target >> t.ownerShift;
                const int32_t previous = counts.GetValue(row);
                repeatedTarget |= previous != 0;
                counts.SetValue(row, previous + 1);
            }
        }
    }

    __aicore__ inline __attribute__((always_inline)) void ProcessPending(const SplineResidentTilingData& t) {
        while (pending) {
            const uint32_t e = static_cast<uint32_t>(GetSFFValue<1>(pending));
            pending &= pending - 1U;
            const uint32_t target = static_cast<uint32_t>(edge.GetValue(t.edges + e));
            const uint32_t row = target >> t.ownerShift;
            const float u = coord.GetValue(2 * e), v = coord.GetValue(2 * e + 1);
            BroadcastRows(expanded, features[e * paddedInput], paddedInput, channels);
            // Weight scaling is independent of feature broadcasting; its
            // barrier completes both before the later product consumes them.
            // Tensor-product linear B-spline on the full [0,1]^2 domain,
            // including endpoint wrapping, equals bilinear interpolation.
            Muls(blend, weights, (1.0F - u) * (1.0F - v), matrixSize);
            PipeBarrier<PIPE_V>();
            SplineAxpyCorner<1, false>(blend, weights, matrixSize, u * (1.0F - v));
            SplineAxpyCorner<2, false>(blend, weights, matrixSize, (1.0F - u) * v);
            SplineAxpyCorner<3, false>(blend, weights, matrixSize, u * v);
            Mul(blend, blend, expanded, matrixSize);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for (uint32_t half = paddedInput / 2U; half; half /= 2U) {
                Add(blend, blend, blend[half * channels], half * channels);
                PipeBarrier<PIPE_V>();
            }
            Add(output[row * channels], output[row * channels], blend, channels);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline __attribute__((always_inline)) void AggregateMean(const SplineResidentTilingData& t) {
        if (t.mean && repeatedTarget) {
            SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if constexpr (SCALAR_MEAN) {
                for (uint32_t row = 0; row < rows; ++row) {
                    const int32_t degree = counts.GetValue(row);
                    if (degree > 1) {
                        Muls(output[row * channels], output[row * channels],
                             1.0F / static_cast<float>(degree), channels);
                        PipeBarrier<PIPE_V>();
                    }
                }
            } else {
                Cast(floatCounts, counts, RoundMode::CAST_NONE, countSize);
                PipeBarrier<PIPE_V>();
                Maxs(floatCounts, floatCounts, 1.0F, countSize);
                PipeBarrier<PIPE_V>();
                BroadcastRows(expanded, floatCounts, countSize, channels);
                PipeBarrier<PIPE_V>();
                Div(output, output, expanded, rows * channels);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void BroadcastRows(LocalTensor<float> dst, LocalTensor<float> src,
                                         uint32_t rows, uint32_t columns) {
        const uint16_t stride = columns / 8U;
        // Each brcb repeats eight source scalars, each into one 32-byte block.
        #pragma unroll
        for (uint32_t block = 0; block < stride; ++block)
            Brcb(dst[block * 8U], src, rows / 8U,
                 BrcbRepeatParams(stride, columns));
    }
};

template <typename T>
class KernelSplineConv {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo,
        GM_ADDR weight, GM_ADDR root, GM_ADDR bias, GM_ADDR out,
        GM_ADDR workspace, const SplineConvTilingData& data) {
        t_ = data;
        x_.SetGlobalBuffer((__gm__ T*)x);
        edges_.SetGlobalBuffer((__gm__ int64_t*)edges);
        pseudo_.SetGlobalBuffer((__gm__ T*)pseudo);
        weights_.SetGlobalBuffer((__gm__ T*)weight);
        root_.SetGlobalBuffer((__gm__ T*)root);
        bias_.SetGlobalBuffer((__gm__ T*)bias);
        out_.SetGlobalBuffer((__gm__ T*)out);
        // Private, cache-line-aligned regions avoid cross-core scalar store races.
        next_.SetGlobalBuffer((__gm__ int32_t*)workspace +
            static_cast<uint64_t>(GetBlockIdx()) * t_.edgeStride);
        pipe_.InitBuffer(headBuffer_, ((t_.nodeTile + 7U) / 8U * 8U) * 4);
        pipe_.InitBuffer(targetBuffer_, 1024 * sizeof(int64_t));
        pipe_.InitBuffer(rawMatrixBuffer_, 128 * 128 * sizeof(T));
        pipe_.InitBuffer(matrixBuffer_, 128 * 128 * sizeof(float));
        pipe_.InitBuffer(rawVectorBuffer_, 128 * sizeof(T));
        pipe_.InitBuffer(xBuffer_, 128 * sizeof(float));
        pipe_.InitBuffer(pseudoBuffer_, 32);
        pipe_.InitBuffer(pseudoFloatBuffer_, 32);
        pipe_.InitBuffer(outputBuffer_, 128 * sizeof(T));
        pipe_.InitBuffer(nodeBuffer_, 128 * 4);
        pipe_.InitBuffer(nodeCompBuffer_, 128 * 4);
        pipe_.InitBuffer(messageBuffer_, 128 * 4);
        pipe_.InitBuffer(messageCompBuffer_, 128 * 4);
        pipe_.InitBuffer(dotBuffer_, 128 * 4);
        pipe_.InitBuffer(dotCompBuffer_, 128 * 4);
        pipe_.InitBuffer(productBuffer_, 128 * 4);
        pipe_.InitBuffer(yBuffer_, 128 * 4);
        pipe_.InitBuffer(sumBuffer_, 128 * 4);
        head_ = headBuffer_.Get<int32_t>();
        target_ = targetBuffer_.Get<int64_t>();
        matrix_ = matrixBuffer_.Get<float>();
        features_ = xBuffer_.Get<float>();
        node_ = nodeBuffer_.Get<float>();
        nodeComp_ = nodeCompBuffer_.Get<float>();
        message_ = messageBuffer_.Get<float>();
        messageComp_ = messageCompBuffer_.Get<float>();
        dot_ = dotBuffer_.Get<float>();
        dotComp_ = dotCompBuffer_.Get<float>();
        product_ = productBuffer_.Get<float>();
        y_ = yBuffer_.Get<float>();
        sum_ = sumBuffer_.Get<float>();
    }

    __aicore__ inline void Process() {
        const uint32_t columnTiles = (t_.outChannels + 127U) / 128U;
        const uint32_t nodeTiles = (t_.nodes + t_.nodeTile - 1U) / t_.nodeTile;
        const uint64_t tasks = static_cast<uint64_t>(nodeTiles) * columnTiles;
        for (uint64_t task = GetBlockIdx(); task < tasks; task += t_.blocks) {
            const uint32_t begin = static_cast<uint32_t>(task / columnTiles) * t_.nodeTile;
            outBegin_ = static_cast<uint32_t>(task % columnTiles) * 128U;
            outCount_ = t_.outChannels - outBegin_ < 128U ? t_.outChannels - outBegin_ : 128U;
            columns_ = (outCount_ + 15U) / 16U * 16U;
            const uint32_t count = t_.nodes - begin < t_.nodeTile ? t_.nodes - begin : t_.nodeTile;
            BuildAdjacency(begin, count);
            for (uint32_t localNode = 0; localNode < count; ++localNode) {
                Zero(node_, nodeComp_);
                uint32_t degree = 0;
                int32_t edge = head_.GetValue(localNode);
                while (edge >= 0) {
                    CalculateMessage(static_cast<uint32_t>(edge));
                    Accumulate(node_, nodeComp_, message_);
                    ++degree;
                    edge = next_.GetValue(static_cast<uint32_t>(edge));
                }
                if (t_.mean && degree) {
                    Muls(node_, node_, 1.0F / static_cast<float>(static_cast<int32_t>(degree)), outCount_);
                    PipeBarrier<PIPE_V>();
                }
                if (t_.hasRoot) {
                    if (t_.inChannels <= 128U) LoadFeatures(begin + localNode);
                    DotMatrix(root_, 0, begin + localNode);
                    Add(node_, node_, dot_, outCount_);
                    PipeBarrier<PIPE_V>();
                }
                if (t_.hasBias) {
                    LoadVector(bias_, outBegin_, outCount_, features_);
                    Add(node_, node_, features_, outCount_);
                    PipeBarrier<PIPE_V>();
                }
                auto output = outputBuffer_.Get<T>();
                if constexpr (IsSameType<T, float>::value) Adds(output, node_, 0.0F, outCount_);
                else Cast(output, node_, RoundMode::CAST_RINT, outCount_);
                PipeBarrier<PIPE_ALL>();
                const DataCopyExtParams copy{1, outCount_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPad(out_[static_cast<uint64_t>(begin + localNode) * t_.outChannels + outBegin_], output, copy);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

private:
    __aicore__ inline void BuildAdjacency(uint32_t begin, uint32_t count) {
        Duplicate(head_, -1, (count + 7U) / 8U * 8U);
        PipeBarrier<PIPE_ALL>();
        const DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
        for (uint32_t offset = 0; offset < t_.edges; offset += 1024) {
            const uint32_t amount = t_.edges - offset < 1024 ? t_.edges - offset : 1024;
            const DataCopyExtParams copy{1, amount * 8U, 0, 0, 0};
            DataCopyPad(target_, edges_[static_cast<uint64_t>(t_.edges) + offset], copy, pad);
            PipeBarrier<PIPE_ALL>();
            // Integer metadata routing; feature arithmetic uses vector instructions.
            for (uint32_t j = 0; j < amount; ++j) {
                const int64_t target = target_.GetValue(j);
                if (target >= begin && target < static_cast<int64_t>(begin + count)) {
                    const uint32_t local = static_cast<uint32_t>(target) - begin;
                    next_.SetValue(offset + j, head_.GetValue(local));
                    head_.SetValue(local, static_cast<int32_t>(offset + j));
                }
            }
        }
    }

    __aicore__ inline void LoadVector(const GlobalTensor<T>& input, uint64_t offset,
        uint32_t count, LocalTensor<float> result) {
        auto raw = rawVectorBuffer_.Get<T>();
        const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, T(0)};
        DataCopyPad(raw, input[offset], copy, pad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSameType<T, float>::value) Adds(result, raw, 0.0F, count);
        else Cast(result, raw, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadFeatures(uint32_t node) {
        LoadVector(x_, static_cast<uint64_t>(node) * t_.inChannels, t_.inChannels, features_);
    }

    __aicore__ inline void LoadMatrix(const GlobalTensor<T>& source, uint64_t offset, uint32_t rows) {
        auto raw = rawMatrixBuffer_.Get<T>();
        const uint32_t bytes = outCount_ * sizeof(T);
        const uint32_t alignedBytes = (bytes + 31U) / 32U * 32U;
        const uint32_t stride = (columns_ * sizeof(T) - alignedBytes) / 32U;
        const DataCopyExtParams copy{static_cast<uint16_t>(rows), bytes,
            (t_.outChannels - outCount_) * static_cast<uint32_t>(sizeof(T)), stride, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, T(0)};
        DataCopyPad(raw, source[offset], copy, pad);
        PipeBarrier<PIPE_ALL>();
        const uint32_t count = rows * columns_;
        if constexpr (IsSameType<T, float>::value) Adds(matrix_, raw, 0.0F, count);
        else Cast(matrix_, raw, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Zero(LocalTensor<float> value, LocalTensor<float> compensation) {
        Duplicate(value, 0.0F, outCount_);
        Duplicate(compensation, 0.0F, outCount_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Accumulate(LocalTensor<float> value,
        LocalTensor<float> compensation, LocalTensor<float> term) {
        // Compensate FP32 sums at dot-product, basis, and neighborhood levels.
        Sub(y_, term, compensation, outCount_);
        PipeBarrier<PIPE_V>();
        Add(sum_, value, y_, outCount_);
        PipeBarrier<PIPE_V>();
        Sub(compensation, sum_, value, outCount_);
        PipeBarrier<PIPE_V>();
        Sub(compensation, compensation, y_, outCount_);
        PipeBarrier<PIPE_V>();
        Adds(value, sum_, 0.0F, outCount_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DotMatrix(const GlobalTensor<T>& source, uint64_t offset, uint32_t node) {
        Zero(dot_, dotComp_);
        for (uint32_t begin = 0; begin < t_.inChannels; begin += 128U) {
            const uint32_t rows = t_.inChannels - begin < 128U ? t_.inChannels - begin : 128U;
            if (t_.inChannels > 128U)
                LoadVector(x_, static_cast<uint64_t>(node) * t_.inChannels + begin, rows, features_);
            LoadMatrix(source, offset + static_cast<uint64_t>(begin) * t_.outChannels + outBegin_, rows);
            for (uint32_t c = 0; c < rows; ++c) {
                Muls(product_, matrix_[c * columns_], features_.GetValue(c), outCount_);
                PipeBarrier<PIPE_V>();
                Accumulate(dot_, dotComp_, product_);
            }
        }
    }

    __aicore__ inline float Polynomial(float v, uint32_t j) {
        if (t_.degree == 1) return j == 0 ? 1.0F - v : v;
        const float v2 = v * v;
        if (t_.degree == 2) {
            if (j == 0) return 0.5F * v2 - v + 0.5F;
            if (j == 1) return -v2 + v + 0.5F;
            return 0.5F * v2;
        }
        const float v3 = v2 * v;
        if (j == 0) { const float u = 1.0F - v; return u * u * u / 6.0F; }
        if (j == 1) return (3.0F * v3 - 6.0F * v2 + 4.0F) / 6.0F;
        if (j == 2) return (-3.0F * v3 + 3.0F * v2 + 3.0F * v + 1.0F) / 6.0F;
        return v3 / 6.0F;
    }

    __aicore__ inline void CalculateMessage(uint32_t edge) {
        const uint32_t sourceNode = static_cast<uint32_t>(edges_.GetValue(edge));
        if (t_.inChannels <= 128U) LoadFeatures(sourceNode);
        auto raw = pseudoBuffer_.Get<T>();
        auto coord = pseudoFloatBuffer_.Get<float>();
        const DataCopyExtParams copy{1, t_.dim * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, T(0)};
        DataCopyPad(raw, pseudo_[static_cast<uint64_t>(edge) * t_.dim], copy, pad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSameType<T, float>::value) Adds(coord, raw, 0.0F, t_.dim);
        else Cast(coord, raw, RoundMode::CAST_NONE, t_.dim);
        PipeBarrier<PIPE_ALL>();
        int32_t base[3];
        float basis[3][4];
        uint32_t slots = 1;
        for (uint32_t d = 0; d < t_.dim; ++d) {
            const float scaled = coord.GetValue(d) *
                static_cast<float>(static_cast<int32_t>(t_.kernelSize[d]) -
                    static_cast<int32_t>(t_.degree * t_.open[d]));
            base[d] = static_cast<int32_t>(scaled);
            if (static_cast<float>(base[d]) > scaled) --base[d];
            const float fraction = scaled - static_cast<float>(base[d]);
            for (uint32_t j = 0; j <= t_.degree; ++j) basis[d][j] = Polynomial(fraction, j);
            slots *= t_.degree + 1;
        }
        Zero(message_, messageComp_);
        for (uint32_t s = 0; s < slots; ++s) {
            uint32_t digits = s, index = 0, stride = 1;
            float coefficient = 1.0F;
            for (uint32_t d = 0; d < t_.dim; ++d) {
                const uint32_t j = digits % (t_.degree + 1);
                digits /= t_.degree + 1;
                int32_t wrapped = (base[d] + static_cast<int32_t>(j)) %
                    static_cast<int32_t>(t_.kernelSize[d]);
                if (wrapped < 0) wrapped += t_.kernelSize[d];
                index += static_cast<uint32_t>(wrapped) * stride;
                stride *= t_.kernelSize[d];
                coefficient *= basis[d][j];
            }
            DotMatrix(weights_, static_cast<uint64_t>(index) * t_.inChannels * t_.outChannels,
                sourceNode);
            Muls(product_, dot_, coefficient, outCount_);
            PipeBarrier<PIPE_V>();
            Accumulate(message_, messageComp_, product_);
        }
    }

    SplineConvTilingData t_;
    uint32_t columns_, outBegin_, outCount_;
    TPipe pipe_;
    GlobalTensor<T> x_, pseudo_, weights_, root_, bias_, out_;
    GlobalTensor<int64_t> edges_;
    GlobalTensor<int32_t> next_;
    TBuf<TPosition::VECCALC> headBuffer_, matrixBuffer_, xBuffer_, pseudoFloatBuffer_;
    TBuf<TPosition::VECIN> targetBuffer_, rawMatrixBuffer_, rawVectorBuffer_, pseudoBuffer_;
    TBuf<TPosition::VECOUT> outputBuffer_;
    TBuf<TPosition::VECCALC> nodeBuffer_, nodeCompBuffer_, messageBuffer_, messageCompBuffer_;
    TBuf<TPosition::VECCALC> dotBuffer_, dotCompBuffer_, productBuffer_, yBuffer_, sumBuffer_;
    LocalTensor<int32_t> head_;
    LocalTensor<int64_t> target_;
    LocalTensor<float> matrix_, features_, node_, nodeComp_, message_, messageComp_;
    LocalTensor<float> dot_, dotComp_, product_, y_, sum_;
};

template <uint32_t NODE_LOG>
__aicore__ inline constexpr uint32_t StaticSplineNodes() {
    if constexpr (NODE_LOG > SPLINE_TINY_SINGLE_NODE_OFFSET)
        return NODE_LOG - SPLINE_TINY_SINGLE_NODE_OFFSET;
    else if constexpr (NODE_LOG != 0U)
        return 1U << (NODE_LOG - 1U);
    else return 0U;
}

template <typename DT_X, uint32_t SCHEDULE, uint32_t PADDED_ROWS, uint32_t EDGE_MODE,
          uint32_t NODE_LOG, uint32_t FULL_WORKERS, uint32_t HALF_WORKERS, uint32_t HALF_MEAN_MODE>
struct SplineDispatch {
    GM_ADDR x;
    GM_ADDR edge_index;
    GM_ADDR edge_attr;
    GM_ADDR weight;
    GM_ADDR root_weight;
    GM_ADDR bias;
    GM_ADDR out;
    GM_ADDR workspace;
    GM_ADDR tiling;
    static constexpr uint32_t fixedNodes = StaticSplineNodes<NODE_LOG>();
    static constexpr bool vectorHalf = PADDED_ROWS >= 32U && PADDED_ROWS <= 64U &&
        fixedNodes >= 8U && fixedNodes <= 64U && fixedNodes % 8U == 0U;
    static constexpr bool irregularMean = HALF_MEAN_MODE == 2U && PADDED_ROWS < 128U &&
        fixedNodes >= 8U && fixedNodes <= 64U && fixedNodes % 8U != 0U;

    __aicore__ inline __attribute__((always_inline)) void Run() {
        if constexpr (SCHEDULE == 1 && (EDGE_MODE == 1 || EDGE_MODE == 3)) Single();
        else if constexpr (SCHEDULE == 1 && FULL_WORKERS == 1) Full();
        else if constexpr (SCHEDULE == 1 && HALF_WORKERS == 1) Half();
        else if constexpr (SCHEDULE == 1) Resident();
        else General();
    }

    template <typename Kernel>
    __aicore__ inline __attribute__((always_inline)) void Owner(const SplineResidentTilingData& data) {
        Kernel op;
        op.Run(x, edge_index, edge_attr, weight, out, data);
    }

    __aicore__ inline __attribute__((always_inline)) void Single() {
        if constexpr (EDGE_MODE == 3U && fixedNodes > 0U && fixedNodes <= 32U &&
                      (fixedNodes * 16U) % 64U != 0U) {
            KernelSplineSingle<PADDED_ROWS, fixedNodes, true, true> op;
            op.Run(x, edge_index, edge_attr, weight, out, tiling);
        } else {
            KernelSplineSingle<PADDED_ROWS, fixedNodes, EDGE_MODE == 3> op;
            op.Run(x, edge_index, edge_attr, weight, out, tiling);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void Full() {
        PreloadSplineRouting(edge_index, edge_attr);
        PreloadSplineWeightsStreamed<PADDED_ROWS>(weight);
        constexpr bool staticVectorTile = NODE_LOG >= SPLINE_FULL_STATIC_NODE_MODE_MIN &&
            NODE_LOG <= SPLINE_FULL_STATIC_NODE_MODE_MAX;
        constexpr bool staticCompactRows = NODE_LOG >= SPLINE_FULL_COMPACT_NODE_MODE_MIN &&
            NODE_LOG <= SPLINE_FULL_COMPACT_NODE_MODE_MAX;
        constexpr uint32_t staticNodes = staticVectorTile ?
            (NODE_LOG - SPLINE_FULL_VECTOR_TILE_NODE_MODE) * SPLINE_FULL_VECTOR_TILE_NODE_MULTIPLE :
            staticCompactRows ? (NODE_LOG - SPLINE_FULL_COMPACT_NODE_OFFSET) *
                SPLINE_FULL_COMPACT_NODE_MULTIPLE : 0U;
        const SplineResidentTilingData data = LoadFullOwnerTiling<PADDED_ROWS, staticNodes, HALF_MEAN_MODE>(tiling);
        if constexpr (PADDED_ROWS >= 128) {
            if constexpr (NODE_LOG == SPLINE_FULL_VECTOR_TILE_NODE_MODE || staticVectorTile)
                Owner<KernelSplineNativeConsumerReadiness<PADDED_ROWS, true, true, true, true>>(data);
            else if constexpr (NODE_LOG == 1U || staticCompactRows)
                Owner<KernelSplineCompactOwner<PADDED_ROWS, true, true, true, true, true>>(data);
            else Owner<KernelSplineCompactOwner<PADDED_ROWS, true, true, NODE_LOG == 1U, true>>(data);
        } else Owner<KernelSplineEdgeOwnerLate<PADDED_ROWS, true>>(data);
    }

    template <bool UNLIKELY_DUPLICATES>
    __aicore__ inline __attribute__((always_inline)) void HalfNumeric(const SplineResidentTilingData& data) {
        if (GetBlockIdx() < 4U) {
            PreloadSplineHalfCoordinates(edge_attr);
            PreloadSplineWeightsHalfBlocks<PADDED_ROWS, 2U>(weight);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
            Owner<KernelSplineNumericResident<PADDED_ROWS, true, true, UNLIKELY_DUPLICATES>>(data);
        } else {
            KernelSplineHalfOriginalSeparateZero op;
            op.Run(edge_index, out, data.nodes);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void Half() {
        if constexpr (irregularMean) {
            PreloadSplineHalfEdges(edge_index);
            const SplineResidentTilingData data = LoadHalfOwnerTiling<PADDED_ROWS, fixedNodes, HALF_MEAN_MODE>();
            HalfNumeric<true>(data);
        } else {
            if constexpr (vectorHalf) PreloadSplineHalfEdges(edge_index);
            else {
                PreloadSplineRouting(edge_index, edge_attr);
                PreloadSplineWeights<PADDED_ROWS>(weight);
            }
            const SplineResidentTilingData data = LoadHalfOwnerTiling<PADDED_ROWS, fixedNodes, HALF_MEAN_MODE>();
            if constexpr (fixedNodes > 0U && fixedNodes < 8U)
                Owner<KernelSplineResident<PADDED_ROWS, true, true>>(data);
            else if constexpr (PADDED_ROWS >= 128)
                Owner<KernelSplineEdgeOwner<PADDED_ROWS, true, false, (fixedNodes <= 512U && fixedNodes % 8U == 0U)>>(data);
            else if constexpr (vectorHalf) HalfNumeric<false>(data);
            else Owner<KernelSplineEdgeOwnerLate<PADDED_ROWS, true, (fixedNodes <= 512U && fixedNodes % 8U == 0U), true>>(data);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void Resident() {
        PreloadSplineRouting(edge_index, edge_attr);
        constexpr bool earlyWeights = EDGE_MODE == 2 && PADDED_ROWS > 0U;
        if constexpr (earlyWeights) PreloadSplineWeights<PADDED_ROWS>(weight);
        const SplineResidentTilingData data = LoadResidentTiling<PADDED_ROWS, earlyWeights>(tiling);
        if constexpr (PADDED_ROWS >= 32) {
            if (data.edges > 1U && data.edges <= 8U && GetBlockNum() == 8U) {
                if constexpr (PADDED_ROWS >= 128) Owner<KernelSplineEdgeOwner<PADDED_ROWS, earlyWeights>>(data);
                else Owner<KernelSplineEdgeOwnerLate<PADDED_ROWS, earlyWeights>>(data);
            } else Owner<KernelSplineResident<PADDED_ROWS, earlyWeights>>(data);
        } else Owner<KernelSplineResident<PADDED_ROWS, earlyWeights>>(data);
    }

    __aicore__ inline __attribute__((always_inline)) void General() {
        GET_TILING_DATA_WITH_STRUCT(SplineConvTilingData, data, tiling);
        KernelSplineConv<DT_X> op;
        op.Init(x, edge_index, edge_attr, weight, root_weight, bias, out, GetUserWorkspace(workspace), data);
        op.Process();
    }
};

template <typename DT_X, uint32_t SCHEDULE, uint32_t PADDED_ROWS, uint32_t EDGE_MODE, uint32_t NODE_LOG, uint32_t FULL_WORKERS, uint32_t HALF_WORKERS, uint32_t HALF_MEAN_MODE>
__aicore__ __attribute__((always_inline)) inline void RunSplineConv(
    GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr, GM_ADDR weight,
    GM_ADDR root_weight, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    SplineDispatch<DT_X, SCHEDULE, PADDED_ROWS, EDGE_MODE, NODE_LOG, FULL_WORKERS, HALF_WORKERS, HALF_MEAN_MODE> dispatch{
        x, edge_index, edge_attr, weight, root_weight, bias, out, workspace, tiling};
    dispatch.Run();
}

template <typename DT_X, uint32_t SCHEDULE, uint32_t PADDED_ROWS, uint32_t EDGE_MODE, uint32_t NODE_LOG, uint32_t FULL_WORKERS, uint32_t HALF_WORKERS, uint32_t HALF_MEAN_MODE>
__global__ __aicore__ void spline_conv(GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr,
    GM_ADDR weight, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    RunSplineConv<DT_X, SCHEDULE, PADDED_ROWS, EDGE_MODE, NODE_LOG, FULL_WORKERS, HALF_WORKERS, HALF_MEAN_MODE>(
        x, edge_index, edge_attr, weight, nullptr, nullptr, out, workspace, tiling);
}

template <typename DT_X, uint32_t SCHEDULE, uint32_t PADDED_ROWS, uint32_t EDGE_MODE, uint32_t NODE_LOG, uint32_t FULL_WORKERS, uint32_t HALF_WORKERS, uint32_t HALF_MEAN_MODE>
__global__ __aicore__ void spline_conv_full(GM_ADDR x, GM_ADDR edge_index, GM_ADDR edge_attr,
    GM_ADDR weight, GM_ADDR root_weight, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SplineConvTilingData);
    RunSplineConv<DT_X, SCHEDULE, PADDED_ROWS, EDGE_MODE, NODE_LOG, FULL_WORKERS, HALF_WORKERS, HALF_MEAN_MODE>(
        x, edge_index, edge_attr, weight, root_weight, bias, out, workspace, tiling);
}


namespace AscendC {
template <typename DT_X, uint32_t SCHEDULE, uint32_t PADDED_ROWS,
          uint32_t EDGE_MODE, uint32_t NODE_LOG, uint32_t FULL_WORKERS,
          uint32_t HALF_WORKERS, uint32_t HALF_MEAN_MODE>
__aicore__ inline void SplineInitializeRequiredWorkspace(GM_ADDR workspace) {
    constexpr uint32_t fixedNodes = ::StaticSplineNodes<NODE_LOG>();
    constexpr bool single = SCHEDULE == 1U && (EDGE_MODE == 1U || EDGE_MODE == 3U);
    constexpr bool paddedZero = EDGE_MODE == 3U && fixedNodes > 0U && fixedNodes <= 32U &&
                                (fixedNodes * 16U) % 64U != 0U;
    constexpr bool half = SCHEDULE == 1U && HALF_WORKERS == 1U;
    constexpr bool privateResident = single || half;
    if constexpr (!privateResident) {
        SetSysWorkspaceForce(workspace);
    }
}
}

// CANN 9 emits TEMPLATE_PARAMS before the wrapper's initialization call.
// Instrumented builds retain the original system-workspace initialization.
#if !ASCENDC_CPU_DEBUG && (!defined(ASCENDC_DUMP) || ASCENDC_DUMP == 0) && \
    !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && !defined(ASCENDC_ACC_DUMP)
#define SetSysWorkspaceForce(workspace) \
    SplineInitializeRequiredWorkspace<TEMPLATE_PARAMS>(workspace)
#endif
