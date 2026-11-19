#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"

// Implementation from kp_conv_statistics.h
using namespace AscendC;

__aicore__ inline void ScaleTemperature(const LocalTensor<float> &values, const LocalTensor<float> &scratch,
                                         float tau, int32_t count)
{
    PipeBarrier<PIPE_V>();
    if (tau == 1.0f) return;
    if (tau == 0.5f || tau == 2.0f) {
        Muls(values, values, tau == 0.5f ? 2.0f : 0.5f, count);
    } else {
        Duplicate(scratch, tau, count);
        PipeBarrier<PIPE_V>();
        Div(values, values, scratch, count);
    }
    PipeBarrier<PIPE_V>();
}

// Exact zero and single-input-channel weight routes, selected from runtime weights.
template <class Tiling, uint32_t Height, uint32_t Width>
class SparseFullImage {
    static constexpr uint32_t Pixels = Height * Width;
    static constexpr uint32_t Pitch = Width + 8;
    static constexpr uint32_t HaloElements = (Height + 2) * Pitch;
    TPipe pipe;
    TBuf<TPosition::VECCALC> maskBuffer, sumsBuffer, valuesBuffer;
    TBuf<TPosition::VECCALC> offsetBuffer, coordinateBuffer, workBuffer;
    TBuf<TPosition::VECCALC> reducedBuffer, scratchBuffer;
    TQue<TPosition::VECIN, 1> weightQueue, biasQueue, inputQueue;
    TQue<TPosition::VECOUT, 1> outputQueue;
    GlobalTensor<float> features, filters, biases, points, confidence;
    int32_t channels[4];
    uint16_t tapMasks[4];

    __aicore__ inline void Write(float x, float y, float score)
    {
        auto packet = outputQueue.AllocTensor<float>();
        packet.SetValue(0, x);
        packet.SetValue(1, y);
        packet.SetValue(8, score);
        PipeBarrier<PIPE_ALL>();
        outputQueue.EnQue(packet);
        packet = outputQueue.DeQue<float>();
        DataCopyPad(points, packet, DataCopyExtParams{1, 8, 0, 0, 0});
        DataCopyPad(confidence, packet[8], DataCopyExtParams{1, 4, 0, 0, 0});
        outputQueue.FreeTensor(packet);
    }

    __aicore__ inline bool Classify(const LocalTensor<float> &weights)
    {
        LocalTensor<uint8_t> mask = maskBuffer.Get<uint8_t>();
        CompareScalar(mask, weights, 0.0f, CMPMODE::NE, 1152);
        PipeBarrier<PIPE_ALL>();
        uint64_t bits[18];
        uint32_t count = 0;
        auto words = mask.ReinterpretCast<uint64_t>();
        for (uint32_t word = 0; word < 18; ++word) {
            bits[word] = words.GetValue(word);
            if (bits[word] == ~static_cast<uint64_t>(0)) return false;
            uint64_t remaining = bits[word];
            while (remaining) {
                if (++count > 36) return false;
                remaining &= remaining - 1;
            }
        }
        if (!count) {
            for (uint32_t keypoint = 0; keypoint < 4; ++keypoint) channels[keypoint] = -1;
            return true;
        }
        for (uint32_t keypoint = 0; keypoint < 4; ++keypoint) {
            channels[keypoint] = -1;
            for (uint32_t channel = 0; channel < 32; ++channel) {
                uint32_t first = keypoint * 288 + channel * 9;
                uint32_t word = first / 64;
                uint32_t shift = first % 64;
                uint64_t active = bits[word] >> shift;
                if (shift > 55 && word + 1 < 18) active |= bits[word + 1] << (64 - shift);
                const uint16_t taps = static_cast<uint16_t>(active & 511U);
                if (taps != 0U) {
                    if (channels[keypoint] >= 0) return false;
                    channels[keypoint] = static_cast<int32_t>(channel);
                    tapMasks[keypoint] = taps;
                }
            }
        }
        return true;
    }

    __aicore__ inline void MakeCoordinates(bool needOffsets)
    {
        auto coordinates = coordinateBuffer.Get<float>();
        LocalTensor<float> values = valuesBuffer.Get<float>();
        auto y = workBuffer.Get<float>();
        auto offsets = offsetBuffer.Get<int32_t>();
        CreateVecIndex(coordinates, 0.0f, Pixels);
        PipeBarrier<PIPE_V>();
        Muls(values, coordinates, 1.0f / static_cast<float>(static_cast<int32_t>(Width)), Pixels);
        PipeBarrier<PIPE_V>();
        Cast(offsets, values, RoundMode::CAST_FLOOR, Pixels);
        PipeBarrier<PIPE_V>();
        Cast(y, offsets, RoundMode::CAST_NONE, Pixels);
        PipeBarrier<PIPE_V>();
        Muls(values, y, static_cast<float>(static_cast<int32_t>(Width)), Pixels);
        if (needOffsets) Muls(offsets, offsets, static_cast<int32_t>(Pitch), Pixels);
        PipeBarrier<PIPE_V>();
        Sub(coordinates, coordinates, values, Pixels);
        PipeBarrier<PIPE_V>();
        if (needOffsets) {
            auto x = values.ReinterpretCast<int32_t>();
            Cast(x, coordinates, RoundMode::CAST_RINT, Pixels);
            PipeBarrier<PIPE_V>();
            Add(offsets, offsets, x, Pixels);
            PipeBarrier<PIPE_V>();
            Muls(offsets, offsets, static_cast<int32_t>(4), Pixels);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyIn(bool centerOnly)
    {
        auto halo = inputQueue.AllocTensor<float>();
        if (centerOnly) {
            DataCopyPad(halo, features, DataCopyExtParams{1, Pixels * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            Duplicate(halo, 0.0f, Pitch);
            Duplicate(halo[(Height + 1) * Pitch], 0.0f, Pitch);
            DataCopyPad(halo[Pitch], features, DataCopyExtParams{Height, Width * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 1, 7, 0});
            PipeBarrier<PIPE_V>();
        }
        inputQueue.EnQue(halo);
    }

public:
    __aicore__ inline bool Process(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias,
                                  GM_ADDR keypoints, GM_ADDR scores, const Tiling &config)
    {
        pipe.InitBuffer(maskBuffer, 256);
        pipe.InitBuffer(weightQueue, 1, 1152 * 4);
        pipe.InitBuffer(biasQueue, 1, 32);
        pipe.InitBuffer(outputQueue, 1, 64);
        filters.SetGlobalBuffer((__gm__ float *)weight);
        biases.SetGlobalBuffer((__gm__ float *)bias);
        auto weights = weightQueue.AllocTensor<float>();
        DataCopyPad(weights, filters, DataCopyExtParams{1, 1152 * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        weightQueue.EnQue(weights);
        weights = weightQueue.DeQue<float>();
        if (!Classify(weights)) {
            weightQueue.FreeTensor(weights);
            return false;
        }
        uint32_t job = GetBlockIdx();
        if (job >= config.batch * 4) {
            weightQueue.FreeTensor(weights);
            return true;
        }
        uint32_t batch = job / 4;
        uint32_t keypoint = job % 4;
        points.SetGlobalBuffer((__gm__ float *)keypoints + job * 2);
        confidence.SetGlobalBuffer((__gm__ float *)scores + job);
        auto biasValues = biasQueue.AllocTensor<float>();
        DataCopyPad(biasValues, biases, DataCopyExtParams{1, 16, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        biasQueue.EnQue(biasValues);
        biasValues = biasQueue.DeQue<float>();
        PipeBarrier<PIPE_ALL>();
        float offset = biasValues.GetValue(keypoint);
        if (channels[keypoint] < 0) {
            const float x = static_cast<float>(static_cast<int32_t>(Width)) * 0.5f * config.stride;
            const float y = static_cast<float>(static_cast<int32_t>(Height)) * 0.5f * config.stride;
            Write(x, y, offset);
            biasQueue.FreeTensor(biasValues);
            weightQueue.FreeTensor(weights);
            return true;
        }
        pipe.InitBuffer(inputQueue, 1, HaloElements * 4);
        pipe.InitBuffer(sumsBuffer, Pixels * 4);
        pipe.InitBuffer(valuesBuffer, Pixels * 4);
        pipe.InitBuffer(offsetBuffer, Pixels * 4);
        pipe.InitBuffer(coordinateBuffer, Pixels * 4);
        pipe.InitBuffer(workBuffer, Pixels * 4);
        pipe.InitBuffer(reducedBuffer, 96);
        pipe.InitBuffer(scratchBuffer, 1024);
        const uint32_t first = keypoint * 288 + static_cast<uint32_t>(channels[keypoint]) * 9U;
        const uint16_t taps = tapMasks[keypoint];
        const bool centerOnly = taps == (1U << 4);
        features.SetGlobalBuffer((__gm__ float *)feat +
            (batch * 32 + static_cast<uint32_t>(channels[keypoint])) * Pixels);
        CopyIn(centerOnly);
        MakeCoordinates(!centerOnly);
        auto halo = inputQueue.DeQue<float>();
        auto sums = sumsBuffer.Get<float>();
        auto values = valuesBuffer.Get<float>();
        auto reduced = reducedBuffer.Get<float>();
        auto scratch = scratchBuffer.Get<float>();
        if (centerOnly) {
            Muls(sums, halo, weights.GetValue(first + 4U), Pixels);
        } else {
            auto offsets = offsetBuffer.Get<uint32_t>();
            Duplicate(sums, 0.0f, Pixels);
            PipeBarrier<PIPE_V>();
            for (uint32_t tap = 0; tap < 9; ++tap) {
                if ((taps & (1U << tap)) != 0U) {
                    const uint32_t base = ((tap / 3U) * Pitch + tap % 3U) * 4U;
                    Gather(values, halo, offsets, base, Pixels);
                    PipeBarrier<PIPE_V>();
                    Axpy(sums, values, weights.GetValue(first + tap), Pixels);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        PipeBarrier<PIPE_V>();
        inputQueue.FreeTensor(halo);
        Adds(sums, sums, offset, Pixels);
        PipeBarrier<PIPE_V>();
        ReduceMax(reduced, sums, scratch, Pixels);
        PipeBarrier<PIPE_ALL>();
        float maximum = reduced.GetValue(0);
        Adds(sums, sums, -maximum, Pixels);
        ScaleTemperature(sums, values, config.tau, Pixels);
        Exp(sums, sums, Pixels);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduced, sums, scratch, Pixels);
        Mul(values, sums, coordinateBuffer.Get<float>(), Pixels);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduced[8], values, scratch, Pixels);
        PipeBarrier<PIPE_V>();
        Mul(values, sums, workBuffer.Get<float>(), Pixels);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduced[16], values, scratch, Pixels);
        PipeBarrier<PIPE_ALL>();
        const float denominator = reduced.GetValue(0);
        Write((reduced.GetValue(8) / denominator + 0.5f) * config.stride,
              (reduced.GetValue(16) / denominator + 0.5f) * config.stride, maximum);
        biasQueue.FreeTensor(biasValues);
        weightQueue.FreeTensor(weights);
        return true;
    }
};

// Channel-parallel convolution and four-keypoint online statistics per batch worker.
template <class Tiling, uint32_t Width, uint32_t Channels>
class BatchedDenseStatistics {
    static constexpr uint32_t Count = 2 * Width;
    static constexpr uint32_t ChannelSize = 4 * Width;
    static constexpr uint32_t ViewSize = Channels * ChannelSize;
    static constexpr uint32_t KeypointsPerPass = Width == 64 ? 2 : 4;
    TPipe pipe;
    Tiling config;
    TQue<TPosition::VECIN, 1> inputQueue, weightQueue, biasQueue;
    TQue<TPosition::VECOUT, 1> outputQueue;
    TBuf<TPosition::VECCALC> partialStore, sumStore;
    TBuf<TPosition::VECCALC> coordinateStore, workStore, weightStore;
    TBuf<TPosition::VECCALC> reductionStore, tileStore, runningStore;
    TBuf<TPosition::VECCALC> broadcastStore, combineStore, packetIndexStore;
    GlobalTensor<float> features, filters, biases, statistics;

    __aicore__ inline void Fence() { PipeBarrier<PIPE_V>(); }

    template <bool Maximum>
    __aicore__ inline void ReduceRows(const LocalTensor<float> &destination,
                                     const LocalTensor<float> &source)
    {
        auto parts = reductionStore.Get<float>();
        Duplicate(parts, 0.0f, 24);
        Fence();
        for (uint32_t segment = 0; segment < Count / 64; ++segment) {
            if constexpr (Maximum) {
                WholeReduceMax(parts[segment * 8], source[segment * 64],
                    static_cast<uint64_t>(64), 4, 1, 1, Count / 8,
                    ReduceOrder::ORDER_ONLY_VALUE);
            } else {
                WholeReduceSum(parts[segment * 8], source[segment * 64],
                    static_cast<uint64_t>(64), 4, 1, 1, Count / 8);
            }
        }
        Fence();
        Adds(destination, parts, 0.0f, 8);
        Fence();
        for (uint32_t segment = 1; segment < Count / 64; ++segment) {
            if constexpr (Maximum) Max(destination, destination, parts[segment * 8], 8);
            else Add(destination, destination, parts[segment * 8], 8);
            Fence();
        }
    }

    __aicore__ inline void MakeIndices()
    {
        auto xy = coordinateStore.Get<float>();
        CreateVecIndex(xy, 0.0f, Width);
        Fence();
        Adds(xy[Width], xy, 0.0f, Width);
        Duplicate(xy[Count], 0.0f, Width);
        Duplicate(xy[Count + Width], 1.0f, Width);
        auto packetIndices = packetIndexStore.Get<uint32_t>();
        for (uint32_t keypoint = 0; keypoint < 4; ++keypoint) {
            for (uint32_t field = 0; field < 4; ++field) {
                packetIndices.SetValue(keypoint * 4 + field, (field * 8 + keypoint) * 4);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyIn(uint32_t batch, uint32_t tile, uint32_t channelBase)
    {
        auto halo = inputQueue.AllocTensor<float>();
        const uint32_t y = tile * 2;
        const bool top = y == 0;
        const bool bottom = y + 2 == config.height;
        const uint32_t firstRow = top ? 1 : 0;
        const uint32_t endRow = bottom ? 3 : 4;
        const uint32_t plane = config.height * Width;
        const uint32_t firstChannel = (batch * 32 + channelBase) * plane;
        for (uint32_t view = 0; view < 3; ++view) {
            auto panel = halo[view * ViewSize];
            if (top || bottom) {
                auto edge = panel[top ? 0 : 3 * Width];
                Duplicate(edge, 0.0f, static_cast<uint64_t>(64), Channels, 1, ChannelSize / 8);
                if constexpr (Width > 64) {
                    Duplicate(edge[64], 0.0f, static_cast<uint64_t>(Width - 64), Channels, 1,
                              ChannelSize / 8);
                }
            }
            // Copy W-1 values for each shifted view and insert its single zero.
            const bool center = view == 1;
            const uint32_t copied = center ? Width : Width - 1;
            const uint8_t left = view == 0 ? 1 : 0;
            const uint8_t right = view == 2 ? 1 : 0;
            const DataCopyExtParams ext{Channels, copied * 4, (plane - copied) * 4,
                                        (ChannelSize - Width) / 8, 0};
            const DataCopyPadExtParams<float> padding{!center, left, right, 0.0f};
            for (uint32_t row = firstRow; row < endRow; ++row) {
                const uint32_t source = firstChannel + (y + row - 1) * Width + right;
                DataCopyPad(panel[row * Width], features[source], ext, padding);
            }
        }
        if (top || bottom) Fence();
        inputQueue.EnQue(halo);
    }

    __aicore__ inline void Accumulate(const LocalTensor<float> &halo, uint32_t channelBase,
                                      uint32_t keypointBase)
    {
        auto partial = partialStore.Get<float>();
        auto coefficients = weightStore.Get<float>();
        const BinaryRepeatParams multiply{1, 1, 0, Count / 8, ChannelSize / 8, 9};
        for (uint32_t tap = 0; tap < 9; ++tap) {
            const uint32_t src = (tap % 3) * ViewSize + (tap / 3) * Width;
            for (uint32_t keypoint = 0; keypoint < KeypointsPerPass; ++keypoint) {
                const uint32_t coefficient = ((keypointBase + keypoint) * 288 + channelBase * 9 + tap) * 8;
                for (uint32_t position = 0; position < Count; position += 64) {
                    MulAddDst(partial[keypoint * Channels * Count + position], halo[src + position],
                              coefficients[coefficient], 64, Channels, multiply);
                }
            }
            Fence();
        }
    }

    __aicore__ inline void ReduceChannels(uint32_t keypointBase)
    {
        auto partial = partialStore.Get<float>();
        // Each filter has a contiguous channel tree, avoiding uint8 stride overflow.
        for (uint32_t half = Channels / 2; half; half /= 2) {
            for (uint32_t keypoint = 0; keypoint < KeypointsPerPass; ++keypoint) {
                auto first = partial[keypoint * Channels * Count];
                Add(first, first, first[half * Count], half * Count);
            }
            Fence();
        }
        auto sums = sumStore.Get<float>();
        for (uint32_t keypoint = 0; keypoint < KeypointsPerPass; ++keypoint) {
            Adds(sums[(keypointBase + keypoint) * Count], partial[keypoint * Channels * Count], 0.0f, Count);
        }
        Fence();
    }

    __aicore__ inline void Convolve(uint32_t batch, uint32_t tile)
    {
        auto partial = partialStore.Get<float>();
        if constexpr (Channels == 32) {
            CopyIn(batch, tile, 0);
            auto halo = inputQueue.DeQue<float>();
            // Share one 32-channel input among two pairs of keypoints to fit UB.
            for (uint32_t keypointBase = 0; keypointBase < 4; keypointBase += KeypointsPerPass) {
                Duplicate(partial, 0.0f, KeypointsPerPass * Channels * Count);
                Fence();
                Accumulate(halo, 0, keypointBase);
                ReduceChannels(keypointBase);
            }
            inputQueue.FreeTensor(halo);
        } else {
            Duplicate(partial, 0.0f, KeypointsPerPass * Channels * Count);
            Fence();
            for (uint32_t channelBase = 0; channelBase < 32; channelBase += Channels) {
                CopyIn(batch, tile, channelBase);
                auto halo = inputQueue.DeQue<float>();
                Accumulate(halo, channelBase, 0);
                inputQueue.FreeTensor(halo);
            }
            ReduceChannels(0);
        }
    }

    __aicore__ inline void CalculateStatistics(uint32_t tile, const LocalTensor<float> &bias)
    {
        auto sums = sumStore.Get<float>();
        auto fields = tileStore.Get<float>();
        auto broadcasts = broadcastStore.Get<float>();
        auto work = workStore.Get<float>();
        auto xy = coordinateStore.Get<float>();
        Brcb(broadcasts, bias, 1, {1, 8});
        Fence();
        BinaryRepeatParams rowScalar{1, 1, 0, Count / 8, Count / 8, 1};
        for (uint32_t position = 0; position < Count; position += 64) {
            Add(sums[position], sums[position], broadcasts, 64, 4, rowScalar);
        }
        Fence();
        ReduceRows<true>(fields, sums);
        Brcb(broadcasts, fields, 1, {1, 8});
        Fence();
        for (uint32_t position = 0; position < Count; position += 64) {
            Sub(work[position], sums[position], broadcasts, 64, 4, rowScalar);
        }
        ScaleTemperature(work, work[4 * Count], config.tau, 4 * Count);
        Exp(work, work, 4 * Count);
        Fence();
        ReduceRows<false>(fields[8], work);
        BinaryRepeatParams coordinate{1, 1, 1, Count / 8, Count / 8, 0};
        for (uint32_t position = 0; position < Count; position += 64) {
            Mul(work[4 * Count + position], work[position], xy[position], 64, 4, coordinate);
        }
        Fence();
        ReduceRows<false>(fields[16], work[4 * Count]);
        for (uint32_t position = 0; position < Count; position += 64) {
            Mul(work[4 * Count + position], work[position], xy[Count + position], 64, 4, coordinate);
        }
        Fence();
        ReduceRows<false>(fields[24], work[4 * Count]);
        float y = static_cast<float>(static_cast<int32_t>(tile * 2));
        Axpy(fields[24], fields[8], y, 8);
        Fence();
    }

    __aicore__ inline void CombineStatistics(uint32_t iteration)
    {
        auto tile = tileStore.Get<float>();
        auto running = runningStore.Get<float>();
        if (!iteration) {
            Adds(running, tile, 0.0f, 32);
            Fence();
            return;
        }
        auto temporary = combineStore.Get<float>();
        Max(temporary, running, tile, 8);
        Fence();
        Sub(temporary[8], running, temporary, 8);
        Sub(temporary[16], tile, temporary, 8);
        Fence();
        if (config.tau == 0.5f || config.tau == 2.0f) {
            Muls(temporary[8], temporary[8], config.tau == 0.5f ? 2.0f : 0.5f, 16);
            Fence();
        } else if (config.tau != 1.0f) {
            Duplicate(temporary[32], config.tau, 8);
            Fence();
            Div(temporary[8], temporary[8], temporary[32], 8);
            Div(temporary[16], temporary[16], temporary[32], 8);
            Fence();
        }
        Exp(temporary[8], temporary[8], 16);
        Fence();
        for (uint32_t field = 1; field < 4; ++field) {
            Mul(running[field * 8], running[field * 8], temporary[8], 8);
            Mul(temporary[24], tile[field * 8], temporary[16], 8);
            Fence();
            Add(running[field * 8], running[field * 8], temporary[24], 8);
            Fence();
        }
        Adds(running, temporary, 0.0f, 8);
        Fence();
    }

public:
    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR workspace, const Tiling &tiling)
    {
        config = tiling;
        features.SetGlobalBuffer((__gm__ float *)feat);
        filters.SetGlobalBuffer((__gm__ float *)weight);
        biases.SetGlobalBuffer((__gm__ float *)bias);
        statistics.SetGlobalBuffer((__gm__ float *)workspace);
        pipe.InitBuffer(inputQueue, 1, 3 * ViewSize * 4);
        pipe.InitBuffer(weightQueue, 1, 1152 * 4);
        pipe.InitBuffer(biasQueue, 1, 32);
        pipe.InitBuffer(outputQueue, 1, 64);
        pipe.InitBuffer(partialStore, KeypointsPerPass * Channels * Count * 4);
        pipe.InitBuffer(sumStore, 4 * Count * 4);
        pipe.InitBuffer(coordinateStore, 2 * Count * 4);
        pipe.InitBuffer(workStore, 8 * Count * 4);
        pipe.InitBuffer(weightStore, 1152 * 8 * 4);
        pipe.InitBuffer(reductionStore, 24 * 4);
        pipe.InitBuffer(tileStore, 32 * 4);
        pipe.InitBuffer(runningStore, 32 * 4);
        pipe.InitBuffer(broadcastStore, 64 * 4);
        pipe.InitBuffer(combineStore, 40 * 4);
        pipe.InitBuffer(packetIndexStore, 16 * 4);
    }

    __aicore__ inline void Process()
    {
        auto weights = weightQueue.AllocTensor<float>();
        auto bias = biasQueue.AllocTensor<float>();
        DataCopyPad(weights, filters, DataCopyExtParams{1, 4608, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(bias, biases, DataCopyExtParams{1, 16, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        weightQueue.EnQue(weights);
        biasQueue.EnQue(bias);
        weights = weightQueue.DeQue<float>();
        bias = biasQueue.DeQue<float>();
        Brcb(weightStore.Get<float>(), weights, 144, {1, 8});
        Fence();
        weightQueue.FreeTensor(weights);
        MakeIndices();
        uint32_t workers = config.coreCount / config.batch;
        uint32_t batch = GetBlockIdx() / workers;
        uint32_t worker = GetBlockIdx() % workers;
        uint32_t iteration = 0;
        for (uint32_t tile = worker; tile < config.tileCount; tile += workers, ++iteration) {
            Convolve(batch, tile);
            CalculateStatistics(tile, bias);
            CombineStatistics(iteration);
        }
        auto packet = outputQueue.AllocTensor<float>();
        Gather(packet, runningStore.Get<float>(), packetIndexStore.Get<uint32_t>(), 0, 16);
        outputQueue.EnQue(packet);
        packet = outputQueue.DeQue<float>();
        DataCopyPad(statistics[GetBlockIdx() * 16], packet, DataCopyExtParams{1, 64, 0, 0, 0});
        outputQueue.FreeTensor(packet);
        biasQueue.FreeTensor(bias);
    }
};

template <class Tiling>
__aicore__ inline void KeypointMerge(
    GM_ADDR statistics, GM_ADDR keypoints, GM_ADDR scores, Tiling config)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inputQueue;
    TQue<TPosition::VECOUT, 1> outputQueue;
    TBuf<TPosition::VECCALC> fieldsBuffer, indexBuffer, tempBuffer, scratchBuffer, scalarBuffer;
    pipe.InitBuffer(inputQueue, 1, config.tileCount * 64);
    pipe.InitBuffer(outputQueue, 1, 64);
    pipe.InitBuffer(fieldsBuffer, 4 * 64 * 4);
    pipe.InitBuffer(indexBuffer, 64 * 4);
    pipe.InitBuffer(tempBuffer, 3 * 64 * 4);
    pipe.InitBuffer(scratchBuffer, 256);
    pipe.InitBuffer(scalarBuffer, 96);
    uint32_t b = GetBlockIdx() / 4;
    uint32_t k = GetBlockIdx() % 4;
    GlobalTensor<float> source, points, confidence;
    source.SetGlobalBuffer((__gm__ float *)statistics + b * config.tileCount * 16);
    points.SetGlobalBuffer((__gm__ float *)keypoints + b * 8 + k * 2);
    confidence.SetGlobalBuffer((__gm__ float *)scores + b * 4 + k);
    auto packet = inputQueue.AllocTensor<float>();
    DataCopyPad(packet, source, {1, config.tileCount * 64, 0, 0, 0}, {false, 0, 0, 0});
    LocalTensor<int32_t> index = indexBuffer.Get<int32_t>();
    CreateVecIndex(index, static_cast<int32_t>(0), config.tileCount);
    PipeBarrier<PIPE_V>();
    Muls(index, index, static_cast<int32_t>(64), config.tileCount);
    inputQueue.EnQue(packet);
    packet = inputQueue.DeQue<float>();
    PipeBarrier<PIPE_V>();
    const auto offsets = index.ReinterpretCast<uint32_t>();
    auto fields = fieldsBuffer.Get<float>();
    for (uint32_t field = 0; field < 4; ++field) {
        Gather(fields[field * 64], packet, offsets, (k * 4 + field) * 4, config.tileCount);
    }
    PipeBarrier<PIPE_V>();
    auto temp = tempBuffer.Get<float>();
    auto scratch = scratchBuffer.Get<float>();
    auto scalar = scalarBuffer.Get<float>();
    ReduceMax(scalar, fields, scratch, config.tileCount);
    PipeBarrier<PIPE_ALL>();
    float maximum = scalar.GetValue(0);
    Adds(temp, fields, -maximum, config.tileCount);
    ScaleTemperature(temp, temp[128], config.tau, config.tileCount);
    Exp(temp, temp, config.tileCount);
    PipeBarrier<PIPE_V>();
    Mul(temp[64], temp, fields[64], config.tileCount);
    Mul(temp[128], temp, fields[128], config.tileCount);
    Mul(fields, temp, fields[192], config.tileCount);
    PipeBarrier<PIPE_V>();
    ReduceSum(scalar, temp[64], scratch, config.tileCount);
    PipeBarrier<PIPE_V>();
    ReduceSum(scalar[8], temp[128], scratch, config.tileCount);
    PipeBarrier<PIPE_V>();
    ReduceSum(scalar[16], fields, scratch, config.tileCount);
    PipeBarrier<PIPE_ALL>();
    const float denominator = scalar.GetValue(0);
    auto result = outputQueue.AllocTensor<float>();
    result.SetValue(0, (scalar.GetValue(8) / denominator + 0.5f) * config.stride);
    result.SetValue(1, (scalar.GetValue(16) / denominator + 0.5f) * config.stride);
    result.SetValue(8, maximum);
    PipeBarrier<PIPE_ALL>();
    outputQueue.EnQue(result);
    result = outputQueue.DeQue<float>();
    DataCopyPad(points, result, {1, 8, 0, 0, 0});
    DataCopyPad(confidence, result[8], {1, 4, 0, 0, 0});
    outputQueue.FreeTensor(result);
    inputQueue.FreeTensor(packet);
}

template <typename DT_FEAT>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints,
    GM_ADDR scores, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, config, tiling);

    bool handled;
    if (config.width == 96) {
        SparseFullImage<decltype(config), 72, 96> sparse;
        handled = sparse.Process(feat, weight, bias, keypoints, scores, config);
    } else if (config.height == 48) {
        SparseFullImage<decltype(config), 48, 64> sparse;
        handled = sparse.Process(feat, weight, bias, keypoints, scores, config);
    } else {
        SparseFullImage<decltype(config), 64, 64> sparse;
        handled = sparse.Process(feat, weight, bias, keypoints, scores, config);
    }
    if (handled) return;

    // CANN 9's generated wrapper records the allocation base separately.
    // This basic-AIV implementation uses the allocation for its own packets.
    // The hardware FFTS barrier has a separate runtime-supplied flag address.
    workspace = ::GetSysWorkSpacePtr();
    if (config.width == 64) {
        BatchedDenseStatistics<decltype(config), 64, 32> convolution;
        convolution.Init(feat, weight, bias, workspace, config);
        convolution.Process();
    } else {
        BatchedDenseStatistics<decltype(config), 96, 16> convolution;
        convolution.Init(feat, weight, bias, workspace, config);
        convolution.Process();
    }

    // All launched AIVs join before reading other cores' statistics.
    // Convolution objects leave scope so the merge can reuse their UB and events.
    AscendC::SyncAll<true>();
    if (AscendC::GetBlockIdx() < config.batch * 4) {
        auto mergeConfig = config;
        mergeConfig.tileCount = config.coreCount / config.batch;
        KeypointMerge(workspace, keypoints, scores, mergeConfig);
    }
}
