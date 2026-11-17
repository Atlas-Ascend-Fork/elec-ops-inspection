#pragma once
#include "fdk_decode_common.h"

namespace fdk {
using namespace AscendC;

// Each block computes one output map. Full-width rows keep all vector
// operands aligned. Separate copies carry the horizontal shifts.
template <int Width>
struct Geo {
    static constexpr int MaxRows = Width == 64 ? 64 : 24;
    static constexpr int Plane = (MaxRows + 2) * Width + 8;  // +8 floats of slack
    static constexpr int Acc = MaxRows * Width;
};

// Copy the (possibly clipped) halo row range of one channel into the three
// shifted plane buffers of a slot.
template <int Width>
__aicore__ inline void LoadShifted(LocalTensor<float> slot, const GlobalTensor<float>& featLm,
                                   const GlobalTensor<float>& featGm,
                                   const GlobalTensor<float>& featRp, uint32_t base, int padRows,
                                   int copyBytes, int totalFloats) {
    constexpr int Plane = Geo<Width>::Plane;
    DataCopyExtParams cp{1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
    const int dst = padRows * Width;
    // Only the two ends of the tensor need bounded DMA reads.
    DataCopyExtParams shortCopy{1, static_cast<uint32_t>(copyBytes - 4), 0, 0, 0};
    if (base == 0)
        DataCopyPad(slot[dst], featGm[0], shortCopy, DataCopyPadExtParams<float>{true, 1, 0, 0.0f});
    else
        DataCopyPad(slot[dst], featLm[base], cp, noPad);
    DataCopyPad(slot[Plane + dst], featGm[base], cp, noPad);
    if (base + static_cast<uint32_t>(copyBytes / 4) == static_cast<uint32_t>(totalFloats))
        DataCopyPad(slot[2 * Plane + dst], featRp[base], shortCopy,
                    DataCopyPadExtParams<float>{true, 0, 1, 0.0f});
    else
        DataCopyPad(slot[2 * Plane + dst], featRp[base], cp, noPad);
}

// Accumulate the 9 taps of one input channel into one output heatmap.
template <int Width>
__aicore__ inline void AccumulateChannel(LocalTensor<float> slot, LocalTensor<float> accum,
                                         const LocalTensor<float>& weights, int c, uint8_t repeats) {
    constexpr int Plane = Geo<Width>::Plane;
    LocalTensor<float> shifted[3] = {slot, slot[Plane], slot[2 * Plane]};
    const int wBase = c * Taps;
    for (int dr = 0; dr < 3; ++dr) {
        for (int dc = 0; dc < 3; ++dc) {
            const int tap = wBase + dr * 3 + dc;
            LocalTensor<float> src = shifted[dc][dr * Width];
            Axpy(accum, src, weights.GetValue(tap), uint64_t(64), repeats,
                 UnaryRepeatParams{1, 1, 8, 8});
        }
    }
}

struct BandStats {
    float maximum, mass, x, y;
    bool constant;
};

template <int Width, int Acc>
__aicore__ inline BandStats ReduceBand(LocalTensor<float> accum, LocalTensor<float> planes, int n,
                                       int r0, float prescale, float invTau) {
    auto tmp = planes;
    auto xIndex = planes[Acc];
    auto yIndex = planes[2 * Acc];
    auto red = planes[3 * Acc];
    auto res = planes[4 * Acc];
    CreateVecIndex(xIndex, 0.0f, n);
    PipeBarrier<PIPE_V>();
    Muls(tmp, xIndex, 1.0f / float(Width), n);
    PipeBarrier<PIPE_V>();
    Cast(tmp.ReinterpretCast<int32_t>(), tmp, RoundMode::CAST_FLOOR, n);
    PipeBarrier<PIPE_V>();
    Cast(yIndex, tmp.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    Muls(tmp, yIndex, float(Width), n);
    Adds(yIndex, yIndex, float(r0), n);
    PipeBarrier<PIPE_V>();
    Sub(xIndex, xIndex, tmp, n);
    PipeBarrier<PIPE_V>();
    ReduceMax(res, accum, red, n, false);
    PipeBarrier<PIPE_ALL>();
    const float bandMax = res.GetValue(0);
    ExpShifted(accum, bandMax, prescale, invTau, n);
    ReduceSum(res, accum, red, n);
    PipeBarrier<PIPE_ALL>();
    const float mass = res.GetValue(0);
    Mul(tmp, accum, xIndex, n);
    ReduceSum(res, tmp, red, n);
    PipeBarrier<PIPE_ALL>();
    const float xMoment = res.GetValue(0);
    Mul(tmp, accum, yIndex, n);
    ReduceSum(res, tmp, red, n);
    PipeBarrier<PIPE_ALL>();
    const float yMoment = res.GetValue(0);
    return {bandMax, mass, xMoment, yMoment, false};
}

template <int Width>
__aicore__ inline BandStats ConvolveBand(GM_ADDR feat, GM_ADDR weight, int batches, int height,
                                        int b, int k0, int band, int rows, float prescale,
                                        float invTau) {
    constexpr int Plane = Geo<Width>::Plane;
    constexpr int Acc = Geo<Width>::Acc;
    const int r0 = band * rows;
    const int n = rows * Width;
    const auto repeats = static_cast<uint8_t>(n / 64);
    const int nsrc = (rows + 2) * Width;
    // Halo row range [first, lastRow] of the plane, clipped to the image.
    const int first = r0 > 0 ? r0 - 1 : 0;
    const int lastRow = (r0 + rows) < height ? (r0 + rows) : (height - 1);
    const int lrows = lastRow - first + 1;
    const int padRows = first - r0 + 1;  // 0 or 1
    const int copyBytes = lrows * Width * 4;
    const int planeBase = (b * Channels) * (height * Width) + first * Width;

    GlobalTensor<float> featGm, featLm, featRp, weightGm;
    featGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
    featLm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat) - 1);
    featRp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat) + 1);
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));

    TPipe pipe;
    TBuf<TPosition::VECCALC> planeBuf, maskBuf, accumBuf, weightBuf;
    pipe.InitBuffer(planeBuf, 2 * 3 * Plane * sizeof(float));
    pipe.InitBuffer(maskBuf, 2 * Plane * sizeof(float));
    pipe.InitBuffer(accumBuf, Acc * sizeof(float));
    pipe.InitBuffer(weightBuf, Channels * Taps * sizeof(float));

    auto planes = planeBuf.Get<float>();
    auto masks = maskBuf.Get<float>();
    auto accum = accumBuf.Get<float>();
    auto wLocal = weightBuf.Get<float>();
    auto maskL = masks;
    auto maskR = masks[Plane];

    constexpr int WeightSize = Channels * Taps;
    DataCopy(wLocal, weightGm[k0 * WeightSize], WeightSize);
    PipeBarrier<PIPE_ALL>();
    if (band == 0) {
        Abs(planes, wLocal, WeightSize);
        PipeBarrier<PIPE_V>();
        ReduceMax(planes[2 * WeightSize], planes, planes[WeightSize], WeightSize, false);
        PipeBarrier<PIPE_ALL>();
        // For an exactly zero filter the heatmap is the bias at every pixel.
        if (planes.GetValue(2 * WeightSize) == 0.0f)
            return {0.0f, 1.0f, (Width - 1) * 0.5f, (height - 1) * 0.5f, true};
    }

    // Column 0 of the L copy and column Width-1 of the R copy wrap around to the
    // neighbouring row; the true zero padding of stride 1 conv needs them zero.
    // Rows outside [first, lastRow] are never written by the copies below, so
    // duplicating the whole slot once keeps them zero for every channel.
    for (int s = 0; s < 2; ++s) Duplicate(planes[s * 3 * Plane], 0.0f, 3 * Plane);
    Duplicate(accum, 0.0f, Acc);
    Duplicate(maskL, 1.0f, nsrc);
    Duplicate(maskR, 1.0f, nsrc);
    PipeBarrier<PIPE_ALL>();
    for (int r = 0; r < rows + 2; ++r) {
        maskL.SetValue(r * Width, 0.0f);
        maskR.SetValue(r * Width + Width - 1, 0.0f);
    }
    PipeBarrier<PIPE_ALL>();

    // FetchEventID() only peeks the first free id, so two slots must be given
    // distinct ids explicitly (a shared id would merge sets and deadlock).
    TEventID evIn[2] = {static_cast<TEventID>(0), static_cast<TEventID>(1)};
    TEventID evOut[2] = {static_cast<TEventID>(2), static_cast<TEventID>(3)};

    // Two slot software pipeline: the copy of channel c+2 is issued while the
    // vector unit still works on channel c+1.
    for (int c = 0; c < Channels && c < 2; ++c) {
        LoadShifted<Width>(planes[c * 3 * Plane], featLm, featGm, featRp,
                           planeBase + c * height * Width, padRows, copyBytes, batches * Channels * height * Width);
        SetFlag<HardEvent::MTE2_V>(evIn[c]);
    }
    for (int c = 0; c < Channels; ++c) {
        const int s = c & 1;
        WaitFlag<HardEvent::MTE2_V>(evIn[s]);
        Mul(planes[s * 3 * Plane], planes[s * 3 * Plane], maskL, nsrc);
        Mul(planes[s * 3 * Plane + 2 * Plane], planes[s * 3 * Plane + 2 * Plane], maskR, nsrc);
        AccumulateChannel<Width>(planes[s * 3 * Plane], accum, wLocal, c, repeats);
        SetFlag<HardEvent::V_MTE2>(evOut[s]);
        if (c + 2 < Channels) {
            WaitFlag<HardEvent::V_MTE2>(evOut[s]);
            LoadShifted<Width>(planes[s * 3 * Plane], featLm, featGm, featRp,
                               planeBase + (c + 2) * height * Width, padRows, copyBytes, batches * Channels * height * Width);
            SetFlag<HardEvent::MTE2_V>(evIn[s]);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(evOut[0]);
    WaitFlag<HardEvent::V_MTE2>(evOut[1]);
    PipeBarrier<PIPE_ALL>();

    return ReduceBand<Width, Acc>(accum, planes, n, r0, prescale, invTau);
}

template <int Width, int Rows, int Acc>
__aicore__ inline BandStats ReduceRows(LocalTensor<float> heat, LocalTensor<float> scratch,
                                       int row0, float prescale, float invTau) {
    constexpr int VectorWidth = 64;
    constexpr int RowStride = Width / 8;
    constexpr int Count = Rows * Width;
    auto temporary = scratch;
    auto redScratch = scratch[Acc];
    auto result = scratch[2 * Acc];
    auto column = scratch[3 * Acc];
    auto row = column[Width];
    auto rowMass = row[64];
    auto weightedRowMass = rowMass[64];
    auto tailMass = weightedRowMass[64];
    static_assert(Rows <= 64 && (Width == 64 || Width == 96));
    CreateVecIndex(column, 0.0f, Width);
    CreateVecIndex(row, float(row0), Rows);
    ReduceMax(result, heat, redScratch, Count, false);
    PipeBarrier<PIPE_ALL>();
    const float maximum = result.GetValue(0);
    ExpShifted(heat, maximum, prescale, invTau, Count);
    const BinaryRepeatParams rowStep{1, 1, 1, RowStride, RowStride, 0};
    WholeReduceSum(rowMass, heat, VectorWidth, Rows, 1, 1, RowStride);
    Mul(temporary, heat, column, uint64_t(VectorWidth), Rows, rowStep);
    if constexpr (Width > VectorWidth) {
        // A 96-wide row spans a full 64-lane repeat and a 32-lane tail.
        constexpr int Tail = Width - VectorWidth;
        WholeReduceSum(tailMass, heat[VectorWidth], Tail, Rows, 1, 1, RowStride);
        Mul(temporary[VectorWidth], heat[VectorWidth], column[VectorWidth], uint64_t(Tail), Rows, rowStep);
        PipeBarrier<PIPE_V>();
        Add(rowMass, rowMass, tailMass, Rows);
    }
    PipeBarrier<PIPE_V>();
    Mul(weightedRowMass, rowMass, row, Rows);
    PipeBarrier<PIPE_V>();
    ReduceSum(result, rowMass, redScratch, Rows);
    PipeBarrier<PIPE_V>();
    ReduceSum(result[Width], temporary, redScratch, Count);
    PipeBarrier<PIPE_V>();
    ReduceSum(result[2 * Width], weightedRowMass, redScratch, Rows);
    PipeBarrier<PIPE_ALL>();
    return {maximum, result.GetValue(0), result.GetValue(Width), result.GetValue(2 * Width), false};
}

template<int Width, int Repeats>
__aicore__ inline void LaneColumn(LocalTensor<float> sum, LocalTensor<float> feature,
                                  LocalTensor<float> coefficients, int group, int dx) {
    const BinaryRepeatParams broadcastBlock{1, 1, 0, 8, 8, 0};
    for (int dy = 0; dy < 3; ++dy) {
        auto weights = coefficients[(dy * 3 + dx) * Channels + group * 8];
        MulAddDst(sum, feature[dy * Width * 8], weights, uint64_t(64), Repeats, broadcastBlock);
    }
}

// With eight channels per pixel a one-column shift is one aligned block.
// For each horizontal tap we clear just the column that would wrap.
template<int Height, int Rows>
__aicore__ inline BandStats ConvolveLane8(GM_ADDR feat, GM_ADDR weight, int b, int k,
                                          float prescale, float invTau, int row0) {
    constexpr int Width = 64;
    constexpr int Lanes = 8;
    constexpr int ChannelPitch = (Rows + 2) * Width;
    constexpr int Count = Rows * Width;
    constexpr int Packed = (Rows + 2) * Width * Lanes;
    constexpr int LanePlane = Count * Lanes;
    constexpr int Acc = Count + 8;
    constexpr int WeightSize = Channels * Taps;
    constexpr int Repeats = Count / Lanes;
    constexpr TEventID Event = static_cast<TEventID>(0);
    const bool top = row0 == 0;
    const bool bottom = row0 + Rows == Height;
    const int first = top ? 0 : row0 - 1;
    const int dst = top ? Width : 0;
    const int copied = (Rows + !top + !bottom) * Width;
    const DataCopyParams copyParams{Lanes, static_cast<uint16_t>(copied / 8),
                                    static_cast<uint16_t>((Height * Width - copied) / 8),
                                    static_cast<uint16_t>((ChannelPitch - copied) / 8)};
    const DataCopyParams saveEdge{Rows + 2, 1, Width - 1, 0};
    const DataCopyParams restoreEdge{Rows + 2, 1, 0, Width - 1};
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, packedBuf, laneBuf, indexBuf, weightsBuf, edgeBuf;
    pipe.InitBuffer(rawBuf, Lanes * ChannelPitch * sizeof(float));
    pipe.InitBuffer(packedBuf, (Packed + 2 * Lanes) * sizeof(float));
    pipe.InitBuffer(laneBuf, LanePlane * sizeof(float));
    pipe.InitBuffer(indexBuf, 2 * Channels * sizeof(int32_t));
    pipe.InitBuffer(weightsBuf, 2 * WeightSize * sizeof(float));
    pipe.InitBuffer(edgeBuf, (Rows + 2) * Lanes * sizeof(float));
    auto raw = rawBuf.Get<float>();
    auto packedHead = packedBuf.Get<float>();
    auto packed = packedHead[Lanes];
    auto edge = edgeBuf.Get<float>();
    auto laneSums = laneBuf.Get<float>();
    auto index = indexBuf.Get<int32_t>();
    auto weightStride = index[Channels];
    auto w = weightsBuf.Get<float>();
    auto coeff = w[WeightSize];

    GlobalTensor<float> featGm, weightGm;
    featGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));
    DataCopy(w, weightGm[k * WeightSize], WeightSize);
    const int firstInput = (b * Channels * Height + first) * Width;
    DataCopy(raw[dst], featGm[firstInput], copyParams);
    SetFlag<HardEvent::MTE2_V>(Event);

    // Convert the 3x3 weights to tap-major order. Each repeat uses a vector of
    // eight adjacent channel coefficients, repeated across spatial blocks.
    CreateVecIndex(index, int32_t(0), Channels);
    PipeBarrier<PIPE_V>();
    ShiftLeft(weightStride, index, int32_t(3), Channels);
    PipeBarrier<PIPE_V>();
    Add(index, index, weightStride, Channels);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, int32_t(2), Channels);
    WaitFlag<HardEvent::MTE2_V>(Event);
    PipeBarrier<PIPE_V>();
    for (int tap = 0; tap < Taps; ++tap)
        Gather(coeff[tap * Channels], w, index.ReinterpretCast<uint32_t>(), tap * sizeof(float), Channels);
    PipeBarrier<PIPE_V>();

    // The b32 instruction transposes sixteen input rows of eight floats.
    // Interleave its output addresses to pack two groups of eight pixels.
    uint64_t source[16], target[16];
    for (int i = 0; i < Lanes; ++i) {
        source[i] = uint64_t(raw[i * ChannelPitch].GetPhyAddr());
        source[Lanes + i] = uint64_t(raw[i * ChannelPitch + Lanes].GetPhyAddr());
        target[2 * i] = uint64_t(packed[i * Lanes].GetPhyAddr());
        target[2 * i + 1] = uint64_t(packed[(Lanes + i) * Lanes].GetPhyAddr());
    }
    const TransDataTo5HDParams packStep{false, false, ChannelPitch / 16, 16, 2};
    for (int channel = 0; channel < Lanes; ++channel) {
        if (top) Duplicate(raw[channel * ChannelPitch], 0.0f, Width);
        if (bottom) Duplicate(raw[channel * ChannelPitch + (Rows + 1) * Width], 0.0f, Width);
    }
    Duplicate(packedHead, 0.0f, Lanes);
    Duplicate(packed[Packed], 0.0f, Lanes);
    Duplicate(laneSums, 0.0f, LanePlane);
    PipeBarrier<PIPE_V>();

    for (int group = 0; group < Channels / Lanes; ++group) {
        if (group != 0) WaitFlag<HardEvent::MTE2_V>(Event);
        TransDataTo5HD<float>(target, source, packStep);
        if (group + 1 < Channels / Lanes) {
            SetFlag<HardEvent::V_MTE2>(Event);
            WaitFlag<HardEvent::V_MTE2>(Event);
            DataCopy(raw[dst], featGm[firstInput + (group + 1) * Lanes * Height * Width], copyParams);
            SetFlag<HardEvent::MTE2_V>(Event);
        }
        PipeBarrier<PIPE_V>();
        LaneColumn<Width, Repeats>(laneSums, packed, coeff, group, 1);
        PipeBarrier<PIPE_ALL>();

        auto lastColumn = packed[(Width - 1) * Lanes];
        DataCopy(edge, lastColumn, saveEdge);
        PipeBarrier<PIPE_ALL>();
        Duplicate(lastColumn, 0.0f, uint64_t(Lanes), Rows + 2, 1, Width);
        PipeBarrier<PIPE_V>();
        LaneColumn<Width, Repeats>(laneSums, packedHead, coeff, group, 0);
        PipeBarrier<PIPE_ALL>();

        DataCopy(lastColumn, edge, restoreEdge);
        Duplicate(packed, 0.0f, uint64_t(Lanes), Rows + 2, 1, Width);
        PipeBarrier<PIPE_ALL>();
        LaneColumn<Width, Repeats>(laneSums, packedHead[2 * Lanes], coeff, group, 2);
        PipeBarrier<PIPE_V>();
    }

    BlockReduceSum(raw, laneSums, Repeats, 64, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    return ReduceRows<Width, Rows, Acc>(raw, packed, row0, prescale, invTau);
}

// Accumulate the three horizontal tap groups at source coordinates. Move
// their sums to destination columns once, with zero outside each row.
template <int Width, int Height, int Rows = Height, int Group = 4>
__aicore__ inline BandStats ConvolveGrouped(GM_ADDR feat, GM_ADDR weight, int b, int k,
                                           float prescale, float invTau, int r0 = 0) {
    if constexpr (Width == 64 && Height == 48 && Rows == 8)
        return ConvolveLane8<Height, Rows>(feat, weight, b, k, prescale, invTau, r0);
    constexpr int n = Rows * Width;
    const bool PadTop = r0 == 0;
    const bool PadBottom = r0 + Rows == Height;
    const int first = PadTop ? 0 : r0 - 1;
    const int copied = (Rows + !PadTop + !PadBottom) * Width;
    const int inputOffset = PadTop ? Width : 0;
    constexpr int Plane = n + 2 * Width;
    constexpr int Acc = n + 8;
    constexpr int SlotSize = Group * Plane;
    constexpr int PlaneBufferSize = 2 * SlotSize > 5 * Acc ? 2 * SlotSize : 5 * Acc;
    constexpr int WeightSize = Channels * Taps;
    const UnaryRepeatParams step{1, 1, 8, 8};

    GlobalTensor<float> featGm, weightGm;
    featGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));
    TPipe pipe;
    TBuf<TPosition::VECCALC> planesBuf, accumBuf, weightBuf;
    pipe.InitBuffer(planesBuf, PlaneBufferSize * sizeof(float));
    pipe.InitBuffer(accumBuf, 3 * Acc * sizeof(float));
    pipe.InitBuffer(weightBuf, WeightSize * sizeof(float));
    auto planes = planesBuf.Get<float>();
    auto sums = accumBuf.Get<float>();
    auto weights = weightBuf.Get<float>();
    DataCopy(weights, weightGm[k * WeightSize], WeightSize);
    PipeBarrier<PIPE_ALL>();
    if constexpr (Rows == Height) {
        Abs(planes, weights, WeightSize);
        PipeBarrier<PIPE_V>();
        ReduceMax(planes[2 * WeightSize], planes, planes[WeightSize], WeightSize, false);
        PipeBarrier<PIPE_ALL>();
        if (planes.GetValue(2 * WeightSize) == 0.0f)
            return {0.0f, 1.0f, (Width - 1) * 0.5f, r0 + (Rows - 1) * 0.5f, true};
    }

    for (int c = 0; c < 2 * Group; ++c) {
        if (PadTop) Duplicate(planes[c * Plane], 0.0f, Width);
        if (PadBottom) Duplicate(planes[c * Plane + Width + n], 0.0f, Width);
    }
    Duplicate(sums, 0.0f, 3 * Acc);
    PipeBarrier<PIPE_ALL>();
    constexpr TEventID input[2] = {static_cast<TEventID>(0), static_cast<TEventID>(1)};
    constexpr TEventID consumed[2] = {static_cast<TEventID>(2), static_cast<TEventID>(3)};
    const DataCopyParams copies{Group, static_cast<uint16_t>(copied / 8),
                                static_cast<uint16_t>((Height * Width - copied) / 8),
                                static_cast<uint16_t>((Plane - copied) / 8)};
    for (int slot = 0; slot < 2; ++slot) {
        const int channel = b * Channels + slot * Group;
        DataCopy(planes[slot * SlotSize + inputOffset], featGm[(channel * Height + first) * Width], copies);
        SetFlag<HardEvent::MTE2_V>(input[slot]);
    }
    for (int group = 0; group < Channels / Group; ++group) {
        const int slot = group & 1;
        WaitFlag<HardEvent::MTE2_V>(input[slot]);
        for (int j = 0; j < Group; ++j) {
            const int c = group * Group + j;
            for (int dy = 0; dy < 3; ++dy) {
                auto src = planes[slot * SlotSize + j * Plane + dy * Width];
                Axpy(sums, src, weights.GetValue(c * Taps + dy * 3), uint64_t(64), n / 64, step);
                Axpy(sums[Acc], src, weights.GetValue(c * Taps + dy * 3 + 1), uint64_t(64), n / 64, step);
                Axpy(sums[2 * Acc], src, weights.GetValue(c * Taps + dy * 3 + 2), uint64_t(64), n / 64, step);
            }
        }
        SetFlag<HardEvent::V_MTE2>(consumed[slot]);
        if (group + 2 < Channels / Group) {
            WaitFlag<HardEvent::V_MTE2>(consumed[slot]);
            const int channel = b * Channels + (group + 2) * Group;
            DataCopy(planes[slot * SlotSize + inputOffset], featGm[(channel * Height + first) * Width], copies);
            SetFlag<HardEvent::MTE2_V>(input[slot]);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(consumed[0]);
    WaitFlag<HardEvent::V_MTE2>(consumed[1]);
    PipeBarrier<PIPE_ALL>();

    auto index = planes.ReinterpretCast<int32_t>();
    auto gathered = planes[Acc];
    auto center = sums[Acc];
    CreateVecIndex(index, int32_t(-1), n);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, int32_t(2), n);
    PipeBarrier<PIPE_V>();
    uint64_t edgeMask[2] = {1, 0};
    Duplicate(index, n * 4, edgeMask, Rows, 1, Width / 8);
    PipeBarrier<PIPE_V>();
    Gather(gathered, sums, index.ReinterpretCast<uint32_t>(), 0, n);
    PipeBarrier<PIPE_V>();
    Add(center, center, gathered, n);
    CreateVecIndex(index, int32_t(1), n);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, int32_t(2), n);
    PipeBarrier<PIPE_V>();
    edgeMask[0] = uint64_t(1) << ((Width - 1) % 64);
    Duplicate(index[((Width - 1) / 64) * 64], n * 4, edgeMask, Rows, 1, Width / 8);
    PipeBarrier<PIPE_V>();
    Gather(gathered, sums[2 * Acc], index.ReinterpretCast<uint32_t>(), 0, n);
    PipeBarrier<PIPE_V>();
    Add(center, center, gathered, n);
    PipeBarrier<PIPE_V>();
    if constexpr (Width == 96 && Rows == 18)
        return ReduceBand<Width, Acc>(center, planes, n, r0, prescale, invTau);
    else
        return ReduceRows<Width, Rows, Acc>(center, planes, r0, prescale, invTau);
}
}  // namespace fdk
