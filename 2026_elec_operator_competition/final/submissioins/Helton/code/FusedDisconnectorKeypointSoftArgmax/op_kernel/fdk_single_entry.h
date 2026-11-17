#pragma once

#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "fdk_vector_body.h"

namespace wave1 {
using Tiling = FusedDisconnectorKeypointSoftArgmaxTilingData;

// Check one real filter for each map. Copy the bias in the same DMA
// stage so the constant-map case can write directly after the check.
__aicore__ inline bool WriteConstantMap(GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints,
                                       GM_ADDR scores, int map, const Tiling& p) {
    using namespace AscendC;
    constexpr int WeightSize = fdk::Channels * fdk::Taps;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (2 * WeightSize + 24) * sizeof(float));
    auto filter = storage.Get<float>();
    auto work = filter[WeightSize];
    auto maximum = work[WeightSize];
    auto biasUb = maximum[8];
    auto xyUb = biasUb[8];
    const int k = map % fdk::KOut;
    GlobalTensor<float> wg, bg, xy, sg;
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    xy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypoints));
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scores));
    const DataCopyExtParams one{1, 4, 0, 0, 0};
    DataCopy(filter, wg[k * WeightSize], WeightSize);
    DataCopyPad(biasUb, bg[k], one, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    PipeBarrier<PIPE_ALL>();
    Abs(filter, filter, WeightSize);
    PipeBarrier<PIPE_V>();
    ReduceMax(maximum, filter, work, WeightSize, false);
    PipeBarrier<PIPE_ALL>();
    if (maximum.GetValue(0) != 0.0f) return false;
    xyUb.SetValue(0, float(p.width) * 0.5f * p.stride);
    xyUb.SetValue(1, float(p.height) * 0.5f * p.stride);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(xy[2 * map], xyUb, DataCopyExtParams{1, 8, 0, 0, 0});
    DataCopyPad(sg[map], biasUb, one);
    PipeBarrier<PIPE_ALL>();
    return true;
}

template<int Width>
__aicore__ inline void RunMap(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints,
                              GM_ADDR scores, const Tiling& p) {
    using namespace AscendC;
    const int map = GetBlockIdx();
    const int batch = map / fdk::KOut;
    const int k = map % fdk::KOut;
    fdk::BandStats stats[3];
    if constexpr (Width == 64) {
        if (p.height == 48)
            stats[0] = fdk::ConvolveGrouped<64, 48>(feat, weight, batch, k, p.prescale, p.invTau);
        else
            stats[0] = fdk::ConvolveGrouped<64, 64>(feat, weight, batch, k, p.prescale, p.invTau);
    } else {
        stats[0] = fdk::ConvolveBand<Width>(feat, weight, p.batch, p.height, batch, k, 0,
                                           p.rows, p.prescale, p.invTau);
    }
    const int bands = stats[0].constant ? 1 : p.height / p.rows;
    for (int band = 1; band < bands; ++band) {
        stats[band] = fdk::ConvolveBand<Width>(feat, weight, p.batch, p.height, batch, k, band,
                                              p.rows, p.prescale, p.invTau);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 24 * sizeof(float));
    auto factors = buffer.Get<float>();
    auto biasUb = factors[8];
    auto xyUb = factors[16];
    GlobalTensor<float> bg, xy, sg;
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    xy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypoints));
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scores));
    DataCopyPad(biasUb, bg[k], DataCopyExtParams{1, 4, 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    float maxValue = stats[0].maximum;
    float mass = stats[0].mass;
    float x = stats[0].x;
    float y = stats[0].y;
    if (bands > 1) {
        for (int band = 0; band < bands; ++band) {
            if (stats[band].maximum > maxValue) maxValue = stats[band].maximum;
            factors.SetValue(band, stats[band].maximum);
        }
        PipeBarrier<PIPE_ALL>();
        fdk::ExpShifted(factors, maxValue, p.prescale, p.invTau, bands);
        PipeBarrier<PIPE_ALL>();
        mass = x = y = 0.0f;
        for (int band = 0; band < bands; ++band) {
            const float factor = factors.GetValue(band);
            mass += factor * stats[band].mass;
            x += factor * stats[band].x;
            y += factor * stats[band].y;
        }
    }
    PipeBarrier<PIPE_ALL>();
    xyUb.SetValue(0, (x / mass + 0.5f) * p.stride);
    xyUb.SetValue(1, (y / mass + 0.5f) * p.stride);
    biasUb.SetValue(0, maxValue + biasUb.GetValue(0));
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(xy[2 * map], xyUb, DataCopyExtParams{1, 8, 0, 0, 0});
    DataCopyPad(sg[map], biasUb, DataCopyExtParams{1, 4, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

// One 32-byte aligned record per block. All blocks finish their record
// before any map leader reads it, so no input or output is used as scratch.
template<int Width, int Height, int Parts, int Group = 4>
__aicore__ inline void RunTiledMap(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints,
                                   GM_ADDR scores, GM_ADDR scratch, const Tiling& p) {
    using namespace AscendC;
    constexpr int Record = 8;
    const int block = GetBlockIdx();
    const int map = block / Parts;
    const int part = block % Parts;
    const int k = map % fdk::KOut;
    constexpr int Rows = (Height + Parts - 1) / Parts;
    const int row0 = part * Rows;
    fdk::BandStats tile;
    if constexpr (Height % Parts != 0) {
        constexpr int Tail = Height - (Parts - 1) * Rows;
        if (part == Parts - 1)
            tile = fdk::ConvolveGrouped<Width, Height, Tail, Group>(feat, weight, map / fdk::KOut, k,
                                                                  p.prescale, p.invTau, row0);
        else
            tile = fdk::ConvolveGrouped<Width, Height, Rows, Group>(feat, weight, map / fdk::KOut, k,
                                                                  p.prescale, p.invTau, row0);
    } else {
        tile = fdk::ConvolveGrouped<Width, Height, Rows, Group>(feat, weight, map / fdk::KOut, k,
                                                              p.prescale, p.invTau, row0);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (Parts + 3) * Record * sizeof(float));
    auto partials = storage.Get<float>();
    auto factors = partials[Parts * Record];
    auto biasUb = factors[Record];
    auto xyUb = biasUb[Record];
    GlobalTensor<float> stats;
    stats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scratch), p.blocks * Record);
    partials.SetValue(0, tile.maximum);
    partials.SetValue(1, tile.mass);
    partials.SetValue(2, tile.x);
    partials.SetValue(3, tile.y);
    PipeBarrier<PIPE_ALL>();
    DataCopy(stats[block * Record], partials, Record);
    SyncAll<true>();
    if (part != 0) return;

    GlobalTensor<float> bg, xy, sg;
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    xy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypoints));
    sg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scores));
    DataCopy(partials, stats[map * Parts * Record], Parts * Record);
    DataCopyPad(biasUb, bg[k], DataCopyExtParams{1, 4, 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    PipeBarrier<PIPE_ALL>();
    float maximum = partials.GetValue(0);
    for (int i = 0; i < Parts; ++i) {
        const float m = partials.GetValue(i * Record);
        if (m > maximum) maximum = m;
        factors.SetValue(i, m);
    }
    PipeBarrier<PIPE_ALL>();
    fdk::ExpShifted(factors, maximum, p.prescale, p.invTau, Parts);
    PipeBarrier<PIPE_ALL>();
    float mass = 0.0f, x = 0.0f, y = 0.0f;
    for (int i = 0; i < Parts; ++i) {
        const float f = factors.GetValue(i);
        mass += f * partials.GetValue(i * Record + 1);
        x += f * partials.GetValue(i * Record + 2);
        y += f * partials.GetValue(i * Record + 3);
    }
    xyUb.SetValue(0, (x / mass + 0.5f) * p.stride);
    xyUb.SetValue(1, (y / mass + 0.5f) * p.stride);
    biasUb.SetValue(0, biasUb.GetValue(0) + maximum);
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(xy[2 * map], xyUb, DataCopyExtParams{1, 8, 0, 0, 0});
    DataCopyPad(sg[map], biasUb, DataCopyExtParams{1, 4, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunVector(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints,
                                 GM_ADDR scores, const Tiling& p, GM_ADDR scratch) {
    if (p.blocks > p.batch * 4) {
        const int parts = p.blocks / (p.batch * fdk::KOut);
        if (p.width == 96) {
            if (parts == 4) RunTiledMap<96, 72, 4>(feat, weight, bias, keypoints, scores, scratch, p);
            else if (parts == 3 && p.batch == 3)
                RunTiledMap<96, 72, 3, 8>(feat, weight, bias, keypoints, scores, scratch, p);
            else if (parts == 3) RunTiledMap<96, 72, 3>(feat, weight, bias, keypoints, scores, scratch, p);
            else RunTiledMap<96, 72, 2>(feat, weight, bias, keypoints, scores, scratch, p);
        } else if (p.height == 48) {
            if (parts == 8) RunTiledMap<64, 48, 8, 8>(feat, weight, bias, keypoints, scores, scratch, p);
            else if (parts == 3) RunTiledMap<64, 48, 3, 8>(feat, weight, bias, keypoints, scores, scratch, p);
            else if (parts == 6) RunTiledMap<64, 48, 6, 8>(feat, weight, bias, keypoints, scores, scratch, p);
            else if (parts == 4) RunTiledMap<64, 48, 4>(feat, weight, bias, keypoints, scores, scratch, p);
            else RunTiledMap<64, 48, 2>(feat, weight, bias, keypoints, scores, scratch, p);
        } else if (parts == 4) {
            RunTiledMap<64, 64, 4>(feat, weight, bias, keypoints, scores, scratch, p);
        } else if (parts == 3) {
            RunTiledMap<64, 64, 3, 8>(feat, weight, bias, keypoints, scores, scratch, p);
        } else if (p.batch == 4) {
            RunTiledMap<64, 64, 2, 8>(feat, weight, bias, keypoints, scores, scratch, p);
        } else {
            RunTiledMap<64, 64, 2>(feat, weight, bias, keypoints, scores, scratch, p);
        }
        return;
    }
    if (p.width == 64) {
        if (p.batch == 4 && WriteConstantMap(weight, bias, keypoints, scores, AscendC::GetBlockIdx(), p)) return;
        RunMap<64>(feat, weight, bias, keypoints, scores, p);
    } else
        RunMap<96>(feat, weight, bias, keypoints, scores, p);
}
}  // namespace wave1
