#pragma once

#include <cmath>
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"

namespace wave1 {
using Tiling = FusedDisconnectorKeypointSoftArgmaxTilingData;

inline bool MakePlan(Tiling& plan, int b, int h, int w, float tau, int stride,
                     int aiv) {
    if (b < 1 || b > 4 || !(((h == 48 || h == 64) && w == 64) || (h == 72 && w == 96)) ||
        !std::isfinite(tau) || tau <= 0 || stride != 4 || aiv <= 0) return false;
    float prescale = 1.0f;
    if (tau < std::ldexp(1.0f, -120)) prescale = std::ldexp(1.0f, 120);
    if (tau > std::ldexp(1.0f, 120)) prescale = 1.0f / 256.0f;
    int maps = 4 * b;
    int parts = 1;
    // Wide spatial workloads and one-batch maps pay back the sync cost.
    // The smaller geometry uses at most two bands for multiple batches.
    if (w == 64 && (b == 1 || h == 64)) {
        if (b <= 2 && maps * 4 <= aiv) parts = 4;
        else if (maps * 2 <= aiv) parts = 2;
    }
    if (w == 64 && (b == 2 || (h == 48 && b == 3)) && maps * 2 <= aiv) parts = 2;
    if (w == 64 && h == 48 && b == 1 && maps * 6 <= aiv) parts = 6;
    if (w == 64 && h == 64 && b == 2 && maps * 3 <= aiv) parts = 3;
    if (w == 96 && b <= 3) {
        if (b == 1 && maps * 4 <= aiv) parts = 4;
        else if (maps * 3 <= aiv) parts = 3;
        else if (maps * 2 <= aiv) parts = 2;
    }
    if (w == 96 && b == 4 && maps * 2 <= aiv) parts = 2;
    const int rows = w == 64 || parts > 1 ? (h + parts - 1) / parts : 24;
    plan = {b, h, w, rows, maps * parts, prescale, 1.0f / (tau * prescale), float(stride)};
    return true;
}

}  // namespace wave1
