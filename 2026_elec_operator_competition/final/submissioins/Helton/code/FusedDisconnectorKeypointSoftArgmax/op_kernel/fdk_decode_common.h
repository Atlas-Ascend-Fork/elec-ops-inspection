#pragma once
#include "kernel_operator.h"

namespace fdk {
using namespace AscendC;

constexpr int Channels = 32;
constexpr int Taps = 9;
constexpr int KOut = 4;

__aicore__ inline void ExpShifted(LocalTensor<float> values, float maxValue, float prescale,
                                  float invTau, int count) {
    Adds(values, values, -maxValue, count);
    PipeBarrier<PIPE_V>();
    if (prescale != 1.0f) {
        Muls(values, values, prescale, count);
        PipeBarrier<PIPE_V>();
    }
    Muls(values, values, invTau, count);
    PipeBarrier<PIPE_V>();
    Exp(values, values, count);
    PipeBarrier<PIPE_V>();
}

}  // namespace fdk
