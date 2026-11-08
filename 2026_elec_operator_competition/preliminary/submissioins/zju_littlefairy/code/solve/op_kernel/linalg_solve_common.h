/*
 * Copyright (c) 2026 LittleFairy.
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
#include <type_traits>
#include "kernel_operator.h"
#include "linalg_solve_tiling.h"

namespace AscendC {
namespace SolveDetail {
// Retain IEEE signed infinities for zero divisors so existing numerical
// guards continue to reject invalid candidates rather than clamping them.
__aicore__ inline float CheckedReciprocal(float value) {
    if(value!=0.0f)return 1.0f/value;
    union {float value;uint32_t bits;} result;
    result.value=value;result.bits=(result.bits&0x80000000U)|0x7f800000U;
    return result.value;
}

__aicore__ inline float BoundedInputScale(uint32_t aBits,uint32_t bBits) {
        int32_t aExponent=(int32_t)((aBits>>23)&255)-127;
        int32_t bExponent=(int32_t)((bBits>>23)&255)-127;
        int32_t scaleExponent=-aExponent;
        const int32_t rhsLimit=126-bExponent;
        scaleExponent=scaleExponent<rhsLimit?scaleExponent:rhsLimit;
        scaleExponent=scaleExponent>126?126:(scaleExponent<-126?-126:scaleExponent);
        union {uint32_t bits;float value;} scale;
        scale.bits=(uint32_t)(127+scaleExponent)<<23;return scale.value;

}
__aicore__ inline void NewtonDefect64(LocalTensor<float> w,GlobalTensor<float> dst,uint32_t lane,uint32_t teams) {
    for(uint32_t row=lane*8;row<64;row+=teams*8) {
        DataCopy(w,dst[row*64],512);PipeBarrier<PIPE_ALL>();Muls(w,w,-1.0f,512);PipeBarrier<PIPE_ALL>();
        for(uint32_t j=0;j<8;++j){uint32_t off=j*64+row+j;w.SetValue(off,2.0f+w.GetValue(off));}
        PipeBarrier<PIPE_ALL>();DataCopy(dst[row*64],w,512);PipeBarrier<PIPE_ALL>();
    }
}

constexpr uint32_t kRowsPerRegion = 512;
constexpr uint32_t kScratchRegions = 32;
constexpr uint32_t kMaxPaddedRows = 512;
// Columns moved by one DataCopy pair. Consecutive columns are contiguous in the
// workspace, so a panel is a single copy; the per-column work stays identical.
constexpr uint32_t kPanel = 8;
// Above this order the strided input copy is cheaper than n scalar global reads.
constexpr uint32_t kInputDmaThreshold = 224;
static_assert(kMaxPaddedRows % 8 == 0, "padded order must keep 8-float alignment");
static_assert(kRowsPerRegion >= kMaxPaddedRows, "region cannot hold a padded row");
static_assert(kPanel >= 2, "panel batching needs at least two columns");
static_assert(2*kPanel*kMaxPaddedRows*sizeof(float) <= 64*1024,
        "panel buffers must stay inside the UB budget");
// Region map (all regions hold kRowsPerRegion floats; rows <= kMaxPaddedRows).
//   0 qh   1 ql   2 ah   3 al   4 dh   5 dl   6 ph   7 pl
//   8 ah2  9 al2 10 dh2 11 dl2 12 ph2 13 pl2
//  14 xh  15 xl  16..23 s[0..7]  24..29 s2[0..5]  30 q1  31 q2
static_assert(kScratchRegions == 32, "region map above must match the allocation in Run");

// A number is hi+lo. QR and back substitution both run on the device.
struct SolvePair {float hi,lo;};

// Common allocation and lane metadata for both orthogonal vector schedules.
class OrthogonalVectorState {
public:
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    LocalTensor<float> w;
    uint32_t lane,teams,batch;
    float inputScale=1.0f;
    __aicore__ inline void InitializeState(uint32_t batches) {
        if(batches==0)return;
        uint32_t id=GetBlockIdx();batch=id%batches;lane=id/batches;
        teams=(2*GetBlockNum()+batches-1-batch)/batches;
        pipe.InitBuffer(buf,160*1024);w=buf.Get<float>();
    }
};

} // namespace SolveDetail
} // namespace AscendC
