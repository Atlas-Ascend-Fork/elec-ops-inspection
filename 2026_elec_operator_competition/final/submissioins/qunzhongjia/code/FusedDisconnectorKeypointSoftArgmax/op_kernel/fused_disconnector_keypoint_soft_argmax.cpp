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

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"
#include "kernel_case5_single_core.h"
namespace fdksa {
using namespace AscendC;
constexpr uint32_t FIXED_CIN = 32U;
constexpr uint32_t FIXED_K = 4U;
constexpr uint32_t WEIGHT_FLOATS = 4U * 32U * 3U * 3U;
constexpr uint32_t WS_K_STRIDE = 8U;
constexpr uint32_t WS_FAST_STRIDE = 32U;
constexpr uint32_t WS_STRIDE = 64U;
constexpr float NEG_INF_FP32 = -3.402823466e+38f;
template <HardEvent E>
__aicore__ inline void Fence()
{
SetFlag<E>(EVENT_ID0);
WaitFlag<E>(EVENT_ID0);
}
__aicore__ inline DataCopyExtParams MakeCopyExt(uint32_t blockCount,
    uint32_t blockLen,
    uint32_t srcStride = 0U,
    uint32_t dstStride = 0U)
{
DataCopyExtParams p{};
p.blockCount = static_cast<uint16_t>(blockCount);
p.blockLen = blockLen;
p.srcStride = srcStride;
p.dstStride = dstStride;
return p;
}
__aicore__ inline DataCopyPadExtParams<float> MakeNoPad()
{
DataCopyPadExtParams<float> p{};
p.isPad = false;
p.leftPadding = 0;
p.rightPadding = 0;
p.paddingValue = 0.0f;
return p;
}
__aicore__ inline uint32_t AlignUp8(uint32_t x)
{
return (x + 7U) & ~7U;
}
__aicore__ inline float U32ToF32(uint32_t x)
{
return static_cast<float>(static_cast<int32_t>(x));
}
__aicore__ inline float WVal(__gm__ const float* wPtr, uint32_t idx)
{
return static_cast<float>(wPtr[idx]);
}
template <uint32_t STATIC_B, uint32_t STATIC_H, uint32_t STATIC_W, uint32_t STATIC_S>
class KernelExactStrike {
public:
static constexpr uint32_t TOTAL_HM = STATIC_B * FIXED_K;
static constexpr uint32_t USED_CORES = TOTAL_HM * STATIC_S;
static constexpr uint32_t STATIC_HLEN = STATIC_H / STATIC_S;
static constexpr uint32_t STATIC_HIN = STATIC_HLEN + 2U;
static constexpr uint32_t STRIPE_ELEMS = STATIC_HLEN * STATIC_W;
static constexpr uint32_t IN_CH_STRIDE = STATIC_HIN * STATIC_W;
static constexpr uint32_t HW_STRIDE = STATIC_H * STATIC_W;
static constexpr uint32_t CH_PER_BLOCK = (STRIPE_ELEMS <= 2048U) ? 4U : 2U;
static constexpr uint32_t NUM_BLOCKS = FIXED_CIN / CH_PER_BLOCK;
static constexpr uint32_t IN_BLOCK_STRIDE = CH_PER_BLOCK * IN_CH_STRIDE;
static constexpr bool REUSE_INBUF_POST_CONV = (STRIPE_ELEMS > 2304U);
static constexpr uint32_t PLANE_STRIDE = STRIPE_ELEMS + 8U;
static constexpr uint32_t OFFSET_STRIDE = STRIPE_ELEMS + 8U;
static constexpr uint32_t CONV_PLANES = 9U;
static constexpr uint8_t CONV_R = static_cast<uint8_t>(STRIPE_ELEMS >> 6U);
static constexpr uint32_t WS_SLOT_FLOATS = 16U;
static constexpr uint32_t WORK_FLOATS = 768U;
static constexpr uint32_t PRELOAD_BYTES =
    (FIXED_CIN * IN_CH_STRIDE + CONV_PLANES * PLANE_STRIDE +
     (REUSE_INBUF_POST_CONV ? 0U : 2U * OFFSET_STRIDE) + WORK_FLOATS) * sizeof(float);
static constexpr bool PRELOAD_ALL = PRELOAD_BYTES <= 192U * 1024U;
static constexpr uint32_t INPUT_FLOATS = PRELOAD_ALL ?
    FIXED_CIN * IN_CH_STRIDE : 2U * IN_BLOCK_STRIDE;
static_assert(448U + 288U <= WORK_FLOATS, "Weight cache exceeds work buffer");
static_assert((INPUT_FLOATS + CONV_PLANES * PLANE_STRIDE +
    (REUSE_INBUF_POST_CONV ? 0U : 2U * OFFSET_STRIDE) + WORK_FLOATS) * sizeof(float)
    <= 192U * 1024U, "Kernel exceeds 910B4 UB capacity");
__aicore__ inline void AccumulateChannel9Tap(const LocalTensor<float>& dst9,
const LocalTensor<float>& curCh,
const LocalTensor<float>& weights,
uint32_t c)
{
const uint32_t wBase = c * 9U;
const UnaryRepeatParams rep{1, 1, 8, 8};
#pragma unroll
for (uint32_t dh = 0; dh < 3U; ++dh) {
LocalTensor<float> srcRow = curCh[dh * STATIC_W];
const uint32_t dOff = dh * 3U;
const uint32_t dhPlaneBase = dh * 3U * PLANE_STRIDE;
Axpy<float, float, false>(
    dst9[dhPlaneBase + 0U * PLANE_STRIDE], srcRow, weights.GetValue(wBase + dOff + 0U), MASK_PLACEHOLDER, CONV_R, rep);
Axpy<float, float, false>(
    dst9[dhPlaneBase + 1U * PLANE_STRIDE], srcRow, weights.GetValue(wBase + dOff + 1U), MASK_PLACEHOLDER, CONV_R, rep);
Axpy<float, float, false>(
    dst9[dhPlaneBase + 2U * PLANE_STRIDE], srcRow, weights.GetValue(wBase + dOff + 2U), MASK_PLACEHOLDER, CONV_R, rep);
}
}
__aicore__ inline void Run(GM_ADDR featAddr,
GM_ADDR weightAddr,
GM_ADDR biasAddr,
GM_ADDR keypointsAddr,
GM_ADDR scoresAddr,
GM_ADDR tilingAddr,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td)
{
const uint32_t blockIdx = GetBlockIdx();
if (blockIdx >= USED_CORES) {
return;
}
const uint32_t hm = blockIdx / STATIC_S;
const uint32_t sIdx = blockIdx % STATIC_S;
const uint32_t bIdx = hm >> 2U;
const uint32_t kIdx = hm & 3U;
const uint32_t v0 = sIdx * STATIC_HLEN;
const uint32_t v1 = v0 + STATIC_HLEN;
__gm__ const float* biasPtr = reinterpret_cast<__gm__ const float*>(biasAddr);
wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightAddr), WEIGHT_FLOATS);
featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(featAddr),
static_cast<uint64_t>(STATIC_B) * FIXED_CIN * HW_STRIDE);
kpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypointsAddr), static_cast<uint64_t>(STATIC_B) * 8U);
scGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scoresAddr), static_cast<uint64_t>(STATIC_B) * 4U);
syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tilingAddr) + 16U, 28U * WS_SLOT_FLOATS);
pipe_.InitBuffer(inBuf_, INPUT_FLOATS * sizeof(float));
if constexpr (!REUSE_INBUF_POST_CONV) {
pipe_.InitBuffer(offsetBuf_, 2U * OFFSET_STRIDE * sizeof(uint32_t));
}
pipe_.InitBuffer(convBuf_, CONV_PLANES * PLANE_STRIDE * sizeof(float));
pipe_.InitBuffer(workBuf_, WORK_FLOATS * sizeof(float));
LocalTensor<float> inAll = inBuf_.Get<float>();
LocalTensor<float> inBlock0 = inAll[0];
LocalTensor<float> inBlock1 = inAll[IN_BLOCK_STRIDE];
LocalTensor<float> convAll = convBuf_.Get<float>();
LocalTensor<float> xConv9 = convAll[0];
LocalTensor<float> xConvL = convAll[0U * PLANE_STRIDE];
LocalTensor<float> xConvC = convAll[1U * PLANE_STRIDE];
LocalTensor<float> xConvR = convAll[2U * PLANE_STRIDE];
LocalTensor<float> xTmp = convAll[3U * PLANE_STRIDE];
LocalTensor<float> workAll = workBuf_.Get<float>();
LocalTensor<float> uVec = workAll[0U];
LocalTensor<float> redOut = workAll[128U];
LocalTensor<float> wsLocal = workAll[256U];
LocalTensor<float> mergeLocal = workAll[320U];
LocalTensor<float> weights = workAll[448U];
DataCopy(weights, wGm_[kIdx * 288U], 288U);
Fence<HardEvent::MTE2_V>();
Abs(inBlock1, weights, 288U);
PipeBarrier<PIPE_V>();
ReduceMax<float>(redOut[0U], inBlock1, xConv9, 288U, false);
Fence<HardEvent::V_S>();
if (redOut.GetValue(0U) == 0.0f) {
if (sIdx == 0U) {
wsLocal.SetValue(0U, 0.5f * U32ToF32(STATIC_W) * td->strideFloat);
wsLocal.SetValue(1U, 0.5f * U32ToF32(STATIC_H) * td->strideFloat);
wsLocal.SetValue(8U, WVal(biasPtr, kIdx));
Fence<HardEvent::S_MTE3>();
DataCopyPad(kpGm_[static_cast<uint64_t>(hm) * 2U], wsLocal[0U], MakeCopyExt(1U, 2U * sizeof(float)));
DataCopyPad(scGm_[static_cast<uint64_t>(hm)], wsLocal[8U], MakeCopyExt(1U, sizeof(float)));
}
return;
}
Fence<HardEvent::V_MTE2>();
const uint32_t gmV0 = (v0 > 0U) ? (v0 - 1U) : 0U;
const uint32_t gmV1 = (v1 < STATIC_H) ? (v1 + 1U) : STATIC_H;
const uint32_t ubRowOff = ((v0 == 0U) ? 1U : 0U) * STATIC_W;
const uint32_t gmCopyElems = (gmV1 - gmV0) * STATIC_W;
const uint64_t batchFeatBase =
static_cast<uint64_t>(bIdx) * FIXED_CIN * HW_STRIDE + static_cast<uint64_t>(gmV0) * STATIC_W;
DataCopyParams mte2Params;
mte2Params.blockCount = static_cast<uint16_t>(PRELOAD_ALL ? FIXED_CIN : CH_PER_BLOCK);
mte2Params.blockLen = static_cast<uint16_t>(gmCopyElems >> 3U);
mte2Params.srcStride = static_cast<uint16_t>((HW_STRIDE - gmCopyElems) >> 3U);
mte2Params.dstStride = static_cast<uint16_t>((IN_CH_STRIDE - gmCopyElems) >> 3U);
DataCopy(inBlock0[ubRowOff], featGm_[batchFeatBase], mte2Params);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
if constexpr (!PRELOAD_ALL) {
DataCopy(inBlock1[ubRowOff],
    featGm_[batchFeatBase + static_cast<uint64_t>(CH_PER_BLOCK) * HW_STRIDE],
    mte2Params);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
}
if (sIdx == 0U) {
#pragma unroll
for (uint32_t dc = 0; dc < (PRELOAD_ALL ? FIXED_CIN : 2U * CH_PER_BLOCK); ++dc) {
Duplicate(inAll[dc * IN_CH_STRIDE], 0.0f, STATIC_W);
}
}
if (sIdx == STATIC_S - 1U) {
#pragma unroll
for (uint32_t dc = 0; dc < (PRELOAD_ALL ? FIXED_CIN : 2U * CH_PER_BLOCK); ++dc) {
Duplicate(inAll[dc * IN_CH_STRIDE + (STATIC_HLEN + 1U) * STATIC_W], 0.0f, STATIC_W);
}
}
Duplicate(xConv9, 0.0f, CONV_PLANES * PLANE_STRIDE);
ArithProgression<float>(uVec, 0.0f, 1.0f, static_cast<int32_t>(STATIC_W));
if constexpr (!REUSE_INBUF_POST_CONV) {
LocalTensor<uint32_t> offAll = offsetBuf_.Get<uint32_t>();
LocalTensor<uint32_t> offsetLeft = offAll[0];
LocalTensor<uint32_t> offsetRight = offAll[OFFSET_STRIDE];
ArithProgression<int32_t>(offsetLeft.ReinterpretCast<int32_t>(),
static_cast<int32_t>(-4),
static_cast<int32_t>(4),
static_cast<int32_t>(STRIPE_ELEMS));
ArithProgression<int32_t>(offsetRight.ReinterpretCast<int32_t>(),
static_cast<int32_t>(4),
static_cast<int32_t>(4),
static_cast<int32_t>(STRIPE_ELEMS));
SetFlag<HardEvent::V_S>(EVENT_ID0);
}
PipeBarrier<PIPE_V>();
SetMaskNorm();
SetVectorMask<float, MaskMode::NORMAL>(0xffffffffffffffffULL, 0xffffffffffffffffULL);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
if constexpr (!PRELOAD_ALL) {
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
}
for (uint32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
const event_t evCur = ((blk & 1U) == 0U) ? EVENT_ID0 : EVENT_ID1;
LocalTensor<float> curBlock = PRELOAD_ALL ? inAll[blk * IN_BLOCK_STRIDE] :
    (((blk & 1U) == 0U) ? inBlock0 : inBlock1);
if constexpr (!PRELOAD_ALL) {
if (blk >= 2U) {
WaitFlag<HardEvent::MTE2_V>(evCur);
}
}
#pragma unroll
for (uint32_t dc = 0; dc < CH_PER_BLOCK; ++dc) {
AccumulateChannel9Tap(xConv9, curBlock[dc * IN_CH_STRIDE], weights, blk * CH_PER_BLOCK + dc);
}
if constexpr (!PRELOAD_ALL) {
if (blk + 2U < NUM_BLOCKS) {
SetFlag<HardEvent::V_MTE2>(evCur);
WaitFlag<HardEvent::V_MTE2>(evCur);
DataCopy(curBlock[ubRowOff],
    featGm_[batchFeatBase + static_cast<uint64_t>((blk + 2U) * CH_PER_BLOCK) * HW_STRIDE],
    mte2Params);
SetFlag<HardEvent::MTE2_V>(evCur);
}
}
}
ResetMask();
PipeBarrier<PIPE_V>();
Add(xConv9[0], xConv9[0], xConv9[3U * PLANE_STRIDE], 3U * PLANE_STRIDE);
PipeBarrier<PIPE_V>();
Add(xConv9[0], xConv9[0], xConv9[6U * PLANE_STRIDE], 3U * PLANE_STRIDE);
LocalTensor<float> vGrid = convAll[5U * PLANE_STRIDE];
LocalTensor<float> redWork = convAll[6U * PLANE_STRIDE];
#pragma unroll
for (uint32_t r = 1U; r < STATIC_HLEN; ++r) {
Duplicate(vGrid[r * STATIC_W], U32ToF32(r), STATIC_W);
}
PipeBarrier<PIPE_V>();
LocalTensor<uint32_t> offsetLeft;
LocalTensor<uint32_t> offsetRight;
if constexpr (!REUSE_INBUF_POST_CONV) {
LocalTensor<uint32_t> offAll = offsetBuf_.Get<uint32_t>();
offsetLeft = offAll[0];
offsetRight = offAll[OFFSET_STRIDE];
WaitFlag<HardEvent::V_S>(EVENT_ID0);
constexpr uint32_t zeroByteOff = STRIPE_ELEMS * static_cast<uint32_t>(sizeof(float));
#pragma unroll
for (uint32_t r = 0; r < STATIC_HLEN; ++r) {
offsetLeft.SetValue(r * STATIC_W, zeroByteOff);
offsetRight.SetValue(r * STATIC_W + STATIC_W - 1U, zeroByteOff);
}
Fence<HardEvent::S_V>();
} else {
offsetLeft = inAll[0].ReinterpretCast<uint32_t>();
offsetRight = inAll[OFFSET_STRIDE].ReinterpretCast<uint32_t>();
ArithProgression<int32_t>(offsetLeft.ReinterpretCast<int32_t>(),
static_cast<int32_t>(-4),
static_cast<int32_t>(4),
static_cast<int32_t>(STRIPE_ELEMS));
ArithProgression<int32_t>(offsetRight.ReinterpretCast<int32_t>(),
static_cast<int32_t>(4),
static_cast<int32_t>(4),
static_cast<int32_t>(STRIPE_ELEMS));
Fence<HardEvent::V_S>();
constexpr uint32_t zeroByteOff = STRIPE_ELEMS * static_cast<uint32_t>(sizeof(float));
#pragma unroll
for (uint32_t r = 0; r < STATIC_HLEN; ++r) {
offsetLeft.SetValue(r * STATIC_W, zeroByteOff);
offsetRight.SetValue(r * STATIC_W + STATIC_W - 1U, zeroByteOff);
}
Fence<HardEvent::S_V>();
}
Gather(xTmp[0U * PLANE_STRIDE], xConvL, offsetLeft, 0U, STRIPE_ELEMS);
Gather(xTmp[1U * PLANE_STRIDE], xConvR, offsetRight, 0U, STRIPE_ELEMS);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, xTmp[0U * PLANE_STRIDE], STRIPE_ELEMS);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, xTmp[1U * PLANE_STRIDE], STRIPE_ELEMS);
PipeBarrier<PIPE_V>();
LocalTensor<float> zBuf = xTmp[0U * PLANE_STRIDE];
LocalTensor<float> eBuf = xTmp[1U * PLANE_STRIDE];
LocalTensor<float> uProd = xConvL[0U];
ReduceMax<float>(redOut[0U], xConvC, redWork, STRIPE_ELEMS, false);
Fence<HardEvent::V_S>();
const float sRaw = redOut.GetValue(0U);
Fence<HardEvent::S_V>();
Adds(zBuf, xConvC, -sRaw, STRIPE_ELEMS);
PipeBarrier<PIPE_V>();
Muls(zBuf, zBuf, td->invTau, STRIPE_ELEMS);
PipeBarrier<PIPE_V>();
Exp(eBuf, zBuf, STRIPE_ELEMS);
PipeBarrier<PIPE_V>();
Mul(zBuf[STATIC_W], eBuf[STATIC_W], vGrid[STATIC_W], (STATIC_HLEN - 1U) * STATIC_W);
uint32_t remRows = STATIC_HLEN;
if constexpr (STATIC_HLEN > 32U) {
constexpr uint32_t fRows = STATIC_HLEN - 32U;
Add(eBuf[0U], eBuf[0U], eBuf[32U * STATIC_W], fRows * STATIC_W);
PipeBarrier<PIPE_V>();
remRows = 32U;
}
if constexpr (STATIC_HLEN > 16U) {
const uint32_t fRows = remRows - 16U;
Add(eBuf[0U], eBuf[0U], eBuf[16U * STATIC_W], fRows * STATIC_W);
PipeBarrier<PIPE_V>();
remRows = 16U;
}
if constexpr (STATIC_HLEN > 8U) {
const uint32_t fRows = remRows - 8U;
Add(eBuf[0U], eBuf[0U], eBuf[8U * STATIC_W], fRows * STATIC_W);
PipeBarrier<PIPE_V>();
remRows = 8U;
}
if constexpr (STATIC_HLEN > 4U) {
const uint32_t fRows = remRows - 4U;
Add(eBuf[0U], eBuf[0U], eBuf[4U * STATIC_W], fRows * STATIC_W);
PipeBarrier<PIPE_V>();
remRows = 4U;
}
if constexpr (STATIC_HLEN > 2U) {
const uint32_t fRows = remRows - 2U;
Add(eBuf[0U], eBuf[0U], eBuf[2U * STATIC_W], fRows * STATIC_W);
PipeBarrier<PIPE_V>();
remRows = 2U;
}
if constexpr (STATIC_HLEN > 1U) {
Add(eBuf[0U], eBuf[0U], eBuf[1U * STATIC_W], STATIC_W);
PipeBarrier<PIPE_V>();
}
ReduceSum<float>(redOut[24U], zBuf[STATIC_W], redWork[0U], (STATIC_HLEN - 1U) * STATIC_W);
ReduceSum<float>(redOut[8U], eBuf[0U], redWork[256U], STATIC_W);
Mul(uProd, eBuf[0U], uVec, STATIC_W);
PipeBarrier<PIPE_V>();
ReduceSum<float>(redOut[16U], uProd, redWork[512U], STATIC_W);
Fence<HardEvent::V_S>();
const float sumLoc = redOut.GetValue(8U);
const float uLoc = redOut.GetValue(16U);
const float vRel = redOut.GetValue(24U);
const float vLoc = U32ToF32(v0) * sumLoc + vRel;
if (sIdx < STATIC_S - 1U) {
const uint32_t slotOff = (hm * (STATIC_S - 1U) + sIdx) * WS_SLOT_FLOATS;
__gm__ volatile float* vSlot = reinterpret_cast<__gm__ volatile float*>(tilingAddr) + 16U + slotOff;
vSlot[0] = sRaw;
vSlot[1] = sumLoc;
vSlot[2] = uLoc;
vSlot[3] = vLoc;
DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(syncGm_[slotOff]);
vSlot[4] = 1.0f;
DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(syncGm_[slotOff]);
return;
}
mergeLocal.SetValue((STATIC_S - 1U) * 4U + 0U, sRaw);
mergeLocal.SetValue((STATIC_S - 1U) * 4U + 1U, sumLoc);
mergeLocal.SetValue((STATIC_S - 1U) * 4U + 2U, uLoc);
mergeLocal.SetValue((STATIC_S - 1U) * 4U + 3U, vLoc);
float sGlob = sRaw;
const uint32_t prodBase = hm * (STATIC_S - 1U);
#pragma unroll
for (uint32_t s = 0U; s < STATIC_S - 1U; ++s) {
const uint32_t slotOff = (prodBase + s) * WS_SLOT_FLOATS;
__gm__ volatile float* vSlot = reinterpret_cast<__gm__ volatile float*>(tilingAddr) + 16U + slotOff;
while (true) {
DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(syncGm_[slotOff]);
if (vSlot[4] == 1.0f) {
break;
}
}
const float sr = vSlot[0];
const float sumS = vSlot[1];
const float uS = vSlot[2];
const float vS = vSlot[3];
vSlot[4] = 0.0f;
DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(syncGm_[slotOff]);
if (sr > sGlob) {
sGlob = sr;
}
mergeLocal.SetValue(s * 4U + 0U, sr);
mergeLocal.SetValue(s * 4U + 1U, sumS);
mergeLocal.SetValue(s * 4U + 2U, uS);
mergeLocal.SetValue(s * 4U + 3U, vS);
}
#pragma unroll
for (uint32_t s = 0U; s < STATIC_S; ++s) {
zBuf.SetValue(s, mergeLocal.GetValue(s * 4U + 0U) - sGlob);
}
for (uint32_t s = STATIC_S; s < 16U; ++s) {
zBuf.SetValue(s, 0.0f);
}
Fence<HardEvent::S_V>();
Muls(zBuf, zBuf, td->invTau, 16U);
PipeBarrier<PIPE_V>();
Exp(eBuf, zBuf, 16U);
Fence<HardEvent::V_S>();
float sumTot = 0.0f;
float uTot = 0.0f;
float vTot = 0.0f;
#pragma unroll
for (uint32_t s = 0U; s < STATIC_S; ++s) {
const float alpha = eBuf.GetValue(s);
const uint32_t base = s * 4U;
sumTot += alpha * mergeLocal.GetValue(base + 1U);
uTot += alpha * mergeLocal.GetValue(base + 2U);
vTot += alpha * mergeLocal.GetValue(base + 3U);
}
const float invS = 1.0f / sumTot;
wsLocal.SetValue(0U, (uTot * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(1U, (vTot * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(8U, sGlob + WVal(biasPtr, kIdx));
Fence<HardEvent::S_MTE3>();
DataCopyPad(kpGm_[static_cast<uint64_t>(hm) * 2U], wsLocal[0U], MakeCopyExt(1U, 2U * sizeof(float)));
DataCopyPad(scGm_[static_cast<uint64_t>(hm)], wsLocal[8U], MakeCopyExt(1U, sizeof(float)));
}
private:
TPipe pipe_;
GlobalTensor<float> wGm_;
GlobalTensor<float> featGm_;
GlobalTensor<float> kpGm_;
GlobalTensor<float> scGm_;
GlobalTensor<float> syncGm_;
TBuf<TPosition::VECCALC> inBuf_;
TBuf<TPosition::VECCALC> offsetBuf_;
TBuf<TPosition::VECCALC> convBuf_;
TBuf<TPosition::VECCALC> workBuf_;
};
template <uint32_t MAX_W_ALIGNED, uint32_t MAX_HLEN, bool IS_ALIGNED_W>
class KernelFastGeneralized {
public:
static constexpr uint32_t MAX_HIN = MAX_HLEN + 2U;
static constexpr bool REUSE_INBUF_POST_CONV = (MAX_W_ALIGNED * MAX_HLEN > 2048U);
static constexpr uint32_t MAX_IN_FLOATS =
(MAX_W_ALIGNED <= 64U) ? (2U * 2U * 50U * 64U)
: ((MAX_W_ALIGNED <= 96U) ? (2U * 4U * 22U * 96U) : (2U * 2U * MAX_HIN * MAX_W_ALIGNED));
static constexpr uint32_t PLANE_STRIDE =
(!IS_ALIGNED_W) ? (((MAX_HIN * MAX_W_ALIGNED + 63U) & ~63U) + 8U) : (MAX_HLEN * MAX_W_ALIGNED + 8U);
static constexpr uint32_t OFFSET_STRIDE = MAX_HLEN * MAX_W_ALIGNED + 8U;
__aicore__ inline void AccumulateChannel9(const LocalTensor<float>& dst9,
const LocalTensor<float>& curCh,
__gm__ const float* kWPtr,
uint32_t c,
uint32_t wAligned,
uint32_t stripeElems)
{
const uint32_t wBase = c * 9U;
if ((stripeElems & 63U) == 0U) {
const uint8_t rCnt = static_cast<uint8_t>(stripeElems >> 6U);
const UnaryRepeatParams rep{1, 1, 8, 8};
#pragma unroll
for (uint32_t dh = 0; dh < 3U; ++dh) {
LocalTensor<float> srcRow = curCh[dh * wAligned];
const uint32_t dOff = dh * 3U;
const uint32_t dhPlaneBase = dh * 3U * PLANE_STRIDE;
Axpy<float, float, false>(
dst9[dhPlaneBase + 0U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 0U), MASK_PLACEHOLDER, rCnt, rep);
Axpy<float, float, false>(
dst9[dhPlaneBase + 1U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 1U), MASK_PLACEHOLDER, rCnt, rep);
Axpy<float, float, false>(
dst9[dhPlaneBase + 2U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 2U), MASK_PLACEHOLDER, rCnt, rep);
}
} else {
#pragma unroll
for (uint32_t dh = 0; dh < 3U; ++dh) {
LocalTensor<float> srcRow = curCh[dh * wAligned];
const uint32_t dOff = dh * 3U;
const uint32_t dhPlaneBase = dh * 3U * PLANE_STRIDE;
Axpy(dst9[dhPlaneBase + 0U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 0U), stripeElems);
Axpy(dst9[dhPlaneBase + 1U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 1U), stripeElems);
Axpy(dst9[dhPlaneBase + 2U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 2U), stripeElems);
}
}
}
__aicore__ inline void AccumulateChannel9Raw1D(const LocalTensor<float>& dst9,
const LocalTensor<float>& rawCh,
__gm__ const float* kWPtr,
uint32_t c,
uint8_t rawRCnt)
{
const uint32_t wBase = c * 9U;
const UnaryRepeatParams rep{1, 1, 8, 8};
#pragma unroll
for (uint32_t p = 0; p < 9U; ++p) {
Axpy<float, float, false>(
dst9[p * PLANE_STRIDE], rawCh, WVal(kWPtr, wBase + p), MASK_PLACEHOLDER, rawRCnt, rep);
}
}
__aicore__ inline void Run(GM_ADDR featAddr,
GM_ADDR weightAddr,
GM_ADDR biasAddr,
GM_ADDR keypointsAddr,
GM_ADDR scoresAddr,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td)
{
const uint32_t blockIdx = GetBlockIdx();
const uint32_t usedCores = td->usedCores;
if (blockIdx >= usedCores) {
return;
}
const uint32_t B = td->B;
const uint32_t H = td->H;
const uint32_t W = td->W;
const uint32_t S = td->stripesPerBatch;
const uint32_t wAligned = IS_ALIGNED_W ? W : AlignUp8(W);
const uint32_t totalHm = B * FIXED_K;
__gm__ const float* wPtr = reinterpret_cast<__gm__ const float*>(weightAddr);
__gm__ const float* biasPtr = reinterpret_cast<__gm__ const float*>(biasAddr);
const uint64_t hwStride = static_cast<uint64_t>(H) * W;
featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(featAddr),
static_cast<uint64_t>(B) * FIXED_CIN * hwStride);
kpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypointsAddr), static_cast<uint64_t>(B) * 8U);
scGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scoresAddr), static_cast<uint64_t>(B) * 4U);
pipe_.InitBuffer(inBuf_, MAX_IN_FLOATS * sizeof(float));
if constexpr (!REUSE_INBUF_POST_CONV) {
pipe_.InitBuffer(offsetBuf_, 2U * OFFSET_STRIDE * sizeof(uint32_t));
}
pipe_.InitBuffer(convBuf_, 9U * PLANE_STRIDE * sizeof(float));
pipe_.InitBuffer(workBuf_, 1024U * sizeof(float));
LocalTensor<float> inAll = inBuf_.Get<float>();
LocalTensor<float> convAll = convBuf_.Get<float>();
LocalTensor<float> xConv9 = convAll[0];
LocalTensor<float> xConvL = convAll[0U * PLANE_STRIDE];
LocalTensor<float> xConvC = convAll[1U * PLANE_STRIDE];
LocalTensor<float> xConvR = convAll[2U * PLANE_STRIDE];
LocalTensor<float> xTmp = convAll[3U * PLANE_STRIDE];
LocalTensor<float> workAll = workBuf_.Get<float>();
LocalTensor<float> uVec = workAll[0U];
LocalTensor<float> padMask = workAll[128U];
LocalTensor<float> validMask = workAll[256U];
LocalTensor<float> redOut = workAll[384U];
LocalTensor<float> wsLocal = workAll[512U];
LocalTensor<float> mergeLocal = workAll[640U];
const float invTau = td->invTau;
const float strideF = td->strideFloat;
for (uint32_t hm = blockIdx; hm < totalHm; hm += usedCores) {
const uint32_t bIdx = hm >> 2U;
const uint32_t kIdx = hm & 3U;
__gm__ const float* kWPtr = wPtr + kIdx * FIXED_CIN * 9U;
for (uint32_t sIdx = 0U; sIdx < S; ++sIdx) {
if (hm > blockIdx || sIdx > 0U) {
PipeBarrier<PIPE_ALL>();
}
const uint32_t v0 = (sIdx * H) / S;
const uint32_t v1 = ((sIdx + 1U) * H) / S;
const uint32_t hLen = v1 - v0;
const uint32_t stripeElems = hLen * wAligned;
const uint32_t chPerBlock = (MAX_W_ALIGNED <= 96U && hLen <= 20U) ? 4U : 2U;
const uint32_t numBlocks = FIXED_CIN / chPerBlock;
const uint32_t rChStride = (hLen + 2U) * wAligned;
const uint32_t rBlockStride = chPerBlock * rChStride;
LocalTensor<float> inBlock0 = inAll[0];
LocalTensor<float> inBlock1 = inAll[rBlockStride];
const uint32_t gmV0 = (v0 > 0U) ? (v0 - 1U) : 0U;
const uint32_t gmV1 = (v1 < H) ? (v1 + 1U) : H;
const uint32_t gmRows = gmV1 - gmV0;
const uint32_t ubRowOff = ((v0 == 0U) ? 1U : 0U) * wAligned;
const uint32_t gmCopyElems = gmRows * wAligned;
const uint64_t batchFeatBase =
static_cast<uint64_t>(bIdx) * FIXED_CIN * hwStride + static_cast<uint64_t>(gmV0) * W;
const uint32_t rawCopyFloats = gmRows * W;
const uint32_t rawVecElems = (rawCopyFloats + 63U) & ~63U;
const uint8_t rawRCnt = static_cast<uint8_t>(rawVecElems >> 6U);
DataCopyParams mte2Params;
DataCopyExtParams padCopyParams;
if constexpr (IS_ALIGNED_W) {
mte2Params.blockCount = static_cast<uint16_t>(chPerBlock);
mte2Params.blockLen = static_cast<uint16_t>(gmCopyElems >> 3U);
mte2Params.srcStride = static_cast<uint16_t>((static_cast<uint32_t>(hwStride) - gmCopyElems) >> 3U);
mte2Params.dstStride = static_cast<uint16_t>((rChStride - gmCopyElems) >> 3U);
DataCopy(inBlock0[ubRowOff], featGm_[batchFeatBase], mte2Params);
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
DataCopy(inBlock1[ubRowOff],
    featGm_[batchFeatBase + static_cast<uint64_t>(chPerBlock) * hwStride],
    mte2Params);
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
if (sIdx == 0U) {
for (uint32_t dc = 0; dc < 2U * chPerBlock; ++dc) {
Duplicate(inAll[dc * rChStride], 0.0f, wAligned);
}
}
if (sIdx == S - 1U) {
for (uint32_t dc = 0; dc < 2U * chPerBlock; ++dc) {
Duplicate(inAll[dc * rChStride + (hLen + 1U) * wAligned], 0.0f, wAligned);
}
}
} else {
padCopyParams = MakeCopyExt(1U, rawCopyFloats * static_cast<uint32_t>(sizeof(float)));
for (uint32_t dc = 0; dc < chPerBlock; ++dc) {
DataCopyPad(inBlock0[dc * rChStride],
    featGm_[batchFeatBase + static_cast<uint64_t>(dc) * hwStride],
    padCopyParams,
    MakeNoPad());
}
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
for (uint32_t dc = 0; dc < chPerBlock; ++dc) {
DataCopyPad(inBlock1[dc * rChStride],
    featGm_[batchFeatBase + static_cast<uint64_t>(chPerBlock + dc) * hwStride],
    padCopyParams,
    MakeNoPad());
}
SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
}
Duplicate(xConv9, 0.0f, 9U * PLANE_STRIDE);
ArithProgression<float>(uVec, 0.0f, 1.0f, static_cast<int32_t>(wAligned));
if constexpr (!IS_ALIGNED_W) {
Duplicate(padMask, 0.0f, wAligned);
Duplicate(validMask, 1.0f, wAligned);
}
if constexpr (!REUSE_INBUF_POST_CONV) {
LocalTensor<uint32_t> offAll = offsetBuf_.Get<uint32_t>();
LocalTensor<uint32_t> offsetLeft = offAll[0];
LocalTensor<uint32_t> offsetRight = offAll[OFFSET_STRIDE];
ArithProgression<int32_t>(offsetLeft.ReinterpretCast<int32_t>(),
static_cast<int32_t>(-4),
static_cast<int32_t>(4),
static_cast<int32_t>(stripeElems));
ArithProgression<int32_t>(offsetRight.ReinterpretCast<int32_t>(),
static_cast<int32_t>(4),
static_cast<int32_t>(4),
static_cast<int32_t>(stripeElems));
SetFlag<HardEvent::V_S>(EVENT_ID0);
}
PipeBarrier<PIPE_V>();
if (!IS_ALIGNED_W || (stripeElems & 63U) == 0U) {
SetMaskNorm();
SetVectorMask<float, MaskMode::NORMAL>(0xffffffffffffffffULL, 0xffffffffffffffffULL);
}
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
for (uint32_t blk = 0; blk < numBlocks; ++blk) {
const event_t evCur = ((blk & 1U) == 0U) ? EVENT_ID0 : EVENT_ID1;
LocalTensor<float> curBlock = ((blk & 1U) == 0U) ? inBlock0 : inBlock1;
if (blk >= 2U) {
WaitFlag<HardEvent::MTE2_V>(evCur);
}
if constexpr (IS_ALIGNED_W) {
for (uint32_t dc = 0; dc < chPerBlock; ++dc) {
AccumulateChannel9(
    xConv9, curBlock[dc * rChStride], kWPtr, blk * chPerBlock + dc, wAligned, stripeElems);
}
} else {
for (uint32_t dc = 0; dc < chPerBlock; ++dc) {
AccumulateChannel9Raw1D(
    xConv9, curBlock[dc * rChStride], kWPtr, blk * chPerBlock + dc, rawRCnt);
}
}
if (blk + 2U < numBlocks) {
SetFlag<HardEvent::V_MTE2>(evCur);
WaitFlag<HardEvent::V_MTE2>(evCur);
const uint64_t nextGmBase = batchFeatBase + static_cast<uint64_t>((blk + 2U) * chPerBlock) * hwStride;
if constexpr (IS_ALIGNED_W) {
DataCopy(curBlock[ubRowOff], featGm_[nextGmBase], mte2Params);
} else {
for (uint32_t dc = 0; dc < chPerBlock; ++dc) {
DataCopyPad(curBlock[dc * rChStride],
    featGm_[nextGmBase + static_cast<uint64_t>(dc) * hwStride],
    padCopyParams,
    MakeNoPad());
}
}
SetFlag<HardEvent::MTE2_V>(evCur);
}
}
if (!IS_ALIGNED_W || (stripeElems & 63U) == 0U) {
ResetMask();
}
PipeBarrier<PIPE_V>();
LocalTensor<float> vGrid = convAll[5U * PLANE_STRIDE];
LocalTensor<float> redWork = convAll[6U * PLANE_STRIDE];
if constexpr (IS_ALIGNED_W) {
Add(xConv9[0], xConv9[0], xConv9[3U * PLANE_STRIDE], 3U * PLANE_STRIDE);
PipeBarrier<PIPE_V>();
Add(xConv9[0], xConv9[0], xConv9[6U * PLANE_STRIDE], 3U * PLANE_STRIDE);
for (uint32_t r = 1U; r < hLen; ++r) {
Duplicate(vGrid[r * wAligned], U32ToF32(r), wAligned);
}
PipeBarrier<PIPE_V>();
LocalTensor<uint32_t> offsetLeft;
LocalTensor<uint32_t> offsetRight;
if constexpr (!REUSE_INBUF_POST_CONV) {
LocalTensor<uint32_t> offAll = offsetBuf_.Get<uint32_t>();
offsetLeft = offAll[0];
offsetRight = offAll[OFFSET_STRIDE];
WaitFlag<HardEvent::V_S>(EVENT_ID0);
} else {
offsetLeft = inAll[0].ReinterpretCast<uint32_t>();
offsetRight = inAll[OFFSET_STRIDE].ReinterpretCast<uint32_t>();
ArithProgression<int32_t>(offsetLeft.ReinterpretCast<int32_t>(),
static_cast<int32_t>(-4),
static_cast<int32_t>(4),
static_cast<int32_t>(stripeElems));
ArithProgression<int32_t>(offsetRight.ReinterpretCast<int32_t>(),
static_cast<int32_t>(4),
static_cast<int32_t>(4),
static_cast<int32_t>(stripeElems));
Fence<HardEvent::V_S>();
}
const uint32_t zeroByteOff = stripeElems * static_cast<uint32_t>(sizeof(float));
for (uint32_t r = 0; r < hLen; ++r) {
offsetLeft.SetValue(r * wAligned, zeroByteOff);
offsetRight.SetValue(r * wAligned + W - 1U, zeroByteOff);
}
Fence<HardEvent::S_V>();
Gather(xTmp[0U * PLANE_STRIDE], xConvL, offsetLeft, 0U, stripeElems);
Gather(xTmp[1U * PLANE_STRIDE], xConvR, offsetRight, 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, xTmp[0U * PLANE_STRIDE], stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, xTmp[1U * PLANE_STRIDE], stripeElems);
PipeBarrier<PIPE_V>();
} else {
const uint32_t rowByteStep = W * static_cast<uint32_t>(sizeof(float));
const uint32_t rawZeroByteOff = rawVecElems * static_cast<uint32_t>(sizeof(float));
LocalTensor<uint32_t> offL = inAll[0U].ReinterpretCast<uint32_t>();
LocalTensor<uint32_t> offC = inAll[rChStride].ReinterpretCast<uint32_t>();
LocalTensor<uint32_t> offR = inAll[2U * rChStride].ReinterpretCast<uint32_t>();
LocalTensor<float> g0 = inAll[3U * rChStride];
LocalTensor<float> g1 = inAll[3U * rChStride + OFFSET_STRIDE];
if (v0 == 0U) {
Duplicate<uint32_t>(offC[0U], rawZeroByteOff, wAligned);
}
if (v1 == H) {
Duplicate<uint32_t>(offC[(hLen + 1U) * wAligned], rawZeroByteOff, wAligned);
}
const uint32_t urBase = (v0 == 0U) ? 1U : 0U;
ArithProgression<int32_t>(offC[urBase * wAligned].ReinterpretCast<int32_t>(),
0,
4,
static_cast<int32_t>(wAligned));
PipeBarrier<PIPE_V>();
uint32_t builtRows = 1U;
while (builtRows < gmRows) {
const uint32_t copyRows = ((builtRows * 2U) <= gmRows) ? builtRows : (gmRows - builtRows);
Adds<int32_t>(offC[(urBase + builtRows) * wAligned].ReinterpretCast<int32_t>(),
offC[urBase * wAligned].ReinterpretCast<int32_t>(),
static_cast<int32_t>(builtRows * rowByteStep),
static_cast<int32_t>(copyRows * wAligned));
PipeBarrier<PIPE_V>();
builtRows += copyRows;
}
Adds<int32_t>(offL.ReinterpretCast<int32_t>(),
offC.ReinterpretCast<int32_t>(),
-4,
static_cast<int32_t>(rChStride));
Adds<int32_t>(offR.ReinterpretCast<int32_t>(),
offC.ReinterpretCast<int32_t>(),
4,
static_cast<int32_t>(rChStride));
PipeBarrier<PIPE_V>();
if (v0 == 0U) {
Duplicate<uint32_t>(offL[0U], rawZeroByteOff, wAligned);
Duplicate<uint32_t>(offR[0U], rawZeroByteOff, wAligned);
}
if (v1 == H) {
Duplicate<uint32_t>(offL[(hLen + 1U) * wAligned], rawZeroByteOff, wAligned);
Duplicate<uint32_t>(offR[(hLen + 1U) * wAligned], rawZeroByteOff, wAligned);
}
Fence<HardEvent::V_S>();
for (uint32_t gr = 0; gr < gmRows; ++gr) {
const uint32_t ur = urBase + gr;
offL.SetValue(ur * wAligned, rawZeroByteOff);
offR.SetValue(ur * wAligned + W - 1U, rawZeroByteOff);
}
const uint32_t lastUr = urBase + gmRows - 1U;
for (uint32_t j = W; j < wAligned; ++j) {
offL.SetValue(lastUr * wAligned + j, rawZeroByteOff);
offC.SetValue(lastUr * wAligned + j, rawZeroByteOff);
offR.SetValue(lastUr * wAligned + j, rawZeroByteOff);
uVec.SetValue(j, 0.0f);
padMask.SetValue(j, -1.0e20f);
validMask.SetValue(j, 0.0f);
}
Fence<HardEvent::S_V>();
Gather(g0, xConv9[0U * PLANE_STRIDE], offL[0U * wAligned], 0U, stripeElems);
Gather(g1, xConv9[1U * PLANE_STRIDE], offC[0U * wAligned], 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, g0, g1, stripeElems);
Gather(g0, xConv9[2U * PLANE_STRIDE], offR[0U * wAligned], 0U, stripeElems);
Gather(g1, xConv9[3U * PLANE_STRIDE], offL[1U * wAligned], 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, g0, stripeElems);
Gather(g0, xConv9[4U * PLANE_STRIDE], offC[1U * wAligned], 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, g1, stripeElems);
Gather(g1, xConv9[5U * PLANE_STRIDE], offR[1U * wAligned], 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, g0, stripeElems);
Gather(g0, xConv9[6U * PLANE_STRIDE], offL[2U * wAligned], 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, g1, stripeElems);
Gather(g1, xConv9[7U * PLANE_STRIDE], offC[2U * wAligned], 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, g0, stripeElems);
Gather(g0, xConv9[8U * PLANE_STRIDE], offR[2U * wAligned], 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, g1, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, g0, stripeElems);
for (uint32_t r = 1U; r < hLen; ++r) {
Duplicate(vGrid[r * wAligned], U32ToF32(r), wAligned);
}
PipeBarrier<PIPE_V>();
for (uint32_t r = 0; r < hLen; ++r) {
Add(xConvC[r * wAligned], xConvC[r * wAligned], padMask, wAligned);
}
PipeBarrier<PIPE_V>();
}
LocalTensor<float> zBuf = xTmp[0U * PLANE_STRIDE];
LocalTensor<float> eBuf = xTmp[1U * PLANE_STRIDE];
LocalTensor<float> uProd = xConvL[0U];
ReduceMax<float>(redOut[0U], xConvC, redWork, stripeElems, false);
Fence<HardEvent::V_S>();
const float sRaw = redOut.GetValue(0U);
Fence<HardEvent::S_V>();
Adds(zBuf, xConvC, -sRaw, stripeElems);
PipeBarrier<PIPE_V>();
Muls(zBuf, zBuf, invTau, stripeElems);
PipeBarrier<PIPE_V>();
Exp(eBuf, zBuf, stripeElems);
PipeBarrier<PIPE_V>();
if (hLen >= 2U) {
Mul(zBuf[wAligned], eBuf[wAligned], vGrid[wAligned], (hLen - 1U) * wAligned);
}
uint32_t remRows = hLen;
for (uint32_t step = 32U; step >= 1U; step >>= 1U) {
if (remRows > step) {
const uint32_t fRows = remRows - step;
Add(eBuf[0U], eBuf[0U], eBuf[step * wAligned], fRows * wAligned);
PipeBarrier<PIPE_V>();
remRows = step;
}
}
if constexpr (!IS_ALIGNED_W) {
Mul(eBuf[0U], eBuf[0U], validMask, wAligned);
PipeBarrier<PIPE_V>();
}
if (hLen >= 2U) {
ReduceSum<float>(redOut[24U], zBuf[wAligned], redWork[0U], (hLen - 1U) * wAligned);
}
ReduceSum<float>(redOut[8U], eBuf[0U], redWork[256U], wAligned);
Mul(uProd, eBuf[0U], uVec, wAligned);
PipeBarrier<PIPE_V>();
ReduceSum<float>(redOut[16U], uProd, redWork[512U], wAligned);
Fence<HardEvent::V_S>();
const float v0f = U32ToF32(v0);
const float sumLoc = redOut.GetValue(8U);
const float uLoc = redOut.GetValue(16U);
const float vRel = (hLen >= 2U) ? redOut.GetValue(24U) : 0.0f;
mergeLocal.SetValue(sIdx * 4U + 0U, sRaw);
mergeLocal.SetValue(sIdx * 4U + 1U, sumLoc);
mergeLocal.SetValue(sIdx * 4U + 2U, uLoc);
mergeLocal.SetValue(sIdx * 4U + 3U, v0f * sumLoc + vRel);
}
if (S == 1U) {
const float sGlob = mergeLocal.GetValue(0U);
const float sumTot = mergeLocal.GetValue(1U);
const float uTot = mergeLocal.GetValue(2U);
const float vTot = mergeLocal.GetValue(3U);
const float invS = 1.0f / sumTot;
wsLocal.SetValue(0U, (uTot * invS + 0.5f) * strideF);
wsLocal.SetValue(1U, (vTot * invS + 0.5f) * strideF);
wsLocal.SetValue(8U, sGlob + WVal(biasPtr, kIdx));
} else {
LocalTensor<float> zBuf = xTmp[0U * PLANE_STRIDE];
LocalTensor<float> eBuf = xTmp[1U * PLANE_STRIDE];
float sGlob = mergeLocal.GetValue(0U);
for (uint32_t s = 1U; s < S; ++s) {
const float v = mergeLocal.GetValue(s * 4U);
if (v > sGlob) {
sGlob = v;
}
}
const uint32_t sAligned = AlignUp8(S);
for (uint32_t s = 0U; s < S; ++s) {
zBuf.SetValue(s, mergeLocal.GetValue(s * 4U) - sGlob);
}
for (uint32_t s = S; s < sAligned; ++s) {
zBuf.SetValue(s, 0.0f);
}
Fence<HardEvent::S_V>();
Muls(zBuf, zBuf, invTau, sAligned);
PipeBarrier<PIPE_V>();
Exp(eBuf, zBuf, sAligned);
Fence<HardEvent::V_S>();
float sumTot = 0.0f;
float uTot = 0.0f;
float vTot = 0.0f;
for (uint32_t s = 0U; s < S; ++s) {
const float alpha = eBuf.GetValue(s);
const uint32_t sOff = s * 4U;
sumTot += alpha * mergeLocal.GetValue(sOff + 1U);
uTot += alpha * mergeLocal.GetValue(sOff + 2U);
vTot += alpha * mergeLocal.GetValue(sOff + 3U);
}
const float invS = 1.0f / sumTot;
wsLocal.SetValue(0U, (uTot * invS + 0.5f) * strideF);
wsLocal.SetValue(1U, (vTot * invS + 0.5f) * strideF);
wsLocal.SetValue(8U, sGlob + WVal(biasPtr, kIdx));
}
Fence<HardEvent::S_MTE3>();
DataCopyPad(kpGm_[static_cast<uint64_t>(hm) * 2U], wsLocal[0U], MakeCopyExt(1U, 2U * sizeof(float)));
DataCopyPad(scGm_[static_cast<uint64_t>(hm)], wsLocal[8U], MakeCopyExt(1U, sizeof(float)));
Fence<HardEvent::MTE3_S>();
}
}
private:
TPipe pipe_;
GlobalTensor<float> featGm_;
GlobalTensor<float> kpGm_;
GlobalTensor<float> scGm_;
TBuf<TPosition::VECCALC> inBuf_;
TBuf<TPosition::VECCALC> offsetBuf_;
TBuf<TPosition::VECCALC> convBuf_;
TBuf<TPosition::VECCALC> workBuf_;
};
class KernelGeneralAligned {
public:
static constexpr uint32_t MAX_W = 256U;
static constexpr uint32_t TILE_H = 8U;
static constexpr uint32_t MAX_HIN = TILE_H + 2U;
static constexpr uint32_t IN_CH_STRIDE = MAX_HIN * MAX_W;
static constexpr uint32_t IN_BLOCK_STRIDE = 4U * IN_CH_STRIDE;
static constexpr uint32_t PLANE_STRIDE = TILE_H * MAX_W + 8U;
__aicore__ inline float ScalarExpScaled(float diffVal,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td,
const LocalTensor<float>& scratch)
{
const float scaled = diffVal * td->invTau;
scratch.SetValue(0, scaled);
Fence<HardEvent::S_V>();
Exp(scratch[8], scratch[0], 8U);
Fence<HardEvent::V_S>();
return scratch.GetValue(8);
}
__aicore__ inline void Run(GM_ADDR featAddr,
GM_ADDR weightAddr,
GM_ADDR biasAddr,
GM_ADDR keypointsAddr,
GM_ADDR scoresAddr,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td)
{
const uint32_t blockIdx = GetBlockIdx();
const uint32_t usedCores = td->usedCores;
if (blockIdx >= usedCores) {
return;
}
const uint32_t B = td->B;
const uint32_t H = td->H;
const uint32_t W = td->W;
const uint32_t totalHm = B * FIXED_K;
const uint64_t hwStride = static_cast<uint64_t>(H) * W;
__gm__ const float* wPtr = reinterpret_cast<__gm__ const float*>(weightAddr);
__gm__ const float* biasPtr = reinterpret_cast<__gm__ const float*>(biasAddr);
featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(featAddr),
static_cast<uint64_t>(B) * FIXED_CIN * hwStride);
kpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypointsAddr), static_cast<uint64_t>(B) * 8U);
scGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scoresAddr), static_cast<uint64_t>(B) * 4U);
pipe_.InitBuffer(inBuf_, 2U * IN_BLOCK_STRIDE * sizeof(float));
pipe_.InitBuffer(offsetBuf_, 2U * PLANE_STRIDE * sizeof(uint32_t));
pipe_.InitBuffer(convBuf_, 9U * PLANE_STRIDE * sizeof(float));
pipe_.InitBuffer(workBuf_, 1792U * sizeof(float));
LocalTensor<float> inAll = inBuf_.Get<float>();
LocalTensor<uint32_t> offAll = offsetBuf_.Get<uint32_t>();
LocalTensor<uint32_t> offsetLeft = offAll[0];
LocalTensor<uint32_t> offsetRight = offAll[PLANE_STRIDE];
LocalTensor<float> convAll = convBuf_.Get<float>();
LocalTensor<float> xConv9 = convAll[0];
LocalTensor<float> xConvL = convAll[0U * PLANE_STRIDE];
LocalTensor<float> xConvC = convAll[1U * PLANE_STRIDE];
LocalTensor<float> xConvR = convAll[2U * PLANE_STRIDE];
LocalTensor<float> xTmp = convAll[3U * PLANE_STRIDE];
LocalTensor<float> zBuf = convAll[3U * PLANE_STRIDE];
LocalTensor<float> eBuf = convAll[4U * PLANE_STRIDE];
LocalTensor<float> redWork = convAll[5U * PLANE_STRIDE];
LocalTensor<float> workAll = workBuf_.Get<float>();
const uint32_t auxBase = 0U;
LocalTensor<float> uVec = workAll[auxBase];
LocalTensor<float> colSum = workAll[auxBase + MAX_W];
LocalTensor<float> uProd = workAll[auxBase + 2U * MAX_W];
LocalTensor<float> vColAcc = workAll[auxBase + 3U * MAX_W];
LocalTensor<float> padMask = workAll[auxBase + 4U * MAX_W];
LocalTensor<float> validMask = workAll[auxBase + 5U * MAX_W];
LocalTensor<float> redOut = workAll[auxBase + 6U * MAX_W];
LocalTensor<float> wsLocal = workAll[auxBase + 6U * MAX_W + 64U];
const uint32_t wAligned = AlignUp8(W);
const bool isAlignedW = ((W & 7U) == 0U);
const uint32_t maxStripeElems = TILE_H * wAligned;
const uint32_t zeroByteOff = maxStripeElems * static_cast<uint32_t>(sizeof(float));
ArithProgression<float>(uVec, 0.0f, 1.0f, static_cast<int32_t>(wAligned));
ArithProgression<int32_t>(offsetLeft.ReinterpretCast<int32_t>(),
static_cast<int32_t>(-4),
static_cast<int32_t>(4),
static_cast<int32_t>(maxStripeElems));
ArithProgression<int32_t>(offsetRight.ReinterpretCast<int32_t>(),
static_cast<int32_t>(4),
static_cast<int32_t>(4),
static_cast<int32_t>(maxStripeElems));
if (!isAlignedW) {
Duplicate(padMask, 0.0f, wAligned);
Duplicate(validMask, 1.0f, wAligned);
}
Fence<HardEvent::V_S>();
for (uint32_t r = 0; r < TILE_H; ++r) {
offsetLeft.SetValue(r * wAligned, zeroByteOff);
offsetRight.SetValue(r * wAligned + W - 1U, zeroByteOff);
}
if (!isAlignedW) {
for (uint32_t j = W; j < wAligned; ++j) {
uVec.SetValue(j, 0.0f);
padMask.SetValue(j, -1.0e20f);
validMask.SetValue(j, 0.0f);
}
}
Fence<HardEvent::S_V>();
PipeBarrier<PIPE_V>();
const uint32_t rChStride = (TILE_H + 2U) * wAligned;
const uint32_t rBlockStride = 4U * rChStride;
LocalTensor<float> inBlock0 = inAll[0];
LocalTensor<float> inBlock1 = inAll[rBlockStride];
for (uint32_t hm = blockIdx; hm < totalHm; hm += usedCores) {
const uint32_t bIdx = hm >> 2U;
const uint32_t kIdx = hm & 3U;
__gm__ const float* kWPtr = wPtr + kIdx * FIXED_CIN * 9U;
float sAcc = NEG_INF_FP32;
float sumAcc = 0.0f;
float uAcc = 0.0f;
float vAcc = 0.0f;
for (uint32_t v0 = 0U; v0 < H; v0 += TILE_H) {
const uint32_t v1 = ((v0 + TILE_H) < H) ? (v0 + TILE_H) : H;
const uint32_t hLen = v1 - v0;
const uint32_t stripeElems = hLen * wAligned;
if (isAlignedW) {
if (v0 == 0U) {
#pragma unroll
for (uint32_t dc = 0; dc < 8U; ++dc) {
Duplicate(inAll[dc * rChStride], 0.0f, wAligned);
}
}
if (v1 == H) {
#pragma unroll
for (uint32_t dc = 0; dc < 8U; ++dc) {
Duplicate(inAll[dc * rChStride + (hLen + 1U) * wAligned], 0.0f, wAligned);
}
}
} else {
Duplicate(inAll, 0.0f, 2U * rBlockStride);
}
Fence<HardEvent::V_MTE2>();
const uint32_t gmV0 = (v0 > 0U) ? (v0 - 1U) : 0U;
const uint32_t gmV1 = (v1 < H) ? (v1 + 1U) : H;
const uint32_t gmRows = gmV1 - gmV0;
const uint32_t ubRowOff = ((v0 == 0U) ? 1U : 0U) * wAligned;
const uint32_t gmCopyElems = gmRows * wAligned;
const uint64_t batchBase =
static_cast<uint64_t>(bIdx) * FIXED_CIN * hwStride + static_cast<uint64_t>(gmV0) * W;
DataCopyParams mte2Params;
DataCopyExtParams padCopyParams;
if (isAlignedW) {
mte2Params.blockCount = 4U;
mte2Params.blockLen = static_cast<uint16_t>(gmCopyElems >> 3U);
mte2Params.srcStride = static_cast<uint16_t>((static_cast<uint32_t>(hwStride) - gmCopyElems) >> 3U);
mte2Params.dstStride = static_cast<uint16_t>((rChStride - gmCopyElems) >> 3U);
DataCopy(inBlock0[ubRowOff], featGm_[batchBase], mte2Params);
} else {
padCopyParams = MakeCopyExt(static_cast<uint16_t>(gmRows), W * static_cast<uint32_t>(sizeof(float)));
#pragma unroll
for (uint32_t dc = 0; dc < 4U; ++dc) {
DataCopyPad(inBlock0[dc * rChStride + ubRowOff],
    featGm_[batchBase + static_cast<uint64_t>(dc) * hwStride],
    padCopyParams,
    MakeNoPad());
}
}
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
Duplicate(xConv9, 0.0f, 9U * PLANE_STRIDE);
PipeBarrier<PIPE_V>();
for (uint32_t blk = 0; blk < 8U; ++blk) {
const event_t evCur = ((blk & 1U) == 0U) ? EVENT_ID0 : EVENT_ID1;
const event_t evNext = ((blk & 1U) == 0U) ? EVENT_ID1 : EVENT_ID0;
LocalTensor<float> curBlock = ((blk & 1U) == 0U) ? inBlock0 : inBlock1;
LocalTensor<float> nextBlock = ((blk & 1U) == 0U) ? inBlock1 : inBlock0;
if (blk + 1U < 8U) {
if (blk + 1U >= 2U) {
WaitFlag<HardEvent::V_MTE2>(evNext);
}
const uint64_t nextGmBase = batchBase + static_cast<uint64_t>((blk + 1U) * 4U) * hwStride;
if (isAlignedW) {
DataCopy(nextBlock[ubRowOff], featGm_[nextGmBase], mte2Params);
} else {
#pragma unroll
for (uint32_t dc = 0; dc < 4U; ++dc) {
DataCopyPad(nextBlock[dc * rChStride + ubRowOff],
    featGm_[nextGmBase + static_cast<uint64_t>(dc) * hwStride],
    padCopyParams,
    MakeNoPad());
}
}
SetFlag<HardEvent::MTE2_V>(evNext);
}
WaitFlag<HardEvent::MTE2_V>(evCur);
#pragma unroll
for (uint32_t dc = 0; dc < 4U; ++dc) {
const uint32_t c = blk * 4U + dc;
const uint32_t wBase = c * 9U;
LocalTensor<float> curCh = curBlock[dc * rChStride];
#pragma unroll
for (uint32_t dh = 0; dh < 3U; ++dh) {
LocalTensor<float> srcRow = curCh[dh * wAligned];
const uint32_t dOff = dh * 3U;
const uint32_t dhPlaneBase = dh * 3U * PLANE_STRIDE;
Axpy(xConv9[dhPlaneBase + 0U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 0U), stripeElems);
Axpy(xConv9[dhPlaneBase + 1U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 1U), stripeElems);
Axpy(xConv9[dhPlaneBase + 2U * PLANE_STRIDE], srcRow, WVal(kWPtr, wBase + dOff + 2U), stripeElems);
}
PipeBarrier<PIPE_V>();
}
if (blk + 2U < 8U) {
SetFlag<HardEvent::V_MTE2>(evCur);
}
}
Add(xConv9[0], xConv9[0], xConv9[3U * PLANE_STRIDE], 3U * PLANE_STRIDE);
PipeBarrier<PIPE_V>();
Add(xConv9[0], xConv9[0], xConv9[6U * PLANE_STRIDE], 3U * PLANE_STRIDE);
PipeBarrier<PIPE_V>();
Fence<HardEvent::V_MTE2>();
Gather(xTmp[0U * PLANE_STRIDE], xConvL, offsetLeft, 0U, stripeElems);
Gather(xTmp[1U * PLANE_STRIDE], xConvR, offsetRight, 0U, stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, xTmp[0U * PLANE_STRIDE], stripeElems);
PipeBarrier<PIPE_V>();
Add(xConvC, xConvC, xTmp[1U * PLANE_STRIDE], stripeElems);
PipeBarrier<PIPE_V>();
if (!isAlignedW) {
for (uint32_t r = 0; r < hLen; ++r) {
Add(xConvC[r * wAligned], xConvC[r * wAligned], padMask, wAligned);
}
PipeBarrier<PIPE_V>();
}
ReduceMax<float>(redOut[0], xConvC, redWork, stripeElems, false);
Fence<HardEvent::V_S>();
const float sTile = redOut.GetValue(0);
Fence<HardEvent::S_V>();
Adds(zBuf, xConvC, -sTile, stripeElems);
PipeBarrier<PIPE_V>();
Muls(zBuf, zBuf, td->invTau, stripeElems);
PipeBarrier<PIPE_V>();
Exp(eBuf, zBuf, stripeElems);
PipeBarrier<PIPE_V>();
Adds(colSum, eBuf[0], 0.0f, wAligned);
PipeBarrier<PIPE_V>();
for (uint32_t r = 1U; r < hLen; ++r) {
Add(colSum, colSum, eBuf[r * wAligned], wAligned);
PipeBarrier<PIPE_V>();
}
if (!isAlignedW) {
Mul(colSum, colSum, validMask, wAligned);
PipeBarrier<PIPE_V>();
}
ReduceSum<float>(redOut[8], colSum, redWork, wAligned);
Mul(uProd, colSum, uVec, wAligned);
PipeBarrier<PIPE_V>();
ReduceSum<float>(redOut[16], uProd, redWork, wAligned);
if (hLen == 2U) {
if (!isAlignedW) {
Mul(vColAcc, eBuf[wAligned], validMask, wAligned);
PipeBarrier<PIPE_V>();
ReduceSum<float>(redOut[24], vColAcc, redWork, wAligned);
} else {
ReduceSum<float>(redOut[24], eBuf[wAligned], redWork, wAligned);
}
} else if (hLen > 2U) {
Adds(vColAcc, eBuf[wAligned], 0.0f, wAligned);
PipeBarrier<PIPE_V>();
for (uint32_t r = 2U; r < hLen; ++r) {
Axpy(vColAcc, eBuf[r * wAligned], U32ToF32(r), wAligned);
PipeBarrier<PIPE_V>();
}
if (!isAlignedW) {
Mul(vColAcc, vColAcc, validMask, wAligned);
PipeBarrier<PIPE_V>();
}
ReduceSum<float>(redOut[24], vColAcc, redWork, wAligned);
}
Fence<HardEvent::V_S>();
const float sumTile = redOut.GetValue(8);
const float uTile = redOut.GetValue(16);
const float vTile = U32ToF32(v0) * sumTile + ((hLen >= 2U) ? redOut.GetValue(24) : 0.0f);
if (v0 == 0U) {
sAcc = sTile;
sumAcc = sumTile;
uAcc = uTile;
vAcc = vTile;
} else if (sTile >= sAcc) {
const float scaleOld = ScalarExpScaled(sAcc - sTile, td, redOut[32]);
sAcc = sTile;
sumAcc = sumAcc * scaleOld + sumTile;
uAcc = uAcc * scaleOld + uTile;
vAcc = vAcc * scaleOld + vTile;
} else {
const float scaleNew = ScalarExpScaled(sTile - sAcc, td, redOut[32]);
sumAcc += sumTile * scaleNew;
uAcc += uTile * scaleNew;
vAcc += vTile * scaleNew;
}
Fence<HardEvent::S_V>();
}
const float invS = 1.0f / sumAcc;
wsLocal.SetValue(0U, (uAcc * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(1U, (vAcc * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(8U, sAcc + WVal(biasPtr, kIdx));
Fence<HardEvent::S_MTE3>();
DataCopyPad(kpGm_[static_cast<uint64_t>(hm) * 2U], wsLocal[0U], MakeCopyExt(1U, 2U * sizeof(float)));
DataCopyPad(scGm_[static_cast<uint64_t>(hm)], wsLocal[8U], MakeCopyExt(1U, sizeof(float)));
Fence<HardEvent::MTE3_S>();
}
}
private:
TPipe pipe_;
GlobalTensor<float> featGm_;
GlobalTensor<float> kpGm_;
GlobalTensor<float> scGm_;
TBuf<TPosition::VECCALC> inBuf_;
TBuf<TPosition::VECCALC> offsetBuf_;
TBuf<TPosition::VECCALC> convBuf_;
TBuf<TPosition::VECCALC> workBuf_;
};
class KernelUniversalFallback {
public:
static constexpr uint32_t CHUNK_W = 256U;
static constexpr uint32_t ROW_STRIDE = CHUNK_W + 16U;
__aicore__ inline float ScalarExpScaled(float diffVal,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td,
const LocalTensor<float>& scratch)
{
const float scaled = diffVal * td->invTau;
scratch.SetValue(0, scaled);
Fence<HardEvent::S_V>();
Exp(scratch[8], scratch[0], 8U);
Fence<HardEvent::V_S>();
return scratch.GetValue(8);
}
__aicore__ inline void Run(GM_ADDR featAddr,
GM_ADDR weightAddr,
GM_ADDR biasAddr,
GM_ADDR keypointsAddr,
GM_ADDR scoresAddr,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td)
{
const uint32_t blockIdx = GetBlockIdx();
const uint32_t usedCores = td->usedCores;
if (blockIdx >= usedCores) {
return;
}
const uint32_t B = td->B;
const uint32_t H = td->H;
const uint32_t W = td->W;
const uint64_t hwStride = static_cast<uint64_t>(H) * W;
__gm__ const float* wPtr = reinterpret_cast<__gm__ const float*>(weightAddr);
__gm__ const float* biasPtr = reinterpret_cast<__gm__ const float*>(biasAddr);
featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(featAddr),
static_cast<uint64_t>(B) * FIXED_CIN * hwStride);
kpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypointsAddr), static_cast<uint64_t>(B) * 8U);
scGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scoresAddr), static_cast<uint64_t>(B) * 4U);
pipe_.InitBuffer(inBuf_, 3U * ROW_STRIDE * sizeof(float));
pipe_.InitBuffer(shiftBuf_, 6U * ROW_STRIDE * sizeof(float));
pipe_.InitBuffer(offsetBuf_, 3U * ROW_STRIDE * sizeof(uint32_t));
pipe_.InitBuffer(convBuf_, 4U * FIXED_K * ROW_STRIDE * sizeof(float));
pipe_.InitBuffer(workBuf_, 8U * ROW_STRIDE * sizeof(float));
LocalTensor<float> fRows = inBuf_.Get<float>();
LocalTensor<float> shiftAll = shiftBuf_.Get<float>();
LocalTensor<float> fLeft = shiftAll[0];
LocalTensor<float> fRight = shiftAll[3U * ROW_STRIDE];
LocalTensor<uint32_t> offAll = offsetBuf_.Get<uint32_t>();
LocalTensor<uint32_t> offL = offAll[0];
LocalTensor<uint32_t> offC = offAll[ROW_STRIDE];
LocalTensor<uint32_t> offR = offAll[2U * ROW_STRIDE];
LocalTensor<float> convAll = convBuf_.Get<float>();
LocalTensor<float> xConv = convAll[0];
LocalTensor<float> xAccL = convAll[1U * FIXED_K * ROW_STRIDE];
LocalTensor<float> xAccC = convAll[2U * FIXED_K * ROW_STRIDE];
LocalTensor<float> xAccR = convAll[3U * FIXED_K * ROW_STRIDE];
LocalTensor<float> workAll = workBuf_.Get<float>();
LocalTensor<float> zBuf = workAll[0];
LocalTensor<float> eBuf = workAll[ROW_STRIDE];
LocalTensor<float> uVec = workAll[2U * ROW_STRIDE];
LocalTensor<float> redWork = workAll[3U * ROW_STRIDE];
LocalTensor<float> redOut = workAll[4U * ROW_STRIDE];
LocalTensor<float> wsLocal = workAll[5U * ROW_STRIDE];
for (uint32_t bIdx = blockIdx; bIdx < B; bIdx += usedCores) {
float sGlob[FIXED_K] = {NEG_INF_FP32, NEG_INF_FP32, NEG_INF_FP32, NEG_INF_FP32};
float sumTot[FIXED_K] = {0.0f, 0.0f, 0.0f, 0.0f};
float uTot[FIXED_K] = {0.0f, 0.0f, 0.0f, 0.0f};
float vTot[FIXED_K] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint32_t vRow = 0; vRow < H; ++vRow) {
float sAcc[FIXED_K] = {NEG_INF_FP32, NEG_INF_FP32, NEG_INF_FP32, NEG_INF_FP32};
float sumAcc[FIXED_K] = {0.0f, 0.0f, 0.0f, 0.0f};
float uAcc[FIXED_K] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint32_t u0 = 0; u0 < W; u0 += CHUNK_W) {
const uint32_t u1 = ((u0 + CHUNK_W) < W) ? (u0 + CHUNK_W) : W;
const uint32_t wLen = u1 - u0;
const uint32_t wAligned = AlignUp8(wLen);
const uint32_t gmU0 = (u0 > 0U) ? (u0 - 1U) : 0U;
const uint32_t gmU1 = (u1 < W) ? (u1 + 1U) : W;
const uint32_t copyCols = gmU1 - gmU0;
const uint32_t hasLeft = (u0 > 0U) ? 1U : 0U;
const uint32_t zeroSlotIdx = CHUNK_W + 8U;
const uint32_t zeroByteOff = zeroSlotIdx * 4U;
ArithProgression<float>(uVec, U32ToF32(u0), 1.0f, static_cast<int32_t>(wAligned));
ArithProgression<int32_t>(offL.ReinterpretCast<int32_t>(), 0, 4, static_cast<int32_t>(wAligned));
ArithProgression<int32_t>(offC.ReinterpretCast<int32_t>(), 4, 4, static_cast<int32_t>(wAligned));
ArithProgression<int32_t>(offR.ReinterpretCast<int32_t>(), 8, 4, static_cast<int32_t>(wAligned));
Fence<HardEvent::V_S>();
for (uint32_t j = wLen; j < wAligned; ++j) {
uVec.SetValue(j, 0.0f);
offL.SetValue(j, zeroByteOff);
offC.SetValue(j, zeroByteOff);
offR.SetValue(j, zeroByteOff);
}
Fence<HardEvent::S_V>();
for (uint32_t c = 0; c < FIXED_CIN; ++c) {
Duplicate(fRows, 0.0f, 3U * ROW_STRIDE);
if ((c & 3U) == 0U) {
Duplicate(xAccL, 0.0f, 3U * FIXED_K * ROW_STRIDE);
}
PipeBarrier<PIPE_V>();
Fence<HardEvent::V_MTE2>();
for (uint32_t dh = 0; dh < 3U; ++dh) {
const int32_t gv = static_cast<int32_t>(vRow) + static_cast<int32_t>(dh) - 1;
if (gv >= 0 && gv < static_cast<int32_t>(H)) {
const uint64_t gmOff =
(static_cast<uint64_t>(bIdx) * FIXED_CIN + c) * hwStride +
static_cast<uint64_t>(gv) * W + gmU0;
DataCopyPadExtParams<float> padP{};
padP.isPad = (hasLeft == 0U);
padP.leftPadding = (hasLeft == 0U) ? 1 : 0;
padP.rightPadding = 0;
padP.paddingValue = 0.0f;
DataCopyPad(fRows[dh * ROW_STRIDE],
    featGm_[gmOff],
    MakeCopyExt(1U, copyCols * sizeof(float)),
    padP);
}
}
Fence<HardEvent::MTE2_V>();
for (uint32_t dh = 0; dh < 3U; ++dh) {
Gather(fLeft[dh * ROW_STRIDE], fRows[dh * ROW_STRIDE], offL, 0U, wAligned);
Gather(fRight[dh * ROW_STRIDE], fRows[dh * ROW_STRIDE], offR, 0U, wAligned);
}
PipeBarrier<PIPE_V>();
for (uint32_t dh = 0; dh < 3U; ++dh) {
Gather(zBuf, fRows[dh * ROW_STRIDE], offC, 0U, wAligned);
PipeBarrier<PIPE_V>();
#pragma unroll
for (uint32_t k = 0; k < FIXED_K; ++k) {
const uint32_t wb = (k * FIXED_CIN + c) * 9U + dh * 3U;
Axpy(xAccL[k * ROW_STRIDE], fLeft[dh * ROW_STRIDE], WVal(wPtr, wb + 0U), wAligned);
Axpy(xAccC[k * ROW_STRIDE], zBuf, WVal(wPtr, wb + 1U), wAligned);
Axpy(xAccR[k * ROW_STRIDE], fRight[dh * ROW_STRIDE], WVal(wPtr, wb + 2U), wAligned);
}
PipeBarrier<PIPE_V>();
}
if ((c & 3U) == 3U) {
Add(xAccL, xAccL, xAccC, FIXED_K * ROW_STRIDE);
PipeBarrier<PIPE_V>();
if (c == 3U) {
Add(xConv, xAccL, xAccR, FIXED_K * ROW_STRIDE);
} else {
Add(xAccL, xAccL, xAccR, FIXED_K * ROW_STRIDE);
PipeBarrier<PIPE_V>();
Add(xConv, xConv, xAccL, FIXED_K * ROW_STRIDE);
}
PipeBarrier<PIPE_V>();
}
}
Fence<HardEvent::V_S>();
for (uint32_t k = 0; k < FIXED_K; ++k) {
LocalTensor<float> xk = xConv[k * ROW_STRIDE];
for (uint32_t j = wLen; j < wAligned; ++j) {
xk.SetValue(j, NEG_INF_FP32);
}
Fence<HardEvent::S_V>();
ReduceMax<float>(redOut[0], xk, redWork, wAligned, false);
Fence<HardEvent::V_S>();
const float sChunk = redOut.GetValue(0);
Fence<HardEvent::S_V>();
Adds(zBuf, xk, -sChunk, wAligned);
PipeBarrier<PIPE_V>();
Muls(zBuf, zBuf, td->invTau, wAligned);
PipeBarrier<PIPE_V>();
Exp(eBuf, zBuf, wAligned);
Fence<HardEvent::V_S>();
for (uint32_t j = wLen; j < wAligned; ++j) {
eBuf.SetValue(j, 0.0f);
}
Fence<HardEvent::S_V>();
ReduceSum<float>(redOut[8], eBuf, redWork, wAligned);
Mul(zBuf, eBuf, uVec, wAligned);
PipeBarrier<PIPE_V>();
ReduceSum<float>(redOut[16], zBuf, redWork, wAligned);
Fence<HardEvent::V_S>();
const float sumChunk = redOut.GetValue(8);
const float uChunk = redOut.GetValue(16);
if (u0 == 0U) {
sAcc[k] = sChunk;
sumAcc[k] = sumChunk;
uAcc[k] = uChunk;
} else if (sChunk >= sAcc[k]) {
const float scOld = ScalarExpScaled(sAcc[k] - sChunk, td, redOut[32]);
sAcc[k] = sChunk;
sumAcc[k] = sumAcc[k] * scOld + sumChunk;
uAcc[k] = uAcc[k] * scOld + uChunk;
} else {
const float scNew = ScalarExpScaled(sChunk - sAcc[k], td, redOut[32]);
sumAcc[k] += sumChunk * scNew;
uAcc[k] += uChunk * scNew;
}
}
Fence<HardEvent::S_V>();
}
const float vRowF = U32ToF32(vRow);
for (uint32_t k = 0; k < FIXED_K; ++k) {
const float vRowVal = vRowF * sumAcc[k];
if (vRow == 0U) {
sGlob[k] = sAcc[k];
sumTot[k] = sumAcc[k];
uTot[k] = uAcc[k];
vTot[k] = vRowVal;
} else if (sAcc[k] >= sGlob[k]) {
const float scOld = ScalarExpScaled(sGlob[k] - sAcc[k], td, redOut[32]);
sGlob[k] = sAcc[k];
sumTot[k] = sumTot[k] * scOld + sumAcc[k];
uTot[k] = uTot[k] * scOld + uAcc[k];
vTot[k] = vTot[k] * scOld + vRowVal;
} else {
const float scNew = ScalarExpScaled(sAcc[k] - sGlob[k], td, redOut[32]);
sumTot[k] += sumAcc[k] * scNew;
uTot[k] += uAcc[k] * scNew;
vTot[k] += vRowVal * scNew;
}
}
Fence<HardEvent::S_V>();
}
for (uint32_t k = 0; k < FIXED_K; ++k) {
const float invS = 1.0f / sumTot[k];
wsLocal.SetValue(k * 2U + 0U, (uTot[k] * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(k * 2U + 1U, (vTot[k] * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(16U + k, sGlob[k] + WVal(biasPtr, k));
}
Fence<HardEvent::S_MTE3>();
DataCopy(kpGm_[static_cast<uint64_t>(bIdx) * 8U], wsLocal[0], 8U);
DataCopyPad(scGm_[static_cast<uint64_t>(bIdx) * 4U],
    wsLocal[16],
    MakeCopyExt(1U, FIXED_K * sizeof(float)));
Fence<HardEvent::MTE3_S>();
}
}
private:
TPipe pipe_;
GlobalTensor<float> featGm_;
GlobalTensor<float> kpGm_;
GlobalTensor<float> scGm_;
TBuf<TPosition::VECCALC> inBuf_;
TBuf<TPosition::VECCALC> shiftBuf_;
TBuf<TPosition::VECCALC> offsetBuf_;
TBuf<TPosition::VECCALC> convBuf_;
TBuf<TPosition::VECCALC> workBuf_;
};
class KernelBaselinePerHeatmap {
public:
static constexpr uint32_t MAX_W = 256U;
static constexpr uint32_t ROW_STRIDE = MAX_W + 16U;
__aicore__ inline float ScalarExpScaled(float diffVal,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td,
const LocalTensor<float>& scratch)
{
const float scaled = (td->exactInvTau != 0U) ? (diffVal * td->invTau) : (diffVal / td->tau);
scratch.SetValue(0, scaled);
Fence<HardEvent::S_V>();
Exp(scratch[8], scratch[0], 8U);
Fence<HardEvent::V_S>();
return scratch.GetValue(8);
}
__aicore__ inline void Run(GM_ADDR featAddr,
GM_ADDR weightAddr,
GM_ADDR biasAddr,
GM_ADDR keypointsAddr,
GM_ADDR scoresAddr,
const FusedDisconnectorKeypointSoftArgmaxTilingHeader* td)
{
const uint32_t blockIdx = GetBlockIdx();
const uint32_t usedCores = td->usedCores;
if (blockIdx >= usedCores) {
return;
}
const uint32_t B = td->B;
const uint32_t H = td->H;
const uint32_t W = td->W;
const uint32_t totalHeatmaps = B * FIXED_K;
const uint64_t hwStride = static_cast<uint64_t>(H) * W;
featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(featAddr),
static_cast<uint64_t>(B) * FIXED_CIN * hwStride);
weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightAddr), WEIGHT_FLOATS);
biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(biasAddr), FIXED_K);
kpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keypointsAddr), static_cast<uint64_t>(B) * 8U);
scGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scoresAddr), static_cast<uint64_t>(B) * 4U);
pipe_.InitBuffer(inBuf_, 3U * ROW_STRIDE * sizeof(float));
pipe_.InitBuffer(offsetBuf_, 2U * ROW_STRIDE * sizeof(uint32_t));
pipe_.InitBuffer(convBuf_, 2U * ROW_STRIDE * sizeof(float));
pipe_.InitBuffer(paramBuf_, (WEIGHT_FLOATS + 64U) * sizeof(float));
pipe_.InitBuffer(workBuf_, (8U * ROW_STRIDE + 1024U) * sizeof(float));
LocalTensor<float> fRows = inBuf_.Get<float>();
LocalTensor<uint32_t> offAll = offsetBuf_.Get<uint32_t>();
LocalTensor<uint32_t> offL = offAll[0];
LocalTensor<uint32_t> offR = offAll[ROW_STRIDE];
LocalTensor<float> xRow = convBuf_.Get<float>();
LocalTensor<float> xBlk = xRow[ROW_STRIDE];
LocalTensor<float> wLocal = paramBuf_.Get<float>();
LocalTensor<float> biasLocal = wLocal[WEIGHT_FLOATS];
LocalTensor<float> workAll = workBuf_.Get<float>();
LocalTensor<float> shiftTmp = workAll[0];
LocalTensor<float> zBuf = workAll[ROW_STRIDE];
LocalTensor<float> eBuf = workAll[2U * ROW_STRIDE];
LocalTensor<float> tauVec = workAll[3U * ROW_STRIDE];
LocalTensor<float> uVec = workAll[4U * ROW_STRIDE];
LocalTensor<float> redWork = workAll[5U * ROW_STRIDE];
LocalTensor<float> redOut = workAll[6U * ROW_STRIDE];
LocalTensor<float> wsLocal = workAll[7U * ROW_STRIDE];
DataCopy(wLocal, weightGm_, WEIGHT_FLOATS);
DataCopyPad(biasLocal, biasGm_, MakeCopyExt(1U, FIXED_K * sizeof(float)), MakeNoPad());
Fence<HardEvent::MTE2_S>();
const uint32_t wAligned = AlignUp8(W);
const uint32_t zeroByteOff = (MAX_W + 8U) * 4U;
for (uint32_t j = 0; j < wAligned; ++j) {
uVec.SetValue(j, (j < W) ? U32ToF32(j) : 0.0f);
offL.SetValue(j, (j == 0U || j >= W) ? zeroByteOff : ((j - 1U) * 4U));
offR.SetValue(j, (j + 1U >= W) ? zeroByteOff : ((j + 1U) * 4U));
}
Fence<HardEvent::S_V>();
for (uint32_t hm = blockIdx; hm < totalHeatmaps; hm += usedCores) {
const uint32_t bIdx = hm / FIXED_K;
const uint32_t kIdx = hm % FIXED_K;
float sAcc = NEG_INF_FP32;
float sumAcc = 0.0f;
float uAcc = 0.0f;
float vAcc = 0.0f;
for (uint32_t vRow = 0; vRow < H; ++vRow) {
for (uint32_t c = 0; c < FIXED_CIN; ++c) {
Duplicate(fRows, 0.0f, 3U * ROW_STRIDE);
PipeBarrier<PIPE_V>();
Fence<HardEvent::V_MTE2>();
for (uint32_t dh = 0; dh < 3U; ++dh) {
const int32_t gv = static_cast<int32_t>(vRow) + static_cast<int32_t>(dh) - 1;
if (gv >= 0 && gv < static_cast<int32_t>(H)) {
const uint64_t gmOff =
(static_cast<uint64_t>(bIdx) * FIXED_CIN + c) * hwStride +
static_cast<uint64_t>(gv) * W;
DataCopyPad(fRows[dh * ROW_STRIDE],
    featGm_[gmOff],
    MakeCopyExt(1U, W * sizeof(float)),
    MakeNoPad());
}
}
Fence<HardEvent::MTE2_V>();
if ((c & 7U) == 0U) {
Duplicate(xBlk, 0.0f, ROW_STRIDE);
PipeBarrier<PIPE_V>();
}
const uint32_t wb = (kIdx * FIXED_CIN + c) * 9U;
for (uint32_t dh = 0; dh < 3U; ++dh) {
Gather(shiftTmp, fRows[dh * ROW_STRIDE], offL, 0U, wAligned);
PipeBarrier<PIPE_V>();
Axpy(xBlk, shiftTmp, wLocal.GetValue(wb + dh * 3U + 0U), wAligned);
PipeBarrier<PIPE_V>();
Axpy(xBlk, fRows[dh * ROW_STRIDE], wLocal.GetValue(wb + dh * 3U + 1U), wAligned);
PipeBarrier<PIPE_V>();
Gather(shiftTmp, fRows[dh * ROW_STRIDE], offR, 0U, wAligned);
PipeBarrier<PIPE_V>();
Axpy(xBlk, shiftTmp, wLocal.GetValue(wb + dh * 3U + 2U), wAligned);
PipeBarrier<PIPE_V>();
}
if ((c & 7U) == 7U) {
if (c == 7U) {
Adds(xRow, xBlk, 0.0f, wAligned);
} else {
Add(xRow, xRow, xBlk, wAligned);
}
PipeBarrier<PIPE_V>();
}
}
Fence<HardEvent::V_S>();
for (uint32_t j = W; j < wAligned; ++j) {
xRow.SetValue(j, NEG_INF_FP32);
}
Fence<HardEvent::S_V>();
ReduceMax<float>(redOut[0], xRow, redWork, wAligned, false);
Fence<HardEvent::V_S>();
const float sRow = redOut.GetValue(0);
Fence<HardEvent::S_V>();
Adds(zBuf, xRow, -sRow, wAligned);
PipeBarrier<PIPE_V>();
if (td->exactInvTau != 0U) {
Muls(zBuf, zBuf, td->invTau, wAligned);
} else {
Duplicate(tauVec, td->tau, wAligned);
PipeBarrier<PIPE_V>();
Div(zBuf, zBuf, tauVec, wAligned);
}
PipeBarrier<PIPE_V>();
Exp(eBuf, zBuf, wAligned);
Fence<HardEvent::V_S>();
for (uint32_t j = W; j < wAligned; ++j) {
eBuf.SetValue(j, 0.0f);
}
Fence<HardEvent::S_V>();
ReduceSum<float>(redOut[8], eBuf, redWork, wAligned);
Mul(zBuf, eBuf, uVec, wAligned);
PipeBarrier<PIPE_V>();
ReduceSum<float>(redOut[16], zBuf, redWork, wAligned);
Fence<HardEvent::V_S>();
const float sumRow = redOut.GetValue(8);
const float uRow = redOut.GetValue(16);
const float vVal = U32ToF32(vRow) * sumRow;
if (vRow == 0U) {
sAcc = sRow;
sumAcc = sumRow;
uAcc = uRow;
vAcc = vVal;
} else if (sRow >= sAcc) {
const float scOld = ScalarExpScaled(sAcc - sRow, td, redOut[32]);
sAcc = sRow;
sumAcc = sumAcc * scOld + sumRow;
uAcc = uAcc * scOld + uRow;
vAcc = vAcc * scOld + vVal;
} else {
const float scNew = ScalarExpScaled(sRow - sAcc, td, redOut[32]);
sumAcc += sumRow * scNew;
uAcc += uRow * scNew;
vAcc += vVal * scNew;
}
Fence<HardEvent::S_V>();
}
const float invS = 1.0f / sumAcc;
wsLocal.SetValue(0U, (uAcc * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(1U, (vAcc * invS + 0.5f) * td->strideFloat);
wsLocal.SetValue(8U, sAcc + biasLocal.GetValue(kIdx));
Fence<HardEvent::S_MTE3>();
DataCopyPad(kpGm_[static_cast<uint64_t>(hm) * 2U], wsLocal[0U], MakeCopyExt(1U, 2U * sizeof(float)));
DataCopyPad(scGm_[static_cast<uint64_t>(hm)], wsLocal[8U], MakeCopyExt(1U, sizeof(float)));
Fence<HardEvent::MTE3_S>();
}
}
private:
TPipe pipe_;
GlobalTensor<float> featGm_;
GlobalTensor<float> weightGm_;
GlobalTensor<float> biasGm_;
GlobalTensor<float> kpGm_;
GlobalTensor<float> scGm_;
TBuf<TPosition::VECCALC> inBuf_;
TBuf<TPosition::VECCALC> offsetBuf_;
TBuf<TPosition::VECCALC> convBuf_;
TBuf<TPosition::VECCALC> paramBuf_;
TBuf<TPosition::VECCALC> workBuf_;
};
}
template <typename DT_FEAT, uint32_t KERNEL_PATH>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(GM_ADDR feat,
GM_ADDR weight,
GM_ADDR bias,
GM_ADDR keypoints,
GM_ADDR scores,
GM_ADDR workspace,
GM_ADDR tiling)
{
REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
if constexpr (KERNEL_PATH == 19U) {
FusedKeypointBranches::KernelCase5SingleCore op;
op.Init(feat, weight, bias, keypoints, scores);
op.Process();
} else {
GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingHeader, tiling_data, tiling);
(void)workspace;
if constexpr (KERNEL_PATH == 1U) {
fdksa::KernelExactStrike<1U, 48U, 64U, 8U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 2U) {
fdksa::KernelExactStrike<2U, 48U, 64U, 4U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 3U) {
fdksa::KernelExactStrike<3U, 48U, 64U, 3U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 4U) {
fdksa::KernelExactStrike<4U, 48U, 64U, 2U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 5U) {
fdksa::KernelExactStrike<1U, 64U, 64U, 8U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 6U) {
fdksa::KernelExactStrike<2U, 64U, 64U, 4U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 7U) {
fdksa::KernelExactStrike<3U, 64U, 64U, 2U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 8U) {
fdksa::KernelExactStrike<4U, 64U, 64U, 2U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 9U) {
fdksa::KernelExactStrike<1U, 72U, 96U, 6U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 10U) {
fdksa::KernelExactStrike<2U, 72U, 96U, 4U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 11U) {
fdksa::KernelExactStrike<3U, 72U, 96U, 3U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 12U) {
fdksa::KernelExactStrike<4U, 72U, 96U, 2U> op;
op.Run(feat, weight, bias, keypoints, scores, tiling, &tiling_data);
} else if constexpr (KERNEL_PATH == 13U) {
fdksa::KernelFastGeneralized<64U, 48U, true> op;
op.Run(feat, weight, bias, keypoints, scores, &tiling_data);
} else if constexpr (KERNEL_PATH == 14U) {
fdksa::KernelFastGeneralized<96U, 32U, true> op;
op.Run(feat, weight, bias, keypoints, scores, &tiling_data);
} else if constexpr (KERNEL_PATH == 15U) {
fdksa::KernelFastGeneralized<128U, 20U, true> op;
op.Run(feat, weight, bias, keypoints, scores, &tiling_data);
} else if constexpr (KERNEL_PATH == 16U) {
fdksa::KernelFastGeneralized<96U, 32U, false> op;
op.Run(feat, weight, bias, keypoints, scores, &tiling_data);
} else if constexpr (KERNEL_PATH == 17U) {
fdksa::KernelGeneralAligned op;
op.Run(feat, weight, bias, keypoints, scores, &tiling_data);
} else if constexpr (KERNEL_PATH == 18U) {
fdksa::KernelUniversalFallback op;
op.Run(feat, weight, bias, keypoints, scores, &tiling_data);
} else {
fdksa::KernelBaselinePerHeatmap op;
op.Run(feat, weight, bias, keypoints, scores, &tiling_data);
}
}
}
