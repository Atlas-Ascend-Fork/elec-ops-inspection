/*
 * Copyright (c) 2026 dihidao.
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
// Private device implementation; included by ld_device.h.
constexpr uint32_t kN256MatrixBytes = 256U * 256U * sizeof(float);

constexpr uint32_t kN256WorkspaceBytes = (2U * 256U * 256U + 512U * 24U + 32U + 256U) * sizeof(float);

constexpr uint32_t kN256GroupBytes = kN256WorkspaceBytes + 32768U + 65536U + 27U * 64U;

#if (defined(__DAV_C220_VEC__))
namespace {
constexpr uint32_t kAlign = 8;
constexpr uint32_t kSmallMaxN = 208;
constexpr uint32_t kPanel = 24;
constexpr uint32_t kTileRows = 8;
constexpr uint32_t kUbW = 512;
constexpr uint32_t kMaxN = 512;
constexpr uint32_t kMaxPerCore = 256;
constexpr uint32_t kSlabMax = 128;
constexpr uint32_t kTileRowsPar = 8;
constexpr uint32_t kBarSlots = 16;
constexpr uint32_t kBarElems = kBarSlots * 16;
__aicore__ inline uint32_t TagLo(uint32_t k) {
    return 0x9E3779B9u ^ (0xA5A5A5A5u + k * 0x7F4A7C15u);
}
__aicore__ inline uint32_t TagHi(uint32_t k) {
    return 0x85EBCA6Bu ^ (0x3C6EF372u + k * 0xD1B54A35u);
}
__aicore__ inline uint32_t T2Lo(uint32_t k) {
    return 0xC2B2AE35u ^ (0x165667B1u + k * 0x27D4EB2Fu);
}
__aicore__ inline uint32_t T2Hi(uint32_t k) {
    return 0xD6E8FEB8u ^ (0x9E3779B9u + k * 0x85EBCA6Bu);
}
__aicore__ inline uint32_t PreLo() {
    return T2Lo(0xFFFFFFFFu);
}
__aicore__ inline uint32_t PreHi() {
    return T2Hi(0xFFFFFFFFu);
}
constexpr uint32_t kSpMat = 0;
constexpr uint32_t kSpCol = kSpMat + kSmallMaxN * kSmallMaxN * sizeof(float);
constexpr uint32_t kSpAbs = kSpCol + kSmallMaxN * sizeof(float);
constexpr uint32_t kSpRowk = kSpAbs + kSmallMaxN * sizeof(float);
constexpr uint32_t kSpTab = kSpRowk + kSmallMaxN * sizeof(float);
constexpr uint32_t kSpColoff = kSpTab + kSmallMaxN * sizeof(float);
constexpr uint32_t kSpTmp = kSpColoff + kSmallMaxN * sizeof(float);
constexpr uint32_t kSpRes = kSpTmp + 256 * sizeof(float);
constexpr uint32_t kSpTabR = kSpRes + 256 * sizeof(float);
constexpr uint32_t kSpBrc = kSpTabR + 8 * kSmallMaxN * sizeof(float);
static_assert(kSpBrc + kSmallMaxN * 8 * sizeof(float) <= 196608, "Small-path broadcast buffer exceeds UB");
constexpr uint32_t kLpPanel = 0;
constexpr uint32_t kLpLbuf = kLpPanel + kMaxN * kPanel * sizeof(float);
constexpr uint32_t kLpUbuf = kLpLbuf + kMaxN * kPanel * sizeof(float);
constexpr uint32_t kLpTile = kLpUbuf + kPanel * kUbW * sizeof(float);
constexpr uint32_t kLpTmp = kLpTile + 2 * kTileRows * kUbW * sizeof(float);
constexpr uint32_t kLpCp = kLpTmp + 256 * sizeof(float);
constexpr uint32_t kLpRes = kLpCp + 4 * kMaxN * sizeof(float);
constexpr uint32_t kLpTab = kLpRes + kMaxPerCore * sizeof(float);
constexpr uint32_t kLpColoff = kLpTab + kMaxN * sizeof(float);
constexpr uint32_t kLpMeta = kLpColoff + kMaxN * sizeof(float);
constexpr int32_t kEvMte2S = 0;
constexpr int32_t kEvMte2V = 1;
constexpr int32_t kEvVS = 2;
constexpr int32_t kEvSV = 3;
constexpr int32_t kEvMte2Mte3 = 4;
constexpr int32_t kEvSMte2 = 4;
constexpr int32_t kEvMte3Mte2 = 5;
constexpr int32_t kEvVMte3 = 6;
constexpr int32_t kEvVMte2 = 7;
constexpr float kTwoPow64 = 18446744073709551616.0f;
constexpr float kTwoPowNeg64 = 5.42101086242752217004e-20f;
constexpr float kTwoPow60 = 1152921504606846976.0f;
constexpr float kTwoPowNeg63 = 1.08420217248550443401e-19f;
}
#endif
