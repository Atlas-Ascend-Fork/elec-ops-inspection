/*
 * Copyright (c) 2026 ceprei_Tnx.
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

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Fixed key9 UB: stage/out, two FP32 arrays, bias pairs and a byte-offset map.
constexpr uint32_t SWIGLU_TINY_RAW_ELEMENTS = 64U * 64U;
constexpr uint32_t SWIGLU_TINY_RAW_UB_BYTES = 16U * SWIGLU_TINY_RAW_ELEMENTS + 768U;

#pragma pack(push, 8)
// v-lean: the seven scalars the cube paths consume sit first and the struct
// is 8-byte packed, so kernel entry touches one 32-byte block per core and
// the compiler can pair the loads. Field names (and therefore every other
// reader) are unchanged; host and kernel rebuild together.
struct SwiGluLeanSlot {
    uint64_t totalOutputs;
    uint64_t blocksPerCore;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t extraBlocks;
    uint32_t usedCoreNum;
    uint32_t vecChunkElements;
};
static_assert(sizeof(SwiGluLeanSlot) == 40U, "lean slot must stay 40B");

struct SwiGluLeanSlotPad {
    SwiGluLeanSlot slot;
    uint8_t pad[512U - sizeof(SwiGluLeanSlot)];
};

struct SwiGluTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t vecChunkElements;
    uint32_t blockDim;
    uint32_t cvTaskM;
    uint32_t cvTaskN;
    uint32_t reserved; // key 4: bit7 branch BK128, bit8 WP BK128; zero keeps BK64.
    uint64_t totalOutputs;
    uint64_t blocksPerCore;
    uint32_t extraBlocks;
    uint32_t usedCoreNum;
    TCubeTiling cubeTilingData;
    SwiGluLeanSlotPad leanSlots[8];
};
#pragma pack(pop)
