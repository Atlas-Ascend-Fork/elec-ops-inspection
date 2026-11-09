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
#include <cstddef>
#include <cstdint>

constexpr uint32_t SBG_MAX_LOCAL_ROWS = 512U;
struct alignas(8) SafetyBeltViolationGeometryTilingData {
    struct Slot {
        uint32_t pack;
        float waistRatio;
    };
    static constexpr uint32_t SLOT_BYTES = 8U;
    static constexpr uint32_t SLOT_COUNT = 1U;
    Slot slots[SLOT_COUNT];
};
static_assert(sizeof(SafetyBeltViolationGeometryTilingData) == 8U, "tiling ABI must be eight bytes");
static_assert(offsetof(SafetyBeltViolationGeometryTilingData::Slot, waistRatio) == 4U, "ratio ABI");
