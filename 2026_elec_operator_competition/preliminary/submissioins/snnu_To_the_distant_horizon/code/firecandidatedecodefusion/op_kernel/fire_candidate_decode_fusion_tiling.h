/*
 * Copyright (c) 2026 zhiyuan (致远队).
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

// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FireCandidateDecodeFusionTilingData {
    uint32_t bDim;            // Batch size B
    uint32_t nDim;            // Candidates per batch N
    uint32_t totalCandidates; // B * N
    uint32_t coreNum;         // Number of active AIV cores
    uint32_t candsPerCore;    // totalCandidates / coreNum
    uint32_t candsRemainder;  // totalCandidates % coreNum
};
