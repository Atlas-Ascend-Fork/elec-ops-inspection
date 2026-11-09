/*
 * Copyright (c) 2026 李冠杰.
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
    uint32_t batch;
    uint32_t candidates;
    uint32_t total;
    uint32_t blockFactor;
    uint32_t tileLength;
};
