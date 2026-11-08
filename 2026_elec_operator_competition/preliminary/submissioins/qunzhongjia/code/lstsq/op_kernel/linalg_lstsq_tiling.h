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

#pragma once
#include <cstdint>

struct LinalgLstsqTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t batches;
    uint32_t mp;
    uint32_t groupSize;
    uint32_t groups;
    uint32_t localCols;
    uint32_t rounds;
};
