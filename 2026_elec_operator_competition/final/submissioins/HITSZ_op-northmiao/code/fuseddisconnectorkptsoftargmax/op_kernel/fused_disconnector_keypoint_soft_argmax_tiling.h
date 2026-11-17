/*
 * Copyright (c) 2026 HITSZ_op-northmiao.
 * SPDX-License-Identifier: Apache-2.0
 *
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

struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t blocks;
    uint32_t tileRows;
    uint32_t tiles;
    float tau;
    float stride;
    uint32_t hideCase;
};
