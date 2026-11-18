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
struct FusedDisconnectorKeypointSoftArgmaxTilingHeader {
uint32_t B;
uint32_t C;
uint32_t H;
uint32_t W;
uint32_t stripesPerBatch;
uint32_t usedCores;
uint32_t exactInvTau;
float tau;
float invTau;
float strideFloat;
};
struct FusedDisconnectorKeypointSoftArgmaxTilingData {
uint32_t B;
uint32_t C;
uint32_t H;
uint32_t W;
uint32_t stripesPerBatch;
uint32_t usedCores;
uint32_t exactInvTau;
float tau;
float invTau;
float strideFloat;
uint32_t pad[6];
float syncSlots[28 * 16];
};
