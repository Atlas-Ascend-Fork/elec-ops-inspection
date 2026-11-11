/*
 * Copyright (c) 2026 aodebiao.
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
struct SplineConvTilingData {
    uint32_t n, e, cIn, cOut;
    uint32_t dim, degree, basisSlots, aggrMean;
    uint32_t kernel0, kernel1, kernel2;
    uint32_t open0, open1, open2;
    uint64_t accStrideFloats;
};
