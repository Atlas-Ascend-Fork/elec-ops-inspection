// Copyright 2026 XDU_XDU-BTD Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once
#ifndef LSTSQ_TILING_DATA_ONCE
#define LSTSQ_TILING_DATA_ONCE

#include <cstdint>

struct LinalgLstsqTilingData{
uint32_t m;
uint32_t n;
uint32_t k;
uint32_t batches;
uint32_t pitch;
uint64_t residualElements;
};
#endif
