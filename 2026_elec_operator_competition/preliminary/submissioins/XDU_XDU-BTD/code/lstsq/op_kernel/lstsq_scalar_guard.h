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

#include <cstdint>

namespace lstsq_detail {
namespace scalar_guard {
// A zero divisor makes the compensated quotient invalid. Preserve that
// signal without executing either of the two floating-point divisions.
__aicore__ inline float QuietNaN()
{
    union { uint32_t bits; float value; } result;
    result.bits = 0x7fc00000u;
    return result.value;
}

// Reciprocal seeds use IEEE-754 signed infinity for either signed zero.
// Existing pivot/range checks and fallback decisions remain at the callers.
__aicore__ inline float SignedInfinity(float zero)
{
    union { uint32_t bits; float value; } result;
    result.value = zero;
    result.bits = (result.bits & 0x80000000u) | 0x7f800000u;
    return result.value;
}
__aicore__ inline float ReciprocalSeed(float diagonal)
{
    if(diagonal==0.0f)return SignedInfinity(diagonal);
    volatile float seed=1.0f/diagonal;
    return seed;
}

}
}
