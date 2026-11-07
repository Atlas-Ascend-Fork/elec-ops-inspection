/*
 * Copyright (c) 2026 lasth0pe.
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
template<uint32_t KEY> struct EighCase;
template<> struct EighCase<1u> {
  static constexpr uint32_t N=16u, B=1u, U=0u, P=1u, Solver=0u;
};
template<> struct EighCase<2u> {
  static constexpr uint32_t N=64u, B=4u, U=0u, P=8u, Solver=2u;
};
template<> struct EighCase<3u> {
  static constexpr uint32_t N=256u, B=4u, U=0u, P=8u, Solver=3u;
};
template<> struct EighCase<4u> {
  static constexpr uint32_t N=15u, B=1u, U=0u, P=1u, Solver=0u;
};
template<> struct EighCase<5u> {
  static constexpr uint32_t N=50u, B=3u, U=0u, P=8u, Solver=2u;
};
template<> struct EighCase<6u> {
  static constexpr uint32_t N=200u, B=3u, U=0u, P=10u, Solver=3u;
};
