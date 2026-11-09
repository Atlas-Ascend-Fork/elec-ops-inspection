/*
 * Copyright (c) 2026 dihidao.
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
// Private device implementation; included by ld_device.h.
#if (defined(__DAV_C220_VEC__))
template <class ClassPlan46> class CyclicG5PriorityRelay {
    #include "../../declarations/ld_common_declarations_02.inc"
#include "ld_cyclicg5_priority_relay_m01.h"

  public:
#include "ld_cyclicg5_priority_relay_m02.h"

  private:
    using LdetPolicy = LdetPriorityPolicy54<ClassPlan46, PREFIX_STREAM>;
};
#endif
