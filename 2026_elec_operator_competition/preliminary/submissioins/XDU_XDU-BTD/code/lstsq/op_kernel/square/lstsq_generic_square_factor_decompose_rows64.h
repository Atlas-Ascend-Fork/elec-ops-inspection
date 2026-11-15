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

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::DecomposeRows200(Zq3 base)
{
    PanelLu200::Factor<false>(*this, base);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::DecomposeBlocked(Zq3 base)
{
    BlockedLu256::Factor<false>(*this, base);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::PrepareColumnFactors256(Zq3 base)
{
    BlockedLu256::PrepareColumnFactors(*this, base, 0u, 1u);
}
