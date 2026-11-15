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

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::DecomposeBlocked(Zq3 base)
{
    BlockedLu256::Factor<true>(*this, base);
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::PrepareColumnFactors256(Zq3 base,Zq2 lane,Zq2 participants)
{
    BlockedLu256::PrepareColumnFactors(*this, base, lane, participants);
}
