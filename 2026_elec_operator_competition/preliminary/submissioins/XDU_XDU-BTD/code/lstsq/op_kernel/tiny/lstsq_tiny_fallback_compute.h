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

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::TinyResidualTile(Zq3 base,Zq2 row,Zq2 slot)
{
    TinyFallbackOperations::TinyResidualTile<TinyN>(*this, base, row, slot);
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::ConsumeTinyInverse(const __gm__ float*in,Zq2 stride)
{
    TinyFallbackOperations::ConsumeTinyInverse<TinyN>(*this, in, stride);
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::PrepareTinyVectorSolve(Zq3 base)
{
    TinyFallbackOperations::PrepareTinyVectorSolve<TinyN>(*this, base);
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::TinyVectorTriangular(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride)
{
    TinyFallbackOperations::TinyVectorTriangular<TinyN>(*this, base, in, out, stride);
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::TinyTriangular(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride)
{
    TinyFallbackOperations::TinyTriangular<TinyN>(*this, base, in, out, stride);
}

template<Zq2 TinyN>
__aicore__ inline bool KernelTiny<TinyN>::BuildTinyInverseFast(Zq3 base)
{
    return TinyFallbackOperations::BuildTinyInverseFast<TinyN>(*this, base);
}

template<Zq2 TinyN>
__aicore__ inline bool KernelTiny<TinyN>::BuildTinyInverse(Zq3 base)
{
    return TinyFallbackOperations::BuildTinyInverse<TinyN>(*this, base);
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::DecomposeSquare(Zq3 base,bool forcePivoted)
{
if constexpr (TinyN == 15u) {
    if (!forcePivoted && (BuildTinyInverseFast(base) || BuildTinyInverse(base))) return;
} else {
    if (BuildTinyInverse(base)) return;
}

Load(base,currentBatch_);
Gather(factors_,panel_[14u*256u],panel_[16u*256u].template Zq6<Zq2>(),0u,256u);ZqB<Zq5::V_S>();
for(Zq2 i=0;i<TinyN;++i)w_[OffB(base)+i]=b_[Zq3(currentBatch_)*TinyN+i];
DecomposeSmallMasked(base);


}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::DecomposeSmallMasked(Zq3 base)
{
    TinyFallbackOperations::DecomposeSmallMasked<TinyN>(*this, base);
}
