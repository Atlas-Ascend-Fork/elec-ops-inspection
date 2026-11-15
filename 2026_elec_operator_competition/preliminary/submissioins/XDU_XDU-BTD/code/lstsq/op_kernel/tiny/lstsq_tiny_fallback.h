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
class KernelTiny : public FixedSquareLayout<TinyN, 1u, 16u>, private SquareWorkspaceState, private KernelScalarOperations {
    friend struct TinyFallbackOperations;
friend struct FixedWorkspace;
friend struct ColumnTriangularSolveOperations;
friend struct RefinementOperations;
friend struct ResidualOperations;
friend struct TriangularSolveOperations;
static_assert(TinyN == 15u || TinyN == 16u, "This kernel supports its registered shape only");
public:
using Layout = FixedSquareLayout<TinyN, 1u, 16u>;
using Layout::M;
using Layout::N;
using Layout::K;
using Layout::FactorPitch;
using Layout::VectorPitch;
using Layout::PerBatch;
using Layout::OffA;
using Layout::OffLU;
using Layout::OffB;
using Layout::OffX;
using Layout::OffR;
using Layout::OffD;
using Layout::OffP;
using Layout::OffInv;

__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);






__aicore__ inline void ProcessOneCold16(Zq2 batch)
{
    static_assert(TinyN == 16u);
    currentRhs_ = 0u;
    ProcessOneMatrix(batch);
}

__aicore__ inline void ProcessOneMatrix(Zq2 batch, bool forcePivoted = false)
{
    const Zq3 base = Zq3(batch)*PerBatch();
    Load(base, batch);
    DecomposeSquare(base, forcePivoted);
    PrepareTinyVectorSolve(base);
    ColumnTriangularAll(base, w_+OffB(base), w_+OffX(base), 1u);
    Refine(base);
    EmitSquareDma(base, batch);
}

private:

TBuf<TPosition::VECCALC>tinyAugmentedBuf_;
bool tinyInverseReady_=false;
__aicore__ inline bool BuildTinyInverseFast(Zq3 base);
__aicore__ inline bool BuildTinyInverse(Zq3 base);
__aicore__ inline void ConsumeTinyInverse(const __gm__ float*in,Zq2 stride);

__aicore__ inline void EmitTinyPrivate(Zq2 batch);
__aicore__ inline void RefineTinyPrivate(Zq3 base){
    RefinementOperations::RefineTinyPrivate(*this, base);
}
__aicore__ inline void PrepareTinyVectorSolve(Zq3 base);
__aicore__ inline void TinyVectorTriangular(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride);
TBuf<TPosition::VECCALC>tinyNegativeBuf_;
Zq7<float>tinyNegative_;
bool tinyVectorSafe_=false,tinyLuDiagonalKnown_=false,tinyLuSafe_=false,tinyNegativeReady_=false;
bool tinyPrivateFlow_=false,tinyLocalRhs_=false;






__aicore__ inline void Load(Zq3 base,Zq2 batch);


__aicore__ inline void SetX(Zq3 base,Zq2 row,Zq2 col,float value){
w_[OffX(base)+Zq3(row)*K()+col]=value;
}
__aicore__ inline float GetX(Zq3 base,Zq2 row,Zq2 col)const{
return w_[OffX(base)+Zq3(row)*K()+col];
}


__aicore__ inline void DecomposeSquare(Zq3 base,bool forcePivoted);


__aicore__ inline void DecomposeSmallMasked(Zq3 base);


__aicore__ inline void TinyTriangular(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride);

__aicore__ inline void ColumnTriangularBatch(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width){
    ColumnTriangularSolveOperations::ColumnTriangularBatch(*this, base, in, out, stride, width);
}

__aicore__ inline void ColumnTriangularAll(Zq3 base,const __gm__ float*in,__gm__ float*out,Zq2 width){
for(Zq2 col=0;col<width;col+=4u)
ColumnTriangularBatch(base,in+col,out+col,width,width-col<4u?width-col:4u);
}


__aicore__ inline void TinyPrefetch(Zq2 baseRow,Zq2 slot){
    ResidualOperations::TinyPrefetch(*this, baseRow, slot);
}

__aicore__ inline void TinyResidualTile(Zq3 base,Zq2 row,Zq2 slot);

__aicore__ inline void RefineTinyRows(Zq3 base){
    RefinementOperations::RefineTinyRows(*this, base);
}

__aicore__ inline void Refine(Zq3 base){
    RefinementOperations::Refine(*this, base);
}


__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col){
    ResidualOperations::PrepareResidualX(*this, base, col);
}


__aicore__ inline void EmitSquareDma(Zq3 base,Zq2 batch);


__aicore__ inline void SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width){
    TriangularSolveOperations::SolveInPlaceWithFactors(*this, base, rhs, width);
}

__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr,*w_=nullptr;
TBuf<TPosition::VECCALC>arenaBuf_;
TPipe pipe_;
LinalgLstsqTilingData tiling_;
};


template<Zq2 TinyN>
using KernelTinyAReuse15 = KernelTiny<TinyN>;

#include "lstsq_tiny_fallback_compute.h"
#include "lstsq_tiny_fallback_transport.h"
