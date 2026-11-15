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
class KernelLarge200 : public FixedSquareLayout<200u, 4u, 256u>, private SquareWorkspaceState, private PrivateSquareStorage {
    friend struct ResidualOperations;
friend struct FixedWorkspace;
friend struct LuFactorizationOperations;
friend struct ProducerResidualOperations;
friend struct TriangularSolveOperations;
friend struct WorkspaceOperations;
friend struct PanelLu200;
static_assert(StaticN == 200u && StaticK == 4u, "This kernel supports its registered shape only");
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);




__aicore__ inline void Process(bool dual200=false);

private:
bool columnArchiveReady_=false;
bool initialForwardReady_=false;
bool pathLocalRhs_=false,pathHasIterate_=false,chainCachesReady_=false,chainSafe_=false;
__aicore__ inline Zq7<float>ProducerResidualInput16(){return factors_[33024u];}
__aicore__ inline Zq7<Zq2>ProducerResidualOffsets16(){return residualTile_;}
__aicore__ inline Zq7<float>ProducerResidualLow16(){return factors_[28896u];}
__aicore__ inline void ProducerPrefetchResidual16(Zq2 row){
    ProducerResidualOperations::ProducerPrefetchResidual16(*this, row);
}
__aicore__ inline void ProducerResidualRowMajor16(Zq3 base,Zq2 row,Zq2 slot){
    ProducerResidualOperations::ProducerResidualRowMajor16(*this, base, row, slot);
}
__aicore__ inline void PrepareChainCaches(Zq3 base);

__aicore__ inline void ProducerRefine16(Zq3 base);


TBuf<TPosition::VECCALC>producerABuf_;
Zq7<float>producerA_;


bool producerPrivate_=false,producerCorrection_=false;







__aicore__ inline float SeedInverse(float diagonal){
return scalar_guard::ReciprocalSeed(diagonal);
}
__aicore__ inline float AbsF(float value){return value<0.0f?-value:value;}


__aicore__ inline Zq2 PivotIndex(Zq7<float>column,Zq2 first);


__aicore__ inline void Load(Zq3 base,Zq2 batch);


__aicore__ inline void DecomposeSquare(Zq3 base){
    LuFactorizationOperations::DecomposeSquare(*this, base);
}

__aicore__ inline void DecomposeLocal(Zq3 base){


{
DecomposeRows200(base);ArchiveRows200(base);return;
}


}


__aicore__ inline void LoadRows200(Zq3 base,Zq2 batch){
    WorkspaceOperations::LoadRows200(*this, base, batch);
}


__aicore__ inline void DecomposeRows200(Zq3 base);

__aicore__ inline void ArchiveRows200(Zq3 base);


__aicore__ inline void ColumnTriangularBatch(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width);

__aicore__ inline void ColumnTriangularAll(Zq3 base,const __gm__ float*in,__gm__ float*out,Zq2 width){
for(Zq2 col=0;col<width;col+=4u)
ColumnTriangularBatch(base,in+col,out+col,width,width-col<4u?width-col:4u);
}


__aicore__ inline void Refine(Zq3 base){
{ProducerRefine16(base);return;}


}


__aicore__ inline Zq7<float>ResidualWork(){
return N()==200u?factors_:block_;
}


__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col);


__aicore__ inline void EmitSplitRhsDma(Zq3 base,Zq2 batch);




__aicore__ inline void SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width){
    TriangularSolveOperations::SolveInPlaceWithFactors(*this, base, rhs, width);
}


};


#include "lstsq_square200_transport.h"
#include "lstsq_square200_compute.h"
