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

template<Zq2 StaticN, Zq2 StaticK, bool ColdTask = false>
class KernelLarge64 : public FixedSquareLayout<64u, 2u, 64u>, private SquareWorkspaceState, private PrivateSquareStorage, private KernelScalarOperations {
    friend struct ResidualOperations;
friend struct FixedWorkspace;
friend struct ColumnTriangularSolveOperations;
friend struct LuFactorizationOperations;
friend struct ProducerResidualOperations;
friend struct RefinementOperations;
friend struct TriangularSolveOperations;
friend struct WorkspaceOperations;
static_assert(StaticN == 64u && StaticK == 2u, "This kernel supports its registered shape only");
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);




__aicore__ inline void Process();

__aicore__ inline void ProcessOneColdTask64(Zq2 task){
static_assert(ColdTask, "This entry processes one cold task");
currentRhs_=task%2u;const Zq2 batch=task/2u;
const Zq3 base=Zq3(task)*PerBatch();
Load(base,batch);DecomposeLocal(base,true);
ColumnTriangularAll(base,w_+OffB(base),w_+OffX(base),K());
Refine(base);WorkspaceOperations::EmitSquareDma(*this,base,batch);
}

private:

bool augmented64Ready_=false;
bool rowInverse64Ready_=false;
__aicore__ inline bool BuildAugmented64Fast();
__aicore__ inline void EliminateAugmented64Fast(
    Zq7<float> aug, Zq7<Zq2> zeros, Zq7<Zq2> rowOffsets, Zq7<float> dvec);
__aicore__ inline void PrepareRowInverseMap64();

__aicore__ inline bool BuildAugmented64(){
    return WorkspaceOperations::BuildAugmented64(*this);
}
__aicore__ inline void ConsumeAugmented64(const __gm__ float*in,Zq2 stride);


__aicore__ inline Zq2 FusedLogicalPivot(Zq7<float>column,
Zq7<Zq2>map,Zq7<float>logical,Zq2 first){
    return LuFactorizationOperations::FusedLogicalPivot(*this, column, map, logical, first);
}

__aicore__ inline void ProducerLazySmallLu(Zq3 base){
    ProducerResidualOperations::ProducerLazySmallLu(*this, base);
}
TBuf<TPosition::VECCALC>producerABuf_;
Zq7<float>producerA_;

__aicore__ inline void WideSmallResidualSingle64(Zq3 base,Zq2 row,Zq2 slot,bool cachedA=false);
__aicore__ inline void WideRefineSmallSingle64(Zq3 base);
__aicore__ inline void RowMajorResidual64All64(Zq3 base, bool cachedHigh);
__aicore__ inline void PrepareRefinementOffsets64();
__aicore__ inline void PrepareColdResidualOffsets64();
__aicore__ inline void ComputeWideResidual64(Zq3 base, Zq2 step);
__aicore__ inline bool ApplyRefinementStep64(Zq3 base, Zq2 step);
bool producerPrivate_=false,producerCorrection_=false,replicaOffsetsReady_=false;










__aicore__ inline void Load(Zq3 base,Zq2 batch);


__aicore__ inline void SetX(Zq3 base,Zq2 row,Zq2 col,float value){
w_[OffX(base)+Zq3(row)*K()+col]=value;
}
__aicore__ inline float GetX(Zq3 base,Zq2 row,Zq2 col)const{
return w_[OffX(base)+Zq3(row)*K()+col];
}


__aicore__ inline void DecomposeSquare(Zq3 base){
    LuFactorizationOperations::DecomposeSquare(*this, base);
}

__aicore__ inline void DecomposeLocal(Zq3 base, bool skipTriedFast = false){

{if((!skipTriedFast&&BuildAugmented64Fast())||BuildAugmented64())return;LoadRows64(currentBatch_);ZqB<Zq5::S_V>();Duplicate(factors_,0.0f,4608u);Zq1<PIPE_V>();
Adds(factors_,producerA_,0.0f,Zq3(64),Zq4(64),UnaryRepeatParams{1,1,9,8});ZqB<Zq5::V_S>();for(Zq2 i=0;i<64u;++i)w_[OffB(base)+i]=negative64_[4224u].GetValue(i);Rows64ToColumns();ProducerLazySmallLu(base);return;}


}


__aicore__ inline void LoadRows64(Zq2 batch);

__aicore__ inline void Rows64ToColumns(){
    WorkspaceOperations::Rows64ToColumns(*this);
}


__aicore__ inline void VectorOnlyTriangular64(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride){
    ColumnTriangularSolveOperations::VectorOnlyTriangular64(*this, base, in, out, stride);
}

__aicore__ inline void ColumnTriangularBatch(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width);

__aicore__ inline void ColumnTriangularAll(Zq3 base,const __gm__ float*in,__gm__ float*out,Zq2 width){
for(Zq2 col=0;col<width;col+=4u)
ColumnTriangularBatch(base,in+col,out+col,width,width-col<4u?width-col:4u);
}


__aicore__ inline void RefineSmallSingle64(Zq3 base){
    RefinementOperations::RefineSmallSingle64(*this, base);
}

__aicore__ inline void Refine(Zq3 base){
{RefineSmallSingle64(base);return;}


}


__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col);


__aicore__ inline void EmitSplitRhsAtomic64(Zq3 base,Zq2 batch);
__aicore__ inline void EmitSplitRhsDma(Zq3 base,Zq2 batch);




__aicore__ inline void SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width){
    TriangularSolveOperations::SolveInPlaceWithFactors(*this, base, rhs, width);
}


};


template<Zq2 StaticN, Zq2 StaticK>
using KernelLarge64Next = KernelLarge64<StaticN, StaticK, true>;

#include "lstsq_square64_transport.h"
#include "lstsq_square64_compute.h"
