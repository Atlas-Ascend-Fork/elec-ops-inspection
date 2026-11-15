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

template<Zq2 StaticN=0u,Zq2 StaticK=0u>
class KernelLarge : private SquareWorkspaceState {
static_assert((StaticN == 0u && (StaticK == 0u || StaticK == 2u)) ||
    (StaticN == 50u && StaticK == 2u), "Unsupported generic entry configuration");
friend struct RefinementOperations;
friend struct RectangularSolveOperations;
friend struct LuFactorizationOperations;
friend struct ProducerResidualOperations;
friend struct TriangularSolveOperations;
friend struct WorkspaceOperations;
friend struct PanelLu200;
friend struct ResidualOperations;
friend struct ColumnTriangularSolveOperations;
friend struct BlockedLu256;
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);


__aicore__ inline Zq3 PerBatch()const{
const Zq3 m=M(),n=N(),k=K();
const Zq3 count=2*m*n+m*k+3*Zq3(n)*k+2*n;
return(count+15u)/16u*16u;
}
__aicore__ inline Zq3 OffA(Zq3 base)const{return base;}
__aicore__ inline Zq3 OffLU(Zq3 base)const{return base+Zq3(M())*N();}
__aicore__ inline Zq3 OffB(Zq3 base)const{
return OffLU(base)+Zq3(M())*N();
}
__aicore__ inline Zq3 OffX(Zq3 base)const{return OffB(base)+Zq3(M())*K();}
__aicore__ inline Zq3 OffR(Zq3 base)const{return OffX(base)+Zq3(N())*K();}
__aicore__ inline Zq3 OffD(Zq3 base)const{return OffR(base)+Zq3(N())*K();}
__aicore__ inline Zq3 OffP(Zq3 base)const{return OffD(base)+Zq3(N())*K();}

__aicore__ inline Zq3 OffInv(Zq3 base)const{return OffP(base)+N();}

__aicore__ inline void Process();

__aicore__ inline void ProcessOneColdTask50(Zq2 task){
static_assert(StaticN==50u&&StaticK==2u,"Cold50only");
currentRhs_=task%2u;const Zq2 batch=task/2u;
const Zq3 base=Zq3(task)*PerBatch();
Load(base,batch);DecomposeLocal(base,true);
ColumnTriangularAll(base,w_+OffB(base),w_+OffX(base),K());
Refine(base);EmitSquareDma(base,batch);
}

private:

bool augmented50Ready_=false;bool augmented50RhsReady_=false;
__aicore__ inline bool BuildAugmented50();
__aicore__ inline void ConsumeAugmented50(const __gm__ float*in,Zq2 stride);


bool pathLocalRhs_=false,pathHasIterate_=false;
__aicore__ inline Zq7<float>ProducerResidualInput16(){return factors_[33024u];}
__aicore__ inline Zq7<Zq2>ProducerResidualOffsets16(){return residualTile_;}
__aicore__ inline Zq7<float>ProducerResidualLow16(){return factors_[28896u];}
__aicore__ inline void ProducerPrefetchResidual16(Zq2 row){
    ProducerResidualOperations::ProducerPrefetchResidual16(*this, row);
}
__aicore__ inline void ProducerResidualRowMajor16(Zq3 base,Zq2 row,Zq2 slot){
    ProducerResidualOperations::ProducerResidualRowMajor16(*this, base, row, slot);
}
__aicore__ inline Zq2 FusedLogicalPivot(Zq7<float>column,
Zq7<Zq2>map,Zq7<float>logical,Zq2 first){
    return LuFactorizationOperations::FusedLogicalPivot(*this, column, map, logical, first);
}

__aicore__ inline void ProducerLazySmallLu(Zq3 base);
TBuf<TPosition::VECCALC>producerABuf_;
Zq7<float>producerA_;

__aicore__ inline void WideSmallResidualSingle64(Zq3 base,Zq2 row,Zq2 slot);


__aicore__ inline void RowMajorResidual50All64(Zq3 base);
__aicore__ inline void RefinePrivate50RowMajor64(Zq3 base);

__aicore__ inline void WideRefineSmallSingle64(Zq3 base);
bool producerPrivate_=false,producerCorrection_=false,replicaOffsetsReady_=false;


__aicore__ inline Zq2 ResidualMaskPlane()const{
return M()==N()&&N()==256u?8u:11u;
}
__aicore__ inline Zq2 ResidualZeroPlane()const{
return M()==N()&&N()==256u?9u:12u;
}

__aicore__ inline Zq2 M()const{
if constexpr(StaticN)return StaticN;else return tiling_.m;
}
__aicore__ inline Zq2 N()const{
if constexpr(StaticN)return StaticN;else return tiling_.n;
}
__aicore__ inline Zq2 K()const{
if constexpr(StaticN)return(StaticN==50u||StaticN==64u||StaticN==200u||StaticN==256u)?1u:StaticK;
else return tiling_.k;
}
__aicore__ inline Zq2 FactorPitch()const{
if constexpr(StaticN)return(StaticN+7u)/8u*8u;else return factorPitch_;
}
__aicore__ inline Zq2 VectorPitch()const{
if constexpr(StaticN)return StaticN<=64u?64u:256u;else return vectorPitch_;
}
__aicore__ inline Zq2 RhsWidth()const{
if constexpr(StaticN)return StaticK;else return rhsWidth_;
}
__aicore__ inline bool UseLocalFactors()const{
if constexpr(StaticN)return StaticN<=200u;else return localFactors_;
}
__aicore__ inline bool UseColumnLocal()const{
if constexpr(StaticN)return StaticN<=200u;else return columnLocal_;
}
__aicore__ inline bool UseVectorSolve()const{
if constexpr(StaticN)return true;else return vectorSolve_;
}

__aicore__ inline bool UseDmaOutputs()const{
if constexpr(StaticN)return true;else return dmaOutputs_;
}
__aicore__ inline bool UseSplitRhs()const{
if constexpr(StaticN)return StaticN==50u||StaticN==64u||StaticN==200u||StaticN==256u;else return splitRhs_;
}


__aicore__ inline float SeedInverse(float diagonal){
return scalar_guard::ReciprocalSeed(diagonal);
}
__aicore__ inline float AbsF(float value){return value<0.0f?-value:value;}


__aicore__ inline Zq2 PivotIndex(Zq7<float>column,Zq2 first);

__aicore__ inline float Factor(Zq3 base,Zq2 row,Zq2 col)const{
const Zq3 index=Zq3(row)*N()+col;
return UseLocalFactors()?factors_.GetValue(UseColumnLocal()?Zq3(col)*FactorPitch()+row:
Zq3(row)*FactorPitch()+col):w_[OffLU(base)+index];
}
__aicore__ inline void PutFactor(Zq3 base,Zq2 row,Zq2 col,float value){
const Zq3 index=Zq3(row)*N()+col;
if(UseLocalFactors())factors_.SetValue(UseColumnLocal()?Zq3(col)*FactorPitch()+row:
Zq3(row)*FactorPitch()+col,value);
else w_[OffLU(base)+index]=value;
}
__aicore__ inline void Load(Zq3 base,Zq2 batch);


__aicore__ inline void SetX(Zq3 base,Zq2 row,Zq2 col,float value){
w_[OffX(base)+Zq3(row)*K()+col]=value;
}
__aicore__ inline float GetX(Zq3 base,Zq2 row,Zq2 col)const{
return w_[OffX(base)+Zq3(row)*K()+col];
}


__aicore__ inline void DecomposeSquare(Zq3 base);

__aicore__ inline void DecomposeLocal(Zq3 base,bool skipTried50=false);


__aicore__ inline bool SelectColumnPivot(
    Zq3 base, Zq2 j, Zq2 n, Zq2 pitch, float& diagonal, float& inverse, bool& safe);
__aicore__ inline void DecomposeSmallMasked(Zq3 base);

__aicore__ inline void DecomposeColumns(Zq3 base);


__aicore__ inline void LoadRows200(Zq3 base,Zq2 batch){
    WorkspaceOperations::LoadRows200(*this, base, batch);
}


__aicore__ inline void LoadRows50(Zq2 batch){
negative64Ready_=false;
GlobalTensor<float>input;input.SetGlobalBuffer(a_+Zq3(batch)*2500u,2500u);


Zq1<PIPE_V>();Duplicate(producerA_,0.0f,4096u);ZqB<Zq5::V_MTE2>();
DataCopyPad(producerA_,input,Zq8{50u,200u,0u,1u,0u},DataCopyPadExtParams<float>{true,0u,6u,0.0f});
ZqB<Zq5::MTE2_V>();
}

__aicore__ inline void PrepareFallbackRows50(){

Duplicate(factors_,0.0f,3136u);Zq1<PIPE_V>();
Adds(factors_,producerA_,0.0f,Zq3(56),Zq4(50),UnaryRepeatParams{1,1,7,8});
ZqB<Zq5::V_S>();
auto offsets=vr_[10u*64u].template Zq6<Zq2>();
for(Zq2 row=0;row<56u;++row)offsets.SetValue(row,row*56u*4u);
}

__aicore__ inline void Rows50ToColumns();


__aicore__ inline void DecomposeRows200(Zq3 base);

__aicore__ inline void ArchiveRows200(Zq3 base){
GlobalTensor<float>target;target.SetGlobalBuffer(w_+OffLU(base),40000u);
DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(target);
ZqB<Zq5::S_MTE3>();ZqB<Zq5::V_MTE3>();
DataCopy(target,factors_,40000u);ZqB<Zq5::MTE3_S>();ZqB<Zq5::S_V>();
LoadBlockedColumns200(base);
}


__aicore__ inline void LoadBlockedColumns200(Zq3 base);


__aicore__ inline void DecomposeBlocked(Zq3 base);


__aicore__ inline void BackSubstitute(Zq3 base,Zq2 col);


__aicore__ inline float TriangularDot(Zq3 base,Zq2 row);

__aicore__ inline void VectorTriangularColumn(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth){
    TriangularSolveOperations::VectorTriangularColumn(*this, base, in, inWidth, out, outWidth);
}


__aicore__ inline void PrepareColumnFactors256(Zq3 base);


__aicore__ inline void ColumnGmTriangular256(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width);


__aicore__ inline void BatchedTriangularDots(Zq3 base,Zq2 row,Zq2 width);

__aicore__ inline float BatchedDotValue(Zq2 col){
const Zq2 p=VectorPitch();float result=0.0f;
for(Zq2 lane=0;lane<8u;++lane)
result=square_math::RoundAdd(result,vr_[(6u+col)*p].GetValue(lane));
return result;
}

__aicore__ inline void VectorTriangularBatch(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width);


__aicore__ inline void VectorOnlyTriangular50(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width);


__aicore__ inline void ColumnTriangularBatch(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width);

__aicore__ inline void ColumnTriangularAll(Zq3 base,const __gm__ float*in,__gm__ float*out,Zq2 width){
for(Zq2 col=0;col<width;col+=4u)
ColumnTriangularBatch(base,in+col,out+col,width,width-col<4u?width-col:4u);
}


__aicore__ inline void SmallPrefetch(Zq2 baseRow,Zq2 slot);

__aicore__ inline void SmallResidualTile(Zq3 base,Zq2 row,Zq2 slot);

__aicore__ inline void RefineSmallRows(Zq3 base);

__aicore__ inline void SmallResidualSingle64(Zq3 base,Zq2 row,Zq2 slot);

__aicore__ inline void RefineSmallSingle64(Zq3 base);

__aicore__ inline void Refine(Zq3 base);


__aicore__ inline Zq7<float>ResidualWork(){
return N()==200u?factors_:block_;
}
__aicore__ inline Zq7<float>ResidualInput(Zq2 slot){
if(N()==200u)return factors_[(K()==1u?20800u:20480u)+slot*2048u];
return slot?residualTile_[2048u].template Zq6<float>():panel_;
}
__aicore__ inline Zq7<Zq2>ResidualOffsets(){
return N()==200u?factors_[K()==1u?24960u:24576u].template Zq6<Zq2>():residualTile_;
}
__aicore__ inline void PrefetchEightRows(Zq2 row,Zq2 slot);


__aicore__ inline Zq7<float>ResidualPairExtra(){
return N()==200u?factors_[26624u]:factors_;
}

__aicore__ inline void ResidualEightRows(Zq3 base,Zq2 row,Zq2 col,
Zq2 groupWidth,Zq2 slot);


__aicore__ inline void ResidualRowMajorEight(Zq3 base,Zq2 row,Zq2 slot);

__aicore__ inline void RefineEightRows(Zq3 base);

__aicore__ inline void PrefetchResidualRow(Zq2 row,Zq2 slot);

__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col);

__aicore__ inline float VectorResidual(Zq3 base,Zq2 row,Zq2 col,Zq2 slot);


__aicore__ inline void RefineRectangular(Zq3 base,Zq2 batch);


__aicore__ inline void EmitSplitRhsAtomic50(Zq3 base,Zq2 batch,Zq2 firstRow);

__aicore__ inline void EmitSplitRhsDma(Zq3 base,Zq2 batch);

__aicore__ inline void EmitSquareDma(Zq3 base,Zq2 batch);

__aicore__ inline void Emit(Zq3 base,Zq2 batch,Zq2 col,bool withResidual);


__aicore__ inline void Qr(Zq3 base);

__aicore__ inline void TriangularSolveQr(Zq3 base,Zq2 col);


__aicore__ inline void SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width);

__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr,*w_=nullptr;
TPipe pipe_;
TBuf<TPosition::VECCALC>negative64Buf_;
Zq7<float>negative64_;
bool negative64Ready_=false;
bool diagonalKnown_=false,diagonalSafe_=false;
LinalgLstsqTilingData tiling_;
};


#include "lstsq_generic_square_factor_build_augmented50.h"
#include "lstsq_generic_square_factor_decompose_rows64.h"

#include "lstsq_generic_square_input.h"

#include "lstsq_generic_square_residual_producer_refine16.h"
#include "lstsq_generic_square_residual_refine_small_single64.h"
#include "lstsq_generic_square_solve_consume_augmented50.h"

#include "lstsq_generic_square_transport.h"
#include "lstsq_generic_square_compute.h"
