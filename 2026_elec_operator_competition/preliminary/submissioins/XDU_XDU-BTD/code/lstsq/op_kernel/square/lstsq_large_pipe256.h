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

template<Zq2 StaticN=0u,Zq2 StaticK=0u,bool EnableBoundedH3=false,bool CompactConsumer200=false,bool ColumnFactored256=false>
class KernelLargePipe256 : public FixedSquareLayout<256u, 3u, 256u>, private SquareWorkspaceState {
    friend struct ResidualOperations;
friend struct FixedWorkspace;
friend struct TriangularSolveOperations;
friend struct WorkspaceOperations;
friend struct ColumnTriangularSolveOperations;
friend struct BlockedLu256;
static_assert(StaticN == 256u && StaticK == 3u, "This kernel supports its registered shape only");
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t,TPipe*externalPipe,bool compactQr32=false,bool dual200=false);




__aicore__ inline void Process();


public:
__aicore__ inline bool QrUniformDomainBatchRead256(Zq2 batches,Zq2 batch,Zq2 rhs,Zq3 shared,Zq2 logicalN=256u,Zq2 rhsWidth=3u);


__aicore__ inline bool QrUniformDomainBatchRead200(Zq2 batches,Zq2 batch,Zq2 rhs,Zq3 shared);
__aicore__ inline bool QrUniformDomain(Zq2 batches,Zq2 batch,Zq2 rhs,Zq3 shared,Zq2 logicalN=256u,Zq2 rhsWidth=3u);


__aicore__ inline void QrCubeDecision(bool accepted,Zq3 shared);
__aicore__ inline void QrCubeApply(Zq3 shared,bool active,bool residualPublished=false,bool prepareEmit=false,bool prepareResidualMap=false);


__aicore__ inline void QrCoopCacheB(Zq3 shared);
__aicore__ inline void QrResidualRows(Zq2 begin,Zq2 end);
__aicore__ inline void QrCoopResidual(Zq3 shared,bool active);

__aicore__ inline void QrDualResidual256(Zq3 shared,bool owner);
__aicore__ inline void QrDualResidual200(Zq3 shared,bool owner);
__aicore__ inline void PrepareEmitIndex256();
__aicore__ inline void EmitQrCoalesced256(Zq2 batch,bool active);
__aicore__ inline void EmitQrFinalFrames256(Zq2 batch,bool active);
__aicore__ inline void EmitQrFinalFrames200(Zq2 batch,bool active);
__aicore__ inline void EmitQrCoalesced200(Zq2 batch,bool active);
__aicore__ inline void EmitQrVectorDma(Zq2 batch);
__aicore__ inline void ProcessQrCubeRhs(Zq2 batch,Zq2 rhs,Zq3 shared);
private:
__aicore__ inline void SolveCooperativeTask(Zq3 base, Zq2 batch);


bool dual200_=false;float qrInverseScale_=1.0f;Zq2 qrDegree_=1u;Zq2 qrResidualEnd_=256u,qrResidualBOffset_=512u;Zq2 qrLogicalN_=256u,qrRhsWidth_=3u;


private:
bool pathLocalRhs_=false,pathHasIterate_=false,chainCachesReady_=false,chainSafe_=false;
bool chainShared_=false;Zq3 chainSharedBase_=0u;bool compactQr32_=false;
Zq2 compactInputSlot_=0u,cachedNextRow_=200u,cachedEnd_=200u;
__aicore__ inline Zq7<float>ProducerResidualInput16(){
if constexpr(CompactConsumer200)return residualTile_.template Zq6<float>()[compactInputSlot_*6656u];
else return residualTile_.template Zq6<float>();
}
__aicore__ inline Zq7<Zq2>ProducerResidualOffsets16(){return panel_.template Zq6<Zq2>();}
__aicore__ inline Zq7<float>ProducerResidualLow16(){return factors_;}
__aicore__ inline void ProducerPrefetchResidual16(Zq2 row);
__aicore__ inline void ProducerResidualRowMajor16(Zq3 base,Zq2 row,Zq2 slot);
template<Zq2 q,Zq2 scratch,Zq2 firstHigh>
__aicore__ inline Zq7<float> PackResidualPlanes(Zq7<float> work, Zq7<float> lo);
template<bool Compact200,Zq2 TileRows,bool KeepInput>
__aicore__ inline Zq7<float> PackResidualRows(Zq7<float> work, Zq7<float> a, Zq7<float> lo);
template<bool Compact200,Zq2 TileRows=32u,bool KeepInput=false>
__aicore__ inline void ProducerResidualRowMajor32Impl(Zq2 row);

__aicore__ inline void ProducerResidualRowMajor32(Zq2 row){
if(qrLogicalN_==200u)ProducerResidualRowMajor32Impl<true>(row);
else ProducerResidualRowMajor32Impl<false>(row);
}
__aicore__ inline void PrefetchCompact200(Zq2 row,Zq2 end,Zq2 slot){
auto input=residualTile_.template Zq6<float>()[slot*6656u];
const Zq2 count=end-row;
ZqB<Zq5::V_MTE2>();Duplicate<float>(input[200u],0.0f,Zq3(8u),Zq4(count),Zq4(1u),Zq4(26u));ZqB<Zq5::V_MTE2>();
GlobalTensor<float>original;original.SetGlobalBuffer(a_+(Zq3(currentBatch_)*200u+row)*200u,Zq3(count)*200u);
DataCopyPad(input,original,Zq8{uint16_t(count),800u,0u,1u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(int32_t(slot));
}
__aicore__ inline void QrResidualRowsDouble200(Zq2 begin,Zq2 end);
__aicore__ inline void QrResidualRows32(Zq2 begin,Zq2 end);
__aicore__ inline void PrepareChainCaches(Zq3 base);

__aicore__ inline void ProducerRefine16(Zq3 base);



__aicore__ inline bool UseLocalFactors()const{
return false;
}
__aicore__ inline bool UseColumnLocal()const{
return false;
}
__aicore__ inline bool UseVectorSolve()const{
return true;
}



__aicore__ inline float SeedInverse(float diagonal){
return scalar_guard::ReciprocalSeed(diagonal);
}
__aicore__ inline float AbsF(float value){return value<0.0f?-value:value;}


__aicore__ inline Zq2 PivotIndex(Zq7<float>column,Zq2 first);

__aicore__ inline float Factor(Zq3 base,Zq2 row,Zq2 col)const{
return w_[OffLU(base)+Zq3(row)*N()+col];
}

__aicore__ inline void Load(Zq3 base,Zq2 batch);


__aicore__ inline void DecomposeSquare(Zq3 base){

{DecomposeBlocked(base);return;}

}


__aicore__ inline void CooperativeLu256(Zq3 base,Zq3 sharedBase,Zq3 stageBase,Zq2 lane);


__aicore__ inline void DecomposeBlocked(Zq3 base);


__aicore__ inline float TriangularDot(Zq3 base,Zq2 row);

__aicore__ inline void VectorTriangularColumn(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth){
    TriangularSolveOperations::VectorTriangularColumn(*this, base, in, inWidth, out, outWidth);
}


__aicore__ inline void PrepareColumnFactors256(Zq3 base,Zq2 lane=0u,Zq2 participants=1u);


__aicore__ inline void ChainVectorTriangular(Zq3 base);

__aicore__ inline void ColumnGmTriangular256(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width);


__aicore__ inline void VectorTriangularBatch(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width){
const Zq2 n=N(),p=VectorPitch();
{
ColumnGmTriangular256(base,in,inWidth,out,outWidth,width);return;
}


}


__aicore__ inline void Refine(Zq3 base){
{ProducerRefine16(base);return;}


}


__aicore__ inline Zq7<float>ResidualWork(){
return N()==200u?factors_:block_;
}


__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col);


__aicore__ inline void EmitSplitRhsDma(Zq3 base,Zq2 batch);




__aicore__ inline void SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width);

__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr,*w_=nullptr;
TPipe*pipe_;
TBuf<TPosition::VECCALC>negative64Buf_;
Zq7<float>negative64_;
bool negative64Ready_=false;
LinalgLstsqTilingData tiling_;
};

#include "lstsq_square256_dispatch.h"
#include "lstsq_square256_factor_qr_uniform_domain_batch_read256.h"
#include "lstsq_square256_factor_decompose_blocked.h"


#include "lstsq_square256_residual_qr_residual_rows.h"
#include "lstsq_square256_residual_producer_residual_row_major32_impl.h"
#include "lstsq_square256_solve.h"
#include "lstsq_square256_transport.h"
