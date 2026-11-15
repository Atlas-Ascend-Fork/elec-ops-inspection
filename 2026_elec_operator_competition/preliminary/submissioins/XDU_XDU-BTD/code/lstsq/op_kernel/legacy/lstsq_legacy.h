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

class KernelLegacy{
friend struct TriangularSolveOperations;
friend struct RefinementOperations;
friend struct RectangularSolveOperations;
friend struct WorkspaceOperations;
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);


__aicore__ inline Zq3 PerBatch()const{
const Zq3 m=tiling_.m,n=tiling_.n,k=tiling_.k;
const Zq3 count=2*m*n+m*k+3*Zq3(n)*k+2*n;
return(count+15u)/16u*16u;
}
__aicore__ inline Zq3 OffA(Zq3 base)const{return base;}
__aicore__ inline Zq3 OffLU(Zq3 base)const{return base+Zq3(tiling_.m)*tiling_.n;}
__aicore__ inline Zq3 OffB(Zq3 base)const{
return OffLU(base)+Zq3(tiling_.m)*tiling_.n;
}
__aicore__ inline Zq3 OffX(Zq3 base)const{return OffB(base)+Zq3(tiling_.m)*tiling_.k;}
__aicore__ inline Zq3 OffR(Zq3 base)const{return OffX(base)+Zq3(tiling_.n)*tiling_.k;}
__aicore__ inline Zq3 OffD(Zq3 base)const{return OffR(base)+Zq3(tiling_.n)*tiling_.k;}
__aicore__ inline Zq3 OffP(Zq3 base)const{return OffD(base)+Zq3(tiling_.n)*tiling_.k;}

__aicore__ inline Zq3 OffInv(Zq3 base)const{return OffP(base)+tiling_.n;}

__aicore__ inline void Process();

private:
__aicore__ inline float AbsF(float value){return value<0.0f?-value:value;}

__aicore__ inline float Factor(Zq3 base,Zq2 row,Zq2 col)const{
const Zq3 index=Zq3(row)*tiling_.n+col;
return localFactors_?factors_.GetValue(Zq3(row)*factorPitch_+col):w_[OffLU(base)+index];
}
__aicore__ inline void PutFactor(Zq3 base,Zq2 row,Zq2 col,float value){
const Zq3 index=Zq3(row)*tiling_.n+col;
if(localFactors_)factors_.SetValue(Zq3(row)*factorPitch_+col,value);
else w_[OffLU(base)+index]=value;
}
__aicore__ inline void Load(Zq3 base,Zq2 batch);


__aicore__ inline void SetX(Zq3 base,Zq2 row,Zq2 col,float value){
w_[OffX(base)+Zq3(row)*tiling_.k+col]=value;
}
__aicore__ inline float GetX(Zq3 base,Zq2 row,Zq2 col)const{
return w_[OffX(base)+Zq3(row)*tiling_.k+col];
}


__aicore__ inline void DecomposeSquare(Zq3 base);

__aicore__ inline void DecomposeLocal(Zq3 base);


__aicore__ inline void ProcessSquare(Zq3 base, Zq2 batch);
__aicore__ inline void ProcessRectangular(Zq3 base, Zq2 batch);
__aicore__ inline void FactorBlockedPanel(Zq3 base, __gm__ float* lu, __gm__ float* piv, Zq2 jb, Zq2 je);
__aicore__ inline void SolveBlockedPanelTail(__gm__ float* lu, Zq2 jb, Zq2 je);
__aicore__ inline void UpdateBlockedTrailingTiles(__gm__ float* lu, Zq2 jb, Zq2 je);
__aicore__ inline void DecomposeBlocked(Zq3 base);


__aicore__ inline void BackSubstitute(Zq3 base,Zq2 col);


__aicore__ inline float TriangularDot(Zq3 base,Zq2 row);

__aicore__ inline void VectorTriangularColumn(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth);


__aicore__ inline void BatchedTriangularDots(Zq2 row,Zq2 width);

__aicore__ inline float BatchedDotValue(Zq2 col){
const Zq2 p=vectorPitch_;float result=0.0f;
for(Zq2 lane=0;lane<8u;++lane)
result=square_math::RoundAdd(result,vr_[(6u+col)*p].GetValue(lane));
return result;
}

__aicore__ inline void VectorTriangularBatch(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width);


__aicore__ inline void Refine(Zq3 base);

__aicore__ inline void PrefetchResidualRow(Zq2 row,Zq2 slot);

__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col);

__aicore__ inline float VectorResidual(Zq3 base,Zq2 row,Zq2 col,Zq2 slot);


__aicore__ inline void RefineRectangular(Zq3 base,Zq2 batch);


__aicore__ inline void EmitSquareDma(Zq3 base,Zq2 batch);

__aicore__ inline void Emit(Zq3 base,Zq2 batch,Zq2 col,bool withResidual);


__aicore__ inline void Qr(Zq3 base);

__aicore__ inline void TriangularSolveQr(Zq3 base,Zq2 col);


__aicore__ inline void SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width);

__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr,*w_=nullptr;
TPipe pipe_;
TBuf<TPosition::VECCALC>factorBuf_,rowBuf_,blockBuf_;
Zq7<float>factors_,rows_,block_;
Zq2 factorPitch_=0u;
bool localFactors_=false;
TBuf<TPosition::VECCALC>solveBuf_;
Zq7<float>solveY_;
bool vectorSolve_=false;
TBuf<TPosition::VECCALC>outputBuf_;
TBuf<TPosition::VECCALC>residualBuf_;
Zq7<float>output_,vr_;
bool vectorResidual_=false;
Zq2 vectorPitch_=8u,currentBatch_=0u;
bool dmaOutputs_=false;
LinalgLstsqTilingData tiling_;
};


#include "lstsq_legacy_factor.h"


#include "lstsq_legacy_transport.h"
#include "lstsq_legacy_compute.h"
