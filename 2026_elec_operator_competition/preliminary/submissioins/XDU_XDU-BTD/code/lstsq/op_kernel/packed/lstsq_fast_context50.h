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

template<Zq2 StaticN=0u,Zq2 StaticK=0u>
class KernelFastContext50 : public FixedSquareLayout<50u, 2u, 64u>, private PackedWorkspaceState {
friend struct WorkspaceOperations;
friend struct ResidualOperations;
static_assert(StaticN == 50u && StaticK == 2u, "This kernel supports its registered shape only");
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);





__aicore__ inline bool TryTask50(Zq2 task);
__aicore__ inline void RefineAccepted50(Zq3 base){
SetMaskNorm();ResetMask();
Adds(negative64_[4096u],solveY_,0.0f,64u);Zq1<PIPE_V>();
PrepareResidualX(base,0u);
RowMajorResidual50All64(base);

producerCorrection_=true;ConsumeAugmented50(w_+OffD(base),1u);
Add(negative64_[4096u],negative64_[4096u],solveY_,64u);Zq1<PIPE_V>();
}

private:

bool directRows50_=false;bool augmented50Ready_=false;bool augmented50RhsReady_=false;
float normalizationScale_=1.0f,inverseNormalizationScale_=1.0f;
__aicore__ inline bool BuildAugmented50();
__aicore__ inline void CanonicalizeRowMajor50();
__aicore__ inline void ConsumeAugmented50(const __gm__ float*in,Zq2 stride);


bool pathLocalRhs_=false,pathHasIterate_=false;


Zq7<float>producerA_;


__aicore__ inline void RowMajorResidual50All64(Zq3 base);


bool producerPrivate_=false,producerCorrection_=false,replicaOffsetsReady_=false;









__aicore__ inline float SeedInverse(float diagonal){
if(diagonal==0.0f)return scalar_guard::SignedInfinity(diagonal);
const float seed=1.0f/diagonal;
return seed;
}
__aicore__ inline float AbsF(float value){return value<0.0f?-value:value;}


__aicore__ inline float GetX(Zq3 base,Zq2 row,Zq2 col)const{
return w_[OffX(base)+Zq3(row)*K()+col];
}


__aicore__ inline void LoadRows50(Zq2 batch){
negative64Ready_=false;
GlobalTensor<float>input;input.SetGlobalBuffer(a_+Zq3(batch)*2500u,2500u);


auto workingA=panel_[22528u];
Zq1<PIPE_V>();Duplicate(producerA_,0.0f,4096u);Duplicate(workingA,0.0f,4608u);ZqB<Zq5::V_MTE2>();
DataCopyPad(workingA,input,Zq8{50u,200u,0u,2u,0u},DataCopyPadExtParams<float>{true,0u,6u,0.0f});


}


__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col);


__aicore__ inline void EmitSplitRhsAtomic50(Zq3 base,Zq2 batch,Zq2 firstRow);

__aicore__ inline void EmitSplitRhsDma(Zq3 base,Zq2 batch);



TBuf<TPosition::VECCALC> packed50Buf_;
};


#include "lstsq_fast50_transport.h"
#include "lstsq_fast50_compute.h"
