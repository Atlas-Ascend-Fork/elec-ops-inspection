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
class KernelFastContext64 : public FixedSquareLayout<64u, 2u, 64u>, private PackedWorkspaceState {
    friend struct WorkspaceOperations;
friend struct ResidualOperations;
static_assert(StaticN == 64u && StaticK == 2u, "This kernel supports its registered shape only");
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);





__aicore__ inline bool TryTask64(Zq2 task);
__aicore__ inline void PreparePrivateResidualX64(){

auto x=vr_[64u],xh=vr_[192u],xl=vr_[320u];
Adds(x,negative64_[4096u],0.0f,64u);Zq1<PIPE_V>();
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
vr_[ResidualMaskPlane()*64u].template Zq6<uint16_t>(),128u);Zq1<PIPE_V>();
Sub(xl,x,xh,64u);Zq1<PIPE_V>();
}
__aicore__ inline bool RefineAccepted64(Zq3 base);

private:

bool augmented64Ready_=false;bool rowInverse64Ready_=false;
bool compactPpt64Ready_=false;
float compactNormalization64_=1.0f;
__aicore__ inline void TransposeOriginalA64();
__aicore__ inline void PrepareCompactPivotMap64();
__aicore__ inline void EliminateCompactPivots64(Zq7<float> aug);
__aicore__ inline bool ValidateCompactInverse64(Zq7<float> aug, float inverseScale, Zq2 scaleExponent);
__aicore__ __attribute__((noinline))bool BuildCompactPpt64();
__aicore__ inline void ConsumeAugmented64(const __gm__ float*in,Zq2 stride);


Zq7<float>producerA_;

__aicore__ inline void FeatureResidual64(bool cachedHigh);
__aicore__ inline void RowMajorResidual64All64(Zq3 base,bool cachedHigh);

bool producerPrivate_=false,producerCorrection_=false,replicaOffsetsReady_=false;









__aicore__ inline float GetX(Zq3 base,Zq2 row,Zq2 col)const{
return w_[OffX(base)+Zq3(row)*K()+col];
}


__aicore__ inline void PrefetchInputA64(Zq2 batch){
negative64Ready_=false;
GlobalTensor<float>input;input.SetGlobalBuffer(a_+Zq3(batch)*4096u,4096u);


Zq1<PIPE_V>();ZqB<Zq5::V_MTE2>();
DataCopyPad(producerA_,input,Zq8{64u,256u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,0u,0.0f});
// A ready0 is consumed exactly once at its first Vector reader.
SetFlag<Zq5::MTE2_V>(0);

ZqB<Zq5::V_S>();


}


__aicore__ inline void EmitSplitRhsAtomic64(Zq3 base,Zq2 batch);

__aicore__ inline void EmitSplitRhsDma(Zq3 base,Zq2 batch);



TBuf<TPosition::VECCALC> packed64Buf_;
};


#include "lstsq_fast64_transport.h"
#include "lstsq_fast64_compute.h"
