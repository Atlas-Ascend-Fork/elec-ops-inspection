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

template<Zq2 TinyN>
class KernelTinyFastContext15 : private SquareWorkspaceState {
static_assert(TinyN == 15u, "This kernel supports its registered shape only");
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t);


__aicore__ inline bool ProcessFastSingle(Zq2 batch);


private:

TBuf<TPosition::VECCALC>tinyAugmentedBuf_;
bool tinyFusedResidual_=false;
bool vectorPivotUsed_=false;

__aicore__ __attribute__((noinline))void RestorePristine15(Zq2 batch);

__aicore__ inline bool BuildTinyInverseFast(Zq2 batch){
vectorPivotUsed_=false;
if(BuildTinyInverseVector15()){vectorPivotUsed_=true;return true;}
Zq1<PIPE_ALL>();
RestorePristine15(batch);
return BuildTinyInverseLegacy15();
}

__aicore__ inline bool BuildTinyInverseVector15();

__aicore__ __attribute__((noinline))bool BuildTinyInverseLegacy15();

__aicore__ inline void ConsumeTinyRawInverse15();

__aicore__ inline void ConsumeTinyInverse();

__aicore__ inline void EmitTinyPrivate(Zq2 batch);
__aicore__ __attribute__((noinline))bool RefineLegacy15(){return RefineTinyPrivate<false>();}

template<bool Direct>
__aicore__ inline bool RefineTinyPrivate();


TBuf<TPosition::VECCALC>tinyNegativeBuf_;
Zq7<float>tinyNegative_;


__aicore__ inline float SeedInverse(float diagonal){
if(diagonal==0.0f)return scalar_guard::SignedInfinity(diagonal);
const float seed=1.0f/diagonal;
return seed;
}


__aicore__ inline void LoadDirect15(Zq2 batch);

__aicore__ inline void Load(Zq2 batch);


template<bool Direct>
__aicore__ inline void TinyResidualTile();


__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr;
TBuf<TPosition::VECCALC>arenaBuf_;
TPipe pipe_;
LinalgLstsqTilingData tiling_;
};


#include "lstsq_tiny15_fast_residual.h"

#include "lstsq_tiny15_fast_transport.h"
#include "lstsq_tiny15_fast_compute.h"
