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

template<Zq2 LogicalN,bool ColumnFactored256>
__aicore__ inline void VectorBounded256::FollowFactors200(GM_ADDR workspace,Zq2 batches)
{
    static_assert(LogicalN == 256u && ColumnFactored256);
    CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
    FollowFactorPanels();
    Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(4);
    CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();
    Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
    CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
    CrossCoreWaitFlag(5);Zq1<PIPE_ALL>();SyncAll<true>();
    CrossCoreSetFlag<2,PIPE_MTE3>(6);CrossCoreWaitFlag(7);Zq1<PIPE_ALL>();SyncAll<true>();

    Zq1<PIPE_ALL>();SyncAll<true>();
    const bool wide = SelectInitialDegree(workspace, batches);
    Zq1<PIPE_ALL>();SyncAll<true>();
    if (wide) {
        FollowWidePolynomial(workspace, batches);
    } else {
        CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();
    }
    Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(14);
    CrossCoreWaitFlag(15);Zq1<PIPE_ALL>();SyncAll<true>();
}

__aicore__ inline void VectorBounded256::FollowFactorPanels()
{
constexpr Zq2 factorN = 256u;
PrepareRetiredInverseTranspose256();
for(Zq2 j=0u;j<factorN;j+=32u){
if(j>0u){
CrossCoreSetFlag<2,PIPE_MTE3>(2);


StageRetiredInverseTranspose256(j/32u-1u);
CrossCoreWaitFlag(1);
}
Zq1<PIPE_ALL>();SyncAll<true>();
CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag(1);


if(j+32u==factorN)StageRetiredInverseTranspose256(j/32u);
Zq1<PIPE_ALL>();SyncAll<true>();
}
}

__aicore__ inline bool VectorBounded256::SelectInitialDegree(GM_ADDR workspace, Zq2 batches)
{
    bool wide = false;
    cubicPolynomial_ = true;
    pooledLightMask_ = 0u;
    reducedH4Eligible_ = true;
    for (Zq2 batch = 0u; batch < batches; ++batch) {
        const float value = ReadQrError(workspace, batch, all_);
        const bool light = value >= 0.0f && value < 0.0009765625f;
        reducedH4Eligible_ = reducedH4Eligible_ && value >= 0.0f && value <= 4.0f;
        if (light) pooledLightMask_ |= 1u << batch;
        wide = wide || !light;
        cubicPolynomial_ = cubicPolynomial_ && value >= 0.0f && value < 0.03125f;
    }
    pooledLightMask_ = RestrictPoolMask(pooledLightMask_, batches);
    return wide;
}

__aicore__ inline void VectorBounded256::FollowWidePolynomial(GM_ADDR workspace, Zq2 batches)
{
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(8);
for(Zq2 phase=0u;phase<4u;++phase){
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(13);Zq1<PIPE_ALL>();SyncAll<true>();
Zq1<PIPE_ALL>();SyncAll<true>();
if(phase==0u){SelectActualE2(workspace,batches,false);}


if((phase==0u&&(quadraticPolynomial_||reducedH4_||pooledLightMask_==0u))||(phase==2u&&quarticPolynomial_)){Zq1<PIPE_ALL>();SyncAll<true>();}
CrossCoreSetFlag<2,PIPE_MTE3>(3);
if((quadraticPolynomial_&&phase==0u)||(cubicPolynomial_&&phase==1u)||(quarticPolynomial_&&phase==(reducedH4_?1u:2u)))break;
}
CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();
}
