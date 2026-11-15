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

template<Zq2 LogicalN>
__aicore__ inline void VectorNormalize200::FollowFactors200(GM_ADDR workspace,Zq2 batches,Zq2 ownBatch)
{
constexpr Zq2 factorN=LogicalN==200u?224u:256u;
auto values=all_;
CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
if constexpr(LogicalN!=200u)PrepareEarlyR64Indices200();
if constexpr(LogicalN==200u){
for(Zq2 j=0u;j<factorN;j+=32u){
if(j>0u){CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();}
if(j==0u){Zq1<PIPE_ALL>();SyncAll<true>();}
CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
if(j+32u<factorN)CrossCoreSetFlag<2,PIPE_MTE3>(2);
}
}else{
for(Zq2 j=0u;j<factorN;j+=32u){
StageEarlyR64Seed200(workspace,ownBatch,j);
Zq1<PIPE_ALL>();SyncAll<true>();
CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
if(j+32u<factorN){CrossCoreSetFlag<2,PIPE_MTE3>(2);CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();}
}
}
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(4);
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(5);Zq1<PIPE_ALL>();SyncAll<true>();
CrossCoreSetFlag<2,PIPE_MTE3>(6);CrossCoreWaitFlag(7);Zq1<PIPE_ALL>();SyncAll<true>();

Zq1<PIPE_ALL>();SyncAll<true>();bool wide=false;
for(Zq2 batch=0u;batch<batches;++batch){GlobalTensor<float>lines;
lines.SetGlobalBuffer(Workspace(workspace,batch)+344320u,64u);
DataCopy(values,lines,64u);ZqB<Zq5::MTE2_S>();float value=values.GetValue(0u);
for(Zq2 lane=1u;lane<4u;++lane)value+=values.GetValue(lane*16u);
wide=wide||!(value>=0.0f&&value<0.0009765625f);
}
Zq1<PIPE_ALL>();SyncAll<true>();
if(!wide){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();}
else{
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(8);
for(Zq2 phase=0u;phase<4u;++phase){
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(13);Zq1<PIPE_ALL>();SyncAll<true>();
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(3);
}
CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();
}
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(14);
if constexpr(LogicalN==256u){
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();

Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
}
CrossCoreWaitFlag(15);Zq1<PIPE_ALL>();SyncAll<true>();
}
