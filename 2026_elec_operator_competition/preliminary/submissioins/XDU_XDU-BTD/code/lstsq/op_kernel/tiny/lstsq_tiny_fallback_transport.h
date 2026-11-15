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

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::EmitTinyPrivate(Zq2 batch)
{
auto iterate=tinyNegative_[272u];output_.SetValue(0u,0.0f);
ZqB<Zq5::S_MTE3>();ZqB<Zq5::V_MTE3>();
GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*TinyN,TinyN);
DataCopyPad(gx,iterate,Zq8{1u,TinyN*4u,0u,0u,0u});
if(r_!=nullptr&&Zq3(batch)<tiling_.residualElements){
GlobalTensor<float>gr;gr.SetGlobalBuffer(r_+batch,1u);
DataCopyPad(gr,output_,Zq8{1u,4u,0u,0u,0u});
}
ZqB<Zq5::MTE3_S>();
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::EmitSquareDma(Zq3 base,Zq2 batch)
{
    TinyFallbackOperations::EmitSquareDma(*this, base, batch);
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
    FixedWorkspace::InitTinyFallback<TinyN>(*this, t);
    FixedWorkspace::BindAddresses(*this, a, b, x, res, workspace);
}

template<Zq2 TinyN>
__aicore__ inline void KernelTiny<TinyN>::Load(Zq3 base,Zq2 batch)
{
tinyInverseReady_=false;currentBatch_=batch;auto input=panel_[14u*256u];
GlobalTensor<float>original;original.SetGlobalBuffer(a_+Zq3(batch)*TinyN*TinyN,TinyN*TinyN);


ZqB<Zq5::V_MTE2>();
if constexpr (TinyN == 15u) {
    Duplicate(input[240u],0.0f,16u);ZqB<Zq5::V_MTE2>();
    DataCopyPad(input,original,Zq8{15u,60u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,1u,0.0f});
} else {
    DataCopyPad(input,original,Zq8{1u,1024u,0u,0u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
}
GlobalTensor<float>originalB;originalB.SetGlobalBuffer(b_+Zq3(batch)*TinyN,TinyN);

DataCopyPad(tinyNegative_,originalB,Zq8{1u,TinyN*4u,0u,0u,0u},DataCopyPadExtParams<float>{TinyN==15u,0u,Zq4(16u-TinyN),0.0f});
ZqB<Zq5::MTE2_V>();ZqB<Zq5::S_V>();


}
