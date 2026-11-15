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

// Shared by the shape-specific kernels. Each kernel owns its workspace and synchronization state.
struct RefinementOperations {
template<class Kernel>
__aicore__ static inline void Refine(Kernel &kernel, Zq3 base)
{
{kernel.RefineTinyRows(base);return;}


}

template<class Kernel>
__aicore__ static inline void RefineSmallSingle64(Kernel &kernel, Zq3 base)
{
{kernel.WideRefineSmallSingle64(base);return;}


}

template<class Kernel>
__aicore__ static inline void RefineTinyPrivate(Kernel &kernel, Zq3 base)
{
constexpr Zq2 p=256u,pitch=16u;auto iterate=kernel.tinyNegative_[272u];
Duplicate(kernel.panel_[10u*p].template Zq6<Zq2>(),Zq2(0xfffff000u),p);
Zq1<PIPE_V>();
Adds(iterate,kernel.solveY_,0.0f,16u);Zq1<PIPE_V>();
for(Zq2 step=0;step<2u;++step){
PrepareTinyResidualColumns(kernel, base, pitch, p);
kernel.TinyResidualTile(base,0u,0u);
kernel.tinyLocalRhs_=true;kernel.SolveInPlaceWithFactors(base,kernel.w_+kernel.OffD(base),1u);
Add(iterate,iterate,kernel.solveY_,16u);Zq1<PIPE_V>();
}
}

template<class Kernel>
__aicore__ static inline void PrepareTinyResidualColumns(Kernel &kernel, Zq3 base, Zq2 pitch, Zq2 p)
{
for(Zq2 rhs=0;rhs<1u;++rhs){
kernel.PrepareResidualX(base,rhs);
auto x=rhs?kernel.panel_[10u*p]:kernel.panel_[p];
auto xh=rhs?kernel.panel_[11u*p]:kernel.panel_[3u*p];
auto xl=rhs?kernel.panel_[12u*p]:kernel.panel_[5u*p];
Brcb(x,kernel.vr_[pitch],Zq4(2),Zq9{2,16});
Brcb(x[8u],kernel.vr_[pitch],Zq4(2),Zq9{2,16});
Brcb(xh,kernel.vr_[3u*pitch],Zq4(2),Zq9{2,16});
Brcb(xh[8u],kernel.vr_[3u*pitch],Zq4(2),Zq9{2,16});
Brcb(xl,kernel.vr_[5u*pitch],Zq4(2),Zq9{2,16});
Brcb(xl[8u],kernel.vr_[5u*pitch],Zq4(2),Zq9{2,16});Zq1<PIPE_V>();
}
}

template<class Kernel>
__aicore__ static inline void RefineTinyRows(Kernel &kernel, Zq3 base)
{
if(kernel.tinyPrivateFlow_){kernel.RefineTinyPrivate(base);return;}
constexpr Zq2 p=256u,pitch=16u;
auto offsets=kernel.panel_[16u*p].template Zq6<Zq2>();
for(Zq2 feature=0;feature<pitch;++feature)for(Zq2 lane=0;lane<16u;++lane)
offsets.SetValue(feature*16u+lane,(lane*pitch+feature)*4u);
ZqB<Zq5::S_V>();
Duplicate(kernel.panel_[10u*p].template Zq6<Zq2>(),Zq2(0xfffff000u),p);
Duplicate(kernel.panel_[14u*p],0.0f,2u*p);Zq1<PIPE_V>();
for(Zq2 step=0;step<2u;++step){
PrepareTinyResidualColumns(kernel, base, pitch, p);
kernel.TinyPrefetch(0u,0u);
for(Zq2 row=0;row<kernel.N();row+=16u){
const Zq2 slot=(row/16u)&1u;WaitFlag<Zq5::MTE2_V>(slot);
if(row+16u<kernel.N())kernel.TinyPrefetch(row+16u,slot^1u);
kernel.TinyResidualTile(base,row,slot);
}
kernel.SolveInPlaceWithFactors(base,kernel.w_+kernel.OffD(base),1u);
for(Zq2 row=0;row<kernel.N();++row)for(Zq2 col=0;col<1u;++col)
kernel.SetX(base,row,col,kernel.GetX(base,row,col)+kernel.w_[kernel.OffD(base)+Zq3(row)+col]);
}
}

template<class Kernel>
__aicore__ static inline void RefineSquareRows(Kernel& kernel, Zq3 base, Zq2 n, Zq2 k)
{
__gm__ float*dx=kernel.w_+kernel.OffD(base);
for(Zq2 step=0;step<2u;++step){
for(Zq2 col=0;col<k;++col){
kernel.PrepareResidualX(base,col);
kernel.PrefetchResidualRow(0u,0u);
for(Zq2 row=0;row<n;++row){
const Zq2 slot=row&1u;
WaitFlag<Zq5::MTE2_V>(slot);
if(row+1u<n)kernel.PrefetchResidualRow(row+1u,slot^1u);
dx[Zq3(row)*k+col]=kernel.VectorResidual(base,row,col,slot);
}
}
kernel.SolveInPlaceWithFactors(base,dx,k);
for(Zq2 col=0;col<k;++col)
for(Zq2 row=0;row<n;++row)
kernel.SetX(base,row,col,kernel.GetX(base,row,col)+dx[Zq3(row)*k+col]);
}
}

};
