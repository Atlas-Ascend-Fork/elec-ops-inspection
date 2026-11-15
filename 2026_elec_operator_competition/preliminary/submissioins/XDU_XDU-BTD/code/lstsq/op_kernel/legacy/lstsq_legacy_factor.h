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

__aicore__ inline void KernelLegacy::DecomposeSquare(Zq3 base)
{
if(localFactors_){DecomposeLocal(base);return;}
if(tiling_.n==256u){DecomposeBlocked(base);return;}
const Zq2 n=tiling_.n;
__gm__ float*lu=w_+OffLU(base);
__gm__ float*piv=w_+OffP(base);
LuFactorizationOperations::FactorScalarSquare(lu, piv, w_+OffInv(base), n);
}

__aicore__ inline void KernelLegacy::DecomposeLocal(Zq3 base)
{
const Zq2 n=tiling_.n;
__gm__ float*piv=w_+OffP(base);
for(Zq2 j=0;j<n;++j){
const Zq2 pick=TriangularSolveOperations::SelectLocalPivot(*this,base,j,n);
piv[j]=static_cast<float>(static_cast<int32_t>(pick));
if(pick!=j){
ZqB<Zq5::S_V>();
Adds(rows_,factors_[j*factorPitch_],0.0f,factorPitch_);
Zq1<PIPE_V>();
Adds(factors_[j*factorPitch_],factors_[pick*factorPitch_],0.0f,factorPitch_);
Zq1<PIPE_V>();
Adds(factors_[pick*factorPitch_],rows_,0.0f,factorPitch_);
ZqB<Zq5::V_S>();
}
const float diagonal=Factor(base,j,j);
if(diagonal==0.0f){w_[OffInv(base)+j]=0.0f;continue;}
const bool safe=AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f;
const float inverse=safe?square_math::Divide(1.0f,diagonal):0.0f;
w_[OffInv(base)+j]=inverse;
if(j+1u==n)continue;
for(Zq2 c=0;c<factorPitch_;++c)
rows_.SetValue(c,c>j&&c<n?Factor(base,j,c):0.0f);

for(Zq2 i=j+1;i<n;++i)
PutFactor(base,i,j,safe?square_math::RoundMul(Factor(base,i,j),inverse):square_math::Divide(Factor(base,i,j),diagonal));
ZqB<Zq5::S_V>();


SetMaskCount();
ZqA<float,MaskMode::COUNTER>(0u,factorPitch_);
for(Zq2 i=j+1;i<n;++i){
const float mult=Factor(base,i,j);
Axpy<float,float,false>(factors_[i*factorPitch_],rows_,-mult,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();SetMaskNorm();

ZqB<Zq5::V_S>();
}
}

__aicore__ inline void KernelLegacy::DecomposeBlocked(Zq3 base)
{
constexpr Zq2 n = 256u;
constexpr Zq2 panel = 8u;
__gm__ float* lu = w_ + OffLU(base);
__gm__ float* piv = w_ + OffP(base);
for (Zq2 jb = 0; jb < n; jb += panel) {
    const Zq2 je = jb + panel;
    FactorBlockedPanel(base, lu, piv, jb, je);
    if (je == n) continue;
    SolveBlockedPanelTail(lu, jb, je);
    UpdateBlockedTrailingTiles(lu, jb, je);
}
}

__aicore__ inline void KernelLegacy::FactorBlockedPanel(Zq3 base, __gm__ float* lu, __gm__ float* piv, Zq2 jb, Zq2 je)
{
constexpr Zq2 n = 256u;
for(Zq2 j=jb;j<je;++j){
Zq2 pick=j;
float best=AbsF(lu[j*n+j]);
for(Zq2 i=j+1;i<n;++i){
const float value=AbsF(lu[i*n+j]);
if(value>best){best=value;pick=i;}
}
piv[j]=static_cast<float>(static_cast<int32_t>(pick));
if(pick!=j)for(Zq2 c=0;c<n;++c){
const float temp=lu[j*n+c];lu[j*n+c]=lu[pick*n+c];lu[pick*n+c]=temp;
}
const float diagonal=lu[j*n+j];
if(diagonal==0.0f)continue;
const float magnitude=AbsF(diagonal);
const bool safe=magnitude>=1.0e-30f&&magnitude<=1.0e30f;
const float inverse=safe?square_math::Divide(1.0f,diagonal):0.0f;
w_[OffInv(base)+j]=inverse;
for(Zq2 i=j+1;i<n;++i){
const float mult=safe?square_math::RoundMul(lu[i*n+j],inverse):square_math::Divide(lu[i*n+j],diagonal);
lu[i*n+j]=mult;
for(Zq2 c=j+1;c<je;++c)lu[i*n+c]-=mult*lu[j*n+c];
}
}
}

__aicore__ inline void KernelLegacy::SolveBlockedPanelTail(__gm__ float* lu, Zq2 jb, Zq2 je)
{
constexpr Zq2 n = 256u;
const Zq2 tail = n-je;
auto upper=block_;
for(Zq2 j=jb;j<je;++j){
const Zq2 row=(j-jb)*256u;
for(Zq2 c=0;c<tail;++c)upper.SetValue(row+c,lu[j*n+je+c]);
ZqB<Zq5::S_V>();
SetMaskCount();
ZqA<float,MaskMode::COUNTER>(0u,tail);
for(Zq2 p=jb;p<j;++p){
const float mult=lu[j*n+p];
Axpy<float,float,false>(upper[row],upper[(p-jb)*256u],-mult,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();
ZqB<Zq5::V_S>();
for(Zq2 c=0;c<tail;++c)lu[j*n+je+c]=upper.GetValue(row+c);
}
}

__aicore__ inline void KernelLegacy::UpdateBlockedTrailingTiles(__gm__ float* lu, Zq2 jb, Zq2 je)
{
constexpr Zq2 n = 256u, tileRows = 32u;
const Zq2 tail = n-je;
auto upper=block_, tile=block_[16u*256u];
GlobalTensor<float>gmLu;gmLu.SetGlobalBuffer(lu,n*n);
DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(gmLu);
ZqB<Zq5::S_MTE2>();
const Zq2 tiles=(n-je+tileRows-1u)/tileRows;
const Zq2 firstCount=n-je<tileRows?n-je:tileRows;
DataCopyPad(tile,gmLu[je*n+je],
Zq8{uint16_t(firstCount),tail*4u,je*4u,je/8u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(0);
for(Zq2 ti=0;ti<tiles;++ti){
const Zq2 ib=je+ti*tileRows,slot=ti&1u;
const Zq2 count=n-ib<tileRows?n-ib:tileRows;
Zq7<float>current=block_[(16u+slot*32u)*256u];
WaitFlag<Zq5::MTE2_V>(slot);
if(ti+1u<tiles){
const Zq2 nextSlot=slot^1u,next=ib+tileRows;
const Zq2 nextCount=n-next<tileRows?n-next:tileRows;
if(ti>=1u)WaitFlag<Zq5::MTE3_MTE2>(nextSlot);
Zq7<float>nextTile=block_[(16u+nextSlot*32u)*256u];
DataCopyPad(nextTile,gmLu[next*n+je],
Zq8{uint16_t(nextCount),tail*4u,je*4u,je/8u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(nextSlot);
}
SetMaskCount();
ZqA<float,MaskMode::COUNTER>(0u,tail);


for(Zq2 i=0;i<count;++i)for(Zq2 pp=jb;pp<je;++pp){
const float mult=lu[(ib+i)*n+pp];
Axpy<float,float,false>(current[i*256u],upper[(pp-jb)*256u],-mult,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();
ZqB<Zq5::V_MTE3>(slot);
DataCopyPad(gmLu[ib*n+je],current,
Zq8{uint16_t(count),tail*4u,je/8u,je*4u,0u});
SetFlag<Zq5::MTE3_MTE2>(slot);
}
WaitFlag<Zq5::MTE3_MTE2>(0);
if(tiles>1u)WaitFlag<Zq5::MTE3_MTE2>(1);
ZqB<Zq5::MTE3_S>();
DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(gmLu);
}

__aicore__ inline void KernelLegacy::Qr(Zq3 base)
{
    RectangularSolveOperations::Qr(w_+OffLU(base), w_+OffB(base), tiling_.m, tiling_.n, tiling_.k);
}
