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

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::Polynomial7()
{
if constexpr(EnableBoundedH3){

CrossCoreWaitFlag(8);Zq2 polynomialDegree=polyDecision_.GetValue(0u);if(polynomialDegree==1u){FinalConsumerZ();CrossCoreSetFlag<2,PIPE_FIX>(9);return;}if(logicalN_==256u){
const Zq2 lightMask=polyDecision_.GetValue(1u);
if(lightMask){PooledPolynomial(lightMask);return;}

CacheA(neg_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
if constexpr(ColumnFactored256)SymmetricUpper256<false>(z_);else CachedProduct(neg_,true,false);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
DataCacheCleanAndInvalid<Zq2,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(polyDecision_);
polynomialDegree=polyDecision_.GetValue(0u);
if(polynomialDegree==2u){FinalConsumerZ();CrossCoreSetFlag<2,PIPE_FIX>(9);return;}
CacheA(second_);CacheB(neg_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedProduct(second_,false,true);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
if(polynomialDegree==3u){if(!ColumnFactored256||logicalN_!=256u)CacheB(qt_);FinalConsumerZ();CrossCoreSetFlag<2,PIPE_FIX>(9);return;}


CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedSquareFromB256(z_);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
if(polynomialDegree==4u){if(!ColumnFactored256||logicalN_!=256u)CacheB(qt_);FinalConsumerZ();CrossCoreSetFlag<2,PIPE_FIX>(9);return;}
FinishPolynomialZ();
return;}
PolynomialDenseTail();

}else{

CrossCoreWaitFlag(8);if(polyDecision_.GetValue(0u)==1u){FinalConsumerZ();CrossCoreSetFlag<2,PIPE_FIX>(9);return;}if(logicalN_==256u){
CacheA(neg_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedProduct(neg_,true,false);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
CacheA(second_);CacheB(neg_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedProduct(second_,false,true);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);


CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedSquareFromB256(z_);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
FinishPolynomialZ();
return;}
PolynomialDenseTail();

}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PooledPolynomial(Zq2 lightMask)
{
const Zq2 heavyMask=((1u<<(GetBlockNum()/4u))-1u)^lightMask;
const PoolTask heavy=PoolGeometry(heavyMask),light=PoolGeometry(lightMask);


CachePool(heavy,NegOff,NegOff);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
ComputePool(heavy,NegOff,false);CrossCoreSetFlag<2,PIPE_FIX>(13);


if constexpr(!ColumnFactored256){CachePool(light,213248u,QtOff);ComputePool(light,278784u,false);}
CrossCoreWaitFlag(3);
DataCacheCleanAndInvalid<Zq2,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(polyDecision_);
Zq2 degree=polyDecision_.GetValue(0u);
if(degree==2u){FinishPooledZ(heavy);return;}

CachePool(heavy,213248u,NegOff);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
ComputePool(heavy,213248u,true);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
if(degree==3u){FinishPooledZ(heavy);return;}

CachePool(heavy,NegOff,NegOff,true);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
ComputePool(heavy,278784u,false);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
if(degree==4u){FinishPooledZ(heavy);return;}

CachePool(heavy,213248u,278784u);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
ComputePool(heavy,213248u,true);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
FinishPooledZ(heavy);
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::Gram()
{
if(logicalN_==200u){CrossCoreWaitFlag(10);InitialGramUpper200();CrossCoreSetFlag<2,PIPE_FIX>(1);return;}


CrossCoreWaitFlag(10);
Nd2NzParams cp;cp.ndNum=uint16_t(lane_==0u?16u:(lane_==1u?12u:(lane_==2u?8u:4u)));cp.nValue=16u;cp.dValue=256u;cp.srcNdMatrixStride=4096u;
cp.srcDValue=256u;cp.dstNzC0Stride=16u;cp.dstNzNStride=1u;cp.dstNzMatrixStride=4096u;
DataCopy(b1_,qt_,cp);
if(lane_==3u)DataCopy(b1_[192u*N],qt_[192u*N],cp);
ZqB<Zq5::MTE2_MTE1>();


const Zq2 groups=lane_==3u?2u:1u;
for(Zq2 group=0u;group<groups;++group){
const Zq2 col=lane_==3u?group*192u:(3u-lane_)*64u;
const Zq2 firstRow=lane_==3u?col:0u;
const Zq2 lastRow=lane_<2u?128u:(lane_==2u?64u:col);
for(Zq2 nb=0u;nb<4u;++nb){LoadData2DParams lb;lb.repeatTimes=32u;lb.srcStride=1u;lb.dstGap=3u;lb.ifTranspose=false;
LoadData(b2_[nb*128u],b1_[((col/16u)+nb)*4096u],lb);}
for(Zq2 row=firstRow;row<=lastRow;row+=64u){
LoadData2DParams la;la.repeatTimes=128u;la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;LoadData(a2_,b1_[row*N],la);
ZqB<Zq5::MTE1_M>();Multiply(64u,64u,256u);Store(g_,row*N+col,64u,64u,false);
}
}
CrossCoreSetFlag<2,PIPE_FIX>(1);
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::FactorPipelined256()
{
for(Zq2 j=0;j<N;j+=32u){
Zq2 end=j+32u,tail=N-end;CrossCoreWaitFlag(0);
if(tail){
CopyNd(a1_,inv_[j/32u*1024u],32u,32u,32u);
CopyNd(a1_[1024u],invNeg_[j/32u*1024u],32u,32u,32u);ZqB<Zq5::MTE2_MTE1>();
for(Zq2 col=lane_*64u;col<tail;col+=4u*64u){
Zq2 cols=tail-col<64u?tail-col:64u;
CopyNd(b1_,g_[j*N+end+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();LoadB(cols,32u);
if constexpr(ColumnFactored256){


IssueDualM32SignedPanel256(j*N+end+col,cols);
}else{
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_,ld);ZqB<Zq5::MTE1_M>();Multiply(32u,cols,32u);Store(g_,j*N+end+col,32u,cols,false);

LoadData(a2_,a1_[1024u],ld);ZqB<Zq5::MTE1_M>();Multiply(32u,cols,32u);Store(neg_,j*N+end+col,32u,cols,false);
}
}
}
if constexpr(ColumnFactored256){if(tail&&lane_==0u)EarlyFirstTail256(j,end,tail);}
CrossCoreSetFlag<2,PIPE_FIX>(1);
if(!tail)break;
CrossCoreWaitFlag(2);
DenseBegin(true);


Zq2 triangularTile=0u,lastLoadedRow=N;
for(Zq2 row=0u;row<tail;row+=64u){
const Zq2 rows=tail-row<64u?tail-row:64u;
for(Zq2 col=row;col<tail;col+=64u,++triangularTile){
if constexpr(ColumnFactored256){if(triangularTile==0u)continue;}
if(triangularTile%4u!=lane_)continue;
const Zq2 cols=tail-col<64u?tail-col:64u;

if constexpr(ColumnFactored256){if(denseIssued_)ZqB<Zq5::MTE1_MTE2>();}else DenseWaitMte2();
const bool loadRow=row!=lastLoadedRow;
if(loadRow)CopyNd(a1_,neg_[j*N+end+row],32u,rows,N);
CopyNd(b1_,g_[j*N+end+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();
if constexpr(ColumnFactored256){if(denseIssued_)ZqB<Zq5::M_MTE1>();}
if(loadRow){LoadATransposeFactor(rows);lastLoadedRow=row;}
LoadB(cols,32u);
ZqB<Zq5::MTE1_M>();DenseMultiply(rows,cols,32u);DenseStore(g_,(end+row)*N+end+col,rows,cols,N);
}
}
DenseEnd();
CrossCoreSetFlag<2,PIPE_FIX>(1);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::FactorSynchronous200()
{
for(Zq2 j=0;j<activeN_;j+=32u){
Zq2 end=j+32u,tail=activeN_-end;CrossCoreWaitFlag(0);
if(tail){
CopyNd(a1_,inv_[j/32u*1024u],32u,32u,32u);
CopyNd(a1_[1024u],invNeg_[j/32u*1024u],32u,32u,32u);ZqB<Zq5::MTE2_MTE1>();
for(Zq2 col=lane_*64u;col<tail;col+=4u*64u){
Zq2 cols=tail-col<64u?tail-col:64u;
CopyNd(b1_,g_[j*N+end+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();LoadB(cols,32u);
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_,ld);ZqB<Zq5::MTE1_M>();Multiply(32u,cols,32u);Store(g_,j*N+end+col,32u,cols,false);

LoadData(a2_,a1_[1024u],ld);ZqB<Zq5::MTE1_M>();Multiply(32u,cols,32u);Store(neg_,j*N+end+col,32u,cols,false);
}
}
if(lane_==3u&&(j==32u||j==96u||j==160u||j==192u))EarlyR64Block200(j==192u?3u:(j-32u)/64u);
if(tail)EarlySchurDiagonal200(j,end,tail);
CrossCoreSetFlag<2,PIPE_FIX>(1);
if(!tail)break;
CrossCoreWaitFlag(2);

Zq2 row=0u,firstCol=0u,lastCol=tail;bool hasWork=true;
if(tail>128u){
row=lane_<2u?0u:(lane_==2u?64u:128u);
firstCol=lane_==0u?0u:(lane_==1u?128u:(lane_==2u?64u:128u));
if(lane_==0u)lastCol=128u;
}else{
row=lane_==2u?64u:0u;firstCol=lane_==0u?0u:64u;
if(lane_==0u)lastCol=tail<64u?tail:64u;
hasWork=lane_<3u&&row<tail&&firstCol<tail;
}
if(row==0u&&firstCol==0u)firstCol=64u;
hasWork=hasWork&&firstCol<lastCol;
if(hasWork){
Zq2 rows=tail-row<64u?tail-row:64u;
CopyNd(a1_,neg_[j*N+end+row],32u,rows,N);ZqB<Zq5::MTE2_MTE1>();LoadATransposeFactor(rows);
for(Zq2 col=firstCol;col<lastCol;col+=64u){
Zq2 cols=tail-col<64u?tail-col:64u;
CopyNd(b1_,g_[j*N+end+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();LoadB(cols,32u);
ZqB<Zq5::MTE1_M>();Multiply(rows,cols,32u);Store(g_,(end+row)*N+end+col,rows,cols,true);
}
}
CrossCoreSetFlag<2,PIPE_FIX>(1);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::SecondGram()
{
if(logicalN_==200u){DenseGram224(second_);return;}
if constexpr(ColumnFactored256){SymmetricUpper256<true>(z_);return;}


Nd2NzParams cp;cp.ndNum=16u;cp.nValue=16u;cp.dValue=256u;cp.srcNdMatrixStride=4096u;
cp.srcDValue=256u;cp.dstNzC0Stride=16u;cp.dstNzNStride=1u;cp.dstNzMatrixStride=4096u;
DataCopy(b1_,qt_,cp);ZqB<Zq5::MTE2_MTE1>();

for(Zq2 nb=0;nb<4u;++nb){LoadData2DParams lb;lb.repeatTimes=32u;lb.srcStride=1u;lb.dstGap=3u;lb.ifTranspose=false;
LoadData(b2_[nb*128u],b1_[((lane_*64u)/16u+nb)*4096u],lb);}
for(Zq2 tile=lane_;tile<16u;tile+=4u){
Zq2 row=tile/4u*64u,col=tile%4u*64u;
LoadData2DParams la;la.repeatTimes=128u;la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;LoadData(a2_,b1_[row*N],la);

ZqB<Zq5::MTE1_M>();Multiply(64u,64u,256u);Store(second_,row*N+col,64u,64u,false);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::DenseGram224(GlobalTensor<float>dst)
{
CacheDense224(b1_,qt_);
const Zq2 col=lane_*64u,cols=lane_==3u?16u:64u;
for(Zq2 nb=0u;nb<cols/16u;++nb){LoadData2DParams lb;lb.repeatTimes=26u;
lb.srcStride=1u;lb.dstGap=uint16_t(cols/16u-1u);lb.ifTranspose=false;
LoadData(b2_[nb*128u],b1_[(col/16u+nb)*3328u],lb);}
for(Zq2 row=0u;row<208u;row+=64u){
const Zq2 rows=row==192u?16u:64u;
LoadData2DParams la;la.repeatTimes=uint16_t(rows*208u/128u);la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;
LoadData(a2_,b1_[row*208u],la);ZqB<Zq5::MTE1_M>();
Multiply(rows,cols,208u);Store(dst,row*N+col,rows,cols,false);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::FinishPolynomialZ()
{
CacheA(second_);CacheB(z_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedProduct(second_,false,true);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
if(!ColumnFactored256||logicalN_!=256u)CacheB(qt_);FinalConsumerZ();CrossCoreSetFlag<2,PIPE_FIX>(9);
}
template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PolynomialDenseTail()
{
CacheA(neg_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedProduct(neg_,true,false);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
CacheA(neg_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedProduct(z_,true,false);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
CacheA(second_);CacheB(neg_);CrossCoreSetFlag<2,PIPE_MTE2>(11);CrossCoreWaitFlag(12);
CachedProduct(second_,false,true);CrossCoreSetFlag<2,PIPE_FIX>(13);CrossCoreWaitFlag(3);
FinishPolynomialZ();
}
