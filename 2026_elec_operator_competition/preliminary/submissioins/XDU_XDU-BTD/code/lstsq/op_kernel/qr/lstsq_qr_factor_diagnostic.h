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


namespace qr_factor_probe{
constexpr Zq2 N=256u,Group=397824u,Goff=256u,NegOff=65792u,QtOff=131328u;
constexpr Zq2 InvOff=196864u,InvNegOff=205056u;

__aicore__ inline __gm__ float*Workspace(GM_ADDR workspace,Zq2 batch){
Zq3 user=reinterpret_cast<Zq3>(GetUserWorkspace(workspace));
return reinterpret_cast<__gm__ float*>((user+63u)&~Zq3(63u))+Zq3(batch)*Group;
}
template<bool EnableBoundedH3=false,bool ColumnFactored256=false,bool CropZ200=false>
class Cube{
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR workspace,Zq2 batch,Zq2 lane,Zq2 logicalN=256u){
lane_=lane;logicalN_=logicalN;activeN_=logicalN==200u?224u:256u;matrix_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a)+batch*logicalN*logicalN,logicalN*logicalN);
auto ptr=Workspace(workspace,batch);workspace_=ptr;g_.SetGlobalBuffer(ptr+Goff,N*N);neg_.SetGlobalBuffer(ptr+NegOff,N*N);
inv_.SetGlobalBuffer(ptr+InvOff,8192u);invNeg_.SetGlobalBuffer(ptr+InvNegOff,8192u);polyDecision_.SetGlobalBuffer(reinterpret_cast<__gm__ Zq2*>(ptr)+(EnableBoundedH3?(64u+lane*16u):48u),EnableBoundedH3?16u:1u);qt_.SetGlobalBuffer(ptr+QtOff,N*N);second_.SetGlobalBuffer(ptr+213248u,N*N);z_.SetGlobalBuffer(ptr+278784u,N*N);
pipe_.InitBuffer(ab_,262144u);pipe_.InitBuffer(bb_,262144u);pipe_.InitBuffer(ax_,65536u);
pipe_.InitBuffer(bx_,65536u);pipe_.InitBuffer(cb_,32768u);
a1_=ab_.Get<float>();b1_=bb_.Get<float>();a2_=ax_.Get<float>();b2_=bx_.Get<float>();c_=cb_.Get<float>();SetHF32Mode(false);
}
__aicore__ inline void Gram();
__aicore__ inline void Factor(){if(logicalN_==256u)FactorPipelined256();else FactorSynchronous200();}
__aicore__ inline void FactorPipelined256();
__aicore__ inline void IssueDualM32SignedPanel256(Zq2 offset,Zq2 cols){
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_,ld);LoadData(a2_[1024u],a1_[1024u],ld);ZqB<Zq5::MTE1_M>();
MmadParams mm;mm.m=32u;mm.n=uint16_t(cols);mm.k=32u;mm.cmatrixInitVal=true;mm.cmatrixSource=false;
Mmad(c_,a2_,b2_,mm);Mmad(c_[4096u],a2_[1024u],b2_,mm);ZqB<Zq5::M_FIX>();
FixpipeParamsV220 fp;fp.mSize=32u;fp.nSize=uint16_t(cols);fp.srcStride=32u;fp.dstStride=N;fp.quantPre=QuantMode_t::NoQuant;
Fixpipe(g_[offset],c_,fp);Fixpipe(neg_[offset],c_[4096u],fp);Zq1<PIPE_ALL>();
}
__aicore__ inline void EarlyFirstTail256(Zq2 j,Zq2 end,Zq2 tail);
__aicore__ inline void FactorSynchronous200();
__aicore__ inline void EarlySchurDiagonal200(Zq2 j,Zq2 end,Zq2 tail){
if(lane_!=0u)return;
const Zq2 rows=tail<64u?tail:64u;

CopyNd(a1_,neg_[j*N+end],32u,rows,N);ZqB<Zq5::MTE2_MTE1>();LoadATransposeFactor(rows);
CopyNd(b1_,g_[j*N+end],32u,rows,N);ZqB<Zq5::MTE2_MTE1>();LoadB(rows,32u);
ZqB<Zq5::MTE1_M>();Multiply(rows,rows,32u);Store(g_,end*N+end,rows,rows,true);
}
__aicore__ inline void LoadATranspose128(){
for(Zq2 mb=0u;mb<8u;++mb){LoadData2dTransposeParams tr;
tr.startIndex=uint16_t(mb);tr.repeatTimes=8u;tr.srcStride=8u;tr.dstGap=1u;tr.dstFracGap=0u;
LoadDataWithTranspose(a2_[mb*2048u],a1_,tr);}
}
__aicore__ inline void LeftTrsm128();
__aicore__ inline void LeftTrsm();


__aicore__ inline bool ComposedPath(){return false;}
__aicore__ inline void ComposeConsumer();


__aicore__ inline void Store(GlobalTensor<float>dst,Zq2 offset,Zq2 rows,Zq2 cols,bool atomic,Zq2 pitch){
FixpipeParamsV220 fp;fp.mSize=uint16_t(rows);fp.nSize=uint16_t(cols);fp.srcStride=uint16_t(rows);fp.dstStride=uint16_t(pitch);fp.quantPre=QuantMode_t::NoQuant;
if(atomic)SetAtomicAdd<float>();Fixpipe(dst[offset],c_,fp);Zq1<PIPE_ALL>();if(atomic)SetAtomicNone();
}


__aicore__ inline void EarlyR64Block200(Zq2 pair);
__aicore__ inline void PrepareR64Direct200();
__aicore__ inline void PrepareR64();
__aicore__ inline void SolveRhsFrame64(GlobalTensor<float>rhs);
__aicore__ inline void PrepareDistributedR256200();
__aicore__ inline void CacheDistributedR256200(){
GlobalTensor<float>wide;wide.SetGlobalBuffer(workspace_+356096u,32768u);
CopyNd(a1_[16384u],wide[(lane_/2u)*16384u+(lane_%2u)*8192u],lane_==3u?16u:64u,lane_<2u?128u:80u,128u);
if(lane_<2u)CopyNd(a1_[24576u],g_[(128u+lane_*64u)*N],64u,80u,N);
ZqB<Zq5::MTE2_MTE1>();
}
__aicore__ inline void SolveDistributedR256200(GlobalTensor<float>src,GlobalTensor<float>dst);
__aicore__ inline void PrepareFactoredApply();
__aicore__ inline void StoreRhs(GlobalTensor<float>dst,Zq2 row,bool atomic){
FixpipeParamsV220 fp;fp.mSize=32u;fp.nSize=16u;fp.srcStride=32u;fp.dstStride=16u;fp.quantPre=QuantMode_t::NoQuant;
if(atomic)SetAtomicAdd<float>();Fixpipe(dst[row*16u],c_,fp);
if(logicalN_==200u){ZqB<Zq5::FIX_M>();ZqB<Zq5::FIX_S>();ZqB<Zq5::FIX_MTE2>();}
else Zq1<PIPE_ALL>();if(atomic)SetAtomicNone();
}
__aicore__ inline void SolveRhsFrame(GlobalTensor<float>rhs);

__aicore__ inline bool CropZMode200(){
if constexpr(CropZ200)return cropZ200_;else return false;
}
__aicore__ inline void ConsumeRhs();
__aicore__ inline __gm__ float*WorkspacePointer(){return workspace_;}
template<bool IsGram>
__aicore__ inline void SymmetricUpper256(GlobalTensor<float>dst);
__aicore__ inline void SecondGram();
__aicore__ inline void CacheA(GlobalTensor<float>src){
if(logicalN_==200u){CacheDense224(a1_,src);return;}

Nd2NzParams cp;cp.ndNum=16u;cp.nValue=16u;cp.dValue=256u;cp.srcNdMatrixStride=4096u;
cp.srcDValue=256u;cp.dstNzC0Stride=16u;cp.dstNzNStride=1u;cp.dstNzMatrixStride=4096u;
DataCopy(a1_,src,cp);ZqB<Zq5::MTE2_MTE1>();
}
__aicore__ inline void CacheB(GlobalTensor<float>src){
if(logicalN_==200u){CacheDense224(b1_,src);return;}

Nd2NzParams cp;cp.ndNum=16u;cp.nValue=16u;cp.dValue=256u;cp.srcNdMatrixStride=4096u;
cp.srcDValue=256u;cp.dstNzC0Stride=16u;cp.dstNzNStride=1u;cp.dstNzMatrixStride=4096u;
DataCopy(b1_,src,cp);ZqB<Zq5::MTE2_MTE1>();
}
__aicore__ inline void CachedProduct(GlobalTensor<float>dst,bool square,bool atomic);
__aicore__ inline void CachedSquareFromB256(GlobalTensor<float>dst);
__aicore__ inline void Polynomial7();
__aicore__ inline void FinishPolynomialZ();
__aicore__ inline void PolynomialDenseTail();
__aicore__ inline void FinalConsumerZ();
__aicore__ inline void CacheDense224(Zq7<float>dst,GlobalTensor<float>src){
Nd2NzParams cp;cp.ndNum=uint16_t(208u/16u);cp.nValue=16u;cp.dValue=uint16_t(208u);
cp.srcNdMatrixStride=16u*N;cp.srcDValue=N;cp.dstNzC0Stride=16u;cp.dstNzNStride=1u;cp.dstNzMatrixStride=16u*208u;
DataCopy(dst,src,cp);ZqB<Zq5::MTE2_MTE1>();
}

__aicore__ inline void InitialGramUpper200();
__aicore__ inline void DenseGram224(GlobalTensor<float>dst);
__aicore__ inline void DenseProduct224(GlobalTensor<float>dst,bool square,bool atomic);
__aicore__ inline void FinalConsumerZ224();

private:


struct PoolTask{Zq2 batch,col,firstRow,rowStep;};
__aicore__ inline PoolTask PoolGeometry(Zq2 mask);
__aicore__ inline __gm__ float*PoolBase(){return workspace_-Zq3(GetBlockIdx()/4u)*Group;}
__aicore__ inline void CachePool(const PoolTask&task,Zq2 aOffset,Zq2 bOffset,bool keepB=false){
const Zq2 batch=task.batch,col=task.col,step=task.rowStep;Zq2 row=task.firstRow;
auto ptr=PoolBase()+Zq3(batch)*Group;GlobalTensor<float>a,b;
a.SetGlobalBuffer(ptr+aOffset,N*N);b.SetGlobalBuffer(ptr+bOffset,N*N);
if(!keepB)CopyNd(b1_,b[col],256u,64u,N);
for(Zq2 slot=0u;row<N;row+=step,++slot)CopyNd(a1_[slot*16384u],a[row*N],64u,256u,N);
ZqB<Zq5::MTE2_MTE1>();
}
__aicore__ inline void ComputePool(const PoolTask&task,Zq2 dstOffset,bool atomic);
__aicore__ inline void FinishPooledZ(const PoolTask&heavy){
if constexpr(!ColumnFactored256){CachePool(heavy,213248u,QtOff);ComputePool(heavy,278784u,false);}
CrossCoreSetFlag<2,PIPE_FIX>(9);
}
__aicore__ inline void PooledPolynomial(Zq2 lightMask);

__aicore__ inline void DenseBegin(bool atomic){
denseSlot_=0u;denseIssued_=false;denseAtomic_=atomic;
densePending_[0]=false;densePending_[1]=false;
if(atomic)SetAtomicAdd<float>();
}
__aicore__ inline void DenseWaitMte1(){
if(denseIssued_)ZqB<Zq5::M_MTE1>();
}
__aicore__ inline void DenseWaitMte2(){

if(denseIssued_)ZqB<Zq5::M_MTE2>();
}
__aicore__ inline void DenseMultiply(Zq2 m,Zq2 n,Zq2 k){
if(densePending_[denseSlot_]){WaitFlag<Zq5::FIX_M>(int32_t(denseSlot_));densePending_[denseSlot_]=false;}
MmadParams mm;mm.m=uint16_t(m);mm.n=uint16_t(n);mm.k=uint16_t(k);mm.cmatrixInitVal=true;mm.cmatrixSource=false;
Mmad(c_[denseSlot_*4096u],a2_,b2_,mm);


SetFlag<Zq5::M_FIX>(int32_t(denseSlot_));WaitFlag<Zq5::M_FIX>(int32_t(denseSlot_));
}
__aicore__ inline void DenseStore(GlobalTensor<float>dst,Zq2 offset,Zq2 rows,Zq2 cols,Zq2 pitch){
FixpipeParamsV220 fp;fp.mSize=rows;fp.nSize=cols;fp.srcStride=rows;fp.dstStride=pitch;fp.quantPre=QuantMode_t::NoQuant;
Fixpipe(dst[offset],c_[denseSlot_*4096u],fp);
SetFlag<Zq5::FIX_M>(int32_t(denseSlot_));densePending_[denseSlot_]=true;denseIssued_=true;denseSlot_^=1u;
}
__aicore__ inline void DenseEnd(){
if(denseIssued_){


ZqB<Zq5::FIX_S>();


ZqB<Zq5::FIX_MTE2>();
for(Zq2 slot=0u;slot<2u;++slot)if(densePending_[slot]){WaitFlag<Zq5::FIX_M>(int32_t(slot));densePending_[slot]=false;}
}
if(denseAtomic_)SetAtomicNone();denseIssued_=false;denseAtomic_=false;
}

__aicore__ inline void CopyNd(Zq7<float>dst,GlobalTensor<float>src,Zq2 rows,Zq2 cols,Zq2 pitch){
Nd2NzParams cp;cp.ndNum=uint16_t(rows/16u);cp.nValue=16u;cp.dValue=uint16_t(cols);cp.srcNdMatrixStride=16u*pitch;
cp.srcDValue=uint16_t(pitch);cp.dstNzC0Stride=16u;cp.dstNzNStride=1u;cp.dstNzMatrixStride=16u*cols;DataCopy(dst,src,cp);
}
__aicore__ inline void LoadB(Zq2 cols,Zq2 k){
for(Zq2 nb=0;nb<cols/16u;++nb){LoadData2dTransposeParams tr;tr.startIndex=uint16_t(nb);
tr.repeatTimes=Zq4(k/16u);tr.srcStride=uint16_t(cols/16u);tr.dstGap=uint16_t(cols/8u-1u);tr.dstFracGap=uint16_t(cols/16u-1u);
LoadDataWithTranspose(b2_[nb*128u],b1_,tr);}
}
__aicore__ inline void LoadATransposeFactor(Zq2 rows){
for(Zq2 mb=0;mb<rows/16u;++mb){LoadData2dTransposeParams tr;tr.startIndex=uint16_t(mb);tr.repeatTimes=2u;tr.srcStride=uint16_t(rows/16u);tr.dstGap=1u;tr.dstFracGap=0u;
LoadDataWithTranspose(a2_[mb*512u],a1_,tr);}
}
__aicore__ inline void LoadATranspose32(){
for(Zq2 mb=0;mb<2u;++mb){LoadData2dTransposeParams tr;tr.startIndex=uint16_t(mb);tr.repeatTimes=2u;tr.srcStride=2u;tr.dstGap=1u;tr.dstFracGap=0u;
LoadDataWithTranspose(a2_[mb*512u],a1_,tr);}
}
__aicore__ inline void Multiply(Zq2 m,Zq2 n,Zq2 k){
MmadParams mm;mm.m=uint16_t(m);mm.n=uint16_t(n);mm.k=uint16_t(k);mm.cmatrixInitVal=true;mm.cmatrixSource=false;
Mmad(c_,a2_,b2_,mm);ZqB<Zq5::M_FIX>();
}
__aicore__ inline void Store(GlobalTensor<float>dst,Zq2 offset,Zq2 rows,Zq2 cols,bool atomic){
FixpipeParamsV220 fp;fp.mSize=rows;fp.nSize=cols;fp.srcStride=rows;fp.dstStride=N;fp.quantPre=QuantMode_t::NoQuant;
if(atomic)SetAtomicAdd<float>();Fixpipe(dst[offset],c_,fp);Zq1<PIPE_ALL>();if(atomic)SetAtomicNone();
}
TPipe pipe_;TBuf<TPosition::A1>ab_;TBuf<TPosition::B1>bb_;TBuf<TPosition::A2>ax_;TBuf<TPosition::B2>bx_;TBuf<TPosition::CO1>cb_;
__gm__ float*workspace_=nullptr;GlobalTensor<Zq2>polyDecision_;Zq7<float>a1_,b1_,a2_,b2_,c_;GlobalTensor<float>matrix_,g_,neg_,inv_,invNeg_,qt_,second_,z_;Zq2 lane_,logicalN_,activeN_;bool factoredApply_=false,cropZ200_=false;bool densePending_[2]={false,false},denseIssued_=false,denseAtomic_=false;Zq2 denseSlot_=0u;
};

struct QrNormalization {
    float inverseScale;
    Zq2 exponent;
};

__aicore__ inline void WriteQrHeader(
    Zq7<float> all, GlobalTensor<float> header, Zq2 batch, Zq2 panelCount,
    QrNormalization normalization, bool valid, Zq2 degree, Zq2 columnLayout, Zq2 reduced)
{
    auto words = all.Zq6<Zq2>();
    ZqB<Zq5::V_S>();
    for (Zq2 i = 0; i < 64u; ++i) words.SetValue(i, 0u);
    words.SetValue(0u, 0x51524631u);
    words.SetValue(1u, N);
    words.SetValue(2u, batch);
    words.SetValue(3u, 0x53544731u);
    words.SetValue(4u, Group);
    words.SetValue(5u, GetBlockNum());
    words.SetValue(6u, 4u);
    words.SetValue(7u, panelCount);
    words.SetValue(24u, QtOff);
    words.SetValue(25u, NegOff);
    words.SetValue(26u, InvOff);
    words.SetValue(27u, InvNegOff);
    words.SetValue(28u, Goff);
    words.SetValue(29u, 5u);
    words.SetValue(30u, 0x51465232u);
    words.SetValue(31u, batch);
    words.SetValue(34u, 213248u);
    words.SetValue(35u, 32u);
    words.SetValue(36u, 278784u);
    union { float value; Zq2 bits; } scale;
    scale.value = normalization.inverseScale;
    words.SetValue(40u, scale.bits);
    words.SetValue(41u, normalization.exponent);
    words.SetValue(42u, valid ? 1u : 0u);
    words.SetValue(43u, degree);
    words.SetValue(44u, columnLayout);
    words.SetValue(45u, 0x48594231u);
    words.SetValue(46u, reduced);
    ZqB<Zq5::S_MTE3>();
    DataCopy(header, all, 64u);
    ZqB<Zq5::MTE3_V>();
}

__aicore__ inline void WriteQrError(
    Zq7<float> all, GlobalTensor<float> errorLines, Zq2 lane, float errorSquared)
{
    ZqB<Zq5::V_S>();
    for (Zq2 i = 0; i < 16u; ++i) all.SetValue(i, 0.0f);
    all.SetValue(0u, errorSquared);
    ZqB<Zq5::S_MTE3>();
    DataCopy(errorLines[lane * 16u], all, 16u);
    ZqB<Zq5::MTE3_S>();
}

__aicore__ inline float ReadQrError(GM_ADDR workspace, Zq2 batch, Zq7<float> scratch)
{
    GlobalTensor<float> lines;
    lines.SetGlobalBuffer(Workspace(workspace, batch) + 344320u, 64u);
    DataCopy(scratch, lines, 64u);
    ZqB<Zq5::MTE2_S>();
    float value = scratch.GetValue(0u);
    for (Zq2 i = 1u; i < 4u; ++i) value += scratch.GetValue(i * 16u);
    return value;
}

#include "lstsq_qr_vector_common.h"

class VectorNormalize200{
    friend struct QrVectorOperations;
public:
template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, Zq2 batch, Zq2 lane);
private:
template<Zq2 LogicalN>
__aicore__ inline QrNormalization Normalize(GM_ADDR a, __gm__ float* ptr, Zq2 batch, Zq2 lane);
template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void StageFactorPanels(__gm__ float* ptr, Zq2 lane, float inverseScale);
__aicore__ inline void PrepareSecondGram(__gm__ float* ptr, Zq2 lane);
__aicore__ inline void PreparePolynomial(GM_ADDR workspace, __gm__ float* ptr, Zq2 lane);
__aicore__ inline void RunWidePolynomial(__gm__ float* ptr, GlobalTensor<float> h, Zq2 lane);
__aicore__ inline void PrepareFinalApply(Zq2 lane);
__aicore__ inline void PrepareEarlyR64Indices200(){
if(lane_!=3u)return;
auto ri=all_.Zq6<int32_t>(),ci=all_[64u].Zq6<int32_t>(),co=all_[128u].Zq6<int32_t>();
ZqB<Zq5::S_V>();CreateVecIndex<int32_t>(ri,int32_t(0),32u);CreateVecIndex<int32_t>(ci,int32_t(0),32u);Zq1<PIPE_V>();
Muls<int32_t>(ri,ri,int32_t(4),32u);Muls<int32_t>(ci,ci,int32_t(128),32u);Zq1<PIPE_V>();
Brcb(co,ri,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
Add<int32_t>(trans_.Zq6<int32_t>(),ci,co,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});Zq1<PIPE_V>();
}
__aicore__ inline void StageEarlyR64Seed200(GM_ADDR workspace,Zq2 ownBatch,Zq2 j);
template<Zq2 LogicalN>
__aicore__ inline void FollowFactors200(GM_ADDR workspace,Zq2 batches,Zq2 ownBatch);
template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void PrepareStagedB(GM_ADDR b,Zq2 batch,Zq2 lane);
template<Zq2 RhsK,Zq2 FactorN>
__aicore__ inline void StageBChunk(__gm__ float*ptr,Zq2 lane,Zq2 row,float inverseScale);
__aicore__ inline void AddSparseDiagonal(Zq7<float>values,Zq2 row,float bias);
__aicore__ inline void InitializeDenseExterior(__gm__ float*ptr,Zq2 lane);
template<Zq2 LogicalN>
__aicore__ inline void ReadOriginal(GlobalTensor<float>matrix,Zq2 row);
template<Zq2 LogicalN>
__aicore__ inline void PrepareTranspose();
template<Zq2 LogicalN>
__aicore__ inline void PrepareFusedIndices();
__aicore__ inline void PrepareDirectSeedTranspose200(){


ZqB<Zq5::V_S>();
auto ri=all_.Zq6<int32_t>(),ci=all_[64u].Zq6<int32_t>(),co=all_[128u].Zq6<int32_t>();
ZqB<Zq5::S_V>();CreateVecIndex<int32_t>(ri,int32_t(0),32u);CreateVecIndex<int32_t>(ci,int32_t(0),32u);Zq1<PIPE_V>();
Muls<int32_t>(ri,ri,int32_t(4),32u);Muls<int32_t>(ci,ci,int32_t(128),32u);Zq1<PIPE_V>();
Brcb(co,ri,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
Add<int32_t>(trans_[3072u].Zq6<int32_t>(),ci,co,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});Zq1<PIPE_V>();
}
__aicore__ inline void PublishPrivateDiagonalSeed200(Zq2 j);
template<Zq2 LogicalN>
__aicore__ inline void Diagonal(Zq2 j);
TPipe pipe_;TBuf<TPosition::VECCALC>fb_,ib_,ob_,tb_;Zq7<float>all_,input_,output_;Zq7<Zq2>trans_;
__gm__ float*seedWorkspace200_=nullptr;GlobalTensor<float>g_,neg_,qt_,inv_,invNeg_,header_,errorLines_;Zq2 lane_;float errorSquared_=0.0f;bool pdOkay_=true,normalScaleDomain_=true,widePolynomial_=false;bool bDomainOkay_[2]={true,true};
};

class VectorBounded256{
    friend struct QrVectorOperations;
public:
template<Zq2 LogicalN,Zq2 RhsK,bool ColumnFactored256=false>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, Zq2 batch, Zq2 lane);
private:
__aicore__ inline void FollowFactorPanels();
__aicore__ inline bool SelectInitialDegree(GM_ADDR workspace, Zq2 batches);
__aicore__ inline void FollowWidePolynomial(GM_ADDR workspace, Zq2 batches);
template<Zq2 LogicalN,bool ColumnFactored256>
__aicore__ inline QrNormalization Normalize(GM_ADDR a, __gm__ float* ptr, Zq2 batch, Zq2 lane);
template<Zq2 LogicalN,Zq2 RhsK,bool ColumnFactored256>
__aicore__ inline void StageFactorPanels(__gm__ float* ptr, Zq2 lane, float inverseScale);
__aicore__ inline void PrepareSecondGram(__gm__ float* ptr, Zq2 lane);
template<Zq2 LogicalN,bool ColumnFactored256>
__aicore__ inline void PreparePolynomial(GM_ADDR workspace, __gm__ float* ptr, Zq2 batch, Zq2 lane);
__aicore__ inline void RunWidePolynomial(GM_ADDR workspace, __gm__ float* ptr, GlobalTensor<float> h, Zq2 batch, Zq2 lane);
__aicore__ inline void MeasureSecondPower(__gm__ float* ptr, GlobalTensor<float> ep, Zq2 lane);
__aicore__ inline void UpdateSecondPower(GM_ADDR workspace, __gm__ float* ptr, GlobalTensor<float> ep, GlobalTensor<float> h, Zq2 batch, Zq2 lane);
__aicore__ inline void StoreSecondPower(GlobalTensor<float> ep, Zq2 lane);
__aicore__ inline void AddPolynomialPower(GlobalTensor<float> h, GlobalTensor<float> power, Zq2 lane);
__aicore__ inline void AddSparseDiagonal(Zq7<float>values,Zq2 row,float bias);
__aicore__ inline Zq2 RestrictPoolMask(Zq2 light,Zq2 batches){
Zq2 heavy=0u;for(Zq2 b=0u;b<batches;++b)if(!(light&(1u<<b)))++heavy;
return heavy&&heavy*2u<=batches?light:0u;
}
__aicore__ inline void SelectActualE2(GM_ADDR workspace,Zq2 batches,bool publish);
__aicore__ inline void PrepareRetiredInverseTranspose256(){
ZqB<Zq5::V_S>();auto ri=all_.Zq6<int32_t>();auto ci=all_[64u].Zq6<int32_t>();auto co=all_[128u].Zq6<int32_t>();
for(Zq2 i=0;i<32u;++i){ri.SetValue(i,int32_t(i*4u));ci.SetValue(i,int32_t(i*128u));}ZqB<Zq5::S_V>();
Brcb(co,ri,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
Add<int32_t>(trans_.Zq6<int32_t>(),ci,co,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});Zq1<PIPE_V>();
}
__aicore__ inline void StageRetiredInverseTranspose256(Zq2 block);
__aicore__ inline void PrepareFoldedSymmetricReadIndices256(){
auto ri=all_[10240u].Zq6<int32_t>(),ci=all_[10304u].Zq6<int32_t>(),co=all_[10368u].Zq6<int32_t>();
CreateVecIndex<int32_t>(ri,int32_t(0),Zq2(32));CreateVecIndex<int32_t>(ci,int32_t(0),Zq2(64));Zq1<PIPE_V>();
Muls<int32_t>(ri,ri,int32_t(4),Zq2(32));Muls<int32_t>(ci,ci,int32_t(128),Zq2(64));Zq1<PIPE_V>();
Brcb(co,ri,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
Add<int32_t>(all_[8192u].Zq6<int32_t>(),ci,co,Zq3(64u),Zq4(32u),Zq0{1,1,0,8,0,1});Zq1<PIPE_V>();
}
__aicore__ inline void LoadFoldedSymmetric32(Zq7<float>dst,GlobalTensor<float>raw,Zq2 row);
template<Zq2 LogicalN,bool ColumnFactored256=false>
__aicore__ inline void FollowFactors200(GM_ADDR workspace,Zq2 batches);
template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void PrepareStagedB(GM_ADDR b,Zq2 batch,Zq2 lane);
template<Zq2 RhsK,Zq2 FactorN>
__aicore__ inline void StageBChunk(__gm__ float*ptr,Zq2 lane,Zq2 row,float inverseScale);
__aicore__ inline void InitializeDenseExterior(__gm__ float*ptr,Zq2 lane);
template<Zq2 LogicalN>
__aicore__ inline void ReadOriginal(GlobalTensor<float>matrix,Zq2 row);
template<Zq2 LogicalN,bool CompactTranspose=false>
__aicore__ inline void PrepareTranspose();
template<Zq2 LogicalN>
__aicore__ inline void PrepareFusedIndices();
template<Zq2 LogicalN,bool PackedPivotCheck=false>
__aicore__ inline void Diagonal(Zq2 j);
TPipe pipe_;TBuf<TPosition::VECCALC>fb_,ib_,ob_,tb_;Zq7<float>all_,input_,output_;Zq7<Zq2>trans_;
__gm__ float*seedWorkspace256_=nullptr;GlobalTensor<float>g_,neg_,qt_,inv_,invNeg_,header_,errorLines_;Zq2 lane_;float errorSquared_=0.0f;bool pdOkay_=true,normalScaleDomain_=true,widePolynomial_=false,cubicPolynomial_=false,quarticPolynomial_=false,quadraticPolynomial_=false,reducedH4Eligible_=false,reducedH4_=false;Zq2 pooledLightMask_=0u;bool bDomainOkay_[2]={true,true};
};

#include "lstsq_qr_cube_solve.h"


#include "lstsq_qr_vector200_factor.h"


#include "lstsq_qr_vector256_dispatch.h"
#include "lstsq_qr_vector256_factor.h"
#include "lstsq_qr_vector256_input.h"
#include "lstsq_qr_cube_transport.h"
#include "lstsq_qr_cube_compute.h"
#include "lstsq_qr_vector200_transport.h"
template<Zq2 LogicalN=256u,Zq2 RhsK=3u,bool ColumnFactored256=false>
__aicore__ inline void Run(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR workspace,LinalgLstsqTilingData t){
if(t.n!=LogicalN||t.k!=RhsK||t.batches>4u||GetBlockNum()!=t.batches*4u)return;
Zq2 task=GetBlockIdx();
if ASCEND_IS_AIV{task/=2u;}
Zq2 batch=task/4u,lane=task%4u;
if ASCEND_IS_AIC{Cube<LogicalN==256u,ColumnFactored256,(LogicalN==200u&&RhsK==4u)>cube;cube.Init(a,workspace,batch,lane,LogicalN);cube.Gram();cube.Factor();
CrossCoreWaitFlag(4);cube.LeftTrsm128();CrossCoreSetFlag<2,PIPE_FIX>(5);
CrossCoreWaitFlag(6);cube.SecondGram();CrossCoreSetFlag<2,PIPE_FIX>(7);
cube.Polynomial7();cube.PrepareFactoredApply();cube.ConsumeRhs();}
if ASCEND_IS_AIV{
if constexpr(LogicalN==200u){
VectorNormalize200 vector;vector.Run<LogicalN,RhsK>(a,b,workspace,batch,lane);
}else{VectorBounded256 vector;vector.Run<LogicalN,RhsK,ColumnFactored256>(a,b,workspace,batch,lane);}
}
}
}
