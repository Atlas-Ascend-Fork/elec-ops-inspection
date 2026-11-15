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
struct WorkspaceOperations {
template<class Kernel>
__aicore__ static inline bool BuildAugmented64(Kernel &kernel)
{
auto aug=kernel.panel_[24576u];auto order=kernel.residualTile_;
auto physicalOffsets=kernel.vr_[640u].template Zq6<Zq2>();
auto diagonal=kernel.negative64_[4160u];
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<64u;++i){order.SetValue(i,i*4u);physicalOffsets.SetValue(i,i*544u);}
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
Duplicate<float>(aug[64u],0.0f,Zq3(64),Zq4(64),1,17);
Adds<int32_t>(aug.template Zq6<int32_t>(),kernel.producerA_.template Zq6<int32_t>(),int32_t(0),Zq3(64),Zq4(64),UnaryRepeatParams{1,1,17,8});
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<64u;++i)aug.SetValue(i*136u+64u+i,1.0f);
for(Zq2 j=0;j<64u;++j){
ZqB<Zq5::S_V>();Gather(kernel.vr_[128u],aug,physicalOffsets,j*4u,64u);Zq1<PIPE_V>();Gather(kernel.solveY_,kernel.vr_[128u],order,0u,64u);Zq1<PIPE_V>();
Abs(kernel.vr_,kernel.solveY_,64u);Zq1<PIPE_V>();
Zq3 active[2]={~Zq3(0u)<<j,0u};
WholeReduceMax(kernel.vr_[64u],kernel.vr_,active,int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);ZqB<Zq5::V_S>();
Zq2 pick=j;float best=kernel.AbsF(kernel.solveY_.GetValue(j));
const Zq2 candidate=kernel.vr_[64u].template Zq6<Zq2>().GetValue(1u);
const float maximum=kernel.vr_[64u].GetValue(0u);
if(candidate>=j&&candidate<64u&&maximum==maximum)pick=candidate;
else for(Zq2 i=j+1u;i<64u;++i){float v=kernel.AbsF(kernel.solveY_.GetValue(i));if(v>best){best=v;pick=i;}}
const Zq2 selected=order.GetValue(pick),physical=selected/4u;
const float d=kernel.solveY_.GetValue(pick),mag=kernel.AbsF(d);
if(!(mag>=1.0e-30f&&mag<=1.0e30f))return false;
const float inv=kernel.SeedInverse(d);
const Zq2 old=order.GetValue(j);order.SetValue(j,selected);order.SetValue(pick,old);diagonal.SetValue(j,inv);
ZqB<Zq5::S_V>();
Zq3 zero[2]={Zq3(1u)<<physical,0u};Duplicate<float>(kernel.vr_[128u],0.0f,zero,Zq4(1),1,8);Zq1<PIPE_V>();
Brcb(kernel.block_,kernel.vr_[128u],Zq4(8),Zq9{1,8});
Muls(kernel.vr_[256u],aug[physical*136u],-inv,128u);Zq1<PIPE_V>();
MulAddDst<float,float>(aug,kernel.vr_[256u],kernel.block_,Zq3(64),Zq4(64),Zq0{1,1,0,17,0,1});
MulAddDst<float,float>(aug[64u],kernel.vr_[320u],kernel.block_,Zq3(64),Zq4(64),Zq0{1,1,0,17,0,1});
Zq1<PIPE_V>();ZqB<Zq5::V_S>();
}
auto offsets=kernel.panel_.template Zq6<int32_t>();
for(Zq2 i=0;i<64u;++i)physicalOffsets.SetValue(i,order.GetValue(i)*136u);
ZqB<Zq5::S_V>();
for(Zq2 c=0;c<64u;++c)Adds<int32_t>(offsets[c*64u],physicalOffsets.template Zq6<int32_t>(),int32_t(256u+c*4u),64u);
Zq1<PIPE_V>();Gather(kernel.panel_[8192u],aug,offsets.template Zq6<Zq2>(),0u,4096u);Zq1<PIPE_V>();
Mul(kernel.negative64_,kernel.panel_[8192u],diagonal,Zq3(64),Zq4(64),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
Abs(kernel.panel_[8192u],kernel.negative64_,4096u);Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(kernel.panel_[8192u],kernel.panel_[8192u],kernel.panel_[8192u+h],h);Zq1<PIPE_V>();}
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<64u;++i){float v=kernel.panel_[8192u].GetValue(i);if(!(v>=0.0f&&v<=1.0e30f))return false;}
kernel.augmented64Ready_=true;return true;
}

template<class Kernel>
__aicore__ static inline void EmitSquareDma(Kernel &kernel, Zq3 base,Zq2 batch)
{
{kernel.EmitSplitRhsDma(base,batch);return;}


}

template<class Kernel>
__aicore__ static inline void LoadRows200(Kernel &kernel, Zq3 base,Zq2 batch)
{
constexpr Zq2 n=200u,p=256u;
GlobalTensor<float>input;input.SetGlobalBuffer(kernel.a_+Zq3(batch)*n*n,n*n);
ZqB<Zq5::V_MTE2>();
DataCopy(kernel.factors_,input,n*n);ZqB<Zq5::MTE2_S>();
auto offsets=kernel.vr_[10u*p].template Zq6<Zq2>();
for(Zq2 row=0;row<n;++row)offsets.SetValue(row,row*n*4u);
}

template<class Kernel>
__aicore__ static inline void Rows64ToColumns(Kernel &kernel)
{
constexpr Zq2 n=64u,rp=72u,p=64u;
auto snapshot=kernel.panel_;
auto offsets=kernel.panel_[n*rp].template Zq6<int32_t>();
auto rowBase=kernel.rows_.template Zq6<int32_t>();
auto colBase=kernel.block_.template Zq6<int32_t>();
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<n;++i){rowBase.SetValue(i,int32_t(i*rp*4u));colBase.SetValue(i,int32_t(i*4u));}
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();


Adds(snapshot.template Zq6<int32_t>(),kernel.factors_.template Zq6<int32_t>(),int32_t(0),n*rp);
Zq1<PIPE_V>();
Brcb(kernel.factors_.template Zq6<int32_t>(),colBase,Zq4(8),Zq9{8,64});
Zq1<PIPE_V>();
Add<int32_t>(offsets,rowBase,kernel.factors_.template Zq6<int32_t>(),Zq3(64),Zq4(64),
Zq0{1,1,0,8,0,8});
Zq1<PIPE_V>();
for(Zq2 col=0;col<n;col+=8u){
Gather(kernel.block_,snapshot,offsets[col*64u].template Zq6<Zq2>(),0u,512u);Zq1<PIPE_V>();
Adds(kernel.factors_[col*72u],kernel.block_,0.0f,Zq3(64),Zq4(8),UnaryRepeatParams{1,1,9,8});Zq1<PIPE_V>();
}
Zq1<PIPE_V>();ZqB<Zq5::V_S>();
auto columnOffsets=kernel.vr_[10u*p].template Zq6<Zq2>();
for(Zq2 col=0;col<n;++col)columnOffsets.SetValue(col,col*72u*4u);
}

template<class Kernel>
__aicore__ static inline void EmitSplitRhsDma64(Kernel& kernel, Zq3 base, Zq2 batch)
{
{
const Zq3 matrixAddress=reinterpret_cast<Zq3>(kernel.x_+Zq3(batch)*64u*kernel.RhsWidth());
if(kernel.UseSplitRhs()&&kernel.RhsWidth()==2u&&(matrixAddress&31u)==0u){kernel.EmitSplitRhsAtomic64(base,batch);return;}
}
const Zq2 n=kernel.N();
{
if(kernel.producerPrivate_){


SetMaskNorm();ResetMask();
Brcb(kernel.output_,kernel.negative64_[4096u],Zq4(8),Zq9{1,8});
Duplicate(kernel.output_[512u],0.0f,Zq2(8));
ZqB<Zq5::V_MTE3>();
}else{
ZqB<Zq5::V_S>();
for(Zq2 row=0u;row<n;++row)kernel.output_.SetValue(row*8u,kernel.w_[kernel.OffX(base)+row]);
kernel.output_.SetValue(n*8u,0.0f);ZqB<Zq5::S_MTE3>();
}
}
GlobalTensor<float>gx;
gx.SetGlobalBuffer(kernel.x_+Zq3(batch)*n*kernel.RhsWidth()+kernel.currentRhs_,Zq3(n)*kernel.RhsWidth()-kernel.currentRhs_);
DataCopyPad(gx,kernel.output_,Zq8{uint16_t(n),4u,0u,(kernel.RhsWidth()-1u)*4u,0u});
const Zq3 residual=Zq3(batch)*kernel.RhsWidth()+kernel.currentRhs_;
if(kernel.r_!=nullptr&&residual<kernel.tiling_.residualElements){
GlobalTensor<float>gr;gr.SetGlobalBuffer(kernel.r_+residual,1u);
DataCopyPad(gr,kernel.output_[n*8u],Zq8{1u,4u,0u,0u,0u});
}
ZqB<Zq5::MTE3_S>();
}

template<bool Cooperative, class Kernel>
__aicore__ static inline void ProcessFixedSquare(Kernel& kernel, Zq2 taskIndex)
{
    const Zq3 per = kernel.PerBatch();
    const Zq2 width = kernel.RhsWidth();
    const Zq2 tasks = kernel.tiling_.batches * width;
    const Zq2 stride = GetBlockNum();
    for (Zq2 task = taskIndex; task < tasks; task += stride) {
        const Zq2 batch = task / width;
        kernel.currentRhs_ = task % width;
        const Zq3 base = Zq3(task) * per;
        kernel.Load(base, batch);
        if constexpr (Cooperative) {
            kernel.SolveCooperativeTask(base, batch);
        } else {
            kernel.DecomposeSquare(base);
            kernel.ColumnTriangularAll(base, kernel.w_ + kernel.OffB(base),
                kernel.w_ + kernel.OffX(base), kernel.K());
            kernel.Refine(base);
            EmitSquareDma(kernel, base, batch);
        }
    }
}


template<class Kernel>
__aicore__ static inline void CopySplitRhsOutput(Kernel& kernel, Zq2 batch, Zq2 n)
{
    const Zq2 width = kernel.RhsWidth();
    const Zq2 rhs = kernel.currentRhs_;
    const Zq3 start = Zq3(batch) * width + rhs;
    GlobalTensor<float> x;
    x.SetGlobalBuffer(kernel.x_ + Zq3(batch)*n*width + rhs, Zq3(n)*width - rhs);
    DataCopyPad(x, kernel.output_, Zq8{uint16_t(n), 4u, 0u, (width-1u)*4u, 0u});
    if (kernel.r_ != nullptr && start < kernel.tiling_.residualElements) {
        GlobalTensor<float> residual;
        residual.SetGlobalBuffer(kernel.r_ + start, 1u);
        DataCopyPad(residual, kernel.output_[n*8u], Zq8{1u, 4u, 0u, 0u, 0u});
    }
    ZqB<Zq5::MTE3_S>();
}

template<class Kernel>
__aicore__ static inline void CopyAtomicRhs64(Kernel& kernel, Zq2 batch,
    Zq7<int32_t> indices, Zq7<int32_t> minimum, Zq7<int32_t> maximum)
{
Gather(minimum,kernel.negative64_[4096u].template Zq6<int32_t>(),indices.template Zq6<Zq2>(),0u,128u);Zq1<PIPE_V>();
const Zq3 other=kernel.currentRhs_==0u?0xaaaaaaaaaaaaaaaaull:0x5555555555555555ull;
Zq3 otherColumns[2]={other,0u};
SetMaskNorm();ResetMask();
Duplicate<int32_t>(minimum,int32_t(2147483647),otherColumns,Zq4(2),1,8);
ZqB<Zq5::V_MTE3>();
GlobalTensor<int32_t>gx;
gx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kernel.x_+Zq3(batch)*128u),128u);


SetAtomicMin<int32_t>();ZqB<Zq5::S_MTE3>();
DataCopy(gx,minimum,128u);


Adds<int32_t>(maximum,minimum,int32_t(0),Zq2(128));Zq1<PIPE_V>();
Duplicate<int32_t>(maximum,int32_t(-2147483647-1),otherColumns,Zq4(2),1,8);
Duplicate(kernel.output_[512u],0.0f,Zq2(8));

ZqB<Zq5::MTE3_S>();
ZqB<Zq5::V_MTE3>();
SetAtomicMax<int32_t>();ZqB<Zq5::S_MTE3>();
DataCopy(gx,maximum,128u);ZqB<Zq5::MTE3_S>();
SetAtomicNone();ZqB<Zq5::S_MTE3>();

const Zq3 residual=Zq3(batch)*kernel.RhsWidth()+kernel.currentRhs_;
if(kernel.r_!=nullptr&&residual<kernel.tiling_.residualElements){
GlobalTensor<float>gr;gr.SetGlobalBuffer(kernel.r_+residual,1u);
DataCopyPad(gr,kernel.output_[512u],Zq8{1u,4u,0u,0u,0u});
}
ZqB<Zq5::MTE3_S>();
}


template<class Kernel>
__aicore__ static inline void CopyAtomicRhs50(Kernel& kernel, Zq2 batch, Zq2 firstRow, Zq2 tailRow,
    Zq7<int32_t> indices, Zq7<int32_t> minimum, Zq7<int32_t> maximum)
{
Gather(minimum,kernel.negative64_[4096u].template Zq6<int32_t>(),indices.template Zq6<Zq2>(),firstRow*4u,96u);Zq1<PIPE_V>();
const Zq3 other=kernel.currentRhs_==0u?0xaaaaaaaaaaaaaaaaull:0x5555555555555555ull;
Zq3 otherColumns[2]={other,0u};
Duplicate<int32_t>(minimum,int32_t(2147483647),otherColumns,Zq4(2),1,8);
ZqB<Zq5::V_MTE3>();
GlobalTensor<int32_t>gx;
gx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kernel.x_+Zq3(batch)*100u+firstRow*2u),96u);
SetAtomicMin<int32_t>();ZqB<Zq5::S_MTE3>();DataCopy(gx,minimum,96u);

Adds<int32_t>(maximum,minimum,int32_t(0),Zq2(96));Zq1<PIPE_V>();
Duplicate<int32_t>(maximum,int32_t(-2147483647-1),otherColumns,Zq4(2),1,8);
ZqB<Zq5::MTE3_S>();ZqB<Zq5::V_MTE3>();
SetAtomicMax<int32_t>();ZqB<Zq5::S_MTE3>();DataCopy(gx,maximum,96u);
ZqB<Zq5::MTE3_S>();SetAtomicNone();ZqB<Zq5::S_MTE3>();
GlobalTensor<float>tail;
tail.SetGlobalBuffer(kernel.x_+Zq3(batch)*100u+tailRow*2u+kernel.currentRhs_,4u-kernel.currentRhs_);
DataCopyPad(tail,kernel.output_,Zq8{2u,4u,0u,4u,0u});
const Zq3 residual=Zq3(batch)*kernel.RhsWidth()+kernel.currentRhs_;
if(kernel.r_!=nullptr&&residual<kernel.tiling_.residualElements){
GlobalTensor<float>gr;gr.SetGlobalBuffer(kernel.r_+residual,1u);
DataCopyPad(gr,kernel.output_[16u],Zq8{1u,4u,0u,0u,0u});
}
ZqB<Zq5::MTE3_S>();
}


template<class Kernel>
__aicore__ static inline void ConsumeRowInverse64(Kernel& kernel)
{
ZqB<Zq5::V_S>();
if(!kernel.producerCorrection_){ZqB<Zq5::S_V>();Adds(kernel.solveY_,kernel.negative64_[4224u],0.0f,64u);Zq1<PIPE_V>();}
if(kernel.rowInverse64Ready_){
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();

auto products=kernel.negative64_;
Mul(products,kernel.panel_[24576u+64u],kernel.solveY_,Zq3(64),Zq4(64),Zq0{1,1,1,8,17,0});Zq1<PIPE_V>();

for(Zq2 width=32u;width>=8u;width/=2u){
Add(products,products,products[width],Zq3(width),Zq4(64),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}

Gather(kernel.block_,products,kernel.residualTile_,0u,512u);Zq1<PIPE_V>();

for(Zq2 width=256u;width>=64u;width/=2u){Add(kernel.block_,kernel.block_,kernel.block_[width],width);Zq1<PIPE_V>();}
Adds(kernel.solveY_,kernel.block_,0.0f,64u);ZqB<Zq5::V_S>();return;
}
ZqB<Zq5::S_V>();Brcb(kernel.block_,kernel.solveY_,Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
auto products=kernel.panel_[24576u];
Mul(products,kernel.negative64_,kernel.block_,Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(products,products,products[h],h);Zq1<PIPE_V>();}
Adds(kernel.solveY_,products,0.0f,64u);ZqB<Zq5::V_S>();
}


__aicore__ static inline void CopySquareOutput(Zq7<float> output, const __gm__ float* solution,
    __gm__ float* x, __gm__ float* r, Zq2 batch, Zq2 n, Zq2 k, Zq3 residualElements)
{
const Zq2 count=n*k,zeroOffset=(count+7u)/8u*8u;
for(Zq2 i=0;i<count;++i)output.SetValue(i,solution[i]);
const Zq3 start=Zq3(batch)*k;
const Zq2 remain=r!=nullptr&&start<residualElements?
Zq2(residualElements-start<k?residualElements-start:k):0u;
for(Zq2 i=0;i<remain;++i)output.SetValue(zeroOffset+i,0.0f);
ZqB<Zq5::S_MTE3>();
GlobalTensor<float>gx;gx.SetGlobalBuffer(x+Zq3(batch)*count,count);
DataCopyPad(gx,output,Zq8{1u,count*4u,0u,0u,0u});
if(remain){
GlobalTensor<float>gr;gr.SetGlobalBuffer(r+start,remain);
DataCopyPad(gr,output[zeroOffset],Zq8{1u,remain*4u,0u,0u,0u});
}
ZqB<Zq5::MTE3_S>();
}

template<class Kernel>
__aicore__ static inline void PrepareAtomicRhs50(Kernel& kernel, Zq3 base, Zq2 firstRow, Zq2 tailRow)
{
ZqB<Zq5::V_S>();
if(!kernel.producerPrivate_){
for(Zq2 row=0u;row<50u;++row)kernel.negative64_.SetValue(4096u+row,kernel.GetX(base,row,0u));
}
kernel.output_.SetValue(0u,kernel.negative64_[4096u].GetValue(tailRow));
kernel.output_.SetValue(8u,kernel.negative64_[4096u].GetValue(tailRow+1u));
kernel.output_.SetValue(16u,0.0f);
}

template<class Kernel>
__aicore__ static inline void PrepareSplitRhsDmaOutput(Kernel& kernel, Zq3 base, Zq2 n)
{
if(kernel.producerPrivate_){
ZqB<Zq5::V_S>();
Brcb(kernel.output_,kernel.negative64_[4096u],Zq4(6),Zq9{1,8});
kernel.output_.SetValue(48u*8u,kernel.negative64_[4096u].GetValue(48u));
kernel.output_.SetValue(49u*8u,kernel.negative64_[4096u].GetValue(49u));
}else{
ZqB<Zq5::V_S>();
for(Zq2 row=0;row<n;++row)kernel.output_.SetValue(row*8u,kernel.w_[kernel.OffX(base)+row]);
}
kernel.output_.SetValue(n*8u,0.0f);
}

template<class Kernel>
__aicore__ static inline void InitializeVectorResidual(Kernel& kernel, Zq2 pitch)
{
auto mask=kernel.vr_[kernel.ResidualMaskPlane()*pitch].template Zq6<Zq2>();
Duplicate(mask,Zq2(0xfffff000u),pitch);
Duplicate(kernel.vr_,0.0f,pitch);
Duplicate(kernel.vr_[kernel.ResidualZeroPlane()*pitch],0.0f,pitch);
Duplicate(kernel.vr_[pitch],0.0f,pitch);
}

__aicore__ static inline void CopySquareMatrix256(Zq7<float> block,
    __gm__ float* inputAddress, __gm__ float* targetAddress)
{
GlobalTensor<float>input,target;
input.SetGlobalBuffer(inputAddress,256u*256u);
target.SetGlobalBuffer(targetAddress,256u*256u);
DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(target);
ZqB<Zq5::S_MTE2>();
for(Zq2 row=0;row<256u;row+=32u){
DataCopy(block,input[row*256u],32u*256u);
ZqB<Zq5::MTE2_MTE3>();
DataCopy(target[row*256u],block,32u*256u);
ZqB<Zq5::MTE3_MTE2>();
}
}

template<class Kernel>
__aicore__ static inline void LoadColumnFactors(Kernel& kernel,
    __gm__ float* inputAddress, Zq2 n)
{
for(Zq2 col=0;col<n;++col)for(Zq2 i=0;i<8u;++i)
kernel.residualTile_.SetValue(col*8u+i,(i*kernel.FactorPitch()+col)*4u);
Duplicate(kernel.factors_,0.0f,n*kernel.FactorPitch());
ZqB<Zq5::S_V>();
GlobalTensor<float>input;input.SetGlobalBuffer(inputAddress,Zq3(n)*n);
for(Zq2 row=0;row<n;row+=8u){
const Zq2 count=n-row<8u?n-row:8u;
ZqB<Zq5::V_MTE2>();
DataCopyPad(kernel.block_,input[row*n],Zq8{uint16_t(count),n*4u,0u,0u,0u},
DataCopyPadExtParams<float>{true,0u,Zq4(kernel.FactorPitch()-n),0.0f});
ZqB<Zq5::MTE2_V>();
Gather(kernel.vr_,kernel.block_,kernel.residualTile_,0u,8u*n);
Zq1<PIPE_V>();
Adds(kernel.factors_[row],kernel.vr_,0.0f,Zq3(count),Zq4(n),
UnaryRepeatParams{1,1,Zq4(kernel.FactorPitch()/8u),1});
}
ResetMask();
Duplicate(kernel.vr_,0.0f,kernel.VectorPitch());
Duplicate(kernel.vr_[kernel.VectorPitch()],0.0f,kernel.VectorPitch());
Duplicate(kernel.vr_[kernel.ResidualZeroPlane()*kernel.VectorPitch()],0.0f,kernel.VectorPitch());
ZqB<Zq5::V_S>();
auto offsets=kernel.vr_[10u*kernel.VectorPitch()].template Zq6<Zq2>();
for(Zq2 col=0;col<n;++col)offsets.SetValue(col,col*kernel.FactorPitch()*4u);
}

template<class Kernel>
__aicore__ static inline void InitializeGlobalBuffers(Kernel& kernel,
    GM_ADDR a, GM_ADDR b, GM_ADDR x, GM_ADDR res, GM_ADDR workspace)
{
kernel.a_=reinterpret_cast<__gm__ float*>(a);
kernel.b_=reinterpret_cast<__gm__ float*>(b);
kernel.x_=reinterpret_cast<__gm__ float*>(x);
kernel.r_=reinterpret_cast<__gm__ float*>(res);
const Zq3 user=reinterpret_cast<Zq3>(AscendC::GetUserWorkspace(workspace));
kernel.w_=reinterpret_cast<__gm__ float*>((user+63u)&~Zq3(63u));
}

template<class Kernel>
__aicore__ static inline bool TryEmitSplitRhsAtomic50(Kernel& kernel, Zq3 base, Zq2 batch)
{
const Zq3 address=reinterpret_cast<Zq3>(kernel.x_+Zq3(batch)*100u);
const Zq2 alignment=Zq2(address&31u);
if(kernel.UseSplitRhs()&&kernel.RhsWidth()==2u&&(alignment==0u||alignment==16u)){
kernel.EmitSplitRhsAtomic50(base,batch,alignment==0u?0u:2u);
return true;
}
return false;
}

template<class Kernel>
__aicore__ static inline void EmitColumn(Kernel& kernel, Zq3 base, Zq2 batch, Zq2 col,
    bool withResidual, Zq2 m, Zq2 n, Zq2 k)
{
for(Zq2 row=0;row<n;++row)
kernel.x_[Zq3(batch)*n*k+Zq3(row)*k+col]=kernel.GetX(base,row,col);
if(withResidual&&m>n&&kernel.r_!=nullptr){
float residual=0.0f;
for(Zq2 row=n;row<m;++row){
const float value=kernel.w_[kernel.OffB(base)+Zq3(row)*k+col];
residual+=value*value;
}
kernel.r_[Zq3(batch)*k+col]=residual;
}
}

struct OutputSchedule {
    bool serial;
    Zq3 residualCount;
};

__aicore__ static inline OutputSchedule CheckOutputSchedule(
    const __gm__ float* x, const __gm__ float* r, Zq2 m, Zq2 n, Zq2 k,
    Zq2 batches, Zq3 residualElements, bool dmaOutputs)
{
const Zq3 xAddress=reinterpret_cast<Zq3>(x);
const Zq3 rAddress=reinterpret_cast<Zq3>(r);
const Zq3 xCount=Zq3(batches)*n*k;
const Zq3 rCount=residualElements<Zq3(batches)*k?
residualElements:Zq3(batches)*k;
const bool outputLinesOverlap=r!=nullptr&&rCount>0u&&
(xAddress&~Zq3(63u))<=((rAddress+rCount*4u-1u)&~Zq3(63u))&&
(rAddress&~Zq3(63u))<=((xAddress+xCount*4u-1u)&~Zq3(63u));
const bool serial=!dmaOutputs&&(outputLinesOverlap||(xAddress&63u)!=0u||
(Zq3(n)*k)%16u!=0u||
(m>n&&residualElements>0u&&r!=nullptr&&
((rAddress&63u)!=0u||k%16u!=0u)));
return {serial, rCount};
}

__aicore__ static inline void EmitZeroResidual200(Zq7<float> block,__gm__ float* r,
    Zq2 batch,Zq2 rhs,Zq3 residualElements,bool active)
{
const Zq3 index=Zq3(batch)*4u+rhs;
if(active&&r!=nullptr&&index<residualElements){
ZqB<Zq5::V_S>();block.SetValue(384u,0.0f);ZqB<Zq5::S_MTE3>();
GlobalTensor<float>gr;gr.SetGlobalBuffer(r+index,1u);
DataCopyPad(gr,block[384u],Zq8{1u,4u,0u,0u,0u});
}
}


};
