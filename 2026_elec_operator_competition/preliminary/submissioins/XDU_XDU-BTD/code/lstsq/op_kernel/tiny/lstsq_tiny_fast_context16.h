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
class KernelTinyFastContext16 : public FixedSquareLayout<16u, 1u, 16u> {
static_assert(TinyN == 16u, "This kernel supports its registered shape only");
public:
__aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t){
static_assert(TinyN==15u||TinyN==16u,"Unsupported tiny shape");
t.m=TinyN;t.n=TinyN;t.k=1u;
InitBuffers(a,b,x,res,t.residualElements);
}
__aicore__ inline void InitImplicitGeometry(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
Zq3 residualCount){
static_assert(TinyN==15u||TinyN==16u,"Unsupported tiny shape");
InitBuffers(a,b,x,res,residualCount);
}
__aicore__ inline bool ProcessFastSingle(Zq2 batch){
tinyPrivateFlow_=true;
const Zq3 base=Zq3(batch)*PerBatch();Load(base,batch);
if(!BuildTinyInverseFast(base)){Zq1<PIPE_ALL>();return false;}
ConsumeTinyInverse<true>();
RefineTinyPrivate(base);EmitTinyPrivate(batch);return true;
}
private:
__aicore__ inline void InitBuffers(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,Zq3 residualCount){
pipe_.InitBuffer(arenaBuf_,(5760u+TinyN*24u)*sizeof(float));
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !(defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
auto arena=Zq7<float>(TPosition::VECCALC,0u,5760u+TinyN*24u);
#else
auto arena=arenaBuf_.template Get<float>();
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !(defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
tinyNegative_=Zq7<float>(TPosition::VECCALC,(4936u+TinyN*24u)*sizeof(float),304u);
#else
tinyNegative_=arena[4936u+TinyN*24u];
#endif


residualElements_=residualCount;

output_=arena[0u];


#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !(defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
factors_=Zq7<float>(TPosition::VECCALC,(4936u+TinyN*24u+304u)*sizeof(float),520u);
#else
factors_=arena[4936u+TinyN*24u+304u];
#endif
rows_=arena[24u+TinyN*16u];
block_=arena[232u+TinyN*16u];
panel_=arena[360u+TinyN*16u];
residualTile_=arena[4712u+TinyN*16u].template Zq6<Zq2>();


vr_=arena[4712u+TinyN*24u];

solveY_=arena[4920u+TinyN*24u];
residualElements_=residualCount;
a_=reinterpret_cast<__gm__ float*>(a);
b_=reinterpret_cast<__gm__ float*>(b);
x_=reinterpret_cast<__gm__ float*>(x);
r_=reinterpret_cast<__gm__ float*>(res);
}
Zq7<float>tinyNegative_;
bool tinyInverseReady_=false,tinyPrivateFlow_=false;


__aicore__ inline void PrepareTinyInverseOffsets(Zq7<Zq2> pairMap, Zq7<int32_t> pivotBytes,
    Zq7<int32_t> features, Zq7<int32_t> pivotOffsets)
{
auto rowBytes=tinyNegative_[64u].template Zq6<int32_t>();
auto broadcast=block_.template Zq6<int32_t>();
Brcb(broadcast,features,Zq4(2),Zq9{1,8});
Brcb(pairMap[40u].template Zq6<int32_t>(),pivotBytes,Zq4(1),Zq9{12,96});Zq1<PIPE_V>();
Add<int32_t>(pairMap[24u].template Zq6<int32_t>(),rowBytes,broadcast,Zq3(16),Zq4(16),Zq0{1,1,0,6,0,1});
Adds<int32_t>(pairMap.template Zq6<int32_t>(),pairMap[40u].template Zq6<int32_t>(),int32_t(4),Zq3(8),Zq4(8),UnaryRepeatParams{1,1,12,12});
Adds<int32_t>(pairMap[48u].template Zq6<int32_t>(),pairMap[40u].template Zq6<int32_t>(),int32_t(68),Zq3(8),Zq4(8),UnaryRepeatParams{1,1,12,12});Zq1<PIPE_V>();
Zq3 finalDiagonal[2]={0x8040201008040201ull,0u};
Duplicate<Zq2>(pairMap[24u],Zq2(256u*4u),finalDiagonal,Zq4(1),6u,8u);
Duplicate<Zq2>(pairMap[416u],Zq2(256u*4u),finalDiagonal,Zq4(1),6u,8u);
Brcb(broadcast,rowBytes,Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
Add<int32_t>(pairMap[8u].template Zq6<int32_t>(),features,broadcast,Zq3(16),Zq4(16),Zq0{1,1,0,6,0,1});

Muls<int32_t>(pivotOffsets,features,int32_t(96),Zq3(8),Zq4(4),UnaryRepeatParams{1,1,2,0});Zq1<PIPE_V>();
Duplicate<Zq2>(pairMap[8u],Zq2(264u*4u),finalDiagonal,Zq4(1),6u,8u);
Duplicate<Zq2>(pairMap[400u],Zq2(264u*4u),finalDiagonal,Zq4(1),6u,8u);
Adds<int32_t>(pivotOffsets,pivotOffsets,int32_t(160),Zq3(8),Zq4(4),UnaryRepeatParams{1,1,2,2});Zq1<PIPE_V>();
Adds<int32_t>(pivotOffsets[8u],pivotOffsets,int32_t(32),Zq3(8),Zq4(4),UnaryRepeatParams{1,1,2,2});
}

__aicore__ inline bool BuildTinyInverseFast(Zq3 base){
constexpr Zq2 n=16u;
auto pairMap=panel_[11u*256u].template Zq6<Zq2>();
auto pivotBytes=tinyNegative_[288u].template Zq6<int32_t>();
auto features=tinyNegative_[272u].template Zq6<int32_t>();
auto pivotOffsets=tinyNegative_[64u].template Zq6<int32_t>();
SetMaskNorm();ResetMask();

Abs<float>(panel_[13u*256u],panel_[14u*256u],Zq3(64),Zq4(4),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();


Muls<float>(tinyNegative_,tinyNegative_,-1.0f,Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});
Muls<int32_t>(pivotBytes,features,int32_t(34),Zq3(8),Zq4(1),UnaryRepeatParams{1,1,8,8});
ReduceMax(vr_[16u],panel_[13u*256u],panel_[12u*256u],int32_t(256),true);ZqB<Zq5::V_S>();
const float inputScale=vr_[16u].GetValue(0u);
if(!(inputScale>=1.0e-30f&&inputScale<=1.0e30f))return false;
const Zq2 maximumBits=vr_[16u].template Zq6<Zq2>().GetValue(0u);
const int32_t exponent=int32_t((maximumBits>>23u)&255u)-127;
const int32_t shift=-80-exponent;
if(shift<-120||shift>120)return false;
union{Zq2 u;float f;}scale;scale.u=Zq2(127+shift)<<23u;
float pivotLower=(inputScale*(1.0f/4096.0f))*scale.f;
const float originalLower=2.0e-30f*scale.f;
if(pivotLower<originalLower)pivotLower=originalLower;
if(pivotLower<0x1p-120f)pivotLower=0x1p-120f;
const float pivotUpper=0x1p-32f;
if(!(pivotLower<=pivotUpper))return false;
int32_t boundExponent=7-exponent;if(boundExponent>95)boundExponent=95;
union{Zq2 u;float f;}stabilityBound;stabilityBound.u=Zq2(127+boundExponent)<<23u;

PrepareTinyInverseOffsets(pairMap, pivotBytes, features, pivotOffsets);
Muls<float>(factors_,panel_[14u*256u],scale.f,Zq3(64),Zq4(4),UnaryRepeatParams{1,1,8,8});
Duplicate<float>(factors_[256u],-1.0f,Zq3(8),Zq4(1),1u,8u);
Duplicate<float>(factors_[264u],1.0f,Zq3(8),Zq4(1),1u,8u);
Zq1<PIPE_V>();

TinyNormalizedActualPair16<true>(0u);
for(Zq2 j=2u;j<16u;j+=2u){TinyNormalizedActualPair16<false>(j);}
ResetMask();SetMaskNorm();


Gather(block_,panel_,pivotOffsets.template Zq6<Zq2>(),0u,64u);
Muls<float>(factors_,factors_,scale.f,Zq3(64),Zq4(4),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
Abs<float>(panel_[12u*256u],block_,Zq2(64));
Abs<float>(panel_[13u*256u],factors_,Zq3(64),Zq4(4),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();

auto normPacked=panel_[11u*256u].template Zq6<int32_t>();
Compares<float,Zq4>(normPacked.template Zq6<Zq4>(),panel_[13u*256u],stabilityBound.f,CMPMODE::LE,Zq2(256));
Compares<float,Zq4>(normPacked[8u].template Zq6<Zq4>(),panel_[12u*256u],pivotLower,CMPMODE::GE,Zq2(64));
Compares<float,Zq4>(normPacked[16u].template Zq6<Zq4>(),panel_[12u*256u],pivotUpper,CMPMODE::LE,Zq2(64));Zq1<PIPE_V>();
auto normBounded=rows_.template Zq6<Zq4>();
Compares<int32_t,Zq4>(normBounded,normPacked,int32_t(-1),CMPMODE::EQ,Zq2(64));
ZqB<Zq5::V_S>();
if((normBounded.template Zq6<Zq2>().GetValue(0u)&Zq2(0x000303ffu))==Zq2(0x000303ffu)){
tinyInverseReady_=true;return true;
}
return false;
}
template<bool PrepareOriginalA>
__aicore__ __attribute__((always_inline))inline void TinyNormalizedActualPair16(Zq2 j){
auto packet=panel_[(j/2u)*96u];
auto indices=panel_[11u*256u+(j/2u)*96u].template Zq6<Zq2>();
Gather(packet,factors_,indices,0u,88u);Zq1<PIPE_V>();
Div<float>(vr_[64u],packet,packet[40u],Zq3(24),Zq4(1),Zq0{1,1,0,8,8,0});
Brcb(block_,packet[24u],Zq4(2),Zq9{1,8});
if constexpr(PrepareOriginalA){Gather(panel_[15u*256u],panel_[14u*256u],panel_[16u*256u].template Zq6<Zq2>(),0u,256u);}
Zq1<PIPE_V>();
Muls<float>(vr_[64u],vr_[64u],-1.0f,Zq3(24),Zq4(1),UnaryRepeatParams{1,1,8,8});
if constexpr(PrepareOriginalA){And<uint16_t>(panel_[4u*256u].template Zq6<uint16_t>(),panel_[15u*256u].template Zq6<uint16_t>(),panel_[10u*256u].template Zq6<uint16_t>(),Zq3(128),Zq4(4),Zq0{1,1,1,8,8,0});}
Zq1<PIPE_V>();
Zq3 rowMask[2]={Zq3(0xffffffu)^(Zq3(1u)<<(j+9u)),0u};
Zq3 colMask[2]={Zq3(0xffffu)^(Zq3(1u)<<(j+1u)),0u};
MulAddDst<float,float>(packet[48u],vr_[64u],block_[(j+1u)*8u],rowMask,Zq4(1),Zq0{1,1,0,8,8,0});
MulAddDst<float,float>(packet[72u],packet[24u],vr_[64u],colMask,Zq4(1),Zq0{1,1,0,8,8,0});
if constexpr(PrepareOriginalA){Sub<float>(panel_[5u*256u],panel_[15u*256u],panel_[4u*256u],Zq3(64),Zq4(4),Zq0{1,1,1,8,8,8});}
Zq1<PIPE_V>();


Div<float>(vr_[32u],packet[56u],packet[48u],Zq3(16),Zq4(1),Zq0{1,1,0,8,8,0});
MulAddDst<float,float>(factors_,vr_[72u],block_,Zq3(16),Zq4(16),Zq0{1,1,0,2,0,1});Zq1<PIPE_V>();
Muls<float>(vr_[32u],vr_[32u],-1.0f,Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});
Brcb(block_,packet[72u],Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
MulAddDst<float,float>(factors_,vr_[32u],block_,Zq3(16),Zq4(16),Zq0{1,1,0,2,0,1});Zq1<PIPE_V>();
}
template<bool Initial>
__aicore__ inline void ConsumeTinyInverse(){
static_assert(TinyN==16u,"Direct canonical row consumer is16-only");
auto input=tinyNegative_;
auto reduced=vr_[64u];
if constexpr(Initial){reduced=vr_[16u];}
else{input=solveY_;}
SetMaskNorm();ResetMask();
auto products=panel_[13u*256u];
Mul<float>(products,factors_,input,Zq3(16),Zq4(16),Zq0{1,1,1,2,2,0});Zq1<PIPE_V>();
BlockReduceSum<float>(block_,products,4,64,1,1,8);Zq1<PIPE_V>();

PairReduceSum<float>(reduced,block_,1,32,1,1,8);Zq1<PIPE_V>();
}
__aicore__ inline void RefineTinyPrivate(Zq3 base){
constexpr Zq2 p=256u,pitch=16u;auto iterate=vr_[16u];


for(Zq2 step=0;step<2u;++step){
for(Zq2 rhs=0;rhs<1u;++rhs){
PrepareResidualX(base,rhs);
auto x=rhs?panel_[10u*p]:panel_[p];


Brcb(x,vr_[pitch],Zq4(6),Zq9{2,16});
Brcb(x[8u],vr_[pitch],Zq4(6),Zq9{2,16});Zq1<PIPE_V>();
}
TinyResidualTile(base,0u,0u,step!=0u);
ConsumeTinyInverse<false>();
Sub(iterate,iterate,vr_[64u],16u);Zq1<PIPE_V>();
}
}
__aicore__ inline void EmitTinyPrivate(Zq2 batch){
auto iterate=vr_[16u];


Muls<float>(iterate,iterate,-1.0f,Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});
Duplicate<float>(output_,0.0f,Zq3(1),Zq4(1),1u,8u);
ZqB<Zq5::V_MTE3>();
GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*16u,16u);
DataCopyPad(gx,iterate,Zq8{1u,64u,0u,0u,0u});
if(r_!=nullptr&&Zq3(batch)<residualElements_){
GlobalTensor<float>gr;gr.SetGlobalBuffer(r_+batch,1u);
DataCopyPad(gr,output_,Zq8{1u,4u,0u,0u,0u});
}


#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !defined(__ASCENDC_ENABLE_SUPER_KERNEL__))
ZqB<Zq5::MTE3_S>();
#endif
}
__aicore__ inline void Load(Zq3 base,Zq2 batch){
static_assert(TinyN==16u,"Full input producer is16-only");
tinyInverseReady_=false;auto input=panel_[14u*256u];
auto features=tinyNegative_[272u].template Zq6<int32_t>();
// No A/B DMA is pending yet: Scalar seeds cannot wait on those inputs.
#pragma unroll 8
for(Zq2 i=0u;i<8u;++i)features.SetValue(i,int32_t(i));
GlobalTensor<float>original;original.SetGlobalBuffer(a_+Zq3(batch)*256u,256u);


#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !defined(__ASCENDC_ENABLE_SUPER_KERNEL__))
ZqB<Zq5::V_MTE2>();
#endif
DataCopyPad(input,original,Zq8{1u,1024u,0u,0u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
GlobalTensor<float>originalB;originalB.SetGlobalBuffer(b_+Zq3(batch)*TinyN,TinyN);

DataCopyPad(tinyNegative_,originalB,Zq8{1u,TinyN*4u,0u,0u,0u},DataCopyPadExtParams<float>{TinyN==15u,0u,Zq4(16u-TinyN),0.0f});

auto transpose=panel_[16u*256u].template Zq6<Zq2>();
auto rowBytes=tinyNegative_[64u].template Zq6<int32_t>();

#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !defined(__ASCENDC_ENABLE_SUPER_KERNEL__))
ZqB<Zq5::S_V>();
#endif
SetMaskNorm();ResetMask();
// Same SDK seed publication, Adds8, internal and external PIPE retirements.
ZqB<Zq5::S_V>();
Adds<int32_t>(features[8u],features,int32_t(8),Zq3(8),Zq4(1),UnaryRepeatParams{1,1,8,8});
Zq1<PIPE_V>();Zq1<PIPE_V>();
Muls<int32_t>(features,features,int32_t(4),Zq2(16));Zq1<PIPE_V>();
Muls<int32_t>(rowBytes,features,int32_t(16),Zq3(16),Zq4(1),UnaryRepeatParams{1,1,2,0});Zq1<PIPE_V>();
Brcb(transpose.template Zq6<int32_t>(),features,Zq4(2),Zq9{2,16});
Brcb(transpose[8u].template Zq6<int32_t>(),features,Zq4(2),Zq9{2,16});Zq1<PIPE_V>();
Add<int32_t>(transpose.template Zq6<int32_t>(),transpose.template Zq6<int32_t>(),rowBytes,
Zq3(16),Zq4(16),Zq0{1,1,1,2,2,0});Zq1<PIPE_V>();


{
auto mask=panel_[10u*256u].template Zq6<Zq2>();
Duplicate(mask,Zq2(0xfffff000u),64u);


}
ZqB<Zq5::MTE2_V>();

Adds<int32_t>(panel_[6u*256u+240u].template Zq6<int32_t>(),tinyNegative_.template Zq6<int32_t>(),int32_t(0),Zq2(16));


}
__aicore__ inline void TinyResidualTile(Zq3 base,Zq2 row,Zq2 slot,bool cachedA){
constexpr Zq2 p=256u,active=p,groupWidth=1u;
auto work=panel_,extra=panel_[10u*p],input=panel_[14u*p+slot*p];
Zq7<float>a=work[15u*p],ah=work[4u*p],al=work[5u*p],u=work[9u*p];


(void)cachedA;

TinyResidualProducts(work, a, ah, al);
Add<float,false>(work[2032u],work[1760u],work[2016u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
TinyResidualRounding(work);

Add<float,false>(work[1008u],work[1008u],work[2048u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Add<float,false>(solveY_,work[752u],work[1008u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
return;
}
__aicore__ inline void PrepareResidualX(Zq3 base,Zq2 col){
const Zq2 p=VectorPitch();
Zq7<float>x=vr_[p],xh=vr_[2u*p],xl=vr_[3u*p];

And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
panel_[10u*256u].template Zq6<uint16_t>(),2u*p);
Zq1<PIPE_V>();
Sub(xl,x,xh,p);
Zq1<PIPE_V>();
}


__gm__ float*a_=nullptr,*b_=nullptr,*x_=nullptr,*r_=nullptr;
TBuf<TPosition::VECCALC>arenaBuf_;
TPipe pipe_;
#if (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
TBuf<TPosition::VECCALC>factorBuf_,rowBuf_,blockBuf_,panelBuf_,residualTileBuf_;
#endif
Zq7<float>factors_,rows_,block_,panel_;
Zq7<Zq2>residualTile_;


#if (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
TBuf<TPosition::VECCALC>solveBuf_;
#endif
Zq7<float>solveY_;

#if (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
TBuf<TPosition::VECCALC>outputBuf_;
#endif
#if (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
TBuf<TPosition::VECCALC>residualBuf_;
#endif
Zq7<float>output_,vr_;


Zq3 residualElements_;

};
