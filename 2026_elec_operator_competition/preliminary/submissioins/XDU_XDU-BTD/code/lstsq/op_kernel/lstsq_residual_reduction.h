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

// One stage of the compensated pairwise reduction. The high planes alternate
// between stages. The low plane accumulates each TwoSum error in place.
template<Zq2 Offset>
__aicore__ inline void MergeCounterResidualHalves(Zq7<float> highOut, Zq7<float> highIn, Zq7<float> low,
    Zq7<float> rounding, Zq7<float> scratch, Zq2 active)
{
ZqA<float,MaskMode::COUNTER>(0u,active);
Add<float,false>(highOut,highIn,highIn[Offset],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Add<float,false>(low,low,low[Offset],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(rounding,highOut,highIn,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(scratch,highOut,rounding,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(scratch,highIn,scratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Sub<float,false>(rounding,highIn[Offset],rounding,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(scratch,scratch,rounding,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(low,low,scratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
}

template<Zq2 Offset, Zq2 Count, Zq4 Repeat, Zq4 Stride>
__aicore__ inline void MergeStridedResidualHalves(Zq7<float> highOut, Zq7<float> highIn, Zq7<float> low,
    Zq7<float> rounding, Zq7<float> scratch)
{
Add<float>(highOut[0u],highIn[0u],highIn[Offset][0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Add<float>(low[0u],low[0u],low[Offset][0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Sub<float>(rounding[0u],highOut[0u],highIn[0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Sub<float>(scratch[0u],highOut[0u],rounding[0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Sub<float>(scratch[0u],highIn[0u],scratch[0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Sub<float>(rounding[0u],highIn[Offset][0u],rounding[0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Add<float>(scratch[0u],scratch[0u],rounding[0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Add<float>(low[0u],low[0u],scratch[0u],Zq3(Count),Zq4(Repeat),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
}

__aicore__ inline void MergeLegacyResidualHalves(Zq7<float> highOut, Zq7<float> highIn, Zq7<float> low,
    Zq7<float> rounding, Zq7<float> scratch, Zq2 offset, Zq2 active)
{
ZqA<float,MaskMode::COUNTER>(0u,active);
Add<float,false>(highOut,highIn,highIn[offset],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(rounding,highOut,highIn,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(scratch,highOut,rounding,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(scratch,highIn,scratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(rounding,highIn[offset],rounding,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(scratch,scratch,rounding,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(low,low,low[offset],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(low,low,scratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
}


__aicore__ inline void TinyResidualProducts(Zq7<float> work, Zq7<float> a,
    Zq7<float> ah, Zq7<float> al)
{
constexpr Zq2 p=256u;
auto left=work[6u*p],right=work[7u*p],low=work[8u*p];
auto x=work[p],xh=work[2u*p],xl=work[3u*p];
SetMaskNorm();ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffffffffffffffull));
Mul<float,false>(left,a,x,Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});
Mul<float,false>(right,a[128u],x[128u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});
Mul<float,false>(low,ah,xh,Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(low,low,left,Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});
Sub<float,false>(low[128u],low[128u],right,Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();

MulAddDst<float,float,false>(low,ah,xl,Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
MulAddDst<float,float,false>(low,al,xh,Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
MulAddDst<float,float,false>(low,al,xl,Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();


Add<float,false>(work[1664u],work[1536u],work[1792u],Zq3(0),Zq4(2),Zq0{1,1,1,32,8,8});
Add<float,false>(work[2048u],work[2048u],work[2176u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffffffull));
Add<float,false>(work[1728u],work[1664u],work[1920u],Zq3(0),Zq4(2),Zq0{1,1,1,32,4,4});
Zq1<PIPE_V>();

ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffull));
Add<float,false>(work[1760u],work[1728u],work[1984u],Zq3(0),Zq4(2),Zq0{1,1,1,32,2,2});
Zq1<PIPE_V>();
}

__aicore__ inline void TinyResidualRounding(Zq7<float> work)
{
ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffffffffffffffull));
Add<float,false>(work[512u],work[1536u],work[1792u],Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Sub<float,false>(work[256u],work[512u],work[1536u],Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Sub<float,false>(work[2304u],work[512u],work[256u],Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Sub<float,false>(work[2304u],work[1536u],work[2304u],Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});
Sub<float,false>(work[256u],work[1792u],work[256u],Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Add<float,false>(work[768u],work[2304u],work[256u],Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Add<float,false>(work[2048u],work[2048u],work[768u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Add<float,false>(work[2048u],work[2048u],work[2112u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Add<float,false>(work[2048u],work[2048u],work[896u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffffffull));
Add<float,false>(work[2048u],work[2048u],work[2080u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Add<float,false>(work[2048u],work[2048u],work[960u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffull));
Add<float,false>(work[2048u],work[2048u],work[2064u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();

Add<float,false>(work[2048u],work[2048u],work[992u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
}


template<Zq4 Rows, Zq4 Stride, Zq2 RightCount>
__aicore__ inline void MergeStridedResidualFirstHalves(Zq7<float> hi, Zq7<float> lo,
    Zq7<float> ah, Zq7<float> al, Zq7<float> u)
{
Add<float>((ah)[0u],(hi)[0u],(hi[128u])[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Add<float>((lo)[0u],(lo)[0u],(lo[128u])[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Add<float>((ah)[64u],(hi)[64u],(hi[128u])[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Add<float>((lo)[64u],(lo)[64u],(lo[128u])[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Sub<float>((al)[0u],(ah)[0u],(hi)[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Sub<float>((al)[64u],(ah)[64u],(hi)[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Sub<float>((u)[0u],(ah)[0u],(al)[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Sub<float>((u)[64u],(ah)[64u],(al)[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Sub<float>((u)[0u],(hi)[0u],(u)[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Sub<float>((al)[0u],(hi[128u])[0u],(al)[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Sub<float>((u)[64u],(hi)[64u],(u)[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Sub<float>((al)[64u],(hi[128u])[64u],(al)[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Add<float>((u)[0u],(u)[0u],(al)[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Add<float>((u)[64u],(u)[64u],(al)[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
Add<float>((lo)[0u],(lo)[0u],(u)[0u],Zq3(64),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Add<float>((lo)[64u],(lo)[64u],(u)[64u],Zq3(RightCount),Zq4(Rows),Zq0{1,1,1,Stride,Stride,Stride});
Zq1<PIPE_V>();
}

__aicore__ inline void WriteCompensatedResidual(
    Zq7<float> out, Zq7<float> bb, Zq7<float> hi, Zq7<float> lo,
    Zq7<float> neg, Zq7<float> ss, Zq7<float> vv, Zq7<float> tt, Zq2 count)
{
Muls(neg,hi,-1.0f,count);Zq1<PIPE_V>();
Add(ss,bb,neg,count);Zq1<PIPE_V>();
Sub(vv,ss,bb,count);Zq1<PIPE_V>();
Sub(tt,ss,vv,count);Zq1<PIPE_V>();
Sub(tt,bb,tt,count);Sub(neg,neg,vv,count);Zq1<PIPE_V>();
Add(tt,tt,neg,count);Zq1<PIPE_V>();
Sub(tt,tt,lo,count);Zq1<PIPE_V>();
Add(out,ss,tt,count);Zq1<PIPE_V>();
}
