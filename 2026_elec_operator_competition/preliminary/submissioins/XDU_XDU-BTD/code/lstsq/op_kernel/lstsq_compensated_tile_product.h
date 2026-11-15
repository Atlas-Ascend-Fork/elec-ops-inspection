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

// Split each product into its rounded FP32 value and its correction.
// The caller owns the counter mask and the tile views for all nine operands.
__aicore__ inline void CompensatedTileProduct(
    Zq7<float> high, Zq7<float> low, Zq7<float> scratch,
    Zq7<float> left, Zq7<float> leftHigh, Zq7<float> leftLow,
    Zq7<float> right, Zq7<float> rightHigh, Zq7<float> rightLow)
{
    Mul<float,false>(high,left,right,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Mul<float,false>(low,leftHigh,rightHigh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Zq1<PIPE_V>();
    Sub<float,false>(low,low,high,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Mul<float,false>(scratch,leftHigh,rightLow,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Zq1<PIPE_V>();
    Add<float,false>(low,low,scratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Zq1<PIPE_V>();
    Mul<float,false>(scratch,leftLow,rightHigh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Zq1<PIPE_V>();
    Add<float,false>(low,low,scratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Zq1<PIPE_V>();
    Mul<float,false>(scratch,leftLow,rightLow,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Zq1<PIPE_V>();
    Add<float,false>(low,low,scratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
    Zq1<PIPE_V>();
}
