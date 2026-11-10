/*
 * Copyright (c) 2026 aodebiao.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "kernel_operator.h"

// Compensated two-component FP32 arithmetic, entirely on the AI Core.
// No FP64 instructions or host-side tensor calculations are used.
struct Wide {
    float h,l;
    __aicore__ inline Wide(float x=0.0f):h(x),l(0.0f) {}
    __aicore__ inline Wide(float x,float y):h(x),l(y) {}
};
#pragma clang fp contract(off) reassociate(off)
__aicore__ inline float WProduct(float a,float b) {return a*b;}
__aicore__ inline Wide WNormalize(float a,float b) {
    const float s=a+b;return Wide(s,b-(s-a));
}
__aicore__ inline Wide operator+(Wide a,Wide b) {
    const float s=a.h+b.h,v=s-a.h;
    const float e=(a.h-(s-v))+(b.h-v);
    return WNormalize(s,e+(a.l+b.l));
}
__aicore__ inline Wide operator-(Wide a) {return Wide(-a.h,-a.l);}
__aicore__ inline Wide operator-(Wide a,Wide b) {return a+(-b);}
__aicore__ inline Wide operator*(Wide a,Wide b) {
    const float p=WProduct(a.h,b.h);
    const float ca=WProduct(4097.0f,a.h),cb=WProduct(4097.0f,b.h);
    const float ah=ca-(ca-a.h),al=a.h-ah,bh=cb-(cb-b.h),bl=b.h-bh;
    const float err=((WProduct(ah,bh)-p)+WProduct(ah,bl)+WProduct(al,bh))+WProduct(al,bl);
    return WNormalize(p,err+(WProduct(a.h,b.l)+WProduct(a.l,b.h)));
}
// Compensated quotient; the caller supplies a nonzero normalized divisor.
__aicore__ inline Wide WQuotient(Wide a,Wide b) {
    const float magnitude=b.h<0.0f?-b.h:b.h;
    if (magnitude<1.0e-30f || magnitude>1.0e30f) {
        const float q=a.h/b.h;Wide r=a-b*Wide(q);
        const float t=r.h/b.h;Wide result=Wide(q)+Wide(t);
        r=a-b*result;return result+Wide(r.h/b.h);
    }
    const float inverse=1.0f/b.h;
    const float q=WProduct(a.h,inverse);Wide r=a-b*Wide(q);
    const float t=WProduct(r.h,inverse);Wide result=Wide(q)+Wide(t);
    r=a-b*result;return result+Wide(WProduct(r.h,inverse));
}
__aicore__ inline Wide operator/(Wide a,Wide b) {return WQuotient(a,b);}
#pragma clang fp contract(on)
__aicore__ inline bool operator<(Wide a,Wide b) {return a.h<b.h || (a.h==b.h && a.l<b.l);}
__aicore__ inline bool operator>(Wide a,Wide b) {return b<a;}
__aicore__ inline bool operator<=(Wide a,Wide b) {return !(b<a);}
__aicore__ inline bool operator>=(Wide a,Wide b) {return !(a<b);}
__aicore__ inline bool operator==(Wide a,Wide b) {return a.h==b.h && a.l==b.l;}
__aicore__ inline bool operator!=(Wide a,Wide b) {return !(a==b);}
struct WideView {
    AscendC::LocalTensor<float> data;uint32_t base;
    __aicore__ inline WideView(AscendC::LocalTensor<float> t,uint32_t b=0U):data(t),base(b) {}
    __aicore__ inline Wide GetValue(uint32_t i) {return Wide(data.GetValue(base+2U*i),data.GetValue(base+2U*i+1U));}
    __aicore__ inline void SetValue(uint32_t i,Wide x) {data.SetValue(base+2U*i,x.h);data.SetValue(base+2U*i+1U,x.l);}
    __aicore__ inline WideView operator[](uint32_t i) {return WideView(data,base+2U*i);}
};

static constexpr uint32_t CAP=512U;
static constexpr uint32_t BYTES=2048U;
