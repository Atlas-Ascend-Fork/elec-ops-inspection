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
#include "linalg_eigh_wide.h"

template<class Kernel> class EighCore {
public:
    __aicore__ inline Kernel &Self() {return static_cast<Kernel&>(*this);}
    // The caller retains its zero-norm/failure policy. Inspect the FP32
    // components directly so the zero condition is explicit for Wide values.
    __aicore__ inline Wide DivideOrZero(Wide numerator,Wide divisor) {
        if (divisor.h==0.0f && divisor.l==0.0f) return Wide(0);
        return WQuotient(numerator,divisor);
    }
    __aicore__ inline void InitializeBuffers(bool small,bool resident,bool keepTridiagonal) {
        auto &op=Self();
        op.pipe_.InitBuffer(op.in_,2,BYTES);op.pipe_.InitBuffer(op.out_,2,BYTES);
        op.pipe_.InitBuffer(op.vBuf_,BYTES);op.pipe_.InitBuffer(op.wBuf_,BYTES);
        op.pipe_.InitBuffer(op.tmp_,2*BYTES);op.pipe_.InitBuffer(op.red_,32);op.pipe_.InitBuffer(op.redWork_,BYTES);
        op.pipe_.InitBuffer(op.meta_,4*BYTES);op.pipe_.InitBuffer(op.offsets_,BYTES);
        if (keepTridiagonal) {op.pipe_.InitBuffer(op.triOriginal_,2U*BYTES);op.pipe_.InitBuffer(op.orientationWork_,13U*BYTES);}
        if (small) {
            op.pipe_.InitBuffer(op.leafMatrix_,((2U*op.n_*op.n_*4U+31U)/32U)*32U);
            op.pipe_.InitBuffer(op.leafMeta_,4U*BYTES);op.pipe_.InitBuffer(op.leafWork_,4U*BYTES);
        }
        if (resident) {op.pipe_.InitBuffer(op.sCache_,op.n_*op.pitch_*4U);op.pipe_.InitBuffer(op.qCache_,op.n_*op.pitch_*4U);}
        else op.pipe_.InitBuffer(op.tile_,32U*op.pitch_*4U);
    }
    __aicore__ inline float Abs(float x) {return x<0?-x:x;}
    __aicore__ inline float Root(float x) {return Self().WideRootSeed(x);}
    __aicore__ inline float Hypot(float a,float b) {
        a=Self().Abs(a);b=Self().Abs(b);const float hi=a>b?a:b;
        if (!(hi>0)) return 0;
        const float lo=(a>b?b:a)/hi;return hi*Self().Root(1.0f+lo*lo);
    }
    __aicore__ inline void VS() {
        auto e=Self().pipe_.FetchEventID(AscendC::HardEvent::V_S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(e);AscendC::WaitFlag<AscendC::HardEvent::V_S>(e);
    }
    __aicore__ inline void SV() {
        auto e=Self().pipe_.FetchEventID(AscendC::HardEvent::S_V);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(e);AscendC::WaitFlag<AscendC::HardEvent::S_V>(e);
    }
    __aicore__ inline float Dot(AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b) {
        auto tmp=Self().tmp_.template Get<float>();AscendC::Mul(tmp,a,b,Self().n_);
        AscendC::ReduceSum(Self().red_.template Get<float>(),tmp,Self().redWork_.template Get<float>(),Self().n_);
        return AscendC::GetAccVal<float>();
    }
    __aicore__ inline float MaxAbs(AscendC::LocalTensor<float> a) {
        auto tmp=Self().tmp_.template Get<float>();AscendC::Abs(tmp,a,Self().n_);
        AscendC::ReduceMax(Self().red_.template Get<float>(),tmp,Self().redWork_.template Get<float>(),Self().n_,false);
        Self().VS();return Self().red_.template Get<float>().GetValue(0);
    }
    __aicore__ inline AscendC::LocalTensor<float> Load(uint32_t r,bool q=false) {
        auto row=Self().in_.template AllocTensor<float>();
        if (Self().UseCache(q)) {
            auto cache=q?Self().qCache_.template Get<float>():Self().sCache_.template Get<float>();
            AscendC::Muls(row,cache[r*Self().pitch_],1.0f,Self().n_);
        } else {
            auto e=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(e);AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(e);
            AscendC::DataCopyExtParams cp{1,Self().n_*4U,0,0,0};
            AscendC::DataCopyPadExtParams<float> pad{false,0,0,0.0f};
            if (q) AscendC::DataCopyPad(row,Self().qt_[r*Self().pitch_],cp,pad);
            else AscendC::DataCopyPad(row,Self().s_[r*Self().pitch_],cp,pad);
        }
        Self().in_.template EnQue<float>(row);return Self().in_.template DeQue<float>();
    }
    __aicore__ inline void Store(uint32_t r,AscendC::LocalTensor<float> row,bool q=false) {
        if (Self().UseCache(q)) {
            auto cache=q?Self().qCache_.template Get<float>():Self().sCache_.template Get<float>();
            AscendC::Muls(cache[r*Self().pitch_],row,1.0f,Self().n_);return;
        }
        auto dst=Self().out_.template AllocTensor<float>();AscendC::Muls(dst,row,1.0f,Self().n_);
        Self().out_.template EnQue<float>(dst);dst=Self().out_.template DeQue<float>();
        AscendC::DataCopyExtParams cp{1,Self().n_*4U,0,0,0};
        if (q) AscendC::DataCopyPad(Self().qt_[r*Self().pitch_],dst,cp);
        else AscendC::DataCopyPad(Self().s_[r*Self().pitch_],dst,cp);
        Self().out_.FreeTensor(dst);
    }
    __aicore__ inline void Prepare(uint32_t b) {
        const uint64_t base=static_cast<uint64_t>(b)*Self().n_*Self().n_;
        Self().matrixScale_=0.0f;Self().inputCenter_=0.0f;Self().referenceCenter_=0.0f;
        Self().CenterDiagonal(base);
        if (Self().UseSmall()) Self().MeasureTriangleSmall(base); else Self().MeasureTriangleTiled(base);
        float scale=Self().matrixScale_;
        if (!(scale>0.0f)) scale=1.0f;
        Self().matrixScale_=scale;
        Self().referenceCenter_=Self().inputCenter_/scale;
        auto row=Self().vBuf_.template Get<float>();
        for (uint32_t r=0;r<Self().n_;++r) {
            if (Self().ScaleWorkspace()) {auto a=Self().Load(r);AscendC::Muls(a,a,1.0f/scale,Self().n_);Self().Store(r,a);Self().in_.FreeTensor(a);}
            AscendC::Duplicate(row,0.0f,Self().n_);Self().VS();row.SetValue(r,1.0f);Self().SV();Self().Store(r,row,true);
        }
        auto d=Self().meta_.template Get<float>();Self().VS();
        AscendC::Duplicate(d,0.0f,4U*CAP);
        Self().SV();
    }
    // The scaling origin is the mean of the diagonal; only large orders use it.
    __aicore__ inline void CenterDiagonal(uint64_t base) {
        if (!(Self().n_>25U)) return;
        float diagonalScale=0.0f;
        for (uint32_t i=0;i<Self().n_;++i) {
            const float magnitude=Self().Abs(Self().a_.GetValue(base+i*Self().n_+i));
            if (magnitude>diagonalScale) diagonalScale=magnitude;
        }
        if (!(diagonalScale>0.0f)) return;
        Wide sum(0),count(0);const Wide inverse=Wide(1)/Wide(diagonalScale);
        for (uint32_t i=0;i<Self().n_;++i) {sum=sum+Wide(Self().a_.GetValue(base+i*Self().n_+i))*inverse;count=count+Wide(1);}
        const Wide center=count!=Wide(0)?(sum/count)*Wide(diagonalScale):Wide(0);
        Self().inputCenter_=center.h+center.l;
    }
    // The identity cache and output tile are not live yet. Reuse them
    // for a strided column gather while respecting the selected triangle.
    __aicore__ inline void MeasureTriangleTiled(uint64_t base) {
        auto column=Self().UseResident()?Self().qCache_.template Get<float>():Self().tile_.template Get<float>();
        for (uint32_t r=0;r<Self().n_;++r) {
            auto event=Self().pipe_.FetchEventID(AscendC::HardEvent::S_MTE2);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event);
            event=Self().pipe_.FetchEventID(AscendC::HardEvent::V_MTE2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            auto input=Self().in_.template AllocTensor<float>();
            AscendC::DataCopyExtParams direct{1,Self().n_*4U,0,0,0};
            AscendC::DataCopyPadExtParams<float> padding{false,0,0,0.0f};
            AscendC::DataCopyPad(input,Self().a_[base+r*Self().n_],direct,padding);
            const uint32_t begin=Self().lower_?r+1U:0U;
            const uint32_t count=Self().lower_?Self().n_-r-1U:r;
            if (count>0U) {
                AscendC::DataCopyExtParams gather{static_cast<uint16_t>(count),4U,(Self().n_-1U)*4U,0,0};
                AscendC::DataCopyPad(column,Self().a_[base+begin*Self().n_+r],gather,padding);
            }
            event=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE2_S);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event);
            for (uint32_t i=0;i<count;++i) input.SetValue(begin+i,column.GetValue(i*8U));
            input.SetValue(r,input.GetValue(r)-Self().inputCenter_);
            Self().SV();Self().in_.template EnQue<float>(input);input=Self().in_.template DeQue<float>();
            const float mag=Self().MaxAbs(input);if (mag>Self().matrixScale_) Self().matrixScale_=mag;
            Self().Store(r,input);Self().in_.FreeTensor(input);
        }
    }
    __aicore__ inline void Tridiagonalize() {
        auto d=Self().meta_.template Get<float>();
        auto v=Self().vBuf_.template Get<float>();
        for (uint32_t step=0;step+2U<Self().n_;++step) {
            const uint32_t j=Self().lower_?step:Self().n_-1U-step;
            const uint32_t first=Self().lower_?j+1U:0U;
            const uint32_t end=Self().lower_?Self().n_:j;
            const uint32_t pivot=Self().lower_?j+1U:j-1U;
            const uint32_t off=Self().lower_?j:j-1U;
            auto row=Self().Load(j);Self().VS();d.SetValue(j,row.GetValue(j));
            AscendC::Muls(v,row,1.0f,Self().n_);Self().in_.FreeTensor(row);Self().VS();
            float t=0.0f;
            if (Self().BuildReflector(j,first,end,pivot,off,t)) Self().ApplyReflector(first,end,t);
            Self().Store(j,v);
        }
        Self().FinalizeTridiagonal();
    }
    // Householder vector of one step. Returns false when the trailing block is
    // already zero, in which case the step only has to be stored back.
    __aicore__ inline bool BuildReflector(uint32_t j,uint32_t first,uint32_t end,uint32_t pivot, uint32_t off,float &t) {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];auto tau=d[2U*CAP];
        auto v=Self().vBuf_.template Get<float>();
        for (uint32_t c=0;c<Self().n_;++c) if (c<first || c>=end) v.SetValue(c,0.0f);
        Self().SV();const float scale=Self().MaxAbs(v);
        if (!(scale>0.0f)) {tau.SetValue(j,0.0f);e.SetValue(off,0.0f);return false;}
        AscendC::Muls(v,v,1.0f/scale,Self().n_);
        const float length=Self().Root(Self().Dot(v,v));Self().VS();const float x0=v.GetValue(pivot);
        if (!(length>0.0f)) {tau.SetValue(j,0.0f);e.SetValue(off,0.0f);return false;}
        const float alpha=x0>=0?-length:length,divisor=x0-alpha;
        if (!(divisor!=0.0f)) {tau.SetValue(j,0.0f);e.SetValue(off,0.0f);return false;}
        t=(alpha-x0)/alpha;
        AscendC::Muls(v,v,1.0f/divisor,Self().n_);Self().VS();v.SetValue(pivot,1.0f);Self().SV();
        e.SetValue(off,alpha*scale);tau.SetValue(j,t);
        return true;
    }
    // w = t*S*v, then the symmetric rank-two update S -= v*w' + w*v'.
    __aicore__ inline void ApplyReflector(uint32_t first,uint32_t end,float t) {
        auto v=Self().vBuf_.template Get<float>();auto w=Self().wBuf_.template Get<float>();auto tmp=Self().tmp_.template Get<float>();
        AscendC::Duplicate(w,0.0f,Self().n_);Self().VS();
        if (Self().UseCache(false)) {
            auto cache=Self().sCache_.template Get<float>();
            for (uint32_t r=first;r<end;++r) w.SetValue(r,t*Self().Dot(cache[r*Self().pitch_],v));
        } else {
            for (uint32_t base=first;base<end;base+=32U) {
                const uint32_t height=end-base<32U?end-base:32U;
                Self().ReadTile(Self().s_,base,height);auto tile=Self().tile_.template Get<float>();
                for (uint32_t r=0;r<height;++r) w.SetValue(base+r,t*Self().Dot(tile[r*Self().pitch_],v));
            }
        }
        Self().SV();const float beta=-0.5f*t*Self().Dot(v,w);
        AscendC::Muls(tmp,v,beta,Self().n_);AscendC::Add(w,w,tmp,Self().n_);Self().VS();
        if (Self().UseCache(false)) {
            auto cache=Self().sCache_.template Get<float>();
            for (uint32_t r=first;r<end;++r) {
                const float vr=v.GetValue(r),wr=w.GetValue(r);auto input=cache[r*Self().pitch_];
                AscendC::Muls(tmp,w,vr,Self().n_);AscendC::Sub(input,input,tmp,Self().n_);
                AscendC::Muls(tmp,v,wr,Self().n_);AscendC::Sub(input,input,tmp,Self().n_);
            }
        } else {
            for (uint32_t base=first;base<end;base+=32U) {
                const uint32_t height=end-base<32U?end-base:32U;
                Self().ReadTile(Self().s_,base,height);auto tile=Self().tile_.template Get<float>();
                for (uint32_t r=0;r<height;++r) {
                    const float vr=v.GetValue(base+r),wr=w.GetValue(base+r);auto input=tile[r*Self().pitch_];
                    AscendC::Muls(tmp,w,vr,Self().n_);AscendC::Sub(input,input,tmp,Self().n_);
                    AscendC::Muls(tmp,v,wr,Self().n_);AscendC::Sub(input,input,tmp,Self().n_);
                }
                Self().WriteTile(Self().s_,base,height);
            }
        }
    }
    // The last two columns are read back once the sweep has finished.
    __aicore__ inline void FinalizeTridiagonal() {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];
        if (Self().n_>=2U) {
            const uint32_t r=Self().lower_?Self().n_-2U:1U;
            const uint32_t c=Self().lower_?Self().n_-1U:0U;
            auto row=Self().Load(r);Self().VS();d.SetValue(r,row.GetValue(r));
            e.SetValue(Self().lower_?Self().n_-2U:0U,row.GetValue(c));Self().in_.FreeTensor(row);
        }
        const uint32_t last=Self().lower_?Self().n_-1U:0U;
        auto row=Self().Load(last);Self().VS();d.SetValue(last,row.GetValue(last));e.SetValue(Self().n_-1U,0.0f);Self().in_.FreeTensor(row);
    }
    __aicore__ inline Wide Abs(Wide x) {return x<Wide(0)?-x:x;}
    __aicore__ inline float WideRootSeed(float x) {
        if (!(x>0.0f)) return 0.0f;
        union FloatBits {float value;uint32_t bits;} seed;
        seed.value=x;
        if (seed.bits>=0x7f800000U) return x;
        const bool subnormal=x<1.1754943508222875e-38f;
        if (subnormal) x*=16777216.0f;
        seed.value=x;seed.bits=(seed.bits>>1U)+0x1fc00000U;
        float g=seed.value;
        g=Self().NewtonRootStep(g,x);g=Self().NewtonRootStep(g,x);g=Self().NewtonRootStep(g,x);
        return subnormal?g*0.000244140625f:g;
    }
    // One Newton step of the square root, with an explicit divisor check.
    __aicore__ inline float NewtonRootStep(float g,float x) {
        return g!=0.0f?0.5f*(g+x/g):0.0f;
    }
    __aicore__ inline Wide Root(Wide x) {
        if (!(x>Wide(0))) return Wide(0);
        const Wide r(Self().WideRootSeed(x.h));
        if (!(r>Wide(0))) return Wide(0);
        return r+(x-r*r)/(Wide(2)*r);
    }
    __aicore__ inline Wide Hypot(Wide a,Wide b) {
        a=Self().Abs(a);b=Self().Abs(b);const Wide hi=a>b?a:b;
        if (!(hi>Wide(0))) return Wide(0);
        const Wide lo=(a>b?b:a)/hi;return hi*Self().Root(Wide(1)+lo*lo);
    }
    __aicore__ inline void ReadTile(AscendC::GlobalTensor<float> &source,uint32_t first,uint32_t height) {
        auto event=Self().pipe_.FetchEventID(AscendC::HardEvent::V_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        event=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(height),Self().n_*4U,(Self().pitch_-Self().n_)*4U,0,0};
        AscendC::DataCopyPadExtParams<float> padding{false,0,0,0.0f};
        AscendC::DataCopyPad(Self().tile_.template Get<float>(),source[first*Self().pitch_],copy,padding);
        event=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE2_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }
    __aicore__ inline void WriteTile(AscendC::GlobalTensor<float> &destination,uint32_t first,uint32_t height) {
        auto event=Self().pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(height),Self().n_*4U,0,(Self().pitch_-Self().n_)*4U,0};
        AscendC::DataCopyPad(destination[first*Self().pitch_],Self().tile_.template Get<float>(),copy);
    }
    __aicore__ inline void TiledBackTransform() {
        auto tau=Self().meta_.template Get<float>()[2U*CAP];auto v=Self().vBuf_.template Get<float>();auto tmp=Self().tmp_.template Get<float>();
        auto tile=Self().tile_.template Get<float>();
        for (uint32_t first=Self().VectorBegin();first<Self().VectorEnd();first+=32U) {
            const uint32_t height=Self().VectorEnd()-first<32U?Self().VectorEnd()-first:32U;
            Self().ReadTile(Self().qt_,first,height);
            for (uint32_t step=0;step+2U<Self().n_;++step) {
                const uint32_t j=Self().lower_?Self().n_-3U-step:step+2U;
                const float t=tau.GetValue(j);if (!(t>0)) continue;
                auto h=Self().Load(j);AscendC::Muls(v,h,1.0f,Self().n_);Self().in_.FreeTensor(h);
                for (uint32_t r=0;r<height;++r) {
                    auto row=tile[r*Self().pitch_];const float coefficient=t*Self().Dot(row,v);
                    AscendC::Muls(tmp,v,coefficient,Self().n_);AscendC::Sub(row,row,tmp,Self().n_);
                }
            }
            for (uint32_t r=0;r<height;++r) {
                auto row=tile[r*Self().pitch_];const float norm=Self().Root(Self().Dot(row,row));
                if (norm>0) AscendC::Muls(row,row,1.0f/norm,Self().n_);
            }
            Self().WriteTile(Self().qt_,first,height);
        }
    }
    __aicore__ inline void BackTransform() {
        if (!Self().UseResident()) {Self().TiledBackTransform();return;}
        auto tau=Self().meta_.template Get<float>()[2U*CAP];auto v=Self().vBuf_.template Get<float>();auto tmp=Self().tmp_.template Get<float>();
        auto q=Self().qCache_.template Get<float>();
        for (uint32_t step=0;step+2U<Self().n_;++step) {
            const uint32_t j=Self().lower_?Self().n_-3U-step:step+2U;
            const float t=tau.GetValue(j);if (!(t>0)) continue;
            auto h=Self().Load(j);AscendC::Muls(v,h,1.0f,Self().n_);Self().in_.FreeTensor(h);
            for (uint32_t r=0;r<Self().n_;++r) {
                auto row=q[r*Self().pitch_];const float coefficient=t*Self().Dot(row,v);
                AscendC::Muls(tmp,v,coefficient,Self().n_);AscendC::Sub(row,row,tmp,Self().n_);
            }
        }
        // Normalize each vector after accumulated rotations/reflections.
        for (uint32_t r=0;r<Self().n_;++r) {
            auto row=q[r*Self().pitch_];const float norm=Self().Root(Self().Dot(row,row));
            if (norm>0) AscendC::Muls(row,row,1.0f/norm,Self().n_);
        }
    }
    __aicore__ inline void Write(AscendC::GlobalTensor<float> &gm,uint64_t offset, AscendC::LocalTensor<float> input,uint32_t count) {
        auto row=Self().out_.template AllocTensor<float>();AscendC::Muls(row,input,1.0f,count);
        Self().out_.template EnQue<float>(row);row=Self().out_.template DeQue<float>();
        AscendC::DataCopyExtParams cp{1,count*4U,0,0,0};AscendC::DataCopyPad(gm[offset],row,cp);Self().out_.FreeTensor(row);
    }
    __aicore__ inline void Output(uint32_t b) {
        auto d=Self().meta_.template Get<float>();auto order=Self().meta_.template Get<uint32_t>()[3U*CAP];
        Self().VS();for (uint32_t i=0;i<Self().n_;++i) order.SetValue(i,i);
        for (uint32_t i=1;i<Self().n_;++i) {
            const float value=d.GetValue(i);const uint32_t index=order.GetValue(i);uint32_t j=i;
            while (j>0 && d.GetValue(j-1U)>value) {d.SetValue(j,d.GetValue(j-1U));order.SetValue(j,order.GetValue(j-1U));--j;}
            d.SetValue(j,value);order.SetValue(j,index);
        }
        Self().SV();auto tmp=Self().tmp_.template Get<float>();AscendC::Muls(tmp,d,Self().matrixScale_,Self().n_);
        if (Self().n_>25U) AscendC::Adds(tmp,tmp,Self().inputCenter_,Self().n_);
        Self().Write(Self().values_,static_cast<uint64_t>(b)*Self().n_,tmp,Self().n_);
        auto offsets=Self().offsets_.template Get<uint32_t>();const uint64_t base=static_cast<uint64_t>(b)*Self().n_*Self().n_;
        if (Self().UseResident()) {
            for (uint32_t i=0;i<Self().n_;++i) offsets.SetValue(i,order.GetValue(i)*Self().pitch_*4U);Self().SV();
            for (uint32_t r=0;r<Self().n_;++r) {
                AscendC::Gather(tmp,Self().qCache_.template Get<float>(),offsets,r*4U,Self().n_);Self().Write(Self().vectors_,base+r*Self().n_,tmp,Self().n_);
            }
        } else {
            auto tile=Self().tile_.template Get<float>();
            for (uint32_t start=0;start<Self().n_;start+=32U) {
                const uint32_t width=Self().n_-start<32U?Self().n_-start:32U;
                auto ev=Self().pipe_.FetchEventID(AscendC::HardEvent::V_MTE2);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ev);AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev);
                ev=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev);AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
                AscendC::DataCopyExtParams cp{1,Self().n_*4U,0,0,0};AscendC::DataCopyPadExtParams<float> pad{false,0,0,0.0f};
                for (uint32_t j=0;j<width;++j) AscendC::DataCopyPad(tile[j*Self().pitch_],Self().qt_[order.GetValue(start+j)*Self().pitch_],cp,pad);
                ev=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE2_V);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
                for (uint32_t j=0;j<width;++j) offsets.SetValue(j,j*Self().pitch_*4U);Self().SV();
                for (uint32_t r=0;r<Self().n_;++r) {
                    AscendC::Gather(tmp,tile,offsets,r*4U,width);Self().Write(Self().vectors_,base+r*Self().n_+start,tmp,width);
                }
            }
        }
    }
    __aicore__ inline void PublishState() {
        auto header=Self().tmp_.template Get<float>();Self().VS();
        header.SetValue(0U,Self().matrixScale_);header.SetValue(1U,Self().splitSign_);
        header.SetValue(2U,Self().referenceBlocks_?1.0f:0.0f);
        header.SetValue(3U,Self().inputCenter_);header.SetValue(4U,Self().referenceCenter_);Self().SV();
        auto event=Self().pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopy(Self().shared_,Self().meta_.template Get<float>(),4U*CAP);
        if (Self().KeepSpectrum()) AscendC::DataCopy(Self().shared_[3U*CAP],Self().meta_.template Get<float>(),CAP);
        AscendC::DataCopy(Self().shared_[4U*CAP],Self().triOriginal_.template Get<float>(),2U*CAP);
        AscendC::DataCopy(Self().shared_[6U*CAP],header,8U);
    }
    __aicore__ inline void ReadState(bool immutable=false) {
        auto event=Self().pipe_.FetchEventID(AscendC::HardEvent::V_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        event=Self().pipe_.FetchEventID(AscendC::HardEvent::S_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event);
        if (immutable) {
            AscendC::DataCopy(Self().meta_.template Get<float>()[CAP],Self().shared_[CAP],3U*CAP);
            AscendC::DataCopy(Self().meta_.template Get<float>(),Self().shared_[3U*CAP],CAP);
        } else AscendC::DataCopy(Self().meta_.template Get<float>(),Self().shared_,4U*CAP);
        AscendC::DataCopy(Self().triOriginal_.template Get<float>(),Self().shared_[4U*CAP],2U*CAP);
        AscendC::DataCopy(Self().tmp_.template Get<float>(),Self().shared_[6U*CAP],8U);
        event=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event);
        event=Self().pipe_.FetchEventID(AscendC::HardEvent::MTE2_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        auto header=Self().tmp_.template Get<float>();Self().matrixScale_=header.GetValue(0U);Self().splitSign_=header.GetValue(1U);
        Self().referenceBlocks_=header.GetValue(2U)>0.0f;
        Self().inputCenter_=header.GetValue(3U);Self().referenceCenter_=header.GetValue(4U);
    }
    // Small path: rows are staged into the scratch buffer while being scanned.
    __aicore__ inline void MeasureTriangleSmall(uint64_t base) {
        auto row=Self().vBuf_.template Get<float>();
        for (uint32_t r=0;r<Self().n_;++r) {
            Self().VS();
            for (uint32_t c=0;c<Self().n_;++c) {
                const uint32_t rr=Self().lower_?(r>=c?r:c):(r<=c?r:c);
                const uint32_t cc=Self().lower_?(r>=c?c:r):(r<=c?c:r);
                const float value=Self().a_.GetValue(base+rr*Self().n_+cc);
                row.SetValue(c,value);const float mag=Self().Abs(value);
                if (mag>Self().matrixScale_) Self().matrixScale_=mag;
            }
            Self().SV();Self().Store(r,row);
        }
    }
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<float> a_,values_,vectors_,s_,qt_;
    AscendC::TQue<AscendC::TPosition::VECIN,2> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT,2> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vBuf_,wBuf_,tmp_,red_,redWork_,meta_,offsets_,leafMatrix_,leafMeta_,leafWork_,triOriginal_,orientationWork_;
    AscendC::TBuf<AscendC::TPosition::VECIN> sCache_,qCache_,tile_;
    uint32_t n_=0,pitch_=0,batch_=0,lower_=1,core_=0,resident_=0;
    bool leafOrientation_=false,referenceBlocks_=false;uint32_t leafWidth_=0U;
    float matrixScale_=1.0f,splitSign_=0.0f,inputCenter_=0.0f,referenceCenter_=0.0f;
};
