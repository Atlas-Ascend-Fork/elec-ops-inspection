/*
 * Copyright (c) 2026 Delicate02. All rights reserved.
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
#include "fire_device_common.h"

namespace FireSoft53 {
constexpr uint64_t INV=0x40471547652b82feULL;
constexpr uint64_t COEFF[3]={0x3ebc6af84b912394ULL,0x3f2ebfce50fac4f3ULL,0x3f962e42ff0c52d6ULL};
constexpr uint64_t TAB[32]={0x3ff0000000000000ULL,0x3ff059b0d3158574ULL,0x3ff0b5586cf9890fULL,0x3ff11301d0125b51ULL,0x3ff172b83c7d517bULL,0x3ff1d4873168b9aaULL,0x3ff2387a6e756238ULL,0x3ff29e9df51fdee1ULL,0x3ff306fe0a31b715ULL,0x3ff371a7373aa9cbULL,0x3ff3dea64c123422ULL,0x3ff44e086061892dULL,0x3ff4bfdad5362a27ULL,0x3ff5342b569d4f82ULL,0x3ff5ab07dd485429ULL,0x3ff6247eb03a5585ULL,0x3ff6a09e667f3bcdULL,0x3ff71f75e8ec5f74ULL,0x3ff7a11473eb0187ULL,0x3ff82589994cce13ULL,0x3ff8ace5422aa0dbULL,0x3ff93737b0cdc5e5ULL,0x3ff9c49182a3f090ULL,0x3ffa5503b23e255dULL,0x3ffae89f995ad3adULL,0x3ffb7f76f2fb5e47ULL,0x3ffc199bdd85529cULL,0x3ffcb720dcef9069ULL,0x3ffd5818dcfba487ULL,0x3ffdfc97337b9b5fULL,0x3ffea4afa2a490daULL,0x3fff50765b6e4540ULL};
__aicore__ inline int topbits(uint32_t v){
 if(!v)return 0;int n=1;
 if(v>>16){v>>=16;n+=16;}if(v>>8){v>>=8;n+=8;}
 if(v>>4){v>>=4;n+=4;}if(v>>2){v>>=2;n+=2;}if(v>>1)++n;return n;
}
struct Big {
 uint32_t w[16]{};
 __aicore__ inline int used()const{int n=16;while(n&&w[n-1]==0)--n;return n;}
 __aicore__ inline int length()const{int n=used();return n?32*(n-1)+topbits(w[n-1]):0;}
 __aicore__ inline bool bit(int b)const{return b>=0&&b<512&&((w[b/32]>>(b%32))&1);}
 __aicore__ inline bool below(int b)const{
 if(b<=0)return false;int q=b/32,s=b%32;
 for(int i=0;i<q&&i<16;++i)if(w[i])return true;
 return q<16&&s&&(w[q]&((uint32_t(1)<<s)-1));
}
 static __aicore__ inline Big of(uint64_t a){Big r;r.w[0]=uint32_t(a);r.w[1]=uint32_t(a>>32);return r;}
 __aicore__ inline Big shift(int b)const{Big r;int n=used(),q=b/32,s=b%32;for(int i=0;i<n;++i){r.w[i+q]|=w[i]<<s;if(s && i+q+1<16)r.w[i+q+1]|=w[i]>>(32-s);}return r;}
 __aicore__ inline uint64_t take(int b)const{
 if(b<0||b>=512)return 0;int q=b/32,s=b%32;
 uint64_t a=w[q],c=q+1<16?w[q+1]:0,d=q+2<16?w[q+2]:0;
 return s?((a>>s)|(c<<(32-s))|(d<<(64-s))):(a|(c<<32));
}
};
__aicore__ inline int compare(const Big&a,const Big&b){for(int i=15;i>=0;--i)if(a.w[i]!=b.w[i])return a.w[i]>b.w[i]?1:-1;return 0;}
__aicore__ inline Big plus(const Big&a,const Big&b){Big r;uint64_t carry=0;for(int i=0;i<16;++i){uint64_t t=uint64_t(a.w[i])+b.w[i]+carry;r.w[i]=uint32_t(t);carry=t>>32;}return r;}
__aicore__ inline Big minus(const Big&a,const Big&b){Big r;uint64_t borrow=0;for(int i=0;i<16;++i){uint64_t t=uint64_t(b.w[i])+borrow;r.w[i]=uint32_t(uint64_t(a.w[i])-t);borrow=uint64_t(a.w[i])<t;}return r;}
__aicore__ inline Big product(uint64_t a,uint64_t b){Big r;uint32_t aw[2]={uint32_t(a),uint32_t(a>>32)},bw[2]={uint32_t(b),uint32_t(b>>32)};for(int i=0;i<2;++i){uint64_t carry=0;for(int j=0;j<2;++j){uint64_t t=uint64_t(aw[i])*bw[j]+r.w[i+j]+carry;r.w[i+j]=uint32_t(t);carry=t>>32;}r.w[i+2]=uint32_t(carry);}return r;}
struct D{uint64_t m=0;int e=0;bool neg=false;}; // magnitude m*2^e; nonzero normal binary64 m has 53 bits
__aicore__ inline D round53(Big m,int e,bool neg){int n=m.length();if(!n)return {};int cut=n-53;uint64_t sig;if(cut>0){sig=m.take(cut);if(m.bit(cut-1)&&(m.below(cut-1)||(sig&1)))++sig;e+=cut;if(sig==(uint64_t(1)<<53)){sig>>=1;++e;}}else{sig=m.take(0)<<(-cut);e+=cut;}return {sig,e,neg};}
__aicore__ inline D unpack(uint64_t u){int ef=int((u>>52)&2047);uint64_t m=u&0xfffffffffffffULL;if(!ef){return {};}return {(uint64_t(1)<<52)|m,ef-1023-52,bool(u>>63)};}
__aicore__ inline uint64_t pack(D d){return d.m?(uint64_t(d.neg)<<63)|(uint64_t(d.e+52+1023)<<52)|(d.m&0xfffffffffffffULL):0;}
__aicore__ inline D from32(uint32_t u){uint32_t f=u&0x7fffff;int ef=int((u>>23)&255);if(!ef&&!f)return {};uint64_t m=ef?f|0x800000:f;int e=ef?ef-127-23:-149;return round53(Big::of(m),e,bool(u>>31));}
__aicore__ inline D integer(int k){return round53(Big::of(uint64_t(k<0?-int64_t(k):k)),0,k<0);}
__aicore__ inline D combine(Big a,int ae,bool as,Big b,int be,bool bs){if(!a.used())return round53(b,be,bs);if(!b.used())return round53(a,ae,as);int e=(ae<be?ae:be);a=a.shift(ae-e);b=b.shift(be-e);if(as==bs)return round53(plus(a,b),e,as);int c=compare(a,b);return c>=0?round53(minus(a,b),e,as):round53(minus(b,a),e,bs);}
__aicore__ inline D add53(D a,D b){return combine(Big::of(a.m),a.e,a.neg,Big::of(b.m),b.e,b.neg);}
__aicore__ inline D mul53(D a,D b){return round53(product(a.m,b.m),a.e+b.e,a.neg!=b.neg);}
__aicore__ inline D fma53(D a,D b,D c){return combine(product(a.m,b.m),a.e+b.e,a.neg!=b.neg,Big::of(c.m),c.e,c.neg);}
__aicore__ inline int away(D z){if(!z.m)return 0;int s=-z.e;if(s<1||s>63){if(s>63)return 0;return 0;}uint64_t k=z.m>>s;uint64_t mask=(uint64_t(1)<<s)-1;if((z.m&mask)>=(uint64_t(1)<<(s-1)))++k;return z.neg?-int(k):int(k);}
__aicore__ inline uint32_t narrow32(D d){if(!d.m)return 0;int e=d.e+52; uint64_t q=d.m>>29,rem=d.m&((uint64_t(1)<<29)-1);if(rem>(uint64_t(1)<<28)||(rem==(uint64_t(1)<<28)&&(q&1)))++q;if(q==(uint64_t(1)<<24)){q>>=1;++e;}return(uint32_t(d.neg)<<31)|(uint32_t(e+127)<<23)|(uint32_t(q)&0x7fffff);}
__aicore__ inline uint32_t evaluate(uint32_t u){
 D x=from32(u),z=mul53(x,unpack(INV));int k=away(z);D r=add53(z,integer(-k));
 D p0=fma53(unpack(COEFF[0]),r,unpack(COEFF[1]));D p1=fma53(unpack(COEFF[2]),r,integer(1));
 D rr=mul53(r,r),y=fma53(p0,rr,p1);int q=k/32,j=k%32;if(j<0){j+=32;--q;}
 D scale=unpack(TAB[j]);scale.e+=q;return narrow32(mul53(y,scale));
}
}

// RN32 reciprocal for a positive normal denominator in the bounded center repair.
__aicore__ inline uint32_t CenterReciprocalBits(uint32_t denominator) {
    uint32_t exponent = (denominator >> 23) & 255;
    uint32_t mantissa = (denominator & 0x7fffff) | 0x800000;
    if (mantissa == 0x800000) return (254 - exponent) << 23;
    uint32_t quotient = 0, remainder = 0x800000;
    for (uint32_t bit = 0; bit < 24; ++bit) {
        remainder <<= 1;
        quotient <<= 1;
        if (remainder >= mantissa) { remainder -= mantissa; ++quotient; }
    }
    if (2 * remainder > mantissa || (2 * remainder == mantissa && (quotient & 1))) ++quotient;
    uint32_t outExponent = 253 - exponent;
    if (quotient == 0x1000000) { quotient >>= 1; ++outExponent; }
    return (outExponent << 23) | (quotient & 0x7fffff);
}

template<class T,bool Scaled>
__aicore__ inline __attribute__((always_inline)) void RepairCenterRows(LocalTensor<float>& values,LocalTensor<float>& priors,GlobalTensor<T>& raw,LocalTensor<float>& scratch,uint32_t pitch,uint32_t count,uint32_t rawOffset) {
    for (uint32_t channel = 0; channel < 2; ++channel) {
        ResetMask(); SetMaskNorm();
        if constexpr(Scaled) Adds(scratch, values[channel * pitch], 0.f, count);
        else Muls(scratch, values[channel * pitch], 2.f, count);
        PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, -.5f, count); PipeBarrier<PIPE_V>();
        Add(scratch, scratch, priors[channel * pitch], count); PipeBarrier<PIPE_V>();
        Abs(scratch, scratch, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1); WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for (uint32_t n = 0; n < count; ++n) {
            if (scratch.GetValue(n) >= 0x1p-4f ||
                priors.GetValue(2 * pitch + n) <= (sizeof(T) == 2 ? 4096.f : 32.f)) continue;
            float original = static_cast<float>(raw.GetValue(uint64_t(rawOffset) + n * 6 + channel));
            if (original > -20.f && original < 0.f) {
                union { float f; uint32_t u; } x;
                x.f = -original;
                uint32_t expBits = FireSoft53::evaluate(x.u);
                uint32_t denominator = FireSoft53::narrow32(FireSoft53::add53(
                    FireSoft53::from32(expBits), FireSoft53::integer(1)));
                values.template ReinterpretCast<uint32_t>().SetValue(channel * pitch + n,
                    CenterReciprocalBits(denominator)+(Scaled?0x800000u:0u));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1); WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
}
template<class T, uint32_t K>
__aicore__ inline void RepairSensitiveCenters(LocalTensor<float>& values,
    LocalTensor<float>& priors, LocalTensor<uint32_t>& offsets, GlobalTensor<T>& raw,
    LocalTensor<float>& scratch, uint32_t pitch, uint32_t count, uint32_t rawOffset) {
    ResetMask(); SetMaskNorm();
    if constexpr (K <= 64) {
        WholeReduceMax(scratch, priors[2 * pitch], count, 1, 1, 1, 8,
            ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        ReduceMax(scratch, priors[2 * pitch], scratch[8], count, false);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID1); WaitFlag<HardEvent::V_S>(EVENT_ID1);
    if (scratch.GetValue(0) <= (sizeof(T) == 2 ? 4096.f : 32.f)) return;
    RepairCenterRows<T,false>(values,priors,raw,scratch,pitch,count,rawOffset);
}

template<uint32_t Channels,bool RestoreNormal=true>
__aicore__ inline void GatherChannelPlanes(const LocalTensor<float>& destination,
    const LocalTensor<float>& source, const LocalTensor<uint32_t>& offsets,
    uint32_t pitch, uint32_t count) {
    // One counter mask covers valid rows in every independent channel.
    SetMaskCount();
    SetVectorMask<float,MaskMode::COUNTER>(count);
    uint32_t base = static_cast<uint32_t>(reinterpret_cast<uint64_t>(source.GetPhyAddr()));
    __ubuf__ uint32_t* indices = (__ubuf__ uint32_t*)offsets.GetPhyAddr();
#pragma unroll
    for (uint32_t channel=0;channel<Channels;++channel) {
        vgather((__ubuf__ uint32_t*)destination[channel*pitch].GetPhyAddr(),
            indices,base+channel*4,8,1);
    }
    PipeBarrier<PIPE_V>();
    if constexpr(RestoreNormal) {ResetMask(); SetMaskNorm();}
}

template<class T,uint32_t K,uint32_t valuesOffset,uint32_t priorOffset,uint32_t scratchOffset>
__aicore__ __attribute__((noinline)) void RepairDenseCold(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset);

template<class T,uint32_t K,bool StaticGuard=false>
__aicore__ inline bool SmallNeedsRepair(LocalTensor<float>& priors,
    LocalTensor<float>& scratch,uint32_t pitch,uint32_t count) {
    ResetMask();SetMaskNorm();
    if constexpr(StaticGuard) {
        static_assert(K==128,"two bounded reduction segments required");
        // The guard only asks whether any valid stride exceeds its bound.
        // Explicit events keep this bounded reduction independent of TPipe.
        const uint32_t first=count<64?count:64;
        WholeReduceMax(scratch,priors[2*pitch],first,1,1,1,8,
            ReduceOrder::ORDER_ONLY_VALUE);
        if(count>64) {
            WholeReduceMax(scratch[8],priors[2*pitch+64],count-64,1,1,1,8,
                ReduceOrder::ORDER_ONLY_VALUE);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const float bound=sizeof(T)==2?4096.f:32.f;
        const bool repair=!(scratch.GetValue(0)<=bound) ||
            (count>64 && !(scratch.GetValue(8)<=bound));
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        return repair;
    } else if constexpr(K<=64) {
        WholeReduceMax(scratch,priors[2*pitch],count,1,1,1,8,
            ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        ReduceMax(scratch,priors[2*pitch],scratch[8],count,false);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
    return !(scratch.GetValue(0)<=(sizeof(T)==2?4096.f:32.f));
}
