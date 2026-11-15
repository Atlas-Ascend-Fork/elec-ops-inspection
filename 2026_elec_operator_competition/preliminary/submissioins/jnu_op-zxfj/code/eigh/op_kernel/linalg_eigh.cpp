/*
 * Copyright (c) 2026 zjyzjy2002.
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

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "linalg_eigh_tiling.h"
#include "tiling_key_linalg_eigh.h"
#include "eigh_tridiagonal_basis.h"
#include "eigh_cube_backtransform.h"
using namespace AscendC;
constexpr bool AlignReferenceSigns = true;
// Distributed symmetric Householder reduction, Sturm bisection,
// inverse iteration, and reflector backtransform.
template <typename DT_A,uint32_t FixedStride=0,uint32_t CubeMode=0>
class KernelLinalgEigh {
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage,wideStorage,smallStorage;
    LocalTensor<float> mat,prod,v,w,tmp,tmp2,red,param,d,e,origD,origE,beta,piv,fac,pack,wideBuf,pcrBand,pcrLeft;
    GlobalTensor<float> input,values,vectors,ws,hist,gw,gparam;
    uint32_t n,dynamicStride,batch,upper,team,groups,rows,laneStride,lane,group,base,count;
    uint32_t privateCapacity,storageCapacity;
    bool packedMatrix,recordCubeSigns,directCoefficients;
    __aicore__ inline uint32_t Stride()const{
        if constexpr(FixedStride!=0)return FixedStride;
        return dynamicStride;
    }
    float spectralBound,spectralCenter;
    bool active;
    // Cyclic ownership balances the work for localized eigenvectors.
    __aicore__ inline uint32_t EigenIndex(uint32_t r){return lane+r*team;}
    __aicore__ inline void Barrier(){PipeBarrier<PIPE_ALL>();}
    __aicore__ inline float Abs(float a){return a<0?-a:a;}
    __aicore__ inline float Root(float a){
        if(a<=0)return 0;
        union Bits{float f;uint32_t u;}b;b.f=a;
        if(a>=1e-30f && a<=1e30f){
            b.u=0x5f375a86u-(b.u>>1);float reciprocal=b.f;
            for(uint32_t i=0;i<2;++i)reciprocal=reciprocal*(1.5f-((0.5f*a)*reciprocal)*reciprocal);
            b.f=a*reciprocal;b.u+=128u;float x=b.f;
            x=0.5f*(x+a/x);x=0.5f*(x+a/x);return x;
        }
        b.u=(b.u>>1)+0x1fc00000u;float x=b.f;
        x=0.5f*(x+a/x);x=0.5f*(x+a/x);x=0.5f*(x+a/x);x=0.5f*(x+a/x);return x;
    }
    __aicore__ inline void VectorToScalar(){
        event_t id=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(id);WaitFlag<HardEvent::V_S>(id);
    }
    __aicore__ inline float Dot(LocalTensor<float>a,LocalTensor<float>b){
        Mul(tmp2,a,b,Stride());PipeBarrier<PIPE_V>();
#if __NPU_ARCH__ == 2201
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(Stride()));
        vcadd((__ubuf__ float*)red.GetPhyAddr(),(__ubuf__ float*)tmp2.GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float result=GetReduceRepeatSumSpr<float>();
        SetMaskNorm();ResetMask();return result;
#else
        ReduceSum(red,tmp2,tmp,Stride());VectorToScalar();return red.GetValue(0);
#endif
    }
    __aicore__ inline float DotTail(LocalTensor<float>a,LocalTensor<float>b,uint32_t begin){
#if __NPU_ARCH__ == 2201
        uint32_t first=begin/64*64,length=Stride()-first;
        Mul(tmp2[first],a[first],b[first],length);PipeBarrier<PIPE_V>();
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
        vcadd((__ubuf__ float*)red.GetPhyAddr(),(__ubuf__ float*)tmp2[first].GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float result=GetReduceRepeatSumSpr<float>();
        SetMaskNorm();ResetMask();return result;
#else
        return Dot(a,b);
#endif
    }
    // Eight rows by eight columns per 256-byte SIMD tile. This is a
    // physical UB layout only; GM tensors and all inverse work stay compact.
    __aicore__ inline void PackMatrix(){
        packedMatrix=n>25 && Stride()%64==0 &&
            (Stride()/64>=(count+7)/8+2 || count*8>4*Stride());
        if(!packedMatrix)return;
        uint32_t padded=(count+7)/8*8;
        Duplicate(prod,0.0f,padded*Stride());PipeBarrier<PIPE_V>();SetVectorMask<float>(8);
        for(uint32_t r=0;r<count;++r)
            Copy<float,false>(prod[(r/8)*8*Stride()+(r%8)*8],mat[r*Stride()],8,uint8_t(Stride()/8),CopyRepeatParams{1,1,8,1});
        Barrier();DataCopy(mat,prod,padded*Stride());Barrier();
    }
    __aicore__ inline void ReadPackedRow(LocalTensor<float> dst,uint32_t r){
        if(packedMatrix){
            Copy(dst,mat[(r/8)*8*Stride()+(r%8)*8],8,uint8_t(Stride()/8),CopyRepeatParams{1,1,1,8});
        }else DataCopy(dst,mat[r*Stride()],Stride());
    }
    __aicore__ inline void UnpackMatrix(){
        if(!packedMatrix)return;
        for(uint32_t r=0;r<count;++r)ReadPackedRow(prod[r*Stride()],r);
        Barrier();DataCopy(mat,prod,count*Stride());Barrier();packedMatrix=false;
    }
    template<uint32_t Width>
    __aicore__ inline void AdjacentLevel(LocalTensor<float> dst,LocalTensor<float> src,uint32_t chunks){
        // Independent adjacent sums share a SIMD repeat. A block holds the
        // same eight row results; changing the instruction mask preserves
        // each scalar pair and the complete three-level reduction tree.
        SetVectorMask<float>(8*Width);
        for(uint32_t first=0;first<chunks;first+=255){
            uint8_t repeats=uint8_t(chunks-first<255?chunks-first:255);
            Add<float,false>(dst[first*Width*8],src[first*Width*16],src[first*Width*16+8],8*Width,repeats,BinaryRepeatParams{1,2,2,uint8_t(Width),uint8_t(2*Width),uint8_t(2*Width)});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DotPackedRows(LocalTensor<float> vector,uint32_t begin){
        uint32_t blocks=(count+7)/8,first=begin/64*64,width=Stride()-first;
        Duplicate(w,0.0f,blocks*8);PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
        for(uint32_t rb=0;rb<blocks;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Mul<float,false>(prod[off],mat[off],vector[first],64,uint8_t((Stride()-first)/8),BinaryRepeatParams{1,1,0,8,8,1});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=0;rb<blocks;++rb)
            BlockReduceSum<float,false>(pcrLeft[rb*width],prod[rb*8*Stride()+first*8],int32_t((Stride()-first)/8),64,1,1,8);
        PipeBarrier<PIPE_V>();
        // VCGADD followed by this adjacent binary tree has the exact VCADD
        // 64-element order. Keep the original left fold between 64-column
        // chunks, including its initial zero.
        uint32_t chunks=blocks*(width/64);
        AdjacentLevel<4>(prod,pcrLeft,chunks);
        AdjacentLevel<2>(pcrLeft,prod,chunks);
        AdjacentLevel<1>(prod,pcrLeft,chunks);
        SetVectorMask<float>(8);
        for(uint32_t j=0;j<width/64;++j){
            Add<float,false>(w,w,prod[j*8],8,uint8_t(blocks),BinaryRepeatParams{1,1,1,1,1,uint8_t(width/64)});
            PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(64);
    }
    __aicore__ inline void UpdatePacked(uint32_t k){
        uint32_t padded=(count+7)/8*8,first=(k+1)/8*8;
        if(directCoefficients){
            Brcb(pcrLeft,w[base],uint8_t(padded/8),{1,8});Brcb(pcrBand,v[base],uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        }else{
        Gather(tmp2,w,piv.template ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
        Brcb(pcrLeft,tmp2,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        Gather(tmp2,v,piv.template ReinterpretCast<uint32_t>(),0,padded);PipeBarrier<PIPE_V>();
        Brcb(pcrBand,tmp2,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(64);
        uint32_t firstBlock=base>k+1?0:(k+1-base)/8;
        uint8_t repeats=uint8_t((Stride()-first)/8);
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Mul<float,false>(prod[off],v[first],pcrLeft[rb*64],64,repeats,BinaryRepeatParams{1,0,1,8,1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            MulAddDst<float,float,false>(prod[off],w[first],pcrBand[rb*64],64,repeats,BinaryRepeatParams{1,0,1,8,1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t rb=firstBlock;rb<padded/8;++rb){
            uint32_t off=rb*8*Stride()+first*8;
            Sub<float,false>(mat[off],mat[off],prod[off],64,repeats,BinaryRepeatParams{1,1,1,8,8,8});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DotRows(LocalTensor<float> vector,uint32_t begin=0){
        if(packedMatrix){DotPackedRows(vector,begin);return;}
        // Complete SIMD blocks share one mask across products, row reductions,
        // and the ordered sum of partials. Padding lanes remain zero.
        if(Stride()%64==0 && count<=64){
            Duplicate(w,0.0f,64);Duplicate(tmp2,0.0f,64);PipeBarrier<PIPE_V>();
            SetVectorMask<float>(64);
            for(uint32_t col=begin/64*64;col<Stride();col+=64){
                Mul<float,false>(prod[col],mat[col],vector[col],64,uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum<float,false>(tmp2,prod[col],64,uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
                Add<float,false>(w,w,tmp2,64,1,BinaryRepeatParams{1,1,1,8,8,8});
            }
            PipeBarrier<PIPE_V>();return;
        }
        uint32_t len=(count+7)/8*8;
        Duplicate(w,0.0f,len);Duplicate(tmp2,0.0f,len);PipeBarrier<PIPE_V>();
        for(uint32_t col=begin/64*64;col<Stride();col+=64){
            uint64_t mask=Stride()-col<64?Stride()-col:64;
            Mul(prod[col],mat[col],vector[col],mask,uint8_t(count),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),0});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tmp2,prod[col],mask,uint8_t(count),1,1,uint8_t(Stride()/8));PipeBarrier<PIPE_V>();
            Add(w,w,tmp2,len);
        }
        VectorToScalar();
    }
    __aicore__ inline ExtendedFloat ReadExtended(uint32_t i){i*=2;return ExtendedFloat(wideBuf.GetValue(i),wideBuf.GetValue(i+1),0);}
    __aicore__ inline void WriteExtended(uint32_t i,ExtendedFloat x){i*=2;wideBuf.SetValue(i,x.h);wideBuf.SetValue(i+1,x.m);}
    __aicore__ inline ExtendedFloat RootExtended(ExtendedFloat x){
        if(x.Zero() || x.Negative())return ExtendedFloat(0);
        ExtendedFloat y(Root(x.h));for(uint32_t i=0;i<1;++i){ExtendedFloat r=x-y*y;y=y+r/(y.Scale(2));}return y;
    }
    __aicore__ inline ExtendedFloat Reflector(uint32_t j){return ExtendedFloat(v.GetValue(j),tmp2.GetValue(j),0);}
    __aicore__ inline ExtendedFloat ProductRow(uint32_t j){return ExtendedFloat(w.GetValue(j),red.GetValue(j),0);}
    __aicore__ inline bool HouseholderFactors(ExtendedFloat a, ExtendedFloat tailSquared,
                                              ExtendedFloat &off, ExtendedFloat &tau, ExtendedFloat &scale) {
        off = a;
        tau = ExtendedFloat(0);
        if (tailSquared.Zero()) return false;
        ExtendedFloat diagonal = RootExtended(a*a + tailSquared);
        if (!a.Negative()) diagonal = -diagonal;
        ExtendedFloat difference = a - diagonal;
        if (diagonal.h == 0.0f || difference.h == 0.0f) return false;
        off = diagonal;
        tau = (diagonal-a).DivideBy(diagonal);
        scale = ExtendedFloat(1) / difference;
        return true;
    }
    __aicore__ inline void PublishExtendedReflector(uint32_t k) {
        uint32_t row=k-base;ExtendedFloat diag=ReadExtended(row*n+k),off(0),tau(0);
        Duplicate(v,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
        if(k+1<n){
            ExtendedFloat a=ReadExtended(row*n+k+1),norm(0);off=a;
            for(uint32_t j=k+2;j<n;++j){ExtendedFloat x=ReadExtended(row*n+j);norm=norm+x*x;}
            v.SetValue(k+1,1);
            ExtendedFloat scale;
            if(HouseholderFactors(a, norm, off, tau, scale)){
                for(uint32_t j=k+2;j<n;++j){ExtendedFloat x=ReadExtended(row*n+j)*scale;v.SetValue(j,x.h);tmp2.SetValue(j,x.m);}
            }
            Barrier();DataCopy(hist[k*Stride()],v,Stride());DataCopy(hist[(n+k)*Stride()],tmp2,Stride());Barrier();
        }
        param.SetValue(0,diag.h);param.SetValue(1,diag.m);param.SetValue(2,off.h);param.SetValue(3,off.m);param.SetValue(4,tau.h);param.SetValue(5,tau.m);
        Barrier();DataCopy(gparam,param,8);Barrier();
    }
    __aicore__ inline void CollectExtendedProduct(uint32_t k) {
        DataCopy(param,gparam,8);if(k+1<n){DataCopy(v,hist[k*Stride()],Stride());DataCopy(tmp2,hist[(n+k)*Stride()],Stride());}Barrier();
        d.SetValue(k,param.GetValue(0));e.SetValue(k,param.GetValue(2));beta.SetValue(k,param.GetValue(4));
        TridiagonalBasis canonical(wideBuf,n);canonical.Save(k,ExtendedFloat(param.GetValue(0),param.GetValue(1),0),ExtendedFloat(param.GetValue(2),param.GetValue(3),0));
        ExtendedFloat tau(param.GetValue(4),param.GetValue(5),0);
        if(k+1<n && !tau.Zero()){
            for(uint32_t r=0;r<count;++r){ExtendedFloat x(0);
                if(base+r>k){for(uint32_t j=k+1;j<n;++j)x=x+ReadExtended(r*n+j)*Reflector(j);x=x*tau;}
                w.SetValue(r,x.h);red.SetValue(r,x.m);
            }
            Barrier();DataCopyPad(gw[base],w,DataCopyExtParams{1,count*4,0,0,0});DataCopyPad(gw[Stride()+base],red,DataCopyExtParams{1,count*4,0,0,0});Barrier();
        }
    }
    __aicore__ inline void ApplyExtendedCorrection(uint32_t k) {
        ExtendedFloat tau(param.GetValue(4),param.GetValue(5),0);
        if(!tau.Zero()){
            DataCopy(w,gw,Stride());DataCopy(red,gw[Stride()],Stride());Barrier();
            ExtendedFloat dot(0);for(uint32_t j=k+1;j<n;++j)dot=dot+Reflector(j)*ProductRow(j);
            ExtendedFloat correction=ExtendedFloat(-0.5f)*tau*dot;
            for(uint32_t j=k+1;j<n;++j){ExtendedFloat x=ProductRow(j)+correction*Reflector(j);w.SetValue(j,x.h);red.SetValue(j,x.m);}
            for(uint32_t r=0;r<count;++r)if(base+r>k){
                ExtendedFloat wi=ProductRow(base+r),vi=Reflector(base+r);
                for(uint32_t j=k+1;j<n;++j){ExtendedFloat x=ReadExtended(r*n+j)-(Reflector(j)*wi+ProductRow(j)*vi);WriteExtended(r*n+j,x);}
            }
        }
    }
    __aicore__ inline void ReduceExtended(){
        if(active)for(uint32_t r=0;r<count;++r)for(uint32_t j=0;j<n;++j)WriteExtended(r*n+j,ExtendedFloat(mat.GetValue(r*Stride()+j)));
        for(uint32_t k=0;k<n;++k){
            if(active && k>=base && k<base+count){
                PublishExtendedReflector(k);
            }
            SyncAll();
            if(active){
                CollectExtendedProduct(k);
            }
            SyncAll();
            if(active && k+1<n){
                ApplyExtendedCorrection(k);
            }
        }
        if(active){Barrier();DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();}
    }
    // Every active AIV owns a complete small matrix. Vector lanes are rows;
    // the column fold keeps the original expansion operation order.
    __aicore__ inline void SmallCopy(LocalTensor<float> a,LocalTensor<float> b){
        Copy<float,false>(a,b,Stride(),1,CopyRepeatParams{1,1,8,8});
    }
    __aicore__ inline void SmallAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Add<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallSub(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c){
        Sub<float,false>(a,b,c,Stride(),1,BinaryRepeatParams{1,1,1,8,8,8});
    }
    __aicore__ inline void SmallMuls(LocalTensor<float> a,LocalTensor<float> b,float x){
        Muls<float,false>(a,b,x,Stride(),1,UnaryRepeatParams{1,1,8,8});
    }
    template<bool Left>
    __aicore__ inline void SmallScale(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,ExtendedFloat b){
        auto scratch=smallStorage.Get<float>()[6*n*Stride()];
        auto hi=scratch[4*Stride()],lo=scratch[5*Stride()];
        auto t=scratch[6*Stride()],u=scratch[7*Stride()];
        float z=4097.0f*b.h,bh=z-(z-b.h),bl=b.h-bh;
        SmallMuls(dh,ah,b.h);SmallMuls(t,ah,4097.0f);PipeBarrier<PIPE_V>();
        SmallSub(hi,t,ah);PipeBarrier<PIPE_V>();SmallSub(hi,t,hi);PipeBarrier<PIPE_V>();
        SmallSub(lo,ah,hi);SmallMuls(dm,hi,bh);PipeBarrier<PIPE_V>();
        SmallSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        if constexpr(Left){
            SmallMuls(t,lo,bh);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
            SmallMuls(t,hi,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        }else{
            SmallMuls(t,hi,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
            SmallMuls(t,lo,bh);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        }
        SmallMuls(t,lo,bl);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        if constexpr(Left){SmallMuls(t,am,b.h);SmallMuls(u,ah,b.m);}
        else{SmallMuls(t,ah,b.m);SmallMuls(u,am,b.h);}
        PipeBarrier<PIPE_V>();SmallAdd(t,t,u);PipeBarrier<PIPE_V>();SmallAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        SmallAdd(t,dh,dm);PipeBarrier<PIPE_V>();SmallSub(u,t,dh);PipeBarrier<PIPE_V>();
        SmallSub(dm,dm,u);SmallCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    template<bool Minus>
    __aicore__ inline void SmallSum(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        auto scratch=smallStorage.Get<float>()[6*n*Stride()];
        auto t=scratch[4*Stride()],q=scratch[5*Stride()],a=scratch[6*Stride()];
        auto b=scratch[7*Stride()],ee=scratch[8*Stride()];
        if constexpr(Minus)SmallSub(t,ah,bh);else SmallAdd(t,ah,bh);
        PipeBarrier<PIPE_V>();SmallSub(q,t,ah);PipeBarrier<PIPE_V>();
        SmallSub(a,t,q);PipeBarrier<PIPE_V>();SmallSub(a,ah,a);
        if constexpr(Minus)SmallMuls(b,bh,-1.0f);else SmallCopy(b,bh);
        PipeBarrier<PIPE_V>();SmallSub(b,b,q);PipeBarrier<PIPE_V>();SmallAdd(ee,a,b);PipeBarrier<PIPE_V>();
        SmallAdd(ee,ee,am);PipeBarrier<PIPE_V>();
        if constexpr(Minus)SmallSub(ee,ee,bm);else SmallAdd(ee,ee,bm);
        PipeBarrier<PIPE_V>();SmallAdd(dh,t,ee);PipeBarrier<PIPE_V>();
        SmallSub(q,dh,t);PipeBarrier<PIPE_V>();SmallSub(dm,ee,q);PipeBarrier<PIPE_V>();
    }
    // Whole-matrix products amortize the expansion arithmetic over SIMD
    // repeats. Columns remain contiguous; row folds below remain ordered.
    __aicore__ inline void DenseExpand(LocalTensor<float> dst,LocalTensor<float> src){
        auto t=smallStorage.Get<float>()[6*n*Stride()];
        Brcb(t,src,uint8_t((n+7)/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(8);
        for(uint32_t row=0;row<Stride();row+=8)
            Copy<float,false>(dst[row],t,8,uint8_t(n),CopyRepeatParams{1,1,uint16_t(Stride()/8),1});
        PipeBarrier<PIPE_V>();SetVectorMask<float>(Stride());
    }
    __aicore__ inline void DenseAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Add<float,false>(a,b,c,Stride(),uint8_t(n),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseSub(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Sub<float,false>(a,b,c,Stride(),uint8_t(n),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseMul(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,uint8_t cs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Mul<float,false>(a,b,c,Stride(),uint8_t(n),BinaryRepeatParams{1,1,1,pitch,pitch,cs==255?pitch:cs});
    }
    __aicore__ inline void DenseMuls(LocalTensor<float> a,LocalTensor<float> b,float x,uint8_t bs=255){
        uint8_t pitch=uint8_t(Stride()/8);
        Muls<float,false>(a,b,x,Stride(),uint8_t(n),UnaryRepeatParams{1,1,pitch,bs==255?pitch:bs});
    }
    __aicore__ inline void DenseCopy(LocalTensor<float> a,LocalTensor<float> b){
        uint16_t pitch=uint16_t(Stride()/8);
        Copy<float,false>(a,b,Stride(),uint8_t(n),CopyRepeatParams{1,1,pitch,pitch});
    }
    template<bool RowB>
    __aicore__ inline void DenseProduct(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        uint32_t dense=n*Stride();auto scratch=smallStorage.Get<float>()[6*dense];
        auto hi=scratch,lo=scratch[dense],bhi=scratch[2*dense],blo=scratch[3*dense],t=scratch[4*dense];
        constexpr uint8_t bs=RowB?0:255;
        DenseMul(dh,ah,bh,bs);DenseMuls(hi,ah,4097.0f);DenseMuls(bhi,bh,4097.0f,bs);PipeBarrier<PIPE_V>();
        DenseSub(lo,hi,ah);DenseSub(blo,bhi,bh,bs);PipeBarrier<PIPE_V>();
        DenseSub(hi,hi,lo);DenseSub(bhi,bhi,blo);PipeBarrier<PIPE_V>();
        DenseSub(lo,ah,hi);
        // bh's repeated view becomes a full contiguous split, so subtraction
        // uses an explicit row repeat when the original b was a row vector.
        DenseMuls(blo,bh,1.0f,bs);PipeBarrier<PIPE_V>();DenseSub(blo,blo,bhi);PipeBarrier<PIPE_V>();
        DenseMul(dm,hi,bhi);PipeBarrier<PIPE_V>();DenseSub(dm,dm,dh);PipeBarrier<PIPE_V>();
        DenseMul(t,hi,blo);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,lo,bhi);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,lo,blo);PipeBarrier<PIPE_V>();DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();
        DenseMul(t,ah,bm,bs);DenseMul(hi,am,bh,bs);PipeBarrier<PIPE_V>();DenseAdd(t,t,hi);PipeBarrier<PIPE_V>();
        DenseAdd(dm,dm,t);PipeBarrier<PIPE_V>();DenseAdd(t,dh,dm);PipeBarrier<PIPE_V>();
        DenseSub(hi,t,dh);PipeBarrier<PIPE_V>();DenseSub(dm,dm,hi);DenseCopy(dh,t);PipeBarrier<PIPE_V>();
    }
    template<bool Minus>
    __aicore__ inline void DenseSum(LocalTensor<float> dh,LocalTensor<float> dm,
        LocalTensor<float> ah,LocalTensor<float> am,LocalTensor<float> bh,LocalTensor<float> bm){
        uint32_t dense=n*Stride();auto scratch=smallStorage.Get<float>()[6*dense];
        auto t=scratch,q=scratch[dense],a=scratch[2*dense],b=scratch[3*dense],ee=scratch[4*dense];
        if constexpr(Minus)DenseSub(t,ah,bh);else DenseAdd(t,ah,bh);
        PipeBarrier<PIPE_V>();DenseSub(q,t,ah);PipeBarrier<PIPE_V>();DenseSub(a,t,q);PipeBarrier<PIPE_V>();
        DenseSub(a,ah,a);if constexpr(Minus)DenseMuls(b,bh,-1.0f);else DenseCopy(b,bh);
        PipeBarrier<PIPE_V>();DenseSub(b,b,q);PipeBarrier<PIPE_V>();DenseAdd(ee,a,b);PipeBarrier<PIPE_V>();
        DenseAdd(ee,ee,am);PipeBarrier<PIPE_V>();if constexpr(Minus)DenseSub(ee,ee,bm);else DenseAdd(ee,ee,bm);
        PipeBarrier<PIPE_V>();DenseAdd(dh,t,ee);PipeBarrier<PIPE_V>();DenseSub(q,dh,t);PipeBarrier<PIPE_V>();DenseSub(dm,ee,q);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadSmallMatrix(uint32_t batchIndex, LocalTensor<float> mh, LocalTensor<float> scratch) {
        // The unused matrix region is large enough for the compact raw
        // input, and is dead before any vector scratch is reused.
        DataCopyPad(scratch,input[uint64_t(batchIndex)*n*n],DataCopyExtParams{1,n*n*4,0,0,0},
            DataCopyPadExtParams<float>{false,0,0,0});Barrier();
        Duplicate(mh,0.0f,2*n*Stride());Barrier();
        for(uint32_t j=0;j<n;++j)for(uint32_t i=0;i<n;++i){
            uint32_t pi=upper?n-1-i:i,pj=upper?n-1-j:j;
            uint32_t row=pi>pj?pi:pj,col=pi>pj?pj:pi;
            if(upper){uint32_t temp=row;row=col;col=temp;}
            mh.SetValue(j*Stride()+i,scratch.GetValue(row*n+col));
        }
    }
    __aicore__ inline void BuildSmallReflector(uint32_t k, LocalTensor<float> mh, LocalTensor<float> ml, ExtendedFloat &off, ExtendedFloat &tau) {
        ExtendedFloat a(mh.GetValue((k+1)*Stride()+k),ml.GetValue((k+1)*Stride()+k),0),norm(0);off=a;
        for(uint32_t j=k+2;j<n;++j){
            ExtendedFloat x(mh.GetValue(j*Stride()+k),ml.GetValue(j*Stride()+k),0);norm=norm+x*x;
        }
        v.SetValue(k+1,1);
        ExtendedFloat scale;
        if(HouseholderFactors(a, norm, off, tau, scale)){
            for(uint32_t j=k+2;j<n;++j){
                ExtendedFloat x=ExtendedFloat(mh.GetValue(j*Stride()+k),ml.GetValue(j*Stride()+k),0)*scale;
                v.SetValue(j,x.h);tmp2.SetValue(j,x.m);
            }
        }
        Barrier();
        if(lane+1==team){DataCopy(hist[k*Stride()],v,Stride());DataCopy(hist[(n+k)*Stride()],tmp2,Stride());Barrier();}
    }
    __aicore__ __attribute__((noinline)) void ReduceReplicatedSmall(uint32_t batchIndex){
        if(active){
            auto mh=smallStorage.Get<float>(),ml=mh[n*Stride()],scratch=mh[2*n*Stride()];
            LoadSmallMatrix(batchIndex, mh, scratch);
            uint32_t dense=n*Stride();auto ph=mh[2*dense],pm=mh[3*dense],qh=mh[4*dense],qm=mh[5*dense];
            auto vh=mh[11*dense],vm=mh[12*dense],wh=mh[13*dense],wm=mh[14*dense];
            auto sh=mh[6*dense],sm=sh[Stride()];
            TridiagonalBasis canonical(wideBuf,n);
            for(uint32_t k=0;k<n;++k){
                Barrier();ExtendedFloat diag(mh.GetValue(k*Stride()+k),ml.GetValue(k*Stride()+k),0);
                ExtendedFloat off(0),tau(0);
                Duplicate(v,0.0f,Stride());Duplicate(tmp2,0.0f,Stride());Barrier();
                if(k+1<n){
                    BuildSmallReflector(k, mh, ml, off, tau);
                }
                d.SetValue(k,diag.h);e.SetValue(k,off.h);beta.SetValue(k,tau.h);canonical.Save(k,diag,off);
                if(k+1<n && !tau.Zero()){
                    Duplicate(w,0.0f,Stride());Duplicate(red,0.0f,Stride());PipeBarrier<PIPE_V>();
                    DenseExpand(vh,v);DenseExpand(vm,tmp2);DenseProduct<false>(ph,pm,mh,ml,vh,vm);
                    for(uint32_t j=k+1;j<n;++j)SmallSum<false>(w,red,w,red,ph[j*Stride()],pm[j*Stride()]);
                    SmallScale<false>(sh,sm,w,red,tau);SmallCopy(w,sh);SmallCopy(red,sm);Barrier();
                    for(uint32_t j=0;j<=k;++j){w.SetValue(j,0);red.SetValue(j,0);}
                    ExtendedFloat dot(0);for(uint32_t j=k+1;j<n;++j)dot=dot+Reflector(j)*ProductRow(j);
                    ExtendedFloat correction=ExtendedFloat(-0.5f)*tau*dot;
                    for(uint32_t j=k+1;j<n;++j){ExtendedFloat x=ProductRow(j)+correction*Reflector(j);w.SetValue(j,x.h);red.SetValue(j,x.m);}
                    Barrier();DenseExpand(wh,w);DenseExpand(wm,red);
                    DenseProduct<true>(ph,pm,vh,vm,w,red);DenseProduct<true>(qh,qm,wh,wm,v,tmp2);
                    DenseSum<false>(ph,pm,ph,pm,qh,qm);DenseSum<true>(mh,ml,mh,ml,ph,pm);
                    Barrier();ResetMask();
                }
            }
            DataCopy(origD,d,Stride());DataCopy(origE,e,Stride());Barrier();
        }
        // The full calculation is invocation-local. Publish the completed
        // reflectors before any AIV begins the original backtransform.
        SyncAll();
    }

    __aicore__ inline float TailDot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> product,LocalTensor<float> result,uint32_t length){
        Mul(product,a,b,length);PipeBarrier<PIPE_V>();
#if __NPU_ARCH__ == 2201
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,uint64_t(length));
        vcadd((__ubuf__ float*)result.GetPhyAddr(),(__ubuf__ float*)product.GetPhyAddr(),1,1,1,8,1);
        VectorToScalar();float value=GetReduceRepeatSumSpr<float>();SetMaskNorm();ResetMask();return value;
#else
        ReduceSum(result,product,product[64],length);VectorToScalar();return result.GetValue(0);
#endif
    }
    struct TailBlock {
        uint32_t rows, columns, pitch, blockRows;
        LocalTensor<float> w, v, matrix, product, sum, broadcastW, broadcastV, pivot;
    };
    __aicore__ inline void TailCorrection(TailBlock &block, uint32_t k, uint32_t localFirst, float tau) {
        Duplicate(block.w,0.0f,128);PipeBarrier<PIPE_V>();
        for(uint32_t rb=(k+1)/8*8;rb<block.rows;rb+=block.blockRows){
            uint8_t live=uint8_t(block.rows-rb<block.blockRows?block.rows-rb:block.blockRows);
            for(uint32_t col=localFirst;col<block.columns;col+=64){
                uint64_t mask=block.columns-col<64?block.columns-col:64;
                Mul(block.product,block.matrix[rb*block.pitch+col],block.v[col],mask,live,BinaryRepeatParams{1,1,1,8,uint8_t(block.pitch/8),0});PipeBarrier<PIPE_V>();
                WholeReduceSum(block.sum,block.product,mask,live,1,1,8);PipeBarrier<PIPE_V>();
                Add(block.w[rb],block.w[rb],block.sum,uint32_t(live));PipeBarrier<PIPE_V>();
            }
        }
        Duplicate(block.w,0.0f,k+1);PipeBarrier<PIPE_V>();Muls(block.w,block.w,tau,block.columns);Barrier();
        float correction=-0.5f*tau*TailDot(block.v[localFirst],block.w[localFirst],block.product,block.sum,block.columns-localFirst);
        Axpy(block.w,block.v,correction,block.columns);Barrier();
        Muls(block.broadcastW,block.v,block.w.GetValue(k+1),block.columns);PipeBarrier<PIPE_V>();
        Axpy(block.broadcastW,block.w,block.v.GetValue(k+1),block.columns);PipeBarrier<PIPE_V>();
        Sub(block.pivot,block.pivot,block.broadcastW,block.columns);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void UpdateTailRows(TailBlock &block, uint32_t k) {
        for(uint32_t rb=(k+1)/8*8;rb<block.rows;rb+=block.blockRows){
            uint8_t live=uint8_t(block.rows-rb<block.blockRows?block.rows-rb:block.blockRows);
            Brcb(block.broadcastW,block.w[rb],uint8_t((live+7)/8),{1,8});Brcb(block.broadcastV,block.v[rb],uint8_t((live+7)/8),{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t col=(k+1)/8*8;col<block.columns;col+=64){
                uint64_t mask=block.columns-col<64?block.columns-col:64;
                Mul(block.product,block.v[col],block.broadcastW,mask,live,BinaryRepeatParams{1,1,0,8,0,1});PipeBarrier<PIPE_V>();
                MulAddDst(block.product,block.w[col],block.broadcastV,mask,live,BinaryRepeatParams{1,1,0,8,0,1});PipeBarrier<PIPE_V>();
                Sub(block.matrix[rb*block.pitch+col],block.matrix[rb*block.pitch+col],block.product,mask,live,BinaryRepeatParams{1,1,1,uint8_t(block.pitch/8),uint8_t(block.pitch/8),8});PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ __attribute__((noinline)) void PrivateTail(uint32_t first){
        uint32_t size=n-first,width=Stride()-first,pitch=size>64?128:64;
        uint32_t batchRows=4*Stride()+1344<=privateCapacity?64:32;
        LocalTensor<float> td=wideBuf,te=wideBuf[Stride()],tb=wideBuf[2*Stride()];
        LocalTensor<float> fullV=wideBuf[3*Stride()],tv=fullV[first];
        LocalTensor<float> tw=wideBuf[4*Stride()],pivot=wideBuf[4*Stride()+128];
        LocalTensor<float> result=wideBuf[4*Stride()+256];
        LocalTensor<float> cw=wideBuf[4*Stride()+320],cv=cw[batchRows*8];
        LocalTensor<float> product=storage.Get<float>()[width*pitch];
        // Preserve all completed metadata and the separately updated pivot
        // before recycling the resident matrix and inverse work storage.
        DataCopy(td,d,Stride());DataCopy(te,e,Stride());DataCopy(tb,beta,Stride());
        DataCopy(pivot,pack[first],width);Duplicate(fullV,0.0f,Stride());Barrier();
        LocalTensor<float> tail=storage.Get<float>();
        TailBlock block{size, width, pitch, batchRows, tw, tv, tail, product, result, cw, cv, pivot};
        Duplicate(tail,0.0f,width*pitch);Barrier();
        auto stage=hist[(n+5)*Stride()];
        if(width==pitch)DataCopy(tail,stage,size*pitch);
        else DataCopyPad(tail,stage,DataCopyExtParams{uint16_t(size),width*4,0,(pitch-width)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
        Barrier();
        for(uint32_t k=0;k<size;++k){
            uint32_t global=first+k,localFirst=(k+1)/64*64;
            td.SetValue(global,pivot.GetValue(k));te.SetValue(global,0);tb.SetValue(global,0);
            float tau=0;
            if(k+2<size){
                DataCopy(tv,pivot,width);Barrier();Duplicate(tv,0.0f,k+1);Barrier();
                float norm=Root(TailDot(tv[localFirst],tv[localFirst],product,result,width-localFirst));float x0=tv.GetValue(k+1);
                te.SetValue(global,x0>=0?-norm:norm);
                if(norm>0){Muls(tv,tv,1.0f/norm,width);Barrier();tv.SetValue(k+1,x0/norm+(x0>=0?1.0f:-1.0f));Barrier();}
                DataCopy(hist[global*Stride()],fullV,Stride());
                if(norm>0)tau=2.0f/TailDot(tv[localFirst],tv[localFirst],product,result,width-localFirst);
                tb.SetValue(global,tau);
            }else if(k+1<size)te.SetValue(global,pivot.GetValue(k+1));
            if(k+1<size){DataCopy(pivot,tail[(k+1)*pitch],width);Barrier();}
            if(k+2<size && tau!=0){
                TailCorrection(block, k, localFirst, tau);
                UpdateTailRows(block, k);
            }
            Barrier();
        }
        // These destinations may overlap the now-dead private matrix.
        DataCopy(d,td,Stride());DataCopy(e,te,Stride());DataCopy(beta,tb,Stride());Barrier();
    }
    __aicore__ inline void GatherTail(uint32_t first){
        uint32_t width=Stride()-first;
        if(active){
            auto stage=hist[(n+5)*Stride()];
            for(uint32_t r=0;r<count;++r)if(base+r>=first){
                ReadPackedRow(tmp,r);Barrier();
                DataCopy(stage[(base+r-first)*width],tmp[first],width);Barrier();
            }
        }
        SyncAll();
        if(active && lane+1==team)PrivateTail(first);
        packedMatrix=false;
    }
    __aicore__ inline void PublishPivot(uint32_t k, GlobalTensor<float> pivotRows) {
        d.SetValue(k,pack.GetValue(k));e.SetValue(k,0);beta.SetValue(k,0);
        if(k+2<n){
            DataCopy(v,pack,Stride());Barrier();Duplicate(v,0.0f,k+1);PipeBarrier<PIPE_V>();
            float norm=Root(DotTail(v,v,k+1));float x0=v.GetValue(k+1),alpha=x0>=0?-norm:norm;
            e.SetValue(k,alpha);
            if(norm>0){Muls(v,v,1.0f/norm,Stride());VectorToScalar();v.SetValue(k+1,x0/norm+(x0>=0?1.0f:-1.0f));Barrier();}
            if(lane+1==team){Barrier();DataCopy(hist[k*Stride()],v,Stride());}
            if(norm>0)beta.SetValue(k,2.0f/DotTail(v,v,k+1));
        }else if(k+1<n)e.SetValue(k,pack.GetValue(k+1));
        // Publish the unmodified next row alongside the matrix-vector product.
        if(k+1<n && k+1>=base && k+1<base+count){
            if(packedMatrix){ReadPackedRow(tmp,k+1-base);Barrier();DataCopy(pivotRows[(k%2)*2*Stride()],tmp,Stride());}
            else DataCopy(pivotRows[(k%2)*2*Stride()],mat[(k+1-base)*Stride()],Stride());
        }
        if(k+2<n){
            if(beta.GetValue(k)!=0){
                DotRows(v,k+1);Muls(w,w,beta.GetValue(k),(count+7)/8*8);Barrier();
                if(base<=k){uint32_t z=k+1-base;z=z<count?z:count;Duplicate(w,0.0f,z);Barrier();}
            }else{Duplicate(w,0.0f,Stride());Barrier();}
            DataCopyPad(gw[(k%2)*2*Stride()+base],w,DataCopyExtParams{1,count*4,0,0,0});
        }
    }
    __aicore__ inline void UpdateTrailingRows(uint32_t k, uint32_t first, LocalTensor<uint32_t> rowOffsets) {
        uint32_t firstRow=base>k?0:k+1-base;firstRow=firstRow<count?firstRow:count;
        if(firstRow<count){
            if(directCoefficients){
                Brcb(pack[Stride()],w[base],uint8_t(count/8),{1,8});Brcb(pack[5*Stride()],v[base],uint8_t(count/8),{1,8});PipeBarrier<PIPE_V>();
            }else{
            Gather(tmp2,w,rowOffsets,0,count);PipeBarrier<PIPE_V>();Brcb(pack[Stride()],tmp2,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();
            Gather(tmp2,v,rowOffsets,0,count);PipeBarrier<PIPE_V>();Brcb(pack[5*Stride()],tmp2,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();
            }
            SetVectorMask<float>(64);
            BinaryRepeatParams br{1,1,0,uint8_t(Stride()/8),0,1};
            for(uint32_t col=first;col<Stride();col+=64){
                uint32_t off=firstRow*Stride()+col;
                Mul<float,false>(prod[off],v[col],pack[Stride()+8*firstRow],64,uint8_t(count-firstRow),br);
            }
            PipeBarrier<PIPE_V>();
            for(uint32_t col=first;col<Stride();col+=64){
                uint32_t off=firstRow*Stride()+col;
                MulAddDst<float,float,false>(prod[off],w[col],pack[5*Stride()+8*firstRow],64,uint8_t(count-firstRow),br);
            }
            PipeBarrier<PIPE_V>();
            for(uint32_t col=first;col<Stride();col+=64){
                uint32_t off=firstRow*Stride()+col;
                Sub<float,false>(mat[off],mat[off],prod[off],64,uint8_t(count-firstRow),BinaryRepeatParams{1,1,1,uint8_t(Stride()/8),uint8_t(Stride()/8),uint8_t(Stride()/8)});
            }
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ApplyCollectedPivot(uint32_t k, LocalTensor<uint32_t> rowOffsets) {
        // One transfer fetches the matrix-vector result and next pivot row.
        if(k+1<n)DataCopy(w,gw[(k%2)*2*Stride()],2*Stride());
        Barrier();
        if(k+2<n && beta.GetValue(k)!=0){
            float correction=-0.5f*beta.GetValue(k)*DotTail(v,w,k+1);
            Axpy(w,v,correction,Stride());VectorToScalar();
            if(packedMatrix || (Stride()%64==0 && count*8<=4*Stride())){
                SetVectorMask<float>(64);uint32_t first=(k+1)/64*64;
                // Apply exactly the same arithmetic to the replicated row.
                Muls<float,false>(tmp[first],v[first],w.GetValue(k+1),64,uint8_t((Stride()-first)/64),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                Axpy<float,float,false>(tmp[first],w[first],v.GetValue(k+1),64,uint8_t((Stride()-first)/64),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                Sub<float,false>(pack[first],pack[first],tmp[first],64,uint8_t((Stride()-first)/64),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                if(packedMatrix)UpdatePacked(k);
                else{
                UpdateTrailingRows(k, first, rowOffsets);
                }
            }else{
                Muls(tmp,v,w.GetValue(k+1),Stride());PipeBarrier<PIPE_V>();Axpy(tmp,w,v.GetValue(k+1),Stride());PipeBarrier<PIPE_V>();Sub(pack,pack,tmp,Stride());PipeBarrier<PIPE_V>();
                for(uint32_t r=0;r<count;++r)if(base+r>k){auto a=mat[r*Stride()];Muls(tmp,v,w.GetValue(base+r),Stride());PipeBarrier<PIPE_V>();Axpy(tmp,w,v.GetValue(base+r),Stride());PipeBarrier<PIPE_V>();Sub(a,a,tmp,Stride());PipeBarrier<PIPE_V>();}
            }
            Barrier();
        }
    }
    __aicore__ __attribute__((noinline)) void Reduce(){
        // Replicate the next pivot row. Two workspace rows prevent a fast core
        // from overwriting a row before another core has consumed it.
        auto pivotRows=gw[Stride()];
        auto rowOffsets=piv.template ReinterpretCast<uint32_t>();
        if(active){Duplicate(d,0.0f,Stride());Duplicate(e,0.0f,Stride());Duplicate(beta,0.0f,Stride());Barrier();for(uint32_t r=0;r<(count+7)/8*8;++r)rowOffsets.SetValue(r,(base+r<n?base+r:n-1)*4);Barrier();}
        if(active && lane==0){
            Duplicate(w,0.0f,Stride());Barrier();DataCopy(gw,w,Stride());DataCopy(gw[2*Stride()],w,Stride());
            if(packedMatrix){ReadPackedRow(tmp,0);Barrier();DataCopy(gparam,tmp,Stride());}
            else DataCopy(gparam,mat,Stride());Barrier();
        }
        SyncAll();
        if(active){DataCopy(pack,gparam,Stride());Barrier();}
        uint32_t privateFirst=(n-1)/64*64;
        for(uint32_t k=0;k<n;++k){
            if(k==privateFirst){GatherTail(privateFirst);break;}
            // A closed row is no longer used by any following reflector.
            // Its w slot has been explicitly zeroed on the last live step.
            bool working=active && k<base+count;
            if(working){
                PublishPivot(k, pivotRows);
            }
            // SyncAll begins with PIPE_ALL before publishing its arrival.
            SyncAll();
            if(working){
                ApplyCollectedPivot(k, rowOffsets);
            }
        }
        // The last row owner remains live throughout the reduction. The
        // last two high reflector rows are unused by the backtransform, so
        // they hold the completed diagonal/off-diagonal for the other cores.
        if(active && lane+1==team){
            DataCopy(hist[(n-2)*Stride()],d,Stride());DataCopy(hist[(n-1)*Stride()],e,Stride());DataCopy(gparam,beta,Stride());Barrier();
        }
        SyncAll();
        if(active){
            DataCopy(origD,hist[(n-2)*Stride()],Stride());DataCopy(origE,hist[(n-1)*Stride()],Stride());DataCopy(beta,gparam,Stride());Barrier();
            DataCopy(d,origD,Stride());DataCopy(e,origE,Stride());Barrier();
        }
    }
    // Independent Sturm recurrences occupy vector lanes; the recurrence along
    // the tridiagonal remains sequential, while different eigenvalues run together.
    // Evaluate a complete binary decision tree in otherwise idle SIMD lanes.
    // Every midpoint uses the original binary arithmetic and every rank stops
    // at the same tolerance; several sequential bisections finish per sweep.
    __aicore__ inline void CountNegativePivots(uint32_t lanes, uint32_t repeats, LocalTensor<float> mid, LocalTensor<float> p, LocalTensor<float> negative, LocalTensor<float> flags, LocalTensor<float> scratch) {
        Barrier();SetVectorMask<float>(lanes);
        Sub<float,false>(p,mat,mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t i=1;i<n;++i){
            Div<float,false>(scratch,wideBuf[(i-1)*8],p,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,mat[i*8],mid,lanes,uint8_t(repeats),BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
            Sub<float,false>(p,p,scratch,lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(p,p,-1e-30f,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            ShiftRight<uint32_t,false>(flags.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,uint8_t(repeats),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),flags.template ReinterpretCast<int32_t>(),lanes,uint8_t(repeats),BinaryRepeatParams{1,1,1,8,8,8});
        }
    }
    __aicore__ inline float OriginalCommonGrid(float lower,float upperBound,float tolerance){
        uint32_t limit=4*Stride()<64*team?4*Stride():64*team;
        uint32_t branches=1;while(2*branches<=limit)branches*=2;
        uint32_t queries=lane<branches-1?(branches-2-lane)/team+1:0;
        uint32_t lanes=queries>64?64:(queries+7)/8*8,repeats=(queries+63)/64;
        auto mid=pack,p=pack[Stride()],negative=pack[2*Stride()],flags=pack[4*Stride()],scratch=pack[5*Stride()];
        Duplicate(piv,lower,64);Duplicate(fac,upperBound,64);Duplicate(mid,0.0f,((queries+63)/64)*64);Barrier();
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        Brcb(mat,origD,uint8_t(Stride()/8),{1,8});Barrier();
        SetGridMidpoints(queries, branches, lower, upperBound, mid);
        CountNegativePivots(lanes, repeats, mid, p, negative, flags, scratch);
        Barrier();Brcb(wideBuf,negative,uint8_t((queries+7)/8),{1,8});Barrier();
        DataCopyPad(gw[lane],wideBuf,DataCopyExtParams{uint16_t(queries),4,0,(team-1)*4,0});Barrier();
        SyncAll();DataCopy(prod,gw,branches);Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t left=0,right=branches;float l=lower,h=upperBound;
            while(right-left>1 && h-l>tolerance){
                uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(value==l || value==h)break;
                if(prod.template ReinterpretCast<uint32_t>().GetValue(j-1)<=EigenIndex(r)){left=j;l=value;}else{right=j;h=value;}
            }
            piv.SetValue(r,l);fac.SetValue(r,h);
        }
        return 1.0f/float(int32_t(branches));
    }
    __aicore__ inline void BisectTree(float lower,float upperBound,float tolerance,bool shared=false,float widthScale=1.0f){
        uint32_t maxLevels=1;while(maxLevels<6 && ((1u<<(maxLevels+1))-1)*count<=Stride()/64*64)++maxLevels;
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        Brcb(mat,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
        auto lo=piv,hi=fac,mid=pack,p=pack[Stride()],negative=pack[2*Stride()];
        auto scratch=pack[5*Stride()],flags=pack[4*Stride()];
        Duplicate(mid,0.0f,Stride());Barrier();
        if(!shared)for(uint32_t r=0;r<count;++r){lo.SetValue(r,lower);hi.SetValue(r,upperBound);}
        float maxWidth=0;
        for(uint32_t r=0;r<count;++r){float x=hi.GetValue(r)-lo.GetValue(r);maxWidth=maxWidth>x?maxWidth:x;}
        for(uint32_t it=0;it<32 && maxWidth>tolerance;++it){
            uint32_t levels=1;while(levels<maxLevels && maxWidth>tolerance*float(int32_t(1u<<levels)))++levels;
            uint32_t branches=1u<<levels,probes=branches-1,slots=probes*count;
            uint32_t lanes=slots>64?64:(slots+7)/8*8,repeats=(slots+63)/64;
            for(uint32_t r=0;r<count;++r){
                float l=lo.GetValue(r),h=hi.GetValue(r);uint32_t off=r*probes;
                for(uint32_t step=branches/2;step;step/=2)for(uint32_t j=step;j<branches;j+=2*step){
                    float a=j==step?l:mid.GetValue(off+j-step-1);
                    float b=j+step==branches?h:mid.GetValue(off+j+step-1);
                    mid.SetValue(off+j-1,0.5f*(a+b));
                }
            }
            CountNegativePivots(lanes, repeats, mid, p, negative, flags, scratch);
            Barrier();
            maxWidth=0;
            for(uint32_t r=0;r<count;++r){
                float l=lo.GetValue(r),h=hi.GetValue(r);uint32_t left=0,right=branches,off=r*probes;
                for(uint32_t depth=0;depth<levels && h-l>tolerance;++depth){
                    uint32_t j=(left+right)/2;float center=mid.GetValue(off+j-1);
                    if(center==l || center==h)break;
                    if(negative.template ReinterpretCast<uint32_t>().GetValue(off+j-1)<=EigenIndex(r)){l=center;left=j;}else{h=center;right=j;}
                }
                lo.SetValue(r,l);hi.SetValue(r,h);float width=h-l;maxWidth=maxWidth>width?maxWidth:width;
            }
        }
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),0.5f*(lo.GetValue(r)+hi.GetValue(r)));
    }
    __aicore__ inline void BisectVector(float lower,float upperBound,float tolerance,bool shared=false,float widthScale=1.0f){
        const uint32_t len=64,lanes=laneStride;
        // Coefficients and shifts come from this invocation's tridiagonal.
        // Materialize all diagonal-minus-shift values with vector repeats.
        Mul(tmp,origE,origE,Stride());PipeBarrier<PIPE_V>();
        Brcb(wideBuf,tmp,uint8_t(Stride()/8),{1,8});
        for(uint32_t col=0;col<lanes;col+=8)
            Brcb(mat[col],origD,uint8_t(Stride()/8),{uint16_t(lanes/8),uint16_t(lanes)});
        PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        auto lo=piv,hi=fac,mid=pack,p=pack[Stride()],negative=pack[2*Stride()];
        auto rank=pack[3*Stride()],one=pack[4*Stride()],scratch=pack[5*Stride()];
        auto mask=pack[6*Stride()].template ReinterpretCast<uint8_t>();
        auto activeMask=pack[6*Stride()+8].template ReinterpretCast<uint8_t>();
        auto oldLo=pack[7*Stride()],oldHi=pack[8*Stride()];
        if(!shared){Duplicate<float,false>(lo,lower,len,1,1,8);Duplicate<float,false>(hi,upperBound,len,1,1,8);}Duplicate<float,false>(one,0.0f,len,1,1,8);Barrier();
        for(uint32_t r=0;r<len;++r)rank.SetValue(r,float(int32_t(EigenIndex(r))));Barrier();
        float width=(upperBound-lower)*widthScale;
        for(uint32_t it=0;it<32 && width>0.5f*tolerance;++it,width*=0.5f){
            Adds<float,false>(oldLo,lo,0.0f,len,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,len,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(scratch,hi,lo,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,scratch,tolerance,CMPMODE::GT,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SetVectorMask<float>(lanes);
            for(uint32_t row=0;row<n;row+=255){
                uint8_t repeats=uint8_t(n-row<255?n-row:255);
                Sub<float,false>(prod[row*lanes],mat[row*lanes],mid,lanes,repeats,BinaryRepeatParams{1,1,1,uint8_t(lanes/8),uint8_t(lanes/8),0});
            }
            PipeBarrier<PIPE_V>();Adds<float,false>(p,prod,0.0f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            // The perturbation makes zero pivots negative before counting and division.
            Adds<float,false>(p,p,-1e-30f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            ShiftRight<uint32_t,false>(negative.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t i=1;i<n;++i){
                Div<float,false>(scratch,wideBuf[(i-1)*8],p,lanes,1,BinaryRepeatParams{1,0,1,8,0,8});PipeBarrier<PIPE_V>();
                Sub<float,false>(p,prod[i*lanes],scratch,lanes,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                Adds<float,false>(p,p,-1e-30f,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                ShiftRight<uint32_t,false>(one.template ReinterpretCast<uint32_t>(),p.template ReinterpretCast<uint32_t>(),31u,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
                // Updating the count is independent of the next recurrence division.
                Add<int32_t,false>(negative.template ReinterpretCast<int32_t>(),negative.template ReinterpretCast<int32_t>(),one.template ReinterpretCast<int32_t>(),lanes,1,BinaryRepeatParams{1,1,1,8,8,8});
            }
            PipeBarrier<PIPE_V>();Cast<float,int32_t,false>(one,negative.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,lanes,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            Compare<float,uint8_t,false>(mask,one,rank,CMPMODE::LE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,mask,mid,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,mask,hi,mid,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
        Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});Barrier();
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),mid.GetValue(r));
    }
    // Congruence elimination preserves the inertia of a symmetric
    // tridiagonal. Odd pivots are independent, so each level runs in SIMD.
    // Retain every eliminated pivot in mat and count their signs at the end.
    __aicore__ inline void SchurInertia(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){uint8_t rep=uint8_t(n-row<255?n-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,rep,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});}
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){uint8_t rep=uint8_t(n-row<255?n-row:255);
            Adds<float,false>(pcrBand[row*len],wideBuf[row*8],0.0f,len,rep,UnaryRepeatParams{1,0,uint8_t(len/8),1});}
        PipeBarrier<PIPE_V>();
        for(uint32_t step=1;step<n;step*=2){
            uint32_t odd=(n-1-step)/(2*step)+1,right=(n-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len],oe=pcrBand[step*len];
            if(dist/8<=255){for(uint32_t first=0;first<odd;first+=255){uint8_t rep=uint8_t(odd-first<255?odd-first:255);
                Adds<float,false>(od[first*dist],od[first*dist],-1e-30f,len,rep,UnaryRepeatParams{1,1,uint8_t(dist/8),uint8_t(dist/8)});}}
            else for(uint32_t i=0;i<odd;++i)Adds<float,false>(od[i*dist],od[i*dist],-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            PcrOp<1>(pcrLeft,len,one,0,od,dist,len,odd);
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<0>(wideBuf,len,wideBuf,len,pcrBand,dist,len,odd);
            PcrOp<2>(mat,dist,mat,dist,wideBuf,len,len,odd);
            if(right){
                PcrOp<0>(wideBuf,len,oe,dist,pcrLeft,len,len,right);
                PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,right);
                PcrOp<2>(mat[dist],dist,mat[dist],dist,wideBuf,len,len,right);
            }
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,odd);
            Muls(wideBuf,wideBuf,-1.0f,odd*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            PcrOp<0>(pcrBand,dist,wideBuf,len,one,0,len,odd);
        }
        Adds<float,false>(mat,mat,-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight(mat.template ReinterpretCast<uint32_t>(),mat.template ReinterpretCast<uint32_t>(),uint32_t(31),n*len);PipeBarrier<PIPE_V>();
        for(uint32_t activeRows=n;activeRows>1;activeRows=(activeRows+1)/2){
            uint32_t half=activeRows/2,next=(activeRows+1)/2;
            Add(mat.template ReinterpretCast<int32_t>(),mat.template ReinterpretCast<int32_t>(),mat[next*len].template ReinterpretCast<int32_t>(),half*len);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
    }
    __aicore__ inline void BisectInertia(float lower,float upperBound,float tolerance,bool shared){
        uint32_t len=(count+7)/8*8;
        auto lo=piv,hi=fac,mid=pack,one=pack[Stride()],rank=pack[2*Stride()],counts=pack[3*Stride()],width=pack[4*Stride()];
        auto oldLo=pack[7*Stride()],oldHi=pack[8*Stride()];
        auto mask=pack[5*Stride()].template ReinterpretCast<uint8_t>(),activeMask=pack[6*Stride()].template ReinterpretCast<uint8_t>();
        if(!shared){Duplicate(lo,lower,len);Duplicate(hi,upperBound,len);}
        Duplicate(one,1.0f,len);Barrier();
        for(uint32_t i=0;i<len;++i)rank.SetValue(i,float(int32_t(EigenIndex(i))));Barrier();
        for(uint32_t it=0;it<32;++it){
            float widest=0;for(uint32_t i=0;i<count;++i){float span=hi.GetValue(i)-lo.GetValue(i);widest=widest>span?widest:span;}
            if(widest<=tolerance)break;
            SetVectorMask<float>(len);
            Adds<float,false>(oldLo,lo,0.0f,len,1,UnaryRepeatParams{1,1,8,8});Adds<float,false>(oldHi,hi,0.0f,len,1,UnaryRepeatParams{1,1,8,8});
            Sub<float,false>(width,hi,lo,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Compares<float,uint8_t,false>(activeMask,width,tolerance,CMPMODE::GT,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            SchurInertia(mid,one,len);
            Cast<float,int32_t,false>(counts,mat.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Compare<float,uint8_t,false>(mask,counts,rank,CMPMODE::LE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,mask,mid,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,mask,hi,mid,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            Select<float,uint8_t,false>(lo,activeMask,lo,oldLo,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});
            Select<float,uint8_t,false>(hi,activeMask,hi,oldHi,SELMODE::VSEL_TENSOR_TENSOR_MODE,len,1,BinaryRepeatParams{1,1,1,8,8,8});Barrier();
        }
        SetVectorMask<float>(len);Add<float,false>(mid,lo,hi,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();Muls<float,false>(mid,mid,0.5f,len,1,UnaryRepeatParams{1,1,8,8});Barrier();
        for(uint32_t i=0;i<count;++i)d.SetValue(EigenIndex(i),mid.GetValue(i));
    }
    __aicore__ inline void ScaleOriginalEdges(LocalTensor<float> dst,LocalTensor<float> src,uint32_t sourceStride,uint32_t firstEdge,uint32_t len,uint32_t number){
        for(uint32_t first=0;first<number;first+=255){
            uint8_t repeats=uint8_t(number-first<255?number-first:255);
            Mul<float,false>(dst[first*len],wideBuf[(firstEdge+2*first)*8],src[first*sourceStride],len,repeats,
                BinaryRepeatParams{1,0,1,uint8_t(len/8),2,uint8_t(sourceStride/8)});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AccumulateOddPivotSigns(uint32_t odd, uint32_t dist, uint32_t len, bool batchable, LocalTensor<float> od) {
        for(uint32_t first=0;first<odd;){
            uint8_t repeats=batchable?uint8_t(odd-first<255?odd-first:255):1;
            Adds<float,false>(od[first*dist],od[first*dist],-1e-30f,len,repeats,
                UnaryRepeatParams{1,1,uint8_t(batchable?dist/8:8),uint8_t(batchable?dist/8:8)});
            first+=repeats;
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<odd;){
            uint8_t repeats=batchable?uint8_t(odd-first<255?odd-first:255):1;
            ShiftRight<uint32_t,false>(prod[first*len].template ReinterpretCast<uint32_t>(),od[first*dist].template ReinterpretCast<uint32_t>(),31u,len,repeats,
                UnaryRepeatParams{1,1,uint8_t(len/8),uint8_t(batchable?dist/8:8)});
            first+=repeats;
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t number=odd;number>1;number=(number+1)/2){
            uint32_t half=number/2,next=(number+1)/2;
            Add(prod.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),prod[next*len].template ReinterpretCast<int32_t>(),half*len);PipeBarrier<PIPE_V>();
        }
        Add(tmp2.template ReinterpretCast<int32_t>(),tmp2.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
    }
    __aicore__ inline void SchurInertiaInplace(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){
            uint8_t repeats=uint8_t(n-row<255?n-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
        // Match the original initialization, including the sign of zero.
        Adds(wideBuf,wideBuf,0.0f,8*Stride());
        Duplicate(tmp2.template ReinterpretCast<int32_t>(),int32_t(0),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t step=1;step<n;step*=2){
            uint32_t odd=(n-1-step)/(2*step)+1,right=(n-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len];bool batchable=dist/8<=255;
            AccumulateOddPivotSigns(odd, dist, len, batchable, od);
            // Once its sign is counted, an odd diagonal stores its reciprocal.
            // Once both neighbor updates finish, the same slot stores the new
            // off-diagonal. Old edges occupy disjoint midpoint slots.
            PcrOp<1>(od,dist,one,0,od,dist,len,odd);
            if(step==1){
                ScaleOriginalEdges(prod,od,dist,0,len,odd);
                ScaleOriginalEdges(prod,prod,len,0,len,odd);
            }else{
                auto left=mat[(step/2)*len];
                PcrOp<0>(prod,len,left,dist,od,dist,len,odd);
                PcrOp<0>(prod,len,prod,len,left,dist,len,odd);
            }
            PcrOp<2>(mat,dist,mat,dist,prod,len,len,odd);
            if(right){
                if(step==1){
                    ScaleOriginalEdges(prod,od,dist,1,len,right);
                    ScaleOriginalEdges(prod,prod,len,1,len,right);
                }else{
                    auto edge=mat[(step+step/2)*len];
                    PcrOp<0>(prod,len,edge,dist,od,dist,len,right);
                    PcrOp<0>(prod,len,prod,len,edge,dist,len,right);
                }
                PcrOp<2>(mat[dist],dist,mat[dist],dist,prod,len,len,right);
                if(step==1){
                    ScaleOriginalEdges(prod,od,dist,0,len,right);
                    ScaleOriginalEdges(prod,prod,len,1,len,right);
                }else{
                    PcrOp<0>(prod,len,mat[(step/2)*len],dist,od,dist,len,right);
                    PcrOp<0>(prod,len,prod,len,mat[(step+step/2)*len],dist,len,right);
                }
                Muls(prod,prod,-1.0f,right*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
                PcrOp<0>(od,dist,prod,len,one,0,len,right);
            }
        }
        Adds<float,false>(mat,mat,-1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        ShiftRight<uint32_t,false>(prod.template ReinterpretCast<uint32_t>(),mat.template ReinterpretCast<uint32_t>(),31u,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Add(mat.template ReinterpretCast<int32_t>(),tmp2.template ReinterpretCast<int32_t>(),prod.template ReinterpretCast<int32_t>(),len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
    }

    __aicore__ inline void SetGridMidpoints(uint32_t queries, uint32_t branches, float lower, float upperBound, LocalTensor<float> mid) {
        for (uint32_t r = 0; r < queries; ++r) {
            uint32_t target = lane + r * team + 1, left = 0, right = branches;
            float low = lower, high = upperBound;
            while (right - left > 1) {
                uint32_t middle = left + (right - left) / 2;
                float value = 0.5f * (low + high);
                if (middle == target) { mid.SetValue(r, value); break; }
                if (target < middle) { right = middle; high = value; }
                else { left = middle; low = value; }
            }
        }
    }
    __aicore__ __attribute__((noinline)) float CommonGrid(float lower,float upperBound,float tolerance){
        if(n<=25 || laneStride==64)return OriginalCommonGrid(lower,upperBound,tolerance);
        uint32_t limit=4*Stride()<64*team?4*Stride():64*team;
        uint32_t branches=1;while(2*branches<=limit)branches*=2;
        uint32_t len=0;
        while(branches>=2){
            uint32_t maxQueries=(branches-2)/team+1;
            len=(maxQueries+7)/8*8;
            uint32_t needed=(n+(n+1)/2)*len+4*Stride()+3*len+64;
            if(len<=64 && needed<=storageCapacity)break;
            branches/=2;
        }
        if(branches<2)return OriginalCommonGrid(lower,upperBound,tolerance);
        uint32_t queries=lane<branches-1?(branches-2-lane)/team+1:0;
        auto oldMat=mat,oldProd=prod,oldTmp2=tmp2;
        auto oldD=d,oldOrigD=origD,oldOrigE=origE,oldBeta=beta;
        DataCopy(wideBuf,origD,Stride());DataCopy(wideBuf[Stride()],origE,Stride());
        DataCopy(wideBuf[2*Stride()],beta,Stride());DataCopy(wideBuf[3*Stride()],d,Stride());Barrier();
        auto memory=storage.Get<float>();uint32_t prefix=(n+(n+1)/2)*len;
        mat=memory;prod=memory[n*len];origD=memory[prefix];origE=memory[prefix+Stride()];
        beta=memory[prefix+2*Stride()];d=memory[prefix+3*Stride()];
        auto mid=memory[prefix+4*Stride()],one=mid[len];tmp2=one[len];
        DataCopy(origD,wideBuf,Stride());DataCopy(origE,wideBuf[Stride()],Stride());
        DataCopy(beta,wideBuf[2*Stride()],Stride());DataCopy(d,wideBuf[3*Stride()],Stride());
        Duplicate(mid,0.0f,len);Duplicate(one,1.0f,len);Barrier();
        SetGridMidpoints(queries, branches, lower, upperBound, mid);
        Barrier();SchurInertiaInplace(mid,one,len);Barrier();
        if(queries){
            Brcb(wideBuf,mat,uint8_t((queries+7)/8),{1,8});Barrier();
            DataCopyPad(gw[lane],wideBuf,DataCopyExtParams{uint16_t(queries),4,0,(team-1)*4,0});Barrier();
        }
        // Save all metadata before rebinding views that may overlap the
        // temporary query matrix. Counts have already been published to GM.
        DataCopy(wideBuf,origD,Stride());DataCopy(wideBuf[Stride()],origE,Stride());
        DataCopy(wideBuf[2*Stride()],beta,Stride());DataCopy(wideBuf[3*Stride()],d,Stride());Barrier();
        mat=oldMat;prod=oldProd;tmp2=oldTmp2;d=oldD;origD=oldOrigD;origE=oldOrigE;beta=oldBeta;
        DataCopy(origD,wideBuf,Stride());DataCopy(origE,wideBuf[Stride()],Stride());
        DataCopy(beta,wideBuf[2*Stride()],Stride());DataCopy(d,wideBuf[3*Stride()],Stride());Barrier();
        SyncAll();DataCopy(prod,gw,branches);Barrier();
        for(uint32_t r=0;r<count;++r){
            uint32_t left=0,right=branches;float l=lower,h=upperBound;
            while(right-left>1 && h-l>tolerance){
                uint32_t j=(left+right)/2;float value=0.5f*(l+h);if(value==l || value==h)break;
                if(prod.template ReinterpretCast<uint32_t>().GetValue(j-1)<=EigenIndex(r)){left=j;l=value;}
                else{right=j;h=value;}
            }
            piv.SetValue(r,l);fac.SetValue(r,h);
        }
        return 1.0f/float(int32_t(branches));
    }
    __aicore__ inline void Bisect(){
        float lower=origD.GetValue(0),upperBound=lower;
        for(uint32_t i=0;i<n;++i){
            float radius=(i>0?Abs(origE.GetValue(i-1)):0)+(i+1<n?Abs(origE.GetValue(i)):0);
            float a=origD.GetValue(i)-radius,b=origD.GetValue(i)+radius;
            lower=lower<a?lower:a;upperBound=upperBound>b?upperBound:b;
        }
        float bound=Abs(lower)>Abs(upperBound)?Abs(lower):Abs(upperBound);
        spectralBound=bound;
        spectralCenter=0.5f*lower+0.5f*upperBound;
        float tolerance=1e-6f*bound+1e-30f;
        lower-=tolerance;upperBound+=tolerance;
        bool shared=Stride()>=64 && batch%groups==0;
        float widthScale=shared?CommonGrid(lower,upperBound,tolerance):1.0f;
        if(count>=2 && n>25){if(laneStride==64 && Stride()>=64)BisectVector(lower,upperBound,tolerance,shared,widthScale);else BisectInertia(lower,upperBound,tolerance,shared);return;}
        for(uint32_t r=0;r<count;++r){
            uint32_t index=EigenIndex(r);float lo=shared?piv.GetValue(r):lower,hi=shared?fac.GetValue(r):upperBound;
            for(uint32_t it=0;it<64 && hi-lo>tolerance;++it){
                float mid=0.5f*(lo+hi);if(mid==lo || mid==hi)break;
                float p=origD.GetValue(0)-mid;uint32_t negative=p<=0;
                for(uint32_t i=1;i<n;++i){
                    if(Abs(p)<1e-30f)p=-1e-30f;
                    float off=origE.GetValue(i-1);p=origD.GetValue(i)-mid-off*off/p;
                    negative+=p<=0;
                }
                if(negative<=index)lo=mid;else hi=mid;
            }
            d.SetValue(index,0.5f*(lo+hi));
        }
    }
    __aicore__ inline float RefineProbe(uint32_t lo,uint32_t hi,float shift,uint32_t boundary){
        float reciprocal=1.0f/SafePivot(piv.GetValue(lo)-shift);red.SetValue(lo,reciprocal);
        for(uint32_t i=lo+1;i<hi;++i){
            float f=fac.GetValue(i-1)*reciprocal;tmp2.SetValue(i-1,f);
            reciprocal=1.0f/SafePivot(piv.GetValue(i)-shift-fac.GetValue(i-1)*f);red.SetValue(i,reciprocal);
        }
        uint32_t first=lo/8*8,span=(hi+7)/8*8-first;Barrier();DataCopy(w[first],tmp[first],span);Barrier();
        for(uint32_t it=0;it<1;++it){
            float previous=w.GetValue(lo);for(uint32_t i=lo+1;i<hi;++i){previous=w.GetValue(i)-tmp2.GetValue(i-1)*previous;w.SetValue(i,previous);}
            float a=w.GetValue(hi-1)*red.GetValue(hi-1);w.SetValue(hi-1,a);float mx=Abs(a);
            for(int i=int(hi)-2;i>=int(lo);--i){a=(w.GetValue(i)-fac.GetValue(i)*a)*red.GetValue(i);w.SetValue(i,a);mx=mx>Abs(a)?mx:Abs(a);}
            float inv=mx>0?1.0f/mx:1.0f,norm=0;
            Barrier();Muls(w[first],w[first],inv,span);VectorToScalar();
            for(uint32_t i=lo;i<hi;++i){a=w.GetValue(i);norm+=a*a;}
            inv=norm>0?1.0f/Root(norm):1.0f;
            Barrier();Muls(w[first],w[first],inv,span);VectorToScalar();
        }
        float dot=0;for(uint32_t i=lo;i<hi;++i)dot+=tmp.GetValue(i)*w.GetValue(i);
        float sign=dot<0?-1.0f:1.0f;
        Barrier();Muls(tmp[first],w[first],sign,span);VectorToScalar();
        return tmp.GetValue(boundary);
    }
    __aicore__ inline float CanonicalSign(LocalTensor<float> q,float shift,uint32_t index){
        uint32_t cut0=n/2;
        float off0=origE.GetValue(upper?n-1-cut0:cut0-1);
        float seam0=q.GetValue(upper?n-cut0:cut0-1)+(off0<0?-1.0f:1.0f)*q.GetValue(upper?n-1-cut0:cut0);
        // This conservative bound only bypasses scratch construction when the
        // original first split would unambiguously return the same sign.
        if(Abs(off0)*Abs(seam0)>1e-10f*spectralBound)
            return seam0<0?1.0f:-1.0f;
        // Nearly deflated components need the original scalar normalization
        // and reduction order to preserve the reference basis orientation.
        float referenceShift=ScalarEigenvector(v,index,param.GetValue(index));
        float alignment=Dot(v,q)<0?-1.0f:1.0f;
        return alignment*CanonicalCore(v,referenceShift);
    }
    __aicore__ inline float IntervalBound(uint32_t lo,uint32_t hi){
        uint32_t length=hi-lo;
        Gather(red,pcrBand,v.template ReinterpretCast<uint32_t>(),lo*4,length);PipeBarrier<PIPE_V>();
        ReduceMax(w,red,tmp2,length,false);VectorToScalar();return w.GetValue(0);
    }
    __aicore__ inline float CanonicalCore(LocalTensor<float> q,float shift){
        if(!upper){DataCopy(piv,origD,Stride());DataCopy(fac,origE,Stride());DataCopy(tmp,q,Stride());Barrier();}
        else{
            auto reversed=pack[7*Stride()].template ReinterpretCast<int32_t>(),offIndices=pack[8*Stride()].template ReinterpretCast<int32_t>();
            CreateVecIndex(reversed,int32_t(0),Stride());PipeBarrier<PIPE_V>();Muls(reversed,reversed,int32_t(-4),Stride());PipeBarrier<PIPE_V>();Adds(reversed,reversed,int32_t((n-1)*4),Stride());PipeBarrier<PIPE_V>();
            Adds(offIndices,reversed,int32_t(-4),Stride());Duplicate(piv,0.0f,Stride());Duplicate(fac,0.0f,Stride());Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();
            Gather(piv,origD,reversed.template ReinterpretCast<uint32_t>(),0,n);
            Gather(fac,origE,offIndices.template ReinterpretCast<uint32_t>(),0,n-1);
            Gather(tmp,q,reversed.template ReinterpretCast<uint32_t>(),0,n);Barrier();
        }
        // Interior Gershgorin row bounds survive each nested split. Only the
        // new boundary row changes, so prepare the interior values once.
        auto indices=v.template ReinterpretCast<int32_t>(),left=pack[6*Stride()].template ReinterpretCast<int32_t>();
        CreateVecIndex(indices,int32_t(0),Stride());PipeBarrier<PIPE_V>();Muls(indices,indices,int32_t(4),Stride());PipeBarrier<PIPE_V>();
        Adds(left,indices,int32_t(-4),Stride());PipeBarrier<PIPE_V>();Maxs(left,left,int32_t(0),Stride());PipeBarrier<PIPE_V>();
        AscendC::Abs(pcrBand,piv,Stride());AscendC::Abs(tmp2,fac,Stride());PipeBarrier<PIPE_V>();
        Duplicate(w,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(w,tmp2,left.template ReinterpretCast<uint32_t>(),0,n);PipeBarrier<PIPE_V>();Duplicate(w,0.0f,1);PipeBarrier<PIPE_V>();
        Add(pcrBand,pcrBand,w,n);PipeBarrier<PIPE_V>();Add(pcrBand,pcrBand,tmp2,n-1);PipeBarrier<PIPE_V>();
        uint32_t lo=0,hi=n;
        while(hi-lo>25){
            uint32_t cut=(lo+hi)/2;float off=fac.GetValue(cut-1);
            float seam=tmp.GetValue(cut-1)+(off<0?-1.0f:1.0f)*tmp.GetValue(cut);
            float bound=IntervalBound(lo,hi);
            float tolerance=6.280369834735101e-16f*bound;
            if(Abs(off)*Abs(seam)>16*tolerance)return seam<0?1.0f:-1.0f;
            piv.SetValue(cut-1,piv.GetValue(cut-1)-Abs(off));piv.SetValue(cut,piv.GetValue(cut)-Abs(off));
            float left=0,right=0;
            for(uint32_t i=lo;i<cut;++i){float x=tmp.GetValue(i);left+=x*x;}
            for(uint32_t i=cut;i<hi;++i){float x=tmp.GetValue(i);right+=x*x;}
            // Retain the Gershgorin upper bound when testing a nearly deflated component.
            uint32_t nextLo=left>=right?lo:cut,nextHi=left>=right?cut:hi;
            float boundary=RefineProbe(nextLo,nextHi,shift,left>=right?cut-1:cut);
            if(Abs(off)*Abs(boundary)>6.280369834735101e-16f*bound)return seam<0?1.0f:-1.0f;
            uint32_t edge=left>=right?nextHi-1:nextLo;
            float rowBound=Abs(piv.GetValue(edge));
            rowBound+=edge>nextLo?Abs(fac.GetValue(edge-1)):0.0f;
            if(edge+1<nextHi)rowBound+=Abs(fac.GetValue(edge));
            pcrBand.SetValue(edge,rowBound);Barrier();
            lo=nextLo;hi=nextHi;
        }
        TridiagonalBasis canonical(wideBuf,hi-lo);
        return canonical.Probe(piv,fac,tmp,lo);
    }
    __aicore__ inline float SafePivot(float x){return Abs(x)<1e-20f?(x<0?-1e-20f:1e-20f):x;}
    __aicore__ inline uint32_t BackPanelWidth(){
        return n>25 && Stride()>=64 && Stride()%64==0 && laneStride>=16 && CubeMode!=0?(CubeMode==1?16:64):0;
    }
    __aicore__ inline uint64_t WorkspacePerMatrix(){
        uint64_t size=uint64_t(n<=25?2*n+3:2*n+5)*Stride();
        if(BackPanelWidth()){
            size+=uint64_t(n)*Stride()+uint64_t((team+1)/2*2)*laneStride*(Stride()+(CubeMode==1?16:64));
            if constexpr(CubeMode==2)size+=uint64_t((n-2+63)/64)*6*64*64;
        }
        return size;
    }
    __aicore__ inline bool ReferenceSensitive(){
        float center=Abs(spectralCenter),spread=spectralBound-center;
        float accumulated=float(int32_t(n))*1.1920928955078125e-7f*center;
        return spread<=0 || accumulated>4e-6f*spread;
    }
    __aicore__ inline void CaptureCubeOrientation(){
        Duplicate(beta,1.0f,Stride());Barrier();recordCubeSigns=true;
        uint32_t needed=0;auto selected=e.template ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];uint32_t cut=n/2;float off=origE.GetValue(upper?n-1-cut:cut-1);
            float seam=q.GetValue(upper?n-cut:cut-1)+(off<0?-1.0f:1.0f)*q.GetValue(upper?n-1-cut:cut);
            if(Abs(off)*Abs(seam)>1e-10f*spectralBound){if(seam>=0){beta.SetValue(r,-1.0f);Muls(q,q,-1.0f,Stride());}}
            else selected.SetValue(needed++,r);
        }
        Barrier();
        if(needed==1 && ReferenceSensitive()){uint32_t r=selected.GetValue(0);auto q=mat[r*Stride()];float sign=CanonicalSign(q,d.GetValue(EigenIndex(r)),EigenIndex(r));Barrier();if(sign<0){beta.SetValue(r,-1.0f);Muls(q,q,-1.0f,Stride());Barrier();}}
        else if(needed)ReferenceBatch(needed);
        recordCubeSigns=false;
    }
    __aicore__ inline void CubeBackHandshake(bool hasRows){
        auto q=hist[uint64_t(3*n+5)*Stride()+uint64_t(lane)*laneStride*Stride()];
        auto y=hist[uint64_t(3*n+5)*Stride()+uint64_t((team+1)/2*2)*laneStride*Stride()+uint64_t(lane)*laneStride*(CubeMode==1?16:64)];
        Barrier();
        Duplicate(prod,0.0f,laneStride*(CubeMode==1?16:64));Barrier();DataCopy(y,prod,laneStride*(CubeMode==1?16:64));
        if(hasRows){
            if(count<laneStride)Duplicate(mat[count*Stride()],0.0f,(laneStride-count)*Stride());
            Barrier();DataCopy(q,mat,laneStride*Stride());
        }else{
            Barrier();Duplicate(prod,0.0f,laneStride*Stride());Barrier();DataCopy(q,prod,laneStride*Stride());
        }
        // Complete Q/Y payload reads before local reference scratch is
        // reused; Cube receives its own immutable GM copy of unoriented Q.
        Barrier();CrossCoreSetFlag<2,PIPE_MTE3>(10);
        bool deferred=AlignReferenceSigns && hasRows;
        if(deferred)CaptureCubeOrientation();
        CrossCoreWaitFlag(9);
        if(hasRows){
            DataCopy(mat,q,laneStride*Stride());Barrier();
            if(deferred){
                for(uint32_t row=0;row<count;++row)if(beta.GetValue(row)<0.0f)Muls(mat[row*Stride()],mat[row*Stride()],-1.0f,Stride());
                Barrier();
            }
        }
    }
    __aicore__ inline void ComputePanelGram(uint32_t number, uint32_t dotTile, uint32_t height, uint32_t low, LocalTensor<float> gramMatrix, LocalTensor<float> partial) {
        for(uint32_t block=0;block<number;block+=dotTile){
            uint32_t cols=number-block<dotTile?number-block:dotTile,total=cols*height;
            Duplicate(prod,0.0f,total*64);PipeBarrier<PIPE_V>();
            for(uint32_t col=(low+1)/64*64;col<Stride();col+=64){
                for(uint32_t j=0;j<cols;++j)
                    Mul<float,false>(prod[j*height*64],mat[col],mat[(block+j)*Stride()+col],64,uint8_t(number),BinaryRepeatParams{1,1,1,8,uint8_t(Stride()/8),0});
                PipeBarrier<PIPE_V>();
                WholeReduceSum<float,false>(partial,prod,64,uint8_t(total),1,1,8);PipeBarrier<PIPE_V>();
                Add(gramMatrix[block*height],gramMatrix[block*height],partial,total);PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void BackSubstitutePanel(uint32_t number, uint32_t height, uint32_t low, LocalTensor<float> gramMatrix) {
        for(int j=int(number)-1;j>=0;--j){
            auto z=prod[uint32_t(j)*Stride()];
            DataCopy(z,mat[uint32_t(j)*Stride()],Stride());Barrier();
            for(int i=int(number)-1;i>j;--i){
                float gram=gramMatrix.GetValue(uint32_t(j)*height+uint32_t(i));
                Axpy(z,prod[uint32_t(i)*Stride()],-gram,Stride());PipeBarrier<PIPE_V>();
            }
            Muls(z,z,beta.GetValue(low+uint32_t(j)),Stride());Barrier();
        }
    }
    __aicore__ __attribute__((noinline)) void PrepareBackPanels(){
        if constexpr(CubeMode==1){
        uint32_t tile=BackPanelWidth();if(!tile)return;
        auto transformed=hist[(2*n+5)*Stride()];
        for(uint32_t low=lane*tile;low<n-2;low+=team*tile){
            uint32_t number=n-2-low;number=number<tile?number:tile;
            uint32_t height=(number+7)/8*8,dotTile=Stride()/64;
            auto gramMatrix=pack,partial=pack[Stride()>256?Stride():256];
            DataCopy(mat,hist[low*Stride()],number*Stride());
            Duplicate(gramMatrix,0.0f,number*height);Barrier();SetVectorMask<float>(64);
            ComputePanelGram(number, dotTile, height, low, gramMatrix, partial);
            VectorToScalar();
            BackSubstitutePanel(number, height, low, gramMatrix);
            Muls(prod,prod,-1.0f,number*Stride());PipeBarrier<PIPE_V>();
            uint32_t stored=number;
            if(low+number==n-2){Duplicate(prod[number*Stride()],0.0f,2*Stride());stored+=2;}
            Barrier();DataCopy(transformed[low*Stride()],prod,stored*Stride());Barrier();
        }

        }else{
        uint32_t tile=64,panels=(n-2+63)/64,parts=team/panels;
        uint32_t panel=lane/parts,part=lane%parts;if(panel>=panels)return;
        uint32_t low=panel*64,number=n-2-low;number=number<64?number:64;
        auto gamma=hist[uint64_t(3*n+5)*Stride()+uint64_t((team+1)/2*2)*laneStride*(Stride()+64)+uint64_t(panel)*6*64*64];
        // The Gram MMA writes its actual number-by-number rectangle only.
        // Copy that rectangle with explicit zero padding into a zeroed tile.
        uint32_t gramPitch=(number+7)/8*8;
        Duplicate(mat,0.0f,64*64);Barrier();
        DataCopyPad(mat,gamma,DataCopyExtParams{uint16_t(number),number*4,(64-number)*4,(64-gramPitch)/8,0},DataCopyPadExtParams<float>{true,0,uint8_t(gramPitch-number),0});Barrier();
        for(uint32_t row=part;row<64;row+=parts){
            // Only the selected strict upper triangle contributes to R.
            // Padding rows are explicitly zero and the identity is complete.
            if(row<number){Mul(prod,mat[row*64],beta[low],64);PipeBarrier<PIPE_V>();Duplicate(prod,0.0f,row+1);}
            else Duplicate(prod,0.0f,64);
            PipeBarrier<PIPE_V>();
            float scale=row<number?beta.GetValue(low+row):0.0f;
            Muls(w,prod,scale,64);Duplicate(tmp,0.0f,64);VectorToScalar();
            w.SetValue(row,-scale);tmp.SetValue(row,1.0f);Barrier();
            DataCopy(gamma[3*64*64+row*64],prod,64);
            DataCopy(gamma[64*64+row*64],w,64);
            DataCopy(gamma[5*64*64+row*64],tmp,64);Barrier();
        }
        if(lane==0){Duplicate(prod,0.0f,2*Stride());Barrier();DataCopy(hist[(2*n+5)*Stride()+(n-2)*Stride()],prod,2*Stride());Barrier();}
        }

    }
    __aicore__ inline void ApplyStoredReflector(int k, LocalTensor<float> current) {
        float b=beta.GetValue(k);
        if(b!=0){
            DotRows(current,k+1);
            if(packedMatrix){
                uint32_t padded=(count+7)/8*8,first=(k+1)/8*8;
                Brcb(pcrLeft,w,uint8_t(padded/8),{1,8});PipeBarrier<PIPE_V>();
                Muls(pcrLeft,pcrLeft,-b,padded*8);PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                for(uint32_t rb=0;rb<padded/8;++rb)
                    MulAddDst<float,float,false>(mat[rb*8*Stride()+first*8],current[first],pcrLeft[rb*64],64,uint8_t((Stride()-first)/8),BinaryRepeatParams{1,0,1,8,1,0});
            }else if(Stride()%64==0){
                Brcb(pack,w,uint8_t((count+7)/8),{1,8});PipeBarrier<PIPE_V>();Muls(pack,pack,-b,8*((count+7)/8*8));PipeBarrier<PIPE_V>();SetVectorMask<float>(64);
                for(uint32_t col=(k+1)/64*64;col<Stride();col+=64)
                    MulAddDst<float,float,false>(mat[col],current[col],pack,64,uint8_t(count),BinaryRepeatParams{1,1,0,uint8_t(Stride()/8),0,1});
            }else for(uint32_t r=0;r<count;++r){auto q=mat[r*Stride()];Axpy(q,current,-b*w.GetValue(r),Stride());}
        }
        Barrier();
    }
    __aicore__ inline void Backtransform(){
        if(BackPanelWidth()){CubeBackHandshake(true);return;}
        if(n<3)return;PackMatrix();
        // Retired solve storage holds two reflector tiles. Each tile comes
        // from this invocation's reduction; every reflector is applied in
        // exactly the original descending order and precision.
        uint32_t tile=laneStride/2;tile=tile<16?tile:16;
        uint32_t first=(n-3)/tile*tile,slot=0;
        DataCopy(pcrBand,hist[first*Stride()],(n-2-first)*Stride());Barrier();
        for(int block=int(first);block>=0;block-=int(tile)){
            if(block>=int(tile))DataCopy(pcrBand[(1-slot)*tile*Stride()],hist[(block-tile)*Stride()],tile*Stride());
            uint32_t last=uint32_t(block)+tile-1;last=last<n-2?last:n-3;
            for(int k=int(last);k>=block;--k){
                auto current=pcrBand[(slot*tile+uint32_t(k-block))*Stride()];
            ApplyStoredReflector(k, current);
            }
            slot=1-slot;
        }
        UnpackMatrix();
    }
    __aicore__ inline void FactorVector(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        auto a=pack[5*Stride()];
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){
            uint8_t repeats=uint8_t(n-row<255?n-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Mul(v,origE,origE,Stride());PipeBarrier<PIPE_V>();Muls(v,v,-1.0f,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        Adds<float,false>(a,mat,1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(mat,one,a,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t i=1;i<n;++i){
            MulAddDst<float,float,false>(mat[i*len],mat[(i-1)*len],wideBuf[(i-1)*8],len,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            Adds<float,false>(a,mat[i*len],1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(mat[i*len],one,a,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        }
        // Reuse factors within this solve. The next factorization overwrites
        // this table; every kernel invocation computes it from its own input.
        Muls(v,origE,-1.0f,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){
            uint8_t repeats=uint8_t(n-row<255?n-row:255);
            Mul<float,false>(pcrBand[row*len],mat[row*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
    }
    // Pair disjoint contiguous halves: no strided gather and no overlapping
    // repeat addresses. An odd middle row remains in the next active prefix.
    __aicore__ inline void SumEigenRows(uint32_t len){
        for(uint32_t activeRows=n;activeRows>1;activeRows=(activeRows+1)/2){
            uint32_t half=activeRows/2,next=(activeRows+1)/2;
            Add(mat,mat,mat[next*len],half*len);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
    }
    __aicore__ inline void RayleighVector(LocalTensor<float> shifts,uint32_t len){
        auto norm=pack[3*Stride()];
        Adds(v,origD,-spectralCenter,Stride());PipeBarrier<PIPE_V>();Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){
            uint8_t repeats=uint8_t(n-row<255?n-row:255);
            Mul<float,false>(mat[row*len],prod[row*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=1;row<n;row+=255){
            uint8_t repeats=uint8_t(n-row<255?n-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row-1)*len],wideBuf[(row-1)*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row+1<n;row+=255){
            uint8_t repeats=uint8_t(n-1-row<255?n-1-row:255);
            MulAddDst<float,float,false>(mat[row*len],prod[(row+1)*len],wideBuf[row*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();Mul(mat,mat,prod,n*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
        Adds<float,false>(shifts,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Mul(mat,prod,prod,n*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
        Adds<float,false>(norm,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(shifts,shifts,norm,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(shifts,shifts,spectralCenter,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    }
    template<uint32_t Op>
    __aicore__ inline void PcrOp(LocalTensor<float> dst,uint32_t ds,LocalTensor<float> a,uint32_t as,LocalTensor<float> b,uint32_t bs,uint32_t len,uint32_t number){
        bool batchable=ds/8<=255 && as/8<=255 && bs/8<=255;
        for(uint32_t first=0;first<number;){
            uint8_t repeats=batchable?uint8_t(number-first<255?number-first:255):1;
            BinaryRepeatParams br{1,1,1,uint8_t(batchable?ds/8:8),uint8_t(batchable?as/8:8),uint8_t(batchable?bs/8:8)};
            if constexpr(Op==0)Mul<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            if constexpr(Op==1)Div<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            if constexpr(Op==2)Sub<float,false>(dst[first*ds],a[first*as],b[first*bs],len,repeats,br);
            first+=repeats;
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void InitializeCyclicReduction(LocalTensor<float> shifts, uint32_t len) {
        Brcb(wideBuf,origD,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){
            uint8_t rep=uint8_t(n-row<255?n-row:255);
            Sub<float,false>(mat[row*len],wideBuf[row*8],shifts,len,rep,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){
            uint8_t rep=uint8_t(n-row<255?n-row:255);
            Adds<float,false>(pcrBand[row*len],wideBuf[row*8],0.0f,len,rep,UnaryRepeatParams{1,0,uint8_t(len/8),1});
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PcrSolve(LocalTensor<float> shifts,LocalTensor<float> one,uint32_t len){
        InitializeCyclicReduction(shifts, len);
        uint32_t step=1;
        for(;step<n;step*=2){
            uint32_t odd=(n-1-step)/(2*step)+1,keep=(n-1)/(2*step)+1,dist=2*step*len;
            auto od=mat[step*len],oq=prod[step*len],oe=pcrBand[step*len];
            // Normalize eliminated rows and retain the left factors temporarily.
            if(dist/8<=255){for(uint32_t first=0;first<odd;first+=255){uint8_t rep=uint8_t(odd-first<255?odd-first:255);Adds<float,false>(od[first*dist],od[first*dist],1e-20f,len,rep,UnaryRepeatParams{1,1,uint8_t(dist/8),uint8_t(dist/8)});}}
            else for(uint32_t j=0;j<odd;++j)Adds<float,false>(od[j*dist],od[j*dist],1e-20f,len,1,UnaryRepeatParams{1,1,8,8});
            PipeBarrier<PIPE_V>();
            PcrOp<1>(pcrLeft,len,one,0,od,dist,len,odd);
            PcrOp<0>(oe,dist,oe,dist,pcrLeft,len,len,odd);
            PcrOp<0>(oq,dist,oq,dist,pcrLeft,len,len,odd);
            PcrOp<0>(pcrLeft,len,pcrBand,dist,pcrLeft,len,len,odd);
            // Right-neighbor corrections use the original left off-diagonal.
            PcrOp<0>(wideBuf,len,pcrBand,dist,pcrLeft,len,len,odd);
            PcrOp<2>(mat,dist,mat,dist,wideBuf,len,len,odd);
            PcrOp<0>(wideBuf,len,pcrBand,dist,oq,dist,len,odd);
            PcrOp<2>(prod,dist,prod,dist,wideBuf,len,len,odd);
            // The original right off-diagonal is D_odd times its normalized value.
            PcrOp<0>(wideBuf,len,od,dist,oe,dist,len,keep-1);
            PcrOp<0>(wideBuf,len,wideBuf,len,oe,dist,len,keep-1);
            PcrOp<2>(mat[dist],dist,mat[dist],dist,wideBuf,len,len,keep-1);
            PcrOp<0>(wideBuf,len,od,dist,oe,dist,len,keep-1);
            PcrOp<0>(wideBuf,len,wideBuf,len,oq,dist,len,keep-1);
            PcrOp<2>(prod[dist],dist,prod[dist],dist,wideBuf,len,len,keep-1);
            // Connect the surviving rows and save both interpolation factors.
            PcrOp<0>(wideBuf,len,pcrBand,dist,oe,dist,len,odd);
            Muls(wideBuf,wideBuf,-1.0f,odd*len);PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
            PcrOp<0>(pcrBand,dist,wideBuf,len,one,0,len,odd);
            PcrOp<0>(od,dist,pcrLeft,len,one,0,len,odd);
        }
        auto scratch=pack[5*Stride()];
        Adds<float,false>(scratch,mat,1e-20f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(prod,prod,scratch,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        for(step/=2;step;step/=2){
            uint32_t odd=(n-1-step)/(2*step)+1,right=(n-1)/(2*step),dist=2*step*len;
            auto od=mat[step*len],oq=prod[step*len],oe=pcrBand[step*len];
            PcrOp<0>(wideBuf,len,od,dist,prod,dist,len,odd);
            PcrOp<2>(oq,dist,oq,dist,wideBuf,len,len,odd);
            if(right){PcrOp<0>(wideBuf,len,oe,dist,prod[dist],dist,len,right);PcrOp<2>(oq,dist,oq,dist,wideBuf,len,len,right);}
        }
    }
    __aicore__ inline void MaximumEigenRows(LocalTensor<float> mx,uint32_t len){
        AscendC::Abs(mat,prod,n*len);PipeBarrier<PIPE_V>();
        for(uint32_t number=n;number>1;number=(number+1)/2){uint32_t half=number/2,next=(number+1)/2;Max(mat,mat,mat[next*len],half*len);PipeBarrier<PIPE_V>();}
        SetVectorMask<float>(len);Adds<float,false>(mx,mat,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
    }
    // The reference orientation uses a prescribed scalar factorization and
    // reduction order. Solve several ambiguous columns together, while retaining
    // scalar reciprocals and the original per-column norms/Rayleigh products.
    __aicore__ inline float ReferenceRayleigh(LocalTensor<float> q){
        auto left=pcrLeft[Stride()].template ReinterpretCast<uint32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<uint32_t>();
        Mul(w,q,origD,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();
        Gather(tmp2,q,left,0,n);Gather(tmp,origE,left,0,n);PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,tmp,Stride());PipeBarrier<PIPE_V>();Duplicate(tmp2,0.0f,1);PipeBarrier<PIPE_V>();
        Add(w,w,tmp2,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(tmp2,q,right,0,n);PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,origE,Stride());PipeBarrier<PIPE_V>();Add(w,w,tmp2,n-1);PipeBarrier<PIPE_V>();
        return Dot(q,w);
    }
    __aicore__ inline void ReferenceFactor(LocalTensor<float> shifts,uint32_t number,uint32_t len){
        for(uint32_t r=0;r<number;++r){
            float ev=shifts.GetValue(r),reciprocal=1.0f/SafePivot(origD.GetValue(0)-ev);
            pcrBand.SetValue(r,reciprocal);
            for(uint32_t i=1;i<n;++i){
                float off=origE.GetValue(i-1),f=off*reciprocal;
                reciprocal=1.0f/SafePivot(origD.GetValue(i)-ev-off*f);
                pcrBand.SetValue(i*len+r,reciprocal);
            }
        }
        Barrier();SetVectorMask<float>(len);
    }
    __aicore__ inline void ForwardReferenceSolve(uint32_t len, uint32_t wideSize) {
        bool compact=len==8;
        if(compact){
            Brcb(wideBuf,origE,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
            Muls(wideBuf,wideBuf,-1.0f,n*len);PipeBarrier<PIPE_V>();Mul(wideBuf,wideBuf,pcrBand,n*len);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
        if(compact){
            for(uint32_t i=1;i<n;++i){MulAddDst<float,float,false>(prod[i*8],wideBuf[(i-1)*8],prod[(i-1)*8],8,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();}
        }else{
            // len>=16 implies a lane tile of at least 16 rows, hence
            // pcrLeft owns at least 8*Stride() floats. The gather indices
            // are generated after this temporary broadcast is consumed.
            Muls(v,origE,-1.0f,Stride());PipeBarrier<PIPE_V>();
            Brcb(pcrLeft,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();
            uint32_t capacity=wideSize/len;
            for(uint32_t first=0;first+1<n;first+=capacity){
                uint32_t numberRows=n-1-first<capacity?n-1-first:capacity;
                SetVectorMask<float>(len);
                for(uint32_t row=0;row<numberRows;row+=255){
                    uint8_t repeats=uint8_t(numberRows-row<255?numberRows-row:255);
                    Mul<float,false>(wideBuf[row*len],pcrBand[(first+row)*len],pcrLeft[(first+row)*8],len,repeats,BinaryRepeatParams{1,1,0,uint8_t(len/8),uint8_t(len/8),1});
                }
                PipeBarrier<PIPE_V>();
                for(uint32_t row=0;row<numberRows;++row){
                    uint32_t i=first+row+1;
                    MulAddDst<float,float,false>(prod[i*len],wideBuf[row*len],prod[(i-1)*len],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
                }
            }
        }
    }
    __aicore__ inline void BackSubstituteReference(uint32_t len, bool parallelMax, LocalTensor<float> mx, LocalTensor<float> a) {
        Duplicate<float,false>(mx,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
        for(int i=int(n)-1;i>=0;--i){
            auto q=prod[i*len];
            if(i+1<int(n)){Axpy<float,float,false>(q,prod[(i+1)*len],-origE.GetValue(i),len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();}
            Mul<float,false>(q,q,pcrBand[i*len],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            if(!parallelMax){AscendC::Abs<float,false>(a,q,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
            Max<float,false>(mx,mx,a,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();}
        }
        if(parallelMax){
            AscendC::Abs(wideBuf,prod,n*len);PipeBarrier<PIPE_V>();
            for(uint32_t rowsLeft=n;rowsLeft>1;rowsLeft=(rowsLeft+1)/2){uint32_t half=rowsLeft/2,next=(rowsLeft+1)/2;Max(wideBuf,wideBuf,wideBuf[next*len],half*len);PipeBarrier<PIPE_V>();}
            SetVectorMask<float>(len);Adds<float,false>(mx,wideBuf,0.0f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void NormalizeReferenceRows(uint32_t number, uint32_t len, LocalTensor<uint32_t> offsets, LocalTensor<uint32_t> selected, LocalTensor<float> invMax, LocalTensor<float> invNorm, LocalTensor<float> mx) {
        // The solve's gather offsets are consecutive integer indices.
        // Build them on the device with SIMD while retaining both boundary
        // entries exactly as in the scalar construction.
        auto index=offsets.template ReinterpretCast<int32_t>(),left=offsets[Stride()].template ReinterpretCast<int32_t>(),right=offsets[2*Stride()].template ReinterpretCast<int32_t>();
        CreateVecIndex(index,int32_t(0),Stride());CreateVecIndex(left,int32_t(-1),Stride());CreateVecIndex(right,int32_t(1),Stride());PipeBarrier<PIPE_V>();
        Muls(index,index,int32_t(len*4),Stride());Maxs(left,left,int32_t(0),Stride());Muls(right,right,int32_t(4),Stride());PipeBarrier<PIPE_V>();
        Muls(left,left,int32_t(4),Stride());VectorToScalar();offsets.SetValue(2*Stride()+n-1,0);Barrier();
            Duplicate(invMax,1.0f,len);Duplicate(invNorm,1.0f,len);VectorToScalar();
            for(uint32_t r=0;r<number;++r){
                float maximum = mx.GetValue(r);
                if (maximum == 0.0f) {
                    selected.SetValue(r, n);
                    continue;
                }
                float scale=1.0f/maximum;
                Duplicate(v,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(v,prod,offsets,r*4,n);PipeBarrier<PIPE_V>();
                Muls(v,v,scale,Stride());PipeBarrier<PIPE_V>();float norm=Root(Dot(v,v));
                if (norm == 0.0f) {
                    selected.SetValue(r, n);
                    continue;
                }
                invMax.SetValue(r,scale);invNorm.SetValue(r,1.0f/norm);
            }
            Barrier();SetVectorMask<float>(len);
            for(uint32_t row=0;row<n;row+=255){uint8_t rep=uint8_t(n-row<255?n-row:255);Mul<float,false>(prod[row*len],prod[row*len],invMax,len,rep,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();
            for(uint32_t row=0;row<n;row+=255){uint8_t rep=uint8_t(n-row<255?n-row:255);Mul<float,false>(prod[row*len],prod[row*len],invNorm,len,rep,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();
    }
    __aicore__ __attribute__((noinline)) void ReferenceBatch(uint32_t number){
        uint32_t len=(number+7)/8*8,wideN=n<=25?n:25;
        uint32_t wideSize=3*wideN*wideN+21*wideN+64;wideSize=wideSize>8*Stride()?wideSize:8*Stride();
        wideSize=wideSize>((Stride()+1)/2)*laneStride?wideSize:((Stride()+1)/2)*laneStride;
        bool parallelMax=n*len<=wideSize;
        auto selected=e.template ReinterpretCast<uint32_t>(),offsets=pcrLeft.template ReinterpretCast<uint32_t>();
        auto shifts=pack,mx=pack[2*Stride()],invMax=pack[3*Stride()],invNorm=pack[4*Stride()],a=pack[5*Stride()];
        Duplicate(shifts,0.0f,len);Duplicate(prod,0.0f,n*len);Duplicate(pcrBand,0.0f,n*len);Duplicate(pcrLeft,0.0f,3*Stride());Barrier();
        // A unit start at a largest component has a nonzero projection on
        // the requested eigenvector and avoids injecting forcing in its tails.
        for(uint32_t r=0;r<number;++r){
            uint32_t localRow=selected.GetValue(r),index=EigenIndex(localRow);
            shifts.SetValue(r,param.GetValue(index));auto q=mat[localRow*Stride()];Barrier();
            AscendC::Abs(w,q,Stride());PipeBarrier<PIPE_V>();
            ReduceMax(red,w,tmp,Stride(),true);VectorToScalar();
            uint32_t peak=red.template ReinterpretCast<uint32_t>().GetValue(1);
            prod.SetValue(peak*len+r,1.0f);
        }
        ReferenceFactor(shifts,number,len);
        ForwardReferenceSolve(len, wideSize);
        BackSubstituteReference(len, parallelMax, mx, a);
        NormalizeReferenceRows(number, len, offsets, selected, invMax, invNorm, mx);
        for(uint32_t r=0;r<number;++r){
            uint32_t localRow=selected.GetValue(r);
            // An empty reference cannot determine the sign of the computed vector.
            if (localRow == n) continue;
            auto q=mat[localRow*Stride()];
            Duplicate(v,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(v,prod,offsets,r*4,n);Barrier();
            float referenceShift=ReferenceRayleigh(v);
            float alignment=Dot(v,q)<0?-1.0f:1.0f;
            float sign=alignment*CanonicalCore(v,referenceShift);Barrier();
            if(sign<0){if(recordCubeSigns)beta.SetValue(localRow,-1.0f);Muls(q,q,-1.0f,Stride());Barrier();}
        }
    }
    __aicore__ inline float CenteredResidual(LocalTensor<float> q,float ev){
        auto left=pcrLeft[Stride()].template ReinterpretCast<uint32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<uint32_t>();
        Adds(tmp2,origD,-ev,Stride());PipeBarrier<PIPE_V>();Mul(w,q,tmp2,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();
        Gather(tmp2,q,left,0,n);Gather(tmp,origE,left,0,n);PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,tmp,Stride());PipeBarrier<PIPE_V>();Duplicate(tmp2,0.0f,1);PipeBarrier<PIPE_V>();Add(w,w,tmp2,Stride());PipeBarrier<PIPE_V>();
        Duplicate(tmp2,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(tmp2,q,right,0,n);PipeBarrier<PIPE_V>();
        Mul(tmp2,tmp2,origE,Stride());PipeBarrier<PIPE_V>();Add(w,w,tmp2,n-1);PipeBarrier<PIPE_V>();
        return Root(Dot(w,w));
    }
    // Certify the invocation-local tridiagonal vectors before reflectors can
    // hide residuals under a large spectral translation. Adjacent columns are
    // shared only within this call and retain their original rank.
    __aicore__ inline void CollectSpectralGaps(GlobalTensor<float> stage) {
        for(uint32_t r=0;r<count;++r){DataCopy(stage[EigenIndex(r)*Stride()],mat[r*Stride()],Stride());pack.SetValue(8*r,d.GetValue(EigenIndex(r)));}
        Barrier();DataCopyPad(gw[lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});Barrier();
        SyncAll();DataCopy(e,gw,Stride());Barrier();
        for(uint32_t r=0;r<count;++r){uint32_t j=EigenIndex(r);float gap=spectralBound;
            if(j)gap=e.GetValue(j)-e.GetValue(j-1);
            if(j+1<n){float other=e.GetValue(j+1)-e.GetValue(j);gap=gap<other?gap:other;}
            pcrBand.SetValue(r,gap);
        }
    }
    __aicore__ __attribute__((noinline)) void CertifyInverse(){
        constexpr bool collective=true;
        auto stage=hist[(n+5)*Stride()];
        if(collective){
            CollectSpectralGaps(stage);
        }
        auto left=pcrLeft[Stride()].template ReinterpretCast<int32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<int32_t>();
        CreateVecIndex(left,int32_t(-1),Stride());CreateVecIndex(right,int32_t(1),Stride());PipeBarrier<PIPE_V>();
        Muls(left,left,int32_t(4),Stride());Muls(right,right,int32_t(4),Stride());PipeBarrier<PIPE_V>();
        Maxs(left,left,int32_t(0),Stride());Mins(right,right,int32_t((n-1)*4),Stride());Barrier();
        float spread=spectralBound-Abs(spectralCenter),limit=2e-6f*spread+1e-25f;
        float gramSquared=0,largestCorrelation=0;
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];uint32_t j=EigenIndex(r);float norm=Dot(q,q),residual=CenteredResidual(q,d.GetValue(j));
            float tolerance=limit;if(collective){float relative=1e-2f*pcrBand.GetValue(r);tolerance=tolerance<relative?tolerance:relative;}
            bool bad=!(residual<=tolerance) || !(Abs(norm-1.0f)<=5e-6f);
            float correlation=0;
            if(collective){
                if(j){DataCopy(tmp,stage[(j-1)*Stride()],Stride());Barrier();float x=Abs(Dot(q,tmp));correlation=x;}
                if(j+1<n){DataCopy(tmp,stage[(j+1)*Stride()],Stride());Barrier();float x=Abs(Dot(q,tmp));gramSquared+=2*x*x;correlation=correlation>x?correlation:x;}
            }
            gramSquared+=(norm-1.0f)*(norm-1.0f);largestCorrelation=largestCorrelation>correlation?largestCorrelation:correlation;
            pcrBand.SetValue(r,correlation);
            if(correlation>3e-5f)bad=true;
            e.SetValue(r,bad?(norm>1e-20f && norm<1e20f?1.0f:2.0f):0.0f);
        }
        if(collective){
            pack.SetValue(0,gramSquared);pack.SetValue(1,largestCorrelation);Barrier();
            DataCopyPad(gw[Stride()+2*lane],pack,DataCopyExtParams{1,8,0,0,0});Barrier();
            SyncAll();DataCopyPad(prod,gw[Stride()],DataCopyExtParams{1,team*8,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
            float total=0,maximum=0;for(uint32_t i=0;i<team;++i){total+=prod.GetValue(2*i);float x=prod.GetValue(2*i+1);maximum=maximum>x?maximum:x;}
            // Budget the measured diagonal and adjacent Gram contributions
            // across the complete matrix, rather than accepting every pair
            // independently. Refine the dominant offending columns together.
            if(total>4e-9f){for(uint32_t r=0;r<count;++r)if(pcrBand.GetValue(r)>=0.25f*maximum && e.GetValue(r)==0)e.SetValue(r,1.0f);}
        }
        for(uint32_t r=0;r<count;++r)if(e.GetValue(r)!=0){
            uint32_t j=EigenIndex(r);float ev=d.GetValue(j);
            union FloatBits{float f;uint32_t u;}bits;bits.f=ev;
            if((bits.u&0x7f800000u)==0x7f800000u)ev=param.GetValue(j);
            ev=PivotedExtendedEigenvector(mat[r*Stride()],j,ev,e.GetValue(r)==1.0f);d.SetValue(j,ev);
        }
        Barrier();
    }

    __aicore__ inline void SeedInverseVectors(uint32_t len) {
        // Generate the same invocation-local hash start vectors with integer
        // SIMD operations. No input-independent tensor is supplied by the host.
        auto rowHash=v.template ReinterpretCast<int32_t>(),colHash=pack[7*Stride()].template ReinterpretCast<int32_t>();
        CreateVecIndex(rowHash,int32_t(1),Stride());PipeBarrier<PIPE_V>();Muls(rowHash,rowHash,int32_t(747796405),Stride());Barrier();
        for(uint32_t r=0;r<len;++r)colHash.SetValue(r,int32_t((EigenIndex(r)+1)*2891336453u));Barrier();
        Brcb(wideBuf,v,uint8_t(Stride()/8),{1,8});PipeBarrier<PIPE_V>();SetVectorMask<float>(len);
        for(uint32_t row=0;row<n;row+=255){
            uint8_t repeats=uint8_t(n-row<255?n-row:255);
            Add<int32_t,false>(prod[row*len].template ReinterpretCast<int32_t>(),wideBuf[row*8].template ReinterpretCast<int32_t>(),colHash,len,repeats,BinaryRepeatParams{1,0,1,uint8_t(len/8),1,0});
        }
        PipeBarrier<PIPE_V>();
        for(uint32_t first=0;first<n*len;first+=4*Stride()){
            uint32_t k=n*len-first<4*Stride()?n*len-first:4*Stride();
            auto h=prod[first].template ReinterpretCast<uint32_t>(),shifted=mat[first].template ReinterpretCast<uint32_t>();
            auto h16=h.template ReinterpretCast<uint16_t>(),s16=shifted.template ReinterpretCast<uint16_t>(),t16=wideBuf.template ReinterpretCast<uint16_t>();
            ShiftRight(shifted,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            Muls(h.template ReinterpretCast<int32_t>(),h.template ReinterpretCast<int32_t>(),int32_t(2246822519u),k);PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(13),k);PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*k);PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*k);PipeBarrier<PIPE_V>();Not(t16,t16,2*k);PipeBarrier<PIPE_V>();And(h16,h16,t16,2*k);PipeBarrier<PIPE_V>();
            ShiftLeft(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();ShiftRight(h,h,uint32_t(16),k);PipeBarrier<PIPE_V>();
            Cast(prod[first],h.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,k);PipeBarrier<PIPE_V>();
            Muls(prod[first],prod[first],1.0f/65536.0f,k);PipeBarrier<PIPE_V>();Adds(prod[first],prod[first],-0.5f,k);PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float>(len);
        Barrier();
    }
    __aicore__ inline void OrientInverseVectors() {
        if(AlignReferenceSigns && !BackPanelWidth()){
        uint32_t needed=0;auto selected=e.template ReinterpretCast<uint32_t>();
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];uint32_t cut=n/2;float off=origE.GetValue(upper?n-1-cut:cut-1);
            float seam=q.GetValue(upper?n-cut:cut-1)+(off<0?-1.0f:1.0f)*q.GetValue(upper?n-1-cut:cut);
            if(Abs(off)*Abs(seam)>1e-10f*spectralBound){if(seam>=0)Muls(q,q,-1.0f,Stride());}
            else selected.SetValue(needed++,r);
        }
        Barrier();
        if(needed==1 && ReferenceSensitive()){uint32_t r=selected.GetValue(0);auto q=mat[r*Stride()];float sign=CanonicalSign(q,d.GetValue(EigenIndex(r)),EigenIndex(r));Barrier();if(sign<0){Muls(q,q,-1.0f,Stride());Barrier();}}
        else if(needed)ReferenceBatch(needed);
        }
    }
    __aicore__ inline void InverseVector(){
        uint32_t len=laneStride;SetVectorMask<float>(len);
        auto shifts=pack,one=pack[Stride()],mx=pack[2*Stride()],norm=pack[3*Stride()];
        auto inv=pack[4*Stride()],a=pack[5*Stride()],b=pack[6*Stride()];
        Duplicate<float,false>(shifts,0.0f,len,1,1,8);Duplicate<float,false>(one,1.0f,len,1,1,8);Barrier();
        for(uint32_t r=0;r<len;++r)shifts.SetValue(r,d.GetValue(EigenIndex(r)<n?EigenIndex(r):n-1));
        SeedInverseVectors(len);
        uint32_t fixedIterations=(n&(n-1))==0?2:3,totalIterations=(n&(n-1))==0?3:5;
        for(uint32_t it=0;it<totalIterations;++it){
            if(it>=1){if(it>=fixedIterations)RayleighVector(shifts,len);FactorVector(shifts,one,len);}
            if(it<1){PcrSolve(shifts,one,len);}else{
            for(uint32_t i=1;i<n;++i){
                MulAddDst<float,float,false>(prod[i*laneStride],pcrBand[(i-1)*laneStride],prod[(i-1)*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            Duplicate<float,false>(mx,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            for(int i=int(n)-1;i>=0;--i){
                auto q=prod[i*laneStride];
                if(i+1<int(n)){MulAddDst<float,float,false>(q,prod[(i+1)*laneStride],wideBuf[i*8],len,1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();}
                Mul<float,false>(q,q,mat[i*laneStride],len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            }
            }
            MaximumEigenRows(mx,len);
            Maxs<float,false>(mx,mx,1e-30f,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,mx,len,1,BinaryRepeatParams{1,1,1,8,8,8});Duplicate<float,false>(norm,0.0f,len,1,1,8);PipeBarrier<PIPE_V>();
            if(it<2){
                for(uint32_t row=0;row<n;row+=255){uint8_t repeats=uint8_t(n-row<255?n-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
                PipeBarrier<PIPE_V>();continue;
            }
            for(uint32_t row=0;row<n;row+=255){uint8_t repeats=uint8_t(n-row<255?n-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();Mul(mat,prod,prod,n*len);PipeBarrier<PIPE_V>();SumEigenRows(len);
            Sqrt<float,false>(norm,mat,len,1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();Div<float,false>(inv,one,norm,len,1,BinaryRepeatParams{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
            for(uint32_t row=0;row<n;row+=255){uint8_t repeats=uint8_t(n-row<255?n-row:255);Mul<float,false>(prod[row*len],prod[row*len],inv,len,repeats,BinaryRepeatParams{1,1,1,uint8_t(len/8),uint8_t(len/8),0});}
            PipeBarrier<PIPE_V>();
        }
        RayleighVector(shifts,len);Barrier();
        for(uint32_t r=0;r<count;++r)d.SetValue(EigenIndex(r),shifts.GetValue(r));
        auto offsets=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t i=0;i<n;++i)offsets.SetValue(i,i*laneStride*4);Barrier();
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            Duplicate(q,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(q,prod,offsets,r*4,n);
        }
        Barrier();
        CertifyInverse();
        OrientInverseVectors();
        Backtransform();
    }
    __aicore__ inline void SeedScalarVector(LocalTensor<float> q, uint32_t index) {
        if(n>25){
            // The same invocation-local start, generated with integer SIMD.
            auto h=q.template ReinterpretCast<uint32_t>(),shifted=w.template ReinterpretCast<uint32_t>();
            auto h16=h.template ReinterpretCast<uint16_t>(),s16=shifted.template ReinterpretCast<uint16_t>(),t16=red.template ReinterpretCast<uint16_t>();
            CreateVecIndex(q.template ReinterpretCast<int32_t>(),int32_t(1),Stride());PipeBarrier<PIPE_V>();
            Muls(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t(747796405),Stride());PipeBarrier<PIPE_V>();
            Adds(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t((index+1)*2891336453u),Stride());PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Not(t16,t16,2*Stride());PipeBarrier<PIPE_V>();And(h16,h16,t16,2*Stride());PipeBarrier<PIPE_V>();
            Muls(q.template ReinterpretCast<int32_t>(),q.template ReinterpretCast<int32_t>(),int32_t(2246822519u),Stride());PipeBarrier<PIPE_V>();
            ShiftRight(shifted,h,uint32_t(13),Stride());PipeBarrier<PIPE_V>();
            And(t16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Or(h16,h16,s16,2*Stride());PipeBarrier<PIPE_V>();Not(t16,t16,2*Stride());PipeBarrier<PIPE_V>();And(h16,h16,t16,2*Stride());PipeBarrier<PIPE_V>();
            ShiftLeft(h,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();ShiftRight(h,h,uint32_t(16),Stride());PipeBarrier<PIPE_V>();
            Cast(q,q.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,Stride());PipeBarrier<PIPE_V>();
            Muls(q,q,1.0f/65536.0f,Stride());PipeBarrier<PIPE_V>();Adds(q,q,-0.5f,Stride());Barrier();
            for(uint32_t i=n;i<Stride();++i)q.SetValue(i,0);Barrier();
        }else{
        for(uint32_t i=0;i<n;++i){uint32_t h=(i+1)*747796405u+(index+1)*2891336453u;h^=h>>16;h*=2246822519u;h^=h>>13;q.SetValue(i,float(int32_t(h&65535u))/65536.0f-0.5f);}        }
    }
    __aicore__ inline void FactorShiftedTridiagonal(float ev) {
        float reciprocal=1.0f/SafePivot(origD.GetValue(0)-ev);piv.SetValue(0,reciprocal);
        for(uint32_t i=1;i<n;++i){
            float f=origE.GetValue(i-1)*reciprocal;fac.SetValue(i-1,f);
            reciprocal=1.0f/SafePivot(origD.GetValue(i)-ev-origE.GetValue(i-1)*f);piv.SetValue(i,reciprocal);
        }
    }
    __aicore__ inline float ScalarRayleigh(LocalTensor<float> q) {
        if(n>25)return ReferenceRayleigh(q);
        for(uint32_t i=0;i<n;++i){
            float a=origD.GetValue(i)*q.GetValue(i);
            if(i>0)a+=origE.GetValue(i-1)*q.GetValue(i-1);
            if(i+1<n)a+=origE.GetValue(i)*q.GetValue(i+1);
            w.SetValue(i,a);
        }
        for(uint32_t i=n;i<Stride();++i)w.SetValue(i,0);
        Barrier();return Dot(q,w);
    }
    __aicore__ inline float ScalarEigenvector(LocalTensor<float> q,uint32_t index,float ev){
        Duplicate(q,0.0f,Stride());Barrier();
        if(n>25){
            auto left=pcrLeft[Stride()].template ReinterpretCast<int32_t>(),right=pcrLeft[2*Stride()].template ReinterpretCast<int32_t>();
            CreateVecIndex(left,int32_t(-1),Stride());CreateVecIndex(right,int32_t(1),Stride());PipeBarrier<PIPE_V>();
            Muls(left,left,int32_t(4),Stride());Muls(right,right,int32_t(4),Stride());PipeBarrier<PIPE_V>();
            Maxs(left,left,int32_t(0),Stride());Mins(right,right,int32_t((n-1)*4),Stride());Barrier();
        }
        FactorShiftedTridiagonal(ev);
        SeedScalarVector(q, index);

        for(uint32_t it=0;it<3;++it){
            if(it==2){
                ev = ScalarRayleigh(q);
                FactorShiftedTridiagonal(ev);
            }
            float previous=q.GetValue(0);for(uint32_t i=1;i<n;++i){previous=q.GetValue(i)-fac.GetValue(i-1)*previous;q.SetValue(i,previous);}
            float a=q.GetValue(n-1)*piv.GetValue(n-1);q.SetValue(n-1,a);float mx=Abs(a);
            for(int i=int(n)-2;i>=0;--i){a=(q.GetValue(i)-origE.GetValue(i)*a)*piv.GetValue(i);q.SetValue(i,a);mx=mx>Abs(a)?mx:Abs(a);}
            Barrier();
            if (mx != 0.0f) {
                Muls(q,q,1.0f/mx,Stride());Barrier();
                float norm=Root(Dot(q,q));
                if (norm != 0.0f) {
                    Muls(q,q,1.0f/norm,Stride());Barrier();
                    continue;
                }
            }
            // Restart a collapsed inverse iterate with a unit vector.
            Duplicate(q,0.0f,Stride());Barrier();
            q.SetValue(index,1.0f);Barrier();
        }
        // Refine each bisection shift with the converged tridiagonal eigenvector.
        return ScalarRayleigh(q);
    }
    __aicore__ inline ExtendedFloat QExtended(LocalTensor<float> q,uint32_t i){return ExtendedFloat(q.GetValue(i),prod.GetValue(i),0);}
    __aicore__ inline void SetQExtended(LocalTensor<float> q,uint32_t i,ExtendedFloat x){q.SetValue(i,x.h);prod.SetValue(i,x.m);}
    __aicore__ inline ExtendedFloat AbsExtended(ExtendedFloat x){return x.Negative()?-x:x;}
    __aicore__ inline ExtendedFloat SafeExtended(ExtendedFloat x){
        float magnitude=Abs(x.h);if(magnitude<1e-25f && Abs(x.m)<1e-25f)return ExtendedFloat(x.Negative()?-1e-25f:1e-25f);return x;
    }
    __aicore__ inline void PivotFactorExtended(ExtendedFloat shift){
        auto swaps=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t i=0;i<n;++i){
            WriteExtended(i,ExtendedFloat(origD.GetValue(i))-shift);
            WriteExtended(n+i,ExtendedFloat(i+1<n?origE.GetValue(i):0));
            WriteExtended(2*n+i,ExtendedFloat(i+1<n?origE.GetValue(i):0));
            WriteExtended(3*n+i,ExtendedFloat(0));swaps.SetValue(i,0);
        }
        for(uint32_t i=0;i+1<n;++i){
            ExtendedFloat diagonal=ReadExtended(i),lower=ReadExtended(n+i);
            if(AbsExtended(diagonal).h>=AbsExtended(lower).h){
                ExtendedFloat multiplier=lower/SafeExtended(diagonal);WriteExtended(n+i,multiplier);
                WriteExtended(i+1,ReadExtended(i+1)-multiplier*ReadExtended(2*n+i));
            }else{
                ExtendedFloat multiplier=diagonal/SafeExtended(lower),next=ReadExtended(i+1);
                WriteExtended(i,lower);WriteExtended(i+1,ReadExtended(2*n+i)-multiplier*next);
                WriteExtended(n+i,multiplier);WriteExtended(2*n+i,next);swaps.SetValue(i,1);
                if(i+2<n){ExtendedFloat upper=ReadExtended(2*n+i+1);WriteExtended(3*n+i,upper);WriteExtended(2*n+i+1,-multiplier*upper);}
            }
        }
    }
    __aicore__ inline ExtendedFloat RayleighExtended(LocalTensor<float> q){
        ExtendedFloat result(0),norm(0);
        for(uint32_t i=0;i<n;++i){
            ExtendedFloat x=QExtended(q,i);norm=norm+x*x;
            result=result+ExtendedFloat(origD.GetValue(i))*x*x;
            if(i+1<n)result=result+(ExtendedFloat(origE.GetValue(i))*x*QExtended(q,i+1)).Scale(2.0f);
        }
        return result/norm;
    }
    __aicore__ __attribute__((noinline)) float PivotedExtendedEigenvector(LocalTensor<float> q,uint32_t index,float ev,bool warmStart=false){
        for(uint32_t i=0;i<n;++i){
            uint32_t hash=(i+1)*747796405u+(index+1)*2891336453u;hash^=hash>>16;hash*=2246822519u;hash^=hash>>13;
            SetQExtended(q,i,ExtendedFloat(warmStart?q.GetValue(i):float(int32_t(hash&65535u))/65536.0f-0.5f));
        }
        ExtendedFloat shift(ev);PivotFactorExtended(shift);
        auto swaps=piv.template ReinterpretCast<uint32_t>();
        for(uint32_t it=0;it<5;++it){
            if(it>=3){shift=RayleighExtended(q);PivotFactorExtended(shift);}
            for(uint32_t i=0;i+1<n;++i){
                ExtendedFloat a=QExtended(q,i),b=QExtended(q,i+1),f=ReadExtended(n+i);
                if(swaps.GetValue(i)){SetQExtended(q,i,b);SetQExtended(q,i+1,a-f*b);}
                else SetQExtended(q,i+1,b-f*a);
            }
            for(int i=int(n)-1;i>=0;--i){
                ExtendedFloat value=QExtended(q,uint32_t(i));
                if(i+1<int(n))value=value-ReadExtended(2*n+uint32_t(i))*QExtended(q,uint32_t(i)+1);
                if(i+2<int(n))value=value-ReadExtended(3*n+uint32_t(i))*QExtended(q,uint32_t(i)+2);
                SetQExtended(q,uint32_t(i),value/SafeExtended(ReadExtended(uint32_t(i))));
            }
            float maximum=0;for(uint32_t i=0;i<n;++i){float x=Abs(QExtended(q,i).h);maximum=maximum>x?maximum:x;}
            ExtendedFloat scale(maximum>0?maximum:1),norm(0);
            for(uint32_t i=0;i<n;++i){ExtendedFloat x=QExtended(q,i)/scale;SetQExtended(q,i,x);norm=norm+x*x;}
            ExtendedFloat length=RootExtended(norm);
            if (length.h == 0.0f) {
                for (uint32_t i=0;i<n;++i) SetQExtended(q,i,ExtendedFloat(i == index ? 1.0f : 0.0f));
                continue;
            }
            for(uint32_t i=0;i<n;++i)SetQExtended(q,i,QExtended(q,i).DivideBy(length));
        }
        ExtendedFloat refined=RayleighExtended(q);
        for(uint32_t i=n;i<Stride();++i)q.SetValue(i,0);
        Barrier();return refined.Float();
    }
    __aicore__ inline void Inverse(){
        if(n>25 && n>=2*team){InverseVector();return;}
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];float ev=d.GetValue(EigenIndex(r));
            if(n>25 && (n&(n-1))!=0)ev=PivotedExtendedEigenvector(q,EigenIndex(r),ev);
            else ev=ScalarEigenvector(q,EigenIndex(r),ev);d.SetValue(EigenIndex(r),ev);
            if constexpr (AlignReferenceSigns) {
            float sign;
            if(n<=25){TridiagonalBasis canonical(wideBuf,n);sign=canonical.ProbeSaved(q,upper);}
            else if((n&(n-1))!=0)sign=CanonicalSign(q,d.GetValue(EigenIndex(r)),EigenIndex(r));
            else sign=CanonicalCore(q,d.GetValue(EigenIndex(r)));Barrier();
            if(sign<0){Muls(q,q,-1.0f,Stride());Barrier();}
            }
        }
        Backtransform();
    }

    __aicore__ inline void Output(uint32_t b){
        if(count){for(uint32_t r=0;r<count;++r)pack.SetValue(8*r,d.GetValue(EigenIndex(r)));Barrier();DataCopyPad(values[uint64_t(b)*n+lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});Barrier();}
        auto offsets=piv.template ReinterpretCast<uint32_t>();
        if(upper){for(uint32_t i=0;i<n;++i)offsets.SetValue(i,(n-1-i)*4);Barrier();}
        for(uint32_t r=0;r<count;++r){
            auto q=mat[r*Stride()];
            if(upper){Duplicate(tmp,0.0f,Stride());PipeBarrier<PIPE_V>();Gather(tmp,q,offsets,0,n);PipeBarrier<PIPE_V>();Brcb(pack,tmp,uint8_t((n+7)/8),{1,8});}
            else Brcb(pack,q,uint8_t((n+7)/8),{1,8});Barrier();
            DataCopyPad(vectors[uint64_t(b)*n*n+EigenIndex(r)],pack,DataCopyExtParams{uint16_t(n),4,0,(n-1)*4,0});Barrier();
        }
    }
    __aicore__ inline void TransposeStagedRows(uint32_t outputRows, uint32_t pitch) {
        if(Stride()%16==0){
            uint64_t source[16],destination[16];
            TransDataTo5HDParams params{false,false,uint8_t(Stride()/16),2,uint16_t(pitch*2)};
            for(uint32_t col=0;col<pitch;col+=8){
                for(uint32_t i=0;i<16;++i)source[i]=uint64_t(prod[i*pitch+col].GetPhyAddr());
                for(uint32_t i=0;i<8;++i){
                    uint32_t logical=col+i,row=logical<outputRows?(upper?outputRows-1-logical:logical):logical;
                    destination[2*i]=uint64_t(mat[row*Stride()].GetPhyAddr());
                    destination[2*i+1]=uint64_t(mat[row*Stride()+8].GetPhyAddr());
                }
                TransDataTo5HD<float>(destination,source,params);PipeBarrier<PIPE_V>();
            }
        }else{
            auto offsets=piv.template ReinterpretCast<int32_t>();
            CreateVecIndex(offsets,int32_t(0),Stride());PipeBarrier<PIPE_V>();Muls(offsets,offsets,int32_t(pitch*4),Stride());Barrier();
            for(uint32_t r=0;r<outputRows;++r){
                auto row=mat[r*Stride()];Duplicate(row,0.0f,Stride());PipeBarrier<PIPE_V>();
                Gather(row,prod,offsets.template ReinterpretCast<uint32_t>(),(upper?outputRows-1-r:r)*4,n);
            }
        }
    }
    __aicore__ inline void OutputStaged(uint32_t b){
        if(active){
            if(count){for(uint32_t r=0;r<count;++r)pack.SetValue(8*r,d.GetValue(EigenIndex(r)));Barrier();DataCopyPad(values[uint64_t(b)*n+lane],pack,DataCopyExtParams{uint16_t(count),4,0,(team-1)*4,0});}
            auto stage=hist[(n+5)*Stride()];Barrier();
            for(uint32_t r=0;r<count;++r)DataCopy(stage[EigenIndex(r)*Stride()],mat[r*Stride()],Stride());
        }
        SyncAll();
        if(active){
            auto stage=hist[(n+5)*Stride()];
            uint32_t outputRows=base<n?(n-base<rows?n-base:rows):0;
            if(outputRows){
                uint32_t pitch=(outputRows+7)/8*8,first=upper?n-base-outputRows:base;
                Duplicate(prod,0.0f,Stride()*pitch);Barrier();
                DataCopyPad(prod,stage[first],DataCopyExtParams{uint16_t(n),outputRows*4,(Stride()-outputRows)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                Barrier();
                TransposeStagedRows(outputRows, pitch);
                Barrier();
                if(n==Stride())DataCopy(vectors[uint64_t(b)*n*n+base*n],mat,outputRows*Stride());
                else DataCopyPad(vectors[uint64_t(b)*n*n+base*n],mat,DataCopyExtParams{uint16_t(outputRows),n*4,0,0,0});
                Barrier();
            }
        }
    }
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR ev,GM_ADDR eq,GM_ADDR work,LinalgEighTilingData t){
        if constexpr(FixedStride==0){if(t.n>=3 && t.n<=25)pipe.InitBuffer(smallStorage,15*t.n*t.stride*4);}
        packedMatrix=false;recordCubeSigns=false;n=t.n;dynamicStride=t.stride;batch=t.batch;upper=t.upper;team=t.team;groups=t.groups;rows=t.rows;laneStride=(rows+7)/8*8;
        uint32_t physicalTeam=GetTaskRation()==2?(team+1)/2*2:team;lane=GetBlockIdx()%physicalTeam;group=GetBlockIdx()/physicalTeam;base=lane*rows;count=base<n?((n-base)<rows?(n-base):rows):0;
        input.SetGlobalBuffer((__gm__ float*)a);values.SetGlobalBuffer((__gm__ float*)ev);vectors.SetGlobalBuffer((__gm__ float*)eq);ws.SetGlobalBuffer((__gm__ float*)work);
        uint32_t wideN=n<=25?n:25;uint32_t wideSize=3*wideN*wideN+21*wideN+64;wideSize=wideSize>8*Stride()?wideSize:8*Stride();wideSize=wideSize>((Stride()+1)/2)*laneStride?wideSize:((Stride()+1)/2)*laneStride;privateCapacity=wideSize+((n>25 && Stride()%64==0)?64:0);pipe.InitBuffer(wideStorage,(wideSize+((n>25 && Stride()%64==0)?128:0))*4);wideBuf=wideStorage.Get<float>()[((n>25 && Stride()%64==0)?64:0)];
        uint32_t dense=rows*Stride()>Stride()*laneStride?rows*Stride():Stride()*laneStride;
        uint32_t color=(n>25 && Stride()%64==0)?1:0,alternate=Stride()%128==64?1:0;
        uint32_t storageSize=3*dense+((dense+1)/2+7)/8*8+22*Stride()+64+(128+128*alternate)*color;storageSize=n>25 && storageSize<8192?8192:storageSize;storageCapacity=storageSize;pipe.InitBuffer(storage,storageSize*4);auto b=storage.Get<float>();mat=b;prod=b[dense+64*color];pcrBand=b[2*dense+64*color];pcrLeft=b[3*dense+64*color];uint32_t off=3*dense+((dense+1)/2+7)/8*8+64*color;
        // Keep matrix/products, reflector data and partial sums at distinct
        // repeated UB offsets. The matrix and vector arithmetic is unchanged.
        v=b[off];tmp=b[off+Stride()+64*alternate*color];tmp2=b[off+2*Stride()+128*alternate*color];red=b[off+3*Stride()+(64+64*alternate)*color];param=b[off+4*Stride()+(64+64*alternate)*color];
        d=b[off+5*Stride()+(64+64*alternate)*color];e=b[off+6*Stride()+(64+64*alternate)*color];origD=b[off+7*Stride()+(64+64*alternate)*color];origE=b[off+8*Stride()+(64+64*alternate)*color];beta=b[off+9*Stride()+(64+64*alternate)*color];
        piv=b[off+10*Stride()+(64+64*alternate)*color];fac=b[off+11*Stride()+(64+64*alternate)*color];w=b[off+12*Stride()+(64+128*alternate)*color];pack=b[off+13*Stride()+(64+128*alternate)*color];
    }
    __aicore__ inline void ReduceByShape(uint32_t b){
        if constexpr(FixedStride==0){
            if(n>=3 && n<=25){ReduceReplicatedSmall(b);return;}
            if(n<=25){ReduceExtended();return;}
        }
        if(active)PackMatrix();Reduce();if(active)UnpackMatrix();
    }
    __aicore__ inline void LoadAlignedMatrixBlock(uint32_t b) {
        uint32_t first=upper?n-base-count:base;
        DataCopy(prod,input[uint64_t(b)*n*n+uint64_t(first)*n],count*Stride());
        DataCopyPad(pcrBand,input[uint64_t(b)*n*n+first],DataCopyExtParams{uint16_t(n),count*4,(n-count)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        Barrier();
        auto columnOffsets=piv.template ReinterpretCast<uint32_t>(),rowOffsets=tmp2.template ReinterpretCast<uint32_t>();

        uint64_t source[16],destination[16];
        TransDataTo5HDParams params{false,false,uint8_t(n/16),2,uint16_t(count*2)};
        for(uint32_t col=0;col<count;col+=8){
            for(uint32_t i=0;i<16;++i)source[i]=uint64_t(pcrBand[i*count+col].GetPhyAddr());
            for(uint32_t i=0;i<8;++i){
                uint32_t row=upper?count-1-col-i:col+i;
                destination[2*i]=uint64_t(mat[row*Stride()].GetPhyAddr());
                destination[2*i+1]=uint64_t(mat[row*Stride()+8].GetPhyAddr());
            }
            TransDataTo5HD<float>(destination,source,params);PipeBarrier<PIPE_V>();
        }
        for(uint32_t r=0;r<count;++r){
            auto a=mat[r*Stride()];uint32_t i=base+r,physical=upper?count-1-r:r;
            if(upper)Gather(tmp,a,rowOffsets,0,n);
            Gather(w,prod[physical*Stride()],rowOffsets,0,n);PipeBarrier<PIPE_V>();
            auto selected=beta.template ReinterpretCast<uint8_t>();Compares(selected,red,float(int32_t(i)),CMPMODE::LE,n);PipeBarrier<PIPE_V>();
            if(upper)Select(a,selected,w,tmp,SELMODE::VSEL_TENSOR_TENSOR_MODE,n);
            else Select(a,selected,w,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,n);
            Barrier();
        }
    }
    __aicore__ inline void LoadMatrixRows(uint32_t b) {
        for(uint32_t r=0;r<count;++r){auto a=mat[r*Stride()];Duplicate(a,0.0f,Stride());Barrier();uint32_t i=base+r;
            if(n>25){
                uint32_t x=upper?n-1-i:i;
                DataCopyPad(tmp,input[uint64_t(b)*n*n+x*n],DataCopyExtParams{1,n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
                DataCopyPad(pack,input[uint64_t(b)*n*n+x],DataCopyExtParams{uint16_t(n),4,(n-1)*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});Barrier();
                if(n%64==0){
                    Gather(a,pack,piv.template ReinterpretCast<uint32_t>(),0,n);Gather(w,tmp,tmp2.template ReinterpretCast<uint32_t>(),0,n);PipeBarrier<PIPE_V>();
                    auto selected=beta.template ReinterpretCast<uint8_t>();Compares(selected,red,float(int32_t(i)),CMPMODE::LE,n);PipeBarrier<PIPE_V>();
                    Select(a,selected,w,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,n);Barrier();
                }else for(uint32_t j=0;j<n;++j){uint32_t y=upper?n-1-j:j;a.SetValue(j,j<=i?tmp.GetValue(y):pack.GetValue(8*y));}
            }else for(uint32_t j=0;j<n;++j){uint32_t x=upper?n-1-i:i,y=upper?n-1-j:j;if((upper && x>y)||(!upper && x<y)){uint32_t z=x;x=y;y=z;}a.SetValue(j,input.GetValue(uint64_t(b)*n*n+x*n+y));}
        }
    }
    __aicore__ __attribute__((noinline)) void Process(){
        uint32_t rounds=(batch+groups-1)/groups;
        for(uint32_t step=0;step<rounds;++step){
            uint32_t b=step*groups+group;active=b<batch && lane<team;count=base<n?((n-base)<rows?(n-base):rows):0;directCoefficients=(base%8==0 && count%8==0 && count>0);
            if(active){
                hist=ws[uint64_t(b)*WorkspacePerMatrix()];gw=hist[(n<=25?2*n:n)*Stride()];gparam=gw[(n<=25?2:4)*Stride()];
                Duplicate(d,0.0f,Stride());Duplicate(e,0.0f,Stride());Duplicate(beta,0.0f,Stride());Barrier();
                if(n>25 && n%64==0){
                    auto columnOffsets=piv.template ReinterpretCast<uint32_t>(),rowOffsets=tmp2.template ReinterpretCast<uint32_t>();
                    CreateVecIndex(rowOffsets.template ReinterpretCast<int32_t>(),int32_t(0),Stride());PipeBarrier<PIPE_V>();
                    Cast(red,rowOffsets.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,Stride());PipeBarrier<PIPE_V>();
                    if(upper){Muls(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(-1),Stride());PipeBarrier<PIPE_V>();Adds(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(n-1),Stride());PipeBarrier<PIPE_V>();}
                    Muls(rowOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(4),Stride());PipeBarrier<PIPE_V>();
                    Muls(columnOffsets.template ReinterpretCast<int32_t>(),rowOffsets.template ReinterpretCast<int32_t>(),int32_t(count && count%8==0?count:8),Stride());Barrier();
                }
                if(n<3 || n>25){
                // Fetch a block of rows and its transposed counterpart using
                // two DMA commands when the owned columns are block aligned.
                if(n>25 && n%64==0 && count && count%8==0){
                    LoadAlignedMatrixBlock(b);
                }else{
                LoadMatrixRows(b);
                }
                }
                Barrier();
            }
            ReduceByShape(b);if(n>25 && BackPanelWidth()){if constexpr(CubeMode==2){if(b<batch){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);}SyncAll();}if(active)PrepareBackPanels();SyncAll();if constexpr(CubeMode==2){if(b<batch)CrossCoreSetFlag<2,PIPE_MTE3>(8);}}count=lane<n?(n-1-lane)/team+1:0;if(active){Bisect();Barrier();DataCopy(param,d,Stride());Barrier();Inverse();}else{if(n>25 && n>=2*team){SyncAll();SyncAll();}if(b<batch && lane>=team){if(Stride()>=64 && batch%groups==0)SyncAll();if(BackPanelWidth())CubeBackHandshake(false);}}if(n>25)OutputStaged(b);else if(active)Output(b);SyncAll();if(BackPanelWidth() && b<batch)CrossCoreSetFlag<2,PIPE_MTE3>(6);
        }
    }
};

template <typename DT_A,uint32_t USE_CUBE>
__global__ __aicore__ void linalg_eigh(GM_ADDR A,GM_ADDR eigenvalues,GM_ADDR eigenvectors,GM_ADDR workspace,GM_ADDR tiling){
    REGISTER_TILING_DEFAULT(LinalgEighTilingData);GET_TILING_DATA_WITH_STRUCT(LinalgEighTilingData,data,tiling);
    if ASCEND_IS_AIC {if constexpr(USE_CUBE==1)CubeBackProcess16(GetUserWorkspace(workspace),data);else if constexpr(USE_CUBE==2)CubeBackProcess(GetUserWorkspace(workspace),data);return;}
    if ASCEND_IS_AIV {
    // Power-of-two SIMD row pitches allow the compiler to fold repeat
    // strides, address arithmetic and bounded vector loops. Other pitches
    // use the identical general algorithm.
    switch(data.stride){
        case 64:{KernelLinalgEigh<DT_A,64,USE_CUBE>op;op.Init(A,eigenvalues,eigenvectors,GetUserWorkspace(workspace),data);op.Process();break;}
        case 128:{KernelLinalgEigh<DT_A,128,USE_CUBE>op;op.Init(A,eigenvalues,eigenvectors,GetUserWorkspace(workspace),data);op.Process();break;}
        case 256:{KernelLinalgEigh<DT_A,256,USE_CUBE>op;op.Init(A,eigenvalues,eigenvectors,GetUserWorkspace(workspace),data);op.Process();break;}
        case 512:{KernelLinalgEigh<DT_A,512,USE_CUBE>op;op.Init(A,eigenvalues,eigenvectors,GetUserWorkspace(workspace),data);op.Process();break;}
        default:{KernelLinalgEigh<DT_A,0,USE_CUBE>op;op.Init(A,eigenvalues,eigenvectors,GetUserWorkspace(workspace),data);op.Process();break;}
    }
}
}
