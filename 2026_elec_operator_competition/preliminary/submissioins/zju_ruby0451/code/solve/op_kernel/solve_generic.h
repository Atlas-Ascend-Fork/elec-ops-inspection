/*
 * Copyright (c) 2026 ruby0451 contributors.
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

#ifndef RUBY0451_SOLVE_SOLVE_GENERIC_H
#define RUBY0451_SOLVE_SOLVE_GENERIC_H
#include "solve_common.h"
template<class DT_A, uint32_t N_VALUE> class KernelLinalgSolve {
public:

 __aicore__ inline void BackSubstituteRows(uint32_t pass,uint32_t batch) {
            for(int32_t p=int32_t(n)-1;p>=0;--p) {
                ReadRow(pivot,p);float inv=1.0f/pivot.GetValue(p);
                Muls(pivot[na],pivot[na],inv,nk);PipeBarrier<PIPE_ALL>();
                if(refine) {
                    if(pass==0) Adds(sol[p*nk],pivot[na],0.0f,nk);
                    else Add(sol[p*nk],sol[p*nk],pivot[na],nk);
                    PipeBarrier<PIPE_V>();
                }
                if constexpr(sizeof(DT_A)==4) {
                    if(refine && pass==1) Adds(io.template ReinterpretCast<float>(),sol[p*nk],0.0f,k);
                    else Adds(io.template ReinterpretCast<float>(),pivot[na],0.0f,k);
                } else Cast(io,pivot[na],RoundMode::CAST_RINT,k);
                Fence<HardEvent::V_MTE3>();
                DataCopyExtParams cp{1,k*uint32_t(sizeof(DT_A)),0,0,0};
                DataCopyPad(x[uint64_t(batch)*n*k+uint64_t(p)*k],io,cp);Fence<HardEvent::MTE3_V>();
                if(grouped) BackGrouped(p);
                else for(uint32_t i=0;i<uint32_t(p);++i) {
                    ReadRow(row,i);float factor=-row.GetValue(p);
                    Axpy(row[na],pivot[na],factor,nk);PipeBarrier<PIPE_ALL>();
                    WriteRow(row,i);
                }
            }

}


 __aicore__ inline void FactorRows() {
            for(uint32_t p=0;p<n;++p) {
                uint32_t best=p;float maxabs=0.0f;
                for(uint32_t i=p;i<n;++i) {
                    float v=column.GetValue(i),av=v<0?-v:v;
                    if(av>maxabs){maxabs=av;best=i;}
                }
                ReadRow(pivot,best);
                if(best!=p){ReadRow(row,p);WriteRow(row,best);}
                WriteRow(pivot,p);
                float inv=1.0f/pivot.GetValue(p);
                if(grouped) ForwardGrouped(p,inv);
                else for(uint32_t i=p+1;i<n;++i) {
                    ReadRow(row,i);
                    float factor=-row.GetValue(p)*inv;
                    Axpy(row,pivot,factor,width);Fence<HardEvent::V_S>();
                    if(p+1<n) column.SetValue(i,row.GetValue(p+1));
                    WriteRow(row,i);
                }
            }

}

    template<HardEvent event> __aicore__ inline void Fence() {
        const event_t id=static_cast<event_t>(pipe.FetchEventID(event));
        SetFlag<event>(id);WaitFlag<event>(id);
    }
    __aicore__ inline void Init(GM_ADDR A,GM_ADDR B,GM_ADDR X,GM_ADDR work,const LinalgSolveTilingData& t) {
        n=N_VALUE?N_VALUE:t.n;k=t.k;batches=t.batches;broadcast=t.broadcastB;cores=t.cores;resident=t.resident;grouped=t.grouped;
        na=(n+7)/8*8;nk=(k+7)/8*8;width=na+nk;
        refine=sizeof(DT_A)==4 && n>64 && k<=8;
        a.SetGlobalBuffer((__gm__ DT_A*)A);b.SetGlobalBuffer((__gm__ DT_A*)B);x.SetGlobalBuffer((__gm__ DT_A*)X);
        if(!resident) w.SetGlobalBuffer((__gm__ float*)work+uint64_t(GetBlockIdx())*n*width);
        pipe.InitBuffer(matrixBuf,resident?n*width*4:(grouped?32*width*4:32));
        pipe.InitBuffer(pivotBuf,width*4);pipe.InitBuffer(rowBuf,width*4);pipe.InitBuffer(tempBuf,width*4);
        pipe.InitBuffer(colBuf,(n+7)/8*32);
        pipe.InitBuffer(ioBuf,(((n>k?n:k)+15)/16*16+16)*sizeof(DT_A));
        mat=matrixBuf.Get<float>();pivot=pivotBuf.Get<float>();row=rowBuf.Get<float>();
        temp=tempBuf.Get<float>();column=colBuf.Get<float>();io=ioBuf.Get<DT_A>();
        if(refine) {
            pipe.InitBuffer(solBuf,n*nk*4);pipe.InitBuffer(rhsBuf,n*nk*4);pipe.InitBuffer(xtBuf,k*na*4);
            sol=solBuf.Get<float>();rhs=rhsBuf.Get<float>();xt=xtBuf.Get<float>();
        }
    }
    __aicore__ inline void ReadInput(LocalTensor<float> dst,GlobalTensor<DT_A> src,uint32_t count) {
        DataCopyExtParams cp{1,count*uint32_t(sizeof(DT_A)),0,0,0};
        DataCopyPadExtParams<DT_A> pad{false,0,0,0};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(io,src,cp,pad);Fence<HardEvent::MTE2_V>();
        if constexpr(sizeof(DT_A)==4) Adds(dst,io.template ReinterpretCast<float>(),0.0f,count);
        else Cast(dst,io,RoundMode::CAST_NONE,count);
        Fence<HardEvent::V_S>();
    }
    __aicore__ inline void ReadRow(LocalTensor<float> dst,uint32_t i) {
        Fence<HardEvent::S_V>();
        if(resident) {
            Adds(dst,mat[i*width],0.0f,width);Fence<HardEvent::V_S>();
        } else {
            Fence<HardEvent::V_MTE2>();DataCopy(dst,w[uint64_t(i)*width],width);
            Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        }
    }
    __aicore__ inline void WriteRow(LocalTensor<float> src,uint32_t i) {
        Fence<HardEvent::S_V>();
        if(resident) {
            Adds(mat[i*width],src,0.0f,width);PipeBarrier<PIPE_V>();
        } else {
            Fence<HardEvent::V_MTE3>();DataCopy(w[uint64_t(i)*width],src,width);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        }
    }
    __aicore__ inline void Process() {
        for(uint32_t batch=GetBlockIdx();batch<batches;batch+=cores) {
            uint64_t aoff=uint64_t(batch)*n*n,boff=broadcast?0:uint64_t(batch)*n*k;
            for(uint32_t pass=0;pass<(refine?2u:1u);++pass) {
            for(uint32_t i=0;i<n;++i) {
                Duplicate(row,0.0f,width);PipeBarrier<PIPE_ALL>();
                ReadInput(row,a[aoff+uint64_t(i)*n],n);
                if(pass==0) ReadInput(row[na],b[boff+uint64_t(i)*k],k);
                else {Adds(row[na],rhs[i*nk],0.0f,nk);Fence<HardEvent::V_S>();}
                column.SetValue(i,row.GetValue(0));WriteRow(row,i);
            }
    FactorRows();
    BackSubstituteRows(pass,batch);
            if(refine && pass==0) ComputeResidual(aoff,boff);
            }
        }
    }
private:
    __aicore__ inline LocalTensor<float> ReadTile(uint32_t base,uint32_t rows) {
        if(resident) return mat[base*width];
        Fence<HardEvent::V_MTE2>();
        DataCopy(mat,w[uint64_t(base)*width],rows*width);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
        return mat;
    }
    __aicore__ inline void WriteTile(uint32_t base,uint32_t rows) {
        if(!resident) {
            Fence<HardEvent::V_MTE3>();
            DataCopy(w[uint64_t(base)*width],mat,rows*width);
            Fence<HardEvent::MTE3_V>();Fence<HardEvent::MTE3_MTE2>();
        }
    }
    __aicore__ inline void ForwardGrouped(uint32_t p,float inv) {
        for(uint32_t base=p+1;base<n;base+=32) {
            uint32_t rows=n-base<32?n-base:32;
            auto tile=ReadTile(base,rows);
            for(uint32_t j=0;j<rows;++j) {
                auto current=tile[j*width];
                float factor=-current.GetValue(p)*inv;
                Axpy(current,pivot,factor,width);Fence<HardEvent::V_S>();
                if(p+1<n) column.SetValue(base+j,current.GetValue(p+1));
            }
            WriteTile(base,rows);
        }
    }
    __aicore__ inline void BackGrouped(uint32_t p) {
        for(uint32_t base=0;base<p;base+=32) {
            uint32_t rows=p-base<32?p-base:32;
            auto tile=ReadTile(base,rows);
            for(uint32_t j=0;j<rows;++j) {
                auto current=tile[j*width];
                float factor=-current.GetValue(p);
                Axpy(current[na],pivot[na],factor,nk);Fence<HardEvent::V_S>();
            }
            WriteTile(base,rows);
        }
    }
    union FloatBits {float number;uint32_t bits;};
    __aicore__ inline float HighPart(float value) {
        FloatBits v;v.number=value;v.bits &= 0xfffff000u;return v.number;
    }
    __aicore__ inline void AddResidualTerm(float term,float& sum,float& comp) {
        // Error-free TwoSum plus accumulated low parts; volatile forbids reassociation.
        volatile float updated=sum+term;
        volatile float displaced=updated-sum;
        volatile float kept=updated-displaced;
        volatile float lowSum=sum-kept;
        volatile float lowTerm=term-displaced;
        volatile float error=lowSum+lowTerm;
        comp=comp+error;sum=updated;
    }
    __aicore__ inline void ComputeResidual(uint64_t aoff,uint64_t boff) {
        Fence<HardEvent::V_S>();
        for(uint32_t q=0;q<k;++q)
            for(uint32_t j=0;j<n;++j) xt.SetValue(q*na+j,sol.GetValue(j*nk+q));
        Fence<HardEvent::S_V>();
        Duplicate(rhs,0.0f,n*nk);Fence<HardEvent::V_S>();
        // Form B-A X in original row order; second solve refactors A from scratch.
        for(uint32_t i=0;i<n;++i) {
            uint32_t original=i;
            ReadInput(row,a[aoff+uint64_t(original)*n],n);
            ReadInput(row[na],b[boff+uint64_t(original)*k],k);
            for(uint32_t q=0;q<k;++q) {
                Fence<HardEvent::S_V>();
                Mul(temp,row,xt[q*na],n);Fence<HardEvent::V_S>();
                float sum=row.GetValue(na+q),comp=0.0f;
                for(uint32_t j=0;j<n;++j) {
                    float av=row.GetValue(j),xv=xt.GetValue(q*na+j);
                    float ah=HighPart(av),xh=HighPart(xv);
                    volatile float al=av-ah,xl=xv-xh;
                    float product=temp.GetValue(j);
                    volatile float hh=ah*xh;
                    volatile float first=hh-product;
                    volatile float hl=ah*xl;
                    volatile float second=first+hl;
                    volatile float lh=al*xh;
                    volatile float third=second+lh;
                    volatile float ll=al*xl;
                    volatile float productError=third+ll;
                    AddResidualTerm(-product,sum,comp);
                    AddResidualTerm(-productError,sum,comp);
                }
                rhs.SetValue(i*nk+q,sum+comp);
            }
        }
        Fence<HardEvent::S_V>();
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf,pivotBuf,rowBuf,tempBuf,colBuf,ioBuf;
    TBuf<TPosition::VECCALC> solBuf,rhsBuf,xtBuf;
    LocalTensor<float> sol,rhs,xt;
    bool refine;
    GlobalTensor<DT_A> a,b,x;GlobalTensor<float> w;
    LocalTensor<float> mat,pivot,row,temp,column;LocalTensor<DT_A> io;
    uint32_t n,k,batches,broadcast,cores,resident,grouped,na,nk,width;
};

#endif
