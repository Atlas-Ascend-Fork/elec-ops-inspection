/*
 * Copyright (c) 2026 FalseShare contributors.
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
#include "linalg_pinv_tiling.h"

// A 16-row capacity for all square inputs that fit: checked seed plus one correction.
// A rejected batch is handled by the existing serial pinv implementation.
class ResidentTinyInverse {
    static constexpr uint32_t K=16,DP=16,L=32;

    AscendC::LocalTensor<float> mat,original,xt,residual,products,corrected,weights,pivot,col,scalar,reduce,partial,repeated;
    AscendC::LocalTensor<uint32_t> indices;
    AscendC::GlobalTensor<float> input,output;
    LinalgPinvTilingData t;
    uint32_t size;
    GM_ADDR staticTiling=nullptr;
    __aicore__ inline void LoadTinyInput(uint64_t base) {
        AscendC::GlobalTensor<uint32_t> payload;
        payload.SetGlobalBuffer((__gm__ uint32_t*)(staticTiling+__builtin_offsetof(LinalgPinvTinyTilingData,packed)));
        if(size==16)AscendC::DataCopy(original,input[base],K*DP);
        else AscendC::DataCopyPad(products,input[base],AscendC::DataCopyExtParams{1,size*size*4,0,0,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)((8-size*size%8)%8),0.0f});
        AscendC::DataCopy(mat.ReinterpretCast<uint32_t>(),payload,K*L);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(indices.ReinterpretCast<int32_t>(),mat[8].ReinterpretCast<half>(),AscendC::RoundMode::CAST_RINT,(uint64_t)DP,(uint8_t)K,AscendC::UnaryRepeatParams{1,1,DP/8,L/8});
        if(size<16) {
            products.SetValue(size*size,0.0f);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(partial.ReinterpretCast<int32_t>(),mat.ReinterpretCast<half>(),AscendC::RoundMode::CAST_RINT,(uint64_t)DP,(uint8_t)K,AscendC::UnaryRepeatParams{1,1,DP/8,L/8});AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(original,products,partial.ReinterpretCast<uint32_t>(),0,K*DP);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(original[size*DP],original[size*DP],mat[size*L+DP],(uint64_t)DP,(uint8_t)(K-size),AscendC::BinaryRepeatParams{1,1,1,DP/8,DP/8,L/8});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void BuildTinySeed() {
        {
            constexpr uint32_t CP=16, LIVE=32;
            auto cm=products,unit=cm[LIVE*CP],cw=unit[K*DP],cd=cw[LIVE*8],pc=cd[64],pr=pc[DP];
            auto pm=partial.ReinterpretCast<int32_t>();
            // Original row-major A is the column-major representation of A^T.
            AscendC::DataCopy(cm,original,AscendC::DataCopyParams{K,DP/8,0,(CP-DP)/8});
            AscendC::DataCopy(unit,mat[DP],AscendC::DataCopyParams{K,DP/8,(L-DP)/8,0});
            // Static-arena Tiny has no AscendC::TPipe; derive indices from the payload.
            AscendC::ShiftRight(pm,indices.ReinterpretCast<int32_t>(),int32_t(CP==16?1:2),K);AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(cm[K*CP],unit,AscendC::DataCopyParams{K,DP/8,0,(CP-DP)/8});
            if constexpr(CP==24){AscendC::Muls(pm,pm,int32_t(3),K);AscendC::PipeBarrier<PIPE_V>();}
            AscendC::Adds(pm[K],pm,int32_t(K*CP*4),K);AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<size;++j) {
                AscendC::Brcb(cd,cm[j*CP+j/8*8],1,AscendC::BrcbRepeatParams{1,8});
                AscendC::Gather(pr,cm[j],pm.ReinterpretCast<uint32_t>(),0,(K+j+8)/8*8);
                AscendC::Sub(pc,unit[j*DP],cm[j*CP],DP);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(pc,pc,cd[j%8*8],(uint64_t)DP,1,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
                AscendC::Brcb(cw,pr,(K+j+8)/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                AscendC::MulAddDst(cm[(j+1)*CP],pc,cw[(j+1)*8],(uint64_t)DP,(uint8_t)K,AscendC::BinaryRepeatParams{1,1,0,CP/8,0,1});AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::DataCopy(mat[DP],cm[K*CP],AscendC::DataCopyParams{K,DP/8,(CP-DP)/8,(L-DP)/8});AscendC::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void FormTinyResidual() {
        AscendC::PipeBarrier<PIPE_V>();AscendC::Gather(xt,mat[DP],indices,0,K*DP);AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams broadcast{1,1,1,8,8,0};
        // Form (I-A*X)^T directly so the correction can consume it.
        for(uint32_t r=0;r<4;++r)AscendC::DataCopy(repeated[r*DP],xt,AscendC::DataCopyParams{K,DP/8,0,6});
        AscendC::PipeBarrier<PIPE_V>();
         {
            for(uint32_t group=0;group<4;++group)
                AscendC::Mul(products[group*64],original[group*64],repeated,(uint64_t)64,(uint8_t)K,AscendC::BinaryRepeatParams{1,1,1,DP*DP/8,0,8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum(partial,products,64,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum(residual,partial,8,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(residual,residual,-1.0f,K*DP);AscendC::PipeBarrier<PIPE_V>();
         {
            for(uint32_t g=0;g<4;++g) {
                uint64_t mask[2]={uint64_t(0x0008000400020001ULL)<<(g*4),0};
                AscendC::Adds(residual[g*64],residual[g*64],1.0f,mask,1,AscendC::UnaryRepeatParams{1,1,8,8});
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool CheckTinyResidual(bool &refine) {
         {
            // Batch three Frobenius statistics into two vector reductions.
            AscendC::Mul(products,original,original,K*DP);
            AscendC::Mul(products[K*DP],mat[DP],mat[DP],(uint64_t)K,(uint8_t)K,AscendC::BinaryRepeatParams{1,1,1,DP/8,L/8,L/8});
            AscendC::Mul(products[2*K*DP],residual,residual,K*DP);AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum(partial,products,12,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(scalar,partial,32,3,8,1,4);
            // R is transposed: the same block sums give rows of X and columns of R.
            AscendC::PairReduceSum(reduce,partial[32],1,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceMax(scalar[24],reduce,(uint64_t)K,2,8,1,K/8,AscendC::ReduceOrder::ORDER_ONLY_VALUE);AscendC::PipeBarrier<PIPE_ALL>();
            float af2=scalar.GetValue(0),xf2=scalar.GetValue(8);float rf2=scalar.GetValue(16);
            float cutoff2=t.rtol*t.rtol*af2;if(t.atol*t.atol>cutoff2)cutoff2=t.atol*t.atol;
            if(!(cutoff2*xf2<0.49f && xf2<=1e6f && rf2<0.0625f && xf2*rf2*rf2<6.4e-9f))return false;
            // Entry error <= maxRowNorm(X)*maxColumnNorm(R)/(1-rho).
            // Use a 9e-5 absolute budget; larger estimates keep the correction.
            float rowX2=scalar.GetValue(24),colR2=scalar.GetValue(32);
            float limit=rf2<1e-4f?7.93881e-9f:4.55625e-9f;
            refine=!(rowX2>=0.0f && colR2>=0.0f && rowX2*colR2<limit);
        }
        return true;
    }
    __aicore__ inline void CorrectTinyInverse(bool refine) {
        if(refine) {
        // Residual rows already represent the columns used by X*R.
        for(uint32_t r=0;r<4;++r)AscendC::DataCopy(repeated[r*DP],mat[DP],AscendC::DataCopyParams{K,DP/8,(L-DP)/8,6});
        AscendC::PipeBarrier<PIPE_V>();
         {
            for(uint32_t group=0;group<4;++group)
                AscendC::Mul(products[group*64],residual[group*64],repeated,(uint64_t)64,(uint8_t)K,AscendC::BinaryRepeatParams{1,1,1,DP*DP/8,0,8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum(partial,products,64,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum(corrected,partial,8,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(corrected,corrected,mat[DP],(uint64_t)K,(uint8_t)K,AscendC::BinaryRepeatParams{1,1,1,DP/8,DP/8,L/8});AscendC::PipeBarrier<PIPE_ALL>();

        } else {
            AscendC::DataCopy(corrected,mat[DP],AscendC::DataCopyParams{K,DP/8,(L-DP)/8,0});AscendC::PipeBarrier<PIPE_ALL>();
        }

    }
    __aicore__ inline bool StoreTinyInverse(uint64_t base) {
        if(size==16)AscendC::DataCopy(output[base],corrected,K*DP);
        else {
            // Compact each capacity row to the logical width, then store n*n values.
            AscendC::Duplicate(indices,uint32_t(((1u<<size)-1)*0x00010001u),8);AscendC::PipeBarrier<PIPE_ALL>();
            uint64_t kept=0;
            AscendC::GatherMask(products,corrected,indices,true,K*DP,AscendC::GatherMaskParams{1,1,8,0},kept);AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(output[base],products,AscendC::DataCopyExtParams{1,size*size*4,0,0,0});
        }
        AscendC::PipeBarrier<PIPE_ALL>();return true;
    }
    __aicore__ inline bool Batch(uint32_t batch) {
        uint64_t base=(uint64_t)batch*size*size;
        LoadTinyInput(base);
        BuildTinySeed();
        FormTinyResidual();
        bool refine=true;
        if(!CheckTinyResidual(refine))return false;
        CorrectTinyInverse(refine);
        return StoreTinyInverse(base);
    }
    template<bool TransposeLeft=false,bool TransposeRight=false>
    __aicore__ inline void DenseProduct(AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> left,AscendC::LocalTensor<float> right) {
        if constexpr(TransposeRight)AscendC::DataCopy(xt,right,K*DP);
        else AscendC::Gather(xt,right,indices,0,K*DP);
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr(TransposeLeft) {AscendC::Gather(dst,left,indices,0,K*DP);AscendC::PipeBarrier<PIPE_ALL>();
            for(uint32_t r=0;r<4;++r)AscendC::DataCopy(repeated[r*DP],dst,AscendC::DataCopyParams{K,DP/8,0,6});
        } else for(uint32_t r=0;r<4;++r)AscendC::DataCopy(repeated[r*DP],left,AscendC::DataCopyParams{K,DP/8,0,6});
        AscendC::PipeBarrier<PIPE_V>();AscendC::BinaryRepeatParams broadcast{1,1,1,8,8,0};
        for(uint32_t i=0;i<K;++i)AscendC::Mul(products[i*DP*DP],xt,repeated[i*64],(uint64_t)64,(uint8_t)4,broadcast);
        AscendC::PipeBarrier<PIPE_V>();AscendC::BlockReduceSum(partial,products,64,64,1,1,8);AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum(dst,partial,8,64,1,1,8);AscendC::PipeBarrier<PIPE_ALL>();
    }
    // Repair drift outside the retained row/column spaces, then correct
    // the retained block. X*=A_tau^+ is a fixed point of both operations.
    __aicore__ inline bool Polish(uint32_t batch) {
        uint64_t base=(uint64_t)batch*size*size;
        AscendC::Duplicate(mat,0.0f,K*L);AscendC::Duplicate(original,0.0f,K*DP);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(original,input[base],AscendC::DataCopyExtParams{(uint16_t)size,size*4,0,0,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)(DP-size),0.0f});
        AscendC::DataCopyPad(mat,output[base],AscendC::DataCopyExtParams{(uint16_t)size,size*4,0,0,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)(DP-size),0.0f});AscendC::PipeBarrier<PIPE_ALL>();
        if(size<16)for(uint32_t j=size;j<K;++j){original.SetValue(j*DP+j,1.0f);mat.SetValue(j*DP+j,1.0f);}
        for(uint32_t r=0;r<K;++r)indices.SetValue(r,r*DP*4);AscendC::PipeBarrier<PIPE_ALL>();
        auto ix=indices.ReinterpretCast<int32_t>();for(uint32_t j=1;j<K;++j)AscendC::Adds(ix[j*DP],ix,int32_t(j*4),DP);AscendC::PipeBarrier<PIPE_ALL>();
        DenseProduct(residual,mat,original);                 // X A
        DenseProduct(corrected,original,mat);                // A X
        float retained=0;for(uint32_t r=0;r<K;++r)retained+=corrected.GetValue(r*DP+r);
        if(retained>K-0.5f)return true; // Preserve the existing full-rank fallback result.
        DenseProduct<true>(mat[K*DP],residual,mat);           // (X A)^T X
        DenseProduct<false,true>(mat,mat[K*DP],corrected);    // (X A)^T X (A X)^T
        DenseProduct(residual,original,mat);                 // A X_new
        AscendC::Muls(residual,residual,-1.0f,K*DP);AscendC::PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<K;++r)residual.SetValue(r*DP+r,residual.GetValue(r*DP+r)+1.0f);
        AscendC::PipeBarrier<PIPE_ALL>();DenseProduct(corrected,mat,residual);
        AscendC::Add(mat,mat,corrected,K*DP);AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(output[base],mat,AscendC::DataCopyExtParams{(uint16_t)size,size*4,(DP-(size+7)/8*8)/8,0,0});AscendC::PipeBarrier<PIPE_ALL>();return true;
    }

    template<bool PolishOutput> __aicore__ inline bool RunArena(GM_ADDR a,GM_ADDR y,const LinalgPinvTilingData &tiling,AscendC::LocalTensor<float> arena,GM_ADDR tables=nullptr) {
        t=tiling;size=t.n;staticTiling=tables;
        mat=arena;original=arena[512];xt=arena[768];residual=arena[1024];products=arena[1280];
        corrected=arena[5376];weights=arena[5632];pivot=arena[5760];col=arena[5792];
        indices=arena[5808].ReinterpretCast<uint32_t>();scalar=arena[6064];reduce=arena[6128];partial=arena[6384];repeated=arena[6896];
        input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)y);
        bool ok=true;
        if constexpr(PolishOutput){for(uint32_t b=AscendC::GetBlockIdx();b<t.batches;b+=AscendC::GetBlockNum())Polish(b);}
        else {for(uint32_t b=AscendC::GetBlockIdx();b<t.batches;b+=AscendC::GetBlockNum())if(!Batch(b)){ok=false;break;}}
        AscendC::PipeBarrier<PIPE_ALL>();return ok;
    }
public:
    template<bool PolishOutput=false> __aicore__ inline bool Run(GM_ADDR a,GM_ADDR y,const LinalgPinvTilingData &tiling,GM_ADDR tables=nullptr) {
        if constexpr(!PolishOutput) {
            // This fast path has one fixed UB arena and no queue/event allocation.
            AscendC::LocalTensor<float> arena(AscendC::TPosition::VECCALC,0,7920);
            return RunArena<PolishOutput>(a,y,tiling,arena,tables);
        } else {
            AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,7920*4);
            bool ok=RunArena<PolishOutput>(a,y,tiling,buffer.Get<float>());pipe.Destroy();return ok;
        }
    }
};
