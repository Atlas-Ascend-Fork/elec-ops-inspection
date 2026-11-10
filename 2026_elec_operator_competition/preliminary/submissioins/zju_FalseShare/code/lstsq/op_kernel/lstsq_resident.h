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

#include "lstsq_vector.h"

template<uint32_t P,bool Parallel=false,bool InverseRows=false,bool InitialOwner=false,bool InitialOnly=false> class ResidentElimination {
    static constexpr uint32_t L=P+8,W=P+8,R=P;
    static_assert(!InverseRows || (Parallel && P==64));
    static_assert(!InitialOwner || InverseRows);
    static_assert(!InitialOnly || (P==64 && !Parallel && !InverseRows));
    __aicore__ inline void Index(AscendC::LocalTensor<int32_t> dst) {
        for(int32_t i=0;i<8;++i)dst.SetValue(i,i);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::UnaryRepeatParams params;
        for(uint32_t i=0;i<P/8-1;++i){AscendC::Adds(dst[(i+1)*8],dst[i*8],(int32_t)8,(uint64_t)8,1,params);AscendC::PipeBarrier<PIPE_V>();}
    }
    __aicore__ inline void ApplyInverseRows(AscendC::LocalTensor<float> result,AscendC::LocalTensor<float> inverse,
        AscendC::LocalTensor<float> value,AscendC::LocalTensor<float> products,uint32_t width) {
        AscendC::Mul(products,inverse,value,(uint64_t)width,(uint8_t)8,AscendC::BinaryRepeatParams{1,1,1,P/8,P/8,0});AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(result,products,(uint64_t)width,8,1,1,P/8);AscendC::PipeBarrier<PIPE_V>();
    }
    LinalgLstsqTilingData t;
    uint32_t n,k,lane,group,groups,inverseLane,augmented,pivotFloorBits,residualRow,residualRows;
    uint32_t pivotRows[P];
    bool packedInput,ownsInverse,nativeInput,refine;
    AscendC::LocalTensor<float> mat,rhs,solution,residual,pivot,candidates,maximum,scales,weights,output,input,product;
    AscendC::LocalTensor<uint32_t> rowIx,outputIndices;
    AscendC::LocalTensor<int32_t> offsets,ix,tmpIx;
    AccurateRows<P,R> dot;
    AscendC::GlobalTensor<float> ga,gb,gx,gr,shared;
    uint64_t transposeSrc[4][16],transposeDst[4][16];
    AscendC::TransDataTo5HDParams transposeParams;
    static constexpr uint32_t vs=0;
    __aicore__ inline void PrepareNativeTranspose() {
        if constexpr(InitialOnly)if(nativeInput) {
            for(uint32_t group=0;group<4;++group) {
                for(uint32_t row=0;row<16;++row)
                    transposeSrc[group][row]=(uint64_t)input[(group*16+row)*P].GetPhyAddr();
                for(uint32_t col=0;col<8;++col) {
                    transposeDst[group][2*col]=(uint64_t)mat[col*L+group*16].GetPhyAddr();
                    transposeDst[group][2*col+1]=(uint64_t)mat[col*L+group*16+8].GetPhyAddr();
                }
            }
            transposeParams.dstHighHalf=false;transposeParams.srcHighHalf=false;
            transposeParams.repeatTimes=n/8;transposeParams.dstRepStride=L;transposeParams.srcRepStride=1;
        }

    }
    __aicore__ inline void InitViews(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR rr,GM_ADDR workspace) {
        n=t.n;k=t.k;
        packedInput=P==16 || ((Parallel || InitialOnly) && n%8!=0);
        constexpr uint32_t BASE=W*L+3*P*8+W+P+64+P+8*W+P+8*P+W;
        mat=AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC,0,BASE+AccurateRows<P,R>::FLOATS+256);
        rhs=mat[W*L];solution=rhs[P*8];residual=solution[P*8];
        pivot=residual[P*8];candidates=pivot[W];maximum=candidates[P];scales=maximum[64];weights=scales[P];
        rowIx=weights[8*W].template ReinterpretCast<uint32_t>();output=rowIx[P].template ReinterpretCast<float>();offsets=output[8*P].template ReinterpretCast<int32_t>();
        dot.Init(mat[BASE]);
        input=dot.raw;product=dot.h;ix=dot.a.template ReinterpretCast<int32_t>();tmpIx=dot.x.template ReinterpretCast<int32_t>();
        static_assert(!InitialOnly || 14*R>=P*8);
        outputIndices=dot.ixBase;
        ga.SetGlobalBuffer((__gm__ float*)aa);gb.SetGlobalBuffer((__gm__ float*)bb);gx.SetGlobalBuffer((__gm__ float*)xx);gr.SetGlobalBuffer((__gm__ float*)rr);

        lane=Parallel?AscendC::GetBlockIdx()%t.teams:0;group=Parallel?AscendC::GetBlockIdx()/t.teams:AscendC::GetBlockIdx();
        groups=Parallel?t.blocks/t.teams:t.blocks;
        ownsInverse=InverseRows && (!InitialOwner || lane!=0);
        inverseLane=InitialOwner?(lane?lane-1:0):lane;augmented=ownsInverse?8:k;
        nativeInput=InitialOnly && n%8==0;


        PrepareNativeTranspose();
        shared.SetGlobalBuffer((__gm__ float*)workspace+(uint64_t)group*2*P*8);

    }
    __aicore__ inline void PrepareBatch(uint32_t batch) {
            residualRow=0;residualRows=0;refine=P!=16 && !InitialOnly;
            if constexpr(Parallel) {
                const uint32_t rowTeams=(t.teams-(InitialOwner?1:0))/k,chunk=((P+rowTeams-1)/rowTeams+7)/8*8;
                residualRow=inverseLane/k*chunk;
                if((!InitialOwner || ownsInverse) && inverseLane<rowTeams*k && residualRow<n)residualRows=n-residualRow<chunk?n-residualRow:chunk;
            }

    }
    __aicore__ inline void LoadResidentInput(uint32_t batch) {
            // A transposed solve gives each lane eight independent inverse rows.
            // The original resident path instead replicates the initial solution.
            if constexpr(P==16) {
                // DMA and transpose-index construction use disjoint UB regions.
                if(n<P){AscendC::Duplicate(input,0.0f,P*P);AscendC::PipeBarrier<PIPE_ALL>();}
                AscendC::DataCopyPad(input,ga[(uint64_t)batch*n*n],AscendC::DataCopyExtParams{1,n*n*4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
                AscendC::DataCopyPad(rhs,gb[(uint64_t)batch*n*k],AscendC::DataCopyExtParams{1,n*k*4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
            }
            Index(tmpIx);
            AscendC::Muls(offsets,tmpIx,(int32_t)4,P);AscendC::PipeBarrier<PIPE_V>();
            if(!nativeInput) {
                AscendC::Muls(tmpIx,tmpIx,(int32_t)((packedInput?n:P)*4),P);AscendC::PipeBarrier<PIPE_V>();
                if(ownsInverse) {
                    AscendC::Brcb(weights.template ReinterpretCast<int32_t>(),tmpIx,P/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(ix,offsets,weights.template ReinterpretCast<int32_t>(),(uint64_t)P,(uint8_t)P,AscendC::BinaryRepeatParams{1,1,0,P/8,0,1});
                } else {
                    AscendC::Brcb(weights.template ReinterpretCast<int32_t>(),offsets,P/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(ix,tmpIx,weights.template ReinterpretCast<int32_t>(),(uint64_t)P,(uint8_t)P,AscendC::BinaryRepeatParams{1,1,0,P/8,0,1});
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr(P!=16)if(n<P)AscendC::Duplicate(input,0.0f,P*P);
            AscendC::Duplicate(solution,0.0f,P*k);AscendC::Duplicate(scales,1.0f,P);AscendC::PipeBarrier<PIPE_ALL>();

    }
    __aicore__ inline void PrepareInitialOutputMap() {
                if constexpr(InitialOnly)if(k>1) {
                    // Row byte offsets already exist here; prepare the final
                    // output mapping while input DMA targets disjoint buffers.
                    if((k&(k-1))==0) {
                        auto rowBytes=dot.l.template ReinterpretCast<int32_t>();
                        auto columnBytes=dot.h.template ReinterpretCast<int32_t>();
                        const uint32_t shift=k==2?1:(k==4?2:3);
                        AscendC::ShiftRight(rowBytes,offsets,(int32_t)(shift+2),P);AscendC::PipeBarrier<PIPE_V>();
                        AscendC::ShiftLeft(rowBytes,rowBytes,(int32_t)2,P);AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(columnBytes,rowBytes,(int32_t)k,P);AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Sub(columnBytes,offsets,columnBytes,P);AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(columnBytes,columnBytes,(int32_t)P,P);AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(outputIndices.template ReinterpretCast<int32_t>(),columnBytes,rowBytes,P);AscendC::PipeBarrier<PIPE_V>();
                        for(uint32_t c=1;c<k;++c)AscendC::Adds(outputIndices[c*P].template ReinterpretCast<int32_t>(),outputIndices.template ReinterpretCast<int32_t>(),(int32_t)(c*(P/k)*4),P);
                    } else {
                        for(uint32_t r=0;r<n;++r)for(uint32_t c=0;c<k;++c)outputIndices.SetValue(r*k+c,(c*P+r)*4);
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                    }
                }

    }
    __aicore__ inline void LoadWideInput(uint32_t batch) {
            if constexpr(P!=16) {
                if(packedInput)AscendC::DataCopyPad(input,ga[(uint64_t)batch*n*n],AscendC::DataCopyExtParams{1,n*n*4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
                else AscendC::DataCopyPad(input,ga[(uint64_t)batch*n*n],AscendC::DataCopyExtParams{(uint16_t)n,n*4,0,(P-(n+7)/8*8)/8,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});
                AscendC::DataCopyPad(rhs,gb[(uint64_t)batch*n*k],AscendC::DataCopyExtParams{1,n*k*4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
                PrepareInitialOutputMap();
                AscendC::PipeBarrier<PIPE_ALL>();
            }

    }
    __aicore__ inline void BuildAugmentedLayout() {
            if constexpr(P==16) {
                AscendC::Gather(product,input,ix.template ReinterpretCast<uint32_t>(),0,n*P);AscendC::PipeBarrier<PIPE_V>();
                AscendC::DataCopy(mat,product,AscendC::DataCopyParams{(uint16_t)n,P/8,0,(L-P)/8});
            } else {
                if(nativeInput) {
                    // Include padded rows so the complete 64-row capacity matches
                    // the original AscendC::Gather; RHS columns are filled afterward.
                    for(uint32_t group=0;group<4;++group) {
                        AscendC::TransDataTo5HD<float>(transposeDst[group],transposeSrc[group],transposeParams);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                } else {
                    if(packedInput){AscendC::Duplicate(mat,0.0f,W*L);AscendC::PipeBarrier<PIPE_V>();}
                    AscendC::Gather(mat,input,ix.template ReinterpretCast<uint32_t>(),0,(uint64_t)(packedInput?n:P),(uint8_t)n,(uint16_t)(L/8));
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmpIx,offsets,(int32_t)k,P);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(offsets,offsets,(int32_t)L,P);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(offsets[P],offsets[P-8],(int32_t)(8*L*4),(uint64_t)8,1,AscendC::UnaryRepeatParams{});AscendC::PipeBarrier<PIPE_ALL>();
            if(ownsInverse) {
                AscendC::Duplicate(mat[n*L],0.0f,8*L);AscendC::PipeBarrier<PIPE_ALL>();
                for(uint32_t c=0;c<8;++c)if(inverseLane*8+c<n)mat.SetValue((n+c)*L+inverseLane*8+c,1.0f);
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                for(uint32_t c=0;c<k;++c)AscendC::Gather(mat[(n+c)*L],rhs,tmpIx.template ReinterpretCast<uint32_t>(),c*4,P);AscendC::PipeBarrier<PIPE_V>();
            }

    }
    __aicore__ inline void FindPivotFloor() {
            pivotFloorBits=0;
            if constexpr(P==64 && !Parallel && !InitialOnly) {
                AscendC::Abs(product,input,P*P);AscendC::PipeBarrier<PIPE_V>();
                if constexpr(P==64) {
                    AscendC::WholeReduceMax(dot.a,product,64,P,1,1,8);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceMax(maximum,dot.a,P,1,1,1,8);
                } else {
                    // Collapse the 16-row input to one reduction without
                    // mixing index fields into the magnitude bound.
                    AscendC::Max(product,product,product[128],128);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Max(product,product,product[64],64);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceMax(maximum,product,64,1,1,1,8);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_S>(vs);AscendC::WaitFlag<AscendC::HardEvent::V_S>(vs);
                uint32_t maximumBits,index;AscendC::GetReduceRepeatMaxMinSpr<float>(maximumBits,index);
                pivotFloorBits=maximumBits>(3U<<23)?maximumBits-(3U<<23):0;
            }

    }
    __aicore__ inline void SelectPivot(uint32_t j,uint64_t *live,uint32_t &row,float &pivotValue) {
                auto col=mat[j*L];uint32_t bits;
                if constexpr(P==16 || (P==64 && (Parallel || InitialOnly))) {
                    AscendC::Abs(candidates,col,P);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceMax(maximum,candidates,live,1,1,1,8);AscendC::SetFlag<AscendC::HardEvent::V_S>(vs);AscendC::WaitFlag<AscendC::HardEvent::V_S>(vs);
                    AscendC::GetReduceRepeatMaxMinSpr<float>(bits,row);
                    if constexpr(P!=16)pivotValue=col.GetValue(row);
                } else {
                    row=63-static_cast<uint32_t>(AscendC::CountLeadingZero(live[0]&(~live[0]+1ULL)));
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(vs);AscendC::WaitFlag<AscendC::HardEvent::V_S>(vs);
                    union {float value;uint32_t bits;} first;first.value=col.GetValue(row);
                    if((first.bits&0x7fffffffU)<pivotFloorBits || (first.bits&0x7fffffffU)==0) {
                        AscendC::Abs(candidates,col,P);AscendC::PipeBarrier<PIPE_V>();AscendC::WholeReduceMax(maximum,candidates,live,1,1,1,8);AscendC::SetFlag<AscendC::HardEvent::V_S>(vs);AscendC::WaitFlag<AscendC::HardEvent::V_S>(vs);
                        AscendC::GetReduceRepeatMaxMinSpr<float>(bits,row);pivotValue=col.GetValue(row);
                    } else pivotValue=first.value;
                }

    }
    __aicore__ inline void ApplyPivot(uint32_t j,uint64_t *live,uint32_t row,float pivotValue) {
                auto col=mat[j*L];
                if constexpr(P==16) {
                    live[0]&=~(1ULL<<row);pivotRows[j]=row;rowIx.SetValue(j,row*4);
                    uint64_t mask[2]={1ULL<<(row%8),0};
                    const uint32_t width=n+augmented-j-1;
                    // Fetch the signed divisor with the remaining pivot row.
                    // The first broadcast block supplies all divisor lanes.
                    AscendC::Gather(pivot,mat,offsets.template ReinterpretCast<uint32_t>(),(j*L+row)*4,width+1);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(weights,pivot,(width+8)/8,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(scales[(row/8)*8],weights,1.0f,mask,1,AscendC::UnaryRepeatParams{1,1,8,8});
                    AscendC::Duplicate(col[(row/8)*8],0.0f,mask,1,1,8);
                    AscendC::Muls(weights,weights,-1.0f,8);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Div(col,col,weights,(uint64_t)P,1,AscendC::BinaryRepeatParams{1,1,0,8,8,0});AscendC::PipeBarrier<PIPE_V>();
                    AscendC::MulAddDst(mat[(j+1)*L],col,weights[8],(uint64_t)P,(uint8_t)width,AscendC::BinaryRepeatParams{1,1,0,L/8,0,1});AscendC::PipeBarrier<PIPE_V>();
                } else {
                    const float scaleValue=CheckedReciprocal(pivotValue);live[0]&=~(1ULL<<row);
                    pivotRows[j]=row;rowIx.SetValue(j,row*4);uint64_t mask[2]={1ULL<<(row%8),0};
                    AscendC::Duplicate(col[(row/8)*8],0.0f,mask,1,1,8);AscendC::Duplicate(scales[(row/8)*8],scaleValue,mask,1,1,8);
                    const uint32_t width=n+augmented-j-1;
                    const uint32_t firstColumn=InitialOnly?((j+1)/8*8):(j+1);
                    const uint32_t gatheredWidth=n+augmented-firstColumn;
                    AscendC::Gather(pivot,mat,offsets.template ReinterpretCast<uint32_t>(),(firstColumn*L+row)*4,gatheredWidth);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(col,col,-scaleValue,P);AscendC::PipeBarrier<PIPE_V>();
                    auto producer=weights,consumer=weights;
                    constexpr uint32_t coefficientStride=InitialOnly?L/8:1;
                    if constexpr(InitialOnly) {
                        // Aligned groups stay inside the reserved column capacity.
                        // Only the still-live columns consume their padding slots.
                        producer=mat[firstColumn*L+P];consumer=mat[(j+1)*L+P];
                    }
                    AscendC::Brcb(producer,pivot,(gatheredWidth+7)/8,AscendC::BrcbRepeatParams{coefficientStride,8*coefficientStride});AscendC::PipeBarrier<PIPE_V>();
                    AscendC::MulAddDst(mat[(j+1)*L],col,consumer,(uint64_t)P,(uint8_t)width,AscendC::BinaryRepeatParams{1,1,0,L/8,0,coefficientStride});AscendC::PipeBarrier<PIPE_V>();
                }

    }
    __aicore__ inline void Eliminate() {
            uint64_t live[2]={n==64?~0ULL:(1ULL<<n)-1,0};
            for(uint32_t j=0;j<n;++j) {
                uint32_t row;float pivotValue;
                SelectPivot(j,live,row,pivotValue);
                ApplyPivot(j,live,row,pivotValue);
            }

    }
    __aicore__ inline void GatherInitialSolution() {
            for(uint32_t c=n;c<n+augmented;++c) {
                if constexpr(P==16)AscendC::Div(mat[c*L],mat[c*L],scales,P);
                else AscendC::Mul(mat[c*L],mat[c*L],scales,P);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr(InverseRows) {
                if(ownsInverse) {
                    AscendC::Duplicate(output,0.0f,8*P);AscendC::PipeBarrier<PIPE_V>();
                    for(uint32_t c=0;c<8;++c)AscendC::Gather(output[c*P],mat[(n+c)*L],rowIx,0,n);AscendC::PipeBarrier<PIPE_V>();
                    if constexpr(!InitialOwner)for(uint32_t c=0;c<k;++c) {
                        AscendC::Gather(residual,rhs,tmpIx.template ReinterpretCast<uint32_t>(),c*4,n);AscendC::PipeBarrier<PIPE_V>();
                        ApplyInverseRows(pivot,output,residual,product,n);AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopy(shared[c*P+inverseLane*8],pivot,8);AscendC::PipeBarrier<PIPE_ALL>();
                    }
                } else {
                    for(uint32_t c=0;c<k;++c)AscendC::Gather(solution[c*P],mat[(n+c)*L],rowIx,0,n);AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(shared,solution,P*k);AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::SyncAll<true>();AscendC::DataCopy(solution,shared,P*k);AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                for(uint32_t c=0;c<k;++c)AscendC::Gather(solution[c*P],mat[(n+c)*L],rowIx,0,n);AscendC::PipeBarrier<PIPE_ALL>();
            }

    }
    __aicore__ inline void DecideRefinement() {
            if constexpr(Parallel) {
                // Decide from the preliminary solution's numerical magnitude.
                AscendC::Abs(product,solution,P*k);AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceMax(maximum,product,(uint64_t)P,(uint8_t)k,1,1,P/8,AscendC::ReduceOrder::ORDER_ONLY_VALUE);AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceMax(maximum[32],maximum,(uint64_t)k,1,1,1,8,AscendC::ReduceOrder::ORDER_ONLY_VALUE);AscendC::PipeBarrier<PIPE_ALL>();
                refine=!(maximum.GetValue(32)<=10.0f);
            }

    }
    __aicore__ inline void ParallelResidual(uint32_t batch) {
                if(residualRows) {
                    const uint32_t row=residualRow,rows=residualRows;
                    if(packedInput) {
                        AscendC::DataCopyPad(input,ga[(uint64_t)batch*n*n+row*n],AscendC::DataCopyExtParams{(uint16_t)rows,n*4,0,(P-(n+7)/8*8)/8,0},AscendC::DataCopyPadExtParams<float>{true,0,(uint8_t)((8-n%8)%8),0});AscendC::PipeBarrier<PIPE_ALL>();
                    } else if(row){AscendC::DataCopy(input,input[row*P],rows*P);AscendC::PipeBarrier<PIPE_ALL>();}
                    {
                        const uint32_t c=inverseLane%k;
                        AscendC::Muls(dot.x,solution[c*P],1.0f,(uint64_t)P,(uint8_t)rows,AscendC::UnaryRepeatParams{1,1,P/8,0});AscendC::PipeBarrier<PIPE_V>();dot.AccurateDot(rows);
                        for(uint32_t r=0;r<rows;++r)tmpIx.SetValue(r,(r*k+c)*4);AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::Gather(dot.a,rhs[row*k],tmpIx.template ReinterpretCast<uint32_t>(),0,rows);AscendC::PipeBarrier<PIPE_V>();
                        dot.D(dot.h,dot.a,dot.h,rows);dot.D(residual[c*P],dot.h,dot.l,rows);AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopyPad(shared[P*8+c*P+row],residual[c*P],AscendC::DataCopyExtParams{1,rows*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }

    }
    __aicore__ inline void SerialResidual(uint32_t batch) {
            for(uint32_t row=0;row<n;row+=R) {
                const uint32_t rows=n-row<R?n-row:R;
                AscendC::PipeBarrier<PIPE_ALL>();
                for(uint32_t c=0;c<k;++c) {
                    AscendC::Muls(dot.x,solution[c*P],1.0f,(uint64_t)P,(uint8_t)rows,AscendC::UnaryRepeatParams{1,1,P/8,0});AscendC::PipeBarrier<PIPE_V>();dot.AccurateDot(rows);
                    for(uint32_t r=0;r<rows;++r)tmpIx.SetValue(r,(r*k+c)*4);AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Gather(dot.a,rhs[row*k],tmpIx.template ReinterpretCast<uint32_t>(),0,rows);AscendC::PipeBarrier<PIPE_V>();dot.D(dot.h,dot.a,dot.h,rows);dot.D(residual[c*P+row],dot.h,dot.l,rows);
                }
            }

    }
    __aicore__ inline void BuildResidual(uint32_t batch) {
        if constexpr(Parallel) {
            if(refine)ParallelResidual(batch);
            AscendC::SyncAll<true>();
            if(refine && (ownsInverse || (!InverseRows && lane==0))){AscendC::DataCopy(residual,shared[P*8],P*k);AscendC::PipeBarrier<PIPE_ALL>();}
        } else if(refine)SerialResidual(batch);

    }
    __aicore__ inline void WriteInverseRows(uint32_t batch,GM_ADDR rr) {
                const uint32_t first=inverseLane*8,rows=ownsInverse && first<n?(n-first<8?n-first:8):0;
                if(rows) {
                    for(uint32_t c=0;c<k;++c) {
                        ApplyInverseRows(pivot,output,residual[c*P],product,n);
                        AscendC::Add(pivot,pivot,solution[c*P+first],8);AscendC::PipeBarrier<PIPE_V>();
                        AscendC::DataCopy(mat[c*8],pivot,8);AscendC::PipeBarrier<PIPE_V>();
                    }
                    for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<k;++c)tmpIx.SetValue(r*k+c,(c*8+r)*4);
                    AscendC::PipeBarrier<PIPE_ALL>();AscendC::Gather(product,mat,tmpIx.template ReinterpretCast<uint32_t>(),0,rows*k);AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyPad(gx[(uint64_t)batch*n*k+first*k],product,AscendC::DataCopyExtParams{1,rows*k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(lane==0 && rr && t.residualElements>=(uint64_t)t.batches*k){AscendC::Duplicate(pivot,0.0f,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(gr[(uint64_t)batch*k],pivot,AscendC::DataCopyExtParams{1,k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();}

    }
    __aicore__ inline void CorrectSolution() {
            if(k>1) {
                for(uint32_t c=0;c<k;++c)offsets.SetValue(c,c*P*4);AscendC::PipeBarrier<PIPE_ALL>();
                for(uint32_t j=0;j<n;++j) {
                    AscendC::Gather(pivot,residual,offsets.template ReinterpretCast<uint32_t>(),pivotRows[j]*4,k);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(weights,pivot,1,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                    AscendC::MulAddDst(residual,mat[j*L],weights,(uint64_t)P,(uint8_t)k,AscendC::BinaryRepeatParams{1,1,0,P/8,0,1});AscendC::PipeBarrier<PIPE_V>();
                }
            } else {
                for(uint32_t j=0;j<n;++j) {
                    const uint32_t row=pivotRows[j];
                    AscendC::Brcb(weights,residual[(row/8)*8],1,AscendC::BrcbRepeatParams{1,8});AscendC::PipeBarrier<PIPE_V>();
                    AscendC::MulAddDst(residual,mat[j*L],weights[(row%8)*8],(uint64_t)P,1,AscendC::BinaryRepeatParams{1,1,0,P/8,L/8,0});AscendC::PipeBarrier<PIPE_V>();
                }
            }
            for(uint32_t c=0;c<k;++c) {
                auto corr=residual[c*P];
                AscendC::Mul(corr,corr,scales,P);AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(pivot,corr,rowIx,0,n);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(solution[c*P],solution[c*P],pivot,n);AscendC::PipeBarrier<PIPE_V>();
            }

    }
    __aicore__ inline void WriteSolution(uint32_t batch,GM_ADDR rr) {
            if(k==1) {
                AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(gx[(uint64_t)batch*n],solution,AscendC::DataCopyExtParams{1,n*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                if constexpr(!InitialOnly) {
                if((k&(k-1))==0) {
                    // Flatten (row,rhs) into column-contiguous solution offsets.
                    // The explicit-event index helper does not need a AscendC::TPipe.
                    Index(tmpIx);const uint32_t shift=k==2?1:(k==4?2:3);
                    AscendC::ShiftRight(ix,tmpIx,(int32_t)shift,P);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(output.template ReinterpretCast<int32_t>(),ix,(int32_t)k,P);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(tmpIx,tmpIx,output.template ReinterpretCast<int32_t>(),P);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(ix,ix,(int32_t)4,P);AscendC::Muls(tmpIx,tmpIx,(int32_t)(P*4),P);AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(tmpIx,tmpIx,ix,P);AscendC::PipeBarrier<PIPE_V>();
                    for(uint32_t c=1;c<k;++c)AscendC::Adds(tmpIx[c*P],tmpIx,(int32_t)(c*(P/k)*4),P);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    for(uint32_t r=0;r<n;++r)for(uint32_t c=0;c<k;++c)tmpIx.SetValue(r*k+c,(c*P+r)*4);AscendC::PipeBarrier<PIPE_ALL>();
                }
                }
                auto resultOffsets=InitialOnly?outputIndices:tmpIx.template ReinterpretCast<uint32_t>();
                AscendC::Gather(output,solution,resultOffsets,0,n*k);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(gx[(uint64_t)batch*n*k],output,AscendC::DataCopyExtParams{1,n*k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
            }
            if(rr && t.residualElements>=(uint64_t)t.batches*k){AscendC::Duplicate(pivot,0.0f,8);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(gr[(uint64_t)batch*k],pivot,AscendC::DataCopyExtParams{1,k*4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();}

    }
public:
    __aicore__ inline void Run(GM_ADDR aa,GM_ADDR bb,GM_ADDR xx,GM_ADDR rr,GM_ADDR workspace,const LinalgLstsqTilingData &data) {
        t=data;InitViews(aa,bb,xx,rr,workspace);
        for(uint32_t batch=group;batch<t.batches;batch+=groups) {
            PrepareBatch(batch);LoadResidentInput(batch);LoadWideInput(batch);
            BuildAugmentedLayout();FindPivotFloor();Eliminate();GatherInitialSolution();
            DecideRefinement();BuildResidual(batch);
            if constexpr(InverseRows)WriteInverseRows(batch,rr);
            else if(lane==0) {
                if(refine)CorrectSolution();
                WriteSolution(batch,rr);
            }
            if constexpr(Parallel)if(batch+groups<t.batches)AscendC::SyncAll<true>();
        }
    }

};
