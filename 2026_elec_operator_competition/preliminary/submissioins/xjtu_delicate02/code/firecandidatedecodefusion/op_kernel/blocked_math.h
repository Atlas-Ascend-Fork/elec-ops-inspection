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
// Inline definitions included after BlockedTransposeDecode is declared.

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) bool BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::JointRangeBitmapAllLE() {
#if __NPU_ARCH__ == 2201 && !defined(ASCENDC_CPU_DEBUG) && !defined(__CCE_KT_TEST__)
        static_assert(sizeof(T)==4 && LargeMetadata,"large Float bitmap resource");
        constexpr uint32_t Bits=3*P,Compared=(Bits+63)/64*64;
        // This guard is the first use of a stable owner. The original
        // zero fill still owns the aligned suffix (gap and initial S).
        constexpr uint32_t FullHalves=Compared/16;
        static_assert((2*P+Compared)*4<=O0,"comparison remains in initialized P/S/V");
        static_assert(FullHalves*2<=1024,"converted predicate overlaps reduction");
        static_assert(1024+128<=4*P*4,"predicate scratch overlaps bitmap");
        LocalTensor<float> source(TPosition::VECCALC,2*P*4,Compared);
        auto bitmap=priorPackedBuf.template Get<uint8_t>()[4*P*4];
        auto failed=bitmap.template ReinterpretCast<uint16_t>();
        auto converted=priorPackedBuf.template Get<half>();
        LocalTensor<half> reduced(TPosition::VECCALC,F0+1024,64);
        CompareScalar(bitmap,source,32.f,CMPMODE::LE,Compared);
        PipeBarrier<PIPE_V>();
        Not(failed,failed,Compared/16);PipeBarrier<PIPE_V>();
        // Every nonzero signed16 integer remains finite and nonzero in Half.
        // Half is only an exact zero/nonzero carrier for this predicate.
        Cast(converted,failed.template ReinterpretCast<int16_t>(),RoundMode::CAST_NONE,FullHalves);
        PipeBarrier<PIPE_V>();Abs(converted,converted,FullHalves);PipeBarrier<PIPE_V>();
        ResetMask();SetMaskNorm();
        constexpr uint32_t Full=FullHalves/128,Tail=FullHalves%128;
        if constexpr(Full) {
            WholeReduceMax(reduced,converted,128,Full,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            if constexpr(Full>1) {
                WholeReduceMax(reduced,reduced,Full,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(Tail) {
            WholeReduceMax(reduced[32],converted[Full*128],Tail,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const auto* result=reinterpret_cast<const volatile __ubuf__ uint32_t*>(reduced.GetPhyAddr());
        bool all=true;
        if constexpr(Full)all=(result[0]&0x7fffu)==0;
        if constexpr(Tail)all=all&&((result[16]&0x7fffu)==0);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        return all;
#else
        return false;
#endif
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool CarryCounter>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::DecodeGeometry(LocalTensor<float> s,LocalTensor<float> v,LocalTensor<float> p,bool shiftGrid) {
        auto* sp=(__ubuf__ float*)s.GetPhyAddr();
                        auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                        auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                        if constexpr(!CarryCounter)SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                        auto* centerSource=vp;
                        if constexpr(sizeof(T)==4 && !BatchOne) {
                            if(shiftGrid)centerSource=sp;
                            else vadds(vp,sp,-.5f,1,1,1,8,8);
                        } else vadds(vp,sp,-.5f,1,1,1,8,8);
                        vmul(vp+2*P,sp+2*P,sp+2*P,1,1,1,1,8,8,8);
                        pipe_barrier(PIPE_V);
                        vadd(sp,centerSource,pp,1,1,1,1,8,8,8);
                        vmul(sp+2*P,vp+2*P,pp+3*P,1,1,1,1,8,8,8);
                        pipe_barrier(PIPE_V);
                        SetVectorMask<float,MaskMode::COUNTER>(P);
        #pragma unroll
                        for(uint32_t c=0;c<4;++c) {
                            if constexpr(sizeof(T)==4 && !BatchOne) {
                                if(c<2 || !shiftGrid)vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                            } else vmul(sp+c*P,sp+c*P,pp+2*P,1,1,1,1,8,8,8);
                        }
                        if constexpr(BatchOne)vmul(vp+4*P,sp+4*P,sp+5*P,1,1,1,1,8,8,8);
                        // HalfEight coordinates and score write disjoint planes;
                        // their original final barrier joins all five outputs.
                        if constexpr(!HalfEight)pipe_barrier(PIPE_V);
                        if constexpr(!DirectScoreDivide) {
                            if constexpr(BatchOne)vmuls(sp+4*P,vp+4*P,.25f,1,1,1,8,8);
                            else vmuls(sp+4*P,sp+4*P,.5f,1,1,1,8,8);
                            pipe_barrier(PIPE_V);
                        }
                        ResetMask();SetMaskNorm();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool InPlace>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::EvaluateSigmoid(LocalTensor<float> s,LocalTensor<float> work) {
        if constexpr(!BatchOne) {
            if constexpr(!HalfEight) {SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);}
            if constexpr(sizeof(T)==4) {
                Muls<float,false>(work,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            }
            Exp<float,false>(s,work,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
            Adds<float,false>(work,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});
        #if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
            if constexpr(!((InPlace && FireInputFloatCompletionJoins) ||
                (!DualOutput && UniformTiles && DirectScoreDivide))) PipeBarrier<PIPE_V>();
        #else
            if constexpr(!(InPlace && FireInputFloatCompletionJoins)) PipeBarrier<PIPE_V>();
        #endif
            if constexpr(DirectScoreDivide && (DualOutput || UniformTiles)) {
                SetVectorMask<float,MaskMode::COUNTER>(4*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),work,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            }
            SetVectorMask<float,MaskMode::COUNTER>(P);
            auto* den=(__ubuf__ float*)work.GetPhyAddr();
            vmul(den+4*P,den+4*P,den+5*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
            if constexpr(DirectScoreDivide) {
                if constexpr(!(DualOutput || UniformTiles)) {
                    SetVectorMask<float,MaskMode::COUNTER>(4*P);
                    Div<float,false>(s,numeratorBuf.template Get<float>(),work,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                    PipeBarrier<PIPE_V>();SetVectorMask<float,MaskMode::COUNTER>(P);
                }
                Div<float,false>(s[4*P],scoreNumeratorBuf.template Get<float>(),work[4*P],MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Div<float,false>(s,numeratorBuf.template Get<float>(),work,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
                PipeBarrier<PIPE_V>();
            }
            if constexpr(!ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::RefreshPrior(uint32_t n,uint8_t repeats,uint32_t& cachedN,bool& needsRepair,bool& shiftGrid) {
        auto p=priorsBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto packed=priorPackedBuf.template Get<float>();
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        UnpackPriors(repeats);
        // Packed priors have expired; the independent slot now owns
        // the guard without touching either previous output buffer.
        if constexpr(BatchOne)BeginOneStrideMax(p[2*P],packed,G*Q);
        else if constexpr(sizeof(T)==4) {
            shiftGrid=false;
            if(stablePriorOwner && endTile-beginTile>1) {
                // One sufficient range test also proves the original
                // stride-only center guard. ScratchV expires here.
                bool bitmapAll=false;
                if constexpr(LargeMetadata)bitmapAll=JointRangeBitmapAllLE();
                if(__builtin_expect(bitmapAll,true))shiftGrid=true;
                else {
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                auto* vp=(__ubuf__ float*)v.GetPhyAddr();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(P);
                vmax(vp,pp+2*P,pp+3*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                vmax(vp,vp,pp+4*P,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
                ResetMask();SetMaskNorm();
                shiftGrid=DirectStrideMax<LargeMetadata>(v,packed,P)<=32.f;
                }
                // A failed joint range test keeps the original
                // stride decision and untouched numerical priors.
                needsRepair=shiftGrid?false:DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
            } else needsRepair=DirectStrideMax<LargeMetadata>(p[2*P],packed,P)>32.f;
            if(shiftGrid) {
                auto* pp=(__ubuf__ float*)p.GetPhyAddr();
                SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(2*P);
                vadds(pp,pp,-.5f,1,1,1,8,8);pipe_barrier(PIPE_V);
                SetVectorMask<float,MaskMode::COUNTER>(P);
                vmul(pp+3*P,pp+3*P,pp+2*P,1,1,1,1,8,8,8);
                vmul(pp+4*P,pp+4*P,pp+2*P,1,1,1,1,8,8,8);
                pipe_barrier(PIPE_V);ResetMask();SetMaskNorm();
            }
        } else needsRepair=DirectStrideMax<HalfEight>(p[2*P],packed,P)>4096.f;
        cachedN=n;
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::ConvertRawHalf(LocalTensor<float> v) {
        if constexpr(sizeof(T)==2 && !BatchOne) {
            // Adding the high bit modulo2^16 toggles only binary16 sign.
            // It preserves every finite significand, including subnormals.
            SetMaskCount();SetVectorMask<int16_t,MaskMode::COUNTER>(6*P);
            Adds<int16_t,false>(valuesBuf.template Get<int16_t>(),
                workBuf.template Get<int16_t>(),static_cast<int16_t>(-32768),
                MASK_PLACEHOLDER,1,{1,1,8,8});
            PipeBarrier<PIPE_V>();
            if constexpr(HalfEight) {
                // Preserve the sign operation's counter mask for widening.
                Cast<float,T,false>(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,
                    MASK_PLACEHOLDER,1,{1,1,8,4});
            } else {
                ResetMask();SetMaskNorm();
                Cast(v,valuesBuf.template Get<T>(),RoundMode::CAST_NONE,6*P);
            }
            PipeBarrier<PIPE_V>();
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::RefreshGuardPrior(uint32_t n,uint8_t repeats,uint32_t& cachedN,uint64_t& retained) {
        auto p=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);UnpackPriors(repeats);
        auto mask=priorPackedBuf.template Get<uint8_t>()[2*P*sizeof(float)];
        CompareScalar(mask,p[2*P],sizeof(T)==2?4096.f:32.f,CMPMODE::LE,G*Q);PipeBarrier<PIPE_V>();
        GatherMask(packed,p[2*P],mask.template ReinterpretCast<uint32_t>(),true,G*Q,{1,1,8,1},retained);
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();cachedN=n;
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) LocalTensor<T> BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::OutputPacked() {
        return stablePriorOwner?priorPackedBuf.template Get<T>():
            (sizeof(T)==2?valuesBuf.template Get<T>():workBuf.template Get<T>());
    }
