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
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::EvaluatePackedRaw(uint32_t count) {
        auto input=packedBuf.template Get<float>();
        auto scratch=workBuf.template Get<float>();
        if constexpr(sizeof(T)==2) {
            Cast(input,workBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);
            PipeBarrier<PIPE_V>();
        }
        // Pointwise FP32 arithmetic commutes with the exact row permutation.
        // Only the valid contiguous stream participates in the sigmoid.
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
        Muls<float,false>(scratch,input,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Exp<float,false>(input,scratch,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(scratch,input,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(input,numeratorBuf.template Get<float>(),scratch,
            MASK_PLACEHOLDER,1,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
        ResetMask();SetMaskNorm();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::EvaluateFullHalfPlanes() {
        auto values=valuesBuf.template Get<float>();
        auto scratch=workBuf.template Get<float>();
        auto bits=packedBuf.template Get<int16_t>();
        SetMaskCount();SetVectorMask<int16_t,MaskMode::COUNTER>(6*P);
        Adds<int16_t,false>(bits,bits,static_cast<int16_t>(-32768),MASK_PLACEHOLDER,1,{1,1,8,8});
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
        Cast(scratch,packedBuf.template Get<T>(),RoundMode::CAST_NONE,6*P);
        PipeBarrier<PIPE_V>();
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*P);
        Exp<float,false>(values,scratch,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(scratch,values,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(values,numeratorBuf.template Get<float>(),scratch,
            MASK_PLACEHOLDER,1,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
        ResetMask();SetMaskNorm();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::UnpackRawStatic(uint8_t repeats) {
        // Install two disjoint VA contexts before either transpose.
                    // Reusing the other bank does not overwrite a queued context.
        #pragma unroll
                    for(uint32_t block=0;block<6;block+=2) {
                        set_va_reg_sb(VA0,rawSrc[block]);set_va_reg_sb(VA1,rawSrc[block]+8);
                        set_va_reg_sb(VA2,rawDst[block]);set_va_reg_sb(VA3,rawDst[block]+8);
                        set_va_reg_sb(VA4,rawSrc[block+1]);set_va_reg_sb(VA5,rawSrc[block+1]+8);
                        set_va_reg_sb(VA6,rawDst[block+1]);set_va_reg_sb(VA7,rawDst[block+1]+8);
                        scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:96);
                        scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:2,repeats==1?0:96);
                    }
                    PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::UnpackRawFallback(uint8_t repeats) {
        #pragma unroll
                for(uint32_t block=0;block<RAWBLOCKS;++block) {
                    if constexpr(BatchOne)LoadAddressContext(block);
                    else {
                    if constexpr(!FloatLayout) {
                        if(block!=0) {set_va_reg_sb(VA0,rawSrc[block]);set_va_reg_sb(VA1,rawSrc[block]+8);}
                    } else {set_va_reg_sb(VA0,rawSrc[block]);set_va_reg_sb(VA1,rawSrc[block]+8);}
                    set_va_reg_sb(VA2,rawDst[block]);set_va_reg_sb(VA3,rawDst[block]+8);
                    }
                    if constexpr(!FloatLayout) {
                        if(block==0)scatter_vnchwconv_b16(VA2,VA4,repeats,repeats==1?0:1,repeats==1?0:RAW_SOURCE_STRIDE);
                        else scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:RAW_SOURCE_STRIDE);
                    } else
                        scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:96);
                }
                PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool FinalFence>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::UnpackRaw(uint8_t repeats) {
        static_assert(FinalFence || (LargeMetadata && FloatLayout && !HalfEight && !BatchOne),
            "unpack completion requires the Hot large-Float event/mask joins");
        if constexpr(NativeHalfPlanes || HalfEight) {
            LoadAddressContext<false>(0);LoadAddressContext<true>(1);
            scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:48);
            scatter_vnchwconv_b16(VA6,VA4,repeats,repeats==1?0:1,repeats==1?0:48);
            LoadAddressContext<false>(2);
            scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:1,repeats==1?0:48);
            PipeBarrier<PIPE_V>();
            return;
        }

        if constexpr(BatchOne || LargeMetadata) {
            UnpackMetadataRaw<FinalFence>(repeats);return;
        }
        if constexpr(FloatLayout && !BatchOne) {
            UnpackRawStatic(repeats);return;
        }
        UnpackRawFallback(repeats);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::UnpackPriors(uint8_t repeats) {
        if constexpr(BatchOne || LargeMetadata) {
            constexpr uint32_t context=HalfEight?3:6;
#pragma unroll
            for(uint32_t block=0;block+1<PRBLOCKS;block+=2) {
                LoadAddressContext<false>(context+block);
                LoadAddressContext<true>(context+1+block);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:10*G);
                scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:2,repeats==1?0:10*G);
            }
            if constexpr(PRBLOCKS%2) {
                LoadAddressContext<false>(context+PRBLOCKS-1);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:10*G);
            }
            PipeBarrier<PIPE_V>();
            return;
        }
#pragma unroll
        for(uint32_t block=0;block<PRBLOCKS;++block) {
            if constexpr(BatchOne)LoadAddressContext(6+block);
            else {
            set_va_reg_sb(VA0,priorSrc[block]);set_va_reg_sb(VA1,priorSrc[block]+8);
            set_va_reg_sb(VA2,priorDst[block]);set_va_reg_sb(VA3,priorDst[block]+8);
            }
            // The two VA destination halves select the low/high eight
            // phases. Half owners use twice as many eight-row repeats.
            const uint8_t priorRepeats=WidePhaseLayout?repeats:2*repeats;
            scatter_vnchwconv_b32(VA2,VA0,priorRepeats,
                priorRepeats==1?0:(WidePhaseLayout?2:1),
                priorRepeats==1?0:(WidePhaseLayout?10*G:80));
        }
        PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::PackMetadataOutput(uint32_t Context,uint8_t repeats) {
        #pragma unroll
                for(uint32_t block=0;block+1<5;block+=2) {
                    LoadAddressContext<false>(Context+block);
                    LoadAddressContext<true>(Context+1+block);
                    scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                    scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:80,repeats==1?0:2);
                }
                LoadAddressContext<false>(Context+4);
                scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<uint32_t Context>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::PackOutputWithContext(uint8_t repeats) {
        static_assert(Context+5<=LargeContexts,"output address contexts exceed metadata");
        PackMetadataOutput(Context,repeats);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::PackOutputStatic(uint8_t repeats) {
        #pragma unroll
                    for(uint32_t block=0;block+1<5;block+=2) {
                        set_va_reg_sb(VA0,outputSrc[block]);set_va_reg_sb(VA1,outputSrc[block]+8);
                        set_va_reg_sb(VA2,outputDst[block]);set_va_reg_sb(VA3,outputDst[block]+8);
                        set_va_reg_sb(VA4,outputSrc[block+1]);set_va_reg_sb(VA5,outputSrc[block+1]+8);
                        set_va_reg_sb(VA6,outputDst[block+1]);set_va_reg_sb(VA7,outputDst[block+1]+8);
                        scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                        scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:80,repeats==1?0:2);
                    }
                    set_va_reg_sb(VA0,outputSrc[4]);set_va_reg_sb(VA1,outputSrc[4]+8);
                    set_va_reg_sb(VA2,outputDst[4]);set_va_reg_sb(VA3,outputDst[4]+8);
                    scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                    PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::PackOutputFallback(uint8_t repeats) {
        #pragma unroll
                for(uint32_t block=0;block<5;++block) {
                    if constexpr(BatchOne)LoadAddressContext(11+block);
                    else {
                    if constexpr(!FloatLayout) {
                        if(block!=0) {set_va_reg_sb(VA0,outputSrc[block]);set_va_reg_sb(VA1,outputSrc[block]+8);}
                    } else {set_va_reg_sb(VA0,outputSrc[block]);set_va_reg_sb(VA1,outputSrc[block]+8);}
                    set_va_reg_sb(VA2,outputDst[block]);set_va_reg_sb(VA3,outputDst[block]+8);
                    }
                    if constexpr(HalfEight) {
                        if(block==0)scatter_vnchwconv_b32(VA2,VA6,repeats,repeats==1?0:80,repeats==1?0:2);
                        else scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                    } else if constexpr(!FloatLayout) {
                        if(block==0)scatter_vnchwconv_b16(VA2,VA6,repeats,repeats==1?0:80,repeats==1?0:1);
                        else scatter_vnchwconv_b16(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:1);
                    } else
                        scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:80,repeats==1?0:2);
                }
                PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::PackOutput(uint8_t repeats) {
        if constexpr(HalfEight) {
            // Keep both legal owner layouts; materialize their addresses locally.
            if(stablePriorOwner)PackOutputWithContext<8>(repeats);
            else PackOutputWithContext<13>(repeats);
            return;
        }
        if constexpr(LargeMetadata && sizeof(T)==4) {
            // Both original owner layouts use locally materialized addresses.
            if(stablePriorOwner)PackOutputWithContext<11>(repeats);
            else PackOutputWithContext<16>(repeats);
            return;
        }
        if constexpr(BatchOne || LargeMetadata) {
            const uint32_t context=HalfEight?(stablePriorOwner?8:13):(LargeMetadata && !stablePriorOwner?16:11);
            PackMetadataOutput(context,repeats);return;
        }
        if constexpr(FloatLayout && !BatchOne) {
            PackOutputStatic(repeats);return;
        }
        PackOutputFallback(repeats);
    }
