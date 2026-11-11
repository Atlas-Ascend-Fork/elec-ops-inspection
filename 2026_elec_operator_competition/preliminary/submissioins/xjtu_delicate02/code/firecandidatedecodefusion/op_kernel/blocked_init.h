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
template<bool Second>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::LoadAddressContext(uint32_t index) {
        auto words=addressBuf.template Get<uint64_t>()[index*((LargeMetadata || BatchOne)?16:8)];
        auto source=(__ubuf__ uint64_t*)words.GetPhyAddr();
        if constexpr(LargeMetadata || BatchOne) {
            if constexpr(Second) {
                vld_va_reg(VA4,source,L128);vld_va_reg(VA5,source+4,L128);
                vld_va_reg(VA6,source+8,L128);vld_va_reg(VA7,source+12,L128);
            } else {
                vld_va_reg(VA0,source,L128);vld_va_reg(VA1,source+4,L128);
                vld_va_reg(VA2,source+8,L128);vld_va_reg(VA3,source+12,L128);
            }
        } else {
            auto destination=(__ubuf__ uint64_t*)words[4].GetPhyAddr();
            if constexpr(Second) {
                vld_va_reg(VA4,source,L128);vld_va_reg(VA5,source,H128);
                vld_va_reg(VA6,destination,L128);vld_va_reg(VA7,destination,H128);
            } else {
                vld_va_reg(VA0,source,L128);vld_va_reg(VA1,source,H128);
                vld_va_reg(VA2,destination,L128);vld_va_reg(VA3,destination,H128);
            }
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::StagePrior(uint32_t n,uint32_t count) {
        auto packed=priorPackedBuf.template Get<float>();
        const uint32_t padded=(count*5+7)/8*8;
        const uint32_t consumed=(count+16*G-1)/(16*G)*16*G*5;
        if constexpr(!BatchOne) {
            if(consumed>padded) {Duplicate(packed[padded],0.f,consumed-padded);PipeBarrier<PIPE_V>();}
        }
        // The previous stride guard uses the released packed-prior slot.
        // Its Scalar read completion is fenced inside DirectStrideMax.
        if constexpr(!BatchOne) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
        CopyDenseInput(packed,prior,n*5,count*5);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::LoadRaw(uint32_t row,uint32_t count) {
        if constexpr(UniformTiles && !DualOutput) {
            auto source=raw[row*6];
            CopyDenseInput(packedBuf.template Get<T>(),source,0,count*6);
        } else if constexpr(sizeof(T)==2 && BatchOne)
            CopyDenseInput(workBuf.template Get<T>(),raw,row*6,count*6);
        else CopyDenseInput(packedBuf.template Get<T>(),raw,row*6,count*6);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::RetainHalfContexts() {
        if constexpr(!FloatLayout && !LargeMetadata) {
            set_va_reg_sb(VA4,rawSrc[0]);set_va_reg_sb(VA5,rawSrc[0]+8);
            set_va_reg_sb(VA6,outputSrc[0]);set_va_reg_sb(VA7,outputSrc[0]+8);
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::PrefetchAddressMetadata(GM_ADDR addressMetadata) {
        GlobalTensor<uint64_t> metadata;
        constexpr uint32_t kind=LargeMetadata?(sizeof(T)==2?4U:3U):((sizeof(T)==2 && BatchOne)?2U:1U);
        metadata.SetGlobalBuffer(fire_metadata::Table<kind,K>::Address());
        LocalTensor<uint64_t> destination(TPosition::VECCALC,MetadataBase,LargeMetadata?LargeContexts*16:(BatchOne?256:128));
        DataCopy(destination,metadata,LargeMetadata?LargeContexts*16:(BatchOne?256:128));
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::InitOwnership(uint32_t batchShift,uint32_t uniformTileCount,uint32_t uniformOwner) {
        const uint32_t total=((d.n+K-1)/K)*d.batch;
        if constexpr(BatchOne) {
            uint32_t part;
            if(!(d.batch&(d.batch-1))) {
                part=GetBlockIdx()>>batchShift;oneBatch=GetBlockIdx()&(d.batch-1);
            } else {
                part=GetBlockIdx()/d.batch;oneBatch=GetBlockIdx()%d.batch;
            }
            // Ownership units follow the GM tensor dtype, independently
            // of the internal8-row FP32 shuffle layout.
            constexpr uint32_t OwnerUnit=sizeof(T)==2?16:8;
            if constexpr(UniformFull) {
                oneN=part*(G*Q);oneCount=G*Q;
            } else {
                oneN=(part*d.unitsPerWorker+(part<d.extraUnits?part:d.extraUnits))*OwnerUnit;
                const uint32_t length=(d.unitsPerWorker+(part<d.extraUnits))*OwnerUnit;
                oneCount=d.n-oneN<length?d.n-oneN:length;
            }
            beginTile=0;endTile=1;stablePriorOwner=true;
        } else {
            if constexpr(UniformTiles) {
                static_assert(DirectScoreDivide,"uniform tile partition requires large Float");
                beginTile=uniformOwner*uniformTileCount;
                endTile=beginTile+uniformTileCount;
            } else {
                beginTile=total*GetBlockIdx()/d.blocks;endTile=total*(GetBlockIdx()+1)/d.blocks;
            }
            stablePriorOwner=beginTile/d.batch==(endTile-1)/d.batch;
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::StageInitialInputs() {
        if constexpr(!BatchOne) {
            const uint32_t firstN=beginTile/d.batch*K;
            const uint32_t firstB=beginTile%d.batch;
            const uint32_t firstCount=d.n-firstN<K?d.n-firstN:K;
            constexpr uint32_t RawAlign=32/sizeof(T);
            const uint32_t padded=(firstCount*6+RawAlign-1)/RawAlign*RawAlign;
            const uint32_t consumed=(firstCount+16*G-1)/(16*G)*16*G*6;
            if(consumed>padded) {
                Duplicate(packedBuf.template Get<T>()[padded],static_cast<T>(0.f),consumed-padded);
                PipeBarrier<PIPE_V>();
            }
            // StagePrior fences both packed input tails before MTE2.
            // Later geometry clears end at O0 and cannot touch either DMA.
            StagePrior(firstN,firstCount);
            LoadRaw(firstB*d.n+firstN,firstCount);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        if constexpr(BatchOne) {
            if(UniformFull || BatchOne || oneCount==G*Q) {
                // Half packed tails are initialized separately from DMA prefixes.
                // Packed prior/raw input destinations are disjoint, so their
                // DMA can overlap that initialization without a V_MTE2 wait.
                StagePrior(oneN,UniformFull?G*Q:oneCount);
                LoadRaw(oneBatch*d.n+oneN,UniformFull?G*Q:oneCount);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool SkipFullPadding>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::InitFullPadding() {
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto pbuf=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();

        // Every packed raw/prior element read by the transposes is
        // written by MTE2 (and the valid half cast) before use.
        // The transposes write G*Q rows in every plane. Initialize
        // the remaining padding in the representation read next.
        if constexpr(!SkipFullPadding) {
        constexpr uint32_t padding=NativeHalfPlanes?16:24;
        SetVectorMask<float,MaskMode::NORMAL>(0,(uint64_t(1)<<padding)-1);
        Duplicate<float,false>(pbuf[G*Q],0.f,MASK_PLACEHOLDER,5,1,P/8);
        if constexpr(NativeHalfPlanes) {
            SetVectorMask<uint16_t,MaskMode::NORMAL>(0,(uint64_t(1)<<padding)-1);
            Duplicate<uint16_t,false>(packedBuf.template Get<uint16_t>()[G*Q],
                0,MASK_PLACEHOLDER,6,1,P/16);
        } else Duplicate<float,false>(s[G*Q],0.f,MASK_PLACEHOLDER,6,1,P/8);
        }
        SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
        Duplicate<float,false>(numeratorBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
        ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::InitPartialPadding() {
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto pbuf=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();

        // One counter-mode clear includes all plane padding, packed tails
        // and scratch. Address metadata starts after the numerator slot.
        LocalTensor<float> zeroRegion(TPosition::VECCALC,0,C0/4);
        SetMaskCount();
        if constexpr(NativeHalfPlanes && !UniformFull) {
            // MTE2 writes the complete padded valid prefixes. Vector
            // writes only their disjoint suffixes and the output planes.
            const uint32_t rawPadded=(oneCount*6+15)/16*8;
            const uint32_t priorPadded=(oneCount*5+7)/8*8;
            SetVectorMask<float,MaskMode::COUNTER>(3*P-rawPadded);
            vector_dup((__ubuf__ float*)v.GetPhyAddr()+rawPadded,0.f,1,1,1,8,0);
            SetVectorMask<float,MaskMode::COUNTER>(5*P-priorPadded);
            vector_dup((__ubuf__ float*)packed.GetPhyAddr()+priorPadded,0.f,1,1,1,8,0);
            SetVectorMask<float,MaskMode::COUNTER>(5*P);
            vector_dup((__ubuf__ float*)pbuf.GetPhyAddr(),0.f,1,1,1,8,0);
            SetVectorMask<float,MaskMode::COUNTER>(3*P);
            vector_dup((__ubuf__ float*)packedBuf.template Get<float>().GetPhyAddr(),0.f,1,1,1,8,0);
        } else if constexpr(sizeof(T)==4 && BatchOne && !UniformFull) {
            // DMA owns the padded prefixes; these suffixes are disjoint.
            const uint32_t rawPadded=(oneCount*6+7)/8*8;
            const uint32_t priorPadded=(oneCount*5+7)/8*8;
            SetVectorMask<float,MaskMode::COUNTER>(6*P-rawPadded);
            vector_dup((__ubuf__ float*)packedBuf.template Get<float>().GetPhyAddr()+rawPadded,0.f,1,1,1,8,0);
            SetVectorMask<float,MaskMode::COUNTER>(5*P-priorPadded);
            vector_dup((__ubuf__ float*)packed.GetPhyAddr()+priorPadded,0.f,1,1,1,8,0);
            // P and S keep their full defined state. V is first written
            // by packed sigmoid or each original geometry/cold stage.
            SetVectorMask<float,MaskMode::COUNTER>(V0/4);
            vector_dup((__ubuf__ float*)zeroRegion.GetPhyAddr(),0.f,1,1,1,8,0);
        } else {
            SetVectorMask<float,MaskMode::COUNTER>(C0/4);
            Duplicate<float,false>(zeroRegion,0.f,MASK_PLACEHOLDER,1,1,8);
        }
        SetVectorMask<float,MaskMode::COUNTER>(64);
        Duplicate<float,false>(numeratorBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
        ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool SkipFullPadding>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::InitScratch() {
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto pbuf=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        if constexpr(BatchOne) {
            if(UniformFull || oneCount==G*Q) InitFullPadding<SkipFullPadding>();
            else InitPartialPadding();
        } else {
            // Raw/prior packed inputs are produced by MTE2 and explicit
            // tail fills. Only the disjoint prior/value/work planes need zero.
            LocalTensor<float> zeroRegion(TPosition::VECCALC,0,O0/4);
            SetMaskCount();
            if constexpr(HalfEight) {
                SetVectorMask<float,MaskMode::COUNTER>(5*P);
                Duplicate<float,false>(pbuf,0.f,MASK_PLACEHOLDER,1,1,8);
                SetVectorMask<uint16_t,MaskMode::COUNTER>(6*P);
                Duplicate<uint16_t,false>(workBuf.template Get<uint16_t>(),0,
                    MASK_PLACEHOLDER,1,1,8);
            } else {
                SetVectorMask<float,MaskMode::COUNTER>(O0/4);
                Duplicate<float,false>(zeroRegion,0.f,MASK_PLACEHOLDER,1,1,8);
            }
            SetVectorMask<float,MaskMode::COUNTER>(64);
            Duplicate<float,false>(numeratorBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
            if constexpr(DirectScoreDivide) {
                Duplicate<float,false>(scoreNumeratorBuf.template Get<float>(),1.f,MASK_PLACEHOLDER,1,1,8);
            }
            ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::InitOutputAddresses(LocalTensor<OutputShuffleT> outputPlanes,LocalTensor<OutputShuffleT> outputPacked) {
        #pragma unroll
                for(uint32_t block=0;block<5;++block) {
        #pragma unroll
                    for(uint32_t i=0;i<16;++i) {
                        if constexpr(!WidePhaseLayout) {
                            const uint32_t j=block*16+i;
                            outputSrc[block][i]=reinterpret_cast<uint64_t>(outputPlanes[(j%5)*P+(j/5)*Q].GetPhyAddr());
                            outputDst[block][i]=reinterpret_cast<uint64_t>(outputPacked[i*5*G+block*16].GetPhyAddr());
                        } else if(i<8) {
                            const uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
                            outputSrc[block][i]=reinterpret_cast<uint64_t>(outputPlanes[offset].GetPhyAddr());
                            outputSrc[block][i+8]=reinterpret_cast<uint64_t>(outputPlanes[offset+8].GetPhyAddr());
                            outputDst[block][2*i]=reinterpret_cast<uint64_t>(outputPacked[i*5*G+block*8].GetPhyAddr());
                            outputDst[block][2*i+1]=reinterpret_cast<uint64_t>(outputPacked[(i+8)*5*G+block*8].GetPhyAddr());
                        }
                    }
                }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::InitAddressTables() {
        auto pbuf=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();

                auto rawPacked=packedBuf.template Get<ShuffleT>();
                auto rawPlanes=FloatLayout?valuesBuf.template Get<ShuffleT>():workBuf.template Get<ShuffleT>();
                auto outputPlanes=WidePhaseLayout?valuesBuf.template Get<OutputShuffleT>():
                    workBuf.template Get<OutputShuffleT>();
                auto outputPacked=stablePriorOwner?priorPackedBuf.template Get<OutputShuffleT>():
                    (WidePhaseLayout?workBuf.template Get<OutputShuffleT>():valuesBuf.template Get<OutputShuffleT>());
        #pragma unroll
                for(uint32_t block=0;block<RAWBLOCKS;++block) {
        #pragma unroll
                    for(uint32_t i=0;i<16;++i) {
                        constexpr uint32_t rawWidth=32/sizeof(ShuffleT);
                        rawSrc[block][i]=reinterpret_cast<uint64_t>(rawPacked[i*6*G+block*rawWidth].GetPhyAddr());
                        if constexpr(!FloatLayout) {
                            const uint32_t j=block*16+i;
                            rawDst[block][i]=reinterpret_cast<uint64_t>(rawPlanes[(j%6)*P+(j/6)*Q].GetPhyAddr());
                        } else if(i<8) {
                            const uint32_t j=block*8+i,offset=(j%6)*P+(j/6)*Q;
                            rawDst[block][2*i]=reinterpret_cast<uint64_t>(rawPlanes[offset].GetPhyAddr());
                            rawDst[block][2*i+1]=reinterpret_cast<uint64_t>(rawPlanes[offset+8].GetPhyAddr());
                        }
                    }
                }
        #pragma unroll
                for(uint32_t block=0;block<PRBLOCKS;++block) {
        #pragma unroll
                    for(uint32_t i=0;i<16;++i) {
                        const uint32_t row=WidePhaseLayout?i:(i<8?2*i:2*(i-8)+1);
                        priorSrc[block][i]=reinterpret_cast<uint64_t>(packed[row*40+block*8].GetPhyAddr());
                        if(i<8) {
                            const uint32_t j=block*8+i,offset=(j%5)*P+(j/5)*Q;
                            priorDst[block][2*i]=reinterpret_cast<uint64_t>(pbuf[offset].GetPhyAddr());
                            priorDst[block][2*i+1]=reinterpret_cast<uint64_t>(pbuf[offset+(WidePhaseLayout?8:8*Q)].GetPhyAddr());
                        }
                    }
                }
                InitOutputAddresses(outputPlanes,outputPacked);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool SkipFullPadding>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,
    FireCandidateDecodeFusionTilingData t,GM_ADDR addressMetadata,uint32_t batchShift,uint32_t uniformTileCount,uint32_t uniformOwner) {
        static_assert(!SkipFullPadding || (BatchOne && UniformFull && K>128 && K<=1024),
            "padding elision requires a complete count-guard owner");
        d=t;
        // The C220 Vector entry already initializes the same SoC state.
        // Address metadata prefetch and scalar packet work do not consume it.
#if __NPU_ARCH__ == 2201
        if constexpr(!(K>1024 && !BatchOne)) {
            if constexpr(!BatchOne)InitSocState();
            ResetMask();SetMaskNorm();
        }
#else
        if constexpr(!BatchOne)InitSocState();
        ResetMask();SetMaskNorm();
#endif
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);
        out.SetGlobalBuffer((__gm__ T*)o);
        InitOwnership(batchShift,uniformTileCount,uniformOwner);
        StageInitialInputs();
        InitScratch<SkipFullPadding>();
        if constexpr(!BatchOne && !LargeMetadata) InitAddressTables();
        RetainHalfContexts();

    }
