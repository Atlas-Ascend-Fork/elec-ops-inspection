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
template<bool FinalFence>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::UnpackMetadataRaw(uint8_t repeats) {
        // Both metadata reads precede the first transpose. The two
                    // matrices use disjoint VA registers and UB destinations.
        #pragma unroll
                    for(uint32_t block=0;block<6;block+=2) {
                        LoadAddressContext<false>(block);
                        LoadAddressContext<true>(block+1);
                        scatter_vnchwconv_b32(VA2,VA0,repeats,repeats==1?0:2,repeats==1?0:96);
                        scatter_vnchwconv_b32(VA6,VA4,repeats,repeats==1?0:2,repeats==1?0:96);
                    }
                    if constexpr(FinalFence)PipeBarrier<PIPE_V>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::StoreTile(LocalTensor<T> outputPacked,uint32_t tile,uint32_t b,uint32_t n,uint32_t count,uint8_t repeats,uint32_t nextCount) {
        // With a recycled prior slot, all Cast and arithmetic are
        // disjoint from MTE3. Wait only at the actual output overwrite.
        if(tile!=beginTile && stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        PackOutput(repeats);
        if constexpr(HalfEight) {
            // O[0,3P) float words belongs to next raw DMA. The disjoint
            // O[3P,6P) slot belongs to the current/previous Half output.
            if(tile!=beginTile)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            auto floatOutput=stablePriorOwner?priorPackedBuf.template Get<float>():
                workBuf.template Get<float>();
            auto halfOutput=packedBuf.template Get<T>()[6*P];
            Cast(halfOutput,floatOutput,RoundMode::CAST_RINT,count*5);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyDenseOutput(out,halfOutput,(b*d.n+n)*5,count*5);
        } else if constexpr(sizeof(T)==2 && BatchOne) {
            Cast(workBuf.template Get<T>(),priorPackedBuf.template Get<float>(),
                RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
        } else {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyDenseOutput(out,outputPacked,(b*d.n+n)*5,count*5);
        }
        if(nextCount)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool Hot>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::UnpackCurrentRaw(uint32_t tile,uint32_t count,uint8_t repeats) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    // Input transpose is disjoint from the previous output: FP32
                    // writes S while MTE3 reads V, FP16 writes V while MTE3 reads S.
                    if constexpr(BatchOne && !NativeHalfPlanes)EvaluatePackedRaw(count);
        #if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
                    // A live raw successor waits on V_MTE2 before reusing O; the
                    // following mask transition joins S before its first math read.
                    UnpackRaw<!(Hot && !BatchOne && !HalfEight && FireInputFloatCompletionJoins)>(repeats);
        #else
                    UnpackRaw(repeats);
        #endif
                    if constexpr(NativeHalfPlanes)EvaluateFullHalfPlanes();
                    if(tile!=beginTile && !stablePriorOwner && !HalfEight)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool Hot>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::PrepareNextTile(uint32_t tile,uint32_t b,uint32_t n,uint32_t& nextB,uint32_t& nextN,uint32_t& nextCount) {
        nextB=b+1;nextN=n;nextCount=0;
                        if(nextB==d.batch){nextB=0;nextN+=K;}
                        if(tile+1<endTile) {
                            nextCount=d.n-nextN<K?d.n-nextN:K;
                            if constexpr(!Hot) if(nextN!=n)StagePrior(nextN,nextCount);
        #if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
                            // Every live raw successor keeps an explicit completion
                            // join. The terminal tile joins at its first math mask.
                            if constexpr(Hot && !BatchOne && !HalfEight && FireInputFloatCompletionJoins)PipeBarrier<PIPE_V>();
        #endif
                            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                            LoadRaw(nextB*d.n+nextN,nextCount);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool Hot>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::RepairCurrentTile(uint32_t b,uint32_t n,uint32_t count,bool& needsRepair) {
        auto packed=priorPackedBuf.template Get<float>();
        if constexpr(BatchOne) {
            needsRepair=FinishOneStrideMax(packed,G*Q)>(sizeof(T)==2?4096.f:32.f);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        }
        if constexpr(!Hot) if(__builtin_expect(needsRepair,false)) {
            if constexpr(ScoreGeometryCounterCarry) {ResetMask();SetMaskNorm();}
            RepairBlockedCenters<T,K,P,G,Q,S0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
            // No assumption about VA preservation across a cold call.
            RetainHalfContexts();
            if constexpr(ScoreGeometryCounterCarry)SetMaskCount();
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::StoreDualTile(uint32_t tile,uint32_t b,uint32_t n,uint32_t count,uint8_t repeats) {
        const uint32_t ordinal=tile-beginTile;
        const bool second=(ordinal&1U)!=0;
        // Completion is requested only for an actual future reuse.
        if(ordinal>=2U) {
            if(second)WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            else WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        // Constant context calls keep VA addresses out of the loop.
        if(second)PackOutputWithContext<16>(repeats);
        else PackOutputWithContext<11>(repeats);
        if(second) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            CopyDenseOutput(out,workBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            if(tile+2U<endTile)SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        } else {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyDenseOutput(out,priorPackedBuf.template Get<T>(),(b*d.n+n)*5,count*5);
            if(tile+2U<endTile)SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
template<bool Hot>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::ProcessKnownOwner(uint32_t cachedN,bool needsRepair,bool shiftGrid) {
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        auto outputPacked=OutputPacked();
        uint32_t b=BatchOne?oneBatch:beginTile%d.batch;
        uint32_t n=BatchOne?oneN:beginTile/d.batch*K;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            const uint32_t count=BatchOne?(UniformFull?G*Q:oneCount):(d.n-n<K?d.n-n:K);
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));
            if constexpr(!Hot) {
            if(n!=cachedN) {
                RefreshPrior(n,repeats,cachedN,needsRepair,shiftGrid);
            }
            }
            UnpackCurrentRaw<Hot>(tile,count,repeats);
            uint32_t nextB,nextN,nextCount;
            if constexpr(HalfEight) PrepareNextTile<Hot>(tile,b,n,nextB,nextN,nextCount);
            ConvertRawHalf(v);
            if constexpr(!HalfEight) PrepareNextTile<Hot>(tile,b,n,nextB,nextN,nextCount);
            if constexpr(DualOutput && Hot) EvaluateSigmoid<true>(s,s);
            else EvaluateSigmoid<false>(s,v);
            RepairCurrentTile<Hot>(b,n,count,needsRepair);
            if constexpr(DualOutput && Hot) DecodeGeometry<ScoreGeometryCounterCarry>(s,s,p,shiftGrid);
            else DecodeGeometry<ScoreGeometryCounterCarry>(s,v,p,shiftGrid);
            if constexpr(sizeof(T)==2 && !BatchOne && !HalfEight) {
                Cast(workBuf.template Get<T>(),s,RoundMode::CAST_RINT,5*P);PipeBarrier<PIPE_V>();
            }
            if constexpr(DualOutput && Hot) StoreDualTile(tile,b,n,count,repeats);
            else StoreTile(outputPacked,tile,b,n,count,repeats,nextCount);
            b=nextB;n=nextN;
        }
        PipeBarrier<PIPE_ALL>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::ProcessOwnerSelected() {
        uint32_t cachedN=d.n;
        bool needsRepair=true,shiftGrid=false;
        if(stablePriorOwner) {
            auto p=priorsBuf.template Get<float>();
            auto v=workBuf.template Get<float>();
            auto packed=priorPackedBuf.template Get<float>();
            const uint32_t n=beginTile/d.batch*K;
            const uint32_t count=d.n-n<K?d.n-n:K;
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));

                RefreshPrior(n,repeats,cachedN,needsRepair,shiftGrid);

            if(!needsRepair) {
                ProcessKnownOwner<true>(cachedN,false,shiftGrid);
                return;
            }
        }
        ProcessKnownOwner<false>(cachedN,needsRepair,shiftGrid);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::Process() {
        if constexpr(DualOutput) ProcessOwnerSelected();
        else ProcessKnownOwner<false>(d.n,true,false);
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline __attribute__((always_inline)) void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::ProcessGuardCount() {
        static_assert(BatchOne,"single-tile owner required");
        auto s=valuesBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorsBuf.template Get<float>();auto packed=priorPackedBuf.template Get<float>();
        auto outputPacked=OutputPacked();
        uint32_t b=oneBatch,n=oneN,cachedN=d.n;
        bool needsRepair=true,shiftGrid=false;
        uint64_t retained=0;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            const uint32_t count=UniformFull?G*Q:oneCount;
            const uint8_t repeats=static_cast<uint8_t>((count+16*G-1)/(16*G));
            if(n!=cachedN) RefreshGuardPrior(n,repeats,cachedN,retained);
            UnpackCurrentRaw<false>(tile,count,repeats);
            uint32_t nextB,nextN,nextCount;
            PrepareNextTile<false>(tile,b,n,nextB,nextN,nextCount);
            needsRepair=false;
            if(__builtin_expect(retained!=G*Q,false))
                needsRepair=DirectStrideMax(p[2*P],packed,G*Q)>(sizeof(T)==2?4096.f:32.f);
            if(__builtin_expect(needsRepair,false)) {
                RepairBlockedCenters<T,K,P,G,Q,S0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                RetainHalfContexts();
            }
            DecodeGeometry<false>(s,v,p,shiftGrid);
            StoreTile(outputPacked,tile,b,n,count,repeats,nextCount);
            b=nextB;n=nextN;
        }
        PipeBarrier<PIPE_ALL>();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::ProcessFullGuardCount() {
        static_assert(BatchOne && UniformFull,"complete owner required");
        ProcessGuardCount();
    }

template<class T,uint32_t K,bool BatchOne,bool UniformFull,bool DualOutput,bool UniformTiles>
__aicore__ inline void BlockedTransposeDecode<T,K,BatchOne,UniformFull,DualOutput,UniformTiles>::ProcessPartialGuardCount() {
        static_assert(BatchOne && !UniformFull,"partial owner required");
        ProcessGuardCount();
    }
