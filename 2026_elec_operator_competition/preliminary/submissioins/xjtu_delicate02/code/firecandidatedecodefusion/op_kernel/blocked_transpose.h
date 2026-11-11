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
#include <type_traits>

// This is the input type of the compiled binary, independent of each TPL
// instantiation. Keep the complete Half image on its original schedule.
#if defined(DTYPE_RAW_PRED) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
static constexpr bool FireInputFloatCompletionJoins = sizeof(DTYPE_RAW_PRED) == 4;
#else
static constexpr bool FireInputFloatCompletionJoins = false;
#endif

template<class T,uint32_t K,uint32_t P,uint32_t G,uint32_t Q,uint32_t S0,uint32_t V0>
// Preserve the next cold resource's measured instruction placement after
// removing manual padding from the preceding shared fallback.
__aicore__ __attribute__((noinline, aligned(
    FireInputFloatCompletionJoins && sizeof(T) == 4 && K == 1664 ? 128 : 4)))
void RepairBlockedCenters(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset) {
    LocalTensor<float> values(TPosition::VECCALC,S0,6*P);
    LocalTensor<float> priors(TPosition::VECCALC,0,5*P);
    LocalTensor<float> scratch(TPosition::VECCALC,V0,6*P);
    GlobalTensor<T> raw;raw.SetGlobalBuffer((__gm__ T*)rawAddress);
    for(uint32_t channel=0;channel<2;++channel) {
        ResetMask();SetMaskNorm();
        Adds(scratch,values[channel*P],-.5f,P);PipeBarrier<PIPE_V>();
        Add(scratch,scratch,priors[channel*P],P);PipeBarrier<PIPE_V>();
        Abs(scratch,scratch,P);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for(uint32_t j=0;j<G*Q;++j) {
            const uint32_t row=(j%Q)*G+j/Q;
            if(row>=count || scratch.GetValue(j)>=0x1p-4f ||
                priors.GetValue(2*P+j)<=(sizeof(T)==2?4096.f:32.f))continue;
            const float original=static_cast<float>(raw.GetValue(uint64_t(rawOffset)+row*6+channel));
            if(original>-20.f && original<0.f) {
                union {float f;uint32_t u;} x;x.f=-original;
                const uint32_t expBits=FireSoft53::evaluate(x.u);
                const uint32_t denominator=FireSoft53::narrow32(FireSoft53::add53(
                    FireSoft53::from32(expBits),FireSoft53::integer(1)));
                values.template ReinterpretCast<uint32_t>().SetValue(channel*P+j,
                    CenterReciprocalBits(denominator)+0x800000u);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
}

// A matrix row contains G adjacent candidates: its raw/prior/output widths
// are whole32-byte blocks. Transpose each block column into the same phase
// permutation in every channel; arithmetic never mixes different candidates.
template<class T,uint32_t K,bool BatchOne=false,bool UniformFull=false,bool DualOutput=false,bool UniformTiles=false> class BlockedTransposeDecode {
    static constexpr bool LargeMetadata=!BatchOne && K>1024;
    static constexpr bool NativeHalfPlanes=sizeof(T)==2 && BatchOne;
    static constexpr bool FloatLayout=sizeof(T)==4 || BatchOne;
    static constexpr bool HalfEight=sizeof(T)==2 && !BatchOne && K>1024;
    static constexpr bool WidePhaseLayout=FloatLayout || HalfEight;
    using ShuffleT=std::conditional_t<FloatLayout,float,T>;
    using OutputShuffleT=std::conditional_t<WidePhaseLayout,float,T>;
    static constexpr uint32_t RAWBLOCKS=HalfEight?3:6;
    static constexpr uint32_t RAW_SOURCE_STRIDE=HalfEight?48:96;
    static constexpr uint32_t G=WidePhaseLayout?8:16;
    static constexpr uint32_t Q=(K+16*G-1)/(16*G)*16;
    static constexpr uint32_t P=G*Q+(NativeHalfPlanes?16:(FloatLayout?24:16));
    static constexpr uint32_t S0=5*P*4+128+(HalfEight?320:0),V0=S0+6*P*4+128;
    static constexpr uint32_t O0=V0+6*P*4+128;
    static constexpr uint32_t RAW_WORDS=(WidePhaseLayout?6:5)*P;
    static constexpr uint32_t F0=O0+RAW_WORDS*4+128,C0=F0+5*P*4+128;
    static_assert(C0+64*4<=196352,"blocked transpose exceeds usable UB");
    static constexpr uint32_t A0=C0+64*4;
    static constexpr bool DirectScoreDivide=sizeof(T)==4 && !BatchOne && K>1024;
#if __NPU_ARCH__ == 2201
    static constexpr bool ScoreGeometryCounterCarry=(DirectScoreDivide && !DualOutput) || HalfEight;
#else
    static constexpr bool ScoreGeometryCounterCarry=false;
#endif
    static_assert(!DualOutput || DirectScoreDivide,"dual output requires large Float resources");
    StaticUBBuffer<A0,(DirectScoreDivide?64*4:0)> scoreNumeratorBuf;
    static_assert(!DirectScoreDivide || A0+64*4<=196352,"score numerator exceeds UB");
    static constexpr uint32_t LargeContexts=HalfEight?18:21;
    static constexpr uint32_t MetadataBase=A0+(DirectScoreDivide?256:0);
    static_assert((!BatchOne && !LargeMetadata) || MetadataBase+(LargeMetadata?LargeContexts*128:16*128)<=196352,"VA metadata exceeds UB");
    StaticUBBuffer<MetadataBase,(LargeMetadata?LargeContexts*128:(BatchOne?16*128:0))> addressBuf;
    static constexpr uint32_t PRBLOCKS=5;
    StaticUBBuffer<0,5*P*4> priorsBuf;
    StaticUBBuffer<S0,6*P*4> valuesBuf;
    StaticUBBuffer<V0,6*P*4> workBuf;
    StaticUBBuffer<O0,RAW_WORDS*4> packedBuf;
    StaticUBBuffer<F0,5*P*4> priorPackedBuf;
    StaticUBBuffer<C0,64*4> numeratorBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint32_t beginTile,endTile;
    uint32_t oneN=0,oneCount=0,oneBatch=0;
    bool stablePriorOwner;
    uint64_t rawSrc[6][16],rawDst[6][16];
    uint64_t priorSrc[PRBLOCKS][16],priorDst[PRBLOCKS][16];
    uint64_t outputSrc[5][16],outputDst[5][16];

    template<bool Second=false>
    __aicore__ inline void LoadAddressContext(uint32_t index);
    __aicore__ inline void StagePrior(uint32_t n,uint32_t count);
    __aicore__ inline void LoadRaw(uint32_t row,uint32_t count);
    __aicore__ inline void RetainHalfContexts();
    __aicore__ inline void EvaluatePackedRaw(uint32_t count);
    __aicore__ inline void EvaluateFullHalfPlanes();
    __aicore__ inline __attribute__((always_inline)) void UnpackRawStatic(uint8_t repeats);
    __aicore__ inline __attribute__((always_inline)) void UnpackRawFallback(uint8_t repeats);
    template<bool FinalFence>
    __aicore__ inline __attribute__((always_inline)) void UnpackMetadataRaw(uint8_t repeats);
    template<bool FinalFence=true>
    __aicore__ inline void UnpackRaw(uint8_t repeats);
    __aicore__ inline void UnpackPriors(uint8_t repeats);
    __aicore__ inline __attribute__((always_inline)) void PackMetadataOutput(uint32_t Context,uint8_t repeats);
    template<uint32_t Context>
    __aicore__ inline __attribute__((always_inline)) void PackOutputWithContext(uint8_t repeats);
    __aicore__ inline __attribute__((always_inline)) void PackOutputStatic(uint8_t repeats);
    __aicore__ inline __attribute__((always_inline)) void PackOutputFallback(uint8_t repeats);
    __aicore__ inline void PackOutput(uint8_t repeats);
public:
    __aicore__ inline static void PrefetchAddressMetadata(GM_ADDR addressMetadata);
    __aicore__ inline __attribute__((always_inline)) void InitOwnership(uint32_t batchShift,uint32_t uniformTileCount,uint32_t uniformOwner);
    __aicore__ inline __attribute__((always_inline)) void StageInitialInputs();
    template<bool SkipFullPadding>
    __aicore__ inline __attribute__((always_inline)) void InitFullPadding();
    __aicore__ inline __attribute__((always_inline)) void InitPartialPadding();
    template<bool SkipFullPadding>
    __aicore__ inline __attribute__((always_inline)) void InitScratch();
    __aicore__ inline __attribute__((always_inline)) void InitOutputAddresses(LocalTensor<OutputShuffleT> outputPlanes,LocalTensor<OutputShuffleT> outputPacked);
    __aicore__ inline __attribute__((always_inline)) void InitAddressTables();
    template<bool SkipFullPadding=false>
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,
        FireCandidateDecodeFusionTilingData t,GM_ADDR addressMetadata=nullptr,uint32_t batchShift=0,uint32_t uniformTileCount=0,uint32_t uniformOwner=0);
    __aicore__ inline __attribute__((always_inline)) bool JointRangeBitmapAllLE();
    template<bool CarryCounter>
    __aicore__ inline __attribute__((always_inline)) void DecodeGeometry(LocalTensor<float> s,LocalTensor<float> v,LocalTensor<float> p,bool shiftGrid);
    __aicore__ inline __attribute__((always_inline)) void StoreTile(LocalTensor<T> outputPacked,uint32_t tile,uint32_t b,uint32_t n,uint32_t count,uint8_t repeats,uint32_t nextCount);
    template<bool InPlace>
    __aicore__ inline __attribute__((always_inline)) void EvaluateSigmoid(LocalTensor<float> s,LocalTensor<float> work);
    __aicore__ inline __attribute__((always_inline)) void RefreshPrior(uint32_t n,uint8_t repeats,uint32_t& cachedN,bool& needsRepair,bool& shiftGrid);
    template<bool Hot>
    __aicore__ inline __attribute__((always_inline)) void UnpackCurrentRaw(uint32_t tile,uint32_t count,uint8_t repeats);
    template<bool Hot>
    __aicore__ inline __attribute__((always_inline)) void PrepareNextTile(uint32_t tile,uint32_t b,uint32_t n,uint32_t& nextB,uint32_t& nextN,uint32_t& nextCount);
    __aicore__ inline __attribute__((always_inline)) void ConvertRawHalf(LocalTensor<float> v);
    template<bool Hot>
    __aicore__ inline __attribute__((always_inline)) void RepairCurrentTile(uint32_t b,uint32_t n,uint32_t count,bool& needsRepair);
    __aicore__ inline __attribute__((always_inline)) void StoreDualTile(uint32_t tile,uint32_t b,uint32_t n,uint32_t count,uint8_t repeats);
    __aicore__ inline __attribute__((always_inline)) void RefreshGuardPrior(uint32_t n,uint8_t repeats,uint32_t& cachedN,uint64_t& retained);
    __aicore__ inline __attribute__((always_inline)) LocalTensor<T> OutputPacked();
    template<bool Hot>
    __aicore__ inline void ProcessKnownOwner(uint32_t cachedN,bool needsRepair,bool shiftGrid);
    __aicore__ inline void ProcessOwnerSelected();
    __aicore__ inline void Process();
    __aicore__ inline __attribute__((always_inline)) void ProcessGuardCount();
    __aicore__ inline void ProcessFullGuardCount();
    __aicore__ inline void ProcessPartialGuardCount();
};

#include "blocked_shuffle.h"

#include "blocked_init.h"

#include "blocked_math.h"

#include "blocked_pipeline.h"
