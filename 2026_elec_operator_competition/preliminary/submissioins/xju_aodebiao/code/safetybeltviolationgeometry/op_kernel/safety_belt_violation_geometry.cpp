/*
 * Copyright (c) 2026 aodebiao.
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
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"
using namespace AscendC;

template <typename T, bool HAS_THRESHOLD, bool THRESHOLD_HALF, bool OUTPUT_HALF, bool SMALL>
class VectorSafetyBelt {
    static constexpr uint32_t Tile = SMALL ? 64U : 128U;
    static constexpr uint32_t OutputByte = Tile*6U*sizeof(T);
    static constexpr uint32_t ExtractByte = OutputByte+Tile*4U;
    static constexpr uint32_t WorkByte = ExtractByte+Tile*6U;
    static constexpr uint32_t MaskByte = WorkByte+Tile*16U;
    static constexpr uint32_t ThresholdByte = MaskByte+32U;
    static constexpr uint32_t ThresholdFloatByte = ThresholdByte+32U;
    static constexpr uint32_t ThresholdBroadcastByte = ThresholdFloatByte+32U;
public:
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR anchors, GM_ADDR threshold,
                               GM_ADDR output, const SafetyBeltViolationGeometryTilingData& t)
    {
        tiling_=t;
        boxes_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(boxes),t.batch*4U);
        anchors_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchors),t.batch*2U);
        output32_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output),t.batch);
        output16_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output),t.batch);
        if constexpr (HAS_THRESHOLD) {
            if constexpr (THRESHOLD_HALF)
                threshold16_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(threshold),1);
            else threshold32_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(threshold),1);
        }
        TBuffAddr address{};
        address.logicPos=static_cast<uint8_t>(TPosition::VECCALC);
        address.bufferAddr=0U;
        address.dataLen=ThresholdBroadcastByte+256U;
        arena_.SetAddr(address);
        Duplicate(arena_.ReinterpretCast<float>()[WorkByte/4U][Tile*3U],1.0F,Tile);
    }
    __aicore__ inline void Process()
    {
        const uint32_t block=static_cast<uint32_t>(GetBlockIdx());
        const uint32_t blocks=static_cast<uint32_t>(GetBlockNum());
        LoadThreshold();
        for (uint32_t start=block*Tile;start<tiling_.batch;start+=blocks*Tile) {
            const uint32_t count=tiling_.batch-start<Tile ? tiling_.batch-start : Tile;
            const uint32_t aligned=(count+63U)/64U*64U;
            LoadTile(start,count);
            Sync<HardEvent::MTE2_V>();
            PrepareThreshold();
            ComputeTile(aligned);
            StoreTile(start,count,aligned);
            if (start+blocks*Tile<tiling_.batch) Sync<HardEvent::MTE3_V>();
        }
    }
private:
    __aicore__ inline void LoadThreshold()
    {
        if constexpr (HAS_THRESHOLD) {
            if constexpr (THRESHOLD_HALF) {
                DataCopyExtParams copy{1,2U,0,0,0};
                DataCopyPadExtParams<half> pad{true,0,0,static_cast<half>(0)};
                DataCopyPad(arena_.ReinterpretCast<half>()[ThresholdByte/2U],threshold16_,copy,pad);
            } else {
                DataCopyExtParams copy{1,4U,0,0,0};
                DataCopyPadExtParams<float> pad{true,0,0,0.0F};
                DataCopyPad(arena_.ReinterpretCast<float>()[ThresholdByte/4U],threshold32_,copy,pad);
            }
        }
    }
    __aicore__ inline void LoadTile(uint32_t start,uint32_t count)
    {
        auto rawBoxes=arena_.ReinterpretCast<T>();
        auto rawAnchors=arena_.ReinterpretCast<T>()[Tile*4U];
        DataCopyExtParams copyBoxes{1,static_cast<uint32_t>(count*4U*sizeof(T)),0,0,0};
        DataCopyExtParams copyAnchors{1,static_cast<uint32_t>(count*2U*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{true,0,0,static_cast<T>(0)};
        DataCopyPad(rawBoxes,boxes_[start*4U],copyBoxes,pad);
        DataCopyPad(rawAnchors,anchors_[start*2U],copyAnchors,pad);
    }
    __aicore__ inline void PrepareThreshold()
    {
        if (!HAS_THRESHOLD || thresholdReady_) return;
        auto value=arena_.ReinterpretCast<float>()[ThresholdByte/4U];
        if constexpr (THRESHOLD_HALF) {
            value=arena_.ReinterpretCast<float>()[ThresholdFloatByte/4U];
            Cast(value,arena_.ReinterpretCast<half>()[ThresholdByte/2U],RoundMode::CAST_NONE,8U);
        }
        Brcb(arena_.ReinterpretCast<float>()[ThresholdBroadcastByte/4U],value,1,{1,8});
        thresholdReady_=true;
    }
    __aicore__ inline void ComputeTile(uint32_t aligned)
    {
        const uint8_t repeats=static_cast<uint8_t>(aligned/64U);
        auto y1=arena_.ReinterpretCast<float>()[WorkByte/4U];auto y2=y1[Tile];
        auto cy=y1[Tile*2U];auto ones=y1[Tile*3U];
        Extract(y1,y2,cy,arena_.ReinterpretCast<T>(),arena_.ReinterpretCast<T>()[Tile*4U],aligned);
        SetVectorMask<float>(64U);
        Sub<float,false>(y2,y2,y1,MASK_PLACEHOLDER,repeats,{1,1,1,8,8,8});
        Muls<float,false>(y2,y2,tiling_.waistRatio,MASK_PLACEHOLDER,repeats,{1,1,8,8});
        Add<float,false>(y2,y2,y1,MASK_PLACEHOLDER,repeats,{1,1,1,8,8,8});
        if constexpr (HAS_THRESHOLD)
            Add<float,false>(y2,y2,arena_.ReinterpretCast<float>()[ThresholdBroadcastByte/4U],
                MASK_PLACEHOLDER,repeats,{1,1,0,8,8,0});
        Compare(arena_.ReinterpretCast<uint8_t>()[MaskByte],cy,y2,CMPMODE::GT,aligned);
        Select(arena_.ReinterpretCast<float>()[OutputByte/4U],arena_.ReinterpretCast<uint8_t>()[MaskByte],
               ones,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,aligned);
    }
    __aicore__ inline void StoreTile(uint32_t start,uint32_t count,uint32_t aligned)
    {
        auto output=arena_.ReinterpretCast<float>()[OutputByte/4U];
        if constexpr (OUTPUT_HALF) {
            auto converted=output.template ReinterpretCast<half>();
            Cast(converted,output,RoundMode::CAST_RINT,aligned);
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams store{1,count*2U,0,0,0};
            DataCopyPad(output16_[start],converted,store);
        } else {
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams store{1,count*4U,0,0,0};
            DataCopyPad(output32_[start],output,store);
        }
    }
    __aicore__ inline void Extract(LocalTensor<float> y1,LocalTensor<float> y2,
        LocalTensor<float> cy,LocalTensor<float> boxes,LocalTensor<float> points,uint32_t aligned)
    {
        uint64_t retained=0;
        GatherMask(y1,boxes,uint8_t(4),false,0,{1,static_cast<uint8_t>(aligned/16U),8,0},retained);
        GatherMask(y2,boxes,uint8_t(6),false,0,{1,static_cast<uint8_t>(aligned/16U),8,0},retained);
        GatherMask(cy,points,uint8_t(2),false,0,{1,static_cast<uint8_t>(aligned/32U),8,0},retained);
    }
    __aicore__ inline void Extract(LocalTensor<float> y1,LocalTensor<float> y2,
        LocalTensor<float> cy,LocalTensor<half> boxes,LocalTensor<half> points,uint32_t aligned)
    {
        auto h1=arena_.ReinterpretCast<half>()[ExtractByte/2U];auto h2=h1[Tile];auto hc=h1[Tile*2U];
        uint64_t retained=0;
        GatherMask(h1,boxes,uint8_t(4),false,0,{1,static_cast<uint8_t>(aligned/32U),8,0},retained);
        GatherMask(h2,boxes,uint8_t(6),false,0,{1,static_cast<uint8_t>(aligned/32U),8,0},retained);
        GatherMask(hc,points,uint8_t(2),false,0,{1,static_cast<uint8_t>(aligned/64U),8,0},retained);
        Cast(y1,h1,RoundMode::CAST_NONE,aligned);
        Cast(y2,h2,RoundMode::CAST_NONE,aligned);
        Cast(cy,hc,RoundMode::CAST_NONE,aligned);
    }
    template<HardEvent Event>
    __aicore__ inline void Sync() {
        SetFlag<Event>(EVENT_ID0);WaitFlag<Event>(EVENT_ID0);
    }
    LocalTensor<uint8_t> arena_;
    GlobalTensor<T> boxes_,anchors_;
    GlobalTensor<float> output32_;
    GlobalTensor<half> output16_;
    SafetyBeltViolationGeometryTilingData tiling_;
    GlobalTensor<float> threshold32_;
    GlobalTensor<half> threshold16_;
    bool thresholdReady_=false;
};

// =================================================================
// Kernel entry
// =================================================================
template <typename DT_PERSON_BOXES, bool HAS_THRESHOLD, bool THRESHOLD_HALF, bool OUTPUT_HALF, bool SMALL>
__global__ __aicore__
void safety_belt_violation_geometry(
    GM_ADDR person_boxes,
    GM_ADDR anchor_points,
    GM_ADDR threshold,
    GM_ADDR violation_score,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        SafetyBeltViolationGeometryTilingData
    );

    GET_TILING_DATA_WITH_STRUCT(
        SafetyBeltViolationGeometryTilingData,
        tiling_data,
        tiling
    );

    VectorSafetyBelt<DT_PERSON_BOXES,HAS_THRESHOLD,THRESHOLD_HALF,OUTPUT_HALF,SMALL> op;
    op.Init(person_boxes,anchor_points,threshold,violation_score,tiling_data);
    op.Process();
}
