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
#include "kernel_operator.h"
#include "fire_candidate_decode_fusion_tiling.h"
#include "fire_static_metadata.h"
using AscendC::Abs;
using AscendC::Add;
using AscendC::Adds;
using AscendC::CMPMODE;
using AscendC::Cast;
using AscendC::CompareScalar;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::Div;
using AscendC::Duplicate;
using AscendC::Exp;
using AscendC::GatherMask;
using AscendC::GatherMaskParams;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::InitSocState;
using AscendC::LocalTensor;
using AscendC::MASK_PLACEHOLDER;
using AscendC::MaskMode;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::Not;
using AscendC::PipeBarrier;
using AscendC::ReduceMax;
using AscendC::ReduceOrder;
using AscendC::ResetMask;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::SetMaskCount;
using AscendC::SetMaskNorm;
using AscendC::SetVectorMask;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TransDataTo5HD;
using AscendC::TransDataTo5HDParams;
using AscendC::WaitFlag;
using AscendC::WholeReduceMax;
template<uint32_t Offset,uint32_t Bytes> struct StaticUBBuffer {
    template<class U> __aicore__ inline LocalTensor<U> Get() {return LocalTensor<U>(TPosition::VECCALC,Offset,Bytes/sizeof(U));}
};
template<class T,uint32_t K> struct FireDensePitchPolicy {
    static constexpr uint32_t Value=sizeof(T)==4 && K>1024?
        ((K+127)/128)*128+24:K+(sizeof(T)==4?24:(K>1024?16:8));
};
template<class T> __aicore__ inline void CopyValidInput(LocalTensor<T> dst,GlobalTensor<T> src,uint32_t offset,uint32_t count) {
    uint32_t bytes=count*sizeof(T);uint8_t pad=static_cast<uint8_t>(((32-bytes%32)%32)/sizeof(T));
    DataCopyPad(dst,src[offset],DataCopyExtParams{1,bytes,0,0,0},DataCopyPadExtParams<T>{pad!=0,0,pad,static_cast<T>(0)});
}
template<class T> __aicore__ inline void CopyDenseInput(LocalTensor<T> dst,
    GlobalTensor<T> src,uint32_t offset,uint32_t count) {
    uint32_t bytes=count*sizeof(T);
    uint64_t address=reinterpret_cast<uint64_t>(src.GetPhyAddr())+uint64_t(offset)*sizeof(T);
    if(((address|bytes)&31u)==0) {
        DataCopy(dst,src[offset],DataCopyParams{1,static_cast<uint16_t>(bytes/32),0,0});
    } else {
        CopyValidInput(dst,src,offset,count);
    }
}
template<class T> __aicore__ inline void CopyDenseOutput(GlobalTensor<T> dst,
    LocalTensor<T> src,uint32_t offset,uint32_t count) {
    uint32_t bytes=count*sizeof(T);
    uint64_t address=reinterpret_cast<uint64_t>(dst.GetPhyAddr())+uint64_t(offset)*sizeof(T);
    if(((address|bytes)&31u)==0) {
        DataCopy(dst[offset],src,DataCopyParams{1,static_cast<uint16_t>(bytes/32),0,0});
    } else {
        DataCopyPad(dst[offset],src,DataCopyExtParams{1,bytes,0,0,0});
    }
}
