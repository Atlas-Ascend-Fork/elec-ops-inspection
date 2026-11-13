/*
 * Copyright (c) 2026 jnu_operators-selling.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TilingKey模板定义的头文件
#ifndef SWI_GLU_TEMPLATE_KEYS_H
#define SWI_GLU_TEMPLATE_KEYS_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SwiGlu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_DECL(CALC_MODE, 3, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7),
    ASCENDC_TPL_UINT_DECL(DIM_TAG, 5, ASCENDC_TPL_UI_LIST,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
    ASCENDC_TPL_UINT_DECL(TAIL_CODE, 4, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ASCENDC_TPL_UINT_DECL(PAD_BLOCKS, 4, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    ASCENDC_TPL_UINT_DECL(ROW_WAVES, 4, ASCENDC_TPL_UI_LIST, 0, 4, 5, 6, 7, 8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 3),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 18),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 10, 11),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 4, 5, 6, 7, 8),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 3),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 18),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0, 8, 9, 10, 11),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 6),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 14),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 3),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 14, 17, 19, 20, 23, 26, 27, 28),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 29),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 0, 3),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 0, 25),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 4, 5),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 28, 30, 31),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 3),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 21, 22, 29),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 0, 7),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 0, 3, 7, 16, 19, 24),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 6),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST,
            0, 2, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE, ASCENDC_TPL_UI_LIST, 7),
        ASCENDC_TPL_UINT_SEL(DIM_TAG, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X,C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(DIM_TAG,ASCENDC_TPL_UI_LIST,24),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X,C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE,ASCENDC_TPL_UI_LIST,6),
        ASCENDC_TPL_UINT_SEL(DIM_TAG,ASCENDC_TPL_UI_LIST,24),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE,ASCENDC_TPL_UI_LIST,2,3,4,5,6,7,8),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X,C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(DIM_TAG,ASCENDC_TPL_UI_LIST,25,26,27),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X,C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DIM_TAG,ASCENDC_TPL_UI_LIST,10),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X,C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(CALC_MODE,ASCENDC_TPL_UI_LIST,6),
        ASCENDC_TPL_UINT_SEL(DIM_TAG,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(TAIL_CODE,ASCENDC_TPL_UI_LIST,2,3,4,5,6,7,8),
        ASCENDC_TPL_UINT_SEL(PAD_BLOCKS,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(ROW_WAVES,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
);

#endif  // SWI_GLU_TEMPLATE_KEYS_H

// Resolve tiling keys after the kernel implementations have been declared.
#if defined(SWIGLU_DEVICE_DISPATCH) && !defined(SWIGLU_DEVICE_DISPATCH_DEFINED)
#define SWIGLU_DEVICE_DISPATCH_DEFINED
template<typename T>
__aicore__ inline void GatherSevenRows13Paired(GM_ADDR ix,GM_ADDR iw1,GM_ADDR ib1,
    GM_ADDR iw2,GM_ADDR ib2,GM_ADDR oy) {
    using namespace AscendC;
    constexpr uint32_t R=7,D=13,P=16,C=16,WC=512;
    const uint32_t row=AscendC::GetBlockIdx()*R;
    AscendC::GlobalTensor<T> x,w1,w2,b1,b2,y;
    BindSwiGluTensors(x, w1, w2, b1, b2, y, ix, iw1, iw2, ib1, ib2, oy);
    AscendC::LocalTensor<T> input(AscendC::TPosition::VECCALC,0,WC+R*P);
    AscendC::LocalTensor<T> bias(AscendC::TPosition::VECCALC,8192,32);
    AscendC::LocalTensor<float> weights(AscendC::TPosition::VECCALC,16384,WC);
    AscendC::LocalTensor<float> expanded(AscendC::TPosition::VECCALC,28672,R*64);
    AscendC::LocalTensor<float> products(AscendC::TPosition::VECCALC,32768,R*WC);
    AscendC::LocalTensor<float> blockSums(AscendC::TPosition::VECCALC,65536,R*64);
    AscendC::LocalTensor<float> sums(AscendC::TPosition::VECCALC,98304,R*32);
    AscendC::LocalTensor<T> pair(AscendC::TPosition::VECCALC,114688,R*32);
    AscendC::LocalTensor<T> out(AscendC::TPosition::VECCALC,114752+2*R*P*2,R*P);
    AscendC::LocalTensor<T> packed(AscendC::TPosition::VECCALC,131072,R*P);
    AscendC::LocalTensor<uint16_t> mask(AscendC::TPosition::VECCALC,163840,16);
    AscendC::LocalTensor<T> raw(AscendC::TPosition::VECCALC,131072,2*208+R*P);
    AscendC::LocalTensor<uint32_t> offsets(AscendC::TPosition::VECCALC,163840,C*P);
    AscendC::DataCopyPadExtParams<T> noPad{false,0,0,0};
    const bool reverse=CopyOrderedTensorPair(raw,208,w1,w2,D*D,noPad);
    AscendC::DataCopyPad(bias[reverse?P:0],b1,AscendC::DataCopyExtParams{1,D*2,0,0,0},noPad);
    AscendC::DataCopyPad(bias[reverse?0:P],b2,AscendC::DataCopyExtParams{1,D*2,0,0,0},noPad);
    AscendC::DataCopyPad(raw[416],x[row*D],AscendC::DataCopyExtParams{1,R*D*2,0,0,0},noPad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    PrepareSevenRowOffsets<T>(offsets);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Gather(input,raw,offsets,0,D*P);
    AscendC::Gather(input[C*P],raw[208],offsets,0,D*P);
    AscendC::Gather(input[WC],raw[416],offsets,0,R*P);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSevenRows(input, weights, expanded, products, blockSums, sums);
    AscendC::Cast(pair,sums,AscendC::RoundMode::CAST_RINT,32,R,AscendC::UnaryRepeatParams{1,1,2,4});
    AscendC::PipeBarrier<PIPE_V>();AscendC::Add(pair,pair,bias,32,R,AscendC::BinaryRepeatParams{1,1,1,2,2,0});
    AscendC::PipeBarrier<PIPE_V>();AscendC::Duplicate(mask,static_cast<uint16_t>(0x1fff),16);
    AscendC::SetVectorMask<T>(R*P);
    auto first=pair[reverse?P:0];auto second=pair[reverse?0:P];
    AscendC::Muls<T,false>(out,first,static_cast<T>(-1),AscendC::MASK_PLACEHOLDER,1,AscendC::UnaryRepeatParams{1,2,8,16});
    AscendC::PipeBarrier<PIPE_V>();AscendC::Exp<T,false>(out,out,AscendC::MASK_PLACEHOLDER,1,AscendC::UnaryRepeatParams{1,1,8,8});
    AscendC::PipeBarrier<PIPE_V>();AscendC::Adds<T,false>(out,out,static_cast<T>(1),AscendC::MASK_PLACEHOLDER,1,AscendC::UnaryRepeatParams{1,1,8,8});
    AscendC::PipeBarrier<PIPE_V>();AscendC::Div<T,false>(out,first,out,AscendC::MASK_PLACEHOLDER,1,AscendC::BinaryRepeatParams{1,2,1,8,16,8});
    AscendC::PipeBarrier<PIPE_V>();AscendC::Mul<T,false>(out,out,second,AscendC::MASK_PLACEHOLDER,1,AscendC::BinaryRepeatParams{1,1,2,8,8,16});
    AscendC::PipeBarrier<PIPE_V>();uint64_t selected=0;
    AscendC::GatherMask(packed,out,mask,true,R*P,AscendC::GatherMaskParams{1,1,8,0},selected);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopyPad(y[row*D],packed,AscendC::DataCopyExtParams{1,R*D*2,0,0,0});
}


template <typename DT_INPUT_X, uint32_t CALC_MODE, uint32_t DIM_TAG, uint32_t TAIL_CODE, uint32_t PAD_BLOCKS, uint32_t ROW_WAVES>
class SwiGluDispatch {
public:
        __aicore__ SwiGluDispatch(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR ws, GM_ADDR td)
        : input_x(x), weight_1(w1), bias_1(b1), weight_2(w2), bias_2(b2), output(y), workspace(ws), tiling(td) {}
    __aicore__ inline void Run() {
        InitializeMask();
        if constexpr (CALC_MODE == 6) DispatchBatch();
        else if constexpr (CALC_MODE == 7) RunPair();
        else if constexpr (CALC_MODE == 3) DispatchCube();
        else if constexpr (CALC_MODE == 0 || CALC_MODE >= 3) RunCubeGeneral();
        else DispatchSmallVector();
    }
private:
        static constexpr uint32_t VECTOR_DATA_DIM = CALC_MODE == 2 && DIM_TAG == 19 ? 32 : (DIM_TAG == 4 ? 3 :
        (DIM_TAG == 5 ? 13 : (DIM_TAG == 3 || DIM_TAG == 6 ? 64 : 1)));
    static constexpr uint32_t BATCH_ROWS = DIM_TAG == 12 ? 1 : (DIM_TAG == 15 || DIM_TAG == 23 ? 3 :
        (DIM_TAG == 7 ? 32 : (DIM_TAG == 6 ? 16 : (DIM_TAG == 3 || DIM_TAG == 9 ? 8 : 4))));
    static constexpr bool BATCH_PADDED = DIM_TAG == 2 || DIM_TAG == 3 || DIM_TAG == 5 || DIM_TAG == 6 ||
        DIM_TAG == 7 || DIM_TAG == 12 || DIM_TAG == 15 || DIM_TAG == 23;
    static constexpr bool BATCH_LONG_K = DIM_TAG == 4 || DIM_TAG == 5 || DIM_TAG == 17 || DIM_TAG == 18;
    static constexpr uint32_t BATCH_DIM = DIM_TAG == 8 || DIM_TAG == 9 ? 16 : (DIM_TAG == 10 ? 32 :
        (DIM_TAG == 11 ? 48 : (DIM_TAG == 12 ? 13 :
        (DIM_TAG == 15 ? 13 :
        (DIM_TAG == 17 ? 112 : (DIM_TAG == 18 ? 128 :
        (DIM_TAG == 23 ? 13 : 0)))))));
    static constexpr bool BATCH_ONE_TILE = DIM_TAG == 12 || DIM_TAG == 15 || DIM_TAG == 23;
    static constexpr bool BATCH_FULL_ROWS = DIM_TAG == 12 || DIM_TAG == 15;
    static constexpr bool CUBE_PIPELINED = CALC_MODE == 3 || CALC_MODE == 4;
    static constexpr bool CUBE_PREFETCH = CALC_MODE >= 4;
    static constexpr bool CUBE_SINGLE_TILE = DIM_TAG == 25 || DIM_TAG == 28 || DIM_TAG == 29 || DIM_TAG == 30 || DIM_TAG == 31;
    static constexpr uint32_t CUBE_DIM = DIM_TAG == 26 || DIM_TAG == 28 ? 128 :
        (DIM_TAG == 14 || DIM_TAG == 17 || DIM_TAG == 27 || DIM_TAG == 29 ? 256 : ((DIM_TAG == 30 || DIM_TAG == 31) ? 33 : 0));
    static constexpr uint32_t CUBE_ROWS = DIM_TAG == 14 || DIM_TAG == 17 ? 128 : (DIM_TAG == 31 ? 16 : 0);
    static constexpr bool CUBE_BALANCED_TAIL = DIM_TAG == 17;
    static constexpr bool VECTOR_ONE_TILE = CALC_MODE == 2 && (DIM_TAG >= 4 || DIM_TAG == 3);
    static constexpr uint32_t VECTOR_DIM = CALC_MODE == 2 && DIM_TAG == 19 ? 32 :
        (DIM_TAG == 4 ? 3 : (DIM_TAG == 5 ? 13 : (DIM_TAG == 3 || DIM_TAG == 6 ? 64 : 0)));

    GM_ADDR input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling;
    __aicore__ inline void InitializeMask() {
        if ASCEND_IS_AIV {
            // First vector operations on these appended classes set count mode
            // and restore normal mode, or ScalarVector sets normal mode itself.
            if constexpr(!((CALC_MODE==2 && (DIM_TAG==25 || DIM_TAG==26 || DIM_TAG==27)) ||
                (CALC_MODE==1 && DIM_TAG==10) || (CALC_MODE==6 && DIM_TAG==1)))
            AscendC::SetMaskNorm();
        }
    }

    __aicore__ inline void RunTinyCurrent() {

        if(AscendC::GetBlockIdx()==0) {
            TinyCurrentMaskSwiGlu<DT_INPUT_X,4> op;
            op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output,tiling);op.Process();
        }
    }

    __aicore__ inline void RunStaticVector() {

        if(DIM_TAG==27 || AscendC::GetBlockIdx()==0) {
            constexpr uint32_t dim=DIM_TAG==26?32:64;
            const SwiGluVectorTilingData data{1,dim,128,16,1};
            if constexpr(DIM_TAG==26) {
                VectorSmallFmaSwiGlu<DT_INPUT_X,2,true,32> op;
                op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output,data,tiling);op.Process();
            } else {
                VectorSwiGlu<DT_INPUT_X,2,true,64> op;
                op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output,data,tiling);op.Process();
            }
        }
    }

    __aicore__ inline void RunScalarCompact() {

        if(AscendC::GetBlockIdx()==0) {
            ScalarVectorFmaCompactSwiGlu<DT_INPUT_X> op;
            op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output);op.Process();
        }
    }

    __aicore__ inline void RunSevenRows() {

        if(AscendC::GetBlockIdx()<TAIL_CODE)
        GatherSevenRows13Paired<DT_INPUT_X>(input_x,weight_1,bias_1,weight_2,bias_2,output);
    }

    __aicore__ inline void RunPair() {

        GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
        PairVectorSwiGlu<DT_INPUT_X, DIM_TAG == 1,
            DIM_TAG == 13 ? 224 : (DIM_TAG == 14 ? 256 : 0)> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data);
        op.Process();
    }

    __aicore__ inline void RunBatchSeven() {

        if (AscendC::GetBlockIdx() < DIM_TAG - 23) {
            const SwiGluVectorTilingData data{7,13,7,16,1};
            BatchedVectorSwiGlu<DT_INPUT_X,7,true,false,13,true,true> op;
            op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output,data,tiling);
            op.Process();
        }
    }

    __aicore__ inline void RunBatchThree() {

        if (AscendC::GetBlockIdx() < AscendC::GetBlockNum() - (23 - DIM_TAG)) {
            const SwiGluVectorTilingData data{3, 13, 3, 16, 1};
            BatchedVectorSwiGlu<DT_INPUT_X, 3, true, false, 13, true, true> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
            op.Process();
        }
    }

    __aicore__ inline void RunBatch33Eight() {

        // A complete local tile descriptor; total rows follow the launch
        // grid, with the remainder and inactive lanes selected by tiling.
        const SwiGluVectorTilingData data{8,33,8,40,1};
        BatchedVectorSwiGlu<DT_INPUT_X,8,true,false,33,true,TAIL_CODE==0,40,TAIL_CODE,true,PAD_BLOCKS> op;
        op.Init(input_x,weight_1,bias_1,weight_2,bias_2,output,data,tiling);
        op.Process();
    }

    __aicore__ inline void RunBatch33Two() {

        const SwiGluVectorTilingData data{2, 33, 2, 40, 1};
        BatchedVectorSwiGlu<DT_INPUT_X, 2, true, false, 33, true, false, 40, 1> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
        op.Process();
    }

    __aicore__ inline void RunBatch33Three() {

        const SwiGluVectorTilingData data{3, 33, 3, 40, 1};
        BatchedVectorSwiGlu<DT_INPUT_X, 3, true, false, 33, true, false, 40, 1> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
        op.Process();
    }

    __aicore__ inline void RunBatchGeneral() {

        SwiGluVectorTilingData data;
        if constexpr (DIM_TAG == 12 || DIM_TAG == 15) {
            // Complete row groups derive their input/output offsets from block IDs.
            data = {DIM_TAG == 12 ? 1U : 3U, 13, DIM_TAG == 12 ? 1U : 3U, 16, 1};
        } else {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, dynamicData, tiling);
            data = dynamicData;
        }
        BatchedVectorSwiGlu<DT_INPUT_X, BATCH_ROWS, BATCH_PADDED, BATCH_LONG_K, BATCH_DIM, BATCH_ONE_TILE, BATCH_FULL_ROWS> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
        op.Process();
    }

    __aicore__ inline void RunScalar() {

        if (AscendC::GetBlockIdx() == 0) {
            ScalarVectorSwiGlu<DT_INPUT_X> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output);
            op.Process();
        }
    }

    __aicore__ inline void RunTiny() {

        if (DIM_TAG != 16 || AscendC::GetBlockIdx() == 0) {
            TinyVectorSwiGlu<DT_INPUT_X, 4> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling);
            op.Process();
        }
    }

    __aicore__ inline void RunCubeWaves() {

        if ASCEND_IS_AIC ComputeProjectionWaves<DT_INPUT_X>(input_x,weight_1,weight_2,workspace);
        if ASCEND_IS_AIV GateProjectionWaves<DT_INPUT_X>(bias_1,bias_2,output,workspace);
    }

    __aicore__ inline void RunCubeFullRow() {

        constexpr uint32_t paddedTailRows = TAIL_CODE == 8 || TAIL_CODE == 10 ? 64 : (TAIL_CODE == 9 || TAIL_CODE == 11 ? 128 : 0);
        // Geometry follows generic schedule parameters, not a list of input
        // shapes. The same body is compiled for all bounded wave/tail plans.
        constexpr uint32_t scheduledRows = ROW_WAVES * (20 / 2 * 128) +
            (20 / 2 - PAD_BLOCKS) * paddedTailRows;
        SwiGluVectorTilingData data{ROW_WAVES != 0 ? scheduledRows : 0,256,128,128,20};
        if constexpr (TAIL_CODE == 0) {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, dynamicData, tiling);
            data.rows = dynamicData.rows; data.cores = dynamicData.cores;
        }
        if ASCEND_IS_AIC ComputeProjections<DT_INPUT_X,true,false,false,64,256,128,true,paddedTailRows,(TAIL_CODE==10 || TAIL_CODE==11),(TAIL_CODE!=0 && ROW_WAVES==0)>(input_x,weight_1,weight_2,workspace,data,tiling);
        if ASCEND_IS_AIV {
            GateProjections<DT_INPUT_X,false,256,128,true,paddedTailRows,(TAIL_CODE==10 || TAIL_CODE==11),(TAIL_CODE!=0 && ROW_WAVES==0)> op;
            op.Init(bias_1,bias_2,output,workspace,data,tiling); op.Process();
        }
    }

    __aicore__ inline void RunCubeEarlyFix() {

        const uint32_t cores=static_cast<uint32_t>(AscendC::GetBlockNum());
        const SwiGluVectorTilingData data{cores/2*64,256,64,128,cores};
        if ASCEND_IS_AIC ComputeSplit128EarlyFix<DT_INPUT_X>(input_x,weight_1,weight_2,workspace);
        if ASCEND_IS_AIV GateSplitFullK<DT_INPUT_X,true>(bias_1,bias_2,output,workspace,data);
    }

    __aicore__ inline void RunCubeFullK() {

        // This path owns one complete64-row tile perN128pair. TotalM and
        // corecount follow the launch grid, and do not specialize a test shape.
        const uint32_t cores=static_cast<uint32_t>(AscendC::GetBlockNum());
        const SwiGluVectorTilingData data{cores/2*64,256,64,128,cores};
        if ASCEND_IS_AIC ComputeSplitFullK<DT_INPUT_X,true>(input_x,weight_1,weight_2,workspace,data);
        if ASCEND_IS_AIV GateSplitFullK<DT_INPUT_X,true>(bias_1,bias_2,output,workspace,data);
    }

    __aicore__ inline void RunCubeDual() {

        GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
        if ASCEND_IS_AIC ComputeDualC0TwoWave<DT_INPUT_X>(input_x,weight_1,weight_2,workspace,data);
        if ASCEND_IS_AIV GateDualC0TwoWave<DT_INPUT_X>(bias_1,bias_2,output,workspace,data);
    }

    __aicore__ inline void RunCubeSplit() {

        GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData,data,tiling);
        if ASCEND_IS_AIC {
            ComputeSplitFullK<DT_INPUT_X>(input_x,weight_1,weight_2,workspace,data);
        }
        if ASCEND_IS_AIV {
            GateSplitFullK<DT_INPUT_X>(bias_1,bias_2,output,workspace,data);
        }
    }

    __aicore__ inline void RunCubeGeneral() {

        // CANN passes the user region after its reserved workspace prefix.
        GM_ADDR userWorkspace = workspace;
        if ASCEND_IS_AIC {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
            if constexpr (DIM_TAG >= 26 && DIM_TAG <= 29) {
                data.dim = DIM_TAG == 26 || DIM_TAG == 28 ? 128 : 256;
                data.tileCols = 128;
            }
            ComputeProjections<DT_INPUT_X, CUBE_PIPELINED, CUBE_PREFETCH, CUBE_SINGLE_TILE, 64, CUBE_DIM, CUBE_ROWS, CUBE_BALANCED_TAIL>(input_x,
                weight_1, weight_2, userWorkspace, data);
        }
        if ASCEND_IS_AIV {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, data, tiling);
            if constexpr (DIM_TAG >= 26 && DIM_TAG <= 29) {
                data.dim = DIM_TAG == 26 || DIM_TAG == 28 ? 128 : 256;
                data.tileCols = 128;
            }
            GateProjections<DT_INPUT_X, CUBE_SINGLE_TILE, CUBE_DIM, CUBE_ROWS, CUBE_BALANCED_TAIL> op;
            op.Init(bias_1, bias_2, output, userWorkspace, data);
            op.Process();
        }
    }

    __aicore__ inline void RunVectorGeneral() {

        SwiGluVectorTilingData data;
        if constexpr ((DIM_TAG >= 4 && DIM_TAG <= 6) || (CALC_MODE == 1 && DIM_TAG == 7) ||
            (CALC_MODE == 2 && (DIM_TAG == 19 || DIM_TAG == 3))) {
            // Block IDs identify each row and column tile. The one-tile vector
            // classes do not need a device-side read of scheduling fields.
            constexpr uint32_t dim = VECTOR_DATA_DIM;
            data = {1, dim, 128, 16,
                    1U};
        } else {
            GET_TILING_DATA_WITH_STRUCT(SwiGluVectorTilingData, dynamicData, tiling);
            data = dynamicData;
            if constexpr (DIM_TAG >= 1 && DIM_TAG <= 3) {
                data.dim = DIM_TAG == 1 ? 3 : (DIM_TAG == 2 ? 13 : 64);
                data.tileCols = 16;
            }
        }
        if (CALC_MODE != 2 || DIM_TAG != 19 || AscendC::GetBlockIdx() == 0) {
            VectorSwiGlu<DT_INPUT_X, CALC_MODE, VECTOR_ONE_TILE, VECTOR_DIM> op;
            op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, data, tiling);
            op.Process();
        }
    }

    __aicore__ inline void DispatchBatch() {
        if constexpr (DIM_TAG == 1 || DIM_TAG == 24) RunSevenRows();
        else if constexpr (DIM_TAG >= 25 && DIM_TAG <= 31) RunBatchSeven();
        else if constexpr (DIM_TAG >= 16 && DIM_TAG <= 22) RunBatchThree();
        else if constexpr (DIM_TAG == 14) RunBatch33Eight();
        else if constexpr (DIM_TAG == 13) RunBatch33Two();
        else if constexpr (DIM_TAG == 11) RunBatch33Three();
        else RunBatchGeneral();
    }
    __aicore__ inline void DispatchCube() {
        if constexpr (DIM_TAG == 19) RunCubeWaves();
        else if constexpr (DIM_TAG == 18) RunCubeFullRow();
        else if constexpr (DIM_TAG == 24) RunCubeEarlyFix();
        else if constexpr (DIM_TAG == 21) RunCubeFullK();
        else if constexpr (DIM_TAG == 20 || DIM_TAG == 23) RunCubeDual();
        else if constexpr (DIM_TAG == 22) RunCubeSplit();
        else RunCubeGeneral();
    }
    __aicore__ inline void DispatchSmallVector() {
        if constexpr (CALC_MODE == 2 && DIM_TAG == 25) RunTinyCurrent();
        else if constexpr (CALC_MODE == 2 && (DIM_TAG == 26 || DIM_TAG == 27)) RunStaticVector();
        else if constexpr (CALC_MODE == 1 && DIM_TAG == 10) RunScalarCompact();
        else if constexpr (CALC_MODE == 1 && DIM_TAG == 7) RunScalar();
        else if constexpr (CALC_MODE == 2 && (DIM_TAG == 24 || DIM_TAG == 16)) RunTiny();
        else RunVectorGeneral();
    }
};

#endif  // SWIGLU_DEVICE_DISPATCH
