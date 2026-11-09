/*
 * Copyright (c) 2026 ceprei_Tnx.
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
#define K_MAX_SHAPE_DIM 0
#include "kernel_tpipe.h"
#include "kernel_tensor.h"
#include "kernel_type.h"
#include "kernel_prof_trace_intf.h"
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_common_intf.h"
#include "kernel_operator_vec_binary_intf.h"
#include "kernel_operator_vec_binary_scalar_intf.h"
#include "kernel_operator_vec_duplicate_intf.h"
#include "kernel_operator_vec_gather_mask_intf.h"
#include "kernel_operator_vec_vconv_intf.h"
#include "kernel_operator_scalar_intf.h"
#include "kernel_operator_vec_cmpsel_intf.h"
#include "kernel_operator_vec_ternary_scalar_intf.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_operator_cache_intf.h"
#include "kernel_operator_utils_intf.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"
#include <type_traits>

namespace Sbg {
using namespace AscendC;

template <typename In, typename Out>
struct Layout {
    static constexpr uint32_t ROWS = SBG_MAX_LOCAL_ROWS;
    static constexpr uint32_t OUTPUT_ROWS = ROWS;
    static constexpr uint32_t PERSON = 0U;
    static constexpr uint32_t ANCHOR = PERSON + ROWS * 4U * sizeof(In) + 32U;
    static constexpr uint32_t HALF_Y1 = ANCHOR + ROWS * 2U * sizeof(In) + 32U;
    static constexpr uint32_t HALF_Y2 = HALF_Y1 + ROWS * sizeof(In) + 32U;
    static constexpr uint32_t HALF_CY = HALF_Y2 + ROWS * sizeof(In) + 32U;
    static constexpr uint32_t Y1 = HALF_CY + ROWS * sizeof(In) + 32U;
    static constexpr uint32_t Y2 = Y1 + ROWS * sizeof(float) + 32U;
    static constexpr uint32_t CY = Y2 + ROWS * sizeof(float) + 32U;
    static constexpr uint32_t OUTPUT = CY + ROWS * sizeof(float) + 32U;
    static constexpr uint32_t MASK = OUTPUT + OUTPUT_ROWS * sizeof(Out) + 32U;
    static constexpr uint32_t ZERO = MASK + ROWS + 32U;
    static constexpr uint32_t BYTES = ZERO + 32U;
    static_assert(BYTES < 192U * 1024U, "UB capacity exceeded");
    static_assert(BYTES % 32U == 0U, "UB layout alignment");
};

template <typename U>
__aicore__ inline LocalTensor<U> At(uint32_t offset, uint32_t count)
{
    return LocalTensor<U>(TPosition::VECCALC, offset, count);
}

template <typename T>
__aicore__ inline void Load(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                            uint32_t elements)
{
    if ((elements * sizeof(T)) % 32U == 0U) {
        DataCopy(dst, src, elements);
    } else {
        DataCopyExtParams copy{1U, elements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src, copy, padding);
    }
}

template <typename T>
__aicore__ inline void Store(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                             uint32_t elements)
{
    if ((elements * sizeof(T)) % 32U == 0U) {
        DataCopy(dst, src, elements);
    } else {
        DataCopyExtParams copy{1U, elements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(dst, src, copy);
    }
}

template <typename T, uint8_t PATTERN>
__aicore__ inline void GatherColumn(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                    uint16_t repeats)
{
    static_assert(PATTERN == 2U || PATTERN == 4U || PATTERN == 6U, "invalid coordinate pattern");
    using Bits = std::conditional_t<sizeof(T) == 2U, uint16_t, uint32_t>;
    auto dstPtr = reinterpret_cast<__ubuf__ Bits *>(dst.GetPhyAddr());
    auto srcPtr = reinterpret_cast<__ubuf__ Bits *>(src.GetPhyAddr());
    auto unusedPattern = srcPtr + 256U * sizeof(Bits);
    vreducev2(dstPtr, srcPtr, unusedPattern, repeats, 1, PATTERN, 8, 0);
}

template <typename In, typename Out>
struct State {
    GlobalTensor<In> personGm;
    GlobalTensor<In> anchorGm;
    GlobalTensor<Out> outputGm;
    LocalTensor<In> person;
    LocalTensor<In> anchor;
    LocalTensor<Out> output;
    LocalTensor<Out> zero;
    LocalTensor<float> y1;
    LocalTensor<float> y2;
    LocalTensor<float> cy;
    LocalTensor<uint8_t> mask;
    uint32_t workRows;
    uint8_t mathRepeats;
    uint16_t personRepeats;
    uint16_t anchorRepeats;
};

template <typename In, typename Out>
__aicore__ inline void InitState(State<In, Out> &s, GM_ADDR personBoxes, GM_ADDR anchorPoints,
                                 GM_ADDR threshold, GM_ADDR violationScore, uint32_t firstRow,
                                 uint32_t validRows, bool hasThreshold)
{
    using L = Layout<In, Out>;
    if (hasThreshold) {
        GlobalTensor<uint64_t> thresholdCache;
        thresholdCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(threshold), 1U);
        DataCachePreload(thresholdCache, static_cast<int16_t>(0));
    }
    s.workRows = ((validRows + 63U) / 64U) * 64U;
    s.mathRepeats = static_cast<uint8_t>(s.workRows / 64U);
    s.personRepeats = static_cast<uint16_t>(
        (validRows + 256U / (4U * sizeof(In)) - 1U) / (256U / (4U * sizeof(In))));
    s.anchorRepeats = static_cast<uint16_t>(
        (validRows + 256U / (2U * sizeof(In)) - 1U) / (256U / (2U * sizeof(In))));
    s.personGm.SetGlobalBuffer(reinterpret_cast<__gm__ In *>(personBoxes) + firstRow * 4U, validRows * 4U);
    s.anchorGm.SetGlobalBuffer(reinterpret_cast<__gm__ In *>(anchorPoints) + firstRow * 2U, validRows * 2U);
    s.outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ Out *>(violationScore) + firstRow, validRows);
    s.person = At<In>(L::PERSON, L::ROWS * 4U);
    s.anchor = At<In>(L::ANCHOR, L::ROWS * 2U);
    s.output = At<Out>(L::OUTPUT, L::OUTPUT_ROWS);
    s.zero = At<Out>(L::ZERO, 32U / sizeof(Out));
    s.y1 = At<float>(L::Y1, L::ROWS);
    s.y2 = At<float>(L::Y2, L::ROWS);
    s.cy = At<float>(L::CY, L::ROWS);
    s.mask = At<uint8_t>(L::MASK, L::ROWS);
    Load(s.person, s.personGm, validRows * 4U);
    Load(s.anchor, s.anchorGm, validRows * 2U);
    Duplicate(s.output, static_cast<Out>(1.0f), s.workRows);
    Duplicate(s.zero, static_cast<Out>(0), 32U / sizeof(Out));
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

template <typename In, typename Out>
__aicore__ inline void StageInputs(State<In, Out> &s)
{
    using L = Layout<In, Out>;
    SetMaskNorm();
    SetVectorMask<In>(0, 0);
    if constexpr (std::is_same_v<In, AscendC::half>) {
        auto halfY1 = At<In>(L::HALF_Y1, L::ROWS);
        auto halfY2 = At<In>(L::HALF_Y2, L::ROWS);
        auto halfCy = At<In>(L::HALF_CY, L::ROWS);
        GatherColumn<In, 4U>(halfY1, s.person, s.personRepeats);
        GatherColumn<In, 6U>(halfY2, s.person, s.personRepeats);
        GatherColumn<In, 2U>(halfCy, s.anchor, s.anchorRepeats);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
        Cast<float, In, false>(s.y1, halfY1, RoundMode::CAST_NONE, 64U, s.mathRepeats, {1, 1, 8, 4});
        Cast<float, In, false>(s.y2, halfY2, RoundMode::CAST_NONE, 64U, s.mathRepeats, {1, 1, 8, 4});
        Cast<float, In, false>(s.cy, halfCy, RoundMode::CAST_NONE, 64U, s.mathRepeats, {1, 1, 8, 4});
    } else {
        GatherColumn<float, 4U>(s.y1, s.person, s.personRepeats);
        GatherColumn<float, 6U>(s.y2, s.person, s.personRepeats);
        GatherColumn<float, 2U>(s.cy, s.anchor, s.anchorRepeats);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, 0xFFFFFFFFFFFFFFFFULL);
}

template <typename In, typename Out>
__aicore__ inline void ComputeAndStore(State<In, Out> &s, GM_ADDR threshold, uint32_t validRows,
                                       float ratio, bool hasThreshold)
{
    Sub<float, false>(s.y2, s.y2, s.y1, 64U, s.mathRepeats, {1, 1, 1, 8, 8, 8});
    Axpy<float, float, false>(s.y1, s.y2, ratio, 64U, s.mathRepeats, {1, 1, 8, 8});
    if (hasThreshold) {
        const float thresholdValue = static_cast<float>(*reinterpret_cast<const __gm__ Out *>(threshold));
        if (thresholdValue != 0.0f) {
            Adds<float, false>(s.y1, s.y1, thresholdValue, 64U, s.mathRepeats, {1, 1, 8, 8});
        }
    }
    if (validRows <= 64U) {
        const uint64_t compareMask[2] = {0xFFFFFFFFFFFFFFFFULL, 0};
        Compare(s.cy, s.y1, CMPMODE::GT, compareMask, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Select<Out, SELMODE::VSEL_CMPMASK_SPR>(s.output, s.output, s.zero, 1U, {1, 1, 0, 8, 8, 0});
    } else {
        Compare(s.mask, s.cy, s.y1, CMPMODE::GT, static_cast<uint64_t>(64U), s.mathRepeats,
                {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        set_cmpmask(reinterpret_cast<__ubuf__ Out *>(s.zero.GetPhyAddr()));
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<Out>(0, s.workRows);
        vsel(reinterpret_cast<__ubuf__ Out *>(s.output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ Out *>(s.output.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t *>(s.mask.GetPhyAddr()),
             1, 1, 1, 1, 8, 8, 8, 1);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    Store(s.outputGm, s.output, validRows);
}

template <typename In, typename Out>
__aicore__ inline void RunTile(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold,
                               GM_ADDR violationScore, uint32_t firstRow, uint32_t validRows,
                               float ratio, bool hasThreshold)
{
    State<In, Out> state;
    InitState(state, personBoxes, anchorPoints, threshold, violationScore, firstRow, validRows,
              hasThreshold);
    StageInputs(state);
    ComputeAndStore(state, threshold, validRows, ratio, hasThreshold);
}
}  // namespace Sbg

template <typename DT_PERSON_BOXES, typename DT_VIOLATION_SCORE, uint32_t GRID_CLASS, uint32_t TAIL_PRESENT,
          uint32_t HAS_THRESHOLD, uint32_t SMALL_ROWS_MINUS_ONE, uint32_t RATIO_ONE>
__global__ __aicore__ void safety_belt_violation_geometry(
    GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold, GM_ADDR violationScore,
    GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(GRID_CLASS == 0U && TAIL_PRESENT == 0U && SMALL_ROWS_MINUS_ONE == 0U &&
                  RATIO_ONE == 0U, "unsupported scheduling class");
    const uint64_t slot = *reinterpret_cast<const __gm__ uint64_t *>(tiling);
    const uint32_t pack = static_cast<uint32_t>(slot);
    union RatioBits {
        uint32_t bits;
        float value;
    } ratioBits;
    ratioBits.bits = static_cast<uint32_t>(slot >> 32U);
    const uint32_t batch = (pack & 0xFFFU) + 1U;
    const uint32_t rowsPerCore = ((pack >> 12U) & 0xFU) * 64U;
    if (rowsPerCore == 0U || rowsPerCore > SBG_MAX_LOCAL_ROWS) {
        return;
    }
    const uint32_t start = AscendC::GetBlockIdx() * rowsPerCore;
    if (start >= batch) {
        return;
    }
    const uint32_t remaining = batch - start;
    const uint32_t validRows = remaining < rowsPerCore ? remaining : rowsPerCore;
    Sbg::RunTile<DT_PERSON_BOXES, DT_VIOLATION_SCORE>(personBoxes, anchorPoints, threshold,
                                                      violationScore, start, validRows,
                                                      ratioBits.value, HAS_THRESHOLD != 0U);
}

// This operator uses direct UB tensors and never consumes the system workspace.
// Experimental generated-entry specialization for CANN 9.0; SDK files stay untouched.
// Preserve the standard entry for debug, timestamp and OOM-instrumented builds.
#include "kernel_common.h"
#if (!defined(ASCENDC_DUMP) || ASCENDC_DUMP == 0) && !defined(ASCENDC_TIME_STAMP_ON) && \
    (!defined(ASCENDC_OOM) || ASCENDC_OOM == 0)
namespace AscendC {
__aicore__ inline void SbgUnusedWorkspaceEntry(GM_ADDR workspace)
{
    (void)workspace;
}
}
#define SetSysWorkspaceForce SbgUnusedWorkspaceEntry
#endif
